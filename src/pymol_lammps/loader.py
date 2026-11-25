"""PyMOL commands for LAMMPS trajectories.

``register()`` adds the ``load_lammps``, ``load_xyz``, ``lammps_wrap``,
``drude_style`` and ``drawgridbox`` commands and lets PyMOL's ``load`` open
``.lammpstrj`` files. Only public PyMOL API is used, except the format table that
connects ``load`` to this reader (``pymol.importing.loadfunctions``); if a PyMOL
update removes that table, ``load_lammps`` keeps working.
"""

import contextlib
import itertools
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pymol
from chempy import Atom, Bond, models
from numpy.typing import NDArray
from pymol import cgo, cmd, importing

from pymol_lammps import xyz
from pymol_lammps.bonds import (
    bond_limits,
    bonding_elements,
    cell_heights,
    guess_bonds,
    max_bond_limit,
    molecule_index,
    periodic_bonds,
    satellite_bonds,
)
from pymol_lammps.completion import FILE_ARGUMENTS, register_completion
from pymol_lammps.datafile import read_data
from pymol_lammps.drude import (
    DEUTERIUM_DISTANCE,
    DRUDE,
    VIRTUAL,
    parse_drude,
    satellite_hosts,
    satellite_mask,
)
from pymol_lammps.dump import Box, parse_cell, read_frames
from pymol_lammps.gridbox import drawgridbox
from pymol_lammps.style import colour_satellites, drude_style, refresh_dashes
from pymol_lammps.topology import Topology, build_topology
from pymol_lammps.wrapping import (
    Molecules,
    attach_satellites,
    place_satellites,
    split_wrapping_molecules,
    wrap_molecules,
)

# Grey stays visible on both black and white backgrounds.
BOX_COLOR = (0.5, 0.5, 0.5)

# Positions, box (None for an xyz frame without cell) and whether the positions are
# unwrapped, for one frame as read from the file.
type _RawFrame = tuple[NDArray[np.float64], Box | None, bool]


@dataclass(frozen=True)
class _Trajectory:
    """What ``lammps_wrap`` needs to re-read the coordinates of a loaded object.

    Attributes
    ----------
    path : str
        Trajectory file.
    start, stop, interval : int
        Frame range as given to the load command.
    molecules : Molecules | None
        Molecules for wrapping, None if unknown.
    satellites : NDArray[np.int64]
        Pairs ``(satellite, host)`` of the Drude particles and virtual sites.
    ids : list[int]
        PyMOL ``ID`` of each atom, in the order of the arrays.
    bonds : NDArray[np.int64] | None
        Bonds given to PyMOL by the loader; None if PyMOL guessed them.
    satellite : NDArray[np.bool_]
        Which atoms are Drude particles or virtual sites.
    unwrapped : bool
        Whether the file has unwrapped coordinates (never for xyz files).
    removed : NDArray[np.bool_]
        Which of ``bonds`` are removed from the display as periodic bonds.
    xyz : bool
        True for objects from ``load_xyz``.
    cell : Box | None
        For xyz objects, the user-given cell, which takes precedence over the
        ``Lattice`` of each frame.
    """

    path: str
    start: int
    stop: int
    interval: int
    molecules: Molecules | None
    satellites: NDArray[np.int64]
    ids: list[int]
    bonds: NDArray[np.int64] | None
    satellite: NDArray[np.bool_]
    unwrapped: bool
    removed: NDArray[np.bool_]
    xyz: bool = False
    cell: Box | None = None


# Trajectories loaded in this session, by object name.
_TRAJECTORIES: dict[str, _Trajectory] = {}


def _remember(name: str, trajectory: _Trajectory) -> None:
    """Store a loaded trajectory as the most recent one.

    The entry moves to the end of ``_TRAJECTORIES`` also when ``name`` was loaded
    before, so that the order of the dict is the order of loading.
    """
    _TRAJECTORIES.pop(name, None)
    _TRAJECTORIES[name] = trajectory


def _last_loaded(_self: Any) -> str:
    """Name of the most recently loaded object that still exists in PyMOL.

    Raises
    ------
    pymol.CmdException
        If no object from ``load_lammps`` or ``load_xyz`` is left in the session.
    """
    present = set(_self.get_names("objects"))
    for name in reversed(_TRAJECTORIES):
        if name in present:
            return name
    raise pymol.CmdException(
        "no object loaded with load_lammps or load_xyz in this session"
    )


@contextlib.contextmanager
def _pymol_messages(command: str) -> Iterator[None]:
    """Print warnings to the PyMOL console and turn ValueError into CmdException."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield
        except ValueError as error:
            raise pymol.CmdException(str(error)) from error
        finally:
            for warning in caught:
                print(f" {command} warning: {warning.message}")


def _can_wrap(molecules: Molecules | None, unwrapped: bool) -> bool:
    """Return whether molecules can be wrapped; warn where it would matter."""
    if molecules is None:
        if unwrapped:
            warnings.warn(
                "no molecule IDs (dump column mol, or molecule IDs in the data "
                "file), so molecules were not wrapped into the box",
                stacklevel=2,
            )
        return False
    return True


def _displayed(
    frames: Iterable[_RawFrame],
    molecules: Molecules | None,
    satellites: NDArray[np.int64],
    wrap: bool,
) -> Iterator[tuple[NDArray[np.float64], Box | None]]:
    """Yield the coordinates to display for each frame, with the box of the frame.

    Whole molecules are wrapped if asked (and known). Drude particles and virtual
    sites are always moved next to their host, because they must not be shown on
    the other side of the box, whatever the file or the wrapping did.
    """
    previous = None
    for positions, box, unwrapped in frames:
        if box is not None:
            if wrap and molecules is not None:
                positions = wrap_molecules(
                    positions,
                    molecules,
                    box,
                    make_whole=not unwrapped,
                    previous=previous,
                )
            if len(satellites):
                positions = place_satellites(positions, satellites, box)
        previous = positions
        yield positions, box


def _chempy_model(topology: Topology, positions: NDArray[Any]) -> Any:
    """Build a chempy model of the first frame, including bonds if known."""
    model = models.Indexed()
    # Python lists instead of NumPy arrays: indexing them per atom is much faster.
    types = topology.types.tolist()
    charges = None if topology.charges is None else topology.charges.tolist()
    for i, coord in enumerate(positions.tolist()):
        atom = Atom()
        atom.coord = coord
        atom.symbol = topology.elements[i]
        atom.name = topology.names[i]
        atom.resn = topology.residue_names[i]
        atom.resi = topology.residue_ids[i]
        atom.numeric_type = types[i]
        atom.text_type = str(types[i])
        if charges is not None:
            atom.partial_charge = charges[i]
        model.add_atom(atom)
    for pair in topology.bonds.tolist() if topology.bonds is not None else ():
        bond = Bond()
        bond.index = pair
        model.add_bond(bond)
    return model


def _add_bonds(name: str, ids: list[int], pairs: NDArray[np.int64], _self: Any) -> None:
    """Add bonds to a loaded object; ``pairs`` are indices into ``ids``."""
    # PyMOL stores atoms in its own order; the atom ID is stable. ``add_bond`` takes
    # PyMOL's atom index and needs no selection, so it is fast for many bonds.
    index_of: dict[int, int] = {}
    _self.iterate(name, "index_of[ID] = index", space={"index_of": index_of})
    for i, j in pairs.tolist():
        _self.add_bond(name, index_of[ids[i]], index_of[ids[j]])


def _remove_bonds(
    name: str, ids: list[int], pairs: NDArray[np.int64], _self: Any
) -> None:
    """Remove bonds from a loaded object; ``pairs`` are indices into ``ids``."""
    for i, j in pairs.tolist():
        # PyMOL stores atoms in its own order; the atom ID is stable.
        _self.unbond(f"{name} and id {ids[i]}", f"{name} and id {ids[j]}")


def _finish_satellites(
    name: str,
    ids: list[int],
    satellite: NDArray[np.bool_],
    guessed: NDArray[np.int64] | None,
    _self: Any,
) -> None:
    """Prepare the Drude particles and virtual sites of a loaded object.

    PyMOL treats the symbol ``D`` as deuterium: it counts as hydrogen in the
    selection ``hydro``, which would make ``hide (hydro)`` hide the Drudes as well.
    Setting the atomic number to 0 removes them from ``hydro``; ``elem D`` and
    ``elem M`` select them. They get the colour and radius of ``drude_style``, but
    keep their representations.

    If PyMOL guessed the bonds itself, ``guessed`` holds the satellite bonds (pairs
    of indices into ``ids``), which replace PyMOL's bonds of the satellites.
    """
    if not satellite.any():
        return
    _self.alter(f"{name} and elem {DRUDE}", "protons = 0")
    colour_satellites(name, _self=_self)
    if guessed is not None:
        _self.unbond(f"{name} and elem {DRUDE}+{VIRTUAL}", name)
        _add_bonds(name, ids, guessed, _self)


def _warn_if_deuterium(
    positions: NDArray[np.float64],
    satellites: NDArray[np.int64],
    elements: list[str],
    box: Box | None,
) -> None:
    """Warn if the atoms with element ``D`` sit too far from their host for Drudes.

    A Drude particle sits about 0.01 A from its core, a deuterium about 1 A from
    its bonded atom; the median over all ``D`` atoms of the first frame decides.
    """
    if len(satellites) == 0:
        return
    drudes = satellites[np.asarray(elements)[satellites[:, 0]] == DRUDE]
    if len(drudes) == 0:
        return
    if box is not None:
        positions = place_satellites(positions, drudes, box)
    distance = np.median(
        np.linalg.norm(positions[drudes[:, 0]] - positions[drudes[:, 1]], axis=1)
    )
    if distance > DEUTERIUM_DISTANCE:
        warnings.warn(
            f"the atoms with element D sit {distance:.2f} A from their bonded atom "
            "(median), but Drude particles sit about 0.01 A from their core; if "
            "they are deuterium, load again with drude=0",
            stacklevel=2,
        )


def _cut_bonds(
    coordinates: NDArray[np.float64],
    bonds: NDArray[np.int64] | None,
    satellite: NDArray[np.bool_],
    box: Box | None,
    check: bool,
) -> NDArray[np.bool_]:
    """Return which bonds to remove as periodic bonds in one frame.

    Only checked if ``check`` is set and the frame has a box; otherwise nothing is
    removed.
    """
    if bonds is None:
        return np.zeros(0, dtype=bool)
    if not check or box is None:
        return np.zeros(len(bonds), dtype=bool)
    return periodic_bonds(coordinates, bonds, satellite, box)


def _guess_dump_bonds(
    positions: NDArray[np.float64],
    topology: Topology,
    satellite: NDArray[np.bool_],
    box: Box,
) -> NDArray[np.int64]:
    """Guess the bonds of a dump without data file from its first frame.

    The bonds are found with the minimum-image convention, so that bonds through
    the periodic boundary are found as well; this is what lets the loader recognise
    molecules cut by the boundary and periodic networks. With molecule IDs in the
    dump, a guessed bond between two molecules is dropped, because the dump says
    that they are separate, and the bond would become long when the two molecules
    are wrapped one by one. The bonds of Drude particles and virtual sites are kept.

    Parameters
    ----------
    positions : NDArray[np.float64]
        Cartesian positions of the first frame in Angstrom, shape (n_atoms, 3).
    topology : Topology
        Per-atom properties of the dump.
    satellite : NDArray[np.bool_]
        Which atoms are Drude particles or virtual sites, shape (n_atoms,).
    box : Box
        Box of the first frame.

    Returns
    -------
    NDArray[np.int64]
        Bonded atom pairs, shape (n_bonds, 2).
    """
    bonds = guess_bonds(positions, bonding_elements(topology.elements), box)
    index = topology.molecule_index()
    if index is None:
        return bonds
    same = index[bonds[:, 0]] == index[bonds[:, 1]]
    return bonds[same | satellite[bonds[:, 0]] | satellite[bonds[:, 1]]]


def _load_states(
    name: str,
    states: Iterator[tuple[NDArray[np.float64], Box | None]],
    trajectory: _Trajectory,
    check: bool,
    show_box: bool,
    _self: Any,
) -> tuple[int, NDArray[np.bool_]]:
    """Load the states after the first one and remove the periodic bonds they add.

    Returns the number of states and which bonds are removed in all states.
    """
    cut = trajectory.removed.copy()
    n_states = 1
    for n_states, (coordinates, box) in enumerate(states, start=2):
        # load_coordset fills atoms in loading order, i.e. by atom ID.
        _self.load_coordset(coordinates, name, n_states)
        if box is not None:
            _set_box(name, box, n_states, show_box, _self)
        cut |= _cut_bonds(
            coordinates, trajectory.bonds, trajectory.satellite, box, check
        )
    if trajectory.bonds is not None:
        # Periodic networks keep the cut of the first frame, so this is rare: a
        # molecule that could not be reassembled can cross the boundary later.
        added = trajectory.bonds[cut & ~trajectory.removed]
        _remove_bonds(name, trajectory.ids, added, _self)
    return n_states, cut


def _periodic_summary(count: int) -> str:
    """Return the summary text for removed periodic bonds, or an empty string."""
    plural = "" if count == 1 else "s"
    return f", {count} periodic bond{plural} removed" if count else ""


def _box_cgo(box: Box) -> list[float]:
    """Return the box edges as a CGO line object."""
    lines: list[float] = [cgo.BEGIN, cgo.LINES, cgo.COLOR, *BOX_COLOR]
    for start, end in box.edges():
        lines += [cgo.VERTEX, *start.tolist(), cgo.VERTEX, *end.tolist()]
    return [*lines, cgo.END]


def _set_box(name: str, box: Box, state: int, show_box: bool, _self: Any) -> None:
    """Store the box of one state as crystal symmetry and, optionally, as CGO."""
    _self.set_symmetry(name, *box.cell_parameters(), "P1", state=state)
    if show_box:
        _self.load_cgo(_box_cgo(box), f"{name}_box", state)


def _default_name(filename: str, _self: Any) -> str:
    """Return an object name from a file name (``dump.lammpstrj.gz`` -> ``dump``)."""
    path = Path(filename)
    if path.suffix == ".gz":
        path = path.with_suffix("")
    return _self.get_legal_name(path.stem)


def load_lammps(
    filename: str,
    data: str = "",
    object: str = "",
    start: int | str = 1,
    stop: int | str = -1,
    interval: int | str = 1,
    elements: str | Sequence[str] = "",
    box: int | str = 1,
    wrap: int | str = 1,
    quiet: int | str = 0,
    drude: int | str | Sequence[int] = "auto",
    *,
    _self: Any = cmd,
) -> None:
    """Load a LAMMPS dump trajectory, with an optional data file for the topology.

    Each dump frame becomes one state of the object. Atoms are ordered by LAMMPS
    atom ID, and the atom ID is stored as PyMOL ``ID``. The box drawn for each
    frame is the one in the dump (``ITEM: BOX BOUNDS``).

    By default, whole molecules are wrapped into the box of each frame (by centre
    of geometry; molecule IDs from the dump's ``mol`` column or the data file,
    without either the groups of atoms connected by the guessed bonds).
    With ``wrap=0``, coordinates are kept as in the dump, unwrapped if it has
    xu yu zu or image flags ix iy iz. ``lammps_wrap`` switches this after loading.

    Without a data file, elements come from the dump's ``element`` or ``mass``
    column, and bonds are guessed from the first frame as in ``load_xyz``: two
    atoms are bonded if their minimum-image distance is below the sum of their
    covalent radii plus 0.4 A, so bonds through the box boundary are found too.
    Element labels are reduced to element symbols (``CL`` is Cl, ``OW`` is O); a
    label that is no element is bonded like carbon, with a warning. With a ``mol``
    column, a guessed bond between two molecules is dropped. With a data file,
    masses, molecule IDs, charges, names and bonds are read from it. The atom type
    is stored as ``numeric_type`` and ``text_type``.

    Drude particles of polarizable force fields are found automatically: element
    label ``D`` (dump ``element`` column or ``elements=``), or, when the elements
    come from masses, a type with 0 < mass < 0.8 g/mol. ``drude=21 22 23`` names the
    Drude atom types instead, ``drude=0`` switches the detection off (a label ``D``
    then means deuterium). A massless site (label ``M`` or mass below 1e-8, the M
    site of SWM4-NDP water) is a virtual site. A data file overrides the labels:
    the types it marks as Drudes (by mass, or by the word ``DP`` in the ``Masses``
    comment) or as virtual sites are loaded as such, unless ``drude=0``. Drudes are
    stored with element ``D`` and virtual sites with element ``M``: select them
    with ``elem D`` and ``elem M``. They get the colour and radius that
    ``drude_style`` uses, but keep their representations. If the ``D`` atoms sit
    more than 0.3 A from their host (median), a warning suggests that they are
    deuterium and ``drude=0``. Their bonds come from the data file; without one,
    and for satellites that have no bond in it, each Drude is bonded to its nearest
    other atom and each virtual site to its nearest oxygen.
    In every frame, also with ``wrap=0``, each of them is shown at the periodic
    image next to its host atom. A dump of the cores only simply has no Drudes.
    ``drude_style`` draws them.

    Bonds longer than half of the smallest box height cannot be real bonds: they
    connect periodic images (a graphene sheet, a crystal). They are removed from the
    display, in every frame, and their number is printed; bonds of Drudes and
    virtual sites are never removed. This holds for bonds from a data file and for
    guessed bonds alike. (With ``wrap=0`` and a dump of wrapped
    coordinates they are kept, because molecules cut by the boundary are shown as
    they are.) A molecule that is bonded through the boundary and has wrapped
    coordinates cannot be reassembled, so its atoms are wrapped one by one in the
    first frame (with a warning) and then follow their own path, so that the same
    bonds are cut in every frame. With unwrapped coordinates it is kept in one piece.

    PyMOL usage::

        load_lammps filename [, data [, object [, start [, stop [, interval
                    [, elements [, box [, wrap [, quiet [, drude ]]]]]]]]]]

    Examples::

        load_lammps traj.lammpstrj
        load_lammps traj.lammpstrj, data=system.data
        load_lammps traj.lammpstrj, interval=10, elements=C H O N
        load_lammps traj.lammpstrj, data=system.data, wrap=0
        load_lammps traj.lammpstrj, data=system.data, drude=21 22 23
        color red, text_type 3
        select drudes, elem D

    Parameters
    ----------
    filename : str
        LAMMPS text dump (``dump atom`` or ``dump custom``), optionally ``.gz``.
    data : str, optional
        LAMMPS data file of the same system.
    object : str, optional
        Object name; defaults to the file name without extension. An existing
        object of that name is replaced.
    start, stop, interval : int, optional
        First frame (1-based), last frame (inclusive, -1 for all) and stride.
    elements : str | Sequence[str], optional
        Element of each atom type in type order, e.g. ``"C H O N"``. Overrides
        the dump and data file.
    box : int, optional
        1 to draw the simulation box of each frame as the object ``<object>_box``.
    wrap : int, optional
        1 to wrap whole molecules into the box, 0 to keep the dump coordinates.
    quiet : int, optional
        1 to suppress the summary line.
    drude : int | str | Sequence[int], optional
        ``auto`` (default), ``0`` to switch Drude detection off, or the Drude atom
        types, e.g. ``"21 22 23"``.

    Raises
    ------
    pymol.CmdException
        If the files cannot be interpreted.
    """
    name = object or _default_name(filename, _self)
    type_elements = elements.split() if isinstance(elements, str) else list(elements)
    show_box = bool(int(box))
    trajectory_path = str(_self.exp_path(filename))

    with _pymol_messages("load_lammps"):
        frames = read_frames(trajectory_path, int(start), int(stop), int(interval))
        first = next(frames, None)
        if first is None:
            raise ValueError(f"{filename}: no complete frame in the selected range")
        data_file = read_data(_self.exp_path(data)) if data else None
        topology = build_topology(
            first, data_file, type_elements or None, parse_drude(drude)
        )
        # Parsing the coordinates is costly; the first frame is used several times.
        first_positions = first.positions()
        satellite = satellite_mask(topology.elements)
        bonds = topology.bonds
        if bonds is None:
            bonds = _guess_dump_bonds(first_positions, topology, satellite, first.box)
        else:
            # A satellite without a bond in the data file (an M site held by fix
            # rigid, for example) is bonded to its nearest host like without one.
            unbonded = satellite.copy()
            unbonded[bonds.ravel()] = False
            if unbonded.any():
                extra = satellite_bonds(
                    first_positions, topology.elements, first.box, only=unbonded
                )
                bonds = np.concatenate([bonds, extra])
        topology = replace(topology, bonds=bonds)
        satellites = satellite_hosts(bonds, topology.elements)
        if not int(quiet):
            _warn_if_deuterium(
                first_positions, satellites, topology.elements, first.box
            )
        index = topology.molecule_index()
        if index is None and data_file is None:
            # Without molecule IDs, the molecules are the groups of atoms connected
            # by the guessed bonds, as for xyz files.
            index = molecule_index(len(topology.ids), bonds)
        molecules = None if index is None else Molecules.from_bonds(index, bonds)
        if molecules is not None and not first.unwrapped:
            # Wrapped coordinates: after reassembly, a network bonded through the
            # boundary still has a bond longer than half the cell; no bond of a real
            # molecule is. Unwrapped coordinates need no reassembly: a sheet stays
            # in one piece and is shifted as a whole.
            half = cell_heights(first.box).min() / 2
            molecules, n_periodic = split_wrapping_molecules(
                first_positions,
                molecules,
                first.box,
                bonds,
                np.full(len(bonds), half),
            )
            if n_periodic:
                molecules = attach_satellites(molecules, bonds, satellites)
                if int(wrap):
                    warnings.warn(
                        f"{n_periodic} molecule(s) are bonded through the box "
                        "boundary (sheet, chain or crystal); their atoms are "
                        "wrapped individually",
                        stacklevel=2,
                    )
        do_wrap = bool(int(wrap)) and _can_wrap(molecules, first.unwrapped)
        # Wrapped coordinates that are not reassembled (wrap=0) show molecules cut
        # by the boundary as the file has them, with their long bonds.
        check = do_wrap or first.unwrapped

        raw = itertools.chain(
            [(first_positions, first.box, first.unwrapped)],
            ((frame.positions(), frame.box, frame.unwrapped) for frame in frames),
        )
        states = _displayed(raw, molecules, satellites, do_wrap)
        coordinates, _ = next(states)
        cut = _cut_bonds(coordinates, bonds, satellite, first.box, check)

        _self.delete(f"{name} {name}_box {name}_drude")
        model = _chempy_model(replace(topology, bonds=bonds[~cut]), coordinates)
        _self.load_model(model, name, 1)
        ids = topology.ids.tolist()
        # chempy cannot carry atom IDs; PyMOL numbers them from 0 on loading.
        _self.alter(name, "ID = ids[rank]", space={"ids": ids})
        _finish_satellites(name, ids, satellite, None, _self)
        _set_box(name, first.box, 1, show_box, _self)

        trajectory = _Trajectory(
            trajectory_path,
            int(start),
            int(stop),
            int(interval),
            molecules,
            satellites,
            ids,
            bonds,
            satellite,
            first.unwrapped,
            cut,
        )
        n_states, cut = _load_states(name, states, trajectory, check, show_box, _self)

    _remember(name, replace(trajectory, removed=cut))

    if not int(quiet):
        print(
            f" load_lammps: {n_states} frames of {len(topology.ids)} atoms "
            f"loaded into '{name}'{_periodic_summary(int(cut.sum()))}"
        )


def _xyz_states(
    path: str, start: int, stop: int, interval: int, cell: Box | None
) -> Iterator[tuple[xyz.XyzFrame, Box | None]]:
    """Yield xyz frames with the cell of each: ``cell``, else the frame's Lattice.

    A frame without either keeps the cell of the previous frame.
    """
    box = None
    for frame in xyz.read_frames(path, start, stop, interval):
        box = cell or frame.box or box
        yield frame, box


def load_xyz(
    filename: str,
    object: str = "",
    start: int | str = 1,
    stop: int | str = -1,
    interval: int | str = 1,
    cell: str = "",
    box: int | str = 1,
    wrap: int | str = 1,
    quiet: int | str = 0,
    drude: int | str = "auto",
    *,
    _self: Any = cmd,
) -> None:
    """Load a multi-frame xyz trajectory (e.g. AIMD) with a periodic cell.

    Each xyz frame becomes one state of the object. xyz files have no box and no
    topology, so both are added here. The cell is ``cell`` if given, otherwise the
    ``Lattice="..."`` of each frame of an extended xyz file (per frame, so NpT
    works). If both are present, ``cell`` is used and a warning is printed.

    With a cell, the box is drawn as ``<object>_box`` and set as crystal symmetry
    (from the cell parameters only), bonds are guessed from the first frame with
    the minimum-image convention (covalent radii plus 0.4 A), and whole molecules
    are reassembled and wrapped into the box in every frame. The bonds are fixed
    for all frames. Molecules are the connected components of these bonds, numbered
    from 1 in the order of their first atom and stored as ``resi``. Atom names
    are the element symbols.

    Alkali and alkaline-earth metals (Li, Na, K, Rb, Cs, Be, Mg, Ca, Sr, Ba) are
    treated as ions and get no bonds. Molecules that are bonded through the cell
    boundary (a chain, slab or crystal) cannot be reassembled; their atoms are
    wrapped one by one in the first frame and then follow their own path, and a
    warning is printed.

    Without a cell there is no box and no wrapping (``wrap`` has no effect);
    PyMOL guesses the bonds, and ``resi`` still holds the molecule number found
    from plain distances.

    The labels ``D`` (Drude particle) and ``M`` (virtual site, e.g. the M site of
    SWM4-NDP water), in any case, are kept as the elements ``D`` and ``M``: select
    them with ``elem D`` and ``elem M``. Each Drude is bonded to its nearest other
    atom and each virtual site to its nearest oxygen, whatever the distance. With a
    cell they are shown at the periodic image next to that atom in every frame,
    also with ``wrap=0``. They get the colour and radius that ``drude_style`` uses,
    but keep their representations. ``drude=0`` reads a label ``D`` as deuterium
    (element H) instead; a warning suggests it if the ``D`` atoms sit more than
    0.3 A from their bonded atom (median).

    With ``wrap=1``, bonds longer than half of the smallest cell height (periodic
    images, which PyMOL would draw through the cell) are removed from the display,
    and their number is printed; bonds of Drudes and virtual sites are never
    removed. With ``wrap=0`` all bonds are kept, as the file coordinates are shown
    as they are.

    PyMOL usage::

        load_xyz filename [, object [, start [, stop [, interval [, cell [, box
                 [, wrap [, quiet [, drude ]]]]]]]]]

    Examples::

        load_xyz aimd.xyz, cell=12.4
        load_xyz aimd.xyz, cell=12.4 12.4 15
        load_xyz aimd.xyz, cell=12.4 12.4 15 90 90 120, interval=5
        load_xyz extended.xyz, wrap=0
        select mol3, resi 3
        select drudes, elem D

    Parameters
    ----------
    filename : str
        Xyz file, plain or extended, optionally ``.gz``.
    object : str, optional
        Object name; defaults to the file name without extension. An existing
        object of that name is replaced.
    start, stop, interval : int, optional
        First frame (1-based), last frame (inclusive, -1 for all) and stride.
    cell : str, optional
        Cell in Angstrom and degree: ``a`` (cube), ``a b c`` (right angles) or
        ``a b c alpha beta gamma``, separated by spaces. (In PyMOL, commas split
        the arguments; they work only when calling this function from Python.)
    box : int, optional
        1 to draw the cell as the object ``<object>_box``; 0 to skip it.
    wrap : int, optional
        1 to wrap whole molecules into the cell, 0 to keep the file coordinates.
    quiet : int, optional
        1 to suppress the summary line.
    drude : int | str, optional
        ``auto`` (default) keeps labels ``D`` as Drude particles; ``0`` reads them
        as deuterium. Atom types do not exist in xyz files.

    Raises
    ------
    pymol.CmdException
        If the file or the cell cannot be interpreted, or ``drude`` is not ``auto``
        or ``0``.
    """
    name = object or _default_name(filename, _self)
    show_box = bool(int(box))
    trajectory_path = str(_self.exp_path(filename))

    with _pymol_messages("load_xyz"):
        user_cell = parse_cell(cell) if cell.strip() else None
        frames = _xyz_states(
            trajectory_path, int(start), int(stop), int(interval), user_cell
        )
        first_state = next(frames, None)
        if first_state is None:
            raise ValueError(f"{filename}: no complete frame in the selected range")
        first, first_box = first_state
        if user_cell is not None and first.box is not None:
            warnings.warn(
                "both cell and an extended-xyz Lattice are given; using cell",
                stacklevel=2,
            )

        drude_setting = parse_drude(drude)
        if drude_setting.mode == "types":
            raise ValueError("load_xyz takes drude=auto or drude=0 (no atom types)")
        elements = first.elements
        if drude_setting.mode == "off":
            elements = ["H" if e == DRUDE else e for e in elements]

        n_atoms = len(elements)
        bonds = guess_bonds(first.positions, elements, first_box)
        satellite = satellite_mask(elements)
        satellites = satellite_hosts(bonds, elements)
        if not int(quiet):
            _warn_if_deuterium(first.positions, satellites, elements, first_box)
        index = molecule_index(n_atoms, bonds)
        molecules = None
        if first_box is not None:
            molecules = Molecules.from_bonds(index, bonds)
            height = cell_heights(first_box).min()
            if max_bond_limit(elements) > height / 2:
                warnings.warn(
                    f"the smallest cell height is {height:.1f} A, less than twice the "
                    "longest bond limit; molecules may be wrapped incorrectly",
                    stacklevel=2,
                )
            molecules, n_periodic = split_wrapping_molecules(
                first.positions,
                molecules,
                first_box,
                bonds,
                bond_limits(elements, bonds),
            )
            if n_periodic:
                molecules = attach_satellites(molecules, bonds, satellites)
                warnings.warn(
                    f"{n_periodic} molecule(s) are bonded through the cell boundary "
                    "(chain, slab or crystal); their atoms are wrapped individually",
                    stacklevel=2,
                )
        do_wrap = bool(int(wrap))

        _, element_types = np.unique(elements, return_inverse=True)
        topology = Topology(
            ids=np.arange(1, n_atoms + 1),
            types=element_types.astype(np.int64) + 1,
            elements=elements,
            names=elements,
            residue_names=[""] * n_atoms,
            residue_ids=[str(i + 1) for i in index],
            charges=None,
            bonds=None if first_box is None else bonds,
        )

        raw = itertools.chain(
            [(first.positions, first_box, False)],
            ((frame.positions, frame_box, False) for frame, frame_box in frames),
        )
        states = _displayed(raw, molecules, satellites, do_wrap)
        coordinates, _ = next(states)
        # With wrap=0 the file coordinates are shown as they are, molecules cut by
        # the boundary included.
        cut = _cut_bonds(coordinates, topology.bonds, satellite, first_box, do_wrap)

        _self.delete(f"{name} {name}_box {name}_drude")
        model = _chempy_model(
            replace(
                topology,
                bonds=None if topology.bonds is None else topology.bonds[~cut],
            ),
            coordinates,
        )
        _self.load_model(model, name, 1)
        ids = topology.ids.tolist()
        _self.alter(name, "ID = ids[rank]", space={"ids": ids})
        if topology.bonds is None:
            _self.rebond(name, 1, pbc=0)
        _finish_satellites(
            name,
            ids,
            satellite,
            None if topology.bonds is not None else satellites,
            _self,
        )
        if first_box is not None:
            _set_box(name, first_box, 1, show_box, _self)

        trajectory = _Trajectory(
            trajectory_path,
            int(start),
            int(stop),
            int(interval),
            molecules,
            satellites,
            ids,
            topology.bonds,
            satellite,
            False,
            cut,
            xyz=True,
            cell=user_cell,
        )
        n_states, cut = _load_states(name, states, trajectory, do_wrap, show_box, _self)

    _remember(name, replace(trajectory, removed=cut))

    if not int(quiet):
        print(
            f" load_xyz: {n_states} frames of {n_atoms} atoms loaded into '{name}'"
            + _periodic_summary(int(cut.sum()))
            + ("" if first_box is not None else " (no cell: no box, no wrapping)")
        )


def _raw_frames(trajectory: _Trajectory) -> Iterator[_RawFrame]:
    """Read the frames of a loaded trajectory again."""
    if trajectory.xyz:
        states = _xyz_states(
            trajectory.path,
            trajectory.start,
            trajectory.stop,
            trajectory.interval,
            trajectory.cell,
        )
        for xyz_frame, xyz_box in states:
            yield xyz_frame.positions, xyz_box, False
    else:
        frames = read_frames(
            trajectory.path, trajectory.start, trajectory.stop, trajectory.interval
        )
        for frame in frames:
            yield frame.positions(), frame.box, frame.unwrapped


def lammps_wrap(
    object: str = "", wrap: int | str = 1, quiet: int | str = 0, *, _self: Any = cmd
) -> None:
    """Wrap or unwrap the molecules of an object from ``load_lammps`` or ``load_xyz``.

    The coordinates are re-read from the dump or xyz file and replace those of
    every state; representations and colours stay. Molecules are wrapped as whole
    molecules by centre of geometry. The bonds through the periodic boundary are
    found again for the new coordinates, as on loading: they are removed when
    wrapping and restored with the file coordinates of a wrapped dump. An xyz object
    loaded without a cell has nothing to wrap into; a warning is printed. All frames
    are read before the object is changed, so an error leaves it as it was.

    Without an object name, the command acts on the object loaded last with
    ``load_lammps`` or ``load_xyz`` that still exists in the session. The summary
    line names the object.

    PyMOL usage::

        lammps_wrap [ object [, wrap [, quiet ]]]

    Examples::

        lammps_wrap wrap=0        # last loaded object, coordinates as in the file
        lammps_wrap wrap=1        # last loaded object, molecules into the box
        lammps_wrap traj          # molecules into the box
        lammps_wrap traj, 0       # coordinates as in the dump (e.g. diffusion)
        lammps_wrap aimd, 0       # coordinates as in the xyz file

    Parameters
    ----------
    object : str, optional
        Object loaded with ``load_lammps`` or ``load_xyz`` in this PyMOL session.
        Empty (default): the object loaded last that still exists.
    wrap : int, optional
        1 to wrap molecules into the box, 0 to restore the file coordinates.
    quiet : int, optional
        1 to suppress the summary line.

    Raises
    ------
    pymol.CmdException
        If the object was not loaded with ``load_lammps`` or ``load_xyz`` in this
        session, or if ``object`` is empty and no such object exists.
    """
    object = object or _last_loaded(_self)
    trajectory = _TRAJECTORIES.get(object)
    if trajectory is None or object not in _self.get_names("objects"):
        raise pymol.CmdException(
            f"'{object}' was not loaded with load_lammps or load_xyz in this session"
        )
    with _pymol_messages("lammps_wrap"):
        do_wrap = bool(int(wrap))
        if trajectory.xyz and do_wrap and trajectory.molecules is None:
            warnings.warn("no cell, so molecules were not wrapped", stacklevel=2)
            do_wrap = False
        elif not trajectory.xyz:
            do_wrap = do_wrap and _can_wrap(trajectory.molecules, trajectory.unwrapped)
        check = do_wrap or trajectory.unwrapped

        # Everything is read and computed first; the object changes only after.
        states = _displayed(
            _raw_frames(trajectory),
            trajectory.molecules,
            trajectory.satellites,
            do_wrap,
        )
        coordinate_sets: list[NDArray[np.float32]] = []
        cut = np.zeros_like(trajectory.removed)
        for coordinates, box in states:
            cut |= _cut_bonds(
                coordinates, trajectory.bonds, trajectory.satellite, box, check
            )
            # PyMOL keeps single precision.
            coordinate_sets.append(coordinates.astype(np.float32))

        for state, coordinate_set in enumerate(coordinate_sets, start=1):
            _self.load_coordset(coordinate_set, object, state)
        if trajectory.bonds is not None:
            restored = trajectory.bonds[trajectory.removed & ~cut]
            _add_bonds(object, trajectory.ids, restored, _self)
            added = trajectory.bonds[cut & ~trajectory.removed]
            _remove_bonds(object, trajectory.ids, added, _self)
        _TRAJECTORIES[object] = replace(trajectory, removed=cut)
    refresh_dashes(object, _self=_self)
    if not int(quiet):
        action = "wrapped into the box" if do_wrap else "as in the file"
        print(
            f" lammps_wrap: coordinates of '{object}' {action}"
            + _periodic_summary(int(cut.sum()))
        )


def register(_self: Any = cmd) -> None:
    """Add the commands to PyMOL and let ``load`` open ``.lammpstrj`` files.

    The commands are ``load_lammps``, ``load_xyz``, ``lammps_wrap``,
    ``drude_style`` and ``drawgridbox``.

    Parameters
    ----------
    _self : Any, optional
        PyMOL command module.
    """
    _self.extend("load_lammps", load_lammps)
    _self.extend("load_xyz", load_xyz)
    _self.extend("lammps_wrap", lammps_wrap)
    _self.extend("drude_style", drude_style)
    _self.extend("drawgridbox", drawgridbox)
    for command, file_keys in FILE_ARGUMENTS.items():
        register_completion(command, file_keys, _self)
    # Internal PyMOL table; skip quietly if a future version renames it.
    loadfunctions = getattr(importing, "loadfunctions", None)
    if loadfunctions is not None:
        loadfunctions["lammpstrj"] = load_lammps
