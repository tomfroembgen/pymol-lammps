"""PyMOL commands for LAMMPS trajectories.

``register()`` adds the ``load_lammps``, ``load_xyz`` and ``lammps_wrap`` commands
and lets PyMOL's ``load`` open ``.lammpstrj`` files. Only public PyMOL API is used, except the format table that
connects ``load`` to this reader (``pymol.importing.loadfunctions``); if a PyMOL
update removes that table, ``load_lammps`` keeps working.
"""

import contextlib
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
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
    cell_heights,
    guess_bonds,
    max_bond_limit,
    molecule_index,
)
from pymol_lammps.datafile import read_data
from pymol_lammps.dump import Box, Frame, parse_cell, read_frames
from pymol_lammps.topology import Topology, build_topology
from pymol_lammps.wrapping import (
    Molecules,
    split_wrapping_molecules,
    wrap_molecules,
)

# Grey stays visible on both black and white backgrounds.
BOX_COLOR = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class _Trajectory:
    """What ``lammps_wrap`` needs to re-read the coordinates of a loaded object.

    ``xyz`` marks objects from ``load_xyz``; ``cell`` is then the user-given cell,
    which takes precedence over the ``Lattice`` of each frame.
    """

    path: str
    start: int
    stop: int
    interval: int
    molecules: Molecules | None
    xyz: bool = False
    cell: Box | None = None


# Trajectories loaded in this session, by object name.
_TRAJECTORIES: dict[str, _Trajectory] = {}


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


def _can_wrap(molecules: Molecules | None, frame: Frame) -> bool:
    """Return whether molecules can be wrapped; warn where it would matter."""
    if molecules is None:
        if frame.unwrapped:
            warnings.warn(
                "no molecule IDs (dump column mol or a data file), so molecules "
                "were not wrapped into the box",
                stacklevel=2,
            )
        return False
    return True


def _coordinates(
    frame: Frame, molecules: Molecules | None, wrap: bool
) -> NDArray[np.float64]:
    """Return the positions of a frame, with whole molecules wrapped if asked."""
    if not wrap or molecules is None:
        return frame.positions()
    return wrap_molecules(
        frame.positions(), molecules, frame.box, make_whole=not frame.unwrapped
    )


def _chempy_model(topology: Topology, positions: NDArray[Any]) -> Any:
    """Build a chempy model of the first frame, including bonds if known."""
    model = models.Indexed()
    for i in range(len(topology.ids)):
        atom = Atom()
        atom.coord = positions[i].tolist()
        atom.symbol = topology.elements[i]
        atom.name = topology.names[i]
        atom.resn = topology.residue_names[i]
        atom.resi = topology.residue_ids[i]
        atom.numeric_type = int(topology.types[i])
        atom.text_type = str(topology.types[i])
        if topology.charges is not None:
            atom.partial_charge = float(topology.charges[i])
        model.add_atom(atom)
    for i, j in topology.bonds if topology.bonds is not None else ():
        bond = Bond()
        bond.index = [int(i), int(j)]
        model.add_bond(bond)
    return model


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
    *,
    _self: Any = cmd,
) -> None:
    """Load a LAMMPS dump trajectory, with an optional data file for the topology.

    Each dump frame becomes one state of the object. Atoms are ordered by LAMMPS
    atom ID, and the atom ID is stored as PyMOL ``ID``. The box drawn for each
    frame is the one in the dump (``ITEM: BOX BOUNDS``).

    By default, whole molecules are wrapped into the box of each frame (by centre
    of geometry; molecule IDs from the dump's ``mol`` column or the data file).
    With ``wrap=0``, coordinates are kept as in the dump, unwrapped if it has
    xu yu zu or image flags ix iy iz. ``lammps_wrap`` switches this after loading.

    Without a data file, elements come from the dump's ``element`` or ``mass``
    column, and bonds are guessed from distances in the first frame. With a data
    file, masses, molecule IDs, charges, names and bonds are read from it. The
    atom type is stored as ``numeric_type`` and ``text_type``.

    PyMOL usage::

        load_lammps filename [, data [, object [, start [, stop [, interval
                    [, elements [, box [, wrap ]]]]]]]]

    Examples::

        load_lammps traj.lammpstrj
        load_lammps traj.lammpstrj, data=system.data
        load_lammps traj.lammpstrj, interval=10, elements=C H O N
        load_lammps traj.lammpstrj, data=system.data, wrap=0
        color red, text_type 3

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
        topology = build_topology(first, data_file, type_elements or None)
        index = topology.molecule_index()
        molecules = (
            None if index is None else Molecules.from_bonds(index, topology.bonds)
        )
        do_wrap = bool(int(wrap)) and _can_wrap(molecules, first)

        _self.delete(f"{name} {name}_box")
        model = _chempy_model(topology, _coordinates(first, molecules, do_wrap))
        _self.load_model(model, name, 1)
        # chempy cannot carry atom IDs; PyMOL numbers them from 0 on loading.
        _self.alter(name, "ID = ids[rank]", space={"ids": topology.ids.tolist()})
        if topology.bonds is None:
            _self.rebond(name, 1, pbc=0)
        _set_box(name, first.box, 1, show_box, _self)

        n_states = 1
        for n_states, frame in enumerate(frames, start=2):
            # load_coordset fills atoms in loading order, i.e. by atom ID.
            coordinates = _coordinates(frame, molecules, do_wrap)
            _self.load_coordset(coordinates, name, n_states)
            _set_box(name, frame.box, n_states, show_box, _self)

    _TRAJECTORIES[name] = _Trajectory(
        trajectory_path, int(start), int(stop), int(interval), molecules
    )

    if not int(quiet):
        print(
            f" load_lammps: {n_states} frames of {len(topology.ids)} atoms "
            f"loaded into '{name}'"
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


def _xyz_coordinates(
    positions: NDArray[np.float64],
    box: Box | None,
    molecules: Molecules | None,
    wrap: bool,
) -> NDArray[np.float64]:
    """Return xyz positions, with whole molecules reassembled and wrapped if asked."""
    if not wrap or molecules is None or box is None:
        return positions
    # xyz files have no image flags, so molecules cut by the boundary are rejoined.
    return wrap_molecules(positions, molecules, box, make_whole=True)


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
    wrapped one by one and a warning is printed.

    Without a cell there is no box and no wrapping (``wrap`` has no effect);
    PyMOL guesses the bonds, and ``resi`` still holds the molecule number found
    from plain distances.

    PyMOL usage::

        load_xyz filename [, object [, start [, stop [, interval [, cell [, box
                 [, wrap ]]]]]]]

    Examples::

        load_xyz aimd.xyz, cell=12.4
        load_xyz aimd.xyz, cell=12.4 12.4 15
        load_xyz aimd.xyz, cell=12.4 12.4 15 90 90 120, interval=5
        load_xyz extended.xyz, wrap=0
        select mol3, resi 3

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

    Raises
    ------
    pymol.CmdException
        If the file or the cell cannot be interpreted.
    """
    name = object or _default_name(filename, _self)
    show_box = bool(int(box))
    trajectory_path = str(_self.exp_path(filename))

    with _pymol_messages("load_xyz"):
        user_cell = parse_cell(cell) if cell.strip() else None
        states = _xyz_states(
            trajectory_path, int(start), int(stop), int(interval), user_cell
        )
        first_state = next(states, None)
        if first_state is None:
            raise ValueError(f"{filename}: no complete frame in the selected range")
        first, first_box = first_state
        if user_cell is not None and first.box is not None:
            warnings.warn(
                "both cell and an extended-xyz Lattice are given; using cell",
                stacklevel=2,
            )

        n_atoms = len(first.elements)
        bonds = guess_bonds(first.positions, first.elements, first_box)
        index = molecule_index(n_atoms, bonds)
        molecules = None
        if first_box is not None:
            molecules = Molecules.from_bonds(index, bonds)
            height = cell_heights(first_box).min()
            if max_bond_limit(first.elements) > height / 2:
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
                bond_limits(first.elements, bonds),
            )
            if n_periodic:
                warnings.warn(
                    f"{n_periodic} molecule(s) are bonded through the cell boundary "
                    "(chain, slab or crystal); their atoms are wrapped individually",
                    stacklevel=2,
                )
        do_wrap = bool(int(wrap))

        _, element_types = np.unique(first.elements, return_inverse=True)
        topology = Topology(
            ids=np.arange(1, n_atoms + 1),
            types=element_types.astype(np.int64) + 1,
            elements=first.elements,
            names=first.elements,
            residue_names=[""] * n_atoms,
            residue_ids=[str(i + 1) for i in index],
            charges=None,
            bonds=None if first_box is None else bonds,
        )

        _self.delete(f"{name} {name}_box")
        model = _chempy_model(
            topology, _xyz_coordinates(first.positions, first_box, molecules, do_wrap)
        )
        _self.load_model(model, name, 1)
        _self.alter(name, "ID = ids[rank]", space={"ids": topology.ids.tolist()})
        if topology.bonds is None:
            _self.rebond(name, 1, pbc=0)
        if first_box is not None:
            _set_box(name, first_box, 1, show_box, _self)

        n_states = 1
        for n_states, (frame, frame_box) in enumerate(states, start=2):
            coordinates = _xyz_coordinates(
                frame.positions, frame_box, molecules, do_wrap
            )
            _self.load_coordset(coordinates, name, n_states)
            if first_box is not None and frame_box is not None:
                _set_box(name, frame_box, n_states, show_box, _self)

    _TRAJECTORIES[name] = _Trajectory(
        trajectory_path,
        int(start),
        int(stop),
        int(interval),
        molecules,
        xyz=True,
        cell=user_cell,
    )

    if not int(quiet):
        print(
            f" load_xyz: {n_states} frames of {n_atoms} atoms loaded into '{name}'"
            + ("" if first_box is not None else " (no cell: no box, no wrapping)")
        )


def lammps_wrap(
    object: str, wrap: int | str = 1, quiet: int | str = 0, *, _self: Any = cmd
) -> None:
    """Wrap or unwrap the molecules of an object from ``load_lammps`` or ``load_xyz``.

    The coordinates are re-read from the dump or xyz file and replace those of
    every state; representations and colours stay. Molecules are wrapped as whole
    molecules by centre of geometry. An xyz object loaded without a cell has
    nothing to wrap into; a warning is printed.

    PyMOL usage::

        lammps_wrap object [, wrap ]

    Examples::

        lammps_wrap traj          # molecules into the box
        lammps_wrap traj, 0       # coordinates as in the dump (e.g. diffusion)
        lammps_wrap aimd, 0       # coordinates as in the xyz file

    Parameters
    ----------
    object : str
        Object loaded with ``load_lammps`` or ``load_xyz`` in this PyMOL session.
    wrap : int, optional
        1 to wrap molecules into the box, 0 to restore the file coordinates.
    quiet : int, optional
        1 to suppress the summary line.

    Raises
    ------
    pymol.CmdException
        If the object was not loaded with ``load_lammps`` or ``load_xyz`` in this
        session.
    """
    trajectory = _TRAJECTORIES.get(object)
    if trajectory is None or object not in _self.get_names("objects"):
        raise pymol.CmdException(
            f"'{object}' was not loaded with load_lammps or load_xyz in this session"
        )
    with _pymol_messages("lammps_wrap"):
        do_wrap = bool(int(wrap))
        if trajectory.xyz:
            if do_wrap and trajectory.molecules is None:
                warnings.warn("no cell, so molecules were not wrapped", stacklevel=2)
                do_wrap = False
            states = _xyz_states(
                trajectory.path,
                trajectory.start,
                trajectory.stop,
                trajectory.interval,
                trajectory.cell,
            )
            for state, (xyz_frame, xyz_box) in enumerate(states, start=1):
                coordinates = _xyz_coordinates(
                    xyz_frame.positions, xyz_box, trajectory.molecules, do_wrap
                )
                _self.load_coordset(coordinates, object, state)
        else:
            frames = read_frames(
                trajectory.path, trajectory.start, trajectory.stop, trajectory.interval
            )
            for state, frame in enumerate(frames, start=1):
                if state == 1:
                    do_wrap = do_wrap and _can_wrap(trajectory.molecules, frame)
                coordinates = _coordinates(frame, trajectory.molecules, do_wrap)
                _self.load_coordset(coordinates, object, state)
    if not int(quiet):
        action = "wrapped into the box" if do_wrap else "as in the file"
        print(f" lammps_wrap: coordinates of '{object}' {action}")


def register(_self: Any = cmd) -> None:
    """Add the commands to PyMOL and let ``load`` open ``.lammpstrj`` files.

    Parameters
    ----------
    _self : Any, optional
        PyMOL command module.
    """
    _self.extend("load_lammps", load_lammps)
    _self.extend("load_xyz", load_xyz)
    _self.extend("lammps_wrap", lammps_wrap)
    # Internal PyMOL table; skip quietly if a future version renames it.
    loadfunctions = getattr(importing, "loadfunctions", None)
    if loadfunctions is not None:
        loadfunctions["lammpstrj"] = load_lammps
