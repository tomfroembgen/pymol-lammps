"""PyMOL commands for LAMMPS trajectories.

``register()`` adds the ``load_lammps`` and ``lammps_wrap`` commands and lets
PyMOL's ``load`` open ``.lammpstrj`` files. Only public PyMOL API is used, except the format table that
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

from pymol_lammps.datafile import read_data
from pymol_lammps.dump import Box, Frame, read_frames
from pymol_lammps.topology import Topology, build_topology
from pymol_lammps.wrapping import Molecules, wrap_molecules

# Grey stays visible on both black and white backgrounds.
BOX_COLOR = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class _Trajectory:
    """What ``lammps_wrap`` needs to re-read the coordinates of a loaded object."""

    path: str
    start: int
    stop: int
    interval: int
    molecules: Molecules | None


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


def lammps_wrap(
    object: str, wrap: int | str = 1, quiet: int | str = 0, *, _self: Any = cmd
) -> None:
    """Wrap or unwrap the molecules of an object loaded with ``load_lammps``.

    The coordinates are re-read from the dump file and replace those of every
    state; representations and colours stay. Molecules are wrapped as whole
    molecules by centre of geometry.

    PyMOL usage::

        lammps_wrap object [, wrap ]

    Examples::

        lammps_wrap traj          # molecules into the box
        lammps_wrap traj, 0       # coordinates as in the dump (e.g. diffusion)

    Parameters
    ----------
    object : str
        Object loaded with ``load_lammps`` in this PyMOL session.
    wrap : int, optional
        1 to wrap molecules into the box, 0 to restore the dump coordinates.
    quiet : int, optional
        1 to suppress the summary line.

    Raises
    ------
    pymol.CmdException
        If the object was not loaded with ``load_lammps`` in this session.
    """
    trajectory = _TRAJECTORIES.get(object)
    if trajectory is None or object not in _self.get_names("objects"):
        raise pymol.CmdException(
            f"'{object}' was not loaded with load_lammps in this session"
        )
    with _pymol_messages("lammps_wrap"):
        frames = read_frames(
            trajectory.path, trajectory.start, trajectory.stop, trajectory.interval
        )
        do_wrap = bool(int(wrap))
        for state, frame in enumerate(frames, start=1):
            if state == 1:
                do_wrap = do_wrap and _can_wrap(trajectory.molecules, frame)
            coordinates = _coordinates(frame, trajectory.molecules, do_wrap)
            _self.load_coordset(coordinates, object, state)
    if not int(quiet):
        action = "wrapped into the box" if do_wrap else "as in the dump"
        print(f" lammps_wrap: coordinates of '{object}' {action}")


def register(_self: Any = cmd) -> None:
    """Add the commands to PyMOL and let ``load`` open ``.lammpstrj`` files.

    Parameters
    ----------
    _self : Any, optional
        PyMOL command module.
    """
    _self.extend("load_lammps", load_lammps)
    _self.extend("lammps_wrap", lammps_wrap)
    # Internal PyMOL table; skip quietly if a future version renames it.
    loadfunctions = getattr(importing, "loadfunctions", None)
    if loadfunctions is not None:
        loadfunctions["lammpstrj"] = load_lammps
