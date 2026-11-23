"""PyMOL commands for LAMMPS trajectories.

``register()`` adds the ``load_lammps`` command and lets PyMOL's ``load`` open
``.lammpstrj`` files. Only public PyMOL API is used, except the format table that
connects ``load`` to this reader (``pymol.importing.loadfunctions``); if a PyMOL
update removes that table, ``load_lammps`` keeps working.
"""

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pymol
from chempy import Atom, Bond, models
from numpy.typing import NDArray
from pymol import cgo, cmd, importing

from pymol_lammps.datafile import read_data
from pymol_lammps.dump import Box, read_frames
from pymol_lammps.topology import Topology, build_topology

# Grey stays visible on both black and white backgrounds.
BOX_COLOR = (0.5, 0.5, 0.5)


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
    quiet: int | str = 0,
    *,
    _self: Any = cmd,
) -> None:
    """Load a LAMMPS dump trajectory, with an optional data file for the topology.

    Each dump frame becomes one state of the object. Atoms are ordered by LAMMPS
    atom ID, and the atom ID is stored as PyMOL ``ID``. Unwrapped coordinates are
    used if the dump has them (xu yu zu, or image flags ix iy iz).

    Without a data file, elements come from the dump's ``element`` or ``mass``
    column, and bonds are guessed from distances in the first frame. With a data
    file, masses, molecule IDs, charges, names and bonds are read from it. The
    atom type is stored as ``numeric_type`` and ``text_type``.

    PyMOL usage::

        load_lammps filename [, data [, object [, start [, stop [, interval
                    [, elements [, box ]]]]]]]

    Examples::

        load_lammps traj.lammpstrj
        load_lammps traj.lammpstrj, data=system.data
        load_lammps traj.lammpstrj, interval=10, elements=C H O N
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

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            frames = read_frames(
                _self.exp_path(filename), int(start), int(stop), int(interval)
            )
            first = next(frames, None)
            if first is None:
                raise ValueError(f"{filename}: no complete frame in the selected range")
            data_file = read_data(_self.exp_path(data)) if data else None
            topology = build_topology(first, data_file, type_elements or None)

            _self.delete(f"{name} {name}_box")
            _self.load_model(_chempy_model(topology, first.positions()), name, 1)
            # chempy cannot carry atom IDs; PyMOL numbers them from 0 on loading.
            _self.alter(name, "ID = ids[rank]", space={"ids": topology.ids.tolist()})
            if topology.bonds is None:
                _self.rebond(name, 1, pbc=0)
            _set_box(name, first.box, 1, show_box, _self)

            n_states = 1
            for n_states, frame in enumerate(frames, start=2):
                # load_coordset fills atoms in loading order, i.e. by atom ID.
                _self.load_coordset(frame.positions(), name, n_states)
                _set_box(name, frame.box, n_states, show_box, _self)
        except ValueError as error:
            raise pymol.CmdException(str(error)) from error
        finally:
            for warning in caught:
                print(f" load_lammps warning: {warning.message}")

    if not int(quiet):
        print(
            f" load_lammps: {n_states} frames of {len(topology.ids)} atoms "
            f"loaded into '{name}'"
        )


def register(_self: Any = cmd) -> None:
    """Add ``load_lammps`` to PyMOL and let ``load`` open ``.lammpstrj`` files.

    Parameters
    ----------
    _self : Any, optional
        PyMOL command module.
    """
    _self.extend("load_lammps", load_lammps)
    # Internal PyMOL table; skip quietly if a future version renames it.
    loadfunctions = getattr(importing, "loadfunctions", None)
    if loadfunctions is not None:
        loadfunctions["lammpstrj"] = load_lammps
