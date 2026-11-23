"""Combine a dump frame and an optional data file into per-atom properties.

Each property is taken from the most specific source available:

- element: explicit per-type list > dump ``element`` column > mass (dump ``mass``
  column, then data-file ``Masses``) > carbon as placeholder, with a warning
- atom name: data-file atom comment > type name (type label or ``Masses`` comment)
  > type number
- residue name: data-file atom comment (fftool writes the species name there)
- residue number: dump ``mol`` column > data-file molecule ID
- charge: dump ``q`` column > data-file charge
- bonds: data file; without a data file they are left to distance-based guessing
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pymol_lammps.datafile import DataFile
from pymol_lammps.dump import Frame
from pymol_lammps.elements import element_from_mass

# Placeholder for atoms whose element cannot be inferred. Carbon gives sensible
# distance-based bonds for organic molecules.
FALLBACK_ELEMENT = "C"


@dataclass(frozen=True)
class Topology:
    """Per-atom properties in ascending atom-ID order.

    Attributes
    ----------
    ids : NDArray[np.int64]
        LAMMPS atom IDs.
    types : NDArray[np.int64]
        Numeric LAMMPS atom types.
    elements : list[str]
        Element symbols.
    names : list[str]
        Atom names.
    residue_names : list[str]
        Residue names, empty where unknown.
    residue_ids : list[str]
        Residue numbers (LAMMPS molecule IDs), empty where unknown.
    charges : NDArray[np.float64] | None
        Partial charges, None if neither dump nor data file has them.
    bonds : NDArray[np.int64] | None
        Bonded atom pairs as indices into ``ids``, shape (n_bonds, 2). None means
        no bond information is available and bonds should be guessed.
    """

    ids: NDArray[np.int64]
    types: NDArray[np.int64]
    elements: list[str]
    names: list[str]
    residue_names: list[str]
    residue_ids: list[str]
    charges: NDArray[np.float64] | None
    bonds: NDArray[np.int64] | None


def _positions_in(
    sorted_ids: NDArray[np.int64], query: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Return the index of each queried ID in a sorted ID array, -1 where absent."""
    index = np.minimum(np.searchsorted(sorted_ids, query), len(sorted_ids) - 1)
    return np.where(sorted_ids[index] == query, index, -1)


def _elements(
    frame: Frame,
    types: NDArray[np.int64],
    data: DataFile | None,
    type_elements: Sequence[str] | None,
) -> list[str]:
    """Resolve the element of each atom, warning about unresolved atom types.

    Raises
    ------
    ValueError
        If ``type_elements`` has fewer entries than the highest atom type.
    """
    if type_elements:
        if len(type_elements) < types.max():
            raise ValueError(
                f"{len(type_elements)} elements given, but the dump has atom "
                f"type {types.max()}; give one element per atom type, in type order"
            )
        return [type_elements[t - 1] for t in types]
    if "element" in frame.columns:
        return frame.column("element").tolist()

    if "mass" in frame.columns:
        masses = frame.column("mass").astype(float)
    elif data is not None:
        masses = np.array([data.type_masses.get(int(t), 0.0) for t in types])
    else:
        masses = np.zeros(len(types))
    by_mass = {m: element_from_mass(m) for m in np.unique(masses)}
    elements = [by_mass[m] or FALLBACK_ELEMENT for m in masses]

    unresolved = sorted({int(t) for t, m in zip(types, masses) if not by_mass[m]})
    if unresolved:
        warnings.warn(
            f"no element known for atom types {unresolved}; shown as "
            f"{FALLBACK_ELEMENT}. Pass a data file with masses, or the elements "
            "per type (e.g. elements=C H O N)",
            stacklevel=3,
        )
    return elements


def build_topology(
    frame: Frame,
    data: DataFile | None = None,
    type_elements: Sequence[str] | None = None,
) -> Topology:
    """Build per-atom properties for the atoms of a dump frame.

    Parameters
    ----------
    frame : Frame
        First frame of the dump.
    data : DataFile | None, optional
        Data file of the same system. The dump may contain a subset of its atoms
        (``dump`` on a group); bonds to atoms outside the dump are dropped.
    type_elements : Sequence[str] | None, optional
        Element of each atom type in type order (type 1 first). Overrides all
        other element sources.

    Returns
    -------
    Topology
        Per-atom properties in ascending atom-ID order.

    Raises
    ------
    ValueError
        If the dump has non-numeric atom types, or atoms missing from the data
        file.
    """
    ids = frame.ids
    type_column = frame.column("type")
    if not all(t.isdigit() for t in type_column):
        raise ValueError(
            "the dump uses atom type labels; write numeric types instead "
            "(dump_modify ... types numeric)"
        )
    types = type_column.astype(np.int64)

    names = [str(t) for t in types]
    residue_names = [""] * len(ids)
    residue_ids = [""] * len(ids)
    charges = None
    bonds = None

    if data is not None:
        rows = _positions_in(data.ids, ids)
        if (rows < 0).any():
            missing = ids[rows < 0]
            raise ValueError(
                f"{len(missing)} atom IDs of the dump are missing from the data file "
                f"(first: {missing[0]}); do the files belong to the same system?"
            )
        names = [
            data.atom_names[r] or data.type_names.get(int(t), str(t))
            for r, t in zip(rows, types)
        ]
        residue_names = [data.residue_names[r] for r in rows]
        if data.mol_ids is not None:
            residue_ids = [str(m) for m in data.mol_ids[rows]]
        if data.charges is not None:
            charges = data.charges[rows]
        # Keep only bonds between dumped atoms (the dump may cover a group).
        index = _positions_in(ids, data.bonds)
        bonds = index[(index >= 0).all(axis=1)]

    if "mol" in frame.columns:
        residue_ids = frame.column("mol").tolist()
    if "q" in frame.columns:
        charges = frame.column("q").astype(float)

    return Topology(
        ids=ids,
        types=types,
        elements=_elements(frame, types, data, type_elements),
        names=names,
        residue_names=residue_names,
        residue_ids=residue_ids,
        charges=charges,
        bonds=bonds,
    )
