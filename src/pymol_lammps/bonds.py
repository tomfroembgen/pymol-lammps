"""Guess bonds and molecules from atom positions, for files without topology.

Two atoms are bonded if their distance is below the sum of their covalent radii plus
a tolerance. In a periodic cell the distance is the minimum-image distance, so bonds
across the cell boundary are found. Molecules are the connected components of the
bond graph.

Alkali and alkaline-earth metals are treated as ions and get no bonds, so that a
cation does not merge with its solvation shell into one molecule. All other
elements, including other metals, are bonded by covalent radii.

Drude particles (``D``) and virtual sites (``M``) have no covalent radius. Each
Drude is bonded to its nearest atom that is no Drude or virtual site, and each
virtual site to its nearest oxygen (or, if the system has no oxygen, its nearest
other atom), whatever the distance. They are left out of the radius-based search,
so they neither get other bonds nor create bonds between other atoms. The host is
found with a cell list (atoms sorted into a grid of small cells), so the search
time grows linearly with the number of atoms.
"""

import itertools
import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from pymol_lammps.drude import DRUDE, SATELLITES, VIRTUAL
from pymol_lammps.dump import Box
from pymol_lammps.elements import element_symbol

# Covalent radii in Angstrom (Cordero et al., Dalton Trans. 2008, 2832; single-bond,
# low-spin values for Mn, Fe and Co), for the elements in ``ELEMENT_MASSES``.
COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
    "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
    "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42,
    "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40, "Cs": 2.44,
    "Ba": 2.15, "La": 2.07, "Gd": 1.96, "W": 1.62, "Pt": 1.36, "Au": 1.36,
    "Hg": 1.32, "Pb": 1.46, "Bi": 1.48, "U": 1.96,
}  # fmt: skip

# Ions: never bonded, each one is its own molecule.
IONS = frozenset({"Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba"})

# A pair is bonded if MIN_DISTANCE < d < r_i + r_j + BOND_TOLERANCE (Angstrom). The
# lower limit drops overlapping atoms, which are not bonds.
BOND_TOLERANCE = 0.4
MIN_DISTANCE = 0.4

# Radius in Angstrom that Drude particles and virtual sites contribute to the distance
# limit of their bond; with the radius of the core and BOND_TOLERANCE the limit is at
# least 1.2 A, well above the usual 0.1-0.4 A of a Drude.
SATELLITE_RADIUS = 0.5

# Atoms per block of the pair search; a block needs about chunk * n_atoms * 24 bytes.
_CHUNK = 128

# Element whose covalent radius bonds an atom with an unknown element label.
FALLBACK_BOND_ELEMENT = "C"

# Radius in Angstrom of the cell-list search for the host of a satellite. A Drude
# sits 0.1-0.4 A from its core and an M site about 0.25 A from its oxygen; a satellite
# without any candidate this close is searched among all atoms instead.
HOST_SEARCH_RADIUS = 2.0


def bonding_elements(labels: Sequence[str]) -> list[str]:
    """Return the element whose covalent radius bonds each atom.

    The element labels of a dump (``element`` column or ``elements=``) are free
    text, so each label is reduced to an element symbol by
    ``pymol_lammps.elements.element_symbol`` (``"CL"`` is Cl, ``"OW"`` is O). A
    label that does not start with an element is bonded like carbon, with a
    warning.

    Parameters
    ----------
    labels : Sequence[str]
        Element label of each atom, with ``D`` and ``M`` for the satellites.

    Returns
    -------
    list[str]
        Element symbol of each atom, a key of ``COVALENT_RADII`` or a satellite.
    """
    symbols = {label: element_symbol(label) for label in set(labels)}
    unknown = sorted(label for label, symbol in symbols.items() if not symbol)
    if unknown:
        warnings.warn(
            f"no covalent radius for the element labels {unknown}; their bonds are "
            f"guessed as for {FALLBACK_BOND_ELEMENT}",
            stacklevel=2,
        )
    return [symbols[label] or FALLBACK_BOND_ELEMENT for label in labels]


def cell_heights(box: Box) -> NDArray[np.float64]:
    """Return the distances between the opposite faces of the cell.

    Parameters
    ----------
    box : Box
        Periodic cell.

    Returns
    -------
    NDArray[np.float64]
        Height of the cell along a, b and c in Angstrom, shape (3,). Rounding the
        fractional difference of two atoms gives their true minimum image only if
        their distance is below half of the smallest height.
    """
    a, b, c = box.matrix.T
    volume = abs(np.linalg.det(box.matrix))
    areas = np.array(
        [np.linalg.norm(np.cross(b, c)), np.linalg.norm(np.cross(c, a))]
        + [np.linalg.norm(np.cross(a, b))]
    )
    return volume / areas


def max_bond_limit(elements: list[str]) -> float:
    """Return the largest distance below which two of these elements are bonded."""
    radii = [COVALENT_RADII[e] for e in set(elements) - IONS - SATELLITES]
    return 2 * max(radii) + BOND_TOLERANCE if radii else 0.0


def _radii(elements: list[str]) -> NDArray[np.float64]:
    """Return the covalent radius of each atom; satellites get ``SATELLITE_RADIUS``."""
    return np.array(
        [
            SATELLITE_RADIUS if element in SATELLITES else COVALENT_RADII[element]
            for element in elements
        ]
    )


def bond_limits(elements: list[str], bonds: NDArray[np.int64]) -> NDArray[np.float64]:
    """Return the distance limit (sum of radii plus tolerance) of each bond."""
    radii = _radii(elements)
    return radii[bonds[:, 0]] + radii[bonds[:, 1]] + BOND_TOLERANCE


def periodic_bonds(
    positions: NDArray[np.float64],
    bonds: NDArray[np.int64],
    satellite: NDArray[np.bool_],
    box: Box,
) -> NDArray[np.bool_]:
    """Find the bonds that PyMOL would draw through the cell.

    A bond longer than half of the smallest cell height cannot be a real bond: it
    connects two periodic images, or two halves of a molecule cut by the boundary.
    The test is made on the coordinates as they are displayed. Bonds of Drude
    particles and virtual sites are never reported, because they are short and move
    with their host atom.

    Parameters
    ----------
    positions : NDArray[np.float64]
        Displayed Cartesian positions in Angstrom, shape (n_atoms, 3).
    bonds : NDArray[np.int64]
        Bonded atom pairs, shape (n_bonds, 2).
    satellite : NDArray[np.bool_]
        Which atoms are Drude particles or virtual sites, shape (n_atoms,).
    box : Box
        Periodic cell of the frame.

    Returns
    -------
    NDArray[np.bool_]
        True for each bond to remove, shape (n_bonds,).
    """
    delta = np.take(positions, bonds[:, 0], axis=0)
    delta -= np.take(positions, bonds[:, 1], axis=0)
    too_long = np.einsum("ij,ij->i", delta, delta) > (cell_heights(box).min() / 2) ** 2
    return too_long & ~(satellite[bonds[:, 0]] | satellite[bonds[:, 1]])


def _distance_block(
    coordinates: NDArray[np.float64],
    rows: NDArray[np.int64],
    box: Box | None,
    shifts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the distances of some atoms to all atoms, shape (len(rows), n_atoms).

    ``coordinates`` are Cartesian without a box and fractional with one; ``shifts``
    are the neighbouring images to test (one zero row for plain rounding).
    """
    delta = coordinates[None, :, :] - coordinates[rows, None, :]
    if box is None:
        return np.linalg.norm(delta, axis=2)
    delta -= np.round(delta)
    return np.min(
        [np.linalg.norm((delta + s) @ box.matrix.T, axis=2) for s in shifts], axis=0
    )


def _search_setup(
    positions: NDArray[np.float64], elements: list[str], box: Box | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the coordinates for ``_distance_block`` and the image shifts to test."""
    if box is None:
        coordinates = positions
    else:
        coordinates = (positions - box.origin) @ np.linalg.inv(box.matrix).T
    skewed = box is not None and max_bond_limit(elements) > cell_heights(box).min() / 2
    shifts = (
        np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)
        if skewed
        else np.zeros((1, 3))
    )
    return coordinates, shifts


def _neighbour_pairs(
    positions: NDArray[np.float64],
    queries: NDArray[np.int64],
    candidates: NDArray[np.int64],
    box: Box | None,
    cutoff: float,
    half: bool = False,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Find all pairs of a query atom and a candidate atom closer than ``cutoff``.

    The atoms are sorted into a grid of cells that are at least ``cutoff`` high, so
    the partners of a query atom lie in its own cell or in one of the 26 cells
    around it (a cell list). In a periodic cell the grid wraps around, and the
    distance is the minimum-image distance, which is exact for ``cutoff`` below half
    of the smallest cell height.

    If the query atoms are also the candidates, ``half`` finds each pair once: of
    two opposite neighbour cells only one is searched, and within the own cell only
    the pairs with the higher candidate index. This halves the number of distances.

    Parameters
    ----------
    positions : NDArray[np.float64]
        Cartesian positions in Angstrom, shape (n_atoms, 3).
    queries, candidates : NDArray[np.int64]
        Indices of the atoms to search for and of the atoms to search among.
    box : Box | None
        Periodic cell, or None for plain distances.
    cutoff : float
        Largest distance in Angstrom.
    half : bool, optional
        True if ``queries`` and ``candidates`` are the same atoms; each pair is then
        returned once, and no atom is paired with itself.

    Returns
    -------
    tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]
        Query atom, candidate atom and distance of each pair closer than ``cutoff``.
    """
    periodic = box is not None
    grid = box
    if grid is None:
        # A non-periodic grid over the extent of the atoms.
        extent = np.maximum(np.ptp(positions, axis=0), cutoff)
        grid = Box(origin=positions.min(axis=0), matrix=np.diag(extent))
    fractional = (positions - grid.origin) @ np.linalg.inv(grid.matrix).T
    in_cell = fractional - np.floor(fractional) if periodic else fractional
    n_cells = np.maximum((cell_heights(grid) / cutoff).astype(np.int64), 1)
    # Few cells per atom are enough; larger cells only add candidates to test.
    limit = 8 * len(candidates) + 27
    if n_cells.prod() > limit:
        scale = (limit / n_cells.prod()) ** (1 / 3)
        n_cells = np.maximum((n_cells * scale).astype(np.int64), 1)
    cells = np.minimum((in_cell * n_cells).astype(np.int64), n_cells - 1)

    keys = np.ravel_multi_index(tuple(cells[candidates].T), tuple(n_cells))
    by_cell = candidates[np.argsort(keys, kind="stable")]
    if half:
        # The same atoms in the order of the cells, which makes the memory access
        # below more local.
        queries = by_cell
    counts = np.bincount(keys, minlength=int(n_cells.prod()))
    starts = np.cumsum(counts) - counts
    if half:
        # The own cell (first) and one cell of each opposite pair.
        stencil = [s for s in itertools.product((-1, 0, 1), repeat=3) if s >= (0, 0, 0)]
    else:
        # Neighbouring cells along each axis; with fewer than three cells the
        # periodic neighbours repeat, so each is taken once.
        steps = [(0,) if n == 1 else (0, 1) if n == 2 else (-1, 0, 1) for n in n_cells]
        stencil = list(
            itertools.product(*(s if periodic else (-1, 0, 1) for s in steps))
        )
    # Cell index of every query atom shifted by -1, 0 and +1 cells along each axis,
    # shape (3, n_queries, 3); the key of a neighbour cell is the sum of its three
    # parts (C order, as np.ravel_multi_index).
    shifted = cells[queries][None, :, :] + np.arange(-1, 2)[:, None, None]
    if periodic:
        shifted %= n_cells
    valid = (shifted >= 0) & (shifted < n_cells)
    part = np.where(
        valid, shifted * np.array([n_cells[1] * n_cells[2], n_cells[2], 1]), 0
    )
    a, b, c = (np.array(stencil) + 1).T
    cell_keys = (part[a, :, 0] + part[b, :, 1] + part[c, :, 2]).ravel()
    inside = (valid[a, :, 0] & valid[b, :, 1] & valid[c, :, 2]).ravel()
    number = np.where(inside, counts[cell_keys], 0)
    # Position of every candidate of these cells in ``by_cell``.
    offsets = np.arange(number.sum()) - np.repeat(np.cumsum(number) - number, number)
    query = np.repeat(np.tile(queries, len(stencil)), number)
    # np.take is faster than fancy indexing for these long index arrays.
    candidate = np.take(by_cell, np.repeat(starts[cell_keys], number) + offsets)
    if half:
        # The pairs of the own cell come first; each is kept once.
        own = number[: len(queries)].sum()
        keep = np.ones(len(query), dtype=bool)
        keep[:own] = candidate[:own] > query[:own]
        query, candidate = query[keep], candidate[keep]
        if periodic and (n_cells < 3).any():
            # With fewer than three cells along an axis, the cells on both sides
            # are the same, so pairs are found twice (and atoms paired with
            # themselves).
            pairs = np.sort(np.column_stack([query, candidate]), axis=1)
            pairs = np.unique(pairs, axis=0)
            query, candidate = pairs[pairs[:, 0] != pairs[:, 1]].T

    if box is None:
        delta = np.take(positions, candidate, axis=0) - np.take(
            positions, query, axis=0
        )
    else:
        shift = np.take(fractional, candidate, axis=0)
        shift -= np.take(fractional, query, axis=0)
        delta = (shift - np.round(shift)) @ box.matrix.T
    squared = np.einsum("ij,ij->i", delta, delta)
    close = squared < cutoff**2
    return query[close], candidate[close], np.sqrt(squared[close])


def _nearest(
    positions: NDArray[np.float64],
    queries: NDArray[np.int64],
    candidates: NDArray[np.int64],
    box: Box | None,
) -> NDArray[np.int64]:
    """Return the nearest candidate atom of each query atom (minimum image in a box).

    The cell list finds the nearest candidate within ``HOST_SEARCH_RADIUS``; query
    atoms without one are compared with all candidates (``_nearest_of_all``).
    """
    radius = HOST_SEARCH_RADIUS
    if box is not None:
        radius = min(radius, cell_heights(box).min() / 2)
    query, candidate, distance = _neighbour_pairs(
        positions, queries, candidates, box, radius
    )
    order = np.lexsort((distance, query))
    query, candidate = query[order], candidate[order]
    # The first pair of each query atom is its closest.
    closest = np.diff(query, prepend=-1) != 0
    nearest = np.full(len(positions), -1)
    nearest[query[closest]] = candidate[closest]

    for atom in queries[nearest[queries] < 0]:
        nearest[atom] = _nearest_of_all(positions, atom, candidates, box)
    return nearest[queries]


def _nearest_of_all(
    positions: NDArray[np.float64],
    atom: int,
    candidates: NDArray[np.int64],
    box: Box | None,
) -> int:
    """Return the nearest of all candidates to one atom, with the exact minimum image.

    Rounding the fractional difference gives an image at distance ``bound``. The
    nearest image is at most ``bound`` away, so its fractional coordinate along
    each cell vector differs from the rounded one by at most
    ``bound / height + 1/2``; all these images are tested. This is exact also in
    strongly skewed cells, and cheap because few atoms need it.
    """
    if box is None:
        delta = positions[candidates] - positions[atom]
        return int(candidates[np.argmin(np.linalg.norm(delta, axis=1))])
    inverse = np.linalg.inv(box.matrix).T
    fractional = (positions[candidates] - positions[atom]) @ inverse
    fractional -= np.round(fractional)
    distance = np.linalg.norm(fractional @ box.matrix.T, axis=1)
    reach = np.ceil(distance.min() / cell_heights(box) + 0.5).astype(np.int64)
    for shift in itertools.product(*(range(-r, r + 1) for r in reach)):
        shifted = np.linalg.norm((fractional + shift) @ box.matrix.T, axis=1)
        distance = np.minimum(distance, shifted)
    return int(candidates[np.argmin(distance)])


def satellite_bonds(
    positions: NDArray[np.float64],
    elements: list[str],
    box: Box | None = None,
    only: NDArray[np.bool_] | None = None,
) -> NDArray[np.int64]:
    """Bond each Drude particle and virtual site to its host atom.

    A Drude (``D``) is bonded to the nearest atom that is neither a Drude nor a
    virtual site, a virtual site (``M``) to the nearest oxygen, or to the nearest
    such atom if the system has no oxygen. The distance does not matter, because
    the host is always the closest candidate (minimum image in a periodic cell).
    A system without any candidate (a dump of the Drudes only) gives no bonds.

    Parameters
    ----------
    positions : NDArray[np.float64]
        Cartesian positions in Angstrom, shape (n_atoms, 3).
    elements : list[str]
        Element symbol of each atom, with ``D`` and ``M`` for the satellites.
    box : Box | None, optional
        Periodic cell for the minimum-image convention. None uses plain distances.
    only : NDArray[np.bool_] | None, optional
        Bond only the satellites marked here, shape (n_atoms,); None for all.

    Returns
    -------
    NDArray[np.int64]
        One pair ``(i, j)`` with ``i < j`` per satellite, shape (n_satellites, 2).
    """
    symbols = np.asarray(elements)
    satellite = np.isin(symbols, [DRUDE, VIRTUAL])
    cores = ~satellite
    if only is not None:
        satellite &= only
    pairs: list[NDArray[np.int64]] = [np.empty((0, 2), dtype=np.int64)]
    if not satellite.any():
        return pairs[0]
    oxygen = cores & (symbols == "O")
    virtual = satellite & (symbols == VIRTUAL)
    # Virtual sites choose among oxygens if there are any.
    for members, allowed in (
        (satellite & ~virtual, cores),
        (virtual, oxygen if oxygen.any() else cores),
    ):
        queries = np.flatnonzero(members)
        candidates = np.flatnonzero(allowed)
        if len(queries) and len(candidates):
            hosts = _nearest(positions, queries, candidates, box)
            pairs.append(np.column_stack([queries, hosts]).astype(np.int64))
    # In the order of the satellites, each pair sorted.
    found = np.concatenate(pairs)
    return np.sort(found[np.argsort(found[:, 0])], axis=1)


def _bonds_by_radii(
    positions: NDArray[np.float64], elements: list[str], box: Box | None
) -> NDArray[np.int64]:
    """Return the pairs closer than their covalent radii plus tolerance (``i < j``).

    Ions and satellites are never bonded by radii, so they are left out before any
    distance is computed.
    """
    radii = _radii(elements)
    kept = np.flatnonzero(~np.isin(np.asarray(elements), [*IONS, *SATELLITES]))
    if len(kept) == 0:
        return np.empty((0, 2), dtype=np.int64)
    cutoff = max_bond_limit(elements)
    if box is None or cutoff < cell_heights(box).min() / 2:
        # Each bonded pair is closer than the largest limit, so a cell list finds
        # all of them; rounding gives the minimum image at these distances.
        i, j, distance = _neighbour_pairs(positions, kept, kept, box, cutoff, True)
        limit = radii[i] + radii[j] + BOND_TOLERANCE
        bonded = (distance < limit) & (distance > MIN_DISTANCE)
        return np.sort(np.column_stack([i[bonded], j[bonded]]), axis=1).astype(np.int64)
    coordinates, shifts = _search_setup(positions[kept], elements, box)
    columns = np.arange(len(kept))
    pairs: list[NDArray[np.int64]] = [np.empty((0, 2), dtype=np.int64)]
    for lo in range(0, len(kept), _CHUNK):
        hi = min(lo + _CHUNK, len(kept))
        distance = _distance_block(coordinates, np.arange(lo, hi), box, shifts)
        limit = radii[kept[lo:hi], None] + radii[None, kept] + BOND_TOLERANCE
        bonded = (distance < limit) & (distance > MIN_DISTANCE)
        # Each pair once: keep the upper triangle.
        bonded &= columns[None, :] > np.arange(lo, hi)[:, None]
        i, j = np.nonzero(bonded)
        pairs.append(np.column_stack([kept[i + lo], kept[j]]).astype(np.int64))
    return np.concatenate(pairs)


def guess_bonds(
    positions: NDArray[np.float64], elements: list[str], box: Box | None = None
) -> NDArray[np.int64]:
    """Find bonded atom pairs from covalent radii.

    The minimum image is found by rounding the fractional difference. This is the
    nearest image if the distance is below half of the smallest cell height
    (``cell_heights``). Usually the longest possible bond is shorter than that, and
    a cell list limits the search to nearby atoms, so the time grows linearly with
    the number of atoms. In a strongly skewed cell all pairs are compared instead,
    in blocks of atoms to keep the memory small, and the 26 neighbouring images are
    tested as well. Ions, Drude particles and virtual sites take no part in the
    search by radii; the satellites are bonded to their host atom by
    ``satellite_bonds``.

    Parameters
    ----------
    positions : NDArray[np.float64]
        Cartesian positions in Angstrom, shape (n_atoms, 3).
    elements : list[str]
        Element symbol of each atom.
    box : Box | None, optional
        Periodic cell for the minimum-image convention. None uses plain distances.

    Returns
    -------
    NDArray[np.int64]
        Bonded atom pairs ``(i, j)`` with ``i < j``, shape (n_bonds, 2).
    """
    pairs = [
        _bonds_by_radii(positions, elements, box),
        satellite_bonds(positions, elements, box),
    ]
    result = np.concatenate(pairs)
    return result[np.lexsort((result[:, 1], result[:, 0]))]


def molecule_index(n_atoms: int, bonds: NDArray[np.int64]) -> NDArray[np.int64]:
    """Label the connected components of the bond graph.

    Parameters
    ----------
    n_atoms : int
        Number of atoms.
    bonds : NDArray[np.int64]
        Bonded atom pairs, shape (n_bonds, 2).

    Returns
    -------
    NDArray[np.int64]
        Molecule of each atom, numbered from 0 in the order of the first atom of
        each molecule.
    """
    # Each atom points to an atom of lower or equal index in its molecule. For every
    # bond, the higher of the two root atoms is pointed to the lower one (all bonds
    # at once), and the pointers are then followed to the roots. This repeats until
    # no bond joins two roots, so every molecule ends at its lowest atom index.
    root = np.arange(n_atoms)
    i, j = bonds.T
    while True:
        lower = np.minimum(root[i], root[j])
        joined = root.copy()
        np.minimum.at(joined, root[i], lower)
        np.minimum.at(joined, root[j], lower)
        while not np.array_equal(joined[joined], joined):
            joined = joined[joined]
        if np.array_equal(joined, root):
            break
        root = joined
    # The root of a component is its lowest atom index, so np.unique numbers the
    # molecules by their first atom.
    _, index = np.unique(root, return_inverse=True)
    return index.astype(np.int64)
