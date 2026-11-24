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

import numpy as np
from numpy.typing import NDArray

from pymol_lammps.drude import SATELLITES, VIRTUAL, satellite_mask
from pymol_lammps.dump import Box

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

# Radius in Angstrom of the cell-list search for the host of a satellite. A Drude
# sits 0.1-0.4 A from its core and an M site about 0.25 A from its oxygen; a satellite
# without any candidate this close is searched among all atoms instead.
HOST_SEARCH_RADIUS = 2.0


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
    elements: list[str],
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
    elements : list[str]
        Element symbol of each atom, with ``D`` and ``M`` for the satellites.
    box : Box
        Periodic cell of the frame.

    Returns
    -------
    NDArray[np.bool_]
        True for each bond to remove, shape (n_bonds,).
    """
    length = np.linalg.norm(positions[bonds[:, 0]] - positions[bonds[:, 1]], axis=1)
    too_long = length > cell_heights(box).min() / 2
    return too_long & ~satellite_mask(elements)[bonds].any(axis=1)


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
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Find all pairs of a query atom and a candidate atom closer than ``cutoff``.

    The atoms are sorted into a grid of cells that are at least ``cutoff`` high, so
    the partners of a query atom lie in its own cell or in one of the 26 cells
    around it (a cell list). In a periodic cell the grid wraps around, and the
    distance is the minimum-image distance, which is exact for ``cutoff`` below half
    of the smallest cell height.

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

    Returns
    -------
    tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]
        Query atom, candidate atom and distance of each pair closer than ``cutoff``.
    """
    periodic = box is not None
    if box is None:
        # A non-periodic grid over the extent of the atoms.
        extent = np.maximum(np.ptp(positions, axis=0), cutoff)
        box = Box(origin=positions.min(axis=0), matrix=np.diag(extent))
    fractional = (positions - box.origin) @ np.linalg.inv(box.matrix).T
    if periodic:
        fractional -= np.floor(fractional)
    n_cells = np.maximum((cell_heights(box) / cutoff).astype(np.int64), 1)
    # Few cells per atom are enough; larger cells only add candidates to test.
    limit = 8 * len(candidates) + 27
    if n_cells.prod() > limit:
        scale = (limit / n_cells.prod()) ** (1 / 3)
        n_cells = np.maximum((n_cells * scale).astype(np.int64), 1)
    cells = np.minimum((fractional * n_cells).astype(np.int64), n_cells - 1)

    keys = np.ravel_multi_index(tuple(cells[candidates].T), tuple(n_cells))
    by_cell = candidates[np.argsort(keys, kind="stable")]
    counts = np.bincount(keys, minlength=int(n_cells.prod()))
    starts = np.cumsum(counts) - counts
    # Neighbouring cells along each axis; with fewer than three cells the periodic
    # neighbours repeat, so each is taken once.
    steps = [
        np.unique(np.array([-1, 0, 1]) % n) if periodic else np.array([-1, 0, 1])
        for n in n_cells
    ]
    query_parts: list[NDArray[np.int64]] = []
    candidate_parts: list[NDArray[np.int64]] = []
    for step in itertools.product(*steps):
        neighbour = cells[queries] + np.array(step)
        if periodic:
            neighbour %= n_cells
        inside = ((neighbour >= 0) & (neighbour < n_cells)).all(axis=1)
        cell_keys = np.ravel_multi_index(tuple(neighbour[inside].T), tuple(n_cells))
        number = counts[cell_keys]
        # Position of every candidate of these cells in ``by_cell``.
        offsets = np.arange(number.sum()) - np.repeat(
            np.cumsum(number) - number, number
        )
        query_parts.append(np.repeat(queries[inside], number))
        candidate_parts.append(by_cell[np.repeat(starts[cell_keys], number) + offsets])
    query = np.concatenate(query_parts)
    candidate = np.concatenate(candidate_parts)

    delta = fractional[candidate] - fractional[query]
    if periodic:
        delta -= np.round(delta)
    distance = np.linalg.norm(delta @ box.matrix.T, axis=1)
    close = distance < cutoff
    return query[close], candidate[close], distance[close]


def _nearest(
    positions: NDArray[np.float64],
    elements: list[str],
    queries: NDArray[np.int64],
    candidates: NDArray[np.int64],
    box: Box | None,
) -> NDArray[np.int64]:
    """Return the nearest candidate atom of each query atom (minimum image in a box).

    The cell list finds the nearest candidate within ``HOST_SEARCH_RADIUS``; query
    atoms without one are compared with all candidates.
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

    far = queries[nearest[queries] < 0]
    coordinates, shifts = _search_setup(positions, elements, box)
    for lo in range(0, len(far), _CHUNK):
        rows = far[lo : lo + _CHUNK]
        distances = _distance_block(coordinates, rows, box, shifts)[:, candidates]
        nearest[rows] = candidates[np.argmin(distances, axis=1)]
    return nearest[queries]


def satellite_bonds(
    positions: NDArray[np.float64], elements: list[str], box: Box | None = None
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

    Returns
    -------
    NDArray[np.int64]
        One pair ``(i, j)`` with ``i < j`` per satellite, shape (n_satellites, 2).
    """
    satellite = satellite_mask(elements)
    cores = ~satellite
    oxygen = cores & (np.array(elements) == "O")
    virtual = np.array(elements) == VIRTUAL
    pairs: list[NDArray[np.int64]] = [np.empty((0, 2), dtype=np.int64)]
    # Virtual sites choose among oxygens if there are any.
    for members, allowed in (
        (satellite & ~virtual, cores),
        (virtual, oxygen if oxygen.any() else cores),
    ):
        queries = np.flatnonzero(members)
        candidates = np.flatnonzero(allowed)
        if len(queries) and len(candidates):
            hosts = _nearest(positions, elements, queries, candidates, box)
            pairs.append(np.column_stack([queries, hosts]).astype(np.int64))
    # In the order of the satellites, each pair sorted.
    found = np.concatenate(pairs)
    return np.sort(found[np.argsort(found[:, 0])], axis=1)


def guess_bonds(
    positions: NDArray[np.float64], elements: list[str], box: Box | None = None
) -> NDArray[np.int64]:
    """Find bonded atom pairs from covalent radii.

    The search runs in blocks of atoms, so memory stays small for several thousand
    atoms. The minimum image is found by rounding the fractional difference. In a
    strongly skewed cell this is the nearest image only if the distance is below half
    of the smallest cell height (``cell_heights``); then the 26 neighbouring images
    are tested as well, which is slower. Drude particles and virtual sites are
    bonded to their host atom by ``satellite_bonds`` and take no part in the search
    by radii.

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
    n_atoms = len(positions)
    radii = _radii(elements)
    coordinates, shifts = _search_setup(positions, elements, box)
    # Ions and satellites are never bonded by radii.
    excluded = np.array([e in IONS or e in SATELLITES for e in elements])
    pairs: list[NDArray[np.int64]] = [np.empty((0, 2), dtype=np.int64)]
    columns = np.arange(n_atoms)
    for lo in range(0, n_atoms, _CHUNK):
        hi = min(lo + _CHUNK, n_atoms)
        distance = _distance_block(coordinates, np.arange(lo, hi), box, shifts)
        limit = radii[lo:hi, None] + radii[None, :] + BOND_TOLERANCE
        bonded = (distance < limit) & (distance > MIN_DISTANCE)
        # Each pair once: keep the upper triangle.
        bonded &= columns[None, :] > np.arange(lo, hi)[:, None]
        bonded &= ~(excluded[lo:hi, None] | excluded[None, :])
        i, j = np.nonzero(bonded)
        pairs.append(np.column_stack([i + lo, j]).astype(np.int64))
    pairs.append(satellite_bonds(positions, elements, box))
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
    parent = list(range(n_atoms))

    def find(atom: int) -> int:
        while parent[atom] != atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    for i, j in bonds.tolist():
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[max(root_i, root_j)] = min(root_i, root_j)

    roots = np.array([find(atom) for atom in range(n_atoms)], dtype=np.int64)
    # The root of a component is its lowest atom index, so np.unique numbers the
    # molecules by their first atom.
    _, index = np.unique(roots, return_inverse=True)
    return index.astype(np.int64)
