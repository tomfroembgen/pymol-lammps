"""Put whole molecules into the simulation box.

Each molecule is shifted by box vectors so that its centre of geometry lies inside
the box. All atoms of a molecule get the same shift, so molecules stay intact. The
work is done in fractional coordinates, which covers triclinic boxes.

Dumps with wrapped coordinates and no image flags have molecules cut by the box
boundary. These are first reassembled: walking along the bonds from the first atom
of each molecule, every atom is moved to the periodic image closest to its bonded
neighbour. This works for molecules of any size, as long as each bond is shorter
than half the box. Atoms without a bond path (no bonds known) are placed next to
the first atom of their molecule instead, which needs molecules smaller than half
the box.

A molecule that is bonded through the boundary to its own periodic image (a sheet,
an infinite chain, a crystal) cannot be reassembled. Its atoms are wrapped one by
one in the first frame. In every later frame, each atom is moved to the image
closest to where it was shown in the previous frame, and the whole network is
shifted so that its centre stays in the box. The network is therefore cut at the
same bonds in every frame.

Drude particles and virtual sites are finally moved to the periodic image next to
their host atom (``place_satellites``), so that they are never shown on the other
side of the box.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pymol_lammps.dump import Box


@dataclass(frozen=True)
class Molecules:
    """Molecule membership and the order in which to reassemble molecules.

    Attributes
    ----------
    index : NDArray[np.int64]
        Molecule of each atom, numbered 0 to n_molecules - 1.
    anchor : NDArray[np.int64]
        Atom next to which each atom is placed when reassembling; the first atom
        of a molecule is its own anchor.
    levels : list[NDArray[np.int64]]
        Atoms grouped by distance (in bonds) from the first atom of their
        molecule, starting at distance 1. Processing the groups in order places
        every anchor before the atoms attached to it.
    network : NDArray[np.int64]
        Periodic network (a molecule bonded through the boundary) of each atom,
        numbered from 0, and -1 for the atoms of ordinary molecules. The atoms of a
        network are split into molecules of one atom each, together with their
        Drude particles and virtual sites.
    """

    index: NDArray[np.int64]
    anchor: NDArray[np.int64]
    levels: list[NDArray[np.int64]]
    network: NDArray[np.int64]

    @classmethod
    def from_bonds(
        cls,
        index: NDArray[np.int64],
        bonds: NDArray[np.int64] | None,
        network: NDArray[np.int64] | None = None,
    ) -> "Molecules":
        """Build the reassembly order by a breadth-first walk along the bonds.

        Parameters
        ----------
        index : NDArray[np.int64]
            Molecule of each atom, numbered 0 to n_molecules - 1.
        bonds : NDArray[np.int64] | None
            Bonded atom pairs as atom indices, shape (n_bonds, 2), or None.
        network : NDArray[np.int64] | None, optional
            Periodic network of each atom, -1 for none. None means no networks.

        Returns
        -------
        Molecules
            Membership and reassembly order.
        """
        n_atoms = len(index)
        neighbours: list[list[int]] = [[] for _ in range(n_atoms)]
        for i, j in bonds if bonds is not None else ():
            neighbours[i].append(int(j))
            neighbours[j].append(int(i))

        _, roots = np.unique(index, return_index=True)
        anchor = np.arange(n_atoms)
        depth = np.full(n_atoms, -1)
        for root in roots:
            depth[root] = 0
            queue = deque([int(root)])
            while queue:
                atom = queue.popleft()
                for other in neighbours[atom]:
                    if depth[other] < 0 and index[other] == index[atom]:
                        depth[other] = depth[atom] + 1
                        anchor[other] = atom
                        queue.append(other)

        # Atoms without a bond path to the first atom of their molecule.
        loose = depth < 0
        anchor[loose] = roots[index[loose]]
        depth[loose] = 1
        levels = [np.flatnonzero(depth == d) for d in range(1, depth.max() + 1)]
        if network is None:
            network = np.full(n_atoms, -1, dtype=np.int64)
        return cls(index=index, anchor=anchor, levels=levels, network=network)


def split_wrapping_molecules(
    positions: NDArray[np.float64],
    molecules: Molecules,
    box: Box,
    bonds: NDArray[np.int64],
    limits: NDArray[np.float64],
) -> tuple[Molecules, int]:
    """Turn molecules that are bonded through the cell boundary into single atoms.

    A chain, slab or crystal that is connected across the periodic boundary cannot
    be reassembled: after walking along the bonds, some bond is still longer than
    its limit. Wrapping such a molecule as a whole would leave part of it outside
    the cell, so its atoms are wrapped one by one instead, as a periodic network.

    Parameters
    ----------
    positions : NDArray[np.float64]
        Cartesian positions, shape (n_atoms, 3).
    molecules : Molecules
        Molecule membership and reassembly order.
    box : Box
        Simulation box.
    bonds : NDArray[np.int64]
        Bonded atom pairs, shape (n_bonds, 2).
    limits : NDArray[np.float64]
        Largest plausible length of each bond in Angstrom, shape (n_bonds,).

    Returns
    -------
    tuple[Molecules, int]
        Molecules for wrapping, with each atom of an affected molecule on its own
        and marked as part of a network, and the number of affected molecules.
    """
    whole = wrap_molecules(positions, molecules, box, make_whole=True)
    length = np.linalg.norm(whole[bonds[:, 0]] - whole[bonds[:, 1]], axis=1)
    cut = np.unique(molecules.index[bonds[length > limits + 1e-6, 0]])
    if len(cut) == 0:
        return molecules, 0
    index = molecules.index.copy()
    atoms = np.isin(index, cut)
    network = np.full(len(index), -1, dtype=np.int64)
    network[atoms] = np.searchsorted(cut, index[atoms])
    index[atoms] = index.max() + 1 + np.arange(atoms.sum())
    _, index = np.unique(index, return_inverse=True)
    return Molecules.from_bonds(index.astype(np.int64), bonds, network), len(cut)


def attach_satellites(
    molecules: Molecules, bonds: NDArray[np.int64], satellites: NDArray[np.int64]
) -> Molecules:
    """Move every Drude particle and virtual site into the molecule of its host.

    After ``split_wrapping_molecules`` each atom of a periodic network is wrapped on
    its own. A Drude wrapped separately from its core would sit on the opposite side
    of the box, so each satellite is assigned to the (single-atom) molecule and to
    the network of the atom it is bonded to.

    Parameters
    ----------
    molecules : Molecules
        Molecule membership and reassembly order.
    bonds : NDArray[np.int64]
        Bonded atom pairs, shape (n_bonds, 2).
    satellites : NDArray[np.int64]
        Pairs ``(satellite, host)``, shape (n_satellites, 2).

    Returns
    -------
    Molecules
        The molecules with the satellites next to their hosts.
    """
    index = molecules.index.copy()
    network = molecules.network.copy()
    satellite, host = satellites.T
    index[satellite] = index[host]
    network[satellite] = network[host]
    _, index = np.unique(index, return_inverse=True)
    return Molecules.from_bonds(index.astype(np.int64), bonds, network)


def place_satellites(
    positions: NDArray[np.float64], satellites: NDArray[np.int64], box: Box
) -> NDArray[np.float64]:
    """Move each Drude particle and virtual site to the periodic image next to its host.

    This keeps the satellites with their hosts however the molecules were wrapped,
    also where the molecules are unknown or could not be reassembled.

    Parameters
    ----------
    positions : NDArray[np.float64]
        Cartesian positions, shape (n_atoms, 3).
    satellites : NDArray[np.int64]
        Pairs ``(satellite, host)``, shape (n_satellites, 2).
    box : Box
        Simulation box of the frame.

    Returns
    -------
    NDArray[np.float64]
        Positions with every satellite at the minimum image of its host.
    """
    placed = positions.copy()
    satellite, host = satellites.T
    offset = (positions[satellite] - positions[host]) @ np.linalg.inv(box.matrix).T
    placed[satellite] -= np.round(offset) @ box.matrix.T
    return placed


def _centres(
    fractional: NDArray[np.float64], index: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Return the centre of geometry of each group in ``index``, shape (n, 3)."""
    counts = np.bincount(index)
    sums = [np.bincount(index, weights=fractional[:, k]) for k in range(3)]
    return np.column_stack(sums) / counts[:, None]


def wrap_molecules(
    positions: NDArray[np.float64],
    molecules: Molecules,
    box: Box,
    make_whole: bool = False,
    previous: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Shift whole molecules so that their centres of geometry lie in the box.

    The pieces of a periodic network (see ``Molecules.network``) are instead moved
    to the image closest to where they were shown in the previous frame, and each
    network is then shifted as a whole so that its centre lies in the box. Without
    a previous frame, the pieces are wrapped one by one.

    Parameters
    ----------
    positions : NDArray[np.float64]
        Cartesian positions, shape (n_atoms, 3).
    molecules : Molecules
        Molecule membership and reassembly order.
    box : Box
        Simulation box of the frame.
    make_whole : bool, optional
        First reassemble molecules that are cut by the boundary. Needed for
        wrapped coordinates without image flags.
    previous : NDArray[np.float64] | None, optional
        Displayed positions of the previous frame, shape (n_atoms, 3).

    Returns
    -------
    NDArray[np.float64]
        Wrapped positions, shape (n_atoms, 3).
    """
    inverse = np.linalg.inv(box.matrix).T
    fractional = (positions - box.origin) @ inverse
    if make_whole:
        for atoms in molecules.levels:
            offset = fractional[atoms] - fractional[molecules.anchor[atoms]]
            fractional[atoms] -= np.round(offset)

    index = molecules.index
    centres = _centres(fractional, index)
    shift = np.floor(centres)
    in_network = molecules.network >= 0
    piece = np.bincount(index, weights=in_network) > 0
    if previous is not None and piece.any():
        # A piece moves by much less than half a box between two frames, so the
        # image closest to its previous position continues its path.
        before = _centres((previous - box.origin) @ inverse, index)
        shift[piece] = np.round(centres - before)[piece]
    fractional -= shift[index]

    if in_network.any():
        # One shift for all atoms of a network keeps its bonds as they are.
        network = molecules.network[in_network]
        network_centres = _centres(fractional[in_network], network)
        fractional[in_network] -= np.floor(network_centres)[network]
    return box.origin + fractional @ box.matrix.T
