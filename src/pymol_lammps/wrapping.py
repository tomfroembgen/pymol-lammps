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
    """

    index: NDArray[np.int64]
    anchor: NDArray[np.int64]
    levels: list[NDArray[np.int64]]

    @classmethod
    def from_bonds(
        cls, index: NDArray[np.int64], bonds: NDArray[np.int64] | None
    ) -> "Molecules":
        """Build the reassembly order by a breadth-first walk along the bonds.

        Parameters
        ----------
        index : NDArray[np.int64]
            Molecule of each atom, numbered 0 to n_molecules - 1.
        bonds : NDArray[np.int64] | None
            Bonded atom pairs as atom indices, shape (n_bonds, 2), or None.

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
        return cls(index=index, anchor=anchor, levels=levels)


def wrap_molecules(
    positions: NDArray[np.float64],
    molecules: Molecules,
    box: Box,
    make_whole: bool = False,
) -> NDArray[np.float64]:
    """Shift whole molecules so that their centres of geometry lie in the box.

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

    Returns
    -------
    NDArray[np.float64]
        Wrapped positions, shape (n_atoms, 3).
    """
    fractional = (positions - box.origin) @ np.linalg.inv(box.matrix).T
    if make_whole:
        for atoms in molecules.levels:
            offset = fractional[atoms] - fractional[molecules.anchor[atoms]]
            fractional[atoms] -= np.round(offset)

    index = molecules.index
    counts = np.bincount(index)
    centres = (
        np.column_stack(
            [np.bincount(index, weights=fractional[:, k]) for k in range(3)]
        )
        / counts[:, None]
    )
    fractional -= np.floor(centres)[index]
    return box.origin + fractional @ box.matrix.T
