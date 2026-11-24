"""Tests for bond and molecule guessing."""

import numpy as np

from pymol_lammps.bonds import COVALENT_RADII, guess_bonds, molecule_index
from pymol_lammps.dump import Box
from pymol_lammps.elements import ELEMENT_MASSES

WATER = ["O", "H", "H"]


def test_radii_cover_all_known_elements() -> None:
    assert COVALENT_RADII.keys() == ELEMENT_MASSES.keys()


def test_bonds_across_the_boundary_need_the_minimum_image() -> None:
    # O at x = 9.8 and H at x = 0.76 are 0.96 A apart through the boundary.
    positions = np.array([[9.8, 5, 5], [0.76, 5, 5], [9.8, 5.96, 5]])
    box = Box.from_cell_parameters(10.0)

    periodic = guess_bonds(positions, WATER, box)
    plain = guess_bonds(positions, WATER)

    assert periodic.tolist() == [[0, 1], [0, 2]]
    assert plain.tolist() == [[0, 2]]


def test_triclinic_cell_uses_the_cell_vectors() -> None:
    box = Box.from_cell_parameters(10, 10, 10, 90, 90, 120)
    # H sits one lattice vector b away from where it would bond to O.
    o = np.array([2.0, 2.0, 2.0])
    h = o + np.array([0.96, 0.0, 0.0]) + box.matrix[:, 1]

    bonds = guess_bonds(np.array([o, h]), ["O", "H"], box)

    assert bonds.tolist() == [[0, 1]]


def test_overlapping_atoms_and_distant_atoms_are_not_bonded() -> None:
    positions = np.array([[0, 0, 0], [0.1, 0, 0], [5, 0, 0]], dtype=float)

    assert len(guess_bonds(positions, ["H", "H", "H"])) == 0


def test_search_in_blocks_gives_the_same_bonds() -> None:
    # 400 atoms exceed one block; chains of H2 molecules spaced 3 A apart.
    n_molecules = 200
    positions = np.zeros((2 * n_molecules, 3))
    positions[0::2, 0] = 3.0 * np.arange(n_molecules)
    positions[1::2, 0] = 3.0 * np.arange(n_molecules) + 0.74

    bonds = guess_bonds(positions, ["H"] * (2 * n_molecules))

    assert bonds.tolist() == [[2 * k, 2 * k + 1] for k in range(n_molecules)]


def test_molecules_are_connected_components() -> None:
    bonds = np.array([[0, 1], [3, 4], [1, 2], [4, 5]])

    index = molecule_index(7, bonds)

    assert index.tolist() == [0, 0, 0, 1, 1, 1, 2]


def test_skewed_cell_finds_bonds_in_the_interior() -> None:
    # Simple cubic 10 A lattice written with c = (30, 0, 10): rounding the fractional
    # difference picks a wrong image for this pair, 1.7 A apart.
    box = Box(
        origin=np.zeros(3), matrix=np.array([[10.0, 0, 30], [0, 10, 0], [0, 0, 10]])
    )
    positions = np.array([[20.0, 5.0, 4.0], [20.0, 5.0, 5.7]])

    assert guess_bonds(positions, ["C", "C"], box).tolist() == [[0, 1]]


def test_ions_get_no_bonds() -> None:
    # Na+ with six waters at 2.4 A (inside the Na-O limit of 2.72 A) and a Cl-.
    rows = [("Na", 5.0, 5.0, 5.0), ("Cl", 1.0, 1.0, 1.0)]
    for axis in np.eye(3):
        for sign in (1, -1):
            o = 5.0 + 2.4 * sign * axis
            side = np.roll(axis, 1)
            rows += [
                ("O", *o),
                ("H", *(o + 0.96 * sign * axis)),
                ("H", *(o + 0.96 * side)),
            ]
    elements = [row[0] for row in rows]
    positions = np.array([row[1:] for row in rows])

    bonds = guess_bonds(positions, elements, Box.from_cell_parameters(15.0))
    index = molecule_index(len(rows), bonds)

    assert len(set(index.tolist())) == 8
    assert index[0] != index[2]
