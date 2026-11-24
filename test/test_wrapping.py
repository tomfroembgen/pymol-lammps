"""Tests for wrapping whole molecules into the box."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from conftest import frame_text

from pymol_lammps.dump import Box, read_frames
from pymol_lammps.wrapping import Molecules, wrap_molecules

WriteFile = Callable[[str, str], Path]

ORTHO = Box.from_bounds(np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]))
# Two diatomic molecules: one sticking out of the box, one far outside it.
POSITIONS = np.array(
    [[9.5, 5.0, 5.0], [10.5, 5.0, 5.0], [25.0, -3.0, 5.0], [26.0, -3.0, 5.0]]
)
MOLECULES = Molecules.from_bonds(np.array([0, 0, 1, 1]), None)


def test_molecules_are_shifted_as_a_whole() -> None:
    wrapped = wrap_molecules(POSITIONS, MOLECULES, ORTHO)

    # Molecule 0 has its centre at x = 10.0, on the boundary: shifted by -10.
    np.testing.assert_allclose(wrapped[:2], [[-0.5, 5, 5], [0.5, 5, 5]], atol=1e-12)
    np.testing.assert_allclose(wrapped[2:], [[5, 7, 5], [6, 7, 5]], atol=1e-12)


def test_molecules_inside_the_box_stay() -> None:
    inside = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])

    single = Molecules.from_bonds(np.array([0, 0]), None)
    wrapped = wrap_molecules(inside, single, ORTHO)

    np.testing.assert_allclose(wrapped, inside, atol=1e-12)


def test_triclinic_box_uses_box_vectors() -> None:
    box = Box.from_bounds(np.array([[0.0, 12.0, 2.0], [0, 10, 0.0], [0, 10, 0.0]]))
    # One box vector b = (2, 10, 0) above the box: fractional y in [1, 2).
    position = np.array([[4.0, 15.0, 5.0]])

    wrapped = wrap_molecules(position, Molecules.from_bonds(np.array([0]), None), box)

    np.testing.assert_allclose(wrapped, [[2.0, 5.0, 5.0]], atol=1e-12)


def test_make_whole_reassembles_cut_molecules() -> None:
    # Wrapped coordinates: the molecule is cut by the boundary at x = 0/10.
    cut = np.array([[9.6, 5.0, 5.0], [0.4, 5.0, 5.0]])

    pair = Molecules.from_bonds(np.array([0, 0]), None)
    whole = wrap_molecules(cut, pair, ORTHO, make_whole=True)

    np.testing.assert_allclose(np.linalg.norm(whole[1] - whole[0]), 0.8)
    centre = whole.mean(axis=0)
    assert ((centre >= 0) & (centre < 10)).all()


def test_frame_knows_whether_positions_are_unwrapped(write_file: WriteFile) -> None:
    rows = [[1, 1, 0, 0, 0, 0, 0, 0]]
    cases = {
        "id type x y z ix iy iz": True,
        "id type xu yu zu ix iy iz": True,
        "id type xs ys zs ix iy iz": True,
    }
    for columns, expected in cases.items():
        path = write_file("t.lammpstrj", frame_text(columns, rows))
        assert next(read_frames(path)).unwrapped is expected

    path = write_file("t.lammpstrj", frame_text("id type x y z", [[1, 1, 0, 0, 0]]))
    assert next(read_frames(path)).unwrapped is False


def test_chain_longer_than_the_box_is_reassembled_along_bonds() -> None:
    # 9 atoms 1.5 A apart: 12 A long chain in a 10 A box, wrapped per atom.
    unwrapped = np.column_stack(
        [np.arange(9) * 1.5 + 0.5, np.full(9, 5.0), np.full(9, 5.0)]
    )
    wrapped_input = unwrapped.copy()
    wrapped_input[:, 0] %= 10.0
    bonds = np.column_stack([np.arange(8), np.arange(1, 9)])
    chain = Molecules.from_bonds(np.zeros(9, dtype=np.int64), bonds)

    whole = wrap_molecules(wrapped_input, chain, ORTHO, make_whole=True)

    np.testing.assert_allclose(np.diff(whole[:, 0]), 1.5)
    assert 0 <= whole[:, 0].mean() < 10


def test_reassembly_order_follows_bonds() -> None:
    # Atoms 0-1-2 bonded in a row, atom 3 in the same molecule but unbonded.
    bonds = np.array([[1, 2], [0, 1]])

    plan = Molecules.from_bonds(np.array([0, 0, 0, 0]), bonds)

    np.testing.assert_array_equal(plan.anchor, [0, 0, 1, 0])
    np.testing.assert_array_equal(plan.levels[0], [1, 3])
    np.testing.assert_array_equal(plan.levels[1], [2])


def test_anchor_is_the_first_atom_that_reaches_it() -> None:
    """In a ring, the far atom is reached from two sides; the walk order decides.

    Molecule 0 is the ring 0-1-2-3-0 (atom 2 is reached from 1 and from 3); molecule
    1 is the chain 4-5 given as bond (5, 4); bond 3-4 joins two molecules and is
    not followed.
    """
    bonds = np.array([[0, 3], [3, 4], [2, 3], [0, 1], [1, 2], [5, 4]])

    plan = Molecules.from_bonds(np.array([0, 0, 0, 0, 1, 1]), bonds)

    # Atom 0 reaches 3 before 1 (bond order), so 3 is walked first and anchors 2.
    np.testing.assert_array_equal(plan.anchor, [0, 0, 3, 0, 4, 4])
    np.testing.assert_array_equal(plan.levels[0], [1, 3, 5])
    np.testing.assert_array_equal(plan.levels[1], [2])
