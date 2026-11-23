"""Tests for combining dump frames, data files and element information."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from conftest import frame_text, water_frame

from pymol_lammps.datafile import read_data
from pymol_lammps.dump import Frame, read_frames
from pymol_lammps.elements import element_from_mass
from pymol_lammps.topology import build_topology

WriteFile = Callable[[str, str], Path]


def first_frame(write_file: WriteFile, text: str) -> Frame:
    return next(read_frames(write_file("t.lammpstrj", text)))


@pytest.mark.parametrize(
    ("mass", "element"),
    [(12.011, "C"), (14.0067, "N"), (16.00, "O"), (1.00794, "H"), (35.453, "Cl")],
)
def test_element_from_mass(mass: float, element: str) -> None:
    assert element_from_mass(mass) == element


@pytest.mark.parametrize("mass", [14.027, 15.035, 0.4, 0.0])
def test_united_atom_and_drude_masses_stay_unresolved(mass: float) -> None:
    assert element_from_mass(mass) == ""


def test_data_file_provides_topology(water_dump: Path, water_data: Path) -> None:
    topology = build_topology(next(read_frames(water_dump)), read_data(water_data))

    assert topology.elements == ["O", "H", "H"] * 2
    assert topology.names == ["OW", "HW1", "HW2"] * 2
    assert topology.residue_names == ["SPC"] * 6
    assert topology.residue_ids == ["1"] * 3 + ["2"] * 3
    assert topology.charges is not None
    np.testing.assert_allclose(topology.charges[:3], [-0.834, 0.417, 0.417])
    assert topology.bonds is not None
    np.testing.assert_array_equal(topology.bonds, [[0, 1], [0, 2], [3, 4], [3, 5]])


def test_dump_of_a_group_keeps_bonds_within_it(
    write_file: WriteFile, water_data: Path
) -> None:
    # Only the second molecule and one atom of the first are dumped.
    rows = [[5, 2, 0, 0, 0], [2, 2, 0, 0, 0], [4, 1, 0, 0, 0], [6, 2, 0, 0, 0]]
    frame = first_frame(write_file, frame_text("id type x y z", rows))

    topology = build_topology(frame, read_data(water_data))

    np.testing.assert_array_equal(topology.ids, [2, 4, 5, 6])
    assert topology.bonds is not None
    np.testing.assert_array_equal(topology.bonds, [[1, 2], [1, 3]])


def test_atom_missing_from_data_raises(write_file: WriteFile, water_data: Path) -> None:
    frame = first_frame(write_file, frame_text("id type x y z", [[99, 1, 0, 0, 0]]))

    with pytest.raises(ValueError, match="missing from the data file"):
        build_topology(frame, read_data(water_data))


def test_without_data_bonds_are_left_to_guessing(write_file: WriteFile) -> None:
    topology = build_topology(first_frame(write_file, water_frame()), None, ["O", "H"])

    assert topology.bonds is None
    assert topology.names == ["1", "2", "2"] * 2
    assert topology.residue_ids == ["1"] * 3 + ["2"] * 3
    assert topology.elements == ["O", "H", "H"] * 2


def test_too_few_elements_raise(write_file: WriteFile) -> None:
    with pytest.raises(ValueError, match="one element per atom type"):
        build_topology(first_frame(write_file, water_frame()), None, ["O"])


def test_element_column_is_used(write_file: WriteFile) -> None:
    rows = [[1, 1, "Na", 0, 0, 0], [2, 2, "Cl", 1, 0, 0]]
    frame = first_frame(write_file, frame_text("id type element x y z", rows))

    assert build_topology(frame).elements == ["Na", "Cl"]


def test_mass_column_is_used(write_file: WriteFile) -> None:
    rows = [[1, 1, 22.99, 0.5, 0, 0, 0], [2, 2, 35.45, -0.5, 1, 0, 0]]
    frame = first_frame(write_file, frame_text("id type mass q x y z", rows))

    topology = build_topology(frame)

    assert topology.elements == ["Na", "Cl"]
    assert topology.charges is not None
    np.testing.assert_allclose(topology.charges, [0.5, -0.5])


def test_unknown_elements_fall_back_with_warning(write_file: WriteFile) -> None:
    frame = first_frame(write_file, water_frame())

    with pytest.warns(UserWarning, match=r"atom types \[1, 2\]"):
        topology = build_topology(frame)

    assert topology.elements == ["C"] * 6


def test_type_labels_in_dump_raise(write_file: WriteFile) -> None:
    frame = first_frame(write_file, frame_text("id type x y z", [[1, "c3", 0, 0, 0]]))

    with pytest.raises(ValueError, match="type labels"):
        build_topology(frame)
