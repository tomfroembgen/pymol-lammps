"""Tests for combining dump frames, data files and element information."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import frame_text, water_frame

from pymol_lammps.datafile import read_data
from pymol_lammps.drude import DrudeSetting
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


def element_frame(write_file: WriteFile, ids: list[int]) -> Frame:
    """Return a frame with the columns ``id element x y z`` for the given water atoms."""
    elements = {1: "O", 2: "H", 3: "H", 4: "O", 5: "H", 6: "H"}
    rows = [[i, elements[i], 0, 0, 0] for i in ids]
    return first_frame(write_file, frame_text("id element x y z", rows))


def test_element_only_dump_takes_types_from_data_file(
    write_file: WriteFile, water_data: Path
) -> None:
    frame = element_frame(write_file, [5, 2, 1, 3, 4, 6])

    topology = build_topology(frame, read_data(water_data))

    np.testing.assert_array_equal(topology.types, [1, 2, 2, 1, 2, 2])
    assert topology.elements == ["O", "H", "H"] * 2
    assert topology.names == ["OW", "HW1", "HW2"] * 2
    assert topology.residue_ids == ["1"] * 3 + ["2"] * 3
    assert topology.bonds is not None
    np.testing.assert_array_equal(topology.bonds, [[0, 1], [0, 2], [3, 4], [3, 5]])


# Data file as written by write_data: no type labels and no Masses comments.
UNNAMED_WATER_DATA = """LAMMPS data file via write_data

3 atoms
2 atom types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 1.008
2 15.999

Atoms # full

1 1 2 -0.8 0 0 0 0 0 0
2 1 1 0.4 1 0 0 0 0 0
3 1 1 0.4 0 1 0 0 0 0
"""


def test_element_only_dump_with_unnamed_data_file_names_atoms_by_element(
    write_file: WriteFile,
) -> None:
    data = read_data(write_file("unnamed.data", UNNAMED_WATER_DATA))

    topology = build_topology(element_frame(write_file, [1, 2, 3]), data)

    np.testing.assert_array_equal(topology.types, [2, 1, 1])
    assert topology.names == ["O", "H", "H"]


def test_element_only_dump_of_a_group_keeps_bonds_within_it(
    write_file: WriteFile, water_data: Path
) -> None:
    topology = build_topology(
        element_frame(write_file, [4, 5, 6, 2]), read_data(water_data)
    )

    np.testing.assert_array_equal(topology.ids, [2, 4, 5, 6])
    np.testing.assert_array_equal(topology.types, [2, 1, 2, 2])
    assert topology.bonds is not None
    np.testing.assert_array_equal(topology.bonds, [[1, 2], [1, 3]])


def test_element_only_dump_with_unknown_atom_raises(
    write_file: WriteFile, water_data: Path
) -> None:
    frame = first_frame(
        write_file, frame_text("id element x y z", [[99, "O", 0, 0, 0]])
    )

    with pytest.raises(ValueError, match="missing from the data file"):
        build_topology(frame, read_data(water_data))


def test_element_only_dump_without_data_gets_one_type_per_element(
    write_file: WriteFile,
) -> None:
    topology = build_topology(element_frame(write_file, [1, 2, 3, 4, 5, 6]))

    # np.unique sorts the elements: H is type 1, O is type 2.
    np.testing.assert_array_equal(topology.types, [2, 1, 1, 2, 1, 1])
    assert topology.elements == ["O", "H", "H"] * 2
    assert topology.names == ["O", "H", "H"] * 2
    assert topology.bonds is None


def test_drude_label_in_element_only_dump_is_kept(write_file: WriteFile) -> None:
    rows = [[1, "O", 0, 0, 0], [2, "D", 0.1, 0, 0]]
    frame = first_frame(write_file, frame_text("id element x y z", rows))

    topology = build_topology(frame)

    assert topology.elements == ["O", "D"]
    np.testing.assert_array_equal(topology.types, [2, 1])


@pytest.mark.parametrize(
    "options",
    [{"type_elements": ["O", "H"]}, {"drude": DrudeSetting("types", frozenset({1}))}],
)
def test_type_options_for_element_only_dump_without_data_raise(
    write_file: WriteFile, options: dict[str, Any]
) -> None:
    frame = element_frame(write_file, [1, 2, 3])

    with pytest.raises(ValueError, match="cannot refer to atom types"):
        build_topology(frame, **options)


def test_dump_without_type_and_element_raises(write_file: WriteFile) -> None:
    with pytest.raises(ValueError, match="type .or element."):
        first_frame(write_file, frame_text("id x y z", [[1, 0, 0, 0]]))
