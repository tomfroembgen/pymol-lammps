"""Tests for cell parsing and the xyz reader."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from pymol_lammps.dump import Box, parse_cell
from pymol_lammps.xyz import normalise_element, read_frames

WriteFile = Callable[[str, str], Path]


def xyz_frame(shift: float = 0.0, comment: str = "", extra: str = "") -> str:
    return (
        f"3\n{comment}\n"
        f"O {shift} 0.0 0.0{extra}\nH {shift + 0.96} 0.0 0.0{extra}\n"
        f"h {shift} 0.96 0.0{extra}\n"
    )


def test_single_length_gives_cube() -> None:
    box = Box.from_cell_parameters(12.3)

    np.testing.assert_allclose(box.matrix, np.diag([12.3] * 3), atol=1e-12)
    np.testing.assert_array_equal(box.origin, 0)


def test_three_lengths_give_right_angles() -> None:
    box = parse_cell("12, 13 15")

    np.testing.assert_allclose(box.cell_parameters(), (12, 13, 15, 90, 90, 90))


@pytest.mark.parametrize(
    "params",
    [(10, 10, 12, 90, 90, 120), (7.1, 8.2, 9.3, 70, 80, 100), (5, 6, 7, 60, 60, 60)],
)
def test_cell_parameters_round_trip(params: tuple[float, ...]) -> None:
    box = Box.from_cell_parameters(*params)

    np.testing.assert_allclose(box.cell_parameters(), params)
    assert box.matrix[1, 0] == 0 == box.matrix[2, 0] == box.matrix[2, 1]


def test_parse_cell_with_six_numbers() -> None:
    box = parse_cell("12 12 15 90 90 120")

    np.testing.assert_allclose(box.cell_parameters(), (12, 12, 15, 90, 90, 120))


@pytest.mark.parametrize("text", ["", "1 2", "1 2 3 4", "1 2 3 4 5 6 7", "a b c"])
def test_parse_cell_rejects_bad_count_or_text(text: str) -> None:
    with pytest.raises(ValueError, match="cell"):
        parse_cell(text)


@pytest.mark.parametrize(
    "args",
    [
        (0.0,),
        (-1.0,),
        (5, 5, 5, 90, 90, 0),
        (5, 5, 5, 90, 90, 180),
        (5, 5, 5, 120, 120, 120),
        (5, 5, 5, 30, 30, 120),
    ],
)
def test_invalid_cells_raise(args: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        Box.from_cell_parameters(*args)


def test_multi_frame_read(write_file: WriteFile) -> None:
    path = write_file("t.xyz", xyz_frame() + xyz_frame(1.0))

    frames = list(read_frames(path))

    assert len(frames) == 2
    assert frames[1].elements == ["O", "H", "H"]
    np.testing.assert_allclose(frames[1].positions[:, 0], [1.0, 1.96, 1.0])
    assert frames[0].box is None


def test_frame_selection_is_one_based_and_inclusive(write_file: WriteFile) -> None:
    path = write_file("t.xyz", "".join(xyz_frame(float(i)) for i in range(5)))

    frames = list(read_frames(path, start=2, stop=4, interval=2))

    assert [f.positions[0, 0] for f in frames] == [1.0, 3.0]


def test_gzip_is_read(write_file: WriteFile) -> None:
    path = write_file("t.xyz.gz", xyz_frame() + xyz_frame(1.0))

    assert len(list(read_frames(path))) == 2


def test_lattice_is_parsed_with_row_vectors(write_file: WriteFile) -> None:
    comment = 'Lattice="10 0 0 5 8.66 0 0 0 12" pbc="T T T"'
    path = write_file("t.xyz", xyz_frame(comment=comment, extra=" 0.1 0.2"))

    (frame,) = read_frames(path)

    assert frame.box is not None
    expected = np.array([[10.0, 5.0, 0.0], [0.0, 8.66, 0.0], [0.0, 0.0, 12.0]])
    np.testing.assert_allclose(frame.box.matrix, expected)
    np.testing.assert_allclose(frame.box.cell_parameters()[5], 60, atol=0.01)
    np.testing.assert_array_equal(frame.box.origin, 0)


def test_origin_tag_is_used(write_file: WriteFile) -> None:
    comment = 'Lattice="4 0 0 0 4 0 0 0 4" Origin="-2 -2 -2"'
    path = write_file("t.xyz", xyz_frame(comment=comment))

    (frame,) = read_frames(path)

    assert frame.box is not None
    np.testing.assert_array_equal(frame.box.origin, [-2, -2, -2])


def test_malformed_lattice_raises(write_file: WriteFile) -> None:
    path = write_file("t.xyz", xyz_frame(comment='Lattice="1 2 3"'))

    with pytest.raises(ValueError, match="Lattice"):
        list(read_frames(path))


def test_truncated_last_frame_is_skipped(write_file: WriteFile) -> None:
    path = write_file("t.xyz", xyz_frame() + xyz_frame(1.0).rsplit("\n", 2)[0] + "\n")

    with pytest.warns(UserWarning, match="incomplete"):
        frames = list(read_frames(path))

    assert len(frames) == 1


def test_changing_atom_count_raises(write_file: WriteFile) -> None:
    path = write_file("t.xyz", xyz_frame() + "1\n\nO 0 0 0\n")

    with pytest.raises(ValueError, match="atoms"):
        list(read_frames(path))


@pytest.mark.parametrize("row", ["O 0 zero 0", "O 0 0"])
def test_bad_row_raises(write_file: WriteFile, row: str) -> None:
    path = write_file("t.xyz", f"1\n\n{row}\n")

    with pytest.raises(ValueError, match="Element x y z"):
        list(read_frames(path))


@pytest.mark.parametrize(
    ("label", "symbol"),
    [("C1", "C"), ("cl", "Cl"), ("NA", "Na"), ("O2", "O"), ("HW", "H"), ("fe3", "Fe")],
)
def test_element_labels_are_normalised(label: str, symbol: str) -> None:
    assert normalise_element(label) == symbol


@pytest.mark.parametrize("label", ["Xx", "12", "?"])
def test_unknown_element_raises(label: str) -> None:
    with pytest.raises(ValueError, match="unknown element"):
        normalise_element(label)
