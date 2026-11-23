"""Tests for the dump reader."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from conftest import WATER_POSITIONS, frame_text, water_frame

from pymol_lammps.dump import Box, read_frames

WriteFile = Callable[[str, str], Path]


def test_frames_are_sorted_by_atom_id(water_dump: Path) -> None:
    frames = list(read_frames(water_dump))

    assert [f.timestep for f in frames] == [0, 100, 200]
    for shift, frame in enumerate(frames):
        np.testing.assert_array_equal(frame.ids, np.arange(1, 7))
        expected = np.array(WATER_POSITIONS) + [shift, 0.0, 0.0]
        np.testing.assert_allclose(frame.positions(), expected)


def test_frame_selection_is_one_based_and_inclusive(write_file: WriteFile) -> None:
    text = "".join(water_frame(100 * i) for i in range(5))
    path = write_file("t.lammpstrj", text)

    frames = read_frames(path, start=2, stop=4, interval=2)

    assert [f.timestep for f in frames] == [100, 300]


def test_gzip_dump_is_read(write_file: WriteFile) -> None:
    path = write_file("t.lammpstrj.gz", water_frame() + water_frame(10))

    assert len(list(read_frames(path))) == 2


def test_image_flags_unwrap_wrapped_positions(write_file: WriteFile) -> None:
    rows = [[1, 1, 4.5, 0.0, 0.0, -1, 0, 0], [2, 1, -4.5, 0.0, 0.0, 1, 0, 2]]
    path = write_file("t.lammpstrj", frame_text("id type x y z ix iy iz", rows))

    positions = next(read_frames(path)).positions()

    np.testing.assert_allclose(positions, [[-5.5, 0.0, 0.0], [5.5, 0.0, 20.0]])


def test_unwrapped_columns_take_precedence(write_file: WriteFile) -> None:
    rows = [[1, 1, 4.5, 0.0, 0.0, -5.5, 0.0, 0.0]]
    path = write_file("t.lammpstrj", frame_text("id type x y z xu yu zu", rows))

    positions = next(read_frames(path)).positions()

    np.testing.assert_allclose(positions, [[-5.5, 0.0, 0.0]])


def test_scaled_positions_use_box_origin(write_file: WriteFile) -> None:
    rows = [[1, 1, 0.25, 0.5, 1.0], [2, 1, 0.0, 0.0, 0.0]]
    path = write_file("t.lammpstrj", frame_text("id type xs ys zs", rows))

    positions = next(read_frames(path)).positions()

    np.testing.assert_allclose(positions, [[-2.5, 0.0, 5.0], [-5.0, -5.0, -5.0]])


def test_scaled_positions_with_image_flags(write_file: WriteFile) -> None:
    rows = [[1, 1, 0.9, 0.5, 0.5, -1, 0, 0]]
    path = write_file("t.lammpstrj", frame_text("id type xs ys zs ix iy iz", rows))

    positions = next(read_frames(path)).positions()

    np.testing.assert_allclose(positions, [[-6.0, 0.0, 0.0]])


def test_triclinic_box() -> None:
    # Box 10x10x10 with xy tilt 2: LAMMPS writes the bounding box in x (0 to 12).
    bounds = np.array([[0.0, 12.0, 2.0], [0.0, 10.0, 0.0], [0.0, 10.0, 0.0]])

    box = Box.from_bounds(bounds)

    np.testing.assert_allclose(box.origin, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(box.matrix, [[10, 2, 0], [0, 10, 0], [0, 0, 10]])
    a, b, c, alpha, beta, gamma = box.cell_parameters()
    np.testing.assert_allclose([a, b, c], [10.0, np.sqrt(104.0), 10.0])
    np.testing.assert_allclose([alpha, beta], [90.0, 90.0])
    np.testing.assert_allclose(gamma, np.degrees(np.arccos(2.0 / np.sqrt(104.0))))


def test_triclinic_scaled_positions(write_file: WriteFile) -> None:
    bounds = ((0.0, 12.0, 2.0), (0.0, 10.0, 0.0), (0.0, 10.0, 0.0))
    text = frame_text(
        "id type xs ys zs", [[1, 1, 0.0, 1.0, 0.0]], bounds=bounds,
        box_header="xy xz yz pp pp pp",
    )  # fmt: skip
    path = write_file("t.lammpstrj", text)

    positions = next(read_frames(path)).positions()

    np.testing.assert_allclose(positions, [[2.0, 10.0, 0.0]])


def test_box_edges() -> None:
    box = Box.from_bounds(np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]))

    lengths = sorted(float(np.linalg.norm(q - p)) for p, q in box.edges())

    assert lengths == [1.0] * 4 + [2.0] * 4 + [3.0] * 4


def test_extra_items_are_skipped(write_file: WriteFile) -> None:
    text = "ITEM: UNITS\nreal\nITEM: TIME\n0.5\n" + water_frame(7)
    path = write_file("t.lammpstrj", text)

    assert next(read_frames(path)).timestep == 7


def test_incomplete_last_frame_warns(write_file: WriteFile) -> None:
    truncated = water_frame(100).splitlines(keepends=True)[:-2]
    path = write_file("t.lammpstrj", water_frame(0) + "".join(truncated))

    with pytest.warns(UserWarning, match="incomplete"):
        frames = list(read_frames(path))

    assert len(frames) == 1


def test_changing_atom_ids_raise(write_file: WriteFile) -> None:
    second = frame_text("id type x y z", [[1, 1, 0, 0, 0], [7, 1, 0, 0, 0]])
    first = frame_text("id type x y z", [[1, 1, 0, 0, 0], [2, 1, 0, 0, 0]])
    path = write_file("t.lammpstrj", first + second)

    with pytest.raises(ValueError, match="atom IDs in frame 2"):
        list(read_frames(path))


def test_missing_position_columns_raise(write_file: WriteFile) -> None:
    path = write_file("t.lammpstrj", frame_text("id type vx vy vz", [[1, 1, 0, 0, 0]]))

    with pytest.raises(ValueError, match="one set of positions"):
        list(read_frames(path))


def test_general_triclinic_box_raises(write_file: WriteFile) -> None:
    bounds = ((10, 0, 0, 0), (0, 10, 0, 0), (0, 0, 10, 0))
    text = frame_text("id type x y z", [[1, 1, 0, 0, 0]], bounds=bounds,
                      box_header="abc origin pp pp pp")  # fmt: skip
    path = write_file("t.lammpstrj", text)

    with pytest.raises(ValueError, match="general triclinic"):
        list(read_frames(path))
