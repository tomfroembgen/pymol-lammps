"""Tests for ``grid_segments`` and the ``drawgridbox`` command."""

import itertools
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

pymol = pytest.importorskip("pymol")

from pymol import cmd  # noqa: E402

from pymol_lammps.gridbox import drawgridbox, grid_segments  # noqa: E402
from pymol_lammps.loader import register  # noqa: E402


def cell_edges(
    lower: np.ndarray, upper: np.ndarray, cells: tuple[int, int, int]
) -> set[tuple[tuple[float, ...], tuple[float, ...]]]:
    """Return the 12 edges of every cell, as in the original script, rounded."""
    step = (upper - lower) / cells
    edges = set()
    for index in itertools.product(*(range(n) for n in cells)):
        corner = lower + np.array(index) * step
        for axis in range(3):
            for a, b in itertools.product((0, 1), repeat=2):
                offset = np.zeros(3)
                others = [k for k in range(3) if k != axis]
                offset[others] = a * step[others[0]], b * step[others[1]]
                start = corner + offset
                end = start.copy()
                end[axis] += step[axis]
                edges.add((tuple(start.round(9)), tuple(end.round(9))))
    return edges


def split(
    segments: np.ndarray,
    cells: tuple[int, int, int],
    lower: np.ndarray,
    upper: np.ndarray,
) -> set[tuple[tuple[float, ...], tuple[float, ...]]]:
    """Cut each full-length line into cell edges, rounded."""
    step = (upper - lower) / cells
    edges = set()
    for start, end in segments:
        axis = int(np.flatnonzero(end - start)[0])
        for i in range(cells[axis]):
            a, b = start.copy(), start.copy()
            a[axis] = lower[axis] + i * step[axis]
            b[axis] = lower[axis] + (i + 1) * step[axis]
            edges.add((tuple(a.round(9)), tuple(b.round(9))))
    return edges


def test_plain_box_has_12_edges() -> None:
    segments = grid_segments(np.zeros(3), np.array([1.0, 2.0, 3.0]), (1, 1, 1))
    assert segments.shape == (12, 2, 3)
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    np.testing.assert_allclose(np.sort(lengths), [1.0] * 4 + [2.0] * 4 + [3.0] * 4)


@pytest.mark.parametrize("cells", [(1, 1, 1), (2, 3, 1), (3, 2, 4)])
def test_lines_cover_the_cell_edges_of_the_original(
    cells: tuple[int, int, int],
) -> None:
    lower, upper = np.array([-1.0, 0.5, 2.0]), np.array([3.0, 2.5, 8.0])
    segments = grid_segments(lower, upper, cells)
    nx, ny, nz = cells
    assert len(segments) == (ny + 1) * (nz + 1) + (nx + 1) * (nz + 1) + (nx + 1) * (
        ny + 1
    )
    assert split(segments, cells, lower, upper) == cell_edges(lower, upper, cells)


@pytest.fixture
def session() -> Iterator[Any]:
    cmd.reinitialize()
    register()
    cmd.pseudoatom("p1", pos=[0.0, 0.0, 0.0])
    cmd.pseudoatom("p2", pos=[2.0, 4.0, 6.0])
    yield cmd
    cmd.reinitialize()


def test_command_draws_box_with_padding(session: Any) -> None:
    name = drawgridbox("all", nx="2", padding="1", pzhi="3", quiet=1)
    assert name in cmd.get_names()
    lower, upper = cmd.get_extent(name)
    np.testing.assert_allclose(lower, [-1.0, -1.0, -1.0])
    np.testing.assert_allclose(upper, [3.0, 5.0, 10.0])


def test_command_through_pymol_parser(session: Any) -> None:
    cmd.do("drawgridbox p1 or p2, 2, 2, 2, quiet=1")
    cmd.do("drawgridbox p1 or p2, quiet=1")
    boxes = [n for n in cmd.get_names() if n.startswith("gridbox")]
    assert len(boxes) == 2


def test_empty_selection_raises(session: Any) -> None:
    with pytest.raises(pymol.CmdException, match="no atoms"):
        drawgridbox("name XYZ")


def test_zero_cells_raises(session: Any) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        drawgridbox("all", nx=0)
