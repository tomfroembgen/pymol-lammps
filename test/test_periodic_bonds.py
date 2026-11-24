"""Tests for the removal of bonds through the periodic boundary.

The test system is a ring of six carbons along x in a 9 A box (spacing 1.5 A), with
a Drude particle on the first carbon. The ring closes through the boundary, so one
of its bonds is a bond between periodic images.
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import frame_text
from numpy.typing import NDArray

pymol = pytest.importorskip("pymol")

from pymol import cmd  # noqa: E402

from pymol_lammps.bonds import cell_heights, periodic_bonds  # noqa: E402
from pymol_lammps.dump import Box  # noqa: E402
from pymol_lammps.loader import (  # noqa: E402
    lammps_wrap,
    load_lammps,
    load_xyz,
    register,
)
from pymol_lammps.wrapping import Molecules, attach_satellites  # noqa: E402

WriteFile = Callable[[str, str], Path]

BOX = 9.0
BOUNDS = ((0.0, BOX),) * 3
CORE_X = [0.45 + 1.5 * k for k in range(6)]
# Atom 7 is the Drude of atom 1, 0.5 A away across the boundary (at 8.95 in a wrapped
# dump).
DRUDE_X = -0.05

DATA_TEXT = """\
ring through the boundary

7 atoms
7 bonds
2 atom types
1 bond types
0.0 9.0 xlo xhi
0.0 9.0 ylo yhi
0.0 9.0 zlo zhi

Masses

1 12.011
2 0.4

Atoms

1 1 1 0.0 0.0 0.0 0.0
2 1 1 0.0 0.0 0.0 0.0
3 1 1 0.0 0.0 0.0 0.0
4 1 1 0.0 0.0 0.0 0.0
5 1 1 0.0 0.0 0.0 0.0
6 1 1 0.0 0.0 0.0 0.0
7 1 2 0.0 0.0 0.0 0.0

Bonds

1 1 1 2
2 1 2 3
3 1 3 4
4 1 4 5
5 1 5 6
6 1 6 1
7 1 7 1
"""


@pytest.fixture(autouse=True)
def clean_session() -> Iterator[Any]:
    cmd.reinitialize()
    register()
    yield cmd


@pytest.fixture
def data_file(write_file: WriteFile) -> str:
    return str(write_file("ring.data", DATA_TEXT))


def ring_frame(columns: str, step: int, shift: float, wrapped: bool) -> str:
    """Return a frame of the ring, all atoms moved by ``shift`` along x."""
    xs = [*CORE_X, DRUDE_X]
    rows = []
    for i, x in enumerate(xs):
        x += shift
        if wrapped:
            x %= BOX
        rows.append([i + 1, 1, 1 if i < 6 else 2, x, 4.5, 4.5])
    return frame_text(columns, rows, step, BOUNDS)


def bond_pairs(name: str) -> set[tuple[int, int]]:
    """Return the PyMOL bonds of an object as pairs of atom IDs."""
    model = cmd.get_model(name)
    pairs = [sorted(model.atom[i].id for i in b.index) for b in model.bond]
    return {(int(i), int(j)) for i, j in pairs}


def test_periodic_bonds_function() -> None:
    box = Box(origin=np.zeros(3), matrix=np.eye(3) * BOX)
    positions = np.array([[0.1, 0, 0], [8.8, 0, 0], [8.9, 0, 0], [3.0, 0, 0]], float)
    bonds = np.array([[0, 1], [1, 2], [2, 3]])
    elements = ["C", "C", "C", "D"]
    # Bond 0-1 is 8.7 A long, bond 1-2 is short, and the long bond 2-3 involves a
    # Drude, which is never removed.
    assert periodic_bonds(positions, bonds, elements, box).tolist() == [
        True,
        False,
        False,
    ]


def test_attach_satellites_moves_them_to_the_host() -> None:
    index = np.array([0, 1, 2])
    molecules = Molecules.from_bonds(index, np.zeros((0, 2), dtype=np.int64))
    bonds = np.array([[2, 0]])
    attached = attach_satellites(molecules, bonds, np.array([[2, 0]]))
    assert attached.index[2] == attached.index[0]
    assert attached.index[1] != attached.index[0]


def coordinates_by_id(name: str, state: int) -> dict[int, NDArray[np.float64]]:
    """Return the coordinates of a state by PyMOL atom ID."""
    shown: dict[int, NDArray[np.float64]] = {}
    cmd.iterate_state(
        state, name, "shown[ID] = np.array([x, y, z])", space={"shown": shown, "np": np}
    )
    return shown


def test_wrapped_ring_is_cut_at_the_same_bond_in_every_frame(
    write_file: WriteFile, data_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ring drifts by 1.6 A per frame, through the boundary several times."""
    columns = "id mol type x y z"
    shifts = [1.6 * k for k in range(6)]
    text = "".join(ring_frame(columns, k, s, True) for k, s in enumerate(shifts))
    load_lammps(str(write_file("wrapped.lammpstrj", text)), data=data_file, object="r")
    out = capsys.readouterr().out
    assert "bonded through the box boundary" in out
    # Only the ring bond 6-1 (7.5 A in frame 1) is removed, in all frames; the
    # Drude bond stays.
    assert "1 periodic bond removed" in out
    assert bond_pairs("r") == {(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 7)}
    for state, shift in enumerate(shifts, start=1):
        shown = coordinates_by_id("r", state)
        for i, j in bond_pairs("r"):
            assert np.linalg.norm(shown[i] - shown[j]) < BOX / 2
        # The ring moves along its path, shifted as a whole by box vectors so that
        # its centre stays in the box.
        cores = np.array([shown[i][0] for i in range(1, 7)])
        images = (cores - np.array(CORE_X) - shift) / BOX
        assert images == pytest.approx(np.full(6, round(images[0])), abs=1e-4)
        assert 0 <= np.mean([shown[i][0] for i in range(1, 8)]) < BOX
        # The Drude stays next to its core although it was across the boundary.
        assert np.linalg.norm(shown[1] - shown[7]) == pytest.approx(0.5, abs=1e-4)


def test_unwrapped_dump_keeps_the_ring_in_one_piece(
    write_file: WriteFile, data_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    text = ring_frame("id mol type xu yu zu", 0, 0.0, False)
    path = str(write_file("unwrapped.lammpstrj", text))
    load_lammps(path, data=data_file, object="w1", wrap=1)
    out = capsys.readouterr().out
    assert "bonded through" not in out
    assert "1 periodic bond removed" in out
    load_lammps(path, data=data_file, object="w0", wrap=0, quiet=1)
    # The same bonds with and without wrapping, and the Drude bond is kept.
    assert bond_pairs("w1") == bond_pairs("w0")
    assert bond_pairs("w1") == {(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 7)}
    # Nothing is moved: the ring is not cut into single atoms.
    assert np.asarray(cmd.get_coords("w1", 1)) == pytest.approx(
        np.asarray(cmd.get_coords("w0", 1))
    )


def test_xyz_ring_loses_its_periodic_bond(
    write_file: WriteFile, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = "\n".join(f"C {0.75 + 1.5 * k} 4.5 4.5" for k in range(6))
    path = str(write_file("ring.xyz", f"6\n\n{rows}\n"))
    load_xyz(path, cell="9", object="x_ring")
    out = capsys.readouterr().out
    assert "bonded through the cell boundary" in out
    assert "1 periodic bond removed" in out
    assert len(cmd.get_model("x_ring").bond) == 5


def test_normal_molecules_lose_no_bonds(
    write_file: WriteFile, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = "\n".join(f"C {0.75 + 1.5 * k} 4.5 4.5" for k in range(3))
    path = str(write_file("chain.xyz", f"3\n\n{rows}\n"))
    load_xyz(path, cell="9", object="three")
    assert "periodic" not in capsys.readouterr().out
    assert len(cmd.get_model("three").bond) == 2


def test_wrap_zero_with_wrapped_coordinates_keeps_all_bonds(
    write_file: WriteFile, data_file: str
) -> None:
    rows = "\n".join(f"C {0.75 + 1.5 * k} 4.5 4.5" for k in range(6))
    path = str(write_file("ring.xyz", f"6\n\n{rows}\n"))
    load_xyz(path, cell="9", object="x_raw", wrap=0, quiet=1)
    assert len(cmd.get_model("x_raw").bond) == 6
    text = ring_frame("id mol type x y z", 0, 0.0, True)
    dump = str(write_file("raw.lammpstrj", text))
    load_lammps(dump, data=data_file, object="d_raw", wrap=0, quiet=1)
    assert len(bond_pairs("d_raw")) == 7


def test_lammps_wrap_recomputes_the_periodic_bonds(
    write_file: WriteFile, data_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    columns = "id mol type x y z"
    text = ring_frame(columns, 0, 0.0, True) + ring_frame(columns, 1, 1.6, True)
    path = str(write_file("wrapped.lammpstrj", text))
    load_lammps(path, data=data_file, object="r", wrap=0, quiet=1)
    every_bond = {(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6), (1, 7)}
    assert bond_pairs("r") == every_bond
    lammps_wrap("r", 1)
    assert "1 periodic bond removed" in capsys.readouterr().out
    assert bond_pairs("r") == every_bond - {(1, 6)}
    for state in (1, 2):
        shown = coordinates_by_id("r", state)
        for i, j in bond_pairs("r"):
            assert np.linalg.norm(shown[i] - shown[j]) < BOX / 2
    # Back to the file coordinates: the molecule is cut as in the file, and its
    # bonds are all shown again.
    lammps_wrap("r", 0, quiet=1)
    assert bond_pairs("r") == every_bond


def test_triclinic_ring_through_the_skewed_boundary(
    write_file: WriteFile, capsys: pytest.CaptureFixture[str]
) -> None:
    """A carbon ring along the tilted cell vector b (gamma = 60 degrees).

    The ring drifts along b by 1.6 A per frame, so it crosses the boundary.
    """
    box = Box.from_cell_parameters(9.0, 9.0, 9.0, 90.0, 90.0, 60.0)
    centre = box.matrix @ np.array([0.5, 0.0, 0.5])
    b = box.matrix[:, 1] / 9.0
    frames = []
    for k in range(4):
        positions = centre + np.outer(0.75 + 1.5 * np.arange(6) + 1.6 * k, b)
        fractional = positions @ np.linalg.inv(box.matrix).T
        wrapped = (fractional % 1.0) @ box.matrix.T
        rows = "\n".join("C {:.5f} {:.5f} {:.5f}".format(*p) for p in wrapped)
        frames.append(f"6\n\n{rows}\n")
    path = str(write_file("skewed.xyz", "".join(frames)))
    load_xyz(path, cell="9 9 9 90 90 60", object="skewed")
    out = capsys.readouterr().out
    assert "1 periodic bond removed" in out
    model = cmd.get_model("skewed")
    assert len(model.bond) == 5
    half = cell_heights(box).min() / 2
    for state in range(1, 5):
        shown = np.asarray(cmd.get_coords("skewed", state))
        for bond in model.bond:
            i, j = bond.index
            assert np.linalg.norm(shown[i] - shown[j]) == pytest.approx(1.5, abs=1e-3)
            assert np.linalg.norm(shown[i] - shown[j]) < half
