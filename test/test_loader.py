"""Tests for the PyMOL commands; skipped where PyMOL is not importable."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import WATER_POSITIONS, frame_text, water_frame

pymol = pytest.importorskip("pymol")

from pymol import cmd  # noqa: E402

from pymol_lammps.loader import load_lammps, register  # noqa: E402

WriteFile = Callable[[str, str], Path]


@pytest.fixture(autouse=True)
def clean_session() -> Iterator[Any]:
    cmd.reinitialize()
    register()
    yield cmd


def atoms(name: str, state: int = 1) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    cmd.iterate_state(
        state, name, "out.append((ID, name, elem, resn, resi, x))", space={"out": out}
    )
    return sorted(out)


def n_bonds(name: str) -> int:
    return len(cmd.get_model(name).bond)


def test_load_with_data_file(water_dump: Path, water_data: Path) -> None:
    load_lammps(str(water_dump), data=str(water_data), object="water")

    assert cmd.count_states("water") == 3
    assert n_bonds("water") == 4
    first = atoms("water")
    assert first[0] == (1, "OW", "O", "SPC", "1", 0.0)
    assert [a[0] for a in first] == [1, 2, 3, 4, 5, 6]
    # Rows are shuffled in the dump; every atom must move +1 A per frame.
    for state in (2, 3):
        x = [a[5] for a in atoms("water", state)]
        expected = np.array(WATER_POSITIONS)[:, 0] + (state - 1)
        np.testing.assert_allclose(x, expected, atol=1e-5)


def test_bonds_are_guessed_without_data_file(water_dump: Path) -> None:
    load_lammps(str(water_dump), elements="O H")

    assert n_bonds("water") == 4
    assert atoms("water")[0][:3] == (1, "1", "O")


def test_box_is_stored_per_state(write_file: WriteFile) -> None:
    # NPT-like run: the box grows from 10 A to 12 A between the frames.
    text = water_frame(0) + water_frame(1).replace("-5.0 5.0", "-6.0 6.0")
    path = write_file("npt.lammpstrj", text)

    load_lammps(str(path), elements="O H", quiet=1)

    assert cmd.get_symmetry("npt", 1)[:3] == [10.0, 10.0, 10.0]
    assert cmd.get_symmetry("npt", 2)[:3] == [12.0, 12.0, 12.0]
    assert cmd.count_states("npt_box") == 2


def test_box_can_be_switched_off(water_dump: Path) -> None:
    load_lammps(str(water_dump), elements="O H", box=0)

    assert "water_box" not in cmd.get_names()


def test_reloading_replaces_the_object(water_dump: Path) -> None:
    load_lammps(str(water_dump), elements="O H")
    load_lammps(str(water_dump), elements="O H", start=2)

    assert cmd.count_states("water") == 2
    assert cmd.count_atoms("water") == 6


def test_pymol_load_opens_gzipped_dumps(write_file: WriteFile) -> None:
    path = write_file("traj.lammpstrj.gz", water_frame(0) + water_frame(1, 1.0))

    cmd.load(str(path))

    assert cmd.count_states("traj") == 2


def test_command_line_usage(water_dump: Path, water_data: Path) -> None:
    cmd.do(f"load_lammps {water_dump}, data={water_data}, interval=2")

    assert cmd.count_states("water") == 2


def test_errors_become_pymol_errors(write_file: WriteFile) -> None:
    path = write_file("bad.lammpstrj", frame_text("id type vx", [[1, 1, 0.0]]))

    with pytest.raises(pymol.CmdException, match="one set of positions"):
        load_lammps(str(path))
