"""Tests for the PyMOL commands; skipped where PyMOL is not importable."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import WATER_POSITIONS, frame_text, water_frame

pymol = pytest.importorskip("pymol")

from pymol import cmd  # noqa: E402

from pymol_lammps.loader import lammps_wrap, load_lammps, register  # noqa: E402

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
    load_lammps(str(water_dump), data=str(water_data), object="water", wrap=0)

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


def box_extent(name: str, state: int) -> tuple[float, float]:
    x: list[float] = []
    cmd.iterate_state(state, name, "x_.append(x)", space={"x_": x})
    return min(x), max(x)


def test_molecules_are_wrapped_by_default(water_dump: Path, water_data: Path) -> None:
    load_lammps(str(water_dump), data=str(water_data), object="water")

    # Frame 3: molecule 2 (O at x = 5.0) has its centre beyond the box edge at 5.
    xs = [a[5] for a in atoms("water", 3)]
    np.testing.assert_allclose(xs, [2.0, 2.96, 2.0, -5.0, -4.04, -5.0], atol=1e-5)


def test_lammps_wrap_switches_after_loading(water_dump: Path, water_data: Path) -> None:
    load_lammps(str(water_dump), data=str(water_data), object="water")
    cmd.show_as("sticks", "water")

    lammps_wrap("water", 0)
    assert box_extent("water", 3)[1] > 5.9
    assert cmd.count_atoms("water and rep sticks") == 6

    lammps_wrap("water", 1)
    assert box_extent("water", 3)[1] < 5.0


def test_lammps_wrap_needs_a_loaded_object() -> None:
    with pytest.raises(pymol.CmdException, match="not loaded with load_lammps"):
        lammps_wrap("nothing")


def test_no_molecule_ids_warns_for_unwrapped_dump(
    write_file: WriteFile, capsys: pytest.CaptureFixture[str]
) -> None:
    """A data file without molecule IDs (atom style atomic) gives nothing to wrap."""
    rows = [[1, 1, 6.0, 0, 0], [2, 1, 7.0, 0, 0]]
    path = write_file("t.lammpstrj", frame_text("id type xu yu zu", rows))
    data = write_file(
        "atomic.data",
        "atomic\n\n2 atoms\n1 atom types\n-5 5 xlo xhi\n-5 5 ylo yhi\n-5 5 zlo zhi\n"
        "\nMasses\n\n1 12.011\n\nAtoms # atomic\n\n1 1 6 0 0\n2 1 7 0 0\n",
    )

    load_lammps(str(path), data=str(data))

    assert "were not wrapped" in capsys.readouterr().out
    assert box_extent("t", 1) == (6.0, 7.0)


def test_dump_without_molecule_ids_is_wrapped_by_guessed_molecules(
    write_file: WriteFile, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without a data file, the bonded pair is one molecule, wrapped as a whole."""
    rows = [[1, 1, 6.0, 0, 0], [2, 1, 7.0, 0, 0]]
    path = write_file("t.lammpstrj", frame_text("id type xu yu zu", rows))

    load_lammps(str(path), elements="C")

    assert "were not wrapped" not in capsys.readouterr().out
    assert box_extent("t", 1) == (-4.0, -3.0)
    assert len(cmd.get_model("t").bond) == 1


@pytest.mark.parametrize(("columns", "expected"), [("id type", 1), ("id mol type", 0)])
def test_guessed_bond_between_two_molecules_is_dropped(
    write_file: WriteFile, columns: str, expected: int
) -> None:
    """Two carbons 1.5 A apart are bonded, unless the dump puts them in two molecules."""
    rows: list[list[object]] = [[1, 1, 0.0, 0, 0], [2, 1, 1.5, 0, 0]]
    if "mol" in columns:
        rows = [[*row[:1], row[0], *row[1:]] for row in rows]
    path = write_file("pair.lammpstrj", frame_text(f"{columns} x y z", rows))

    load_lammps(str(path), elements="C", quiet=1)

    assert n_bonds("pair") == expected


def test_unknown_element_label_is_bonded_like_carbon(
    write_file: WriteFile, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [[1, 1, "X", 0.0, 0, 0], [2, 1, "X", 1.5, 0, 0]]
    path = write_file("labels.lammpstrj", frame_text("id type element x y z", rows))

    load_lammps(str(path))

    assert "no covalent radius for the element labels ['X']" in capsys.readouterr().out
    assert n_bonds("labels") == 1
    assert atoms("labels")[0][2] == "X"


def element_dump(write_file: WriteFile) -> Path:
    """Two-frame water dump with the columns ``id element xu yu zu``, rows shuffled."""
    elements = ["O", "H", "H", "O", "H", "H"]
    texts = []
    for step, (shift, order) in enumerate([(0.0, range(6)), (1.0, [5, 3, 1, 0, 4, 2])]):
        rows = [
            [i + 1, elements[i], WATER_POSITIONS[i][0] + shift, *WATER_POSITIONS[i][1:]]
            for i in order
        ]
        texts.append(frame_text("id element xu yu zu", rows, step))
    return write_file("elements.lammpstrj", "".join(texts))


@pytest.mark.parametrize("wrap", [0, 1])
def test_element_only_dump_with_data_file(
    write_file: WriteFile, water_data: Path, wrap: int
) -> None:
    load_lammps(
        str(element_dump(write_file)), data=str(water_data), object="e", wrap=wrap
    )

    assert cmd.count_states("e") == 2
    assert n_bonds("e") == 4
    first = atoms("e")
    assert first[0] == (1, "OW", "O", "SPC", "1", 0.0)
    x = [a[5] for a in atoms("e", 2)]
    np.testing.assert_allclose(x, np.array(WATER_POSITIONS)[:, 0] + 1, atol=1e-5)


def test_element_only_dump_without_data_file(write_file: WriteFile) -> None:
    load_lammps(str(element_dump(write_file)), object="e", wrap=0)

    assert cmd.count_states("e") == 2
    assert n_bonds("e") == 4
    assert [a[2] for a in atoms("e")] == ["O", "H", "H", "O", "H", "H"]
