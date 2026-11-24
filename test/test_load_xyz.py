"""Tests for ``load_xyz``; skipped where PyMOL is not importable."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pymol = pytest.importorskip("pymol")

from pymol import cmd  # noqa: E402

from pymol_lammps.loader import lammps_wrap, load_xyz, register  # noqa: E402

WriteFile = Callable[[str, str], Path]

# Water A is cut by the boundary of a 10 A cube: O at x = 9.8, one H wrapped to 0.76.
# Water B is whole, in the middle of the cell.
CUT_WATER = [
    ("O", 9.8, 5.0, 5.0),
    ("H", 0.76, 5.0, 5.0),
    ("H", 9.8, 5.96, 5.0),
    ("O", 3.0, 3.0, 3.0),
    ("H", 3.96, 3.0, 3.0),
    ("H", 3.0, 3.96, 3.0),
]


@pytest.fixture(autouse=True)
def clean_session() -> Iterator[Any]:
    cmd.reinitialize()
    register()
    yield cmd


def xyz_text(comment: str = "", shift: float = 0.0) -> str:
    rows = [f"{e} {x + shift} {y} {z}" for e, x, y, z in CUT_WATER]
    return f"{len(rows)}\n{comment}\n" + "\n".join(rows) + "\n"


def coordinates(name: str, state: int = 1) -> np.ndarray:
    """Return the coordinates of a selection, in the order of the file."""
    out: list[tuple[int, float, float, float]] = []
    cmd.iterate_state(state, name, "out.append((ID, x, y, z))", space={"out": out})
    return np.array([row[1:] for row in sorted(out)])


def molecule_atoms(name: str, resi: int) -> np.ndarray:
    return coordinates(f"{name} and resi {resi}")


def test_cut_molecule_is_whole_and_inside_the_cell(write_file: WriteFile) -> None:
    path = write_file("waters.xyz", xyz_text())

    load_xyz(str(path), cell="10", quiet=1)

    assert cmd.count_atoms("waters") == 6
    assert len(cmd.get_model("waters").bond) == 4
    # Each water is O plus two H at 0.96 A, so all distances to O are short.
    for resi in (1, 2):
        xyz = molecule_atoms("waters", resi)
        assert np.all(np.linalg.norm(xyz - xyz[0], axis=1) < 2.0)
    for resi in (1, 2):
        centre = molecule_atoms("waters", resi).mean(axis=0)
        assert np.all((centre >= 0) & (centre <= 10))


def test_object_stores_elements_and_molecule_numbers(write_file: WriteFile) -> None:
    path = write_file("waters.xyz", xyz_text())

    load_xyz(str(path), cell="10", quiet=1)

    out: list[tuple[int, str, str, str, str]] = []
    cmd.iterate(
        "waters", "out.append((ID, name, elem, resn, resi))", space={"out": out}
    )
    out.sort()
    assert out[0] == (1, "O", "O", "", "1")
    assert [a[4] for a in out] == ["1", "1", "1", "2", "2", "2"]


def test_cell_string_defaults(write_file: WriteFile) -> None:
    path = write_file("waters.xyz", xyz_text())
    expected = {
        "10": (10, 10, 10, 90, 90, 90),
        "10 11 12": (10, 11, 12, 90, 90, 90),
        "10,11,12,90,90,120": (10, 11, 12, 90, 90, 120),
    }

    for text, parameters in expected.items():
        load_xyz(str(path), cell=text, quiet=1)
        np.testing.assert_allclose(cmd.get_symmetry("waters")[:6], parameters)


def test_box_object_and_switch(write_file: WriteFile) -> None:
    path = write_file("waters.xyz", xyz_text())

    load_xyz(str(path), cell="10", quiet=1)
    assert "waters_box" in cmd.get_names()

    load_xyz(str(path), cell="10", box=0, quiet=1)
    assert "waters_box" not in cmd.get_names()


def test_extended_xyz_lattice_is_used_per_frame(write_file: WriteFile) -> None:
    lattice = 'Lattice="{0} 0 0 0 {0} 0 0 0 {0}" Properties=species:S:1:pos:R:3'
    path = write_file(
        "npt.xyz", xyz_text(lattice.format(10)) + xyz_text(lattice.format(12))
    )

    load_xyz(str(path), quiet=1)

    assert cmd.get_symmetry("npt", 1)[:3] == [10.0, 10.0, 10.0]
    assert cmd.get_symmetry("npt", 2)[:3] == [12.0, 12.0, 12.0]
    assert cmd.count_states("npt_box") == 2
    assert len(cmd.get_model("npt").bond) == 4


def test_cell_wins_over_lattice_and_warns_once(
    write_file: WriteFile, capsys: pytest.CaptureFixture[str]
) -> None:
    lattice = 'Lattice="12 0 0 0 12 0 0 0 12"'
    path = write_file("both.xyz", xyz_text(lattice) + xyz_text(lattice))

    load_xyz(str(path), cell="10", quiet=1)

    assert cmd.get_symmetry("both", 2)[:3] == [10.0, 10.0, 10.0]
    assert capsys.readouterr().out.count("using cell") == 1


def test_without_cell_nothing_is_changed(
    write_file: WriteFile, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_file("waters.xyz", xyz_text())

    load_xyz(str(path), quiet=1)

    assert "waters_box" not in cmd.get_names()
    np.testing.assert_allclose(
        coordinates("waters"), [row[1:] for row in CUT_WATER], atol=1e-5
    )
    assert "warning" not in capsys.readouterr().out
    # Plain distances still number the molecules (the cut water falls apart).
    assert cmd.count_atoms("waters and resi 1") == 2


def test_wrap_zero_keeps_file_coordinates(write_file: WriteFile) -> None:
    path = write_file("waters.xyz", xyz_text())

    load_xyz(str(path), cell="10", wrap=0, quiet=1)

    np.testing.assert_allclose(
        coordinates("waters"), [row[1:] for row in CUT_WATER], atol=1e-5
    )
    assert len(cmd.get_model("waters").bond) == 4
    assert "waters_box" in cmd.get_names()


def test_lammps_wrap_works_on_xyz_objects(write_file: WriteFile) -> None:
    path = write_file("waters.xyz", xyz_text() + xyz_text(shift=0.5))
    load_xyz(str(path), cell="10", quiet=1)
    cmd.show_as("sticks", "waters")
    wrapped = coordinates("waters", 2)

    lammps_wrap("waters", 0, quiet=1)
    np.testing.assert_allclose(
        coordinates("waters", 2),
        [(x + 0.5, y, z) for _, x, y, z in CUT_WATER],
        atol=1e-5,
    )
    assert cmd.count_atoms("waters and rep sticks") == 6

    lammps_wrap("waters", 1, quiet=1)
    np.testing.assert_allclose(coordinates("waters", 2), wrapped, atol=1e-5)


def test_invalid_cell_becomes_a_pymol_error(write_file: WriteFile) -> None:
    path = write_file("waters.xyz", xyz_text())

    with pytest.raises(pymol.CmdException, match="1 .* 3 .* 6"):
        load_xyz(str(path), cell="1 2")


def test_command_line_usage(write_file: WriteFile) -> None:
    path = write_file("waters.xyz", xyz_text() + xyz_text())

    cmd.do(f"load_xyz {path}, cell=10 10 10, interval=2, quiet=1")

    assert cmd.count_states("waters") == 1


def test_chain_through_the_boundary_is_wrapped_by_atom(
    write_file: WriteFile, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [("C", 0.75 + 1.5 * i, 6.0, 6.0) for i in range(8)]
    text = "8\n\n" + "\n".join(f"{e} {x} {y} {z}" for e, x, y, z in rows) + "\n"
    path = write_file("alkane.xyz", text)

    load_xyz(str(path), cell="12", quiet=1)

    x = coordinates("alkane")[:, 0]
    assert np.all((x >= 0) & (x <= 12))
    assert "1 molecule(s)" in capsys.readouterr().out


def test_crystal_stays_inside_the_cell(write_file: WriteFile) -> None:
    rows = [
        ("Si", 2.3 * i + 0.5, 2.3 * j + 0.5, 2.3 * k + 0.5)
        for i in range(4)
        for j in range(4)
        for k in range(4)
    ]
    text = "64\n\n" + "\n".join(f"{e} {x} {y} {z}" for e, x, y, z in rows) + "\n"
    path = write_file("crystal.xyz", text)

    load_xyz(str(path), cell="9.2", quiet=1)

    xyz = coordinates("crystal")
    assert np.all((xyz >= 0) & (xyz <= 9.2))


def test_atom_at_zero_stays_at_zero(write_file: WriteFile) -> None:
    path = write_file("edge.xyz", "1\n\nAr 0.0 5.0 5.0\n")

    load_xyz(str(path), cell="10", quiet=1)

    assert coordinates("edge")[0, 0] == 0.0
