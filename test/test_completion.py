"""Tests for the tab completion of file arguments."""

from pathlib import Path
from types import SimpleNamespace

import pytest

pymol = pytest.importorskip("pymol")

from pymol import cmd  # noqa: E402

from pymol_lammps.completion import (  # noqa: E402
    PathCompleter,
    register_completion,
)
from pymol_lammps.loader import register  # noqa: E402


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """Folder with two data files sharing a prefix, one other file and a directory."""
    for name in ("system.data", "system.restart", "x.lammpstrj"):
        (tmp_path / name).write_text("")
    (tmp_path / "runs").mkdir()
    return tmp_path


def test_first_argument_unique(folder: Path) -> None:
    completer = PathCompleter(("filename", "data"))
    assert completer.interpret(f"{folder}/x.l") == f"{folder}/x.lammpstrj"


def test_directory_gets_slash(folder: Path) -> None:
    completer = PathCompleter(("data",))
    assert completer.interpret(f"data={folder}/ru") == f"data={folder}/runs/"


def test_tilde_is_expanded(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(folder))
    assert PathCompleter(("data",)).interpret("data=~/sys") is not None
    assert PathCompleter(("data",)).interpret("~/x.l") == f"{folder}/x.lammpstrj"


def test_ambiguous_prefix(folder: Path) -> None:
    result = PathCompleter(("data",)).interpret(f"data={folder}/sys")
    assert result == [f"data={folder}/system.data", f"data={folder}/system.restart"]


def test_no_match_and_other_keywords(folder: Path) -> None:
    completer = PathCompleter(("data",))
    assert completer.interpret(f"data={folder}/nothing") is None
    assert completer.interpret(f"start={folder}/x") is None


def test_register_skips_without_auto_arg() -> None:
    register_completion("load_lammps", ("data",), SimpleNamespace())


def test_register_extends_positions() -> None:
    fake = SimpleNamespace(auto_arg=[{}])
    register_completion("load_xyz", ("filename",), fake)
    assert len(fake.auto_arg) >= 4
    assert all("load_xyz" in position for position in fake.auto_arg)


@pytest.fixture
def parser() -> object:
    cmd.reinitialize()
    register()
    return cmd._parser


def test_pymol_completes_data_keyword(parser: object, folder: Path) -> None:
    line = f"load_lammps x.lammpstrj, data={folder}/system.d"
    assert cmd._parser.complete(line) == (
        f"load_lammps x.lammpstrj, data={folder}/system.data"
    )


def test_pymol_completes_data_at_position_two(parser: object, folder: Path) -> None:
    line = f"load_lammps x.lammpstrj, object=a, data={folder}/system.d"
    assert cmd._parser.complete(line) == (
        f"load_lammps x.lammpstrj, object=a, data={folder}/system.data"
    )


def test_pymol_completes_first_argument(parser: object, folder: Path) -> None:
    assert cmd._parser.complete(f"load_xyz {folder}/ru") == (f"load_xyz {folder}/runs/")


def test_pymol_extends_ambiguous_to_common_prefix(parser: object, folder: Path) -> None:
    line = f"load_lammps x.lammpstrj, data={folder}/sy"
    assert cmd._parser.complete(line) == (
        f"load_lammps x.lammpstrj, data={folder}/system."
    )


def test_pymol_other_keyword_does_not_fail(parser: object) -> None:
    assert cmd._parser.complete("load_lammps x.lammpstrj, start=1") is None
