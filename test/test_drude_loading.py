"""Tests that load Drude particles and virtual sites into PyMOL.

The fixtures are described in ``drude_helpers.py``.
"""

import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import frame_text
from drude_helpers import (
    BOX,
    DATA,
    N_ATOMS,
    N_DRUDES,
    N_VIRTUAL,
    WriteFile,
    cores_only,
    file_distances,
    frame_elements_and_positions,
    with_mol_column,
    without_elements,
    wrapped_dump,
)
from numpy.typing import NDArray

pymol = pytest.importorskip("pymol")

from pymol import cmd  # noqa: E402

from pymol_lammps.datafile import read_data  # noqa: E402
from pymol_lammps.drude import satellite_mask  # noqa: E402
from pymol_lammps.dump import read_frames  # noqa: E402
from pymol_lammps.loader import (  # noqa: E402
    lammps_wrap,
    load_lammps,
    load_xyz,
    register,
)
from pymol_lammps.topology import build_topology  # noqa: E402
from pymol_lammps.xyz import read_frames as read_xyz  # noqa: E402

DUMP = str(DATA / "drude.lammpstrj")
DATA_FILE = str(DATA / "drude.data")
XYZ = str(DATA / "drude.xyz")


@pytest.fixture(autouse=True)
def clean_session() -> Iterator[Any]:
    cmd.reinitialize()
    register()
    yield cmd


def count(name: str, selection: str) -> int:
    return int(cmd.count_atoms(f"{name} and ({selection})"))


def pair_distances(
    name: str, pairs: NDArray[np.int64], state: int = 1
) -> NDArray[np.float64]:
    """Return the distance in PyMOL of each atom pair (indices are ID - 1)."""
    c = coordinates_by_id(name, state)
    return np.linalg.norm(c[pairs[:, 0]] - c[pairs[:, 1]], axis=1)


def coordinates_by_id(name: str, state: int = 1) -> NDArray[np.float64]:
    """Return the coordinates of a state in the order of the atom ID.

    PyMOL stores the atoms of an object in its own order, not the one of the IDs.
    """
    ids: list[int] = []
    cmd.iterate(name, "ids.append(ID)", space={"ids": ids})
    c = np.asarray(cmd.get_coords(name, state), dtype=float)
    return c[np.argsort(ids)]


def satellite_bond_pairs(name: str) -> list[tuple[int, int]]:
    """Return the PyMOL bonds that touch a Drude or an M site, as (ID - 1) pairs."""
    model = cmd.get_model(name)
    ids = [a.id - 1 for a in model.atom]
    satellite = {i for i, a in enumerate(model.atom) if a.symbol in ("D", "M")}
    pairs = [
        sorted(ids[i] for i in b.index) for b in model.bond if satellite & set(b.index)
    ]
    return sorted((int(i), int(j)) for i, j in pairs)


def sorted_pairs(pairs: NDArray[np.int64]) -> list[tuple[int, int]]:
    return sorted((int(min(p)), int(max(p))) for p in pairs.tolist())


def dump_positions() -> list[NDArray[np.float64]]:
    return [f.positions() for f in read_frames(DATA / "drude.lammpstrj")]


def xyz_positions() -> list[NDArray[np.float64]]:
    return [f.positions for f in read_xyz(DATA / "drude.xyz")]


def test_load_lammps_with_data_file(drude_pairs: NDArray[np.int64]) -> None:
    load_lammps(DUMP, data=DATA_FILE, object="d", quiet=1)
    assert cmd.count_atoms("d") == N_ATOMS
    assert count("d", "elem D") == N_DRUDES
    assert count("d", "elem M") == N_VIRTUAL
    # PyMOL counts D as hydrogen; the loader removes the Drudes from "hydro".
    assert count("d", "hydro") == count("d", "elem H")
    # The atom types are untouched.
    assert count("d", "numeric_type 20") == N_VIRTUAL
    assert count("d", "numeric_type 21-34") == N_DRUDES
    # Drude and M bonds are PyMOL bonds, one per satellite, as in the data file.
    assert satellite_bond_pairs("d") == sorted_pairs(drude_pairs)
    assert len(cmd.get_model("d").bond) == len(read_data(DATA / "drude.data").bonds)


def test_wrapping_keeps_all_bond_lengths_dump_with_data(
    drude_data: Any, drude_pairs: NDArray[np.int64]
) -> None:
    """Molecule 1 straddles x = 0 and is moved as a whole, with its Drudes."""
    load_lammps(DUMP, data=DATA_FILE, object="d", quiet=1)
    bonds = drude_data.bonds - 1
    for state, positions in enumerate(dump_positions(), start=1):
        for pairs in (drude_pairs, bonds):
            reference = np.linalg.norm(
                positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1
            )
            assert pair_distances("d", pairs, state) == pytest.approx(
                reference, abs=1e-5
            )
    # The Drude of atom 1 is 0.3 A from it in frame 1.
    first = drude_pairs[(drude_pairs == 0).any(axis=1)]
    assert pair_distances("d", first, 1) == pytest.approx([0.3], abs=1e-5)


def test_wrapping_puts_molecule_centres_in_the_box(drude_data: Any) -> None:
    load_lammps(DUMP, data=DATA_FILE, object="d", quiet=1)
    c = coordinates_by_id("d")
    for mol in np.unique(drude_data.mol_ids):
        centre = c[drude_data.mol_ids == mol].mean(axis=0)
        assert ((centre >= 0) & (centre < BOX)).all()


def test_dump_only_with_mol_column(
    drude_pairs: NDArray[np.int64], write_file: WriteFile, drude_data: Any
) -> None:
    text = with_mol_column((DATA / "drude.lammpstrj").read_text(), drude_data.mol_ids)
    load_lammps(str(write_file("mol.lammpstrj", text)), object="d", quiet=1)
    assert count("d", "elem D") == N_DRUDES
    assert count("d", "elem M") == N_VIRTUAL
    # Without a data file each satellite has one bond, to its host atom.
    assert satellite_bond_pairs("d") == sorted_pairs(drude_pairs)
    for state, positions in enumerate(dump_positions(), start=1):
        reference = np.linalg.norm(
            positions[drude_pairs[:, 0]] - positions[drude_pairs[:, 1]], axis=1
        )
        assert pair_distances("d", drude_pairs, state) == pytest.approx(
            reference, abs=1e-5
        )


def test_dump_without_mol_and_data(drude_pairs: NDArray[np.int64]) -> None:
    """No molecule IDs: nothing to wrap with; the satellites still get their bonds."""
    load_lammps(DUMP, object="d", quiet=1, wrap=0)
    assert count("d", "elem D") == N_DRUDES
    assert satellite_bond_pairs("d") == sorted_pairs(drude_pairs)


@pytest.mark.parametrize("wrap", [0, 1])
def test_wrapped_dump_without_mol_and_data_keeps_drudes_at_their_cores(
    drude_pairs: NDArray[np.int64], write_file: WriteFile, wrap: int
) -> None:
    """No Drude is drawn across the box from its core, wrapped or not.

    In frame 1 atom 1 is at x = 0.1 and its Drude at x = 29.8 in the wrapped file.
    """
    path = write_file("wrapped.lammpstrj", wrapped_dump())
    load_lammps(str(path), object="d", quiet=1, wrap=wrap)
    assert satellite_bond_pairs("d") == sorted_pairs(drude_pairs)
    for state, positions in enumerate(dump_positions(), start=1):
        reference = np.linalg.norm(
            positions[drude_pairs[:, 0]] - positions[drude_pairs[:, 1]], axis=1
        )
        assert pair_distances("d", drude_pairs, state) == pytest.approx(
            reference, abs=1e-4
        )
    frame = next(read_frames(path))
    cores = ~satellite_mask(build_topology(frame).elements)
    shown = coordinates_by_id("d")
    if wrap:
        # The molecules found from the guessed bonds are put together again.
        bonds = read_data(DATA_FILE).bonds - 1
        assert pair_distances("d", bonds) == pytest.approx(
            file_distances(frame.positions(), bonds), abs=1e-4
        )
    else:
        # The cores stay where the file has them.
        assert shown[cores] == pytest.approx(frame.positions()[cores], abs=1e-4)


def without_satellite_bonds(pairs: NDArray[np.int64]) -> str:
    """Return the fixture data file without the bonds of the Drudes and M sites."""
    satellite_bonds = {tuple(sorted(p)) for p in (pairs + 1).tolist()}
    lines = (DATA / "drude.data").read_text().splitlines()
    start = lines.index("Bonds") + 2
    kept = [
        line
        for line in lines[start:]
        if line.strip()
        and tuple(sorted(int(v) for v in line.split()[2:4])) not in satellite_bonds
    ]
    head = [
        f"{len(kept)} bonds" if line.endswith(" bonds") else line
        for line in lines[:start]
    ]
    return "\n".join([*head, *kept]) + "\n"


@pytest.mark.parametrize("wrap", [0, 1])
def test_satellites_without_bonds_in_the_data_file_get_their_host(
    drude_pairs: NDArray[np.int64], write_file: WriteFile, wrap: int
) -> None:
    """An M site held by fix rigid has no bond in the data file, for example.

    The dump has wrapped coordinates, so in frame 1 the Drude of atom 1 is at
    x = 29.8 while atom 1 is at x = 0.1.
    """
    data = write_file("nosat.data", without_satellite_bonds(drude_pairs))
    dump = write_file("wrapped.lammpstrj", wrapped_dump())
    load_lammps(str(dump), data=str(data), object="d", wrap=wrap, quiet=1)
    assert satellite_bond_pairs("d") == sorted_pairs(drude_pairs)
    for state, positions in enumerate(dump_positions(), start=1):
        assert pair_distances("d", drude_pairs, state) == pytest.approx(
            file_distances(positions, drude_pairs), abs=1e-4
        )


def test_long_molecule_from_mol_column_keeps_its_drude(write_file: WriteFile) -> None:
    """A chain longer than half the box, molecule IDs but no bonds (no data file).

    Without bonds the atoms of a molecule are placed next to its first atom; the
    last carbon stays at x = 5.498, and its Drude at 5.502 must stay with it rather
    than jump next to the first atom at x = 0.5.
    """
    xs = [0.5, 1.75, 3.0, 4.25, 5.498, 5.502]
    columns = "id mol type element x y z"
    bounds = ((0.0, 10.0),) * 3
    text = ""
    for step, shift in enumerate((0.0, 0.3)):
        rows = [
            [i + 1, 1, 2 if i == 5 else 1, "D" if i == 5 else "C", x + shift, 5, 5]
            for i, x in enumerate(xs)
        ]
        text += frame_text(columns, rows, step, bounds)
    load_lammps(str(write_file("long.lammpstrj", text)), object="long", quiet=1)
    for state in (1, 2):
        shown = coordinates_by_id("long", state)
        assert np.linalg.norm(shown[5] - shown[4]) == pytest.approx(0.004, abs=1e-4)
    assert satellite_bond_pairs("long") == [(4, 5)]


def test_xyz_with_cell_and_boundary_drude(
    drude_data: Any, drude_pairs: NDArray[np.int64]
) -> None:
    load_xyz(XYZ, object="xyzobj", quiet=1)
    assert cmd.count_atoms("xyzobj") == N_ATOMS
    assert count("xyzobj", "elem D") == N_DRUDES
    assert count("xyzobj", "elem M") == N_VIRTUAL
    assert count("xyzobj", "hydro") == count("xyzobj", "elem H")
    assert satellite_bond_pairs("xyzobj") == sorted_pairs(drude_pairs)
    bonds = drude_data.bonds - 1
    # Wrapped and reassembled: every distance equals the minimum-image file value.
    for state, positions in enumerate(xyz_positions(), start=1):
        for pairs in (drude_pairs, bonds):
            assert pair_distances("xyzobj", pairs, state) == pytest.approx(
                file_distances(positions, pairs), abs=1e-5
            )
    # With the file coordinates the cores stay where the file has them, but the
    # Drude of atom 1 is not shown across the box from it.
    lammps_wrap("xyzobj", 0, quiet=1)
    elements, positions = frame_elements_and_positions()
    cores = ~satellite_mask(elements)
    shown = coordinates_by_id("xyzobj")
    assert shown[cores] == pytest.approx(positions[cores], abs=1e-4)
    assert pair_distances("xyzobj", drude_pairs) == pytest.approx(
        file_distances(xyz_positions()[0], drude_pairs), abs=1e-5
    )
    lammps_wrap("xyzobj", 1, quiet=1)
    assert pair_distances("xyzobj", drude_pairs) == pytest.approx(
        file_distances(xyz_positions()[0], drude_pairs), abs=1e-5
    )


def test_xyz_molecules_include_drudes_and_m_sites(drude_data: Any) -> None:
    load_xyz(XYZ, object="xyzobj", quiet=1)
    resi: list[int] = []
    cmd.iterate("xyzobj", "resi_list.append(int(resi))", space={"resi_list": resi})
    # Same molecule sizes as in the data file, in the same order.
    expected = np.bincount(drude_data.mol_ids)[1:].tolist()
    assert np.bincount(resi)[1:].tolist() == expected


def test_xyz_drude_off_reads_deuterium() -> None:
    load_xyz(XYZ, object="xyzobj", quiet=1, drude=0)
    assert count("xyzobj", "elem D") == 0
    assert count("xyzobj", "elem M") == N_VIRTUAL


def test_xyz_rejects_atom_types() -> None:
    with pytest.raises(pymol.CmdException, match="drude=auto"):
        load_xyz(XYZ, object="xyzobj", quiet=1, drude="21 22")


def test_lammps_masses_and_explicit_types(write_file: WriteFile) -> None:
    path = str(write_file("nolabel.lammpstrj", without_elements()))
    load_lammps(path, data=DATA_FILE, object="auto", quiet=1)
    assert count("auto", "elem D") == N_DRUDES
    assert count("auto", "elem M") == N_VIRTUAL
    types = " ".join(str(t) for t in range(21, 35))
    load_lammps(path, data=DATA_FILE, object="listed", quiet=1, drude=types)
    assert count("listed", "elem D") == N_DRUDES


D2O_ROWS = [[1, 1, "O", 5.0, 5.0, 5.0], [2, 2, "D", 5.96, 5.0, 5.0]]
D2O_ROWS.append([3, 2, "D", 4.76, 5.93, 5.0])


def test_deuterium_labelled_d_gives_a_warning(
    write_file: WriteFile, capsys: pytest.CaptureFixture[str]
) -> None:
    bounds = ((0.0, 10.0),) * 3
    text = frame_text("id type element x y z", D2O_ROWS, 0, bounds)
    path = str(write_file("d2o.lammpstrj", text))
    load_lammps(path, object="heavy")
    assert "deuterium, load again with drude=0" in capsys.readouterr().out
    load_lammps(path, object="heavy", quiet=1)
    assert capsys.readouterr().out == ""
    rows = "\n".join(" ".join(str(v) for v in row[2:]) for row in D2O_ROWS)
    load_xyz(str(write_file("d2o.xyz", f"3\n\n{rows}\n")), cell="10")
    assert "deuterium" in capsys.readouterr().out


def test_drude_particles_give_no_deuterium_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    load_lammps(DUMP, data=DATA_FILE, object="d")
    load_lammps(DUMP, object="nodata")
    load_xyz(XYZ, object="xyzobj")
    assert "deuterium" not in capsys.readouterr().out


def test_cores_only_dump_has_no_drudes(write_file: WriteFile) -> None:
    """The project's production dumps hold only the cores; they must load cleanly."""
    text = cores_only((DATA / "drude.lammpstrj").read_text())
    path = write_file("cores.lammpstrj", text)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_lammps(str(path), data=DATA_FILE, object="c", quiet=1)
    assert cmd.count_atoms("c") == N_ATOMS - N_DRUDES - N_VIRTUAL
    assert count("c", "elem D") == 0
    assert count("c", "elem M") == 0
    assert len(cmd.get_model("c").bond) > 0


REAL = Path.home() / "test/pymol-lammps-drude/water-pol_tt"


@pytest.mark.skipif(
    not (REAL / "traj5.lammpstrj").exists(),
    reason="real trajectory not on this machine",
)
def test_real_cores_only_dump() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_lammps(
            str(REAL / "traj5.lammpstrj"),
            data=str(REAL / "data-p.lmp"),
            object="r",
            quiet=1,
        )
    assert cmd.count_states("r") == 5
    assert count("r", "elem D") == 0
