"""Tests for Drude particles and virtual sites.

The fixtures ``test/data/drude.*`` hold three c4c1im+ cations, three tf2N- anions and
three SWM4-NDP waters with their Drudes and M sites (see ``make_drude_fixtures.py``):
78 Drudes and 3 M sites among 210 atoms, in a 30 A cube. The first core atom (ID 1)
sits at x = 0.1 A and its Drude 0.3 A below it in x in frame 1, across the lower x
face. Atoms are numbered by molecule, so atom i of the fixtures has index i - 1 in
the arrays below.
"""

import itertools
import re
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import frame_text
from numpy.typing import NDArray

from pymol_lammps.bonds import guess_bonds, satellite_bonds
from pymol_lammps.datafile import read_data
from pymol_lammps.drude import (
    DrudeSetting,
    label_sites,
    mass_sites,
    parse_drude,
    satellite_mask,
)
from pymol_lammps.dump import Box, read_frames
from pymol_lammps.topology import build_topology
from pymol_lammps.xyz import normalise_element
from pymol_lammps.xyz import read_frames as read_xyz

DATA = Path(__file__).parent / "data"
BOX = 30.0
N_ATOMS = 210
N_DRUDES = 78
N_VIRTUAL = 3
CUBE = Box(origin=np.zeros(3), matrix=np.eye(3) * BOX)
WriteFile = Callable[[str, str], Path]


def file_distances(
    positions: NDArray[np.float64], pairs: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Return the minimum-image distance of each atom pair in the 30 A cube."""
    delta = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    return np.linalg.norm(delta - BOX * np.round(delta / BOX), axis=1)


@pytest.fixture(scope="module")
def data_file() -> Any:
    return read_data(DATA / "drude.data")


@pytest.fixture(scope="module")
def drude_pairs(data_file: Any) -> NDArray[np.int64]:
    """Core-Drude and O-M bonds of the data file as atom indices, Drude first."""
    bonds = data_file.bonds - 1
    satellite = data_file.types >= 20
    return bonds[satellite[bonds].any(axis=1)]


def select_columns(path: Path, keep: list[str]) -> str:
    """Return a dump file's text with only the named columns."""
    out = []
    positions: list[int] = []
    for line in path.read_text().splitlines():
        if line.startswith("ITEM: ATOMS"):
            names = line.split()[2:]
            positions = [names.index(k) for k in keep]
            line = "ITEM: ATOMS " + " ".join(keep)
        elif positions and not line.startswith("ITEM") and len(line.split()) > 2:
            fields = line.split()
            line = " ".join(fields[p] for p in positions)
        out.append(line)
    return "\n".join(out) + "\n"


def without_elements() -> str:
    """Return the fixture dump without its ``element`` column."""
    return select_columns(DATA / "drude.lammpstrj", ["id", "type", "xu", "yu", "zu"])


def with_mol_column(text: str, mol_ids: NDArray[np.int64]) -> str:
    """Insert a ``mol`` column (after ``id``) into the dump text."""
    out = []
    in_atoms = False
    for line in text.splitlines():
        if line.startswith("ITEM: ATOMS"):
            fields = line.split()
            line = " ".join([*fields[:3], "mol", *fields[3:]])
            in_atoms = True
        elif line.startswith("ITEM"):
            in_atoms = False
        elif in_atoms:
            fields = line.split()
            line = " ".join([fields[0], str(mol_ids[int(fields[0]) - 1]), *fields[1:]])
        out.append(line)
    return "\n".join(out) + "\n"


def cores_only(text: str) -> str:
    """Drop Drudes and M sites from a dump with columns ``id type element ...``."""
    frames = text.split("ITEM: TIMESTEP\n")[1:]
    out = []
    for frame in frames:
        lines = frame.splitlines()
        head, rows = lines[:8], lines[8:]
        rows = [r for r in rows if r.split()[2] not in ("D", "M")]
        head[2] = str(len(rows))
        out += ["ITEM: TIMESTEP", *head, *rows]
    return "\n".join(out) + "\n"


def wrapped_dump() -> str:
    """Return the fixture dump with wrapped coordinates ``x y z`` (no image flags)."""
    out = []
    for line in (DATA / "drude.lammpstrj").read_text().splitlines():
        if line.startswith("ITEM: ATOMS"):
            line = "ITEM: ATOMS id type element x y z"
        elif not line.startswith("ITEM") and len(line.split()) == 6:
            fields = line.split()
            wrapped = [f"{float(v) % BOX:.4f}" for v in fields[3:]]
            line = " ".join(fields[:3] + wrapped)
        out.append(line)
    return "\n".join(out) + "\n"


# ---- detection rules -------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "mode", "types"),
    [
        ("auto", "auto", frozenset()),
        ("", "auto", frozenset()),
        (0, "off", frozenset()),
        ("0", "off", frozenset()),
        ("21 22 23", "types", frozenset({21, 22, 23})),
        ("21,22", "types", frozenset({21, 22})),
        ([21, 22], "types", frozenset({21, 22})),
    ],
)
def test_parse_drude(value: Any, mode: str, types: frozenset[int]) -> None:
    assert parse_drude(value) == DrudeSetting(mode, types)  # type: ignore[arg-type]


def test_parse_drude_rejects_text() -> None:
    with pytest.raises(ValueError, match="drude="):
        parse_drude("yes")


@pytest.mark.parametrize(
    ("label", "symbol"),
    [
        ("D", "D"),
        ("d", "D"),
        ("M", "M"),
        ("m", "M"),
        ("D1", "D"),
        ("HW", "H"),
        ("Hw", "H"),
        ("MG", "Mg"),
        ("Mn", "Mn"),
        ("C1", "C"),
        ("CL", "Cl"),
    ],
)
def test_xyz_labels(label: str, symbol: str) -> None:
    assert normalise_element(label) == symbol


@pytest.mark.parametrize("label", ["MW", "DP", "Dw"])
def test_xyz_still_rejects_other_unknown_labels(label: str) -> None:
    with pytest.raises(ValueError, match="unknown element label"):
        normalise_element(label)


def test_label_sites() -> None:
    types = np.array([1, 2, 3, 4])
    auto = label_sites(["C", "d", "M", "H"], types, DrudeSetting())
    assert auto == ["C", "D", "M", "H"]
    off = label_sites(["C", "D", "M", "H"], types, DrudeSetting("off"))
    assert off == ["C", "H", "M", "H"]
    listed = label_sites(["C", "C", "C", "H"], types, parse_drude("2 3"))
    assert listed == ["C", "D", "D", "H"]


def test_mass_sites() -> None:
    types = np.array([1, 2, 3, 4, 5])
    masses = np.array([12.011, 0.4, 1.008, 1e-16, np.nan])
    assert list(mass_sites(types, masses, DrudeSetting())) == ["", "D", "", "M", ""]
    assert list(mass_sites(types, masses, DrudeSetting("off"))) == ["", "", "", "M", ""]
    listed = mass_sites(types, masses, parse_drude("1"))
    assert list(listed) == ["D", "", "", "M", ""]


# ---- topology --------------------------------------------------------------


@pytest.fixture(scope="module")
def first_frame() -> Any:
    return next(read_frames(DATA / "drude.lammpstrj"))


def test_topology_from_element_column(first_frame: Any, data_file: Any) -> None:
    topology = build_topology(first_frame, data_file)
    assert topology.elements.count("D") == N_DRUDES
    assert topology.elements.count("M") == N_VIRTUAL
    # Core-Drude and O-M bonds come from the data file.
    sat = satellite_mask(topology.elements)
    assert sat[topology.bonds].any(axis=1).sum() == N_DRUDES + N_VIRTUAL
    # Atom types and names stay what the data file says.
    assert set(topology.types[sat]) <= {*range(21, 35), 20}


def test_topology_from_masses_equals_element_column(
    first_frame: Any, data_file: Any, write_file: WriteFile
) -> None:
    """Without an element column, masses give D, M and also the reduced core masses."""
    path = write_file("nolabel.lammpstrj", without_elements())
    frame = next(read_frames(path))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        by_mass = build_topology(frame, data_file)
    assert by_mass.elements == build_topology(first_frame, data_file).elements


def test_topology_explicit_types(data_file: Any, write_file: WriteFile) -> None:
    frame = next(read_frames(write_file("nolabel.lammpstrj", without_elements())))
    listed = build_topology(frame, data_file, drude=parse_drude(range(21, 35)))
    assert listed.elements.count("D") == N_DRUDES
    # With a partial list the other Drude types are no Drudes, and the mass rule is
    # off: they stay unresolved and are reported.
    with pytest.warns(UserWarning, match="no element known"):
        partial = build_topology(frame, data_file, drude=parse_drude("21"))
    assert partial.elements.count("D") == int((partial.types == 21).sum())


def test_topology_drude_off(first_frame: Any, data_file: Any) -> None:
    topology = build_topology(first_frame, data_file, drude=parse_drude(0))
    # Label D is read as deuterium; the M site stays a virtual site.
    assert "D" not in topology.elements
    default = build_topology(first_frame, data_file)
    assert topology.elements.count("H") == default.elements.count("H") + N_DRUDES
    assert topology.elements.count("M") == N_VIRTUAL


def test_topology_without_any_mass_has_no_massless_sites(
    write_file: WriteFile,
) -> None:
    """No data file and no element column: zero masses are unknown, not massless."""
    path = write_file("nolabel.lammpstrj", without_elements())
    with pytest.warns(UserWarning, match="no element known"):
        topology = build_topology(next(read_frames(path)))
    assert set(topology.elements) == {"C"}


def test_topology_with_elements_list(first_frame: Any, data_file: Any) -> None:
    labels = [*"N C C C H C H H C C H C C F S N O O H M".split(), *["D"] * 14]
    topology = build_topology(first_frame, data_file, type_elements=labels)
    assert topology.elements == build_topology(first_frame, data_file).elements


def relabel_drudes(label: str) -> str:
    """Return the fixture dump with the element label of every Drude replaced."""
    out = []
    for line in (DATA / "drude.lammpstrj").read_text().splitlines():
        fields = line.split()
        if len(fields) == 6 and fields[2] == "D":
            line = " ".join([*fields[:2], label, *fields[3:]])
        out.append(line)
    return "\n".join(out) + "\n"


def test_data_file_types_override_the_element_column(
    first_frame: Any, data_file: Any, write_file: WriteFile
) -> None:
    """Drude types of mass 0.4 in the data file are Drudes whatever the dump says."""
    frame = next(read_frames(write_file("x.lammpstrj", relabel_drudes("X"))))
    topology = build_topology(frame, data_file)
    assert topology.elements.count("D") == N_DRUDES
    assert "X" not in topology.elements
    assert topology.elements.count("M") == N_VIRTUAL
    # The same for elements given per type.
    labels = [*"N C C C H C H H C C H C C F S N O O H O".split(), *["C"] * 14]
    listed = build_topology(frame, data_file, type_elements=labels)
    assert listed.elements == build_topology(first_frame, data_file).elements


def test_dp_comment_marks_drude_types_of_any_mass(write_file: WriteFile) -> None:
    """fftool writes ``# NA DP``; the word DP makes a type a Drude type."""
    text = (DATA / "drude.data").read_text().replace("0.400  # ", "1.500  # ")
    data = read_data(write_file("dp.data", text))
    assert data.type_names[21] == "NA"
    assert data.type_comments[21] == ["NA", "DP"]
    frame = next(read_frames(write_file("x.lammpstrj", relabel_drudes("X"))))
    assert build_topology(frame, data).elements.count("D") == N_DRUDES


def test_drude_off_keeps_the_labels(data_file: Any, write_file: WriteFile) -> None:
    frame = next(read_frames(write_file("x.lammpstrj", relabel_drudes("X"))))
    topology = build_topology(frame, data_file, drude=parse_drude(0))
    assert "D" not in topology.elements
    assert topology.elements.count("X") == N_DRUDES


def test_cores_only_dump_without_elements_uses_drude_masses_of_the_data_file(
    data_file: Any, write_file: WriteFile
) -> None:
    """The cores of a Drude force field are lighter by the Drude mass (N 13.607)."""
    cores = write_file(
        "cores.lammpstrj", cores_only(DATA.joinpath("drude.lammpstrj").read_text())
    )
    labelled = next(read_frames(cores))
    path = write_file(
        "nolabel.lammpstrj", select_columns(cores, ["id", "type", "xu", "yu", "zu"])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        by_mass = build_topology(next(read_frames(path)), data_file)
    assert by_mass.elements == build_topology(labelled, data_file).elements
    assert {"N", "O", "S", "F"} <= set(by_mass.elements)


def test_unknown_type_warning_does_not_ask_for_the_given_data_file(
    data_file: Any, write_file: WriteFile
) -> None:
    frame = next(read_frames(write_file("nolabel.lammpstrj", without_elements())))
    # With only type 21 as Drude type, the other Drude types stay unresolved.
    with pytest.warns(UserWarning, match="no element known") as caught:
        build_topology(frame, data_file, drude=parse_drude("21"))
    assert "data file" not in str(caught[0].message)
    with pytest.warns(UserWarning, match="a data file"):
        build_topology(frame)


# ---- bonds -----------------------------------------------------------------


def frame_elements_and_positions(n: int = 0) -> tuple[list[str], NDArray[np.float64]]:
    frames = list(read_xyz(DATA / "drude.xyz"))
    return frames[n].elements, frames[n].positions


def test_satellite_bonds_match_the_data_file(drude_pairs: NDArray[np.int64]) -> None:
    elements, positions = frame_elements_and_positions()
    pairs = satellite_bonds(positions, elements, CUBE)
    # One bond per Drude and per M site, to the host the data file names.
    assert len(pairs) == N_DRUDES + N_VIRTUAL
    assert {tuple(p) for p in pairs.tolist()} == {
        tuple(sorted(p)) for p in drude_pairs.tolist()
    }


def test_drude_bond_crosses_the_cell_boundary() -> None:
    """Atom 1 is at x = 0.1 and its Drude at 29.8 in the wrapped xyz file."""
    elements, positions = frame_elements_and_positions()
    assert positions[0, 0] == pytest.approx(0.1)
    pairs = satellite_bonds(positions, elements, CUBE)
    drude = [j for i, j in pairs.tolist() if i == 0]
    assert len(drude) == 1
    assert positions[drude[0], 0] == pytest.approx(29.8)
    # Without the minimum image the nearest atom would be another one.
    plain = satellite_bonds(positions, elements, None)
    assert [j for i, j in plain.tolist() if i == 0] != drude


def test_guess_bonds_keeps_core_bonds_and_isolates_satellites() -> None:
    elements, positions = frame_elements_and_positions()
    bonds = guess_bonds(positions, elements, CUBE)
    sat = satellite_mask(elements)
    # Each satellite has exactly one bond, to a non-satellite atom.
    counts = np.bincount(bonds.ravel(), minlength=len(elements))
    assert (counts[sat] == 1).all()
    assert not sat[bonds[sat[bonds].any(axis=1)]].all(axis=1).any()
    # The bonds between cores are those found for the cores alone.
    core_index = np.flatnonzero(~sat)
    cores = guess_bonds(positions[core_index], [elements[i] for i in core_index], CUBE)
    expected = {tuple(core_index[pair]) for pair in cores.tolist()}
    assert {tuple(b) for b in bonds[~sat[bonds].any(axis=1)].tolist()} == expected


def test_m_site_bonds_to_oxygen_even_if_a_hydrogen_is_closer() -> None:
    elements = ["O", "H", "H", "M", "C"]
    positions = np.array(
        [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0], [0.9, 0.1, 0], [3, 0, 0]], float
    )
    pairs = satellite_bonds(positions, elements)
    assert pairs.tolist() == [[0, 3]]


def test_satellites_do_not_bond_cores_together() -> None:
    """Two carbons 2.5 A apart are not bonded just because a Drude sits between."""
    elements = ["C", "D", "C"]
    positions = np.array([[0, 0, 0], [1.25, 0, 0], [2.5, 0, 0]], float)
    assert guess_bonds(positions, elements).tolist() == [[0, 1]]


def test_drude_without_cores_in_system_gets_oxygenless_fallback() -> None:
    elements = ["C", "M"]
    positions = np.array([[0, 0, 0], [0.5, 0, 0]], float)
    assert satellite_bonds(positions, elements).tolist() == [[0, 1]]


def nearest_by_all_images(
    positions: NDArray[np.float64], box: Box, satellite: int, candidates: list[int]
) -> int:
    """Return the nearest candidate of an atom, testing all 27 neighbouring images."""
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3))) @ box.matrix.T
    delta = positions[candidates] - positions[satellite]
    distance = np.linalg.norm(delta[:, None, :] + shifts[None], axis=2).min(axis=1)
    return candidates[int(np.argmin(distance))]


def test_satellite_hosts_in_a_skewed_cell_match_a_search_of_all_images() -> None:
    """Hosts found with the cell list agree with a search over all atoms and images.

    Most Drudes sit within 0.4 A of a random core, some across the boundary; one
    lies 3 A from every core, beyond the cell-list radius.
    """
    rng = np.random.default_rng(7)
    box = Box.from_cell_parameters(14.0, 15.0, 16.0, 80.0, 95.0, 110.0)
    n_cores = 300
    fractional = rng.random((n_cores, 3))
    cores = box.origin + fractional @ box.matrix.T
    hosts = rng.choice(n_cores, 60, replace=False)
    offsets = rng.normal(size=(60, 3))
    offsets *= (
        rng.uniform(0.05, 0.4, (60, 1)) / np.linalg.norm(offsets, axis=1)[:, None]
    )
    drudes = cores[hosts] + offsets
    lonely = np.array([[7.0, 7.0, 8.0]])
    positions = np.vstack([cores, drudes, lonely])
    elements = ["C"] * n_cores + ["D"] * 61
    # Keep the lonely Drude 3 A from every core.
    distance = np.linalg.norm(cores - lonely, axis=1)
    positions[:n_cores][distance < 3.0] += 10.0

    pairs = satellite_bonds(positions, elements, box)

    host_of = {j: i for i, j in pairs.tolist()}
    candidates = list(range(n_cores))
    for satellite in range(n_cores, n_cores + 61):
        expected = nearest_by_all_images(positions, box, satellite, candidates)
        assert host_of[satellite] == expected


# ---- PyMOL -----------------------------------------------------------------

pymol = pytest.importorskip("pymol")

from pymol import cmd  # noqa: E402

from pymol_lammps.loader import (  # noqa: E402
    lammps_wrap,
    load_lammps,
    load_xyz,
    register,
)

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
    data_file: Any, drude_pairs: NDArray[np.int64]
) -> None:
    """Molecule 1 straddles x = 0 and is moved as a whole, with its Drudes."""
    load_lammps(DUMP, data=DATA_FILE, object="d", quiet=1)
    bonds = data_file.bonds - 1
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


def test_wrapping_puts_molecule_centres_in_the_box(data_file: Any) -> None:
    load_lammps(DUMP, data=DATA_FILE, object="d", quiet=1)
    c = coordinates_by_id("d")
    for mol in np.unique(data_file.mol_ids):
        centre = c[data_file.mol_ids == mol].mean(axis=0)
        assert ((centre >= 0) & (centre < BOX)).all()


def test_dump_only_with_mol_column(
    drude_pairs: NDArray[np.int64], write_file: WriteFile, data_file: Any
) -> None:
    text = with_mol_column((DATA / "drude.lammpstrj").read_text(), data_file.mol_ids)
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


def test_wrapped_dump_without_mol_and_data_keeps_drudes_at_their_cores(
    drude_pairs: NDArray[np.int64], write_file: WriteFile
) -> None:
    """Nothing can be wrapped, but no Drude is drawn across the box from its core.

    In frame 1 atom 1 is at x = 0.1 and its Drude at x = 29.8 in the wrapped file.
    """
    path = write_file("wrapped.lammpstrj", wrapped_dump())
    load_lammps(str(path), object="d", quiet=1)
    assert satellite_bond_pairs("d") == sorted_pairs(drude_pairs)
    for state, positions in enumerate(dump_positions(), start=1):
        reference = np.linalg.norm(
            positions[drude_pairs[:, 0]] - positions[drude_pairs[:, 1]], axis=1
        )
        assert pair_distances("d", drude_pairs, state) == pytest.approx(
            reference, abs=1e-4
        )
    # The cores stay where the file has them.
    frame = next(read_frames(path))
    cores = ~satellite_mask(build_topology(frame).elements)
    shown = coordinates_by_id("d")
    assert shown[cores] == pytest.approx(frame.positions()[cores], abs=1e-4)


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
    data_file: Any, drude_pairs: NDArray[np.int64]
) -> None:
    load_xyz(XYZ, object="xyzobj", quiet=1)
    assert cmd.count_atoms("xyzobj") == N_ATOMS
    assert count("xyzobj", "elem D") == N_DRUDES
    assert count("xyzobj", "elem M") == N_VIRTUAL
    assert count("xyzobj", "hydro") == count("xyzobj", "elem H")
    assert satellite_bond_pairs("xyzobj") == sorted_pairs(drude_pairs)
    bonds = data_file.bonds - 1
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


def test_xyz_molecules_include_drudes_and_m_sites(data_file: Any) -> None:
    load_xyz(XYZ, object="xyzobj", quiet=1)
    resi: list[int] = []
    cmd.iterate("xyzobj", "resi_list.append(int(resi))", space={"resi_list": resi})
    # Same molecule sizes as in the data file, in the same order.
    assert np.bincount(resi)[1:].tolist() == np.bincount(data_file.mol_ids)[1:].tolist()


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
