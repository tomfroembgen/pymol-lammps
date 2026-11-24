"""Tests for Drude particles and virtual sites that need only NumPy.

The tests that load files into PyMOL are in ``test_drude_loading.py``.

The fixtures ``test/data/drude.*`` hold three c4c1im+ cations, three tf2N- anions and
three SWM4-NDP waters with their Drudes and M sites (see ``make_drude_fixtures.py``):
78 Drudes and 3 M sites among 210 atoms, in a 30 A cube. The first core atom (ID 1)
sits at x = 0.1 A and its Drude 0.3 A below it in x in frame 1, across the lower x
face. Atoms are numbered by molecule, so atom i of the fixtures has index i - 1 in
the arrays below.
"""

import itertools
import warnings
from typing import Any

import numpy as np
import pytest
from drude_helpers import (
    CUBE,
    DATA,
    N_DRUDES,
    N_VIRTUAL,
    WriteFile,
    cores_only,
    frame_elements_and_positions,
    relabel_drudes,
    select_columns,
    without_elements,
)
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


def test_topology_from_element_column(first_frame: Any, drude_data: Any) -> None:
    topology = build_topology(first_frame, drude_data)
    assert topology.elements.count("D") == N_DRUDES
    assert topology.elements.count("M") == N_VIRTUAL
    # Core-Drude and O-M bonds come from the data file.
    sat = satellite_mask(topology.elements)
    assert sat[topology.bonds].any(axis=1).sum() == N_DRUDES + N_VIRTUAL
    # Atom types and names stay what the data file says.
    assert set(topology.types[sat]) <= {*range(21, 35), 20}


def test_topology_from_masses_equals_element_column(
    first_frame: Any, drude_data: Any, write_file: WriteFile
) -> None:
    """Without an element column, masses give D, M and also the reduced core masses."""
    path = write_file("nolabel.lammpstrj", without_elements())
    frame = next(read_frames(path))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        by_mass = build_topology(frame, drude_data)
    assert by_mass.elements == build_topology(first_frame, drude_data).elements


def test_topology_explicit_types(drude_data: Any, write_file: WriteFile) -> None:
    frame = next(read_frames(write_file("nolabel.lammpstrj", without_elements())))
    listed = build_topology(frame, drude_data, drude=parse_drude(range(21, 35)))
    assert listed.elements.count("D") == N_DRUDES
    # With a partial list the other Drude types are no Drudes, and the mass rule is
    # off: they stay unresolved and are reported.
    with pytest.warns(UserWarning, match="no element known"):
        partial = build_topology(frame, drude_data, drude=parse_drude("21"))
    assert partial.elements.count("D") == int((partial.types == 21).sum())


def test_topology_drude_off(first_frame: Any, drude_data: Any) -> None:
    topology = build_topology(first_frame, drude_data, drude=parse_drude(0))
    # Label D is read as deuterium; the M site stays a virtual site.
    assert "D" not in topology.elements
    default = build_topology(first_frame, drude_data)
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


def test_topology_with_elements_list(first_frame: Any, drude_data: Any) -> None:
    labels = [*"N C C C H C H H C C H C C F S N O O H M".split(), *["D"] * 14]
    topology = build_topology(first_frame, drude_data, type_elements=labels)
    assert topology.elements == build_topology(first_frame, drude_data).elements


def test_data_file_types_override_the_element_column(
    first_frame: Any, drude_data: Any, write_file: WriteFile
) -> None:
    """Drude types of mass 0.4 in the data file are Drudes whatever the dump says."""
    frame = next(read_frames(write_file("x.lammpstrj", relabel_drudes("X"))))
    topology = build_topology(frame, drude_data)
    assert topology.elements.count("D") == N_DRUDES
    assert "X" not in topology.elements
    assert topology.elements.count("M") == N_VIRTUAL
    # The same for elements given per type.
    labels = [*"N C C C H C H H C C H C C F S N O O H O".split(), *["C"] * 14]
    listed = build_topology(frame, drude_data, type_elements=labels)
    assert listed.elements == build_topology(first_frame, drude_data).elements


def test_dp_comment_marks_drude_types_of_any_mass(write_file: WriteFile) -> None:
    """fftool writes ``# NA DP``; the word DP makes a type a Drude type."""
    text = (DATA / "drude.data").read_text().replace("0.400  # ", "1.500  # ")
    data = read_data(write_file("dp.data", text))
    assert data.type_names[21] == "NA"
    assert data.type_comments[21] == ["NA", "DP"]
    frame = next(read_frames(write_file("x.lammpstrj", relabel_drudes("X"))))
    assert build_topology(frame, data).elements.count("D") == N_DRUDES


def test_drude_off_keeps_the_labels(drude_data: Any, write_file: WriteFile) -> None:
    frame = next(read_frames(write_file("x.lammpstrj", relabel_drudes("X"))))
    topology = build_topology(frame, drude_data, drude=parse_drude(0))
    assert "D" not in topology.elements
    assert topology.elements.count("X") == N_DRUDES


def test_cores_only_dump_without_elements_uses_drude_masses_of_the_data_file(
    drude_data: Any, write_file: WriteFile
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
        by_mass = build_topology(next(read_frames(path)), drude_data)
    assert by_mass.elements == build_topology(labelled, drude_data).elements
    assert {"N", "O", "S", "F"} <= set(by_mass.elements)


def test_unknown_type_warning_does_not_ask_for_the_given_data_file(
    drude_data: Any, write_file: WriteFile
) -> None:
    frame = next(read_frames(write_file("nolabel.lammpstrj", without_elements())))
    # With only type 21 as Drude type, the other Drude types stay unresolved.
    with pytest.warns(UserWarning, match="no element known") as caught:
        build_topology(frame, drude_data, drude=parse_drude("21"))
    assert "data file" not in str(caught[0].message)
    with pytest.warns(UserWarning, match="a data file"):
        build_topology(frame)


# ---- bonds -----------------------------------------------------------------


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


def distance_over_images(delta: NDArray[np.float64], box: Box) -> NDArray[np.float64]:
    """Return the minimum-image length of difference vectors, testing 13^3 images.

    In the cells of the test (edges 5-15 A, tilt at most half an edge) the nearest
    image is never more than six cell vectors away from the rounded one.
    """
    fractional = delta @ np.linalg.inv(box.matrix).T
    fractional -= np.round(fractional)
    shifts = np.array(list(itertools.product(range(-6, 7), repeat=3)), dtype=float)
    images = (fractional[:, None, :] + shifts[None]) @ box.matrix.T
    return np.linalg.norm(images, axis=2).min(axis=1)


# Random cells in which rounding the fractional difference alone picks a wrong host
# for one satellite (found by a search over 200 seeds).
@pytest.mark.parametrize("seed", [79, 98, 198])
def test_far_satellites_in_skewed_cells_get_the_nearest_host(seed: int) -> None:
    """Satellites far from every core are compared with all cores and images.

    The cells are triclinic as in LAMMPS, with a tilt of up to half an edge.
    """
    rng = np.random.default_rng(seed)
    a, b, c = rng.uniform(5.0, 15.0, 3)
    xy, xz, yz = rng.uniform(-0.5, 0.5, 3) * np.array([a, a, b])
    matrix = np.array([[a, xy, xz], [0, b, yz], [0, 0, c]])
    box = Box(origin=np.zeros(3), matrix=matrix)
    n_cores, n_drudes = 30, 12
    positions = rng.uniform(-0.2, 1.2, (n_cores + n_drudes, 3)) @ matrix.T
    elements = ["C"] * n_cores + ["D"] * n_drudes

    pairs = satellite_bonds(positions, elements, box)

    assert len(pairs) == n_drudes
    for core, drude in pairs.tolist():
        distances = distance_over_images(positions[:n_cores] - positions[drude], box)
        assert distances[core] == pytest.approx(distances.min(), abs=1e-9)


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
