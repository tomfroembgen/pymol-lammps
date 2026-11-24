"""Tests for ``drude_style``.

The fixtures ``test/data/drude.*`` are described in ``test_drude.py``: 78 Drudes and
3 M sites among 210 atoms, three frames, the box as the object ``<object>_box``.
"""

import pickle
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

pymol = pytest.importorskip("pymol")

from pymol import cmd  # noqa: E402

from pymol_lammps import loader, style  # noqa: E402
from pymol_lammps.dump import read_frames  # noqa: E402
from pymol_lammps.loader import lammps_wrap, load_lammps, register  # noqa: E402
from pymol_lammps.style import drude_style, refresh_dashes  # noqa: E402

DATA = Path(__file__).parent / "data"
DUMP = str(DATA / "drude.lammpstrj")
DATA_FILE = str(DATA / "drude.data")
N_DRUDES = 78
N_VIRTUAL = 3
N_SATELLITES = N_DRUDES + N_VIRTUAL


@pytest.fixture(autouse=True)
def clean_session() -> Iterator[Any]:
    cmd.reinitialize()
    register()
    yield cmd


@pytest.fixture
def loaded() -> str:
    """Load the fixture as object ``d`` with the sticks of the style ``tf``."""
    load_lammps(DUMP, data=DATA_FILE, object="d", quiet=1)
    cmd.hide("everything", "(d)")
    cmd.show("sticks", "d")
    cmd.show("spheres", "(d) and not bound_to all")
    return "d"


def dash_points(name: str, state: int) -> NDArray[np.float64]:
    """Return the two end points of every dash of a distance object, in one state."""
    session = pickle.loads(zlib.decompress(cmd.get_session(name, 1, 1, 1, 1)))
    entry = next(item for item in session["names"] if item and item[0] == name)
    distance_set = entry[5][2][state - 1]
    if not distance_set:
        return np.zeros((0, 3))
    return np.array(distance_set[1]).reshape(-1, 2, 3)


def per_atom(selection: str, expression: str) -> list[Any]:
    """Return an expression evaluated for each atom of a selection."""
    values: list[Any] = []
    cmd.iterate(
        selection, f"values.append({expression})", space={"values": values, "cmd": cmd}
    )
    return values


def bond_radii() -> list[float | None]:
    """Return the per-bond stick radii that are set between satellites and hosts."""
    result = cmd.get_bond("stick_radius", "d and elem D+M", "d and not elem D+M")
    return [row[2] for _, rows in result for row in rows]


@pytest.mark.parametrize("loader_name", ["load_lammps", "load_xyz"])
def test_loading_gives_the_colours_and_radii_but_keeps_the_representations(
    loader_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def load(name: str) -> None:
        if loader_name == "load_lammps":
            load_lammps(DUMP, data=DATA_FILE, object=name, quiet=1)
        else:
            loader.load_xyz(str(DATA / "drude.xyz"), object=name, quiet=1)

    load("d")
    drude_colour = per_atom("d and elem D", "cmd.get_color_tuple(color)")
    msite_colour = per_atom("d and elem M", "cmd.get_color_tuple(color)")
    assert drude_colour == [pytest.approx(style.COLORS["drude_gray"], abs=1e-3)] * (
        N_DRUDES
    )
    assert msite_colour == [pytest.approx(style.COLORS["vsite_gray"], abs=1e-3)] * (
        N_VIRTUAL
    )
    assert per_atom("d and elem D", "vdw") == [pytest.approx(style.DRUDE_RADIUS)] * (
        N_DRUDES
    )
    assert per_atom("d and elem M", "vdw") == [pytest.approx(style.VSITE_RADIUS)] * (
        N_VIRTUAL
    )
    # The representations are those of a load without the colouring step.
    monkeypatch.setattr(loader, "colour_satellites", lambda *_, **__: None)
    load("plain")
    assert per_atom("d", "(ID, reps)") == per_atom("plain", "(ID, reps)")
    assert "d_drude" not in cmd.get_names("objects")


def test_colours_radii_and_representations(loaded: str) -> None:
    cores_before = per_atom("d and not elem D+M", "vdw")
    drude_style("d", quiet=1)
    drudes = per_atom("d and elem D", "cmd.get_color_tuple(color)")
    msites = per_atom("d and elem M", "cmd.get_color_tuple(color)")
    assert len(drudes) == N_DRUDES
    assert len(msites) == N_VIRTUAL
    assert drudes == [pytest.approx(style.COLORS["drude_gray"], abs=1e-3)] * N_DRUDES
    assert msites == [pytest.approx(style.COLORS["vsite_gray"], abs=1e-3)] * N_VIRTUAL
    # Radii only on the satellites, and as drawn: the global scale is overridden.
    assert per_atom("d and elem D", "vdw") == [pytest.approx(style.DRUDE_RADIUS)] * (
        N_DRUDES
    )
    assert per_atom("d and elem M", "vdw") == [pytest.approx(style.VSITE_RADIUS)] * (
        N_VIRTUAL
    )
    assert per_atom("d and not elem D+M", "vdw") == cores_before
    # Spheres on the satellites, no sticks; the cores keep their sticks.
    assert cmd.count_atoms("d and elem D+M and rep spheres") == N_SATELLITES
    assert cmd.count_atoms("d and elem D+M and rep sticks") == 0
    assert cmd.count_atoms("d and not elem D+M and rep sticks") == 210 - N_SATELLITES
    # Smaller than every atom sphere of the style tf (van der Waals radius 1.5 A).
    assert style.DRUDE_RADIUS < 0.3 * 1.5


def test_dashes_connect_each_satellite_with_its_host(loaded: str) -> None:
    drude_style("d", quiet=1)
    assert "d_drude" in cmd.get_names("objects")
    assert cmd.count_states("d_drude") == 3
    for state, frame in enumerate(read_frames(DATA / "drude.lammpstrj"), start=1):
        points = dash_points("d_drude", state)
        assert len(points) == N_SATELLITES
        lengths = np.linalg.norm(points[:, 0] - points[:, 1], axis=1)
        # A Drude is 0.1-0.4 A from its core in the fixtures; the M site is 0.25 A.
        assert lengths.max() < 0.5
        assert lengths.min() > 0.0
        # The end points are atoms of the object in this state.
        coordinates = np.asarray(cmd.get_coords("d", state))
        for end in (points[:, 0], points[:, 1]):
            distance = np.linalg.norm(end[:, None] - coordinates[None], axis=2)
            assert distance.min(axis=1) == pytest.approx(0.0, abs=1e-3)
        assert frame.positions().shape == (210, 3)


def test_dash_settings(loaded: str) -> None:
    drude_style("d", quiet=1)
    assert float(cmd.get("dash_radius", "d_drude")) == pytest.approx(style.DASH_RADIUS)
    assert float(cmd.get("dash_gap", "d_drude")) == pytest.approx(style.DASH_GAP)
    assert cmd.get("dash_round_ends", "d_drude") == "on"


def test_idempotent_and_box_untouched(loaded: str) -> None:
    extent = cmd.get_extent("d_box")
    drude_style("d", quiet=1)
    first = dash_points("d_drude", 2)
    drude_style("d", quiet=1)
    assert cmd.get_names("objects").count("d_drude") == 1
    assert dash_points("d_drude", 2) == pytest.approx(first)
    assert cmd.count_atoms("d and elem D+M and rep spheres") == N_SATELLITES
    assert "d_box" in cmd.get_names("objects", enabled_only=1)
    assert cmd.get_extent("d_box") == extent
    assert cmd.count_states("d_box") == 3
    # Loading again replaces the dashes with the object.
    load_lammps(DUMP, data=DATA_FILE, object="d", quiet=1)
    assert "d_drude" not in cmd.get_names("objects")


def test_selection_limits_the_styled_atoms(loaded: str) -> None:
    drude_style("d and resn tf2N-", quiet=1)
    styled = cmd.count_atoms("d and elem D+M and rep spheres")
    assert 0 < styled < N_SATELLITES
    assert styled == cmd.count_atoms("d and resn tf2N- and elem D+M")
    assert len(dash_points("d_drude", 1)) == styled


def test_two_objects_get_their_own_dashes(loaded: str) -> None:
    load_lammps(DUMP, data=DATA_FILE, object="e", quiet=1)
    drude_style("all", quiet=1)
    assert len(dash_points("d_drude", 1)) == N_SATELLITES
    assert len(dash_points("e_drude", 1)) == N_SATELLITES


def test_solid_fallback(loaded: str) -> None:
    drude_style("d", dashed=1, quiet=1)
    drude_style("d", dashed=0, quiet=1)
    # The dash object is gone, the bonds are thin sticks.
    assert "d_drude" not in cmd.get_names("objects")
    assert cmd.count_atoms("d and elem D+M and rep sticks") == N_SATELLITES
    radii = bond_radii()
    assert len(radii) == N_SATELLITES
    assert radii == [pytest.approx(style.THIN_STICK_RADIUS)] * N_SATELLITES
    # Going back to dashes restores the normal stick radius and hides the sticks.
    drude_style("d", dashed=1, quiet=1)
    assert cmd.count_atoms("d and elem D+M and rep sticks") == 0
    assert "d_drude" in cmd.get_names("objects")
    assert set(bond_radii()) == {None}


def test_solid_bonds_leave_the_hosts_alone(loaded: str) -> None:
    """In a view of spheres, dashed=0 must not draw the bonds between cores."""
    cmd.hide("everything", "(d)")
    cmd.show("spheres", "d")
    drude_style("d", dashed=0, quiet=1)
    assert cmd.count_atoms("d and not elem D+M and rep sticks") == 0
    assert cmd.count_atoms("d and not elem D+M and rep spheres") == 210 - N_SATELLITES
    assert cmd.count_atoms("d and elem D+M and rep sticks") == N_SATELLITES


@pytest.mark.parametrize("selection", ["d_box", "d_drude", "nothere"])
def test_selection_without_atoms_gives_a_clear_error(
    loaded: str, selection: str
) -> None:
    drude_style("d", quiet=1)
    with pytest.raises(pymol.CmdException, match="is no atom selection"):
        drude_style(selection, quiet=1)


def test_without_satellites_nothing_happens(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cmd.fragment("ala", "frag")
    cmd.show("sticks", "frag")
    drude_style("all")
    assert "no Drudes" in capsys.readouterr().out
    assert cmd.get_names("objects") == ["frag"]
    assert cmd.count_atoms("frag and rep sticks") == cmd.count_atoms("frag")


def test_summary_line(loaded: str, capsys: pytest.CaptureFixture[str]) -> None:
    drude_style("d")
    assert f"{N_SATELLITES} Drudes and virtual sites" in capsys.readouterr().out
    drude_style("d", quiet=1)
    assert capsys.readouterr().out == ""


def ends_match_atoms(state: int) -> bool:
    """Return whether all dash end points of a state are atom positions."""
    points = dash_points("d_drude", state)
    coordinates = np.asarray(cmd.get_coords("d", state))
    ends = points.reshape(-1, 3)
    distance = np.linalg.norm(ends[:, None] - coordinates[None], axis=2)
    return bool(np.allclose(distance.min(axis=1), 0.0, atol=1e-3))


def test_refresh_follows_new_coordinate_sets(loaded: str) -> None:
    drude_style("d", quiet=1)
    # PyMOL does not move a distance object when a coordinate set is replaced.
    cmd.load_coordset(np.asarray(cmd.get_coords("d", 2)) + 1.0, "d", 2)
    assert not ends_match_atoms(2)
    refresh_dashes("d")
    assert ends_match_atoms(2)
    assert float(cmd.get("dash_radius", "d_drude")) == pytest.approx(style.DASH_RADIUS)
    # No dash object, no effect.
    refresh_dashes("nothing")


def test_lammps_wrap_refreshes_the_dashes(
    loaded: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    refreshed: list[str] = []
    monkeypatch.setattr(
        loader, "refresh_dashes", lambda name, **_: refreshed.append(name)
    )
    lammps_wrap("d", 0, quiet=1)
    assert refreshed == ["d"]


def test_dashes_survive_a_deleted_selection(loaded: str) -> None:
    """The dashes remember their atoms, not the name of the user's selection."""
    cmd.select("mysel", "d and resi 1-3")
    drude_style("mysel", quiet=1)
    styled = cmd.count_atoms("mysel and elem D+M")
    cmd.delete("mysel")
    lammps_wrap("d", 0, quiet=1)
    for state in (1, 2, 3):
        assert len(dash_points("d_drude", state)) == styled
        assert ends_match_atoms(state)


def test_dashes_of_several_calls_add_up(loaded: str) -> None:
    drude_style("d and resi 1", quiet=1)
    drude_style("d and resi 2", quiet=1)
    expected = cmd.count_atoms("d and resi 1+2 and elem D+M")
    assert len(dash_points("d_drude", 1)) == expected
    # Solid bonds for molecule 1 take its dashes away, not those of molecule 2.
    drude_style("d and resi 1", dashed=0, quiet=1)
    assert len(dash_points("d_drude", 1)) == cmd.count_atoms(
        "d and resi 2 and elem D+M"
    )


def test_hidden_molecules_stay_hidden(loaded: str) -> None:
    cmd.hide("everything", "d and resn tf2N-")
    hidden = "d and resn tf2N-"
    drude_style("d", quiet=1)
    assert cmd.count_atoms(f"{hidden} and rep spheres+sticks+lines") == 0
    shown = "d and not resn tf2N- and elem D+M"
    assert cmd.count_atoms(f"{shown} and rep spheres") == cmd.count_atoms(shown)
    assert len(dash_points("d_drude", 1)) == cmd.count_atoms(shown)
    drude_style("d", dashed=0, quiet=1)
    assert cmd.count_atoms(f"{hidden} and rep spheres+sticks+lines") == 0


def test_lammps_wrap_leaves_the_object_unchanged_on_a_read_error(
    write_file: Any,
) -> None:
    """A file that became unreadable after loading changes no state."""
    # All atoms one box length along x, so that wrapping moves every molecule.
    lines = Path(DUMP).read_text().splitlines()
    for k, line in enumerate(lines):
        fields = line.split()
        if len(fields) == 6 and not line.startswith("ITEM"):
            fields[3] = f"{float(fields[3]) + 30.0:.4f}"
            lines[k] = " ".join(fields)
    path = write_file("drude.lammpstrj", "\n".join(lines) + "\n")
    load_lammps(str(path), data=DATA_FILE, object="d", quiet=1)
    before = [np.asarray(cmd.get_coords("d", state)) for state in (1, 2, 3)]
    # Spoil one coordinate in the last frame.
    fields = lines[-1].split()
    lines[-1] = " ".join([*fields[:3], "unreadable", *fields[4:]])
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(pymol.CmdException):
        lammps_wrap("d", 0, quiet=1)
    for state, coordinates in enumerate(before, start=1):
        assert np.asarray(cmd.get_coords("d", state)) == pytest.approx(coordinates)


REAL = Path.home() / "test/pymol-lammps-drude/db-case2-rep2"


@pytest.mark.skipif(
    not (REAL / "traj10.lammpstrj").exists(), reason="real trajectory not here"
)
def test_real_system_with_graphene_and_drudes() -> None:
    load_lammps(
        str(REAL / "traj10.lammpstrj"),
        data=str(REAL / "data_from_restart.lmp"),
        object="r",
        quiet=1,
    )
    drude_style("r", quiet=1)
    assert cmd.count_atoms("r and elem D") == 3036
    assert len(dash_points("r_drude", 1)) == 3036
    assert len(dash_points("r_drude", 10)) == 3036
