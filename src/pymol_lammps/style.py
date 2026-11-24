"""Display style of Drude particles and virtual sites in PyMOL.

``drude_style`` draws Drude particles (``elem D``) as small dark-gray spheres and
virtual sites (``elem M``) as small light-gray spheres. Their bonds to the host atom
are replaced by thin dashed lines: a PyMOL ``distance`` object in bond mode, with
the measurement labels hidden. PyMOL cannot dash sticks, and a distance object follows
all states, so it moves with the trajectory. With ``dashed=0``, thin solid sticks
are used instead. Only satellites that are shown, or whose host atom is shown, are
changed, so that hidden molecules stay hidden.

The dashed satellites of each object are remembered by atom ID, so that the dashes
can be recomputed after new coordinates are loaded, whatever happened to the
selection that the user gave.

Sizes and colours are the module constants below; the colours are registered in
PyMOL as ``drude_gray`` and ``vsite_gray``.
"""

from typing import Any

import pymol
from pymol import cmd

from pymol_lammps.drude import DRUDE, VIRTUAL

# Sphere radii in Angstrom as drawn, independent of the global ``sphere_scale``. An
# ion under the style ``tf`` is drawn with about 0.45-0.55 A (van der Waals radius
# 1.5-1.8 A times ``sphere_scale`` 0.3), and a stick has radius 0.2 A, so a Drude
# must be larger than a stick to be seen and clearly smaller than an ion.
DRUDE_RADIUS = 0.25
VSITE_RADIUS = 0.25

# Not used by the element colours of PyMOL or of ``tf`` (orange C, red O, white H,
# blue N, yellow S).
DRUDE_COLOR = "drude_gray"
VSITE_COLOR = "vsite_gray"
COLORS = {DRUDE_COLOR: (0.3, 0.3, 0.3), VSITE_COLOR: (0.72, 0.72, 0.72)}

# Dashed bond: length and gap of a dash and its radius, all in Angstrom; the bond
# of a Drude is only 0.1-0.4 A long, so the dashes are short.
DASH_LENGTH = 0.08
DASH_GAP = 0.07
DASH_RADIUS = 0.045
# Radius of the solid fallback.
THIN_STICK_RADIUS = 0.05
# Longest bond in Angstrom that gets a dash. With a cutoff, PyMOL searches only
# nearby pairs, which in our measurements made the dash object about 40 times
# faster to create; a satellite is placed at the minimum image of its host, so its
# bond is far shorter.
DASH_CUTOFF = 5.0

SATELLITES = f"{DRUDE}+{VIRTUAL}"

# Representations that make an atom visible; a label alone does not count.
SHOWN = "rep " + "+".join(
    [
        "lines",
        "sticks",
        "spheres",
        "nb_spheres",
        "nonbonded",
        "dots",
        "mesh",
        "surface",
        "cartoon",
        "ribbon",
        "ellipsoids",
    ]
)

# Atom IDs of the Drudes and virtual sites with dashed bonds, per object.
_DASHED: dict[str, list[int]] = {}


def dash_name(object: str) -> str:
    """Return the name of the dashed-bond object of an object."""
    return f"{object}_drude"


def _id_selection(name: str, ids: list[int]) -> str:
    """Return a selection of the atoms of an object with the given IDs."""
    return f"({name}) and id {'+'.join(str(i) for i in ids)}"


# Radius and colour of each satellite element.
_LOOK = ((DRUDE, DRUDE_RADIUS, DRUDE_COLOR), (VIRTUAL, VSITE_RADIUS, VSITE_COLOR))


def colour_satellites(selection: str, *, _self: Any = cmd) -> None:
    """Give the Drudes and virtual sites of a selection their colour and radius.

    The loaders call this, so that the particles have their look as soon as they
    are shown as spheres; the representations are not changed.

    Parameters
    ----------
    selection : str
        Atoms whose Drudes (``elem D``) and virtual sites (``elem M``) are changed.
    """
    for name, rgb in COLORS.items():
        _self.set_color(name, list(rgb))
    for symbol, radius, color in _LOOK:
        atoms = f"({selection}) and elem {symbol}"
        _self.alter(atoms, f"vdw = {radius}")
        _self.color(color, atoms)


def _style_atoms(satellites: str, _self: Any) -> None:
    """Show Drudes and virtual sites as small coloured spheres without sticks."""
    colour_satellites(satellites, _self=_self)
    for symbol, _, _ in _LOOK:
        atoms = f"({satellites}) and elem {symbol}"
        # Parentheses make it an atom selection; "all" would also hide the box CGO.
        _self.hide("everything", f"({atoms})")
        _self.set("sphere_scale", 1.0, atoms)
        _self.show("spheres", atoms)
        _self.rebuild(atoms, "spheres")


def drude_style(
    selection: str = "all",
    dashed: int | str = 1,
    quiet: int | str = 0,
    *,
    _self: Any = cmd,
) -> None:
    """Draw Drude particles and virtual sites as small spheres with thin bonds.

    Drude particles (``elem D``) become dark-gray spheres, virtual sites
    (``elem M``, e.g. the M site of SWM4-NDP water) light-gray spheres, both much
    smaller than the atoms. Their sticks are hidden; the bond to the host atom is
    drawn as a thin dashed line in the distance object ``<object>_drude``, which
    follows all states. With ``dashed=0``, the bonds are thin solid sticks instead.

    The command only changes the Drudes and virtual sites of the selection that are
    shown or whose host atom is shown; hidden molecules stay hidden. The
    representations of the host atoms are never changed. A stick bond is drawn
    only between two atoms that both show sticks, so with ``dashed=0`` the thin
    bond appears only where the host shows sticks; in a view of lines or spheres
    the satellites get no visible bond. The command can be repeated: the dashes of
    several calls add up, and ``dashed=0`` replaces the dashes of the atoms it
    styles. Call it after ``tf`` or after any command that shows sticks on all
    atoms. The box object is untouched.

    The dash object holds one set of dashes per state, so the time to create and
    update it grows with the number of frames: for 3036 Drudes, the update took
    0.02 s for 1 frame and 0.06 s for 10 frames in our test. Bonds longer than
    ``DASH_CUTOFF`` (5 A) get no dash.

    PyMOL usage::

        drude_style [ selection [, dashed [, quiet ]]]

    Examples::

        drude_style
        drude_style traj, dashed=0
        hide everything, traj_drude    # only the dashes (also: delete traj_drude)
        show dashes, traj_drude
        alter elem D, vdw=0.2          # smaller Drudes
        rebuild

    Parameters
    ----------
    selection : str, optional
        Atoms to style; their Drudes and virtual sites are changed, in every object
        that the selection touches.
    dashed : int, optional
        1 for dashed bonds, 0 for thin solid sticks.
    quiet : int, optional
        1 to suppress the summary line.

    Raises
    ------
    pymol.CmdException
        If the selection names no atoms of a molecular object (for example the box
        or the dash object, or a name that does not exist).
    """
    use_dashes = bool(int(dashed))
    objects = _self.get_object_list(f"({selection})")
    if objects is None:
        raise pymol.CmdException(
            f"drude_style: '{selection}' is no atom selection; give atoms or a "
            "molecular object, not the box or the dash object"
        )
    n_styled = 0
    n_objects = 0
    for name in objects:
        hosts = f"({name}) and not elem {SATELLITES}"
        shown_hosts = f"({hosts}) and ({SHOWN})"
        ids: list[int] = []
        _self.iterate(
            f"({selection}) and ({name}) and elem {SATELLITES} "
            f"and (({SHOWN}) or neighbor ({shown_hosts}))",
            "ids.append(ID)",
            space={"ids": ids},
        )
        if not ids:
            continue
        n_styled += len(ids)
        n_objects += 1
        # The IDs fix the atoms before their representations change.
        satellites = _id_selection(name, ids)
        _style_atoms(satellites, _self)
        dashed_ids: set[int] = set()
        if dash_name(name) in _self.get_names("objects"):
            dashed_ids = set(_DASHED.get(name, []))
        if use_dashes:
            _self.unset_bond("stick_radius", satellites, hosts)
            dashed_ids |= set(ids)
        else:
            # A stick bond is drawn only if both atoms show sticks; the hosts keep
            # their representations.
            _self.show("sticks", satellites)
            _self.set_bond("stick_radius", THIN_STICK_RADIUS, satellites, hosts)
            dashed_ids -= set(ids)
        _self.delete(dash_name(name))
        _DASHED.pop(name, None)
        if dashed_ids:
            _dashed_bonds(name, sorted(dashed_ids), _self)
    if not int(quiet):
        print(
            f" drude_style: {n_styled} Drudes and virtual sites styled in "
            f"{n_objects} object(s)"
            if n_objects
            else " drude_style: no Drudes or virtual sites shown in the selection"
        )


def _measure(name: str, reset: int, _self: Any) -> None:
    """Create or update the dashed-bond object of ``name`` from its atom IDs."""
    satellites = _id_selection(name, _DASHED[name])
    hosts = f"({name}) and not elem {SATELLITES}"
    dashes = dash_name(name)
    # Mode 1 measures only bonded pairs. The atoms are selected by ID, so it does
    # not matter in which order PyMOL stores them. State 0 covers all states.
    _self.distance(
        dashes,
        satellites,
        hosts,
        cutoff=DASH_CUTOFF,
        mode=1,
        state=0,
        zoom=0,
        reset=reset,
        quiet=1,
    )
    _self.hide("labels", dashes)


def _dashed_bonds(name: str, ids: list[int], _self: Any) -> None:
    """Draw the bonds of the satellites with these IDs as a dashed distance object."""
    dashes = dash_name(name)
    _DASHED[name] = ids
    _measure(name, 0, _self)
    _self.set("dash_length", DASH_LENGTH, dashes)
    _self.set("dash_gap", DASH_GAP, dashes)
    _self.set("dash_radius", DASH_RADIUS, dashes)
    _self.set("dash_round_ends", 1, dashes)
    _self.set("dash_color", DRUDE_COLOR, dashes)


def refresh_dashes(object: str, *, _self: Any = cmd) -> None:
    """Recompute the dashes of an object after its coordinates were replaced.

    A distance object follows moving atoms, but not new coordinate sets loaded with
    ``load_coordset`` (as ``lammps_wrap`` does). Nothing happens if the object has
    no dash object from ``drude_style``.

    Parameters
    ----------
    object : str
        Name of the molecular object.
    """
    if object in _DASHED and dash_name(object) in _self.get_names("objects"):
        _measure(object, 1, _self)
