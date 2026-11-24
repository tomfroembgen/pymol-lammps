"""Display style of Drude particles and virtual sites in PyMOL.

``drude_style`` draws Drude particles (``elem D``) as small dark-gray spheres and
virtual sites (``elem M``) as small light-gray spheres. Their bonds to the host atom
are replaced by thin dashed lines: a PyMOL ``distance`` object in bond mode, with
the measurement labels hidden. PyMOL cannot dash sticks, and a distance object follows
all states, so it moves with the trajectory. With ``dashed=0``, thin solid sticks
are used instead.

Sizes and colours are the module constants below; the colours are registered in
PyMOL as ``drude_gray`` and ``vsite_gray``.
"""

from typing import Any

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

SATELLITES = f"{DRUDE}+{VIRTUAL}"

# Selections of the dashed-bond object of each object: Drudes and virtual sites, hosts.
_DASH_SELECTIONS: dict[str, tuple[str, str]] = {}


def dash_name(object: str) -> str:
    """Return the name of the dashed-bond object of an object."""
    return f"{object}_drude"


def _style_atoms(satellites: str, hosts: str, _self: Any) -> None:
    """Show Drudes and virtual sites as small coloured spheres without sticks."""
    for symbol, radius, color in (
        (DRUDE, DRUDE_RADIUS, DRUDE_COLOR),
        (VIRTUAL, VSITE_RADIUS, VSITE_COLOR),
    ):
        atoms = f"({satellites}) and elem {symbol}"
        # Parentheses make it an atom selection; "all" would also hide the box CGO.
        _self.hide("everything", f"({atoms})")
        _self.alter(atoms, f"vdw = {radius}")
        _self.set("sphere_scale", 1.0, atoms)
        _self.show("spheres", atoms)
        _self.color(color, atoms)
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

    The command only changes the atoms of the selection that are Drudes or virtual
    sites. It can be repeated: the dash object is replaced. Call it after ``tf`` or
    after any command that shows sticks on all atoms. The box object is untouched.

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
    """
    use_dashes = bool(int(dashed))
    objects = [
        name
        for name in _self.get_object_list(f"({selection})")
        if _self.count_atoms(f"({selection}) and ({name}) and elem {SATELLITES}")
    ]
    for name, rgb in COLORS.items():
        _self.set_color(name, list(rgb))
    for name in objects:
        satellites = f"({selection}) and ({name}) and elem {SATELLITES}"
        hosts = f"({name}) and not elem {SATELLITES}"
        _self.delete(dash_name(name))
        _style_atoms(satellites, hosts, _self)
        if use_dashes:
            _self.unset_bond("stick_radius", satellites, hosts)
            _dashed_bonds(name, satellites, hosts, _self)
        else:
            # A bond is drawn only if both atoms show sticks.
            _self.show(
                "sticks", f"({satellites}) or (({hosts}) and neighbor ({satellites}))"
            )
            _self.set_bond("stick_radius", THIN_STICK_RADIUS, satellites, hosts)
    if not int(quiet):
        count = _self.count_atoms(f"({selection}) and elem {SATELLITES}")
        print(
            f" drude_style: {count} Drudes and virtual sites styled in "
            f"{len(objects)} object(s)"
            if objects
            else " drude_style: no Drudes or virtual sites in the selection"
        )


def _measure(name: str, reset: int, _self: Any) -> None:
    """Create or update the dashed-bond object of ``name`` from its selections."""
    satellites, hosts = _DASH_SELECTIONS[name]
    dashes = dash_name(name)
    # Mode 1 measures only bonded pairs. The atoms are selected by element, so it
    # does not matter in which order PyMOL stores them. State 0 covers all states.
    _self.distance(
        dashes, satellites, hosts, mode=1, state=0, zoom=0, reset=reset, quiet=1
    )
    _self.hide("labels", dashes)


def _dashed_bonds(name: str, satellites: str, hosts: str, _self: Any) -> None:
    """Draw the satellite bonds of an object as a dashed distance object."""
    dashes = dash_name(name)
    _DASH_SELECTIONS[name] = (satellites, hosts)
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
    if object in _DASH_SELECTIONS and dash_name(object) in _self.get_names("objects"):
        _measure(object, 1, _self)
