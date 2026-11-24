"""Drude particles and virtual sites of polarizable force fields.

Polarizable force fields such as CL&Pol attach a light charged *Drude particle* to
a core atom by a stiff spring. Water models such as SWM4-NDP add a massless
*virtual site* (the M site) that carries the oxygen charge. Neither is a chemical
element, so they are stored with the pseudo-elements ``D`` and ``M``. Both are
called *satellites* in the code: they always belong to one core atom.

Detection, in order of precedence (``drude=auto``):

1. Drude types given by the user (``drude=21 22 23``): these atom types are Drudes,
   and only these (plus atoms labelled ``D``). The mass rule below is then off.
2. Element label ``D`` (dump ``element`` column, ``elements=`` list, xyz label;
   case-insensitive) is a Drude, label ``M`` is a virtual site.
3. Only if the elements are taken from masses (no element column, no ``elements=``):
   a type with 0 < mass < 0.8 g/mol is a Drude (Drude masses are about 0.4, the
   lightest real atom H has 1.008), and a type with mass below 1e-8 g/mol is a
   virtual site. This needs a known mass, so a dump without ``mass`` column and
   without data file never produces satellites this way.

A data file overrides the element labels of the dump and of ``elements=``: an
atom type that the data file marks as Drude (rule 3 on its ``Masses``, or the word
``DP`` in the comment of the ``Masses`` line, as fftool writes it) becomes ``D``,
and a massless type becomes ``M``. With ``drude=21 22 23`` only the listed types
are Drudes, and the ``DP`` word is not used.

``drude=0`` switches the Drude detection off, including the override by the data
file. A label ``D`` is then read as deuterium (element H). Virtual sites found from
masses are not affected by this setting.

A label ``D`` can also mean deuterium. Real Drude particles sit about 0.01 A from
their core, deuterium about 1 A from its bonded atom, so the loaders warn if the
median distance of the ``D`` atoms to their host is above ``DEUTERIUM_DISTANCE``.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

DRUDE = "D"
VIRTUAL = "M"
SATELLITES = frozenset({DRUDE, VIRTUAL})

# Masses in g/mol: a type lighter than DRUDE_MASS_MAX but heavier than zero is a
# Drude; below VIRTUAL_MASS_MAX it is massless.
DRUDE_MASS_MAX = 0.8
VIRTUAL_MASS_MAX = 1e-8

# Word in the comment of a ``Masses`` line that marks a Drude type (fftool).
DRUDE_TOKEN = "DP"

# Median distance in Angstrom of the ``D`` atoms to their host above which they look
# like deuterium (O-D 0.96 A) rather than Drude particles (about 0.01 A, rarely more
# than 0.1 A).
DEUTERIUM_DISTANCE = 0.3


@dataclass(frozen=True)
class DrudeSetting:
    """How Drude particles are detected.

    Attributes
    ----------
    mode : {"auto", "off", "types"}
        ``auto``: labels and masses, ``off``: no Drudes, ``types``: the atom types
        in ``types`` (and labels ``D``).
    types : frozenset[int]
        Drude atom types, used for ``mode="types"``.
    """

    mode: Literal["auto", "off", "types"] = "auto"
    types: frozenset[int] = frozenset()


def parse_drude(value: int | str | Sequence[int]) -> DrudeSetting:
    """Interpret the ``drude`` argument of the load commands.

    Parameters
    ----------
    value : int | str | Sequence[int]
        ``"auto"`` (or empty) for automatic detection, ``0`` or ``"0"`` to switch it
        off, or atom types as numbers separated by spaces (``"21 22 23"``) or as a
        sequence.

    Returns
    -------
    DrudeSetting
        The parsed setting.

    Raises
    ------
    ValueError
        If the value is neither ``auto``, ``0`` nor a list of positive integers.
    """
    tokens = (
        re.split(r"[\s,]+", value.strip())
        if isinstance(value, str)
        else [str(v) for v in np.atleast_1d(value)]
    )
    tokens = [t for t in tokens if t]
    if not tokens or tokens == ["auto"]:
        return DrudeSetting()
    if tokens == ["0"]:
        return DrudeSetting("off")
    if not all(t.isdigit() and int(t) > 0 for t in tokens):
        raise ValueError(
            f"drude={value!r} is not 'auto', 0 or a list of atom types such as "
            "'21 22 23'"
        )
    return DrudeSetting("types", frozenset(int(t) for t in tokens))


def data_sites(
    type_masses: dict[int, float],
    type_comments: dict[int, list[str]],
    setting: DrudeSetting,
) -> dict[int, str]:
    """Return the atom types that a data file marks as Drudes or virtual sites.

    Parameters
    ----------
    type_masses : dict[int, float]
        Mass of each atom type in g/mol (data file ``Masses``).
    type_comments : dict[int, list[str]]
        Words of the ``Masses`` comment of each atom type.
    setting : DrudeSetting
        Drude detection setting; ``off`` marks nothing.

    Returns
    -------
    dict[int, str]
        ``"D"`` or ``"M"`` for each marked atom type; other types are left out.
    """
    if setting.mode == "off":
        return {}
    types = np.array(sorted(type_masses), dtype=np.int64)
    masses = np.array([type_masses[t] for t in types.tolist()])
    sites = mass_sites(types, masses, setting)
    marked = {int(t): str(s) for t, s in zip(types, sites) if s}
    if setting.mode == "auto":
        for atom_type, words in type_comments.items():
            if DRUDE_TOKEN in words and atom_type not in marked:
                marked[atom_type] = DRUDE
    return marked


def satellite_mask(elements: Sequence[str]) -> NDArray[np.bool_]:
    """Return which atoms are Drude particles or virtual sites."""
    return np.isin(np.asarray(elements), [DRUDE, VIRTUAL])


def label_sites(
    labels: Sequence[str], types: NDArray[np.int64], setting: DrudeSetting
) -> list[str]:
    """Apply the Drude rules to element labels given by the user or the file.

    Parameters
    ----------
    labels : Sequence[str]
        Element label of each atom (dump ``element`` column or ``elements=``).
    types : NDArray[np.int64]
        LAMMPS atom type of each atom.
    setting : DrudeSetting
        Drude detection setting.

    Returns
    -------
    list[str]
        The labels, with ``D`` for Drude particles and ``M`` for virtual sites.
        With ``drude=0`` a label ``D`` becomes ``H`` (deuterium).
    """
    # The rules depend only on the label and the type, so they are applied to each
    # distinct label once.
    unique, inverse = np.unique(np.asarray(labels), return_inverse=True)
    symbols = [u.upper() if u.upper() in SATELLITES else str(u) for u in unique]
    if setting.mode == "off":
        symbols = ["H" if symbol == DRUDE else symbol for symbol in symbols]
    elements = np.array(symbols, dtype=object)[inverse]
    if setting.mode == "types":
        elements[np.isin(types, list(setting.types))] = DRUDE
    return elements.tolist()


def mass_sites(
    types: NDArray[np.int64], masses: NDArray[np.float64], setting: DrudeSetting
) -> NDArray[np.str_]:
    """Find Drude particles and virtual sites from atom masses.

    Parameters
    ----------
    types : NDArray[np.int64]
        LAMMPS atom type of each atom.
    masses : NDArray[np.float64]
        Mass of each atom in g/mol; NaN where unknown.
    setting : DrudeSetting
        Drude detection setting.

    Returns
    -------
    NDArray[np.str_]
        ``"D"``, ``"M"`` or an empty string for each atom.
    """
    sites = np.full(len(types), "", dtype="U1")
    sites[masses < VIRTUAL_MASS_MAX] = VIRTUAL
    if setting.mode == "auto":
        sites[(masses > VIRTUAL_MASS_MAX) & (masses < DRUDE_MASS_MAX)] = DRUDE
    elif setting.mode == "types":
        sites[np.isin(types, list(setting.types))] = DRUDE
    return sites


def satellite_hosts(
    bonds: NDArray[np.int64], elements: Sequence[str]
) -> NDArray[np.int64]:
    """Return the bonds of the Drude particles and virtual sites, satellite first.

    Parameters
    ----------
    bonds : NDArray[np.int64]
        Bonded atom pairs, shape (n_bonds, 2).
    elements : Sequence[str]
        Element symbol of each atom, with ``D`` and ``M`` for the satellites.

    Returns
    -------
    NDArray[np.int64]
        Pairs ``(satellite, host)``, shape (n_satellite_bonds, 2).
    """
    satellite = satellite_mask(elements)
    pairs = bonds[satellite[bonds].any(axis=1)]
    satellite_second = ~satellite[pairs[:, 0]]
    pairs[satellite_second] = pairs[satellite_second, ::-1]
    return pairs
