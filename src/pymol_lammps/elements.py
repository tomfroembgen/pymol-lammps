"""Infer chemical elements from atomic masses.

LAMMPS stores masses, not elements. Matching a mass against standard atomic weights
recovers the element for all-atom force fields. The tolerance is tight on purpose:
united-atom sites such as CH2 (14.027) lie close to real elements (N, 14.007) and must
stay unresolved instead of being assigned the wrong element.
"""

import numpy as np

# Conventional standard atomic weights (IUPAC, abridged to 3 decimals), H to Ba plus
# the heavier elements common in simulations.
ELEMENT_MASSES: dict[str, float] = {
    "H": 1.008, "He": 4.003, "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.95,
    "Ru": 101.07, "Rh": 102.906, "Pd": 106.42, "Ag": 107.868, "Cd": 112.414,
    "In": 114.818, "Sn": 118.710, "Sb": 121.760, "Te": 127.60, "I": 126.904,
    "Xe": 131.293, "Cs": 132.905, "Ba": 137.327, "La": 138.905, "Gd": 157.25,
    "W": 183.84, "Pt": 195.084, "Au": 196.967, "Hg": 200.592, "Pb": 207.2,
    "Bi": 208.980, "U": 238.029,
}  # fmt: skip

# Absolute tolerance in g/mol. Covers the usual rounding of force-field masses
# (14.0067 vs 14.007, 16.00 vs 15.999) but not united-atom sites.
MASS_TOLERANCE = 0.01

_SYMBOLS = np.array(list(ELEMENT_MASSES))
_MASSES = np.array(list(ELEMENT_MASSES.values()))


def element_from_mass(mass: float) -> str:
    """Return the element whose standard atomic weight matches a mass.

    Parameters
    ----------
    mass : float
        Atomic mass in g/mol.

    Returns
    -------
    str
        Element symbol, or an empty string if no element lies within
        ``MASS_TOLERANCE``.
    """
    deviation = np.abs(_MASSES - mass)
    best = int(np.argmin(deviation))
    return str(_SYMBOLS[best]) if deviation[best] <= MASS_TOLERANCE else ""
