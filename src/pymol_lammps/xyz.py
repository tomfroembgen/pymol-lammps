"""Read multi-frame ``.xyz`` files, plain or extended.

A frame is a count line, a comment line and one ``Element x y z`` row per atom.
Extended xyz stores the cell in the comment line as ``Lattice="ax ay az bx by bz cx cy
cz"`` (box vectors as rows) and optionally ``Origin="x y z"``.
"""

import itertools
import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pymol_lammps.drude import SATELLITES
from pymol_lammps.dump import Box, open_text
from pymol_lammps.elements import ELEMENT_MASSES

_LATTICE = re.compile(r'Lattice\s*=\s*"([^"]*)"', re.IGNORECASE)
_ORIGIN = re.compile(r'Origin\s*=\s*"([^"]*)"', re.IGNORECASE)


@dataclass(frozen=True)
class XyzFrame:
    """One xyz frame.

    Attributes
    ----------
    positions : NDArray[np.float64]
        Cartesian positions in Angstrom, shape (n_atoms, 3).
    elements : list[str]
        Normalised element symbols, one per atom; ``D`` and ``M`` mark Drude
        particles and virtual sites.
    box : Box | None
        Cell from the ``Lattice`` tag of the comment line, or None if absent.
    """

    positions: NDArray[np.float64]
    elements: list[str]
    box: Box | None


def normalise_element(label: str) -> str:
    """Convert an xyz atom label such as ``"c1"`` or ``"CL"`` to an element symbol.

    Digits and other trailing characters are dropped, then the capitalisation is
    fixed. A two-letter label that is no element falls back to its first letter
    (``"HW"`` becomes ``"H"``). The labels ``D`` (Drude particle) and ``M`` (virtual
    site of a water model) are not elements and are kept as ``"D"`` and ``"M"``,
    independent of case; this applies to the bare letter only (``"MG"`` is still
    magnesium and ``"MW"`` is unknown). Digits are dropped first, so ``"D1"`` is a
    Drude.

    Parameters
    ----------
    label : str
        Atom label from the first column of an xyz row.

    Returns
    -------
    str
        Element symbol.

    Raises
    ------
    ValueError
        If the label does not start with a known element.
    """
    letters = re.match(r"[A-Za-z]+", label)
    word = letters.group().capitalize() if letters else ""
    if word.upper() in SATELLITES:
        return word.upper()
    for candidate in (word[:2], word[:1]):
        if candidate in ELEMENT_MASSES:
            return candidate
    raise ValueError(f"unknown element label {label!r} in xyz file")


def _parse_box(comment: str, path: Path, index: int) -> Box | None:
    """Return the box from the comment line of an extended xyz frame, if any."""
    match = _LATTICE.search(comment)
    if match is None:
        return None
    try:
        lattice = np.array(match.group(1).split(), dtype=float)
        origin_match = _ORIGIN.search(comment)
        origin = (
            np.array(origin_match.group(1).split(), dtype=float)
            if origin_match
            else np.zeros(3)
        )
        return Box(origin=origin.reshape(3), matrix=lattice.reshape(3, 3).T)
    except ValueError:
        raise ValueError(
            f"{path}: frame {index} has a malformed Lattice or Origin tag; Lattice "
            "needs 9 numbers and Origin 3"
        ) from None


def read_frames(
    filename: str | Path, start: int = 1, stop: int = -1, interval: int = 1
) -> Iterator[XyzFrame]:
    """Iterate over the frames of an xyz file.

    Frame numbering follows ``pymol_lammps.dump.read_frames``: the first frame is 1
    and ``stop`` is inclusive.

    Parameters
    ----------
    filename : str | Path
        Xyz file, optionally gzip-compressed (``.gz``).
    start : int, optional
        First frame to read.
    stop : int, optional
        Last frame to read, or -1 for all frames.
    interval : int, optional
        Read every ``interval``-th frame.

    Yields
    ------
    XyzFrame
        Positions, element symbols and the cell, if the file gives one.

    Raises
    ------
    ValueError
        If a frame header, row or element label cannot be parsed, or the atom
        count changes between frames.
    """
    path = Path(filename)
    n_first: int | None = None
    with open_text(path) as fh:
        for index in itertools.count(1):
            if stop != -1 and index > stop:
                return
            count_line = fh.readline()
            if not count_line.strip():
                return
            try:
                n_atoms = int(count_line)
            except ValueError:
                raise ValueError(
                    f"{path}: expected an atom count for frame {index}, got "
                    f"{count_line.strip()!r}"
                ) from None
            if n_atoms == 0:
                raise ValueError(f"{path}: frame {index} has no atoms")
            comment = fh.readline()
            rows = list(itertools.islice(fh, n_atoms))
            if len(rows) < n_atoms:
                # Typical for a file that is still being written.
                warnings.warn(
                    f"{path}: frame {index} is incomplete and was skipped",
                    stacklevel=2,
                )
                return
            if index < start or (index - start) % interval:
                continue
            if n_first is None:
                n_first = n_atoms
            elif n_atoms != n_first:
                raise ValueError(
                    f"{path}: frame {index} has {n_atoms} atoms but the first has "
                    f"{n_first}; trajectories with a changing set of atoms are "
                    "not supported"
                )
            fields = [row.split() for row in rows]
            try:
                positions = np.array([f[1:4] for f in fields], dtype=float)
            except ValueError:
                raise ValueError(
                    f"{path}: frame {index} has rows that are not 'Element x y z'"
                ) from None
            if positions.shape != (n_atoms, 3):
                raise ValueError(
                    f"{path}: frame {index} has rows that are not 'Element x y z'"
                )
            yield XyzFrame(
                positions,
                [normalise_element(f[0]) for f in fields],
                _parse_box(comment, path, index),
            )
