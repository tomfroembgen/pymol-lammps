"""Shared fixtures: small dump and data files written on the fly.

The water system has two molecules (atom IDs 1-3 and 4-6, O H H) in a 10 A box
centred at the origin, as written by fftool.
"""

import gzip
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from pymol_lammps.datafile import DataFile, read_data

WATER_DATA = """\
created by fftool

6 atoms
4 bonds
2 atom types
1 bond types
-5.000000 5.000000 xlo xhi
-5.000000 5.000000 ylo yhi
-5.000000 5.000000 zlo zhi

Masses

   1   15.999  # OW
   2    1.008  # HW

Bond Coeffs

   1   450.000000     1.000000  # OW-HW

Atoms

      2       1    2   0.417000  9.600000e-01  0.000000e+00  0.000000e+00  # HW1 SPC
      1       1    1  -0.834000  0.000000e+00  0.000000e+00  0.000000e+00  # OW SPC
      3       1    2   0.417000  0.000000e+00  9.600000e-01  0.000000e+00  # HW2 SPC
      4       2    1  -0.834000  3.000000e+00  0.000000e+00  0.000000e+00  # OW SPC
      5       2    2   0.417000  3.960000e+00  0.000000e+00  0.000000e+00  # HW1 SPC
      6       2    2   0.417000  3.000000e+00  9.600000e-01  0.000000e+00  # HW2 SPC

Velocities

      1  0.001 0.002 0.003

Bonds

      1    1       1       2  # OW-HW
      2    1       1       3  # OW-HW
      3    1       4       5  # OW-HW
      4    1       4       6  # OW-HW
"""

# Water positions by atom ID (index 0 is atom 1).
WATER_POSITIONS = [
    [0.0, 0.0, 0.0],
    [0.96, 0.0, 0.0],
    [0.0, 0.96, 0.0],
    [3.0, 0.0, 0.0],
    [3.96, 0.0, 0.0],
    [3.0, 0.96, 0.0],
]

ORTHO_BOUNDS = ((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0))


def frame_text(
    columns: str,
    rows: Sequence[Sequence[object]],
    timestep: int = 0,
    bounds: Sequence[Sequence[float]] = ORTHO_BOUNDS,
    box_header: str = "pp pp pp",
) -> str:
    """Return one dump frame as text, in the layout LAMMPS writes."""
    lines = [
        "ITEM: TIMESTEP",
        str(timestep),
        "ITEM: NUMBER OF ATOMS",
        str(len(rows)),
        f"ITEM: BOX BOUNDS {box_header}",
        *(" ".join(map(str, b)) for b in bounds),
        f"ITEM: ATOMS {columns}",
        *(" ".join(map(str, row)) for row in rows),
    ]
    return "\n".join(lines) + "\n"


def water_frame(
    timestep: int = 0, shift: float = 0.0, order: Sequence[int] = range(6)
) -> str:
    """Return a water frame with all atoms shifted along x, rows in given order."""
    rows = []
    for i in order:
        x, y, z = WATER_POSITIONS[i]
        rows.append([i + 1, 1 + i // 3, 1 if i % 3 == 0 else 2, x + shift, y, z])
    return frame_text("id mol type x y z", rows, timestep)


@pytest.fixture
def write_file(tmp_path: Path) -> Callable[[str, str], Path]:
    """Return a function that writes text to a file in tmp_path (gzip for .gz)."""

    def write(name: str, text: str) -> Path:
        path = tmp_path / name
        if name.endswith(".gz"):
            with gzip.open(path, "wt") as fh:
                fh.write(text)
        else:
            path.write_text(text)
        return path

    return write


@pytest.fixture
def water_data(write_file: Callable[[str, str], Path]) -> Path:
    """Data file of the water system in fftool layout."""
    return write_file("water.data", WATER_DATA)


@pytest.fixture
def water_dump(write_file: Callable[[str, str], Path]) -> Path:
    """Three-frame water dump; atoms move +1 A in x per frame, rows shuffled."""
    text = (
        water_frame(0, 0.0, [0, 1, 2, 3, 4, 5])
        + water_frame(100, 1.0, [5, 3, 1, 0, 4, 2])
        + water_frame(200, 2.0, [2, 4, 0, 1, 3, 5])
    )
    return write_file("water.lammpstrj", text)


@pytest.fixture(scope="module")
def drude_data() -> DataFile:
    """Data file of the Drude fixtures (see ``drude_helpers.py``)."""
    return read_data(Path(__file__).parent / "data" / "drude.data")


@pytest.fixture(scope="module")
def drude_pairs(drude_data: DataFile) -> NDArray[np.int64]:
    """Core-Drude and O-M bonds of the Drude fixtures as atom indices, Drude first."""
    bonds = drude_data.bonds - 1
    satellite = drude_data.types >= 20
    return bonds[satellite[bonds].any(axis=1)]
