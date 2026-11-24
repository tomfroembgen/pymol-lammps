"""Shared constants, fixtures and file helpers for the Drude tests.

The fixtures ``test/data/drude.*`` hold three c4c1im+ cations, three tf2N- anions and
three SWM4-NDP waters with their Drudes and M sites (see ``make_drude_fixtures.py``):
78 Drudes and 3 M sites among 210 atoms, in a 30 A cube. The first core atom (ID 1)
sits at x = 0.1 A and its Drude 0.3 A below it in x in frame 1, across the lower x
face. Atoms are numbered by molecule, so atom i of the fixtures has index i - 1 in
the arrays of the tests.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pymol_lammps.dump import Box
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


def relabel_drudes(label: str) -> str:
    """Return the fixture dump with the element label of every Drude replaced."""
    out = []
    for line in (DATA / "drude.lammpstrj").read_text().splitlines():
        fields = line.split()
        if len(fields) == 6 and fields[2] == "D":
            line = " ".join([*fields[:2], label, *fields[3:]])
        out.append(line)
    return "\n".join(out) + "\n"


def frame_elements_and_positions(n: int = 0) -> tuple[list[str], NDArray[np.float64]]:
    frames = list(read_xyz(DATA / "drude.xyz"))
    return frames[n].elements, frames[n].positions
