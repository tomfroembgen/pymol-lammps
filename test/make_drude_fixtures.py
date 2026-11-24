"""Cut a small Drude test system out of a CL&Pol data file (run by hand, not by pytest).

The fixtures in ``test/data/`` were written with::

    python test/make_drude_fixtures.py path/to/data-p.lmp test/data

The source is the fftool data file of an ionic liquid (c4c1im+ / tf2N-) with SWM4-NDP
water. The script keeps three cations, three anions and three waters together with
their Drude particles and M sites, renumbers atoms and bonds, and writes

- ``drude.data``: LAMMPS data file (all 34 atom types, atom style full, bonds only),
- ``drude.lammpstrj``: three frames, columns ``id type element xu yu zu``,
- ``drude.xyz``: the same frames as extended xyz with wrapped atoms and a Lattice.

The molecules are moved into a 30 A cube. Molecule 1 sits on the lower x face, so
its first core atom is at x = 0.1 A and its Drude at x = -0.2 A in frame 1; the
xyz file wraps that Drude to x = 29.8 A. In every frame each Drude lies 0.1 to
0.4 A from its core, in a random direction, and the molecules drift by 0.2 A in x.
"""

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

BOX = 30.0
N_FRAMES = 3
# Molecule IDs of the source file to keep: cations, anions, waters.
KEEP_MOLECULES = (1, 2, 3, 253, 254, 255, 505, 506, 507)
# Element of each of the 34 atom types of the source file, in type order.
TYPE_ELEMENTS = [
    *"N C C C H C H H C C H C C F S N O O H M".split(),
    *["D"] * 14,
]
DRUDE_TYPES = range(21, 35)


def read_source(path: Path) -> tuple[list[str], list[list[str]], list[list[str]]]:
    """Return the Masses lines, Atoms rows and Bonds rows of the source data file.

    An Atoms row is ``[id, mol, type, q, x, y, z, comment]``; a Bonds row is
    ``[id, type, atom1, atom2]``.
    """
    lines = path.read_text().splitlines()

    def section(keyword: str) -> list[str]:
        start = lines.index(keyword) + 2
        stop = start
        while stop < len(lines) and lines[stop].strip():
            stop += 1
        return lines[start:stop]

    atoms = []
    for row in section("Atoms"):
        values, _, comment = row.partition("#")
        atoms.append([*values.split(), comment.strip()])
    bonds = [row.split("#")[0].split() for row in section("Bonds")]
    return section("Masses"), atoms, bonds


def place_molecules(
    atoms: list[list[str]],
) -> tuple[list[list[str]], NDArray[np.float64]]:
    """Move the kept molecules into the cube.

    Returns
    -------
    tuple[list[list[str]], NDArray[np.float64]]
        The kept Atoms rows in molecule order, and their unwrapped positions
        (whole molecules, shape (n_atoms, 3)).
    """
    kept = [a for a in atoms if int(a[1]) in KEEP_MOLECULES]
    kept.sort(key=lambda a: (KEEP_MOLECULES.index(int(a[1])), int(a[0])))
    positions = np.array([a[4:7] for a in kept], dtype=float)
    molecule = np.array([KEEP_MOLECULES.index(int(a[1])) for a in kept])
    is_core = ~np.isin([int(a[2]) for a in kept], DRUDE_TYPES)
    # Molecule centres on a 3 x 3 grid; the first one on the x = 0 face.
    for k in range(len(KEEP_MOLECULES)):
        centre = [0.0 if k == 0 else 5.0 + 10.0 * (k % 3), 5.0 + 10.0 * (k // 3), 15.0]
        selected = molecule == k
        positions[selected] += centre - positions[selected & is_core].mean(axis=0)
    # The first core atom of molecule 1 goes to x = 0.1 A.
    positions[molecule == 0, 0] += 0.1 - positions[0, 0]
    return kept, positions


def displace_drudes(
    positions: NDArray[np.float64],
    drude_core: list[tuple[int, int]],
    rng: np.random.Generator,
    across_boundary: bool,
) -> NDArray[np.float64]:
    """Place every Drude 0.1 to 0.4 A from its core, in a random direction.

    With ``across_boundary``, the Drude of the first atom is put 0.3 A below it in x,
    which is across the lower box face.
    """
    out = positions.copy()
    for drude, core in drude_core:
        direction = rng.normal(size=3)
        out[drude] = out[core] + rng.uniform(0.1, 0.4) * direction / np.linalg.norm(
            direction
        )
    if across_boundary:
        first_drude = next(d for d, c in drude_core if c == 0)
        out[first_drude] = out[0] + np.array([-0.3, 0.0, 0.0])
    return out


def main(source: Path, target: Path) -> None:
    """Write ``drude.data``, ``drude.lammpstrj`` and ``drude.xyz`` into ``target``."""
    masses, atoms, bonds = read_source(source)
    kept, positions = place_molecules(atoms)
    new_id = {a[0]: i + 1 for i, a in enumerate(kept)}
    kept_bonds = [b for b in bonds if b[2] in new_id and b[3] in new_id]
    types = np.array([int(a[2]) for a in kept])
    # The source lists Drude bonds as (core, Drude).
    drude_core = [
        (new_id[b[3]] - 1, new_id[b[2]] - 1)
        for b in kept_bonds
        if int(types[new_id[b[3]] - 1]) in DRUDE_TYPES
    ]

    rng = np.random.default_rng(20261124)
    frames = [
        displace_drudes(positions + [0.2 * n, 0.0, 0.0], drude_core, rng, n == 0)
        for n in range(N_FRAMES)
    ]

    data = [
        "Drude test system cut from a CL&Pol (fftool) data file",
        "",
        f"{len(kept)} atoms",
        f"{len(kept_bonds)} bonds",
        "34 atom types",
        "28 bond types",
        f"0.0 {BOX} xlo xhi",
        f"0.0 {BOX} ylo yhi",
        f"0.0 {BOX} zlo zhi",
        "",
        "Masses",
        "",
        *masses,
        "",
        "Atoms # full",
        "",
    ]
    data += [
        f"{i + 1} {KEEP_MOLECULES.index(int(a[1])) + 1} {a[2]} {a[3]} "
        + " ".join(f"{c:.6f}" for c in frames[0][i])
        + f" # {a[7]}"
        for i, a in enumerate(kept)
    ]
    data += ["", "Bonds", ""]
    data += [
        f"{i + 1} {b[1]} {new_id[b[2]]} {new_id[b[3]]}"
        for i, b in enumerate(kept_bonds)
    ]
    (target / "drude.data").write_text("\n".join(data) + "\n")

    elements = [TYPE_ELEMENTS[t - 1] for t in types]
    dump: list[str] = []
    extended: list[str] = []
    lattice = f"{BOX} 0 0 0 {BOX} 0 0 0 {BOX}"
    for n, coordinates in enumerate(frames):
        dump += [
            "ITEM: TIMESTEP",
            str(1000 * n),
            "ITEM: NUMBER OF ATOMS",
            str(len(kept)),
            "ITEM: BOX BOUNDS pp pp pp",
            *[f"0.0 {BOX}"] * 3,
            "ITEM: ATOMS id type element xu yu zu",
        ]
        dump += [
            f"{i + 1} {types[i]} {elements[i]} " + " ".join(f"{c:.4f}" for c in row)
            for i, row in enumerate(coordinates)
        ]
        extended += [str(len(kept)), f'Lattice="{lattice}"']
        extended += [
            f"{elements[i]} " + " ".join(f"{c:.4f}" for c in row)
            for i, row in enumerate(coordinates % BOX)
        ]
    (target / "drude.lammpstrj").write_text("\n".join(dump) + "\n")
    (target / "drude.xyz").write_text("\n".join(extended) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
