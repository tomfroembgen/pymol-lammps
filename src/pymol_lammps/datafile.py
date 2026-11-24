"""Read the topology parts of a LAMMPS data file.

Only what helps visualisation is read: atom types, molecule IDs, charges, masses,
type names and bonds. Coordinates come from the dump. Names are taken from type
labels and from the comments that tools such as fftool write after each line
(``Masses``: ``1 12.011 # CT``; ``Atoms``: ``... # C1 c4mim``).
"""

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pymol_lammps.dump import open_text

# Header count that gives the number of lines in each section that is read.
_SECTION_COUNTS = {
    "Atom Type Labels": "atom types",
    "Masses": "atom types",
    "Atoms": "atoms",
    "Bonds": "bonds",
}

# Leading columns of the Atoms section for each atom style (LAMMPS read_data manual).
_STYLE_COLUMNS = {
    "atomic": ("id", "type"),
    "charge": ("id", "type", "q"),
    "molecular": ("id", "mol", "type"),
    "bond": ("id", "mol", "type"),
    "angle": ("id", "mol", "type"),
    "full": ("id", "mol", "type", "q"),
}

# Atom style from the number of columns, for files without an "Atoms # style"
# comment. Each style may carry three image flags at the end. 6 and 9 columns are
# missing on purpose: they fit both "molecular" and "charge".
_STYLE_BY_N_COLUMNS = {5: "atomic", 8: "atomic", 7: "full", 10: "full"}


@dataclass(frozen=True)
class DataFile:
    """Topology from a LAMMPS data file, with atoms sorted by atom ID.

    Attributes
    ----------
    ids : NDArray[np.int64]
        Atom IDs in ascending order.
    types : NDArray[np.int64]
        Numeric atom type of each atom.
    mol_ids : NDArray[np.int64] | None
        Molecule ID of each atom, None for atom styles without molecules.
    charges : NDArray[np.float64] | None
        Charge of each atom, None for atom styles without charges.
    atom_names : list[str]
        Atom name from the comment of each ``Atoms`` line, empty if absent.
    residue_names : list[str]
        Residue (species) name from the comment of each ``Atoms`` line, empty if
        absent.
    type_masses : dict[int, float]
        Mass of each atom type.
    type_names : dict[int, str]
        Name of each atom type, from type labels or the first word of the
        ``Masses`` comment.
    type_comments : dict[int, list[str]]
        All words of the ``Masses`` comment of each atom type (fftool marks Drude
        types with ``DP``, as in ``# NA DP``); empty lists where there is none.
    bonds : NDArray[np.int64]
        Bonded atom ID pairs, shape (n_bonds, 2).
    """

    ids: NDArray[np.int64]
    types: NDArray[np.int64]
    mol_ids: NDArray[np.int64] | None
    charges: NDArray[np.float64] | None
    atom_names: list[str]
    residue_names: list[str]
    type_masses: dict[int, float]
    type_names: dict[int, str]
    type_comments: dict[int, list[str]]
    bonds: NDArray[np.int64]


@dataclass(frozen=True)
class _Line:
    """One data-file line split into its values and its comment tokens."""

    values: list[str]
    comment: list[str]

    @classmethod
    def parse(cls, raw: str) -> "_Line":
        content, _, comment = raw.partition("#")
        return cls(content.split(), comment.split())


def _is_number(token: str) -> bool:
    """Return True if a token parses as a number."""
    try:
        float(token)
    except ValueError:
        return False
    return True


def _read_sections(path: Path) -> tuple[dict[str, list[_Line]], dict[str, str]]:
    """Split a data file into the sections this module reads.

    Returns
    -------
    tuple[dict[str, list[_Line]], dict[str, str]]
        Lines of each section, and the comment after each section keyword (the
        ``Atoms`` comment names the atom style).

    Raises
    ------
    ValueError
        If a section is present but the header lacks its count.
    """
    counts: dict[str, int] = {}
    sections: dict[str, list[_Line]] = {}
    keyword_comments: dict[str, str] = {}
    with open_text(path) as fh:
        lines = iter(fh)
        next(lines)  # the first line is a free-form title
        in_header = True
        for raw in lines:
            line = _Line.parse(raw)
            keyword = " ".join(line.values)
            if line.values and not _is_number(line.values[0]):
                # The header ends with the first section keyword.
                in_header = False
            if keyword in _SECTION_COUNTS:
                count_key = _SECTION_COUNTS[keyword]
                if count_key not in counts:
                    raise ValueError(
                        f"{path}: section '{keyword}' found, but the header does not "
                        f"give the number of {count_key}"
                    )
                body = itertools.islice(filter(str.strip, lines), counts[count_key])
                sections[keyword] = [_Line.parse(row) for row in body]
                keyword_comments[keyword] = " ".join(line.comment)
            elif in_header and len(line.values) >= 2 and line.values[0].isdigit():
                # Header counts such as "1200 atoms" or "8 atom types". Box lines
                # ("0.0 40.0 xlo xhi") start with a float and are skipped.
                counts[" ".join(line.values[1:])] = int(line.values[0])
    return sections, keyword_comments


def _atom_style(path: Path, comment: str, n_columns: int) -> str:
    """Return the atom style from the Atoms comment or the number of columns.

    Raises
    ------
    ValueError
        If the style is not supported or cannot be inferred.
    """
    style = comment.split()[0] if comment else _STYLE_BY_N_COLUMNS.get(n_columns, "")
    if style in _STYLE_COLUMNS:
        return style
    if comment:
        raise ValueError(
            f"{path}: atom style '{style}' is not supported; supported styles are "
            f"{', '.join(_STYLE_COLUMNS)}"
        )
    raise ValueError(
        f"{path}: cannot infer the atom style from {n_columns} columns in the Atoms "
        "section; name it in the section header, e.g. 'Atoms # molecular' or "
        "'Atoms # charge'"
    )


def read_data(filename: str | Path) -> DataFile:
    """Read the topology from a LAMMPS data file.

    Parameters
    ----------
    filename : str | Path
        Data file as written by ``write_data``, fftool, moltemplate or similar,
        optionally gzip-compressed (``.gz``).

    Returns
    -------
    DataFile
        Atoms sorted by atom ID, with type information and bonds.

    Raises
    ------
    ValueError
        If the file has no Atoms section, or its atom style is not supported.
    """
    path = Path(filename)
    sections, keyword_comments = _read_sections(path)
    if "Atoms" not in sections:
        raise ValueError(f"{path}: no Atoms section found")

    labels = {
        line.values[1]: int(line.values[0])
        for line in sections.get("Atom Type Labels", [])
    }

    def type_id(token: str) -> int:
        # Files written with type labels may use the label instead of the number.
        return int(token) if token.isdigit() else labels[token]

    type_masses: dict[int, float] = {}
    type_comments: dict[int, list[str]] = {}
    type_names = {number: label for label, number in labels.items()}
    for line in sections.get("Masses", []):
        number = type_id(line.values[0])
        type_masses[number] = float(line.values[1])
        type_comments[number] = line.comment
        if line.comment and number not in type_names:
            type_names[number] = line.comment[0]

    atoms = sections["Atoms"]
    style = _atom_style(path, keyword_comments["Atoms"], len(atoms[0].values))
    columns = _STYLE_COLUMNS[style]
    table = np.array([line.values[: len(columns)] for line in atoms])
    ids = table[:, columns.index("id")].astype(np.int64)
    order = np.argsort(ids)
    table = table[order]
    comments = [atoms[i].comment for i in order]

    return DataFile(
        ids=ids[order],
        types=np.array([type_id(t) for t in table[:, columns.index("type")]]),
        mol_ids=(
            table[:, columns.index("mol")].astype(np.int64)
            if "mol" in columns
            else None
        ),
        charges=table[:, columns.index("q")].astype(float) if "q" in columns else None,
        atom_names=[c[0] if c else "" for c in comments],
        residue_names=[c[1] if len(c) > 1 else "" for c in comments],
        type_masses=type_masses,
        type_names=type_names,
        type_comments=type_comments,
        bonds=np.array(
            [line.values[2:4] for line in sections.get("Bonds", [])], dtype=np.int64
        ).reshape(-1, 2),
    )
