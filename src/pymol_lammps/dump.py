"""Read LAMMPS text dump files (``dump atom`` and ``dump custom``).

Each frame is returned with its rows sorted by atom ID. LAMMPS writes atoms in
processor order, which changes between frames unless ``dump_modify sort id`` is set,
so frames cannot be stacked without this reordering.
"""

import gzip
import io
import itertools
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Position column sets, from most to least preferred for visualisation. Unwrapped
# coordinates keep molecules whole across the periodic boundary.
_POSITION_COLUMNS = (
    (("xu", "yu", "zu"), False),
    (("x", "y", "z"), False),
    (("xsu", "ysu", "zsu"), True),
    (("xs", "ys", "zs"), True),
)
_IMAGE_COLUMNS = ("ix", "iy", "iz")
# Columns that are parsed to numbers, and are the only ones read after the first frame.
# All other columns (type, element, mol, q, mass, ...) are only needed to build the
# topology, and are kept as the text written in the file.
_NUMERIC_COLUMNS = frozenset(
    {"id", *_IMAGE_COLUMNS, *(name for names, _ in _POSITION_COLUMNS for name in names)}
)


@dataclass(frozen=True)
class Box:
    """Simulation box of one frame.

    Attributes
    ----------
    origin : NDArray[np.float64]
        Lower corner (xlo, ylo, zlo) in Angstrom.
    matrix : NDArray[np.float64]
        3x3 matrix whose columns are the box vectors a, b, c in Angstrom, in the
        LAMMPS convention (a along x, b in the xy plane).
    """

    origin: NDArray[np.float64]
    matrix: NDArray[np.float64]

    @classmethod
    def from_bounds(cls, bounds: NDArray[np.float64]) -> "Box":
        """Build a box from the ``ITEM: BOX BOUNDS`` block of a dump.

        Parameters
        ----------
        bounds : NDArray[np.float64]
            Shape (3, 2) for orthogonal boxes, or (3, 3) for triclinic boxes, where
            the third column holds the tilt factors xy, xz, yz.

        Returns
        -------
        Box
            The box.
        """
        xy, xz, yz = bounds[:, 2] if bounds.shape[1] == 3 else (0.0, 0.0, 0.0)
        # Triclinic dumps store the bounding box of the tilted cell, not xlo/xhi.
        # Conversion from the LAMMPS manual, section "Triclinic (non-orthogonal)
        # simulation boxes".
        xlo = bounds[0, 0] - min(0.0, xy, xz, xy + xz)
        xhi = bounds[0, 1] - max(0.0, xy, xz, xy + xz)
        ylo = bounds[1, 0] - min(0.0, yz)
        yhi = bounds[1, 1] - max(0.0, yz)
        zlo, zhi = bounds[2, 0], bounds[2, 1]
        matrix = np.array(
            [[xhi - xlo, xy, xz], [0.0, yhi - ylo, yz], [0.0, 0.0, zhi - zlo]]
        )
        return cls(origin=np.array([xlo, ylo, zlo]), matrix=matrix)

    @classmethod
    def from_cell_parameters(
        cls,
        a: float,
        b: float | None = None,
        c: float | None = None,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
    ) -> "Box":
        """Build a box at the origin from cell lengths and angles.

        Parameters
        ----------
        a : float
            Length of the first box vector in Angstrom.
        b : float | None, optional
            Length of the second box vector; defaults to ``a``.
        c : float | None, optional
            Length of the third box vector; defaults to ``a``.
        alpha, beta, gamma : float, optional
            Angles between b and c, a and c, a and b in degree.

        Returns
        -------
        Box
            The box in the LAMMPS convention (a along x, b in the xy plane).

        Raises
        ------
        ValueError
            If a length is not positive or the angles cannot form a cell.
        """
        b = a if b is None else b
        c = a if c is None else c
        if min(a, b, c) <= 0:
            raise ValueError(f"cell lengths must be positive, got {a} {b} {c}")
        if not all(0 < angle < 180 for angle in (alpha, beta, gamma)):
            raise ValueError(
                f"cell angles must lie between 0 and 180 degree, got "
                f"{alpha} {beta} {gamma}"
            )
        # Exact zeros for right angles, so that atoms at x = 0 stay at 0.
        cos_a, cos_b, cos_g = (
            0.0 if angle == 90 else np.cos(np.radians(angle))
            for angle in (alpha, beta, gamma)
        )
        sin_g = 1.0 if gamma == 90 else np.sin(np.radians(gamma))
        cx = c * cos_b
        cy = c * (cos_a - cos_b * cos_g) / sin_g
        cz_squared = c**2 - cx**2 - cy**2
        if cz_squared <= 1e-12 * c**2:
            raise ValueError(
                f"the angles {alpha} {beta} {gamma} degree do not describe a "
                "three-dimensional cell"
            )
        matrix = np.array(
            [[a, b * cos_g, cx], [0.0, b * sin_g, cy], [0.0, 0.0, np.sqrt(cz_squared)]]
        )
        return cls(origin=np.zeros(3), matrix=matrix)

    def cell_parameters(self) -> tuple[float, float, float, float, float, float]:
        """Return the cell lengths (Angstrom) and angles (degree).

        Returns
        -------
        tuple[float, float, float, float, float, float]
            a, b, c, alpha, beta, gamma.
        """
        a, b, c = self.matrix.T

        def angle(u: NDArray[np.float64], v: NDArray[np.float64]) -> float:
            cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            return float(np.degrees(np.arccos(cos)))

        length = np.linalg.norm
        return (
            float(length(a)),
            float(length(b)),
            float(length(c)),
            angle(b, c),
            angle(a, c),
            angle(a, b),
        )

    def edges(self) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return the 12 box edges as pairs of Cartesian end points.

        Returns
        -------
        list[tuple[NDArray[np.float64], NDArray[np.float64]]]
            Start and end point of each edge.
        """
        corners = {
            idx: self.origin + self.matrix @ np.array(idx, float)
            for idx in itertools.product((0, 1), repeat=3)
        }
        # Two corners share an edge when their fractional indices differ in one axis.
        return [
            (corners[p], corners[q])
            for p, q in itertools.combinations(corners, 2)
            if sum(i != j for i, j in zip(p, q)) == 1
        ]


def parse_cell(text: str) -> Box:
    """Build a box from a user string of cell parameters.

    Parameters
    ----------
    text : str
        One number (cube), three numbers (a b c, right angles) or six numbers
        (a b c alpha beta gamma), separated by whitespace and/or commas, with
        optional surrounding quotes. Lengths are in Angstrom, angles in degree.
        Commas work from Python only; PyMOL splits command arguments at commas, so
        use spaces in ``load_xyz``.

    Returns
    -------
    Box
        The box, see ``Box.from_cell_parameters``.

    Raises
    ------
    ValueError
        If the string does not hold 1, 3 or 6 numbers, or the cell is invalid.
    """
    try:
        values = [
            float(token) for token in text.strip("\"' ").replace(",", " ").split()
        ]
    except ValueError:
        raise ValueError(f"cell {text!r} must contain only numbers") from None
    if len(values) not in (1, 3, 6):
        raise ValueError(
            f"cell {text!r} has {len(values)} numbers; give 1 (a), 3 (a b c) or "
            "6 (a b c alpha beta gamma)"
        )
    return Box.from_cell_parameters(*values)


@dataclass(frozen=True)
class Frame:
    """One dump frame with its rows sorted by atom ID.

    Attributes
    ----------
    timestep : int
        LAMMPS timestep.
    box : Box
        Simulation box.
    data : pd.DataFrame
        Per-atom columns, sorted by atom ID. ``id`` is integer, positions and image
        flags are floats, and every other column holds the strings of the file. Only
        the first frame of a trajectory holds all dump columns; later frames hold
        the numeric ones.
    """

    timestep: int
    box: Box
    data: pd.DataFrame

    @property
    def columns(self) -> list[str]:
        """Names of the columns held by this frame."""
        return self.data.columns.tolist()

    def column(self, name: str) -> NDArray[Any]:
        """Return one column.

        Parameters
        ----------
        name : str
            Column name, e.g. ``"type"``.

        Returns
        -------
        NDArray[Any]
            Values sorted by atom ID: integers for ``id``, floats for positions and
            image flags, strings for all other columns.
        """
        return self.data[name].to_numpy()

    @property
    def ids(self) -> NDArray[np.int64]:
        """Atom IDs in ascending order."""
        return self.column("id")

    def _position_columns(self) -> tuple[tuple[str, str, str], bool]:
        """Return the preferred position columns and whether they are scaled."""
        return next(
            (names, scaled)
            for names, scaled in _POSITION_COLUMNS
            if all(name in self.columns for name in names)
        )

    @property
    def unwrapped(self) -> bool:
        """True if ``positions()`` gives unwrapped coordinates.

        Unwrapped coordinates keep molecules whole across the periodic boundary.
        """
        names, _ = self._position_columns()
        return names[0].endswith("u") or all(n in self.columns for n in _IMAGE_COLUMNS)

    def positions(self) -> NDArray[np.float64]:
        """Return Cartesian positions in Angstrom, unwrapped where possible.

        Unwrapped columns are used directly. Wrapped coordinates are unwrapped with
        image flags (ix, iy, iz) when the dump contains them.

        Returns
        -------
        NDArray[np.float64]
            Positions, shape (n_atoms, 3), sorted by atom ID.
        """
        names, scaled = self._position_columns()
        xyz = np.column_stack([self.column(name) for name in names])
        wrapped = not names[0].endswith("u")
        if wrapped and all(name in self.columns for name in _IMAGE_COLUMNS):
            images = np.column_stack([self.column(n) for n in _IMAGE_COLUMNS])
            if scaled:
                xyz += images
            else:
                xyz += images @ self.box.matrix.T
        if scaled:
            xyz = self.box.origin + xyz @ self.box.matrix.T
        return xyz


def open_text(path: Path) -> TextIO:
    """Open a dump file as text, transparently decompressing ``.gz`` files."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def _check_columns(columns: list[str], path: Path) -> None:
    """Raise if a dump lacks the columns needed to build a trajectory."""
    missing = "id" not in columns or not {"type", "element"} & set(columns)
    has_positions = any(
        all(name in columns for name in names) for names, _ in _POSITION_COLUMNS
    )
    if missing or not has_positions:
        raise ValueError(
            f"{path}: dump needs the columns id, type (or element) and one set of "
            f"positions (x y z, xu yu zu, xs ys zs or xsu ysu zsu); found {' '.join(columns)}"
        )


def _read_header(
    fh: TextIO, path: Path
) -> tuple[int, int, NDArray[np.float64], list[str]] | None:
    """Read the ITEM blocks before the atom rows of one frame.

    Returns
    -------
    tuple[int, int, NDArray[np.float64], list[str]] | None
        Timestep, number of atoms, box bounds and column names, or None at the end
        of the file.

    Raises
    ------
    ValueError
        If the file contains a box format this reader does not support.
    """
    timestep = n_atoms = -1
    bounds = np.empty((0, 0))
    for line in fh:
        item = line.removeprefix("ITEM:").strip()
        if item == "TIMESTEP":
            timestep = int(fh.readline())
        elif item == "NUMBER OF ATOMS":
            n_atoms = int(fh.readline())
        elif item.startswith("BOX BOUNDS"):
            if "abc" in item.split():
                raise ValueError(
                    f"{path}: general triclinic boxes ('BOX BOUNDS abc origin') are "
                    "not supported; write the dump with the restricted triclinic "
                    "convention (dump_modify triclinic/general no)"
                )
            n_values = 3 if "xy" in item.split() else 2
            bounds = np.array(
                [fh.readline().split()[:n_values] for _ in range(3)], dtype=float
            )
        elif item.startswith("ATOMS"):
            return timestep, n_atoms, bounds, item.split()[1:]
        # Other items (UNITS, TIME, user-defined) are not needed and skipped.
    return None


def _read_table(
    text: str, sep: str, columns: list[str], kept: list[str], dtypes: dict[str, Any]
) -> pd.DataFrame:
    """Read the atom rows with the pandas C engine, see ``_parse_rows``."""
    return pd.read_csv(
        io.StringIO(text),
        sep=sep,
        header=None,
        names=columns,
        usecols=kept,
        dtype=dtypes,
        index_col=False,
        engine="c",
    )


def _parse_rows(rows: list[str], columns: list[str], keep_all: bool) -> pd.DataFrame:
    """Parse the atom rows of one frame with pandas' C parser and sort them by ID.

    Parameters
    ----------
    rows : list[str]
        Lines of the ``ITEM: ATOMS`` block, with line breaks.
    columns : list[str]
        Column names from the ``ITEM: ATOMS`` line.
    keep_all : bool
        If True, parse all columns; otherwise only the numeric ones.

    Returns
    -------
    pd.DataFrame
        The parsed columns, sorted by atom ID.

    Raises
    ------
    ValueError
        If rows have fewer fields than there are columns.
    """
    # Parsing text to numbers is by far the slowest step for long trajectories, and
    # the C engine does it about ten times faster than ``str.split`` plus ``astype``.
    kept = [name for name in columns if keep_all or name in _NUMERIC_COLUMNS]
    dtypes: dict[str, Any] = {
        name: str if name not in _NUMERIC_COLUMNS else float for name in kept
    }
    dtypes["id"] = np.int64
    text = "".join(rows)
    # LAMMPS separates fields by single spaces, which the C engine parses fastest. A
    # user format with padded fields leaves empty fields, which show up as missing
    # values (or as an error for the integer ID); only then is the slower
    # whitespace separator needed.
    try:
        frame = _read_table(text, " ", columns, kept, dtypes)
        padded = bool(frame.isna().to_numpy().any())
    except ValueError:
        padded = True
    if padded:
        frame = _read_table(text, r"\s+", columns, kept, dtypes)
        # Missing values that remain mean rows with too few fields, e.g. a line cut
        # off while LAMMPS writes the dump; a literal "nan" (a blown-up run) is kept.
        if frame.isna().to_numpy().any() and "nan" not in text.lower():
            raise ValueError("atom rows have fewer fields than the ITEM: ATOMS line")
    ids = frame["id"].to_numpy()
    if (ids[1:] < ids[:-1]).any():
        frame = frame.iloc[np.argsort(ids)]
    return frame


def read_frames(
    filename: str | Path, start: int = 1, stop: int = -1, interval: int = 1
) -> Iterator[Frame]:
    """Iterate over the frames of a LAMMPS text dump file.

    Frame numbering follows PyMOL's ``load_traj``: the first frame is 1 and ``stop``
    is inclusive.

    Parameters
    ----------
    filename : str | Path
        Dump file, optionally gzip-compressed (``.gz``).
    start : int, optional
        First frame to read.
    stop : int, optional
        Last frame to read, or -1 for all frames.
    interval : int, optional
        Read every ``interval``-th frame.

    Yields
    ------
    Frame
        Frames with rows sorted by atom ID.

    Raises
    ------
    ValueError
        If the dump lacks the required columns, if rows have fewer fields than
        columns, or if the atom IDs differ between frames (atoms added or removed
        during the run).
    """
    path = Path(filename)
    first_ids: NDArray[np.int64] | None = None
    with open_text(path) as fh:
        for index in itertools.count(1):
            if stop != -1 and index > stop:
                return
            header = _read_header(fh, path)
            if header is None:
                return
            timestep, n_atoms, bounds, columns = header
            rows = list(itertools.islice(fh, n_atoms))
            if len(rows) < n_atoms:
                # Typical for a dump that LAMMPS is still writing.
                warnings.warn(
                    f"{path}: frame {index} (timestep {timestep}) is incomplete and "
                    "was skipped",
                    stacklevel=2,
                )
                return
            if index < start or (index - start) % interval:
                continue
            if first_ids is None:
                _check_columns(columns, path)
            try:
                data = _parse_rows(rows, columns, first_ids is None)
            except ValueError as error:
                raise ValueError(
                    f"{path}: frame {index} (timestep {timestep}): {error}"
                ) from None
            frame = Frame(timestep, Box.from_bounds(bounds), data)
            if first_ids is None:
                first_ids = frame.ids
            elif not np.array_equal(frame.ids, first_ids):
                raise ValueError(
                    f"{path}: atom IDs in frame {index} (timestep {timestep}) differ "
                    "from the first frame; trajectories with a changing set of atoms "
                    "are not supported"
                )
            yield frame
