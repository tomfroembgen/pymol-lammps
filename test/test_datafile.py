"""Tests for the data-file reader."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from pymol_lammps.datafile import read_data

WriteFile = Callable[[str, str], Path]

HEADER = """\
LAMMPS data file

2 atoms
2 atom types
0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 12.011
2 1.008

"""


def test_fftool_layout(water_data: Path) -> None:
    data = read_data(water_data)

    np.testing.assert_array_equal(data.ids, np.arange(1, 7))
    np.testing.assert_array_equal(data.types, [1, 2, 2, 1, 2, 2])
    assert data.mol_ids is not None
    np.testing.assert_array_equal(data.mol_ids, [1, 1, 1, 2, 2, 2])
    assert data.charges is not None
    np.testing.assert_allclose(data.charges[:3], [-0.834, 0.417, 0.417])
    assert data.atom_names == ["OW", "HW1", "HW2"] * 2
    assert data.residue_names == ["SPC"] * 6
    assert data.type_masses == {1: 15.999, 2: 1.008}
    assert data.type_names == {1: "OW", 2: "HW"}
    np.testing.assert_array_equal(data.bonds, [[1, 2], [1, 3], [4, 5], [4, 6]])


def test_write_data_layout_with_type_labels(write_file: WriteFile) -> None:
    text = """\
LAMMPS data file via write_data

2 atoms
2 atom types
0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Atom Type Labels

1 c3
2 hc

Masses

c3 12.011
hc 1.008

Atoms # full

2 1 hc 0.1 1.0 0.0 0.0 0 0 0
1 1 c3 -0.1 0.0 0.0 0.0 0 0 0
"""
    data = read_data(write_file("t.data", text))

    np.testing.assert_array_equal(data.types, [1, 2])
    assert data.type_names == {1: "c3", 2: "hc"}
    assert data.type_masses == {1: 12.011, 2: 1.008}
    assert data.atom_names == ["", ""]
    assert data.bonds.shape == (0, 2)


def test_atomic_style_is_inferred(write_file: WriteFile) -> None:
    text = HEADER + "Atoms\n\n1 1 0.0 0.0 0.0\n2 2 1.0 0.0 0.0\n"

    data = read_data(write_file("t.data", text))

    assert data.mol_ids is None
    assert data.charges is None
    np.testing.assert_array_equal(data.types, [1, 2])


def test_named_charge_style(write_file: WriteFile) -> None:
    text = HEADER + "Atoms # charge\n\n1 1 -0.5 0.0 0.0 0.0\n2 2 0.5 1.0 0.0 0.0\n"

    data = read_data(write_file("t.data", text))

    assert data.mol_ids is None
    assert data.charges is not None
    np.testing.assert_allclose(data.charges, [-0.5, 0.5])


def test_ambiguous_style_raises(write_file: WriteFile) -> None:
    text = HEADER + "Atoms\n\n1 1 1 0.0 0.0 0.0\n2 1 2 1.0 0.0 0.0\n"

    with pytest.raises(ValueError, match="Atoms # molecular"):
        read_data(write_file("t.data", text))


def test_unsupported_style_raises(write_file: WriteFile) -> None:
    text = HEADER + "Atoms # sphere\n\n1 1 1.0 1.0 0.0 0.0 0.0\n2 2 1.0 1.0 1.0 0 0\n"

    with pytest.raises(ValueError, match="'sphere' is not supported"):
        read_data(write_file("t.data", text))


def test_missing_atoms_section_raises(write_file: WriteFile) -> None:
    with pytest.raises(ValueError, match="no Atoms section"):
        read_data(write_file("t.data", HEADER))


def test_section_without_header_count_raises(write_file: WriteFile) -> None:
    text = HEADER + "Atoms # atomic\n\n1 1 0 0 0\n2 2 0 0 0\n\nBonds\n\n1 1 1 2\n"

    with pytest.raises(ValueError, match="number of bonds"):
        read_data(write_file("t.data", text))
