# pymol-lammps

Load LAMMPS trajectories into PyMOL.

PyMOL cannot read LAMMPS dump files (`.lammpstrj`) out of the box. This add-on
teaches it to, and optionally reads the LAMMPS data file of the system to get
elements, atom names, molecules, charges and bonds right.

```
load_lammps traj.lammpstrj, data=system.data
```

- Each dump frame becomes one PyMOL state; play them with the movie controls.
- The simulation box of every frame is drawn as a grey object, also for NPT runs.
- Atoms are matched by LAMMPS atom ID in every frame, so dumps written without
  `dump_modify sort id` load correctly.
- Pure Python (NumPy only, which ships with PyMOL): no compiling, and it keeps
  working when PyMOL is updated.

Tested with open-source PyMOL 3.1 (Homebrew and conda-forge) on macOS. It should
work with the Schrödinger (incentive) PyMOL, but that is untested.


## Installation

1. Get the code (any location works):

   ```
   git clone https://github.com/tomfroembgen/pymol-lammps.git ~/Software/pymol-lammps
   ```

2. Add these lines to the end of `~/.pymolrc` (create the file if it does not
   exist; on Windows it is `pymolrc.pml` in your home folder). Adjust the path
   if you cloned elsewhere:

   ```
   python
   import os, sys
   _path = os.path.expanduser("~/Software/pymol-lammps/src")
   if os.path.isdir(_path):
       sys.path.insert(0, _path)
       from pymol_lammps.loader import register
       register()
   python end
   ```

3. Restart PyMOL and check:

   ```
   help load_lammps
   ```

To update later: `git pull` in the repository folder and restart PyMOL.


## Quick start

```
load_lammps traj.lammpstrj, data=system.data      # recommended
load traj.lammpstrj                               # also works, without topology
```

Then, for example:

```
hide everything, (all)    # the parentheses keep the box visible
show sticks
util.cbag
```

A plain `hide everything` also hides the box object; bring it back with
`show cgo, *_box`.


## The data file: why `data=` matters

A dump file contains only numbers per frame: atom ID, type number and
coordinates. It does not say which element type 3 is, which atoms are bonded, or
what an atom is called. That information is in the LAMMPS **data file**, the file
your input script reads with `read_data` (written by fftool, moltemplate, VMD
TopoTools or LAMMPS' own `write_data`). It plays the role of a topology file, like
a PSF next to a DCD in VMD or a `.top` next to an `.xtc` in GROMACS.

With `data=system.data`, the add-on reads from it:

- `Masses` → the element of each type (12.011 → C)
- `Atoms` → molecule ID (`resi`), charge, and atom/ion names if the file has them
  as comments (fftool writes `# NA c2c1im+`)
- `Bonds` → the bonds of the force field

The data file is matched to the dump by atom ID; its coordinates are ignored, so
the starting data file of the run is fine. The dump may contain only a subset of
the atoms (a dump of one group).

Without a data file, elements come from an `element` or `mass` column in the dump
if present (otherwise all atoms are shown as carbon, with a warning), and bonds
are guessed from distances in the first frame. For the ionic liquid
[C2C1im][NTf2], for example, distance guessing finds 400 "bonds" instead of the
real 330.


## Preparing the LAMMPS run

The dump needs the columns `id`, `type` and positions. Recommended:

```
dump TRAJ all custom 1000 traj.lammpstrj id mol type q xu yu zu
```

- Use **unwrapped** coordinates (`xu yu zu`), or wrapped ones together with image
  flags (`x y z ix iy iz`). With purely wrapped coordinates, molecules cut by the
  box boundary show bonds stretching across the box.
- Scaled coordinates (`xs ys zs`, the default of `dump atom`) and triclinic boxes
  work too.
- Gzipped dumps (`traj.lammpstrj.gz`) are read directly.


## Command reference

```
load_lammps filename [, data [, object [, start [, stop [, interval
            [, elements [, box ]]]]]]]
```

| Argument | Meaning |
|---|---|
| `filename` | text dump (`dump atom` or `dump custom`), optionally `.gz` |
| `data` | LAMMPS data file of the same system (optional) |
| `object` | object name; default: file name without extension. An existing object of that name is replaced |
| `start`, `stop`, `interval` | first frame (1-based), last frame (inclusive, -1 = all), stride |
| `elements` | element per atom type in type order, e.g. `C H O N`; overrides everything else |
| `box` | 1 (default) draws the box of every frame as object `<object>_box`; 0 to skip |

Examples:

```
load_lammps traj.lammpstrj, data=system.data, interval=10    # every 10th frame
load_lammps traj.lammpstrj, elements=C H O N                 # no data file
load_lammps traj.lammpstrj, data=system.data, object=run1, box=0
```

From a Python script: `from pymol_lammps.loader import load_lammps`.

### What is stored where

| PyMOL property | Source, first available wins |
|---|---|
| `elem` | `elements=` > dump `element` column > mass (dump `mass` column, data-file `Masses`) > `C` with a warning |
| `name` | data-file comment after the atom (fftool) > type label or `Masses` comment > type number |
| `resn` | data-file comment after the atom (fftool writes the species, e.g. `c2c1im+`) |
| `resi` | dump `mol` column > data-file molecule ID |
| `partial_charge` | dump `q` column > data file |
| `ID` | LAMMPS atom ID |
| `numeric_type`, `text_type` | LAMMPS atom type |
| bonds | data file; without one, guessed from distances in the first frame |

Elements are matched to masses within 0.01 g/mol, so united-atom sites
(CH2 = 14.027) and Drude particles are not mistaken for N or H; give them with
`elements=`.

Useful selections:

```
color red, text_type 3          # LAMMPS atom type 3
select cation, resn c2c1im*     # fftool residue names (see Troubleshooting)
select mol12, resi 12           # LAMMPS molecule 12
select first100, ID 1-100       # LAMMPS atom IDs
```


## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `load_lammps` is an unknown command | the `.pymolrc` block did not run; check the path in it and restart PyMOL |
| warning "no element known for atom types ..." | give `data=` or `elements=` |
| long bonds across the box | the dump has wrapped coordinates; dump `xu yu zu` or add `ix iy iz` |
| `resn c2c1im+` selects nothing | `+` separates list items in PyMOL selections; use `resn c2c1im*` |
| "Selector-Error" or "cannot find object" | the object name is a PyMOL keyword (e.g. `elem`, `name`); pass `object=...` |
| PyMOL runs out of memory | all frames are held in memory; use `interval=` or `start=`/`stop=` |
| the box disappeared | `hide everything` (or `hide everything, all`) hides all objects; use `hide everything, (all)` and restore with `show cgo, *_box` |
| `show cell` draws the box in the wrong place | PyMOL draws the cell from the origin; use the `<object>_box` object instead |


## Limitations

- Not supported: a changing number of atoms during the run, type labels in the
  dump (`dump_modify types labels`), general triclinic boxes
  (`dump_modify triclinic/general yes`), binary, NetCDF or H5MD dumps.
- Coordinates are taken as Angstrom. With `units lj`, distance-based bond guessing
  is meaningless; give a data file.
- Bonds are fixed for the whole trajectory (from the data file or from the first
  frame); reactive simulations are not reflected.


## Development

```
conda env create -f environment.yaml
conda activate pymol-lammps
pre-commit install
pytest
```

The parsers (`dump.py`, `datafile.py`, `topology.py`) need only NumPy; the PyMOL
commands are in `loader.py`. The tests in `test/test_loader.py` need PyMOL and
are skipped where it is not importable.

License: MIT (see `LICENSE.md`). Author: Tom Frömbgen.
