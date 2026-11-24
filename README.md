# pymol-lammps

Load LAMMPS trajectories into PyMOL.

PyMOL cannot read LAMMPS dump files (`.lammpstrj`) out of the box. This add-on
teaches it to, and optionally reads the LAMMPS data file of the system to get
elements, atom names, molecules, charges and bonds right. A second command,
`load_xyz`, loads multi-frame xyz files such as ab initio MD output, which have
no box and no topology (see [Loading xyz trajectories](#loading-xyz-trajectories)).

```
load_lammps traj.lammpstrj, data=system.data
```

- Each dump frame becomes one PyMOL state; play them with the movie controls.
- The simulation box of every frame, as written in the dump, is drawn as a grey
  object, also for NPT runs.
- Whole molecules are wrapped into the box of each frame (switchable).
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

Playback speed is a PyMOL setting, in frames per second (default 30):

```
set movie_fps, 5      # each frame is shown for 0.2 s
```

Put it in your `~/.pymolrc` to keep it. To play through a long run faster, load
only every n-th frame instead: `load_lammps traj.lammpstrj, interval=10`.


## Wrapping molecules into the box

By default, every molecule is shifted by whole box vectors so that its centre of
geometry lies inside the box of that frame. Molecules stay intact; this also
works for triclinic boxes. Molecules are taken from the LAMMPS molecule IDs (dump
column `mol` or the data file).

```
lammps_wrap traj, 0       # back to the coordinates of the dump (e.g. to follow diffusion)
lammps_wrap traj          # wrap again
load_lammps traj.lammpstrj, data=system.data, wrap=0      # load without wrapping
```

`lammps_wrap` re-reads the coordinates from the dump file and keeps
representations and colours. A molecule that crosses the boundary jumps to the
opposite side between two frames; this is expected for wrapped coordinates.

If the dump has wrapped coordinates without image flags, molecules cut by the
boundary are first reassembled by walking along the bonds. This needs the bonds
from a data file for molecules longer than half the box (peptides, polymers).


## Loading xyz trajectories

Ab initio molecular dynamics (AIMD) programs usually write the trajectory as a
multi-frame `.xyz` file. Such a file holds only element symbols and coordinates,
so PyMOL knows neither the periodic cell nor which atoms form a molecule. The
command `load_xyz` takes the cell from you (or from the file) and works out the
molecules itself. It is meant mainly for NVT simulations, in which the cell stays
constant, but a cell that changes between frames (NpT) is handled as well.

```
load_xyz aimd.xyz, cell=12.4                      # cubic cell, 12.4 A
load_xyz aimd.xyz, cell=12.4 12.4 15              # orthorhombic cell
load_xyz aimd.xyz, cell=12.4 12.4 15 90 90 120    # a b c alpha beta gamma
load_xyz aimd.xyz, cell=12.4, interval=10, wrap=0
```

**The cell.** The `cell` argument is a string in Angstrom and degree. One number
gives a cube, three numbers give the edge lengths a, b and c with right angles,
and six numbers give a, b, c, alpha, beta and gamma. Numbers are separated by
spaces. Do not use commas inside the PyMOL command, because PyMOL splits the
arguments of a command at commas; the Python function accepts both. If `cell` is not given, the loader looks for the `Lattice="..."`
tag in the comment line of each frame of an extended xyz file and uses that cell
for the frame; this is why a cell that changes between frames works. If a file has a
`Lattice` and you also give `cell`, the cell argument is used and a warning
is printed once. If there is no cell at all, no box is drawn and nothing is
wrapped, and `wrap` has no effect.

**Bonds and molecules.** With a cell, the loader guesses the bonds from the first
frame. Two atoms are bonded if their distance, measured with the minimum-image
convention in the cell, is smaller than the sum of their covalent radii plus
0.4 A. Because of the minimum image, bonds across the cell boundary are found.
Alkali and alkaline-earth metals (Li, Na, K, Rb, Cs, Be, Mg, Ca, Sr, Ba) are
treated as ions and receive no bonds, so that a cation does not become part of one
molecule together with the solvent molecules around it. Every other metal is bonded
by the covalent radii like any other element, which can join a metal complex or a
metal surface with its ligands into one molecule. The molecules are the groups of
atoms connected by these bonds. They are stored
as `resi` (numbered from 1 in the order of their first atom), so `select mol3,
resi 3` selects the third molecule. The atom name is the element symbol and the
residue name is empty.

**Wrapping.** xyz files carry no image flags, so a molecule that was cut by the
cell boundary when the file was written is first put together again, by walking
along its bonds. Then every molecule is shifted by whole cell vectors so that its
centre of geometry lies inside the cell, as in `load_lammps`. With `wrap=0` the
coordinates stay as in the file. The command `lammps_wrap object [, 0|1]` also
works on objects from `load_xyz` and switches between the two views after loading.

**Limitations.**

- A molecule that is bonded through the cell boundary to its own periodic image,
  such as an infinite chain, a slab or a crystal, cannot be put together again.
  Such molecules are detected, their atoms are wrapped one by one into the cell,
  and a warning gives their number. The bonds to the other side of the cell are
  then drawn across the box.
- In a strongly skewed cell, rounding to the nearest image can fail for bonds. The
  bond search tests the neighbouring cells as well, but wrapping does not. A warning
  is printed if the smallest height of the cell (the distance between two opposite
  faces) is smaller than twice the longest bond limit.
- The bonds and molecules are determined once, from the first frame, and kept
  for all frames. Reactions during the run, such as proton transfer, are therefore
  not reflected, and a molecule that is already broken in frame 1 stays broken.
- Without a cell, PyMOL guesses the bonds itself, and `resi` is numbered from
  plain distances without the minimum image.
- The cell is drawn as the `<object>_box` object from the cell vectors, and wrapping
  uses the same vectors. PyMOL's crystal symmetry (`show cell`, `get_symmetry`) can
  only store lengths and angles. A `Lattice` is used as written in the file and is
  not rotated into the LAMMPS orientation, so for a cell whose first vector does not
  point along x the symmetry setting describes the same cell shape but not its
  orientation. Use the `_box` object for display.
- The elements must be given in the first column of the xyz file; labels such as
  `C1` or `HW` are reduced to the element symbol.


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

- Write the molecule ID (`mol`) or give a data file, so that molecules can be
  wrapped as a whole.
- Prefer **unwrapped** coordinates (`xu yu zu`), or wrapped ones together with
  image flags (`x y z ix iy iz`). Purely wrapped coordinates work when molecules
  can be reassembled (see above), but `wrap=0` then shows molecules cut by the
  box boundary.
- Scaled coordinates (`xs ys zs`, the default of `dump atom`) and triclinic boxes
  work too.
- Gzipped dumps (`traj.lammpstrj.gz`) are read directly.


## Command reference

```
load_lammps filename [, data [, object [, start [, stop [, interval
            [, elements [, box [, wrap ]]]]]]]]
lammps_wrap object [, wrap ]
load_xyz filename [, object [, start [, stop [, interval [, cell [, box
         [, wrap ]]]]]]]
```

| Argument | Meaning |
|---|---|
| `filename` | text dump (`dump atom` or `dump custom`), optionally `.gz` |
| `data` | LAMMPS data file of the same system (optional) |
| `object` | object name; default: file name without extension. An existing object of that name is replaced |
| `start`, `stop`, `interval` | first frame (1-based), last frame (inclusive, -1 = all), stride |
| `elements` | element per atom type in type order, e.g. `C H O N`; overrides everything else |
| `box` | 1 (default) draws the box of every frame as object `<object>_box`; 0 to skip |
| `wrap` | 1 (default) wraps whole molecules into the box; 0 keeps the coordinates of the dump |

Examples:

```
load_lammps traj.lammpstrj, data=system.data, interval=10    # every 10th frame
load_lammps traj.lammpstrj, elements=C H O N                 # no data file
load_lammps traj.lammpstrj, data=system.data, object=run1, box=0
```

For `load_xyz`, the arguments `object`, `start`, `stop`, `interval`, `box` and
`wrap` mean the same as above, and `cell` is the periodic cell as described in
[Loading xyz trajectories](#loading-xyz-trajectories). `lammps_wrap` accepts
objects from both loaders.

From a Python script:
`from pymol_lammps.loader import lammps_wrap, load_lammps, load_xyz`.

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
| `load_lammps` or `load_xyz` is an unknown command | the `.pymolrc` block did not run; check the path in it and restart PyMOL |
| warning "no element known for atom types ..." | give `data=` or `elements=` |
| long bonds across the box | molecules could not be reassembled: with `wrap=0` and a wrapped dump, or a molecule longer than half the box without a data file. Dump `xu yu zu` (or add `ix iy iz`), or give `data=` |
| warning "molecules were not wrapped" | no molecule IDs: dump the `mol` column or give `data=` |
| molecules jump between frames | they cross the boundary while wrapping is on; `lammps_wrap object, 0` shows continuous paths |
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

The parsers (`dump.py`, `datafile.py`, `topology.py`, `xyz.py`, `bonds.py`) need only NumPy; the PyMOL
commands are in `loader.py`. The tests in `test/test_loader.py` need PyMOL and
are skipped where it is not importable.

License: MIT (see `LICENSE.md`). Author: Tom Frömbgen.
