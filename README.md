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
column `mol` or the data file). If neither is available, a molecule is a group of
atoms that are connected by bonds, with the bonds guessed from the first frame (see
[The data file](#the-data-file-why-data-matters)).

```
lammps_wrap traj, 0       # back to the coordinates of the dump (e.g. to follow diffusion)
lammps_wrap traj          # wrap again
load_lammps traj.lammpstrj, data=system.data, wrap=0      # load without wrapping
```

`lammps_wrap` re-reads the coordinates from the dump file and keeps
representations and colours. A molecule that crosses the boundary jumps to the
opposite side between two frames; this is expected for wrapped coordinates.

If the dump has wrapped coordinates without image flags, molecules cut by the
boundary are first reassembled by walking along the bonds. The bonds come from the
data file or, without one, are guessed from the first frame. Guessed bonds are
reliable only if the elements are known (see
[The data file](#the-data-file-why-data-matters)).


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
  Such molecules are detected, and a warning gives their number. Their atoms are
  wrapped one by one into the cell in the first frame and then follow their own
  path, and the bonds to the other side of the cell are removed from the display
  (see [Periodic bonds](#periodic-bonds)).
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


## Drude particles

Polarizable force fields such as CL&Pol represent the electronic polarization of an
atom by a light, charged *Drude particle* that is tied to its atom by a stiff spring.
Some water models (for example SWM4-NDP) add a massless *M site* (a virtual site that
carries the charge of the oxygen). We call the atom that a Drude particle or an M
site belongs to its *host*: the polarizable atom of a Drude particle, and the oxygen
of an M site. Drude particles and M sites are loaded as ordinary atoms, so that the
coordinates of the dump can be used unchanged.

**Detection.** A Drude particle gets the element `D` and an M site the element `M`.
The rules below are tried in this order, and the first one that applies to an atom
type decides:

1. `drude=0` switches the detection of Drude particles off. A label `D` then means
   deuterium, and only the M sites are recognised (by rules 4 and 5).
2. With `drude=21 22 23`, the atom types in the list are Drude particles, whatever
   the `element` column of the dump or `elements=` says. Masses and the data file
   then mark no further Drude types, but a label `D` still does (rule 4).
3. With a data file, every atom type that the data file marks as a Drude type is a
   Drude particle, and every massless type (mass below 1e-8 g/mol) an M site,
   whatever the `element` column or `elements=` says. The data file marks a Drude
   type by a mass between 0 and 0.8 g/mol or by the word `DP` in the comment of its
   `Masses` line (fftool writes `# NA DP`). A dump written with another label for
   the Drude particles therefore still loads correctly. With `drude=21 22 23`, only
   the M sites are taken from the data file.
4. The element label `D` or `M` (case-insensitive) in the `element` column of the
   dump, in `elements=`, or in an xyz file.
5. If the elements come from the masses, a type with a mass between 0 and
   0.8 g/mol is a Drude particle (their masses are about 0.4 g/mol, and hydrogen is
   1.008 g/mol), and a type with a mass below 1e-8 g/mol is an M site.

A dump without the Drude particles has none and loads as before.

Because the label `D` is also the symbol of deuterium, the loaders check the
distance of the `D` atoms to their host in the first frame. A Drude particle sits
about 0.01 A from its host, whereas a deuterium atom sits about 0.96 A from its
oxygen. If the median distance is larger than 0.3 A, the loaders print a warning
that the atoms look like deuterium and suggest loading again with `drude=0`. The
warning is not printed with `quiet=1`.

**Selections and bonds.** `select drudes, elem D` and `select msites, elem M` find
the particles, and the LAMMPS atom type stays available as `numeric_type`. PyMOL
treats the symbol `D` as deuterium, which would make `hide (hydro)` hide the Drude
particles too; the loader removes them from `hydro`. Each Drude particle and M site
is bonded to its host as the data file says. Without a data file, for xyz files, and
for particles that have no bond in the data file (an M site held in place by
`fix rigid`, for example), the host is the nearest atom that is no Drude particle or
M site (the nearest oxygen for an M site). Drude particles and M sites travel with
their molecule when it is wrapped. In addition, every Drude particle and M site is
moved in every frame to the periodic image next to its host, also with `wrap=0`. A
particle is therefore never shown on the other side of the box from its host, even
if the dump has wrapped coordinates and the molecule is not reassembled, for example
with `wrap=0` or with a data file that has no molecule IDs.

**Display style.** A Drude particle lies very close to its host. In the first frame
of a simulation of an ionic liquid on graphene with 3036 Drude particles, the median
distance between a Drude particle and its host is 0.0095 A and the largest 0.085 A.
In the usual stick style the particle is therefore hidden inside its host. Already
on loading, Drude particles and M sites get the colours and radii that are described
below, but they keep the representations of all other atoms, so that loading does
not change what is shown. `drude_style` then gives the particles their own look:

```
drude_style                     # all objects
drude_style traj                # one object
drude_style traj, dashed=0      # thin solid bonds instead of dashed ones
```

For every Drude particle or M site in the selection, the command hides the sticks,
shows a sphere, and colours it. Only particles that are shown, or whose host is
shown, are changed, so that molecules hidden before stay hidden. Drude particles
become dark-gray spheres (`drude_gray`, RGB 0.3 0.3 0.3) with a radius of 0.25 A,
and M sites light-gray spheres (`vsite_gray`, 0.72 0.72 0.72) with the same
radius. The radius is the radius as drawn: it does not depend on `sphere_scale`. It
is clearly smaller than an ion sphere in the style `tf` (about 0.5 A, which is a van
der Waals radius of 1.5 to 1.8 A times `sphere_scale` 0.3) and larger than a stick
(0.2 A), so a Drude particle remains visible on its host. The two colours are not
used for any element by PyMOL or by `tf`. The selection must contain atoms: the box
object, the dash object or a name that does not exist give an error.

The bond of a particle to its host is drawn as a thin dashed line. PyMOL cannot dash
sticks, so the command creates a distance object `<object>_drude` that measures only
the bonded pairs of the Drude particles and M sites, with its labels hidden. Bonds
longer than 5 A get no dash; this limit lets PyMOL search only nearby atoms, and the
bond of a particle that is placed next to its host is far shorter. The object
contains one set of dashes for each state, so the dashes follow the trajectory, and
the time to create or update it grows with the number of frames. The command
remembers the dashed particles by atom ID, so that `lammps_wrap` can update the
dashes when it changes the coordinates, also after the selection used for
`drude_style` was deleted. The dashes have a length of 0.08 A, a gap of 0.07 A and a
radius of 0.045 A. Calling `drude_style` again on another part of the object adds
its dashes to the existing ones, and `dashed=0` removes the dashes of the particles
it styles. Loading the object again deletes the dash object.

With `dashed=0`, the bonds are instead sticks with a radius of 0.05 A. The command
shows sticks on the particles only and never changes the representations of their
hosts. Because PyMOL draws a stick bond only between two atoms that both show
sticks, the thin bond appears where the host shows sticks; in a view of lines or of
spheres the particles get no visible bond. We chose this behaviour because showing
sticks on the hosts as well would also draw the bonds between the hosts, which
turns a view of spheres into one of spheres and sticks.

For the ionic liquid on graphene (8237 atoms, 3036 Drude particles, 10 frames), we
measured 0.29 s for `drude_style` with dashes and 0.22 s with solid sticks, and
0.19 s for a following `lammps_wrap` with dashes, of which 0.06 s is the update of
the dashes (Apple M3 Pro). Ray-tracing one frame took the same time, within 10 %,
with the dashes, with the solid sticks and with neither. The box object is not
touched.

Run `drude_style` after the command that sets the general style (such as `tf`),
because that command shows sticks on all atoms. To change the look afterwards:

```
hide dashes, traj_drude              # hide the dashes (show them again with: show dashes, traj_drude)
delete traj_drude                    # remove them
set dash_radius, 0.03, traj_drude    # thinner, also dash_gap, dash_length, dash_color
set_color drude_gray, [0.2, 0.2, 0.5]
color drude_gray, elem D             # another colour
alter elem D, vdw=0.2                # another radius; then: rebuild
```

The sizes and colours are also constants at the top of `pymol_lammps/style.py`.
Dashes only become visible if a Drude particle is displaced by more than about
0.3 A from its host; for smaller displacements the dashed line is hidden in the
spheres and sticks, and the solid fallback behaves in the same way.


## Periodic bonds

If a molecule is bonded through the periodic boundary to its own periodic image (a
graphene sheet, an infinite chain or a crystal; we call it a *periodic network*),
its bonds include bonds between atoms and the periodic images of their neighbours.
This holds for the bonds of a data file and for the bonds that the loaders guess
without one, because the guess uses the minimum-image distance. PyMOL draws these
bonds as sticks through the whole box. After loading, bonds longer than half of the
smallest height of the box are therefore removed from the display, and the summary
line says how many (150 for the five graphene layers of the ionic-liquid example,
whose dump has unwrapped coordinates, with or without the data file). No real bond
is that long, and the bonds of Drude particles and M sites are never removed. The
same rules apply to `load_lammps` with and without a data file and to `load_xyz`.
With `wrap=0` and a file of wrapped coordinates, all bonds are kept, because
molecules cut by the boundary are then shown as they are in the file. `lammps_wrap`
finds the periodic bonds again for the new coordinates, so switching between the
two views removes and restores them.

PyMOL stores bonds per object, not per frame, so a bond removed in one frame is
missing in all frames. The loader therefore keeps the set of removed bonds the same
in all frames. A periodic network cannot be reassembled into one piece if its
coordinates are wrapped, because it is connected to its own images. For a dump with
wrapped coordinates, `load_lammps` therefore wraps the atoms of such a network one
by one in the first frame, with a warning, in the same way as `load_xyz`; a Drude
particle is wrapped together with its host. In every later frame, each atom of the
network is placed at the periodic image closest to its position in the previous
frame, and the whole network is shifted by box vectors so that its centre stays in
the box. The network is therefore cut at the same bonds in every frame. This needs
atoms that move by less than half a box between two loaded frames, which holds for
any usual dump interval.

The warning counts the molecules that form networks. Without a data file and without
a `mol` column, every connected network counts on its own, so for the ionic-liquid
example the warning reports five networks (the five graphene layers), whereas the
data file puts all layers into one molecule. The display is the same in both cases.

Only the centre of the network is kept in the box, not each of its atoms. A network
that drifts through the box is shown as a continuous piece that follows its own
path, and its atoms can lie outside the box by up to about half a box length when
its centre is near a face of the box. A graphene sheet that is fixed in space, as in
the ionic-liquid example, stays where it is. With unwrapped coordinates
(`xu yu zu`), the network is already in one piece and is shifted as a whole.


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
if present (otherwise all atoms are shown as carbon, with a warning). The bonds are
then guessed from the first frame in the same way as for xyz files: two atoms are
bonded if their distance, measured with the minimum-image convention, is smaller
than the sum of their covalent radii plus 0.4 A. Because of the minimum image,
molecules cut by the box boundary and periodic networks are recognised as with a
data file. Element labels such as `CL` or `OW` are read as Cl and O, and a label
that does not start with an element symbol is bonded like carbon, with a warning.
If the dump has a `mol` column, a guessed bond between two molecules is dropped,
because the molecule IDs say that the molecules are separate. Without a `mol`
column, a molecule is a group of atoms connected by the guessed bonds.

The guess is only as good as the elements. With an `element` column, the guessed
bonds were exactly the bonds of the data file for the three systems we tested: an
ionic liquid (9945 bonds), the same ionic liquid with water (9954 bonds) and an
ionic liquid on five graphene layers (9216 bonds, of which 150 pass through the box
boundary). Without element information every atom is treated as carbon, and the
guess finds 12325 bonds instead of 9954 for the ionic liquid with water.


## Preparing the LAMMPS run

The dump needs the columns `id`, positions, and `type` or `element`. Recommended:

```
dump TRAJ all custom 1000 traj.lammpstrj id mol type q xu yu zu
```

- The `type` column may be replaced by `element` (`dump_modify TRAJ element O H`),
  for example `id element xu yu zu`. With a data file (`data=`), the atom types are
  then read from the data file by atom ID, so the dump may cover a subset of its
  atoms. Without a data file, each distinct element becomes one atom type (in
  alphabetical order, starting at 1), and the bonds are guessed from distances.
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
            [, elements [, box [, wrap [, quiet [, drude ]]]]]]]]]]
lammps_wrap object [, wrap [, quiet ]]
drude_style [ selection [, dashed [, quiet ]]]
load_xyz filename [, object [, start [, stop [, interval [, cell [, box
         [, wrap [, quiet [, drude ]]]]]]]]]
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
| `quiet` | 1 suppresses the summary line; 0 (default) prints it |
| `drude` | `auto` (default) finds Drude particles by data file, label and mass; `0` switches this off (label `D` is then deuterium); a list of atom types such as `21 22 23` names the Drude types (see [Drude particles](#drude-particles)) |
| `selection` | `drude_style`: atoms whose Drude particles and M sites are styled; default `all` |
| `dashed` | `drude_style`: 1 (default) draws the bonds of the particles as dashes, 0 as thin sticks |

Examples:

```
load_lammps traj.lammpstrj, data=system.data, interval=10    # every 10th frame
load_lammps traj.lammpstrj, elements=C H O N                 # no data file
load_lammps traj.lammpstrj, data=system.data, object=run1, box=0
```

For `load_xyz`, the arguments `object`, `start`, `stop`, `interval`, `box`, `wrap`
and `quiet` mean the same as above, `drude` accepts only `auto` and `0`, and `cell`
is the periodic cell as described in
[Loading xyz trajectories](#loading-xyz-trajectories). `lammps_wrap` accepts
objects from both loaders.

From a Python script:
`from pymol_lammps.loader import lammps_wrap, load_lammps, load_xyz`.

### What is stored where

| PyMOL property | Source, first available wins |
|---|---|
| `elem` | Drude types of `drude=21 22 23` (`D`) > Drude and M-site types of the data file (`D`, `M`, unless `drude=0`) > `elements=` > dump `element` column > mass (dump `mass` column, data-file `Masses`) > `C` with a warning |
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
| long bonds across the box | bonds between periodic images are removed after loading (see [Periodic bonds](#periodic-bonds)); other long bonds mean that molecules were not reassembled, for example with `wrap=0` and a wrapped dump. Dump `xu yu zu` (or add `ix iy iz`), or wrap with `lammps_wrap object` |
| warning "molecules were not wrapped" | the data file has no molecule IDs (atom style `atomic` or `charge`): dump the `mol` column |
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
- PyMOL's mass table (in its `chempy` module) has no entry for the symbols `D` and
  `M`. The PyMOL functions that need masses, such as `centerofmass` and
  `util.compute_mass`, therefore stop with a `KeyError` on a selection that contains
  Drude particles or M sites. Leave them out of the selection, for example
  `centerofmass mol12 and not elem D+M`.


## Development

```
conda env create -f environment.yaml
conda activate pymol-lammps
pre-commit install
pytest
```

The modules `dump.py`, `datafile.py`, `topology.py`, `xyz.py`, `bonds.py`,
`drude.py` and `wrapping.py` need only NumPy; the PyMOL commands are in `loader.py`
and `style.py`. The tests in `test_loader.py`, `test_load_xyz.py`,
`test_drude_loading.py`, `test_periodic_bonds.py` and `test_style.py` need PyMOL and
are skipped where it is not importable; all other tests need only NumPy.

License: MIT (see `LICENSE.md`). Author: Tom Frömbgen.
