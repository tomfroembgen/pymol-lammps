# Adapted from drawgridbox.py by Cunliang Geng (2016), published on the PyMOL Wiki
# (https://pymolwiki.org/index.php/DrawGridBox) under the BSD-2-Clause license, and
# itself based on DrawBoundingBox by Jason Vertrees. License of the original:
#
# Copyright (c) 2016 Cunliang Geng
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this
#    list of conditions and the following disclaimer in the documentation and/or
#    other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Grid box around a selection, drawn as a CGO object.

``drawgridbox`` encloses the extent of a selection in a box divided into
``nx * ny * nz`` cells and draws all cell edges as lines. Unlike the original
script, each grid line is drawn once instead of once per adjacent cell, and the
object gets the next free name ``gridbox<NN>`` instead of a random one.
"""

import itertools
from typing import Any

import numpy as np
import pymol
from numpy.typing import NDArray
from pymol import cgo, cmd


def grid_segments(
    lower: NDArray[np.float64], upper: NDArray[np.float64], cells: tuple[int, int, int]
) -> NDArray[np.float64]:
    """Return the lines of a box divided into a regular grid of cells.

    Every line runs through the whole box along one axis, at one grid point of the
    two other axes, so each cell edge is part of exactly one line.

    Parameters
    ----------
    lower : NDArray[np.float64]
        Lower corner of the box, shape (3,).
    upper : NDArray[np.float64]
        Upper corner of the box, shape (3,).
    cells : tuple[int, int, int]
        Number of cells along x, y and z, each at least 1.

    Returns
    -------
    NDArray[np.float64]
        Start and end point of each line, shape (n_lines, 2, 3) with
        ``n_lines = sum over axes of the product of (cells + 1) of the two others``.
    """
    ticks = [np.linspace(lower[k], upper[k], cells[k] + 1) for k in range(3)]
    segments = []
    for axis in range(3):
        others = [k for k in range(3) if k != axis]
        for a, b in itertools.product(ticks[others[0]], ticks[others[1]]):
            start = np.empty(3)
            start[others] = a, b
            end = start.copy()
            start[axis], end[axis] = lower[axis], upper[axis]
            segments.append((start, end))
    return np.array(segments)


def drawgridbox(
    selection: str = "(all)",
    nx: int | str = 1,
    ny: int | str = 1,
    nz: int | str = 1,
    padding: float | str = 0.0,
    lw: float | str = 2.0,
    r: float | str = 0.0,
    g: float | str = 0.0,
    b: float | str = 0.0,
    pxlo: float | str = 0.0,
    pylo: float | str = 0.0,
    pzlo: float | str = 0.0,
    pxhi: float | str = 0.0,
    pyhi: float | str = 0.0,
    pzhi: float | str = 0.0,
    quiet: int | str = 0,
    *,
    _self: Any = cmd,
) -> str:
    """Draw a grid box around a selection.

    The box spans the extent of the selection (atom centres, current state),
    widened by ``padding`` on all sides and by ``pxlo`` ... ``pzhi`` on single
    sides. It is divided into ``nx * ny * nz`` cells and loaded as a new CGO
    object.

    PyMOL usage::

        drawgridbox [ selection [, nx [, ny [, nz [, padding [, lw [, r [, g [, b
                    [, pxlo [, pylo [, pzlo [, pxhi [, pyhi [, pzhi [, quiet ]]]]]]]]]]]]]]]

    Examples::

        drawgridbox
        drawgridbox resn LIG, nx=4, ny=4, nz=4, padding=2, r=1

    Parameters
    ----------
    selection : str, optional
        Atoms to enclose. Default all.
    nx, ny, nz : int | str, optional
        Number of cells along x, y and z. Default 1 (a plain box).
    padding : float | str, optional
        Margin in Angstrom added on all six sides. Default 0.
    lw : float | str, optional
        Line width in pixels. Default 2.
    r, g, b : float | str, optional
        Line colour, each component in [0, 1]. Default black.
    pxlo, pylo, pzlo, pxhi, pyhi, pzhi : float | str, optional
        Extra margin in Angstrom on the lower (``lo``) or upper (``hi``) side of
        x, y and z, added to ``padding``. Default 0.
    quiet : int | str, optional
        1 suppresses the line with the box dimensions. Default 0.
    _self : Any, optional
        PyMOL command module.

    Returns
    -------
    str
        Name of the new CGO object.

    Raises
    ------
    ValueError
        If a number of cells is smaller than 1.
    pymol.CmdException
        If the selection is empty.
    """
    cells = (int(nx), int(ny), int(nz))
    if min(cells) < 1:
        raise ValueError(f"drawgridbox: nx, ny and nz must be at least 1, got {cells}")
    # For an empty selection, get_extent returns a unit cube around the origin
    # without an error, which would draw a box at a meaningless place.
    if _self.count_atoms(selection) == 0:
        raise pymol.CmdException(f"drawgridbox: selection '{selection}' has no atoms")
    extent = _self.get_extent(selection)
    lower = (
        np.array(extent[0]) - float(padding) - [float(pxlo), float(pylo), float(pzlo)]
    )
    upper = (
        np.array(extent[1]) + float(padding) + [float(pxhi), float(pyhi), float(pzhi)]
    )

    lines: list[float] = [cgo.LINEWIDTH, float(lw), cgo.BEGIN, cgo.LINES]
    lines += [cgo.COLOR, float(r), float(g), float(b)]
    for start, end in grid_segments(lower, upper, cells):
        lines += [cgo.VERTEX, *start.tolist(), cgo.VERTEX, *end.tolist()]
    lines.append(cgo.END)

    name: str = _self.get_unused_name("gridbox")
    _self.load_cgo(lines, name)
    if not int(quiet):
        size = ", ".join(f"{x:.2f}" for x in upper - lower)
        print(f" drawgridbox: box ({size}) A with {cells} cells drawn as '{name}'")
    return name
