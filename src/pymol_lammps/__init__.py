"""Load LAMMPS dump trajectories and periodic xyz trajectories into PyMOL.

The commands are ``load_lammps`` (dump files, with optional data file), ``load_xyz``
(multi-frame xyz files with a cell) and ``lammps_wrap``. The parsers (``dump``,
``datafile``, ``topology``, ``xyz``, ``bonds``) depend only on NumPy. The PyMOL
glue lives in ``loader`` and is imported only when registering the commands.
"""

__version__ = "0.1.0"


def __init_plugin__(app: object = None) -> None:
    """Register the commands when installed through PyMOL's plugin manager.

    Parameters
    ----------
    app : object, optional
        Plugin manager application object (unused).
    """
    from pymol_lammps.loader import register

    register()
