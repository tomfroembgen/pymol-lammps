"""Tab completion of file paths for the keyword arguments of the PyMOL commands.

PyMOL completes the text after the last comma as a file path unless a command has
an entry in ``cmd.auto_arg``. For ``load_lammps traj.lammpstrj, data=fil`` it thus
globs ``data=fil*`` and finds nothing. The completer in this module replaces that
fallback for the file arguments: it strips a ``key=`` prefix, completes the path
like PyMOL does (``~`` and environment variables expanded, directories get a
trailing ``/``) and puts the prefix back. It relies on the internal PyMOL list
``cmd.auto_arg`` and the contract of ``pymol.parser.complete_sc``, which calls
``sc().interpret(text, mode)`` and expects ``None`` (no match), a string (unique
match, replaces the text) or a list of strings (ambiguous matches).
"""

import glob
import os
from typing import Any

from pymol import cmd

# Number of comma-separated argument positions that get a completer.
N_POSITIONS = 12

# Keyword arguments that hold a file path, per command.
FILE_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "load_lammps": ("filename", "data"),
    "load_xyz": ("filename",),
}


class PathCompleter:
    """Complete the file path of one argument, with an optional ``key=`` prefix.

    Parameters
    ----------
    file_keys : tuple[str, ...]
        Keyword names whose value is a file path.
    """

    def __init__(self, file_keys: tuple[str, ...]) -> None:
        self.file_keys = file_keys

    def __call__(self) -> "PathCompleter":
        """Return itself, because PyMOL calls the shortcut object to get the matcher.

        Returns
        -------
        PathCompleter
            This completer.
        """
        return self

    def interpret(self, text: str, mode: int = 0) -> str | list[str] | None:
        """Complete the argument text as a path.

        Parameters
        ----------
        text : str
            Text of the argument before the cursor, for example ``data=~/sim/fi``.
        mode : int, optional
            Unused; part of the interface PyMOL expects.

        Returns
        -------
        str | list[str] | None
            The completed argument for a unique match, the candidate arguments for
            an ambiguous match, and None if nothing matches or the keyword is not
            a file argument.
        """
        key, equals, value = text.partition("=")
        if equals:
            if key not in self.file_keys:
                return None
            prefix = key + equals
        else:
            prefix, value = "", text
        matches = glob.glob(os.path.expandvars(os.path.expanduser(value)) + "*")
        if len(matches) == 1:
            slash = "/" if os.path.isdir(matches[0]) else ""
            return prefix + matches[0] + slash
        if not matches:
            return None
        return [prefix + m for m in sorted(matches)]


def register_completion(
    command: str, file_keys: tuple[str, ...], _self: Any = cmd
) -> None:
    """Make Tab complete the file arguments of a command at every position.

    Parameters
    ----------
    command : str
        Name of the PyMOL command.
    file_keys : tuple[str, ...]
        Keyword names whose value is a file path.
    _self : Any, optional
        PyMOL command module. Nothing happens if it has no ``auto_arg`` list.
    """
    auto_arg = getattr(_self, "auto_arg", None)
    if auto_arg is None:
        return
    while len(auto_arg) < N_POSITIONS:
        auto_arg.append({})
    completer = PathCompleter(file_keys)
    for position in auto_arg[:N_POSITIONS]:
        position[command] = [completer, "file", ""]
