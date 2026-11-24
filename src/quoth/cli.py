# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""The `quoth` command.

Kept free of GTK so a keybinding's `quoth --toggle` can hand the toggle to a
running instance over D-Bus and exit before the app's imports are paid; only
when no instance answers does it load the application, which then either
becomes the instance or has GIO forward the option to one that appeared.
A lone `quoth --quit` uses the same D-Bus hand-off, and the script commands
(`--start`, `--stop`, `--cancel`, `--status`) call the `dev.quoth.Dictation1`
interface; neither loads the application.
"""

from __future__ import annotations

import sys

from gi.repository import GLib

from quoth import ipc


def main() -> int:
    """Run the `quoth` command and return its exit status."""
    # `--help` and `--version` name the program, whatever interpreter started it.
    GLib.set_prgname("quoth")
    args = sys.argv[1:]
    # Only a lone toggle takes the fast path; `-t --help` or a typo still gets answered.
    if args in (["-t"], ["--toggle"]) and ipc.activate_action("toggle"):
        return 0
    # A lone quit never starts Quoth just to stop it, whether or not one answers.
    if args in (["-q"], ["--quit"]):
        ipc.activate_action("quit")
        return 0
    # A lone script command acts only on a running Quoth and never loads the app.
    if args in (["--start"], ["--stop"], ["--cancel"], ["--status"]):
        return _run_script_command(args[0].removeprefix("--"))
    from quoth.app import main as run_app

    return run_app()


def _run_script_command(command: str) -> int:
    """Run one script command against the running instance and return its status.

    1 means the instance refused or did not answer, with the reason on stderr; 2 that
    none is running or there is no session bus.
    """
    try:
        if command == "status":
            print(ipc.dictation_state())
        else:
            ipc.control_dictation(command.capitalize())
    except ipc.NoSessionBus:
        print(
            "quoth: there is no D-Bus session bus, so no running Quoth can be reached",
            file=sys.stderr,
        )
        return 2
    except ipc.NotRunning:
        print(
            "quoth: Quoth is not running; start it with quoth --background",
            file=sys.stderr,
        )
        return 2
    except ipc.Refused as error:
        print(f"quoth: {error}", file=sys.stderr)
        return 1
    return 0
