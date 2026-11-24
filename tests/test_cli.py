# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `cli`.

A toggle goes to a running instance without loading the app; anything else, or no
instance, loads and runs it. A quit never loads it, and a script command acts only on a
running instance, with an exit status a script can branch on.
"""

import sys
from unittest.mock import patch

import pytest
from gi.repository import GLib

from quoth import cli, ipc


def _run(argv, handed_over, status=0):
    with (
        patch.object(sys, "argv", ["quoth", *argv]),
        patch("quoth.ipc.activate_action", return_value=handed_over) as activate,
        patch("quoth.app.main", return_value=status) as run_app,
    ):
        exit_status = cli.main()
    return exit_status, activate, run_app


def test_toggle_handed_to_running_instance_skips_the_app():
    exit_status, activate, run_app = _run(["--toggle"], handed_over=True)
    activate.assert_called_once_with("toggle")
    run_app.assert_not_called()
    assert exit_status == 0


def test_short_flag_takes_the_fast_path_too():
    _, activate, run_app = _run(["-t"], handed_over=True)
    activate.assert_called_once_with("toggle")
    run_app.assert_not_called()


def test_toggle_without_an_instance_runs_the_app():
    _, activate, run_app = _run(["--toggle"], handed_over=False)
    activate.assert_called_once_with("toggle")
    run_app.assert_called_once()


@pytest.mark.parametrize("flag", ["-q", "--quit"])
def test_quit_goes_to_a_running_instance_and_never_runs_the_app(flag):
    # A launcher's Quit on a stopped Quoth would otherwise start it only to exit.
    for handed_over in (True, False):
        exit_status, activate, run_app = _run([flag], handed_over=handed_over)
        activate.assert_called_once_with("quit")
        run_app.assert_not_called()
        assert exit_status == 0


def test_plain_start_never_asks_the_bus():
    _, activate, run_app = _run([], handed_over=True)
    activate.assert_not_called()
    run_app.assert_called_once()


def test_a_toggle_with_anything_else_goes_to_the_app():
    # So `-t --help` prints help and `-t --bogus` is refused, not toggled.
    _, activate, run_app = _run(["-t", "--help"], handed_over=True)
    activate.assert_not_called()
    run_app.assert_called_once()


def test_the_exit_status_is_the_apps():
    exit_status, _, _ = _run(["--bogus"], handed_over=False, status=1)
    assert exit_status == 1


def test_help_and_version_name_the_program_quoth():
    GLib.set_prgname("python3")
    _run(["--help"], handed_over=False)
    assert GLib.get_prgname() == "quoth"


def _script(argv, *, state="idle", error=None):
    """Run a script command on a fake instance; return (status, run_app, control)."""
    with (
        patch.object(sys, "argv", ["quoth", *argv]),
        patch("quoth.ipc.dictation_state", return_value=state, side_effect=error),
        patch("quoth.ipc.control_dictation", side_effect=error) as control,
        patch("quoth.app.main") as run_app,
    ):
        status = cli.main()
    return status, run_app, control


@pytest.mark.parametrize(
    ("flag", "method"),
    [("--start", "Start"), ("--stop", "Stop"), ("--cancel", "Cancel")],
)
def test_a_script_command_reaches_the_instance_without_the_app(flag, method):
    status, run_app, control = _script([flag])
    control.assert_called_once_with(method)
    run_app.assert_not_called()
    assert status == 0


def test_status_prints_the_state_alone(capsys):
    status, run_app, _ = _script(["--status"], state="recording")
    assert capsys.readouterr() == ("recording\n", "")
    run_app.assert_not_called()
    assert status == 0


@pytest.mark.parametrize("flag", ["--start", "--status"])
def test_a_script_command_without_quoth_running_exits_2(flag, capsys):
    status, run_app, _ = _script([flag], error=ipc.NotRunning())
    assert capsys.readouterr() == (
        "",
        "quoth: Quoth is not running; start it with quoth --background\n",
    )
    run_app.assert_not_called()
    assert status == 2


def test_a_script_command_without_a_session_bus_says_so(capsys):
    # Starting Quoth would not help: no instance can be reached without a bus.
    status, run_app, _ = _script(["--stop"], error=ipc.NoSessionBus())
    assert capsys.readouterr() == (
        "",
        "quoth: there is no D-Bus session bus, so no running Quoth can be reached\n",
    )
    run_app.assert_not_called()
    assert status == 2


def test_a_refused_script_command_exits_1_with_the_reason(capsys):
    error = ipc.Refused("Finish setting up Quoth first")
    status, _, _ = _script(["--start"], error=error)
    assert capsys.readouterr() == ("", "quoth: Finish setting up Quoth first\n")
    assert status == 1


def test_a_script_command_with_anything_else_goes_to_the_app():
    # The app lists it in --help and refuses it beside other options.
    _, run_app, control = _script(["--status", "--help"])
    control.assert_not_called()
    run_app.assert_called_once()


def test_a_script_command_needs_its_dashes():
    _, run_app, control = _script(["start"])
    control.assert_not_called()
    run_app.assert_called_once()
