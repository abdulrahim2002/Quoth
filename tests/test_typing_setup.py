# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `typing_setup`. Each builds the real window, so each needs a display."""

import threading
from unittest.mock import Mock

import pytest

from tests.support import descendants, wait_until


@pytest.fixture(scope="module")
def app():
    from gi.repository import Adw, Gio

    # Parallel test processes must each own a local application.
    app = Adw.Application(
        application_id="dev.quoth.TestTypingSetup",
        flags=Gio.ApplicationFlags.NON_UNIQUE,
    )
    assert app.register(None)
    return app


@pytest.fixture
def text_scale(display):
    from gi.repository import Gtk

    settings = Gtk.Settings.get_default()
    previous = settings.get_property("gtk-xft-dpi")

    def scale(factor: float) -> None:
        settings.set_property("gtk-xft-dpi", int(96 * 1024 * factor))

    yield scale
    settings.set_property("gtk-xft-dpi", previous)


@pytest.fixture
def open_dialog(app, display):
    from quoth.typing_setup import TypingSetupDialog

    dialogs = []

    def open_dialog(on_run, transcript_on_clipboard):
        dialog = TypingSetupDialog(
            app, on_run, transcript_on_clipboard=transcript_on_clipboard
        )
        dialogs.append(dialog)
        return dialog

    yield open_dialog
    for dialog in dialogs:
        dialog._win.destroy()


@pytest.mark.parametrize(
    ("transcript_on_clipboard", "factor"), [(False, 1.5), (True, 1.0)]
)
def test_the_window_opens_with_its_action_in_view(
    text_scale, open_dialog, transcript_on_clipboard, factor
):
    # The button stays inside the window at large text and with the clipboard note.
    text_scale(factor)
    dialog = open_dialog(lambda: True, transcript_on_clipboard)
    win = dialog._win
    wait_until(lambda: dialog._btn.get_mapped() and win.get_height() > 0)
    visible, bounds = dialog._btn.compute_bounds(win)
    assert visible
    assert bounds.get_y() + bounds.get_height() <= win.get_height()


@pytest.mark.parametrize("transcript_on_clipboard", [False, True])
def test_a_waiting_transcript_is_pointed_out(open_dialog, transcript_on_clipboard):
    from gi.repository import Adw

    dialog = open_dialog(lambda: True, transcript_on_clipboard)
    status = next(w for w in descendants(dialog._win) if isinstance(w, Adw.StatusPage))
    note = "your transcription is on the clipboard"
    assert (note in status.get_description()) is transcript_on_clipboard


@pytest.mark.parametrize(
    ("succeeds", "outcome", "retry"),
    [(True, "Log out and back in", False), (False, "Try again", True)],
)
def test_setup_runs_off_the_ui_thread_and_a_failure_can_be_retried(
    open_dialog, succeeds, outcome, retry
):
    # The polkit prompt blocks the thread it runs on, which must not be the UI's.
    ran_on = []

    def on_run():
        ran_on.append(threading.current_thread())
        return succeeds

    dialog = open_dialog(on_run, transcript_on_clipboard=False)
    dialog._btn.emit("clicked")
    wait_until(lambda: outcome in dialog._label.get_label())
    assert len(ran_on) == 1 and ran_on[0] is not threading.main_thread()
    assert dialog._btn.get_sensitive() is retry


def test_a_setup_that_raises_still_ends_in_a_retry(open_dialog, caplog):
    # An unexpected error must not leave the dialog waiting with its button off.
    def on_run():
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    dialog = open_dialog(on_run, transcript_on_clipboard=False)
    dialog._btn.emit("clicked")
    wait_until(lambda: "Setup didn't complete" in dialog._label.get_label())
    assert dialog._btn.get_sensitive()
    [failure] = [r for r in caplog.records if r.getMessage() == "uinput setup failed"]
    assert failure.exc_info is not None


@pytest.mark.parametrize("succeeds", [True, False])
def test_the_outcome_is_read_out(open_dialog, succeeds):
    # The button that had the focus goes insensitive, so nothing else leads a screen
    # reader to the outcome and its log-out step.
    from gi.repository import Gtk

    dialog = open_dialog(lambda: succeeds, transcript_on_clipboard=False)
    dialog._label.announce = Mock()
    dialog._btn.emit("clicked")
    wait_until(lambda: dialog._label.announce.called)
    dialog._label.announce.assert_called_once_with(
        dialog._label.get_label(), Gtk.AccessibleAnnouncementPriority.MEDIUM
    )
