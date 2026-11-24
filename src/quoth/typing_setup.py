# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""The one-time "enable typing" disclosure dialog.

A standalone window that shares nothing with the settings window, shown where the
Flatpak lacks host /dev/uinput access: at launch, or at the first paste failure that
proves it when the launch check could not tell.
"""

import threading
from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, GLib, Gtk
from loguru import logger

from quoth import APP_ID


class TypingSetupDialog:
    """The one-time "enable typing" window, presented as soon as it is built.

    It appears only in the Flatpak, where ydotool needs /dev/uinput access because
    the desktop's virtual keyboard cannot paste everywhere. `on_run` performs the
    privileged setup (`typer.run_uinput_setup`) and returns whether it succeeded; it
    runs off the GTK thread so the polkit prompt never blocks the UI.
    `transcript_on_clipboard` says a failed dictation is waiting there.
    """

    def __init__(
        self,
        app: Adw.Application,
        on_run: Callable[[], bool],
        *,
        transcript_on_clipboard: bool,
    ):
        self._on_run = on_run
        win = Adw.ApplicationWindow(application=app)
        # A height of -1 fits the window to its content, button included, up to the
        # screen's height, past which the page scrolls.
        win.set_default_size(500, -1)
        win.set_title("Enable typing")

        description = (
            "To paste on this desktop, Quoth needs a one-time system setting to "
            "simulate keyboard input. It applies to anyone signed in at this "
            "computer's screen, so on a shared computer enable it only if you "
            "trust its other users. Click below to set it up (you'll be asked "
            "for your password), then log out and back in."
        )
        if transcript_on_clipboard:
            description += (
                "\n\nMeanwhile your transcription is on the clipboard: paste it "
                "with Ctrl+V or Shift+Insert."
            )
        status = Adw.StatusPage(
            icon_name=APP_ID, title="Enable typing", description=description
        )
        child = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self._btn = Gtk.Button(label="Enable typing…")
        self._btn.add_css_class("suggested-action")
        self._btn.add_css_class("pill")
        self._btn.connect("clicked", self._on_click)
        child.append(self._btn)
        self._label = Gtk.Label(label="")
        self._label.add_css_class("dim-label")
        self._label.set_wrap(True)
        child.append(self._label)
        status.set_child(child)

        toolbar = Adw.ToolbarView()
        toolbar.add_top_bar(Adw.HeaderBar())
        toolbar.set_content(status)
        win.set_content(toolbar)
        self._win = win
        win.present()

    def _on_click(self, _btn: Gtk.Button) -> None:
        # pkexec blocks on the polkit prompt: run it off the GTK main thread and
        # report back via idle_add so the dialog stays responsive.
        self._btn.set_sensitive(False)
        self._label.set_label("Setting up…")

        def worker() -> None:
            # Any raise still reports a failure, or the dialog would wait forever.
            try:
                ok = self._on_run()
            except Exception:
                logger.exception("uinput setup failed")
                ok = False
            GLib.idle_add(self._on_done, ok)

        threading.Thread(target=worker, name="uinput-setup", daemon=True).start()

    def _on_done(self, ok: bool) -> bool:
        text = (
            "Installed. Log out and back in to finish."
            if ok
            else "Setup didn't complete. Try again."
        )
        self._label.set_label(text)
        # The focused button went insensitive, so nothing leads a screen reader here.
        self._label.announce(text, Gtk.AccessibleAnnouncementPriority.MEDIUM)
        if not ok:
            self._btn.set_sensitive(True)
        return False  # one-shot idle callback
