#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Focused GTK text buffer for the installed dictation check."""

import sys
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

output = Path(sys.argv[1])
app = Gtk.Application(application_id="dev.quoth.QuothTestTarget")


def activate(application):
    window = Gtk.ApplicationWindow(application=application, title="Quoth GTK target")
    window.set_default_size(640, 480)
    view = Gtk.TextView()
    window.set_child(view)

    def save(_window):
        buffer = view.get_buffer()
        output.write_text(buffer.get_text(*buffer.get_bounds(), False))
        return False

    window.connect("close-request", save)
    window.present()
    view.grab_focus()


app.connect("activate", activate)
app.run([sys.argv[0]])
