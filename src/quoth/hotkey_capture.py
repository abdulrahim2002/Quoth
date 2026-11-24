# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Dictation hotkey capture, and the user-facing labels of a hotkey token and the modes.

Capture uses `Gtk.EventControllerKey`: one key event carries both the physical key (the
hardware keycode minus the XKB offset of 8 is the layout-independent evdev token the
listener binds to) and the layout's character (`keyval`) for a label that matches the
keycap. `hotkey_label` derives that glyph from a stored token through Gdk, so Settings
and onboarding show the same label at rest as at capture. The listener reads evdev; only
capture uses GTK, because evdev names keys by US-QWERTY position and so mislabels other
layouts.

Shared by Settings and the onboarding shortcut step, it knows nothing of `Config` or the
controller: it hands the chosen evdev token back through `on_selected` and each caller
persists it its own way.
"""

from __future__ import annotations

from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gdk, GLib, Gtk
from loguru import logger

from quoth import keynames
from quoth.listener import key_name_for_scancode, scancode_for_key_name
from quoth.trigger import PUSH_TO_HOLD, TAP_TO_TOGGLE

_PROMPT = "Press the key you'd like to use to start dictation."
# A GTK hardware keycode is the evdev scancode plus this (the X11 legacy XKB kept).
_XKB_OFFSET = 8

# What onboarding and Settings call each input mode, so the two read the same.
MODE_LABELS = {
    TAP_TO_TOGGLE: "Tap to start and stop",
    PUSH_TO_HOLD: "Hold while you talk",
}


def token_for_keycode(keycode: int) -> str | None:
    """Return the evdev token for a GTK hardware keycode, or None if unmapped."""
    return key_name_for_scancode(keycode - _XKB_OFFSET)


def hotkey_label(token: str) -> str:
    """Return the user's keycap label for a hotkey `token`.

    That is the glyph the first configured layout prints (`KEY_Y` -> "Z" on a German
    layout), else the friendly evdev name ("F13", "Space"). The token is the physical
    binding; only the label follows the layout.
    """
    code = scancode_for_key_name(token)
    display = Gdk.Display.get_default()
    if code is None or display is None:
        return keynames.format_key(token)
    ok, keyval, *_ = display.translate_key(code + _XKB_OFFSET, Gdk.ModifierType(0), 0)
    codepoint = Gdk.keyval_to_unicode(keyval) if ok else 0
    return keynames.printable_label(codepoint, token)


def capture_hotkey(
    parent: Gtk.Widget,
    on_selected: Callable[[str], None],
    on_closed: Callable[[], None] | None = None,
) -> None:
    """Open a "press a key" dialog over `parent` and report the key the user confirms.

    A risky key gets a warning the user can override; `on_selected(evdev_token)` runs
    once a key is confirmed, and Cancel or Escape closes with no callback. `on_closed`,
    if given, fires however the dialog closes, so a caller can pause its own key
    handling around the capture (a PreferencesWindow's typeahead search would otherwise
    swallow the first keystroke).
    """
    _HotkeyCapture(parent, on_selected, on_closed).begin()


class _HotkeyCapture:
    """A staged capture dialog that reads one key through a `Gtk.EventControllerKey`.

    The stages run press -> warn/confirm. `_listening` gates the key handler so the
    warning-stage buttons keep working, and Escape always cancels, so it can never be
    bound by accident.
    """

    def __init__(
        self,
        parent: Gtk.Widget,
        on_selected: Callable[[str], None],
        on_closed: Callable[[], None] | None = None,
    ):
        self._parent = parent
        self._on_selected = on_selected
        self._listening = False
        self._dialog = Adw.Dialog(title="Dictation key")
        self._dialog.set_content_width(380)
        if on_closed is not None:
            self._dialog.connect("closed", lambda _d: on_closed())
        controller = Gtk.EventControllerKey()
        controller.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        controller.connect("key-pressed", self._on_key_pressed)
        self._dialog.add_controller(controller)

    def begin(self) -> None:
        logger.debug("Hotkey capture started")
        self._show_prompt()
        self._dialog.present(self._parent)

    def _on_key_pressed(self, _controller, keyval, keycode, _state) -> bool:
        if keyval == Gdk.KEY_Escape:
            self._dialog.close()
            return True
        if not self._listening:
            return False  # warning stage: let the keys reach the buttons
        token = token_for_keycode(keycode)
        if token is None:
            logger.debug(
                "Hotkey capture: unmapped scancode ignored (keycode={})", keycode
            )
            return True  # unmapped scancode: ignore, keep listening
        risk = keynames.hotkey_risk(token)
        if risk is None:
            self._finish(token)
        else:
            self._show_warning(token, risk)
        return True

    # --- stages ---

    def _show_prompt(self) -> None:
        self._set_stage(
            "Press a key",
            _PROMPT,
            [("Cancel", self._dialog.close, False)],
            icon_name="input-keyboard-symbolic",
        )
        self._listening = True

    def _show_warning(self, token: str, risk: str) -> None:
        self._listening = False
        logger.debug("Hotkey capture: '{}' flagged risky ({})", token, risk)
        label = hotkey_label(token)
        # No icon: the key name is the headline.
        self._set_stage(
            label,
            keynames.hotkey_warning_text(label, risk),
            [
                ("Choose another", self._show_prompt, False),
                ("Use anyway", lambda: self._confirm_risky(token), True),
            ],
        )

    def _set_stage(
        self,
        title: str,
        description: str,
        buttons: list[tuple[str, Callable[[], None], bool]],
        icon_name: str | None = None,
    ) -> None:
        page = Adw.StatusPage(
            icon_name=icon_name,
            title=title,
            description=GLib.markup_escape_text(description),
        )
        # A named group, so a screen reader speaks the stage as the focus lands on it.
        box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            accessible_role=Gtk.AccessibleRole.GROUP,
        )
        box.update_property(
            [Gtk.AccessibleProperty.LABEL, Gtk.AccessibleProperty.DESCRIPTION],
            [title, description],
        )
        # Hold keyboard focus inside the dialog across a stage rebuild: the controller
        # hears only its own subtree, and destroying the focused button would drop
        # focus to the parent window. The box takes it, so the prompt shows no ring.
        box.set_focusable(True)
        box.append(page)
        row = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=10, halign=Gtk.Align.CENTER
        )
        row.set_margin_bottom(12)
        for label, callback, suggested in buttons:
            button = Gtk.Button(label=label)
            button.add_css_class("pill")
            if suggested:
                button.add_css_class("suggested-action")
            button.connect("clicked", lambda _b, cb=callback: cb())
            row.append(button)
        box.append(row)
        toolbar = Adw.ToolbarView()
        toolbar.add_top_bar(Adw.HeaderBar())
        toolbar.set_content(box)
        self._dialog.set_child(toolbar)
        box.grab_focus()  # re-arm: keep the capture controller on the key-delivery path

    def _confirm_risky(self, token: str) -> None:
        logger.info("Hotkey capture: risk override, using '{}' anyway", token)
        self._finish(token)

    def _finish(self, token: str) -> None:
        self._listening = False
        logger.debug("Hotkey capture finished: token='{}'", token)
        self._on_selected(token)
        self._dialog.close()
