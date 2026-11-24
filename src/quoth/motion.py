# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Reduced-motion preference detection.

Watches the system reduced-motion preference so the overlay can drop
decorative animation. Any reduction request wins:

1. The settings portal key org.freedesktop.appearance/reduced-motion
   (xdg-desktop-portal >= 1.21).
2. Gtk.Settings gtk-interface-reduced-motion on GTK 4.22+.
3. Gtk.Settings gtk-enable-animations == False.

On desktops without a settings portal backend (COSMIC, bare Sway) these
report defaults: the theme's [animation] enabled key is the manual override
there.
"""

from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gio, GLib, Gtk
from loguru import logger

from quoth.scheduler import guarded

_PORTAL_TIMEOUT_MS = 500


class ReducedMotionWatcher:
    """A watcher for the effective system reduced-motion preference.

    Calls `on_change(reduced: bool)` whenever the effective preference changes.
    Keep a reference to this object alive: the D-Bus proxy dies with it.
    """

    def __init__(self, on_change: Callable[[bool], None]):
        self._on_change = on_change
        self._portal_reduced = False

        self._settings = Gtk.Settings.get_default()
        if self._settings is not None:
            self._settings.connect(
                "notify::gtk-enable-animations", guarded(self._on_gtk_setting)
            )
            if self._settings.find_property("gtk-interface-reduced-motion"):
                self._settings.connect(
                    "notify::gtk-interface-reduced-motion",
                    guarded(self._on_gtk_setting),
                )
        else:
            logger.debug("No default Gtk.Settings; reduced-motion via portal only")

        self._proxy = None
        try:
            # No property load or portal start here: either can take the bus's 25 s
            # default at login, while the ReadOne below is bounded and starts it.
            self._proxy = Gio.DBusProxy.new_for_bus_sync(
                Gio.BusType.SESSION,
                Gio.DBusProxyFlags.DO_NOT_LOAD_PROPERTIES
                | Gio.DBusProxyFlags.DO_NOT_AUTO_START_AT_CONSTRUCTION,
                None,
                "org.freedesktop.portal.Desktop",
                "/org/freedesktop/portal/desktop",
                "org.freedesktop.portal.Settings",
                None,
            )
            # Listening first, so a portal slower to start than the read below still
            # reports a later change.
            self._proxy.connect("g-signal", guarded(self._on_portal_signal))
            ret = self._proxy.call_sync(
                "ReadOne",
                GLib.Variant("(ss)", ("org.freedesktop.appearance", "reduced-motion")),
                Gio.DBusCallFlags.NONE,
                _PORTAL_TIMEOUT_MS,
                None,
            )
            self._portal_reduced = _reduced_value(ret.unpack()[0])
            logger.debug("Reduced-motion portal: reduced={}", self._portal_reduced)
        except GLib.Error as e:
            # No portal or no such key (normal on COSMIC and Sway), or a slow start.
            logger.debug("Reduced-motion portal unavailable: {}", e.message)
        self._reduced = self._effective_reduced()

    @property
    def reduced(self) -> bool:
        """Whether any source currently requests reduced motion."""
        return self._reduced

    def _effective_reduced(self) -> bool:
        native_reduced = False
        if self._settings is not None:
            if self._settings.find_property("gtk-interface-reduced-motion"):
                native_reduced = _reduced_value(
                    self._settings.get_property("gtk-interface-reduced-motion")
                )
            return (
                self._portal_reduced
                or native_reduced
                or not self._settings.get_property("gtk-enable-animations")
            )
        return self._portal_reduced

    def _emit_if_changed(self) -> None:
        reduced = self._effective_reduced()
        if reduced == self._reduced:
            return
        self._on_change(reduced)
        self._reduced = reduced

    def _on_gtk_setting(self, settings, pspec):
        logger.debug(
            "Reduced-motion GTK setting changed: reduced={}",
            self._effective_reduced(),
        )
        self._emit_if_changed()

    def _on_portal_signal(self, proxy, sender, signal_name, params):
        if signal_name != "SettingChanged":
            return
        namespace, key, value = params.unpack()
        if namespace == "org.freedesktop.appearance" and key == "reduced-motion":
            self._portal_reduced = _reduced_value(value)
            logger.debug(
                "Reduced-motion portal changed: reduced={}", self._portal_reduced
            )
            self._emit_if_changed()


def _reduced_value(value: object) -> bool:
    """Read the portal/native enum without importing GTK 4.22-only symbols."""
    if isinstance(value, GLib.Variant):
        value = value.unpack()
    return value == 1
