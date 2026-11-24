# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""How Quoth surfaces feedback on this system: the one source of overlay capability.

The floating waveform pill needs wlr-layer-shell: precise positioning plus a surface
that never takes keyboard focus. Where the compositor grants it (native on wlroots,
COSMIC, Hyprland, niri and KDE; the Flatpak on KDE) the pill shows. Where it does not
(GNOME has no layer-shell; COSMIC, Sway, Hyprland and niri deny it to a Flatpak's
security context) the pill must not show: a Wayland toplevel cannot refuse keyboard
focus (those hints are X11-only), so it would steal focus from the editor and the paste
would land on the pill.
There the tray carries state and errors go to notifications.

Every module derives its presentation decisions from here; nothing else re-tests the
environment. `overlay_available()` is a protocol probe; the desktop name only feeds the
"would the native build help?" hint, which cannot be probed (it asks about a
hypothetical unsandboxed run); see `environment.desktop_supports_layer_shell()`.
"""

from __future__ import annotations

import gi
from loguru import logger

from quoth import environment

# gtk4-layer-shell may be absent (not built or installed). Importing it says nothing
# about whether the compositor offers the protocol; the `is_supported()` probe tells.
try:
    gi.require_version("Gtk4LayerShell", "1.0")
    from gi.repository import Gtk4LayerShell
except (ImportError, ValueError) as e:  # pragma: no cover - depends on the build
    Gtk4LayerShell = None
    logger.debug("gtk4-layer-shell not available: {}", e)

# Where to send users who want the on-screen overlay the sandbox can't provide.
NATIVE_INSTALL_URL = "https://github.com/abullard1/Quoth#install"


def overlay_available() -> bool:
    """Whether the compositor grants this process wlr-layer-shell.

    That decides whether the floating pill can show without stealing keyboard focus.
    Call it after GTK has initialised (the support check needs a display connection);
    it is cheap enough to call on demand.
    """
    return Gtk4LayerShell is not None and Gtk4LayerShell.is_supported()


def suggests_native_overlay() -> bool:
    """Whether to point the user at the native package for the overlay.

    True only when sandboxed, without the pill here, and on a desktop where an
    unsandboxed build would get it, so installing the native package restores the
    pill (never shown on GNOME).
    """
    suggest = (
        environment.in_flatpak()
        and not overlay_available()
        and environment.desktop_supports_layer_shell()
    )
    logger.debug("Native-overlay suggestion: {}", suggest)
    return suggest


def feedback_enabled(setting: str) -> bool:
    """Resolve a tri-state feedback setting ("auto", "on" or "off") to on or off.

    "auto" follows overlay availability: feedback fires where there is no pill and stays
    quiet where the pill already shows state. This is the one home of the auto rule,
    mirroring `device="auto"`.
    """
    if setting == "on":
        return True
    if setting == "off":
        return False
    return not overlay_available()  # "auto"


def error_uses_notification(essential: bool, overlay: bool) -> bool:
    """Whether an error should raise a desktop notification.

    Always when there is no pill; also for an essential error even when the pill shows,
    so its guidance outlives the pill's auto-dismiss and reaches screen readers, which
    cannot read the pill.
    """
    uses = essential or not overlay
    logger.debug(
        "Error channel: notification={} (essential={}, overlay={})",
        uses,
        essential,
        overlay,
    )
    return uses
