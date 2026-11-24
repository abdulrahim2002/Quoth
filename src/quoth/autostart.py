# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Launch-at-login via the freedesktop autostart spec.

Writes (or removes) `<app-id>.desktop` in the XDG autostart directory
(~/.config/autostart by default) so the desktop session launches Quoth on login. It is
named after the application id because startup-application managers key off the
desktop id, and the spec's Hidden=true override only works for same-named files.

Inside a Flatpak that path is the sandbox's own config dir, which the session never
reads, so there the entry is the Background portal's to write: it puts the right
`flatpak run` line on the host, with the user's consent.
"""

import shutil
from collections.abc import Callable
from pathlib import Path

import gi

gi.require_version("Gio", "2.0")
from gi.repository import Gio, GLib
from loguru import logger

from quoth import APP_ID, atomic_write, environment, paths

try:
    gi.require_version("Xdp", "1.0")
    from gi.repository import Xdp as _Xdp
except (ImportError, ValueError):
    _Xdp = None  # only the Flatpak asks the portal, and it bundles libportal

DESKTOP_FILENAME = f"{APP_ID}.desktop"

# A login start opens no window, since nobody at the desktop asked for one.
_DESKTOP_ENTRY = f"""[Desktop Entry]
Type=Application
Name=Quoth
Comment=Private voice dictation
Exec=quoth --background
Icon={APP_ID}
Terminal=false
X-GNOME-Autostart-enabled=true
"""


def _desktop_path() -> Path:
    return paths.autostart_dir() / DESKTOP_FILENAME


def set_enabled(
    enabled: bool, on_declined: Callable[[bool], None] | None = None
) -> None:
    """Create or remove the autostart entry to match `enabled`.

    Natively, disabling leaves an entry the desktop has switched off as the desktop
    left it. Failures are logged rather than raised, so they reach the log file.

    `on_declined` runs when the desktop declines the request, which only the portal
    path can do, and gets True when the desktop cannot start apps at login at all.
    """
    if environment.in_flatpak():
        _request_background(enabled, on_declined)
        return
    path = _desktop_path()
    try:
        if enabled:
            atomic_write.write_text(path, _DESKTOP_ENTRY)
            logger.info("Autostart enabled ({})", path)
            # The entry runs `quoth` from PATH, where a uv, pipx or source install may
            # not be at login.
            logger.debug("Autostart command resolves to {}", shutil.which("quoth"))
        elif is_enabled():
            path.unlink()
            logger.info("Autostart disabled ({})", path)
    except OSError as e:
        logger.error(
            "Autostart {} failed ({}): {}", "enable" if enabled else "disable", path, e
        )


def is_enabled() -> bool:
    """Whether the native autostart entry exists and the desktop has not turned it off.

    Desktops switch an entry off in place, with `Hidden=true` or GNOME's
    `X-GNOME-Autostart-enabled=false`. Inside a Flatpak the entry is the portal's, on
    the host, and not visible from here: the config is the record.
    """
    entry = GLib.KeyFile()
    try:
        entry.load_from_file(str(_desktop_path()), GLib.KeyFileFlags.NONE)
    except GLib.Error:
        return False
    return not _flag(entry, "Hidden", False) and _flag(
        entry, "X-GNOME-Autostart-enabled", True
    )


def _flag(entry: GLib.KeyFile, key: str, default: bool) -> bool:
    try:
        return entry.get_boolean(GLib.KEY_FILE_DESKTOP_GROUP, key)
    except GLib.Error:
        return default


def _request_background(
    enabled: bool, on_declined: Callable[[bool], None] | None = None
) -> None:
    """Ask the Background portal to write (or drop) the host autostart entry.

    Asynchronous, since this runs during activation and the portal may prompt.
    """
    try:
        portal = _Xdp.Portal.initable_new()
    except GLib.Error as e:
        logger.warning("Autostart portal: no session bus ({})", e.message)
        return

    def on_done(portal, result, *_) -> None:
        try:
            # A disable is answered with the background grant rather than the entry,
            # and the portal drops the entry on any successful answer.
            done = portal.request_background_finish(result) or not enabled
        except GLib.Error as e:
            # The portal lacks the interface when no backend implements it, as on
            # COSMIC, Sway and Hyprland.
            if e.matches(Gio.DBusError.quark(), Gio.DBusError.UNKNOWN_METHOD):
                logger.warning(
                    "Autostart portal: this desktop has no Background portal"
                )
                if on_declined:
                    on_declined(True)
                return
            if not e.matches(Gio.io_error_quark(), Gio.IOErrorEnum.CANCELLED):
                logger.warning("Autostart portal unavailable: {}", e.message)
                return
            done = False  # the user cancelled, or background use is denied
        if done:
            logger.info("Autostart {} via portal", "enabled" if enabled else "disabled")
            return
        logger.info("Autostart portal declined: wanted {}", enabled)
        if on_declined:
            on_declined(False)

    # The portal writes `flatpak run --command=quoth <app id> --background`.
    portal.request_background(
        None,
        "Start Quoth when you log in",
        ["quoth", "--background"],
        _Xdp.BackgroundFlags.AUTOSTART if enabled else _Xdp.BackgroundFlags.NONE,
        None,
        on_done,
        None,
    )
