# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Runtime environment facts: how this process was packaged and where it runs.

Whether we run inside a Flatpak, Snap or AppImage, where the host copy of our files
lives, how to run a command on the host, which desktop we are on, whether a Wayland
compositor is reachable and how many CPUs we may use. Standard library only, with no
GI or quoth imports, so it stays a leaf.
"""

from __future__ import annotations

import configparser
import os

_FLATPAK_INFO = "/.flatpak-info"


def in_flatpak() -> bool:
    """Whether we run inside a Flatpak sandbox."""
    return os.path.exists(_FLATPAK_INFO)


def in_snap() -> bool:
    """Whether we run from a Snap, whose runtime sets `$SNAP` to the mount path."""
    return bool(os.environ.get("SNAP"))


def in_appimage() -> bool:
    """Whether we run from an AppImage, whose runtime sets `$APPIMAGE`."""
    return bool(os.environ.get("APPIMAGE"))


def flatpak_app_path() -> str | None:
    """Return the host-side path of the app's `/app` tree, or None.

    Read from `Instance/app-path` in `/.flatpak-info` and used to run our bundled
    binaries on the host. None when not sandboxed or the key is absent.
    """
    if not in_flatpak():
        return None
    parser = configparser.ConfigParser()
    parser.read(_FLATPAK_INFO)
    return parser.get("Instance", "app-path", fallback="") or None


def host_argv(argv: list[str]) -> list[str]:
    """Wrap `argv` so it runs on the host.

    Inside a sandbox that prefixes `flatpak-spawn --host`; outside, `argv` comes back
    unchanged. The one place that knows the host-escape mechanism.
    """
    return ["flatpak-spawn", "--host", *argv] if in_flatpak() else list(argv)


def usable_cpus() -> int:
    """Return how many CPUs this process may run on, for sizing decode threads.

    Reads the affinity mask so a cgroup, taskset or Flatpak CPU limit is respected,
    falling back to the logical CPU count. Always at least 1.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def wayland_session() -> bool:
    """Whether the clipboard tools can reach a Wayland compositor.

    True when the socket libwayland opens, `$WAYLAND_DISPLAY` or else `wayland-0` in
    `$XDG_RUNTIME_DIR`, exists. Flatpak sets the name even when the host has no socket
    to share, so the name alone proves nothing.
    """
    display = os.environ.get("WAYLAND_DISPLAY") or "wayland-0"
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if not (runtime_dir or os.path.isabs(display)):
        return False
    return os.path.exists(os.path.join(runtime_dir or "", display))


def desktop_names() -> list[str]:
    """Return the desktops named in `$XDG_CURRENT_DESKTOP`, lower-cased.

    The variable is a `:`-separated list (`pop:GNOME` -> `["pop", "gnome"]`); the list
    is empty when it is unset. Per the freedesktop spec, match against entries rather
    than the whole string, which keeps working when new entries appear.
    """
    raw = os.environ.get("XDG_CURRENT_DESKTOP", "")
    return [name for name in (part.strip().lower() for part in raw.split(":")) if name]


# Desktops whose compositors grant wlr-layer-shell to unsandboxed clients but withhold
# it from sandboxed (security-context) ones, so the native build restores the overlay
# the Flatpak cannot show. KWin grants it to both, and GNOME has none.
_LAYER_SHELL_DESKTOPS = frozenset({"cosmic", "hyprland", "niri", "sway"})


def desktop_supports_layer_shell() -> bool:
    """Whether a native build would get the layer-shell overlay a sandbox is denied.

    Only on such a desktop is the native package worth suggesting. The actual
    in-process capability is probed separately (see `presentation`).
    """
    return bool(_LAYER_SHELL_DESKTOPS.intersection(desktop_names()))
