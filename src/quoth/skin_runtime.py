# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""The app skin at runtime: the rendered skin and the watch on its active file.

`SkinRuntime` loads skin.toml, renders it, follows edits through a Gio directory
monitor and copies presets and shelf skins over it (removing it for Default), all
on the GTK main thread.
"""

from collections.abc import Callable
from pathlib import Path

import gi

gi.require_version("Gio", "2.0")
from gi.repository import Gio, GLib
from loguru import logger

from quoth import skin
from quoth.scheduler import Scheduler


class SkinRuntime:
    """The live app skin for one config directory.

    `render` draws a skin, and `on_changed(message)` tells the Settings view to
    refresh, with a message to show when there is one.
    """

    def __init__(
        self,
        config_dir: Path,
        scheduler: Scheduler,
        *,
        render: Callable[[skin.Skin], None],
        on_changed: Callable[[str | None], None],
    ):
        self._path = config_dir / "skin.toml"
        self._user_dir = skin.user_skin_dir(config_dir)
        self._scheduler = scheduler
        self._render = render
        self._on_changed = on_changed
        self._skin = skin.DEFAULT_SKIN
        self._load_error: str | None = None
        self._monitor: Gio.FileMonitor | None = None
        self._reload_queued = False

    def start(self) -> None:
        """Load the active skin, then follow edits to it."""
        self._reload()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            directory = Gio.File.new_for_path(str(self._path.parent))
            # The directory, not the file: an atomic save replaces the file's inode.
            self._monitor = directory.monitor_directory(
                Gio.FileMonitorFlags.WATCH_MOVES, None
            )
            self._monitor.connect("changed", self._on_file_changed)
        except (GLib.Error, OSError) as e:
            logger.warning("App skin hot-reload unavailable: {}", e)

    def shutdown(self) -> None:
        if self._monitor is not None:
            self._monitor.cancel()
            self._monitor = None

    def user_skins(self) -> list[str]:
        return skin.list_user_skins(self._user_dir)

    def current(self) -> str | None:
        """Return the active skin's id, None when it is unreadable or the user's own."""
        return skin.identify_skin(self._path, self._user_dir)

    def skin_of(self, skin_id: str) -> skin.Skin:
        return skin.skin_from_id(skin_id, self._user_dir)

    def load_error(self) -> str | None:
        return self._load_error

    def apply(self, skin_id: str) -> None:
        """Copy a skin over the active file, then render whatever the file holds."""
        try:
            skin.apply_skin_id(skin_id, self._user_dir, self._path)
        finally:
            self._reload()

    def import_file(self, path: Path) -> str:
        """Copy a skin file onto the shelf without changing the active skin."""
        return skin.import_user_skin(path, self._user_dir)

    def _on_file_changed(self, monitor, gfile, other_file, event) -> None:
        if not any(
            file is not None and file.get_basename() == self._path.name
            for file in (gfile, other_file)
        ):
            return
        if event not in (
            Gio.FileMonitorEvent.CHANGES_DONE_HINT,
            Gio.FileMonitorEvent.DELETED,
            Gio.FileMonitorEvent.RENAMED,
            Gio.FileMonitorEvent.MOVED_IN,
            Gio.FileMonitorEvent.MOVED_OUT,
        ):
            return
        if not self._reload_queued:
            self._reload_queued = True
            self._scheduler.dispatch(self._reload_queued_edit)

    def _reload_queued_edit(self) -> None:
        self._reload_queued = False
        self._reload()

    def _reload(self) -> None:
        message = None
        try:
            current = skin.load_skin(self._path)
        except (ValueError, OSError) as e:
            logger.warning("Could not load app skin {}: {}", self._path, e)
            message = "Couldn't load the skin. Keeping the current appearance."
        else:
            try:
                if current != self._skin:
                    self._render(current)
            except Exception:
                logger.exception("Could not render app skin")
                message = "Couldn't update the app appearance."
            else:
                self._skin = current
        self._load_error = message
        self._on_changed(message)
