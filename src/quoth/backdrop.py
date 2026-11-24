# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Onboarding backdrop: a video, or a still image.

One widget covers both so the asset can be swapped without touching the pages.
Video decoding needs GStreamer plugins that not every channel pulls in by
default; a missing decoder makes GTK report an error instead of drawing, so a
failed video falls back to the still rather than leaving the page blank.
"""

from __future__ import annotations

from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")

from gi.repository import Gdk, Gtk
from loguru import logger


class Backdrop(Gtk.Picture):
    """Full-bleed background, cropped to fill its allocation.

    Shows `video` when one is given, motion is allowed and GTK is not 4.24.1,
    otherwise `still`; `start` plays the video once. With neither, or when both fail to
    load, it draws nothing and the container's own background shows through.
    """

    def __init__(
        self,
        still: Path | None = None,
        video: Path | None = None,
        *,
        motion_enabled: bool = True,
    ):
        # Pictures carry the "img" role by default, which would have a screen
        # reader announce the backdrop as content; it carries no meaning.
        super().__init__(accessible_role=Gtk.AccessibleRole.PRESENTATION)
        version = (
            Gtk.get_major_version(),
            Gtk.get_minor_version(),
            Gtk.get_micro_version(),
        )
        if video is not None and version == (4, 24, 1):
            # On this GTK, releasing a media file makes GStreamer terminate the EGL
            # display GTK draws with, so under its GL renderer the next GL use aborts.
            logger.debug("Backdrop video skipped on GTK 4.24.1, using still")
            video = None
        self._still = still
        self._video = video
        self._motion_enabled = motion_enabled
        self._play_requested = False
        self._media: Gtk.MediaFile | None = None
        self._media_error_handler: int | None = None
        self._closed = False

        self.set_content_fit(Gtk.ContentFit.COVER)
        self.set_overflow(Gtk.Overflow.HIDDEN)
        # Decoration only: never take clicks from the page content above it.
        self.set_can_target(False)
        self.connect("hide", self._on_hide)

        if self._video is not None and motion_enabled:
            self._load_video(self._video)
        else:
            self._load_still()

    def start(self):
        """Begin playback, or defer it while motion is disabled. No-op for a still."""
        if self._closed:
            return
        self._play_requested = True
        if not self._motion_enabled:
            return
        if self._media is None and self._video is not None:
            self._load_video(self._video)
        if self._media is not None:
            self.set_paintable(self._media)
            self._media.play()

    def stop(self):
        """Pause playback so a hidden page costs no decoding. No-op for a still."""
        self._play_requested = False
        if self._media is not None:
            self._media.pause()

    def close(self) -> None:
        """Release the decoder and ignore callbacks queued during teardown."""
        if self._closed:
            return
        self._closed = True
        self._play_requested = False
        media = self._media
        self._media = None
        if media is None:
            return
        media.pause()
        if self._media_error_handler is not None:
            media.disconnect(self._media_error_handler)
            self._media_error_handler = None
        media.clear()
        self.set_paintable(None)

    def _on_hide(self, *_args) -> None:
        if self._media is not None:
            self._media.pause()

    def set_motion_enabled(self, enabled: bool) -> None:
        """Show the still while motion is off; resume a requested playback when on."""
        if enabled == self._motion_enabled:
            return
        self._motion_enabled = enabled
        if enabled:
            if self._play_requested:
                self.start()
            return
        if self._media is not None:
            self._media.pause()
        self._load_still()

    def _load_video(self, video: Path):
        # The clip plays once, and a finished clip rewinds to its first frame, so
        # one meant to rest has to end where it began.
        if self._closed:
            return
        media = Gtk.MediaFile.new()
        # Decode errors arrive asynchronously (a missing codec is only discovered
        # on prepare), so the fallback has to be a callback, not a return check.
        self._media = media
        self._media_error_handler = media.connect("notify::error", self._on_media_error)
        media.set_filename(str(video))
        if media.get_error() is not None:
            self._on_media_error(media, None)
        if self._media is media:
            # The media backend must exist before setting volume or mute.
            media.set_muted(True)
            self.set_paintable(media)

    def _on_media_error(self, media: Gtk.MediaFile, _param):
        error = media.get_error()
        if error is None or media is not self._media or self._closed:
            return
        logger.warning(
            "Backdrop video failed to decode ({}), using still", error.message
        )
        self._video = None
        self._media = None
        if self._media_error_handler is not None:
            media.disconnect(self._media_error_handler)
            self._media_error_handler = None
        media.clear()
        self.set_paintable(None)
        self._load_still()

    def _load_still(self):
        self.set_paintable(None)
        if self._still is None:
            return
        try:
            self.set_paintable(Gdk.Texture.new_from_filename(str(self._still)))
        except Exception:
            logger.exception("Backdrop still failed to load: {}", self._still)
