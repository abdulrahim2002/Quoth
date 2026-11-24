# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Human-readable text for a `DownloadProgress`, shared by both download surfaces.

Onboarding and Settings render model downloads identically without importing each
other.
"""

from gi.repository import GLib

from quoth.downloader import DownloadProgress

# Weight on the newest rate sample (tqdm's default): responsive enough to follow a
# real change of speed, damped enough that one fast burst does not halve the estimate.
_EMA_WEIGHT = 0.3
_MIN_SAMPLES = 3  # samples before an estimate is worth showing


def format_eta(seconds: float) -> str:
    """Return a coarse "time left" for `seconds` remaining.

    The estimate cannot support per-second accuracy, and a figure that counts down
    unevenly reads as broken.
    """
    if seconds < 10:
        return "almost done"
    if seconds < 60:
        return f"about {round(seconds / 10) * 10} seconds left"
    minutes = round(seconds / 60)
    if minutes <= 1:
        return "about a minute left"
    return f"about {minutes} minutes left"


class EtaEstimator:
    """An estimator that turns reported progress into a human "time left".

    `update` returns None until there is a trustworthy estimate. The downloader's rate
    covers only the last interval and swings with every stall, so this smooths it and
    withholds an answer until enough samples arrive.
    """

    def __init__(self):
        self._rate = 0.0
        self._samples = 0

    def update(self, progress: DownloadProgress) -> str | None:
        if progress.total <= 0 or progress.bytes_per_sec <= 0:
            return None
        self._rate = (
            progress.bytes_per_sec
            if self._samples == 0
            else _EMA_WEIGHT * progress.bytes_per_sec + (1 - _EMA_WEIGHT) * self._rate
        )
        self._samples += 1
        remaining = progress.total - progress.downloaded
        if self._samples < _MIN_SAMPLES or remaining <= 0 or self._rate <= 0:
            return None
        return format_eta(remaining / self._rate)


def format_download_progress(progress: DownloadProgress) -> str:
    """Return progress text, e.g. "12.3 MB of 466.0 MB (42%), 4.1 MB/s".

    It degrades gracefully: "Downloading…" before the first bytes, and "12.3 MB,
    4.1 MB/s" once bytes flow without a known total (the server sent no
    Content-Length).
    """
    if progress.downloaded == 0:
        return "Downloading…"
    downloaded = GLib.format_size(progress.downloaded)
    speed = f"{GLib.format_size(int(progress.bytes_per_sec))}/s"
    if progress.total > 0:
        pct = round(progress.fraction * 100)
        return f"{downloaded} of {GLib.format_size(progress.total)} ({pct}%), {speed}"
    return f"{downloaded}, {speed}"
