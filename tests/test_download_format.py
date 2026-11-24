# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `download_format`.

Sizes are expected as `GLib.format_size` renders them, since its decimals depend on
the locale.
"""

import pytest
from gi.repository import GLib

from quoth.download_format import (
    EtaEstimator,
    format_download_progress,
    format_eta,
)
from quoth.downloader import DownloadProgress


def test_full_line_has_size_percent_and_speed():
    s = format_download_progress(
        DownloadProgress(0.42, 12_300_000, 466_000_000, 4_100_000)
    )
    assert s == (
        f"{GLib.format_size(12_300_000)} of {GLib.format_size(466_000_000)} (42%), "
        f"{GLib.format_size(4_100_000)}/s"
    )


def test_unknown_total_omits_total_and_percent():
    s = format_download_progress(DownloadProgress(0.0, 12_300_000, 0, 4_100_000))
    assert s == f"{GLib.format_size(12_300_000)}, {GLib.format_size(4_100_000)}/s"


def test_before_first_bytes_is_placeholder():
    assert format_download_progress(DownloadProgress(0.0, 0, 0, 0.0)) == "Downloading…"


class TestFormatEta:
    @pytest.mark.parametrize(
        ("seconds", "text"),
        [
            # Near the end, a countdown would only flicker.
            (4, "almost done"),
            # Seconds round to ten: precision the estimate cannot support reads as
            # broken.
            (38, "about 40 seconds left"),
            (42, "about 40 seconds left"),
            (63, "about a minute left"),
            (190, "about 3 minutes left"),
        ],
    )
    def test_text(self, seconds, text):
        assert format_eta(seconds) == text


class TestEtaEstimator:
    def _progress(self, downloaded, rate, total=142_000_000):
        return DownloadProgress(
            fraction=downloaded / total if total else 0.0,
            downloaded=downloaded,
            total=total,
            bytes_per_sec=rate,
        )

    def test_withholds_an_estimate_until_the_rate_settles(self):
        eta = EtaEstimator()
        assert eta.update(self._progress(1_000_000, 2_000_000)) is None
        assert eta.update(self._progress(3_000_000, 2_000_000)) is None
        # 137 MB left at a steady 2 MB/s: the first sample seeds the rate.
        assert eta.update(self._progress(5_000_000, 2_000_000)) == "about a minute left"

    def test_no_estimate_without_a_known_total(self):
        eta = EtaEstimator()
        for _ in range(5):
            assert eta.update(self._progress(1_000_000, 2_000_000, total=0)) is None

    def test_no_estimate_while_stalled(self):
        eta = EtaEstimator()
        for _ in range(5):
            assert eta.update(self._progress(1_000_000, 0)) is None

    def test_smoothing_damps_a_rate_spike(self):
        # The downloader reports only the last interval's rate, so one burst must not
        # slash the estimate: 132 MB left at a spiked 3 MB/s would read "about 40
        # seconds", while the smoothed rate keeps it at about a minute.
        eta = EtaEstimator()
        for _ in range(5):
            eta.update(self._progress(10_000_000, 1_000_000))
        assert (
            eta.update(self._progress(10_000_000, 3_000_000)) == "about a minute left"
        )
