# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `backdrop`, the onboarding video or still; they need a display."""

import pytest


def _pin_gtk(monkeypatch, version):
    from gi.repository import Gtk

    major, minor, micro = version
    monkeypatch.setattr(Gtk, "get_major_version", lambda: major)
    monkeypatch.setattr(Gtk, "get_minor_version", lambda: minor)
    monkeypatch.setattr(Gtk, "get_micro_version", lambda: micro)


@pytest.fixture(autouse=True)
def _isolate_gtk_version(monkeypatch):
    """Video behaviour must not depend on the GTK the suite happens to run on."""
    _pin_gtk(monkeypatch, (4, 24, 2))


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "backdrop.webm"
    path.write_bytes(b"")
    return path


@pytest.fixture
def still(tmp_path):
    import cairo

    path = tmp_path / "still.png"
    cairo.ImageSurface(cairo.FORMAT_ARGB32, 32, 24).write_to_png(str(path))
    return path


def test_gtk_4_24_1_shows_the_still_and_never_opens_the_video(
    display, still, video, monkeypatch
):
    from unittest.mock import Mock

    from quoth import backdrop as backdrop_module
    from quoth.backdrop import Backdrop

    _pin_gtk(monkeypatch, (4, 24, 1))
    new_media = Mock()
    monkeypatch.setattr(backdrop_module.Gtk.MediaFile, "new", new_media)

    backdrop = Backdrop(still=still, video=video)
    backdrop.start()

    new_media.assert_not_called()
    assert backdrop.get_paintable().get_intrinsic_width() == 32


@pytest.mark.parametrize("version", [(4, 24, 0), (4, 25, 0)])
def test_gtk_either_side_of_4_24_1_opens_the_video(
    display, video, monkeypatch, version
):
    from quoth.backdrop import Backdrop

    _pin_gtk(monkeypatch, version)
    loaded = []
    monkeypatch.setattr(Backdrop, "_load_video", lambda self, path: loaded.append(path))

    Backdrop(video=video)

    assert loaded == [video]


def test_video_load_is_deferred_until_motion_is_reenabled_and_requested(
    display, video, monkeypatch
):
    from quoth.backdrop import Backdrop

    loaded = []
    monkeypatch.setattr(Backdrop, "_load_video", lambda self, path: loaded.append(path))
    backdrop = Backdrop(video=video, motion_enabled=False)

    backdrop.start()
    assert loaded == []
    backdrop.set_motion_enabled(True)
    assert loaded == [video]


def test_a_still_is_shown(display, still):
    from quoth.backdrop import Backdrop

    paintable = Backdrop(still=still).get_paintable()

    assert paintable.get_intrinsic_width() == 32
    assert paintable.get_intrinsic_height() == 24


def test_disabling_video_pauses_and_selects_the_still(display, monkeypatch):
    from unittest.mock import Mock

    from quoth.backdrop import Backdrop

    backdrop = Backdrop()
    backdrop._media = Mock()
    use_still = Mock()
    monkeypatch.setattr(backdrop, "_load_still", use_still)

    backdrop.set_motion_enabled(False)

    backdrop._media.pause.assert_called_once_with()
    use_still.assert_called_once_with()


def test_close_releases_media_once_and_suppresses_late_errors(display):
    from unittest.mock import Mock

    from quoth.backdrop import Backdrop

    backdrop = Backdrop()
    media = Mock()
    media.get_error.return_value = Mock(message="late")
    backdrop._media = media
    backdrop._media_error_handler = 7

    backdrop.close()
    backdrop.close()
    backdrop._on_media_error(media, None)

    media.pause.assert_called_once_with()
    media.disconnect.assert_called_once_with(7)
    media.clear.assert_called_once_with()
    assert backdrop._media is None


def test_immediate_media_error_never_installs_failed_paintable(
    display, video, monkeypatch
):
    from unittest.mock import Mock

    from quoth import backdrop as backdrop_module
    from quoth.backdrop import Backdrop

    media = Mock()
    media.connect.return_value = 9
    error = Mock(message="missing decoder")
    media.get_error.return_value = error

    def fail_during_filename(_filename):
        media.connect.call_args.args[1](media, None)

    media.set_filename.side_effect = fail_during_filename
    monkeypatch.setattr(backdrop_module.Gtk.MediaFile, "new", lambda: media)
    backdrop = Backdrop()
    paintables = []
    backdrop.set_paintable = paintables.append
    backdrop._load_still = Mock()

    backdrop._load_video(video)

    assert paintables == [None]
    assert backdrop._media is None
    media.disconnect.assert_called_once_with(9)
    media.clear.assert_called_once_with()
    media.set_muted.assert_not_called()
    backdrop._load_still.assert_called_once_with()


def test_video_initializes_backend_before_muting(display, video, monkeypatch):
    from unittest.mock import Mock

    from quoth import backdrop as backdrop_module
    from quoth.backdrop import Backdrop

    media = Mock()
    media.get_error.return_value = None
    monkeypatch.setattr(backdrop_module.Gtk.MediaFile, "new", lambda: media)
    backdrop = Backdrop()
    backdrop.set_paintable = Mock()

    backdrop._load_video(video)

    calls = [call[0] for call in media.mock_calls]
    assert calls.index("connect") < calls.index("set_filename")
    assert calls.index("set_filename") < calls.index("set_muted")
    media.set_muted.assert_called_once_with(True)
    backdrop.set_paintable.assert_called_once_with(media)
