# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `sound`, the recording-feedback earcon player.

Gst is mocked, so no audio device is needed. A cue is one persistent playbin: prerolled
to PAUSED, rewound and set PLAYING by `play`, PAUSED again on EOS, evicted on ERROR.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from quoth.sound import SoundPlayer


def _playbin(gst, *, state=None):
    """Return a playbin mock whose `get_state` returns a real 3-tuple.

    The state defaults to READY, so `play` skips the rewind seek as on a fresh cue.
    """
    pb = MagicMock()
    pb.get_state.return_value = (
        gst.StateChangeReturn.SUCCESS,
        state if state is not None else gst.State.READY,
        gst.State.VOID_PENDING,
    )
    return pb


def test_play_missing_file_is_noop():
    sp = SoundPlayer()
    with patch("quoth.sound.Gst") as gst:
        sp.play("does-not-exist")
        gst.ElementFactory.make.assert_not_called()
        assert not sp._cues


def test_play_prerolls_then_plays_and_caches():
    sp = SoundPlayer()
    with patch("quoth.sound.Gst") as gst:
        pb = _playbin(gst)
        gst.ElementFactory.make.return_value = pb
        sp.play("start")  # start.wav ships in quoth/sounds/
        gst.ElementFactory.make.assert_called_once_with("playbin", None)
        prop, uri = pb.set_property.call_args_list[0].args
        assert prop == "uri"
        # playbin refuses a bare path as an invalid URI.
        assert uri.startswith("file:///") and uri.endswith("/quoth/sounds/start.wav")
        # Preroll (PAUSED) precedes playback (PLAYING).
        assert pb.set_state.call_args_list == [
            call(gst.State.PAUSED),
            call(gst.State.PLAYING),
        ]
        assert sp._cues["start"] is pb


def test_play_reuses_persistent_cue():
    sp = SoundPlayer()
    with patch("quoth.sound.Gst") as gst:
        pb = _playbin(gst, state=gst.State.PAUSED)
        gst.ElementFactory.make.return_value = pb
        sp.play("start")
        sp.play("start")
        # Built once, reused on the second press.
        gst.ElementFactory.make.assert_called_once()
        # A prerolled (PAUSED) cue is flush-seeked to the start before replay.
        pb.seek_simple.assert_called_with(
            gst.Format.TIME,
            gst.SeekFlags.FLUSH | gst.SeekFlags.KEY_UNIT,
            0,
        )


def test_still_playing_cue_is_not_retriggered():
    """A `play` while the cue is still PLAYING only sets the volume.

    A flushing seek mid-playback would reset the sink ringbuffer at a non-zero sample
    and click.
    """
    sp = SoundPlayer()
    with patch("quoth.sound.Gst") as gst:
        pb = _playbin(gst, state=gst.State.PLAYING)
        gst.ElementFactory.make.return_value = pb
        sp.play("start", 40)
        pb.set_property.assert_called_with("volume", 0.4)
        pb.seek_simple.assert_not_called()
        # No second PLAYING transition: the in-flight cue is left to finish.
        assert call(gst.State.PLAYING) not in pb.set_state.call_args_list


def test_fresh_cue_skips_rewind_seek():
    sp = SoundPlayer()
    with patch("quoth.sound.Gst") as gst:
        pb = _playbin(gst, state=gst.State.READY)  # not yet prerolled
        gst.ElementFactory.make.return_value = pb
        sp.play("start")
        pb.seek_simple.assert_not_called()


@pytest.mark.parametrize(
    ("args", "expected"), [((), 1.0), ((70,), 0.7), ((0,), 0.0), ((100,), 1.0)]
)
def test_play_sets_the_volume_before_the_cue_sounds(args, expected):
    sp = SoundPlayer()
    with patch("quoth.sound.Gst") as gst:
        pb = _playbin(gst)
        gst.ElementFactory.make.return_value = pb
        sp.play("start", *args)
        calls = pb.mock_calls
        volume_set = calls.index(call.set_property("volume", pytest.approx(expected)))
        assert volume_set < calls.index(call.set_state(gst.State.PLAYING))


def test_prewarm_prerolls_without_playing():
    sp = SoundPlayer()
    with patch("quoth.sound.Gst") as gst:
        gst.ElementFactory.make.side_effect = lambda *_: _playbin(gst)
        sp.prewarm()  # as the app calls it: both cues it plays
        assert set(sp._cues) == {"start", "stop"}
        for pb in sp._cues.values():
            pb.set_state.assert_called_once_with(gst.State.PAUSED)
            assert call(gst.State.PLAYING) not in pb.set_state.call_args_list


def test_eos_returns_cue_to_paused_and_keeps_it():
    sp = SoundPlayer()
    with patch("quoth.sound.Gst") as gst:
        pb = _playbin(gst, state=gst.State.PAUSED)
        gst.ElementFactory.make.return_value = pb
        sp.play("start")
        message = MagicMock()
        message.type = gst.MessageType.EOS
        sp._on_message(pb.get_bus.return_value, message, "start")
        # Rewound-ready, not torn down.
        assert pb.set_state.call_args_list[-1] == call(gst.State.PAUSED)
        assert sp._cues["start"] is pb


def test_error_evicts_cue_for_rebuild():
    sp = SoundPlayer()
    with patch("quoth.sound.Gst") as gst:
        pb = _playbin(gst, state=gst.State.PAUSED)
        gst.ElementFactory.make.return_value = pb
        sp.play("start")
        message = MagicMock()
        message.type = gst.MessageType.ERROR
        message.parse_error.return_value = (MagicMock(message="boom"), "debug")
        sp._on_message(pb.get_bus.return_value, message, "start")
        assert "start" not in sp._cues
        assert pb.set_state.call_args_list[-1] == call(gst.State.NULL)
        pb.get_bus.return_value.remove_signal_watch.assert_called_once()


def test_warning_keeps_the_cue():
    # Non-fatal (an underrun, sink latency): evicting would cost the next press its
    # prerolled cue.
    sp = SoundPlayer()
    with patch("quoth.sound.Gst") as gst:
        pb = _playbin(gst, state=gst.State.PAUSED)
        gst.ElementFactory.make.return_value = pb
        sp.play("start")
        message = MagicMock()
        message.type = gst.MessageType.WARNING
        message.parse_warning.return_value = (MagicMock(message="underrun"), "debug")
        sp._on_message(pb.get_bus.return_value, message, "start")
        assert sp._cues["start"] is pb
        assert call(gst.State.NULL) not in pb.set_state.call_args_list


def test_shutdown_tears_down_all_cues():
    sp = SoundPlayer()
    with patch("quoth.sound.Gst") as gst:
        gst.ElementFactory.make.side_effect = lambda *_: _playbin(gst)
        sp.prewarm(("start", "stop"))
        cues = list(sp._cues.values())
        sp.shutdown()
        assert not sp._cues
        for pb in cues:
            assert pb.set_state.call_args_list[-1] == call(gst.State.NULL)
