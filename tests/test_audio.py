# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `audio`."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from tests.support import wait_until

# One second of 16 kHz mono float32 in ten buffers, then the stream ends.
_SECOND = (
    "num-buffers=10 samplesperbuffer=1600 "
    "! audio/x-raw,format=F32LE,rate=16000,channels=1"
)
# A live input that never ends, in 10 ms buffers.
_LIVE = (
    "is-live=true samplesperbuffer=160 ! audio/x-raw,format=F32LE,rate=16000,channels=1"
)


@pytest.fixture
def capture(monkeypatch):
    """Build a real `AudioPipeline` over audiotestsrc, whatever sound server runs here.

    `capture(props, stays_playing)` returns it beside the chunks and levels it
    delivered, every graph it built, in order, and one entry per stream that ended.
    `props` may end in caps, which pin the format the source produces.
    """
    from quoth import audio

    rig = SimpleNamespace(chunks=[], levels=[], graphs=[], ended=[], pipeline=None)
    parse = audio.Gst.parse_launch

    def build(launch):
        graph = parse(launch)
        graph.get_bus().connect("message::eos", lambda *_: rig.ended.append(True))
        rig.graphs.append(graph)
        return graph

    def make(props, stays_playing):
        source = audio._AudioSource("audiotestsrc", None, props, stays_playing)
        monkeypatch.setattr(audio, "_select_source", lambda: source)
        monkeypatch.setattr(audio.Gst, "parse_launch", build)
        rig.pipeline = audio.AudioPipeline(
            on_audio_buffer=rig.chunks.append, on_level=rig.levels.append
        )
        return rig

    yield make
    if rig.pipeline is not None:
        rig.pipeline.destroy()


def _state(graph):
    """The state `graph` settles in.

    A live graph reaches PLAYING only once its first buffer arrives.
    """
    from quoth import audio

    return graph.get_state(3 * audio.Gst.SECOND)[1]


def _posting(graph) -> bool:
    """Whether the meter in `graph` posts levels."""
    return graph.get_by_name("level").get_property("post-messages")


def _stand_in_pipeline():
    """A real `AudioPipeline` whose GStreamer pipeline and level element are stand-ins.

    The gating under test is Quoth's own, and a real PLAYING needs a running sound
    server. The stand-in accepts every state change unless told otherwise.
    """
    from quoth import audio

    pipeline = audio.AudioPipeline()
    pipeline._level = MagicMock()
    pipeline._pipeline = MagicMock()
    pipeline._pipeline.set_state.return_value = audio.Gst.StateChangeReturn.SUCCESS
    pipeline._pipeline.get_state.return_value = (  # let destroy() unpack cleanly
        audio.Gst.StateChangeReturn.SUCCESS,
        audio.Gst.State.NULL,
        audio.Gst.State.VOID_PENDING,
    )
    pipeline._pipeline_playing = False
    return pipeline


def _source(element):
    """The capture source that drives `element`."""
    from quoth import audio

    return {s.element: s for s in audio._SOURCES}[element]


def test_audio_pipeline_conforms_to_audio_backend():
    # Drift guard: a rename of any capture method RecordingController drives
    # (start/stop/set_monitoring/warm_up/set_mic_source/destroy) breaks this.
    from quoth.audio import AudioBackend, AudioPipeline

    assert issubclass(AudioPipeline, AudioBackend)


class TestAudioPipelineInit:
    def test_creates_pipeline(self):
        from quoth.audio import AudioPipeline

        pipeline = AudioPipeline()
        assert pipeline._recording is False
        pipeline.destroy()


class TestLevelPosting:
    """The meter posts for a recording or a monitor; a monitor never opens the gate."""

    def test_monitoring_posts_without_recording(self, capture):
        # The level sits on its own tee branch, so the meter can watch the input
        # without opening the software gate that feeds the transcription buffer.
        from quoth import audio

        rig = capture(_SECOND, stays_playing=False)
        [graph] = rig.graphs
        rig.pipeline.set_monitoring(True)  # starts the stopped source
        wait_until(lambda: rig.ended)  # every buffer has met the gate
        assert _posting(graph) is True
        assert rig.chunks == []
        rig.pipeline.set_monitoring(False)
        assert _state(graph) == audio.Gst.State.NULL

    def test_stopping_a_recording_keeps_a_monitor_alive(self, capture):
        # Recording while Settings is open, then stopping, must not turn posting off
        # and leave the meter dead until Settings reopens.
        from quoth import audio

        rig = capture(_LIVE, stays_playing=False)
        [graph] = rig.graphs
        rig.pipeline.set_monitoring(True)
        rig.pipeline.start()
        rig.pipeline.stop()
        assert (_state(graph), _posting(graph)) == (audio.Gst.State.PLAYING, True)
        rig.pipeline.set_monitoring(False)
        assert (_state(graph), _posting(graph)) == (audio.Gst.State.NULL, False)

    def test_dropping_a_monitor_keeps_a_recording_alive(self, capture):
        from quoth import audio

        rig = capture(_LIVE, stays_playing=False)
        [graph] = rig.graphs
        rig.pipeline.start()
        assert _posting(graph) is True
        rig.pipeline.set_monitoring(True)
        rig.pipeline.set_monitoring(False)
        assert (_state(graph), _posting(graph)) == (audio.Gst.State.PLAYING, True)
        rig.pipeline.stop()
        assert (_state(graph), _posting(graph)) == (audio.Gst.State.NULL, False)


class TestPlayingState:
    """A requested PLAYING is not an achieved one."""

    def _pipeline(self, set_state_results):
        pipeline = _stand_in_pipeline()
        pipeline._pipeline.set_state.side_effect = set_state_results
        return pipeline

    def test_a_refused_start_raises_closes_the_gate_and_retries_next_time(self, caplog):
        from quoth import audio

        refused, ok = (
            audio.Gst.StateChangeReturn.FAILURE,
            audio.Gst.StateChangeReturn.ASYNC,
        )
        pipeline = self._pipeline([refused, ok, ok])  # PLAYING refused, NULL, PLAYING
        with pytest.raises(audio.AudioUnavailable):
            pipeline.start()
        assert pipeline._recording is False and pipeline._pipeline_playing is False
        # The refusal names the source element and the microphone it tried.
        assert (
            f"Audio pipeline refused to start ({pipeline._source.element}, mic auto)"
            in caplog.messages
        )
        pipeline.start()  # the next recording tries again rather than trusting a flag
        assert pipeline._pipeline_playing is True
        states = [c.args[0] for c in pipeline._pipeline.set_state.call_args_list]
        assert states == [
            audio.Gst.State.PLAYING,
            audio.Gst.State.NULL,
            audio.Gst.State.PLAYING,
        ]

    def test_stop_names_the_default_device_once_per_change(self, caplog):
        pipeline = self._pipeline(None)
        src = pipeline._pipeline.get_by_name.return_value
        for device in ("alsa_input.usb", "alsa_input.usb", "", "alsa_input.pci"):
            src.get_property.return_value = device
            pipeline.stop()
        assert [m for m in caplog.messages if m.startswith("Default input")] == [
            "Default input device: alsa_input.usb",
            "Default input device: alsa_input.pci",
        ]

    def test_the_default_device_line_masks_a_bluetooth_address(self, caplog):
        pipeline = self._pipeline(None)
        src = pipeline._pipeline.get_by_name.return_value
        for device in (
            "bluez_input.AA_BB_CC_DD_EE_FF.0",
            "bluez_input.11:22:33:44:55:66",
        ):
            src.get_property.return_value = device
            pipeline.stop()
        assert [m for m in caplog.messages if m.startswith("Default input")] == [
            "Default input device: bluez_input.XX_XX_XX_XX_XX_XX.0",
            "Default input device: bluez_input.XX_XX_XX_XX_XX_XX",
        ]

    def test_a_bus_error_forgets_the_stream(self):
        from quoth import audio

        pipeline = self._pipeline(None)
        pipeline._pipeline_playing = True
        message = MagicMock()
        message.parse_error.return_value = (MagicMock(message="device gone"), "debug")
        pipeline._on_bus_error(None, message)
        pipeline._pipeline.set_state.assert_called_once_with(audio.Gst.State.NULL)
        assert pipeline._pipeline_playing is False

    @pytest.mark.parametrize(
        ("applied", "rebuilt"), [("usb-mic", True), ("auto", False)]
    )
    def test_a_bus_error_looks_a_named_microphone_up_again(self, applied, rebuilt):
        # PulseAudio refuses a stream to an unplugged device instead of moving it to
        # the default, so the next request must not trust the applied name.
        pipeline = self._pipeline(None)
        pipeline._requested_mic_source = pipeline._applied_mic_source = applied
        message = MagicMock()
        message.parse_error.return_value = (MagicMock(message="No such entity"), "")
        with (
            patch.object(
                pipeline, "_resolve_mic_source", return_value="auto"
            ) as resolve,
            patch.object(pipeline, "_rebuild_pipeline") as rebuild,
        ):
            pipeline._on_bus_error(None, message)
            assert pipeline.set_mic_source(applied) is rebuilt
        assert rebuild.call_args_list == ([call("auto")] if rebuilt else [])
        if rebuilt:  # the fallback to the default is logged as a warning
            assert resolve.call_args.kwargs["warn"] is True


class TestMicrophoneRelease:
    """Pulse and auto sources close their stream between recordings."""

    def test_stop_stops_the_source_and_the_next_start_reopens(self, capture):
        # NULL, not PAUSED: a corked Pulse stream keeps the mic indicator lit and the
        # device held, and replays its last few milliseconds on resume.
        from quoth import audio

        rig = capture(_SECOND, stays_playing=False)
        [graph] = rig.graphs
        rig.pipeline.start()
        wait_until(lambda: rig.ended)
        rig.pipeline.stop()
        assert _state(graph) == audio.Gst.State.NULL
        rig.pipeline.start()
        wait_until(lambda: len(rig.ended) == 2)
        # The second take heard a whole second afresh, from the same graph.
        assert np.concatenate(rig.chunks).shape == (32000,)
        assert len(rig.graphs) == 1

    def test_pipewiresrc_stays_playing(self, capture):
        from quoth import audio

        rig = capture(_SECOND, stays_playing=True)
        rig.pipeline.start()
        wait_until(lambda: rig.ended)
        rig.pipeline.stop()
        assert _state(rig.graphs[0]) == audio.Gst.State.PLAYING
        assert len(rig.graphs) == 1


class TestLevelFromDb:
    """Speech RMS maps onto 0..1 over a window calibrated on a laptop mic."""

    def test_floor_and_below_are_silent(self):
        from quoth.audio import LEVEL_FLOOR_DB, level_from_db

        assert level_from_db(LEVEL_FLOOR_DB) == 0.0
        assert level_from_db(-57.0) == 0.0  # measured room tone

    def test_ceiling_and_above_are_full(self):
        from quoth.audio import LEVEL_CEILING_DB, level_from_db

        assert level_from_db(LEVEL_CEILING_DB) == 1.0
        assert level_from_db(0.0) == 1.0

    def test_dictation_lands_mid_scale(self):
        from quoth.audio import level_from_db

        assert 0.5 < level_from_db(-36.0) < 0.7  # measured normal dictation

    def test_digital_silence_and_bad_samples_read_as_silence(self):
        from quoth.audio import level_from_db

        # GStreamer reports digital silence as -700 dB; a NaN sample poisons the RMS.
        assert level_from_db(-700.0) == 0.0
        assert level_from_db(float("nan")) == 0.0
        assert level_from_db(float("inf")) == 0.0

    def test_is_monotonic(self):
        from quoth.audio import level_from_db

        levels = [level_from_db(db) for db in range(-60, 1)]
        assert levels == sorted(levels)


class TestLevelForwarding:
    """A posted level reaches `on_level` only while recording or monitoring."""

    def _pipeline(self, on_level):
        from quoth.audio import AudioPipeline

        pipeline = AudioPipeline()
        pipeline._on_level = on_level
        return pipeline

    def _level_message(self, db: float):
        struct = MagicMock()
        struct.get_name.return_value = "level"
        struct.get_value.return_value = [db]
        message = MagicMock()
        message.get_structure.return_value = struct
        return message

    def test_forwards_while_monitoring_without_recording(self):
        # The Settings meter rides this path; without it the bus handler drops every
        # level and the preview never reacts.
        on_level = MagicMock()
        pipeline = self._pipeline(on_level)
        pipeline._monitoring = True
        pipeline._on_bus_message(None, self._level_message(-10.0))
        on_level.assert_called_once()
        pipeline.destroy()

    def test_forwards_the_rms_not_the_peak(self):
        from quoth.audio import level_from_db

        on_level = MagicMock()
        pipeline = self._pipeline(on_level)
        pipeline._monitoring = True
        message = self._level_message(0.0)
        fields = {"rms": [-36.0], "peak": [-10.0]}
        message.get_structure.return_value.get_value.side_effect = fields.get
        pipeline._on_bus_message(None, message)
        on_level.assert_called_once_with(level_from_db(-36.0))
        pipeline.destroy()

    def test_forwards_while_recording(self):
        on_level = MagicMock()
        pipeline = self._pipeline(on_level)
        pipeline._recording = True
        pipeline._on_bus_message(None, self._level_message(-10.0))
        on_level.assert_called_once()
        pipeline.destroy()

    def test_dropped_when_neither_recording_nor_monitoring(self):
        on_level = MagicMock()
        pipeline = self._pipeline(on_level)
        pipeline._on_bus_message(None, self._level_message(-10.0))
        on_level.assert_not_called()
        pipeline.destroy()


class TestListMicrophones:
    def test_pulse_uses_internal_name_instead_of_display_or_node_name(self):
        from quoth import audio

        microphone = MagicMock()
        microphone.get_properties().get_string.return_value = "wrong-node-name"
        microphone.get_display_name.return_value = "USB microphone"
        microphone.get_property.return_value = "alsa_input.usb"
        microphone.get_device_class.return_value = "Audio/Source"
        pulse = audio._AudioSource("pulsesrc", "device", "", False)
        with patch.object(audio, "_pulse_devices", return_value=[microphone]):
            microphones = audio.list_microphones(pulse)

        assert microphones == [
            {
                "name": "alsa_input.usb",
                "display_name": "USB microphone",
                "device_class": "Audio/Source",
            }
        ]
        microphone.get_property.assert_called_once_with("internal-name")

    def test_pipewire_names_each_input_by_its_node_for_target_object(self):
        from quoth import audio

        def device(display_name, props):
            dev = MagicMock()
            dev.get_display_name.return_value = display_name
            dev.get_properties().get_string.side_effect = props.get
            return dev

        named = device(
            "USB microphone",
            {"node.name": "alsa_input.usb", "media.class": "Audio/Source/Virtual"},
        )
        unnamed = device("Webcam", {})
        monitor = MagicMock()
        monitor.get_devices.return_value = [named, unnamed]
        pipewire = audio._AudioSource("pipewiresrc", "target-object", "", True)
        with patch.object(audio.Gst.DeviceMonitor, "new", return_value=monitor):
            microphones = audio.list_microphones(pipewire)

        assert microphones == [
            {
                "name": "alsa_input.usb",
                "display_name": "USB microphone",
                "device_class": "Audio/Source/Virtual",
            },
            {
                "name": "Webcam",
                "display_name": "Webcam",
                "device_class": "Audio/Source",
            },
        ]
        monitor.add_filter.assert_called_once_with("Audio/Source", None)
        monitor.stop.assert_called_once_with()

    def test_auto_source_exposes_only_the_default_route(self):
        from quoth import audio

        source = audio._AudioSource("autoaudiosrc", None, "", False)
        with patch.object(audio.Gst.DeviceMonitor, "new") as monitor:
            assert audio.list_microphones(source) == []
        monitor.assert_not_called()

    def test_pipeline_enumerates_for_its_lifetime_source(self):
        from quoth import audio

        source = audio._AudioSource("pipewiresrc", "target-object", "", True)
        pipeline = object.__new__(audio.AudioPipeline)
        pipeline._source = source
        with patch.object(audio, "list_microphones", return_value=[]) as enumerate_mics:
            assert pipeline.list_microphones() == []
        enumerate_mics.assert_called_once_with(source)


class TestSourceSelection:
    """Prefer an available Pulse endpoint, otherwise use PipeWire then auto."""

    def test_prefers_pulse_when_its_endpoint_is_available(self):
        from quoth import audio

        with (
            patch.object(audio.Gst.ElementFactory, "find", return_value=object()),
            patch.object(audio, "_pulse_devices", return_value=[]),
        ):
            src = audio._select_source()
        assert src.element == "pulsesrc"
        assert src.stays_playing is False
        assert src.device_prop == "device"

    def test_falls_back_to_pipewire_when_pulse_provider_is_unavailable(self):
        from quoth import audio

        with (
            patch.object(audio.Gst.ElementFactory, "find", return_value=object()),
            patch.object(audio, "_pulse_devices", return_value=None),
        ):
            src = audio._select_source()
        assert src.element == "pipewiresrc"
        assert src.stays_playing is True
        assert src.device_prop == "target-object"

    def test_autoaudiosrc_is_last_resort(self):
        from quoth import audio

        with patch.object(audio.Gst.ElementFactory, "find", return_value=None):
            src = audio._select_source()
        assert src.element == "autoaudiosrc"
        assert src.device_prop is None
        assert src.stays_playing is False


class TestPulseDeviceProvider:
    def _provider(self):
        from quoth import audio

        provider = MagicMock()
        return audio, provider

    def test_returns_empty_devices_and_stops_provider(self):
        audio, provider = self._provider()
        provider.start.return_value = True
        provider.get_devices.return_value = []
        with patch.object(
            audio.Gst.DeviceProviderFactory, "get_by_name", return_value=provider
        ):
            assert audio._pulse_devices() == []
        provider.stop.assert_called_once_with()

    def test_excludes_monitors_and_sinks(self):
        audio, provider = self._provider()

        def device(device_class, pulse_class):
            dev = MagicMock()
            dev.get_device_class.return_value = device_class
            dev.get_properties().get_string.return_value = pulse_class
            return dev

        microphone = device("Audio/Source", "source")
        monitor = device("Audio/Source", "monitor")
        sink = device("Audio/Sink", "sink")
        provider.start.return_value = True
        provider.get_devices.return_value = [microphone, monitor, sink]
        with patch.object(
            audio.Gst.DeviceProviderFactory, "get_by_name", return_value=provider
        ):
            assert audio._pulse_devices() == [microphone]

    def test_stops_provider_when_enumeration_raises(self):
        audio, provider = self._provider()
        provider.start.return_value = True
        provider.get_devices.side_effect = RuntimeError("unexpected enumeration error")
        with (
            patch.object(
                audio.Gst.DeviceProviderFactory, "get_by_name", return_value=provider
            ),
            pytest.raises(RuntimeError, match="unexpected enumeration error"),
        ):
            audio._pulse_devices()
        provider.stop.assert_called_once_with()

    def test_failed_start_does_not_stop_provider(self):
        audio, provider = self._provider()
        provider.start.return_value = False
        with patch.object(
            audio.Gst.DeviceProviderFactory, "get_by_name", return_value=provider
        ):
            assert audio._pulse_devices() is None
        provider.get_devices.assert_not_called()
        provider.stop.assert_not_called()

    def test_missing_provider_is_not_an_available_endpoint(self):
        from quoth import audio

        with patch.object(
            audio.Gst.DeviceProviderFactory, "get_by_name", return_value=None
        ):
            assert audio._pulse_devices() is None


def _built(element):
    """The launch string `AudioPipeline` builds when `element` is the only source."""
    from quoth import audio

    with (
        patch.object(
            audio.Gst.ElementFactory,
            "find",
            side_effect=lambda n: object() if n == element else None,
        ),
        patch.object(audio, "_pulse_devices", return_value=[]),
        patch.object(audio.Gst, "parse_launch", return_value=MagicMock()) as parse,
    ):
        audio.AudioPipeline()
    return parse.call_args.args[0]


class TestPipelineComposition:
    """The pipeline string is built from the selected source."""

    def test_capture_is_whisper_input_format(self):
        # whisper.cpp takes 16 kHz mono float32 samples and never resamples them.
        assert "audio/x-raw,format=F32LE,rate=16000,channels=1 " in _built("pulsesrc")

    def test_pipewire_carries_always_copy(self):
        built = _built("pipewiresrc")
        assert built.startswith("pipewiresrc name=src always-copy=true")
        assert "media.role" not in built
        assert "node.latency" not in built

    def test_pulse_has_no_pipewire_only_props(self):
        built = _built("pulsesrc")
        assert built.startswith("pulsesrc name=src client-name=Quoth")
        assert "always-copy" not in built
        assert "media.role" not in built

    def test_meter_branch_is_leaky_and_starts_silent(self):
        """The branch is leaky so the meter can never stall the appsink branch."""
        built = _built("pipewiresrc")
        assert "level name=level" in built and "leaky=downstream" in built
        assert "post-messages=false" in built

    def test_meter_branch_high_passes_before_the_level(self):
        """Rumble stays off the wave while transcription hears the audio unfiltered."""
        speech, meter = _built("pipewiresrc").split("t. ! ")[1:]
        assert "audiocheblimit" not in speech
        assert meter.index("audiocheblimit mode=high-pass") < meter.index("level name=")


class TestMicSourceSwitch:
    @staticmethod
    def _gst_pipeline(audio, playing=None):
        pipeline = MagicMock()
        source, sink, level, bus = (MagicMock() for _ in range(4))
        pipeline.get_by_name.side_effect = {
            "src": source,
            "sink": sink,
            "level": level,
        }.get
        pipeline.get_bus.return_value = bus
        pipeline.set_state.return_value = (
            audio.Gst.StateChangeReturn.SUCCESS if playing is None else playing
        )
        return pipeline, source, sink, level, bus

    def test_default_aliases_are_unchanged_and_leave_warmup_alone(self):
        from quoth import audio

        pipeline = _stand_in_pipeline()
        pipeline._warmup_timer_id = 41
        with (
            patch.object(pipeline, "_rebuild_pipeline") as rebuild,
            patch.object(audio.GLib, "source_remove") as remove_timer,
        ):
            assert pipeline.set_mic_source(None) is False
            assert pipeline.set_mic_source("auto") is False
        rebuild.assert_not_called()
        remove_timer.assert_not_called()
        assert pipeline._requested_mic_source == "auto"

    def test_saved_missing_microphone_is_applied_when_it_appears(self):
        from quoth import audio

        old, old_source, _, _, _ = self._gst_pipeline(audio)
        candidate, candidate_source, _, _, _ = self._gst_pipeline(audio)
        with (
            patch.object(audio, "_select_source", return_value=_source("pipewiresrc")),
            patch.object(audio.Gst, "parse_launch", side_effect=[old, candidate]),
            patch.object(
                audio,
                "list_microphones",
                side_effect=[[], [{"name": "usb-mic"}]],
            ) as list_mics,
        ):
            pipeline = audio.AudioPipeline(mic_source="usb-mic")
            changed = pipeline.set_mic_source("usb-mic")

        assert changed is True
        assert pipeline._pipeline is candidate
        assert pipeline._applied_mic_source == "usb-mic"
        old_source.set_property.assert_not_called()
        candidate_source.set_property.assert_called_once_with(
            "target-object", "usb-mic"
        )
        assert list_mics.call_count == 2

    def test_a_microphone_is_opened_by_its_name_and_logged_masked(self, caplog):
        # The log masks a Bluetooth address on every line; the device needs it whole.
        from quoth import audio

        headset = "bluez_input.AA_BB_CC_DD_EE_FF.0"
        current, src, _, _, _ = self._gst_pipeline(audio)
        refused, _, _, _, _ = self._gst_pipeline(
            audio, playing=audio.Gst.StateChangeReturn.FAILURE
        )
        with (
            patch.object(audio, "_select_source", return_value=_source("pulsesrc")),
            patch.object(audio.Gst, "parse_launch", return_value=current),
            patch.object(audio, "list_microphones", return_value=[{"name": headset}]),
        ):
            pipeline = audio.AudioPipeline(mic_source=headset)
            pipeline._try_playing(refused, headset)
        with patch.object(audio, "list_microphones", return_value=[]):
            pipeline._resolve_mic_source("bluez_input.11:22:33:44:55:66")

        named = [m for m in caplog.messages if "bluez_input" in m]
        assert named == [
            "Mic device applied: bluez_input.XX_XX_XX_XX_XX_XX.0",
            "Audio pipeline refused to start (pulsesrc, mic "
            "bluez_input.XX_XX_XX_XX_XX_XX.0)",
            "Configured mic 'bluez_input.XX_XX_XX_XX_XX_XX' not available; "
            "using default device",
        ]
        src.set_property.assert_called_once_with("device", headset)

    def test_applied_request_is_a_constant_time_noop(self):
        from quoth import audio

        current, _, _, _, _ = self._gst_pipeline(audio)
        with (
            patch.object(audio, "_select_source", return_value=_source("pipewiresrc")),
            patch.object(audio.Gst, "parse_launch", return_value=current) as parse,
            patch.object(
                audio, "list_microphones", return_value=[{"name": "usb-mic"}]
            ) as list_mics,
        ):
            pipeline = audio.AudioPipeline(mic_source="usb-mic")
            pipeline._warmup_timer_id = 40
            assert pipeline.set_mic_source("usb-mic") is False

        parse.assert_called_once()
        list_mics.assert_called_once_with(pipeline._source)
        assert pipeline._warmup_timer_id == 40

    def test_repeated_missing_request_retries_without_rebuild_or_warmup_reset(
        self, caplog
    ):
        from quoth import audio

        caplog.set_level("WARNING")
        current, _, _, _, _ = self._gst_pipeline(audio)
        with (
            patch.object(audio, "_select_source", return_value=_source("pipewiresrc")),
            patch.object(audio.Gst, "parse_launch", return_value=current) as parse,
            patch.object(audio, "list_microphones", side_effect=[[], []]) as list_mics,
        ):
            pipeline = audio.AudioPipeline(mic_source="usb-mic")
            pipeline._warmup_timer_id = 41
            pipeline._warmup_cycle = 2
            assert pipeline.set_mic_source("usb-mic") is False

        assert pipeline._requested_mic_source == "usb-mic"
        assert pipeline._applied_mic_source == "auto"
        assert list_mics.call_count == 2
        parse.assert_called_once()
        [record] = caplog.records
        assert "'usb-mic' not available" in record.getMessage()
        assert pipeline._warmup_timer_id == 41
        assert pipeline._warmup_cycle == 2

    def test_changed_missing_request_updates_intent_without_replacing_default(
        self, caplog
    ):
        from quoth import audio

        caplog.set_level("WARNING")
        current, _, _, _, _ = self._gst_pipeline(audio)
        with (
            patch.object(audio, "_select_source", return_value=_source("pipewiresrc")),
            patch.object(audio.Gst, "parse_launch", return_value=current) as parse,
            patch.object(audio, "list_microphones", side_effect=[[], []]),
        ):
            pipeline = audio.AudioPipeline(mic_source="missing-a")
            pipeline._warmup_timer_id = 42
            pipeline._warmup_cycle = 2
            assert pipeline.set_mic_source("missing-b") is False

        assert pipeline._requested_mic_source == "missing-b"
        assert pipeline._applied_mic_source == "auto"
        parse.assert_called_once()
        assert len(caplog.records) == 2
        assert pipeline._warmup_timer_id == 42
        assert pipeline._warmup_cycle == 2

    def test_missing_replacement_moves_an_explicit_input_to_default(self):
        from quoth import audio

        old, old_source, _, _, _ = self._gst_pipeline(audio)
        candidate, candidate_source, _, _, _ = self._gst_pipeline(audio)
        with (
            patch.object(audio, "_select_source", return_value=_source("pipewiresrc")),
            patch.object(audio.Gst, "parse_launch", side_effect=[old, candidate]),
            patch.object(
                audio,
                "list_microphones",
                side_effect=[[{"name": "mic-a"}], []],
            ),
        ):
            pipeline = audio.AudioPipeline(mic_source="mic-a")
            pipeline._warmup_timer_id = 43
            pipeline._warmup_cycle = 2
            with patch.object(audio.GLib, "source_remove") as remove_timer:
                changed = pipeline.set_mic_source("missing-b")

        assert changed is True
        assert pipeline._requested_mic_source == "missing-b"
        assert pipeline._applied_mic_source == "auto"
        old_source.set_property.assert_called_once_with("target-object", "mic-a")
        candidate_source.set_property.assert_not_called()
        remove_timer.assert_called_once_with(43)
        assert pipeline._warmup_cycle == 0

    def test_nonselectable_backend_keeps_intent_without_inventory_or_rebuild(self):
        from quoth import audio

        current, _, _, _, _ = self._gst_pipeline(audio)
        with (
            patch.object(audio, "_select_source", return_value=_source("autoaudiosrc")),
            patch.object(audio.Gst, "parse_launch", return_value=current) as parse,
            patch.object(audio, "list_microphones") as list_mics,
        ):
            pipeline = audio.AudioPipeline(mic_source="mic-a")
            pipeline._warmup_timer_id = 44
            pipeline._warmup_cycle = 2
            assert pipeline.set_mic_source("mic-b") is False

        assert pipeline._requested_mic_source == "mic-b"
        assert pipeline._applied_mic_source == "auto"
        list_mics.assert_not_called()
        parse.assert_called_once()
        assert pipeline._warmup_timer_id == 44
        assert pipeline._warmup_cycle == 2

    def test_a_monitored_switch_moves_the_meter_to_the_new_stream(self, capture):
        from quoth import audio

        rig = capture(_LIVE, stays_playing=False)
        rig.pipeline.set_monitoring(True)
        wait_until(lambda: rig.levels)
        with patch.object(rig.pipeline, "_resolve_mic_source", return_value="usb-mic"):
            assert rig.pipeline.set_mic_source("usb-mic") is True
        old, new = rig.graphs
        assert _state(old) == audio.Gst.State.NULL
        rig.levels.clear()
        wait_until(lambda: rig.levels)
        assert _state(new) == audio.Gst.State.PLAYING
        rig.pipeline.set_monitoring(False)  # closing Settings releases the new input
        assert _state(new) == audio.Gst.State.NULL

    def test_refused_candidate_keeps_the_monitored_stream(self, capture):
        from quoth import audio

        rig = capture(_LIVE, stays_playing=False)
        rig.pipeline.set_monitoring(True)
        wait_until(lambda: rig.levels)
        # The replacement's source cannot open, as for a device that vanished.
        rig.pipeline._source = audio._AudioSource(
            "filesrc", None, "location=/nonexistent", False
        )
        with (
            patch.object(rig.pipeline, "_resolve_mic_source", return_value="usb-mic"),
            pytest.raises(audio.AudioUnavailable),
        ):
            rig.pipeline.set_mic_source("usb-mic")
        old, refused = rig.graphs
        assert _state(refused) == audio.Gst.State.NULL
        assert _state(old) == audio.Gst.State.PLAYING
        rig.levels.clear()
        wait_until(lambda: rig.levels)
        rig.pipeline.set_monitoring(False)
        assert _state(old) == audio.Gst.State.NULL

    @pytest.mark.parametrize("refused", [False, True], ids=["unparsable", "refused"])
    def test_a_failed_switch_keeps_the_existing_stream(self, refused):
        # A replacement that does not parse, or that the meter needs live and that
        # refuses to start, leaves the old stream and the warm-up budget as they were.
        from quoth import audio

        old, _, old_sink, old_level, old_bus = self._gst_pipeline(audio)
        unparsable = audio.GLib.Error("candidate parse failed")
        refusing, *_ = self._gst_pipeline(audio, audio.Gst.StateChangeReturn.FAILURE)
        candidate = refusing if refused else unparsable
        with (
            patch.object(audio, "_select_source", return_value=_source("pipewiresrc")),
            patch.object(audio.Gst, "parse_launch", side_effect=[old, candidate]),
            patch.object(audio, "list_microphones", return_value=[{"name": "usb-mic"}]),
        ):
            pipeline = audio.AudioPipeline()
            pipeline.set_monitoring(True)
            pipeline._warmup_timer_id = 44
            pipeline._warmup_cycle = 2
            with pytest.raises(audio.AudioUnavailable if refused else audio.GLib.Error):
                pipeline.set_mic_source("usb-mic")

        assert pipeline._pipeline is old
        assert pipeline._appsink is old_sink
        assert pipeline._level is old_level
        assert pipeline._requested_mic_source == "usb-mic"
        assert pipeline._applied_mic_source == "auto"
        old_bus.remove_signal_watch.assert_not_called()
        assert call(audio.Gst.State.NULL) not in old.set_state.call_args_list
        assert pipeline._warmup_timer_id == 44
        assert pipeline._warmup_cycle == 2


class TestDeviceSelection:
    """A configured mic applies only if present; otherwise warn and use the default."""

    @staticmethod
    def _pipewire():
        from quoth import audio

        return patch.object(
            audio.Gst.ElementFactory,
            "find",
            side_effect=lambda n: object() if n == "pipewiresrc" else None,
        )

    def test_warns_when_configured_mic_absent(self, caplog):
        from quoth import audio

        caplog.set_level("WARNING")
        with (
            self._pipewire(),
            patch.object(audio.Gst, "parse_launch", return_value=MagicMock()),
            patch.object(audio, "list_microphones", return_value=[{"name": "present"}]),
        ):
            audio.AudioPipeline(mic_source="absent")
        [record] = caplog.records
        assert "'absent' not available" in record.getMessage()

    def test_sets_device_when_configured_mic_present(self):
        from quoth import audio

        pipeline, src = MagicMock(), MagicMock()
        pipeline.get_by_name.return_value = src
        with (
            self._pipewire(),
            patch.object(audio.Gst, "parse_launch", return_value=pipeline),
            patch.object(audio, "list_microphones", return_value=[{"name": "present"}]),
        ):
            audio.AudioPipeline(mic_source="present")
        src.set_property.assert_any_call("target-object", "present")

    def test_sets_nondefault_pulse_device_when_present(self):
        from quoth import audio

        pipeline, src = MagicMock(), MagicMock()
        pipeline.get_by_name.return_value = src
        with (
            patch.object(
                audio.Gst.ElementFactory,
                "find",
                side_effect=lambda name: object() if name == "pulsesrc" else None,
            ),
            patch.object(audio, "_pulse_devices", return_value=[]),
            patch.object(
                audio,
                "list_microphones",
                return_value=[{"name": "alsa_input.usb"}],
            ) as list_mics,
            patch.object(audio.Gst, "parse_launch", return_value=pipeline),
        ):
            capture = audio.AudioPipeline(mic_source="alsa_input.usb")
        list_mics.assert_called_once_with(capture._source)
        src.set_property.assert_any_call("device", "alsa_input.usb")


class TestWarmupGating:
    def test_warmup_noops_when_source_does_not_need_it(self, capture):
        from quoth import audio

        rig = capture(_SECOND, stays_playing=False)
        with patch.object(audio.GLib, "timeout_add") as arm:
            rig.pipeline.warm_up()
        arm.assert_not_called()
        assert _state(rig.graphs[0]) == audio.Gst.State.NULL

    def test_rearming_warmup_retires_the_previous_timer(self):
        from quoth import audio

        pipeline = _stand_in_pipeline()
        pipeline._source = audio._AudioSource("pipewiresrc", "target-object", "", True)
        pipeline._warmup_timer_id = 51
        with (
            patch.object(audio.GLib, "source_remove") as remove_timer,
            patch.object(audio.GLib, "timeout_add", return_value=52),
        ):
            pipeline.warm_up()
        remove_timer.assert_called_once_with(51)
        assert pipeline._warmup_timer_id == 52

    @pytest.mark.parametrize("error_type", ["parse", "playing"])
    def test_rebuild_failure_ends_the_bounded_warmup_attempt(self, error_type, caplog):
        from quoth import audio

        pipeline = _stand_in_pipeline()
        old_pipeline = pipeline._pipeline
        pipeline._source = audio._AudioSource("pipewiresrc", "target-object", "", True)
        pipeline._monitoring = True
        failure = (
            audio.GLib.Error("warmup rebuild failed")
            if error_type == "parse"
            else audio.AudioUnavailable("replacement refused")
        )
        with (
            patch.object(pipeline, "_rebuild_pipeline", side_effect=failure),
            patch.object(audio.GLib, "timeout_add") as add_timer,
        ):
            pipeline.warm_up()
            assert add_timer.call_args.args[1]() is False  # the window elapses

        # The refusal was already explained where it happened; a failed build was not.
        assert any(r.exc_info for r in caplog.records) is (error_type == "parse")
        add_timer.assert_called_once()  # and is not armed again
        assert pipeline._pipeline is old_pipeline
        assert pipeline._monitoring is True

    def test_discovery_failure_ends_warmup_without_disturbing_the_live_pipeline(
        self, caplog
    ):
        from quoth import audio

        caplog.set_level("ERROR")
        pipeline = _stand_in_pipeline()
        old_pipeline = pipeline._pipeline
        pipeline._source = audio._AudioSource("pipewiresrc", "target-object", "", True)
        pipeline._requested_mic_source = "usb-mic"
        with (
            patch.object(
                audio,
                "list_microphones",
                side_effect=RuntimeError("provider enumeration failed"),
            ),
            patch.object(pipeline, "_rebuild_pipeline") as rebuild,
            patch.object(audio.GLib, "timeout_add") as arm,
        ):
            pipeline.warm_up()
            assert arm.call_args.args[1]() is False  # the window elapses

        [record] = caplog.records
        assert record.exc_info[0] is RuntimeError
        rebuild.assert_not_called()
        assert pipeline._pipeline is old_pipeline
        assert pipeline._requested_mic_source == "usb-mic"
        assert pipeline._applied_mic_source == "auto"

    def test_stalled_warmup_rebuilds_remain_bounded(self, capture, caplog):
        # Ten buffers are too few to prove any graph warm, however late the window.
        from quoth import audio

        caplog.set_level("WARNING")
        rig = capture(_SECOND, stays_playing=True)
        with patch.object(audio.GLib, "timeout_add") as arm:
            rig.pipeline.warm_up()
            for _ in range(audio.WARMUP_MAX_CYCLES):
                assert arm.call_args.args[1]() is False  # the window elapses

        assert len(rig.graphs) == audio.WARMUP_MAX_CYCLES
        [record] = caplog.records
        assert f"gave up after {audio.WARMUP_MAX_CYCLES} cycles" in record.getMessage()

    def test_a_switch_during_warm_up_gives_the_new_stream_a_full_one(self, capture):
        # The controller warms the new stream up after a microphone change, which may
        # come inside the startup window.
        from quoth import audio

        rig = capture(_SECOND, stays_playing=True)
        rig.pipeline.warm_up()
        with patch.object(rig.pipeline, "_resolve_mic_source", return_value="usb-mic"):
            assert rig.pipeline.set_mic_source("usb-mic") is True
        with patch.object(audio.GLib, "timeout_add") as arm:
            rig.pipeline.warm_up()
            for _ in range(audio.WARMUP_MAX_CYCLES):
                assert arm.call_args.args[1]() is False  # the window elapses

        assert len(rig.graphs) == 1 + audio.WARMUP_MAX_CYCLES

    def test_a_recording_started_in_the_window_is_never_rebuilt(self, capture):
        # A press inside the warm-up window opens the gate on the live stream; a stall
        # rebuild then would cut the take short.
        from quoth import audio

        rig = capture(_SECOND, stays_playing=True)  # ten buffers look like a stall
        with patch.object(audio.GLib, "timeout_add") as arm:
            rig.pipeline.warm_up()
        rig.pipeline.start()
        assert arm.call_args.args[1]() is False  # the window elapses
        assert len(rig.graphs) == 1

    def test_stalled_warmup_resolves_latest_intent_within_current_budget(self):
        from quoth import audio

        old, _, _, _, _ = TestMicSourceSwitch._gst_pipeline(audio)
        candidate, candidate_source, _, _, _ = TestMicSourceSwitch._gst_pipeline(audio)
        with (
            patch.object(audio, "_select_source", return_value=_source("pipewiresrc")),
            patch.object(audio.Gst, "parse_launch", side_effect=[old, candidate]),
            patch.object(
                audio,
                "list_microphones",
                side_effect=[[], [], [{"name": "mic-b"}]],
            ) as list_mics,
        ):
            pipeline = audio.AudioPipeline(mic_source="missing-a")
            assert pipeline.set_mic_source("mic-b") is False
            with patch.object(audio.GLib, "timeout_add", return_value=90) as arm:
                pipeline.warm_up()
                assert arm.call_args.args[1]() is False  # the window elapses

        assert pipeline._requested_mic_source == "mic-b"
        assert pipeline._applied_mic_source == "mic-b"
        candidate_source.set_property.assert_called_once_with("target-object", "mic-b")
        assert list_mics.call_count == 3
        assert pipeline._warmup_cycle == 2
        assert pipeline._warmup_timer_id == 90


class TestStreamProperties:
    """The name and icon the capture stream gives the sound server."""

    # Each source is built by name, not by which sound server the host runs, so both
    # are checked wherever their plugin is installed.
    @pytest.mark.parametrize("element", ["pulsesrc", "pipewiresrc"])
    def test_source_carries_quoth_identity_without_a_role(self, element):
        from quoth import audio

        if audio.Gst.ElementFactory.find(element) is None:
            pytest.skip(f"{element} is not installed here")
        with patch.object(audio, "_select_source", return_value=_source(element)):
            pipeline = audio.AudioPipeline()
        try:
            src = pipeline._pipeline.get_by_name("src")
            # PulseAudio names the client through client-name; PipeWire also takes the
            # name and icon (what the mic indicator shows) as stream properties.
            assert src.get_property("client-name") == "Quoth"
            props = src.get_property("stream-properties")
            # On PulseAudio and WirePlumber 0.4 a call role would switch Bluetooth
            # headphones to the headset profile.
            assert props is None or not props.has_field("media.role")
            if src.get_factory().get_name() == "pipewiresrc":
                assert props.get_string("application.name") == "Quoth"
                assert props.get_string("application.icon-name") == "dev.quoth.Quoth"
                # A latency request would pull the whole graph's quantum down
                # for every app while Quoth idles with its warm pipeline.
                assert not props.has_field("node.latency")
        finally:
            pipeline.destroy()


class TestRealCapture:
    """The real graph, fed by audiotestsrc so it runs without a sound server."""

    def test_a_recording_delivers_16khz_mono_float32(self, capture):
        # One second of a half-scale sine as 48 kHz stereo 16-bit: whisper must get it
        # as 16000 mono float32 samples, whatever format the source produced.
        rig = capture(
            "num-buffers=10 samplesperbuffer=4800 volume=0.5 "
            "! audio/x-raw,format=S16LE,rate=48000,channels=2",
            stays_playing=False,
        )
        rig.pipeline.start()
        wait_until(lambda: rig.ended)
        captured = np.concatenate(rig.chunks)
        assert captured.dtype == np.float32
        assert captured.shape == (16000,)
        assert np.abs(captured).max() == pytest.approx(0.5, abs=0.01)

    def test_warmup_counts_flow_behind_the_closed_gate_and_keeps_the_stream(
        self, capture, caplog
    ):
        # 25 buffers (WARMUP_MIN_BUFFERS) reach the appsink while warm-up holds the gate
        # shut: none may reach the capture buffer, and they prove the stream warm.
        from quoth import audio

        caplog.set_level("INFO")
        rig = capture(
            "num-buffers=25 samplesperbuffer=160 "
            "! audio/x-raw,format=F32LE,rate=16000,channels=1",
            stays_playing=True,
        )
        with patch.object(audio.GLib, "timeout_add") as arm:
            rig.pipeline.warm_up()
        wait_until(lambda: rig.ended)
        assert arm.call_args.args[1]() is False  # the window elapses
        [record] = caplog.records
        assert record.getMessage().startswith("Audio pipeline warm (25 buffers")
        assert len(rig.graphs) == 1
        assert rig.chunks == []

    def test_a_recording_after_an_idle_mic_switch_hears_the_new_stream(self, capture):
        # The switch replaces a stream left PLAYING with a stopped pipeline, which the
        # next recording must start.
        rig = capture(_SECOND, stays_playing=True)
        rig.pipeline.start()
        wait_until(lambda: rig.ended)
        rig.pipeline.stop()
        with patch.object(rig.pipeline, "_resolve_mic_source", return_value="usb-mic"):
            assert rig.pipeline.set_mic_source("usb-mic")
        rig.chunks.clear()
        rig.pipeline.start()
        wait_until(lambda: len(rig.ended) == 2)
        assert np.concatenate(rig.chunks).shape == (16000,)
