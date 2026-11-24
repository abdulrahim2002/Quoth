# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.recording`: the `RecordingController` state machine and flows.

A `FakeScheduler` (synchronous dispatch, inspectable timers) and fake audio and whisper
collaborators run the full lifecycle without GTK, GStreamer or whisper.cpp.
"""

import logging
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from quoth import recording
from quoth.audio import AudioUnavailable
from quoth.calibration import CellResult
from quoth.config import Config
from quoth.recording import (
    ContextLoadOutcome,
    ContextStatus,
    RecordingCallbacks,
    RecordingController,
    RecordingState,
)
from quoth.typer import InjectionResult
from quoth.whisper import ModelLoadFailed, WarmupAborted
from tests.support import (
    DeferredScheduler,
    FakeAudio,
    FakeModelManager,
    FakeScheduler,
    FakeWhisper,
    join_thread,
)


def _config(**overrides):
    config = Config()
    values = {"model": "small", "keep_warm": False, "vad": False} | overrides
    for name, value in values.items():
        setattr(config, name, value)
    return config


def _make(
    *,
    config=None,
    model_mgr=None,
    device="CPU",
    scheduler=None,
    audio_factory=None,
    whisper_factory=FakeWhisper,
    type_text=lambda text: InjectionResult.SUCCESS,
):
    """Build a controller plus a record of callback events."""
    events = {
        "state": [],
        "level": [],
        "error": [],
        "error_calls": [],
        "typing_failed": [],
        "no_input": 0,
        "dictation_success": 0,
        "context": [],
    }

    def _on_error(m, detail=None, essential=False, guidance=None, /):
        events["error"].append(m)  # the terse message alone, for message-only asserts
        events["error_calls"].append((m, detail, essential, guidance))

    cb = RecordingCallbacks(
        on_state_change=lambda s: events["state"].append(s),
        on_level=lambda a: events["level"].append(a),
        on_error=_on_error,
        on_typing_failed=lambda result: events["typing_failed"].append(result),
        on_no_input=lambda: events.__setitem__("no_input", events["no_input"] + 1),
        on_dictation_success=lambda: events.__setitem__(
            "dictation_success", events["dictation_success"] + 1
        ),
        on_context_changed=lambda status: events["context"].append(status),
    )
    sched = scheduler or FakeScheduler()
    audio = FakeAudio()

    def build_audio(**kw):
        audio.mic_source = kw.get("mic_source") or "auto"
        audio.on_audio_buffer = kw.get("on_audio_buffer")
        audio.on_level = kw.get("on_level")
        return audio

    controller = RecordingController(
        config or _config(),
        model_mgr or FakeModelManager(encoder=False),
        device,
        sched,
        cb,
        audio_factory=audio_factory or build_audio,
        whisper_factory=whisper_factory,
        type_text=type_text,
    )
    return controller, sched, audio, events


def _dictate(c, audio):
    """Record a second of audio and wait for its transcription."""
    c.start()
    audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
    c.stop()
    join_thread(c, "_transcribe_thread")


def _calibrate(c, models, devices, measure, on_done=lambda report: None):
    """Start a benchmark of every `models` x `devices` cell, measured by `measure`."""
    return c.calibrate(
        models,
        devices,
        "FP",
        on_progress=lambda *a: None,
        on_done=on_done,
        measure_fn=measure,
    )


class OffloadWhisper(FakeWhisper):
    """A context whose encoder runs through OpenVINO exactly when built for the NPU."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.openvino_active = kwargs["device"] == "NPU"


class TestNoInput:
    """The no-input watchdog fires only when no buffer arrives within its timeout."""

    def test_fires_without_audio(self):
        c, sched, _audio, events = _make()
        c.start()
        [(_id, watchdog, args, delay_ms)] = sched.timers
        assert delay_ms == 2500
        # The no-input window elapses with no buffer. A True return would make GLib
        # repeat the hint every 2.5 s.
        assert watchdog(*args) is False
        assert events["no_input"] == 1

    def test_not_fired_when_audio_arrives(self):
        c, sched, audio, events = _make()
        c.start()
        audio.on_audio_buffer(np.zeros(160, dtype=np.float32))  # first buffer
        sched.fire_all()
        assert events["no_input"] == 0

    def test_timer_cancelled_on_stop(self):
        c, _sched, _audio, _events = _make()
        c.start()
        assert c._no_input_timeout_id is not None
        c.stop()
        assert c._no_input_timeout_id is None

    def test_timer_cancelled_on_cancel(self):
        c, sched, _audio, events = _make()
        c.start()
        c.cancel()
        assert c._no_input_timeout_id is None
        sched.fire_all()
        assert events["no_input"] == 0

    def test_timeout_noop_when_not_recording(self):
        c, _sched, _audio, events = _make()
        c._on_no_input_timeout()  # a stale fire while idle
        assert events["no_input"] == 0

    def test_got_audio_resets_between_recordings(self):
        c, sched, audio, events = _make()
        c.start()
        audio.on_audio_buffer(
            np.zeros(160, dtype=np.float32)
        )  # first recording had audio
        c.cancel()
        c.start()  # second recording, no audio this time
        sched.fire_all()
        assert events["no_input"] == 1


class TestIsIdle:
    """`is_idle` says whether any work holds the engine, so context swaps wait."""

    def test_true_in_idle(self):
        c, _, _, _ = _make()
        assert c.is_idle is True

    def test_false_while_recording(self):
        c, _, _, _ = _make()
        c.start()
        assert c.is_idle is False

    def test_false_while_transcribing(self):
        gate = threading.Event()
        c, _, audio, _ = _make(whisper_factory=FakeWhisper.factory(gate=gate))
        c.start()
        audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
        c.stop()
        assert c.is_idle is False
        gate.set()
        join_thread(c, "_transcribe_thread")


class TestWhenIdle:
    """Work that must not share the engine waits for its holder, then runs once."""

    def test_runs_at_once_when_idle(self):
        c, _, _, _ = _make()
        ran = []
        c.when_idle(lambda: ran.append(True))
        assert ran == [True]

    def test_waits_for_the_dictation_to_end(self):
        c, _, audio, _ = _make()
        ran = []
        c.start()
        c.when_idle(lambda: ran.append(c.state))
        audio.on_audio_buffer(np.zeros(1600, dtype=np.float32))
        assert ran == []
        c.stop()
        join_thread(c, "_transcribe_thread")
        assert ran == [RecordingState.IDLE]

    def test_waits_for_the_benchmark_and_runs_after_its_result(self):
        c, _, _, _ = _make()
        order = []
        gate = threading.Event()

        def measure(model, device, model_mgr):
            gate.wait(5)
            return CellResult(compiled=True, ms=10.0)

        _calibrate(c, ["small"], ["CPU"], measure, lambda report: order.append("done"))
        c.when_idle(lambda: order.append("waiter"))
        assert order == []
        gate.set()
        join_thread(c, "_calibrate_thread")
        assert order == ["done", "waiter"]

    def test_the_same_work_waits_once(self):
        c, _, _, _ = _make()
        ran = []

        def work():
            ran.append(True)

        c.start()
        c.when_idle(work)
        c.when_idle(work)
        c.cancel()
        assert ran == [True]


class TestInputMonitoring:
    """Input meters ride the one shared pipeline but must never look like recording."""

    def test_enables_on_the_backend(self):
        c, _, _, _ = _make()
        c.set_input_monitoring("settings", True)
        assert c._audio.monitoring is True

    def test_disables_again(self):
        c, _, _, _ = _make()
        c.set_input_monitoring("settings", True)
        c.set_input_monitoring("settings", False)
        assert c._audio.monitoring is False

    def test_stays_on_while_another_watcher_remains(self):
        # Settings and onboarding's meter share the pipeline; closing one must not
        # freeze the other.
        c, _, _, _ = _make()
        c.set_input_monitoring("settings", True)
        c.set_input_monitoring("try-it", True)
        c.set_input_monitoring("settings", False)
        assert c._audio.monitoring is True
        c.set_input_monitoring("try-it", False)
        assert c._audio.monitoring is False

    def test_does_not_start_a_recording(self):
        c, _, _, _ = _make()
        c.set_input_monitoring("settings", True)
        assert c.is_idle is True
        assert c._audio.started == 0

    def test_builds_and_primes_the_pipeline_if_it_is_the_first_thing_to_ask(self):
        # Settings or onboarding can meter before the first recording ever happens.
        c, _, audio, _ = _make()
        assert c._audio is None
        c.set_input_monitoring("settings", True)
        assert c._audio is audio
        assert audio.warmed == 1
        c.prepare_audio()
        assert audio.warmed == 1  # activation finds it already up


class TestTake:
    """The guided take runs on the dictation's engine but hands its transcript back."""

    def _take(self, c, audio, *, chunk=True, device="GPU"):
        texts = []
        assert c.begin_take() is True
        if chunk:
            audio.on_audio_buffer(np.ones(1600, dtype=np.float32))
        c.end_take(device, texts.append)
        join_thread(c, "_transcribe_thread")
        return texts

    def test_transcribes_on_the_requested_device_without_pasting(self):
        typed = []
        c, sched, audio, events = _make(
            config=_config(keep_warm=True), type_text=lambda t: typed.append(t)
        )
        assert self._take(c, audio) == ["hello world"]
        assert c._loaded_key == ("small", "GPU")
        sched.fire_all()  # a paste would wait on a timer
        join_thread(c, "_inject_threads")
        assert typed == []
        # The caller shows its own feedback, so no state callback fires.
        assert events["state"] == []
        assert c.is_idle is True

    def test_uses_the_saved_language_without_translating(self):
        captured = {}
        c, _, audio, _ = _make(
            config=_config(language="de", translate=True),
            whisper_factory=FakeWhisper.factory(captured=captured),
        )
        self._take(c, audio)
        assert captured == {"language": "de", "translate": False}

    def test_nothing_captured_yields_empty(self):
        c, _, audio, _ = _make()
        assert self._take(c, audio, chunk=False) == [""]
        assert c.is_idle is True

    def test_holds_the_engine_until_it_lands(self):
        c, _, audio, _ = _make()
        ran = []
        c.begin_take()
        assert c.is_idle is False
        assert c.start() is False
        assert (
            c.calibrate(["small"], ["CPU"], "fp", on_progress=None, on_done=None)
            is False
        )
        c.when_idle(lambda: ran.append(True))
        assert ran == []
        audio.on_audio_buffer(np.ones(1600, dtype=np.float32))
        c.end_take("CPU", lambda _text: ran.append("text"))
        join_thread(c, "_transcribe_thread")
        assert ran == [True, "text"]

    def test_refused_while_busy(self):
        c, _, audio, _ = _make()
        assert c.start() is True
        assert c.begin_take() is False
        assert audio.started == 1

    def test_a_microphone_that_cannot_start_is_refused(self):
        c, _, audio, events = _make()

        def refuse():
            raise AudioUnavailable("the audio pipeline refused to start")

        c._ensure_audio()
        audio.start = refuse
        assert c.begin_take() is False
        assert c.is_idle is True
        assert events["error"] == []

    @pytest.mark.parametrize(
        "failure",
        [
            {"raise_on_init": RuntimeError("the engine would not build")},
            {"raise_on_transcribe": RuntimeError("the decode failed")},
        ],
    )
    def test_a_failed_take_reports_the_engine_failure(self, failure):
        c, _, audio, events = _make(whisper_factory=FakeWhisper.factory(**failure))
        assert self._take(c, audio) == [""]
        assert len(events["error"]) == 1
        assert c.is_idle is True

    @pytest.mark.parametrize("keep_warm", [False, True])
    def test_cancel_drops_the_transcript_and_frees_the_engine(self, keep_warm):
        scheduler = DeferredScheduler()
        c, _, audio, _ = _make(config=_config(keep_warm=keep_warm), scheduler=scheduler)
        texts = []
        c.begin_take()
        audio.on_audio_buffer(np.ones(1600, dtype=np.float32))
        c.end_take("CPU", texts.append)
        join_thread(c, "_transcribe_thread")
        c.cancel()
        assert c.is_idle is True
        scheduler.fire_dispatches()
        assert texts == []
        # The decode loaded the model; the cancel must free it or let it expire.
        assert c.model_loaded() is keep_warm
        scheduler.fire_all()
        assert not c.model_loaded()

    def test_a_mic_change_waits_for_the_take(self):
        config = _config(mic_source="mic_a")
        c, _, audio, _ = _make(config=config)
        c.begin_take()
        config.mic_source = "mic_x"
        assert c.apply_mic_source() is True
        assert audio.mic_source == "mic_a"
        c.end_take("CPU", lambda _text: None)
        assert audio.mic_source == "mic_x"


class TestContextReconcile:
    """The context is keyed on (model, device) and reloads only when the key changes."""

    def test_first_ensure_loads_and_keys(self):
        c, _, _, _ = _make(model_mgr=FakeModelManager(ggml=True))
        c.ensure_context()
        assert c.model_loaded()
        assert c._loaded_key == (c._config.model, c._device)

    def test_same_key_keeps_the_context(self):
        c, _, _, _ = _make(model_mgr=FakeModelManager(ggml=True))
        c.ensure_context()
        first = c._whisper
        c.ensure_context()
        assert c._whisper is first  # not reloaded

    def test_model_change_reloads_on_next_ensure(self):
        cfg = _config(model="small")
        c, _, _, _ = _make(config=cfg, model_mgr=FakeModelManager(ggml=True))
        c.ensure_context()
        first = c._whisper
        cfg.model = "base"  # mutate config, no explicit free_context
        c.ensure_context()
        assert c._whisper is not first
        assert c._loaded_key == ("base", c._device)

    def test_device_change_reloads_on_next_ensure(self):
        c, _, _, _ = _make(model_mgr=FakeModelManager(ggml=True), device="CPU")
        c.ensure_context()
        first = c._whisper
        c.set_device("GPU")  # no explicit free_context
        c.ensure_context()
        assert c._whisper is not first
        assert c._loaded_key == (c._config.model, "GPU")

    def test_the_saved_vad_setting_reaches_the_engine(self):
        c, _, _, _ = _make(
            config=_config(vad=True), model_mgr=FakeModelManager(vad=True)
        )
        c.ensure_context()
        vad_model = c._whisper.init_kwargs["vad_model_path"]
        assert vad_model == "/models/ggml-silero-v6.2.0.bin"


class TestSafeTeardown:
    """`free_context` defers to the next IDLE while inference holds the lock."""

    def test_free_when_idle_closes_immediately(self):
        c, _, _, _ = _make(model_mgr=FakeModelManager(ggml=True))
        c.ensure_context()
        c.free_context()
        assert not c.model_loaded()
        assert c._loaded_key is None

    def test_free_defers_while_inference_holds_lock(self):
        c, _, _, _ = _make(model_mgr=FakeModelManager(ggml=True))
        c.ensure_context()
        c._inference_lock.acquire()  # stand in for an in-flight whisper_full
        try:
            c.free_context()
            assert c._free_pending is True
            assert c.model_loaded()  # not freed mid-inference
        finally:
            c._inference_lock.release()

    def test_idle_transition_reconciles_a_pending_free(self):
        # Keep-warm on, so only the deferred free can unload the model.
        c, _, _, _ = _make(
            config=_config(keep_warm=True), model_mgr=FakeModelManager(ggml=True)
        )
        c.ensure_context()
        c._inference_lock.acquire()
        try:
            c.free_context()  # deferred
            assert c.model_loaded()
        finally:
            c._inference_lock.release()
        c.start()
        c.cancel()  # back to IDLE, which applies the pending free
        assert not c.model_loaded()

    def test_a_free_asked_for_while_a_swap_closes_the_old_context_applies(self):
        # Keep-warm on, so only the deferred free can unload the new context.
        c, _, _, _ = _make(config=_config(keep_warm=True))
        c.ensure_context()
        c._whisper.close = c.free_context  # a free requested while the close runs
        done = threading.Event()
        c.ensure_context_async("base", "CPU", lambda *_result: done.set())
        assert done.wait(2)
        assert not c.model_loaded()


class TestTransitions:
    def test_start_from_idle_enters_recording(self):
        c, _, audio, events = _make()
        assert c.start() is True
        assert c.state is RecordingState.RECORDING
        assert audio.started == 1
        assert events["state"] == [RecordingState.RECORDING]

    @pytest.mark.parametrize(
        "busy", [RecordingState.RECORDING, RecordingState.TRANSCRIBING]
    )
    def test_start_rejected_when_not_idle(self, busy):
        # A start while transcribing would begin a second capture under the decode.
        gate = threading.Event()
        c, _, audio, _ = _make(whisper_factory=FakeWhisper.factory(gate=gate))
        c.start()
        if busy is RecordingState.TRANSCRIBING:
            audio.on_audio_buffer(np.ones(1600, dtype=np.float32))
            c.stop()
        audio.started = 0
        assert c.start() is False
        assert c.state is busy
        assert audio.started == 0  # pipeline not touched again
        gate.set()
        join_thread(c, "_transcribe_thread")

    def test_stop_rejected_when_idle(self):
        c, _, _, _ = _make()
        assert c.stop() is False
        assert c.state is RecordingState.IDLE

    @pytest.mark.parametrize(
        ("failure", "traceback"),
        [(AudioUnavailable("no sound server"), False), (RuntimeError("defect"), True)],
    )
    def test_only_an_unexpected_audio_failure_keeps_a_traceback(
        self, failure, traceback, caplog
    ):
        c, _, audio, events = _make()

        def refuse():
            raise failure

        audio.start = refuse
        assert c.start() is False
        assert c.state is RecordingState.IDLE
        assert events["error"] == ["Microphone unavailable"]
        assert any(record.exc_info for record in caplog.records) is traceback

    def test_an_unavailable_microphone_is_an_essential_error(self):
        # It blocks every dictation until fixed, so its guidance must outlive the pill.
        c, _, audio, events = _make()

        def refuse():
            raise AudioUnavailable("no sound server")

        audio.start = refuse
        assert c.start() is False
        assert events["error_calls"] == [
            ("Microphone unavailable", None, True, "Check microphone in Settings")
        ]

    def test_audio_failure_returns_to_idle_and_errors(self):
        def boom(**kw):
            raise RuntimeError("no mic")

        c, _sched, _audio, events = _make(
            config=_config(keep_warm=True), audio_factory=boom
        )
        assert c.start() is False
        assert c.state is RecordingState.IDLE
        assert events["error"] == ["Microphone unavailable"]
        assert events["error_calls"][0][3] == "Check microphone in Settings"
        # start() cancelled the keep-warm timer; a failed start must re-arm it, or the
        # loaded model would stay in memory until the next successful dictation.
        assert c._warm_timeout_id is not None


class TestModelGate:
    """`start` refuses to record when no model could produce a transcription."""

    def test_model_loaded_false_without_context(self):
        c, _, _, _ = _make()
        assert c.model_loaded() is False  # lazy, no context until transcribe

    def test_model_available_false_when_no_model_and_no_context(self):
        c, _, _, _ = _make(model_mgr=FakeModelManager(ggml=False))
        assert c.model_available() is False

    def test_start_refused_when_no_model(self):
        c, _, audio, events = _make(model_mgr=FakeModelManager(ggml=False))
        assert c.start() is False
        assert c.state is RecordingState.IDLE  # never entered RECORDING
        assert events["state"] == []
        assert audio.started == 0  # audio pipeline never touched
        assert events["error"] == ["No model installed"]  # terse pill message
        assert events["error_calls"][-1] == (
            "No model installed",
            "No model installed. Open Settings to download one.",
            True,  # essential → also raises a notification
            "Download in Settings",
        )

    def test_start_allowed_when_context_loaded_despite_missing_file(self):
        # A warm session keeps working even if the model file was deleted on disk:
        # the loaded context can still transcribe, so the gate must not refuse it.
        c, _, _, _ = _make(model_mgr=FakeModelManager(ggml=False))
        c._whisper = FakeWhisper()
        assert c.model_available() is True
        assert c.start() is True
        assert c.state is RecordingState.RECORDING

    @pytest.mark.parametrize(("model", "started"), [("small", True), ("base", False)])
    def test_start_needs_the_configured_model_on_disk(self, model, started):
        mgr = FakeModelManager(ggml={"small"})
        c, _, _, _ = _make(config=_config(model=model), model_mgr=mgr)
        assert c.start() is started

    def test_start_refuses_unsupported_translation_before_capture(self):
        c, _, audio, events = _make(
            config=_config(model="large-v3-turbo", translate=True)
        )
        assert c.start() is False
        assert c.state is RecordingState.IDLE
        assert audio.started == 0
        assert events["state"] == []
        assert events["error_calls"][-1] == (
            "Translation unavailable",
            "Turbo cannot translate to English. Turn off translation or "
            "choose a different model.",
            True,
            "Change model or translation",
        )


class TestMicSource:
    def test_a_pipeline_that_cannot_be_built_leaves_the_recording_to_retry(self):
        c, _, _, events = _make()

        def missing_plugin(**_kw):
            raise RuntimeError("no element pipewiresrc")

        c._audio_factory = missing_plugin
        c.prepare_audio()  # must not stop activation
        assert c.start() is False
        assert events["error"] == ["Microphone unavailable"]

    def test_apply_rebuilds_and_rewarms_when_idle(self):
        config = _config(mic_source="mic_a")
        c, _, audio, _ = _make(config=config)
        c.prepare_audio()
        config.mic_source = "mic_x"
        assert c.apply_mic_source() is True
        assert audio.mic_source == "mic_x"
        assert audio.warmed == 2

    def test_same_selection_does_not_rewarm(self):
        c, _, audio, _ = _make(config=_config(mic_source="mic_a"))
        c.prepare_audio()
        assert c.apply_mic_source() is True
        assert c.start() is True
        c.cancel()
        assert audio.warmed == 1

    @pytest.mark.parametrize(
        "outcome",
        ["cancel-recording", "cancel-transcribing", "success", "empty", "error"],
    )
    def test_busy_changes_apply_latest_selection_on_idle(self, outcome):
        config = _config(mic_source="mic_a")
        gate = threading.Event()
        c, _, audio, _ = _make(
            config=config,
            whisper_factory=FakeWhisper.factory(
                gate=gate,
                raise_on_transcribe=RuntimeError("decode failed")
                if outcome == "error"
                else None,
            ),
        )
        try:
            c.start()
            config.mic_source = "mic_b"
            c.apply_mic_source()
            assert audio.mic_source == "mic_a"
            if outcome == "cancel-recording":
                c.cancel()
            elif outcome == "empty":
                c.stop()
            else:
                audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
                c.stop()
                config.mic_source = "mic_c"
                c.apply_mic_source()
                assert audio.mic_source == "mic_a"
                if outcome == "cancel-transcribing":
                    c.cancel()
                gate.set()
                join_thread(c, "_transcribe_thread")
            assert c.state is RecordingState.IDLE
            assert audio.mic_source == config.mic_source
            assert audio.warmed == 2  # primed when built, then once for the new input
            assert c.start() is True
            assert audio.mic_source == config.mic_source
        finally:
            gate.set()
            c.shutdown()

    def test_failed_change_preserves_completion_but_blocks_next_capture(
        self, monkeypatch
    ):
        config = _config(mic_source="mic_a")
        typed = []
        c, sched, audio, events = _make(
            config=config,
            type_text=lambda text: typed.append(text) or InjectionResult.SUCCESS,
        )
        c.start()
        audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
        config.mic_source = "mic_b"

        def reject(_source):
            raise AudioUnavailable("device refused to start")

        monkeypatch.setattr(audio, "set_mic_source", reject)
        c.stop()
        join_thread(c, "_transcribe_thread")
        assert c.state is RecordingState.IDLE
        # It blocks the next dictation, so its guidance must outlive the pill.
        assert events["error_calls"] == [
            ("Microphone change failed", None, True, "Check microphone in Settings")
        ]
        sched.fire_all()
        join_thread(c, "_inject_threads")
        assert typed == ["hello world "]
        assert config.mic_source == "mic_b"
        assert audio.mic_source == "mic_a"
        assert c.start() is False
        assert c.state is RecordingState.IDLE
        assert audio.started == 1
        c.shutdown()

    def test_discovery_uses_one_lazy_pipeline_and_current_devices(self):
        c, _, audio, _ = _make(config=_config(mic_source="mic_a"))
        audio.microphones = [{"name": "mic_a", "display_name": "First mic"}]
        assert c.list_microphones() == audio.microphones
        first_pipeline = c._audio
        audio.microphones = [{"name": "mic_b", "display_name": "Second mic"}]
        assert c.list_microphones() == audio.microphones
        assert c._audio is first_pipeline
        assert audio.started == 0


class TestRecordingFlow:
    def test_empty_buffer_finishes_without_thread(self):
        c, _sched, audio, events = _make()
        c.start()
        assert c.stop() is True  # no audio captured
        assert c.state is RecordingState.IDLE
        assert audio.stopped == 1
        assert events["state"][-1] is RecordingState.IDLE

    def test_full_flow_transcribes_and_types(self):
        typed = []
        c, sched, audio, _events = _make(
            type_text=lambda t: typed.append(t) or InjectionResult.SUCCESS
        )
        assert c.last_text is None
        _dictate(c, audio)
        assert c.state is RecordingState.IDLE
        # Without the ~50 ms delay the paste chord's key events can be dropped.
        [(_id, paste, args, delay_ms)] = sched.timers
        assert delay_ms == 50
        assert paste(*args) is False  # a True return would make GLib paste again
        join_thread(c, "_inject_threads")  # injection runs on a worker thread
        assert typed == ["hello world "]
        assert c.last_text == "hello world"

    @pytest.mark.parametrize(
        ("result", "pasted"),
        [
            ("Hello world.", "Hello world. "),
            ("你好。", "你好。"),
            ("こんにちは", "こんにちは"),
            ("Quoth 支持 Linux。", "Quoth 支持 Linux。"),
        ],
    )
    def test_a_space_follows_only_text_written_with_spaces(self, result, pasted):
        typed = []
        c, sched, audio, _events = _make(
            whisper_factory=FakeWhisper.factory(result=result),
            type_text=lambda t: typed.append(t) or InjectionResult.SUCCESS,
        )
        _dictate(c, audio)
        sched.fire_all()
        join_thread(c, "_inject_threads")
        assert typed == [pasted]

    def test_passes_language_and_translate_to_whisper(self):
        captured = {}
        c, _sched, audio, _ = _make(
            config=_config(language="de", translate=True),
            whisper_factory=FakeWhisper.factory(captured=captured),
        )
        _dictate(c, audio)
        assert captured == {"language": "de", "translate": True}

    def test_passes_the_captured_audio_to_whisper(self):
        decoded = []

        class KeepsAudio(FakeWhisper):
            def transcribe_buffer(self, audio, language="auto", translate=False):
                decoded.append(audio)
                return super().transcribe_buffer(audio, language, translate)

        c, _, audio, _ = _make(whisper_factory=KeepsAudio)
        first = np.full(1600, 0.25, dtype=np.float32)
        second = np.full(3200, -0.5, dtype=np.float32)
        c.prepare_audio()
        audio.on_audio_buffer(np.ones(10, dtype=np.float32))  # landed before start
        c.start()
        audio.on_audio_buffer(first)
        audio.on_audio_buffer(second)
        c.stop()
        join_thread(c, "_transcribe_thread")
        [heard] = decoded
        np.testing.assert_array_equal(heard, np.concatenate([first, second]))

    def test_dictation_keeps_settings_captured_at_start(self):
        captured = {}
        config = _config(
            model="small",
            language="de",
            translate=True,
            keep_warm=True,
        )
        c, _sched, audio, _ = _make(
            config=config,
            device="CPU",
            whisper_factory=FakeWhisper.factory(captured=captured),
        )
        c.start()
        config.model = "large-v3-turbo"
        config.language = "it"
        config.translate = False
        c.set_device("GPU")
        audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
        c.stop()
        join_thread(c, "_transcribe_thread")
        assert captured == {"language": "de", "translate": True}
        assert c._loaded_key == ("small", "CPU")

    def test_enabling_translation_mid_turbo_recording_applies_next_time(self):
        captured = {}
        config = _config(model="large-v3-turbo", translate=False)
        c, _sched, audio, _ = _make(
            config=config,
            whisper_factory=FakeWhisper.factory(captured=captured),
        )
        assert c.start() is True
        config.translate = True
        audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
        c.stop()
        join_thread(c, "_transcribe_thread")
        assert captured["translate"] is False
        assert c.start() is False

    @pytest.mark.parametrize(
        "result",
        [InjectionResult.CLIPBOARD_ONLY, InjectionResult.UINPUT_SETUP_REQUIRED],
    )
    def test_type_failure_signals_typing_failed(self, result, caplog):
        c, sched, audio, events = _make(type_text=lambda t: result)
        _dictate(c, audio)
        sched.fire_all()
        join_thread(c, "_inject_threads")  # injection runs on a worker thread
        assert events["typing_failed"] == [result]
        assert f"Paste did not land ({result.value})" in caplog.messages
        assert events["dictation_success"] == 0
        assert (
            events["error"] == []
        )  # UI routing is the app's job, not the state machine

    def test_successful_paste_signals_dictation_success(self):
        c, sched, audio, events = _make(type_text=lambda t: InjectionResult.SUCCESS)
        _dictate(c, audio)
        sched.fire_all()
        join_thread(c, "_inject_threads")  # injection runs on a worker thread
        assert events["dictation_success"] == 1
        assert events["typing_failed"] == []
        # Named, so the log file's thread column tells the workers apart.
        assert c._transcribe_thread.name == "decode"
        assert [t.name for t in c._inject_threads] == ["inject"]

    def test_injection_exception_signals_plain_failure(self):
        def fail(_text):
            raise RuntimeError("wl-copy crashed")

        c, sched, audio, events = _make(type_text=fail)
        _dictate(c, audio)
        sched.fire_all()
        join_thread(c, "_inject_threads")
        assert events["typing_failed"] == [InjectionResult.FAILED]

    def test_shutdown_joins_every_pending_paste(self):
        # A second dictation's paste waits behind the first (typer serialises them);
        # shutdown must join both, not only the most recent thread.
        release = threading.Event()
        typed = []

        def slow_type(text):
            if text == "first":
                release.wait(2)
            typed.append(text)
            return InjectionResult.SUCCESS

        c, _, _, _ = _make(type_text=slow_type)
        c._inject("first")
        c._inject("second")
        assert len(c._inject_threads) == 2
        # The first paste outlasts the second.
        threading.Timer(0.15, release.set).start()
        c._join_workers(2.0)
        assert sorted(typed) == ["first", "second"]

    def test_transcription_error_is_surfaced(self):
        c, _, audio, events = _make(
            whisper_factory=FakeWhisper.factory(
                raise_on_transcribe=RuntimeError("whisper crashed")
            )
        )
        _dictate(c, audio)
        assert "Transcription failed" in events["error"]
        assert events["error_calls"][-1] == (
            "Transcription failed",
            None,
            False,
            "Try dictating again",
        )
        assert c.state is RecordingState.IDLE

    def test_level_forwarded_to_callback(self):
        scheduler = DeferredScheduler()
        c, _, audio, events = _make(scheduler=scheduler)
        c.start()
        audio.on_level(0.5)
        assert events["level"] == []  # the GStreamer thread must not touch the UI
        scheduler.fire_dispatches()
        assert events["level"] == [0.5]

    def test_a_failed_load_reports_on_the_main_loop(self):
        # The decode worker must not call into the UI itself.
        scheduler = DeferredScheduler()
        c, _, audio, events = _make(
            scheduler=scheduler,
            whisper_factory=FakeWhisper.factory(raise_on_init=RuntimeError("boom")),
        )
        _dictate(c, audio)
        assert events["error"] == []
        scheduler.fire_dispatches()
        assert events["error"] == ["Speech model failed to load"]

    def test_context_load_failure_reports_a_model_load_failure(self):
        c, _, audio, events = _make(
            whisper_factory=FakeWhisper.factory(
                raise_on_init=RuntimeError("CPU backend did not register")
            )
        )  # is_downloaded=True by default
        _dictate(c, audio)
        assert "Speech model failed to load" in events["error"]
        assert events["error_calls"][-1] == (
            "Speech model failed to load",
            "The speech model couldn't load. Delete and re-download the model in "
            "Settings. If it still fails, reinstall Quoth, then restart it.",
            True,
            "Re-download in Settings",
        )


class TestNoSpeechLog:
    """The log says "No speech detected" only for a finished decode that heard none."""

    def test_an_empty_decode_says_so(self, caplog):
        c, _, audio, _ = _make(whisper_factory=FakeWhisper.factory(result=""))
        _dictate(c, audio)
        [said] = [r for r in caplog.records if r.getMessage() == "No speech detected"]
        assert said.levelname == "INFO"
        # The captured level, numbers only, tells silence from quiet speech.
        assert "Captured peak 1.000, RMS 0.0 dBFS" in caplog.messages

    @pytest.mark.parametrize(
        "factory",
        [
            FakeWhisper.factory(raise_on_transcribe=RuntimeError("decode failed")),
            FakeWhisper.factory(raise_on_init=RuntimeError("bad model")),
        ],
        ids=["transcription failed", "load failed"],
    )
    def test_a_failed_decode_does_not(self, caplog, factory):
        c, _, audio, _ = _make(whisper_factory=factory)
        _dictate(c, audio)
        assert "No speech detected" not in caplog.messages

    def test_a_recording_with_no_audio_does_not(self, caplog):
        c, _, _, _ = _make(whisper_factory=FakeWhisper.factory(result=""))
        c.start()
        c.stop()
        join_thread(c, "_transcribe_thread")
        assert "No speech detected" not in caplog.messages

    def test_a_cancelled_decode_does_not(self, caplog):
        entered, gate = threading.Event(), threading.Event()
        factory = FakeWhisper.factory(result="", gate=gate, entered=entered)
        c, _, audio, _ = _make(whisper_factory=factory)
        c.start()
        audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
        c.stop()
        assert entered.wait(2)
        c.cancel()
        gate.set()
        join_thread(c, "_transcribe_thread")
        assert "No speech detected" not in caplog.messages

    def test_a_decode_cut_short_by_shutdown_does_not(self, caplog):
        entered, gate = threading.Event(), threading.Event()

        class Aborting(FakeWhisper):
            def cancel(self):
                gate.set()  # an abort ends the decode with empty text

        factory = Aborting.factory(result="", gate=gate, entered=entered)
        c, _, audio, _ = _make(whisper_factory=factory)
        c.start()
        audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
        c.stop()
        assert entered.wait(2)
        c.shutdown()
        join_thread(c, "_transcribe_thread")  # the abort ended the decode
        assert "No speech detected" not in caplog.messages

    def test_a_missing_context_is_not_logged_again_as_an_error(self, caplog):
        caplog.set_level("ERROR")
        c, _, audio, _ = _make(
            whisper_factory=FakeWhisper.factory(raise_on_init=RuntimeError("bad"))
        )
        _dictate(c, audio)
        assert "No whisper context available" not in caplog.messages


class TestTranscriptPrivacy:
    @pytest.mark.parametrize("pasted", [True, False])
    def test_the_transcript_is_logged_only_at_trace(self, caplog, pasted):
        # Transcript text is logged only at TRACE, which no sink writes to disk.
        result = InjectionResult.SUCCESS if pasted else InjectionResult.CLIPBOARD_ONLY
        c, sched, audio, _ = _make(
            whisper_factory=FakeWhisper.factory(result="sentinel dictation"),
            type_text=lambda _text: result,
        )
        _dictate(c, audio)
        sched.fire_all()  # the paste
        join_thread(c, "_inject_threads")
        heard = {r.levelname for r in caplog.records if "sentinel" in r.getMessage()}
        assert heard == {"TRACE"}


class TestPrepareAudio:
    def test_prepare_creates_and_warms_pipeline(self):
        c, _, audio, _ = _make()
        c.prepare_audio()
        assert c._audio is audio
        assert audio.warmed == 1
        assert audio.started == 0  # warming is not recording

    def test_start_after_prepare_reuses_pipeline(self):
        c, _, audio, _ = _make()
        c.prepare_audio()
        assert c.start() is True
        assert audio.warmed == 1  # not warmed again
        assert audio.started == 1


class TestCancel:
    def test_cancel_recording_resets_and_clears(self):
        c, _, audio, _ = _make()
        c.start()
        audio.on_audio_buffer(np.ones(10, dtype=np.float32))
        c.cancel()
        assert c.state is RecordingState.IDLE
        assert audio.stopped == 1
        with c._buffer_lock:
            assert c._buffer == []

    def test_cancel_stops_the_running_decode(self):
        entered, gate = threading.Event(), threading.Event()
        c, _, audio, _ = _make(
            whisper_factory=FakeWhisper.factory(gate=gate, entered=entered)
        )
        c._ensure_context()
        whisper = c._whisper
        c.start()
        audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
        c.stop()
        assert entered.wait(2)
        c.cancel()
        gate.set()
        join_thread(c, "_transcribe_thread")
        assert whisper.cancelled == 1

    def test_cancel_while_idle_leaves_the_engine_alone(self):
        c, _, _, _ = _make()
        c._ensure_context()
        c.cancel()
        assert c._whisper.cancelled == 0

    def test_shutdown_cancels_whatever_holds_the_engine(self):
        c, _, _, _ = _make()
        c._ensure_context()
        whisper = c._whisper
        c.shutdown()
        assert whisper.cancelled == 1

    def test_cancel_while_recording_rearms_keep_warm(self):
        c, sched, _, _ = _make(
            config=_config(keep_warm=True, keep_warm_timeout=300),
            model_mgr=FakeModelManager(ggml=True),
        )
        c.ensure_context()
        c.start()
        c.cancel()
        assert c.model_loaded()
        sched.fire_all()  # the keep-warm expiry
        assert not c.model_loaded()

    def test_cancel_while_transcribing_frees_once_the_decode_exits(self):
        # Keep-warm off: the cancel's free is deferred while the decode holds the lock,
        # and the engine is already idle, so the worker must apply it.
        entered, gate = threading.Event(), threading.Event()
        c, _, audio, _ = _make(
            config=_config(keep_warm=False),
            whisper_factory=FakeWhisper.factory(gate=gate, entered=entered),
        )
        c._ensure_context()
        c.start()
        audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
        c.stop()
        assert entered.wait(2)
        c.cancel()
        assert c.model_loaded()
        gate.set()
        join_thread(c, "_transcribe_thread")
        assert not c.model_loaded()
        assert c._free_pending is False

    def test_cancel_during_the_model_load_skips_the_decode(self):
        loading, release = threading.Event(), threading.Event()
        built = []

        def slow_load(*args, **kwargs):
            loading.set()
            release.wait(2)
            built.append(FakeWhisper(*args, **kwargs))
            return built[-1]

        c, _, audio, _ = _make(whisper_factory=slow_load)
        c.start()
        audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
        c.stop()
        assert loading.wait(2)
        c.cancel()
        release.set()
        join_thread(c, "_transcribe_thread")
        assert [w.calls for w in built] == [0]


class TestKeepWarm:
    def test_keep_warm_schedules_timer(self):
        c, sched, _, _ = _make(config=_config(keep_warm=True, keep_warm_timeout=300))
        c._schedule_warm_timeout()
        assert len(sched.timers) == 1

    def test_disable_keep_warm_frees_and_cancels(self):
        cfg = _config(keep_warm=True)
        c, sched, _, _ = _make(config=cfg)
        c._whisper = FakeWhisper()
        c._schedule_warm_timeout()
        cfg.keep_warm = False
        c.apply_keep_warm()
        assert c._whisper is None
        assert sched.timers == []

    def test_turning_keep_warm_on_arms_the_timer(self):
        cfg = _config(keep_warm=False, keep_warm_timeout=300)
        c, sched, _, _ = _make(config=cfg)
        cfg.keep_warm = True
        c.apply_keep_warm()
        assert [t[3] for t in sched.timers] == [300_000]

    def test_a_new_timeout_rearms_a_pending_timer(self):
        cfg = _config(keep_warm=True, keep_warm_timeout=300)
        c, sched, _, _ = _make(config=cfg)
        c.apply_keep_warm()
        cfg.keep_warm_timeout = 60
        c.reschedule_warm_timeout()
        assert [t[3] for t in sched.timers] == [60_000]

    def test_a_new_timeout_arms_nothing_while_none_is_pending(self):
        cfg = _config(keep_warm=True, keep_warm_timeout=300)
        c, sched, _, _ = _make(config=cfg)
        cfg.keep_warm_timeout = 60
        c.reschedule_warm_timeout()
        assert sched.timers == []

    def test_a_finished_take_arms_the_timer(self):
        c, sched, audio, _ = _make(config=_config(keep_warm=True))
        assert c.begin_take() is True
        audio.on_audio_buffer(np.ones(1600, dtype=np.float32))
        c.end_take("CPU", lambda _text: None)
        join_thread(c, "_transcribe_thread")
        assert c.model_loaded()
        sched.fire_all()  # the keep-warm expiry
        assert not c.model_loaded()

    def test_a_pasted_dictation_arms_the_timer(self):
        c, sched, audio, _ = _make(
            config=_config(keep_warm=True, keep_warm_timeout=300)
        )
        _dictate(c, audio)
        sched.fire_all()  # the paste
        join_thread(c, "_inject_threads")
        assert [t[3] for t in sched.timers] == [300_000]
        assert c.model_loaded()

    def test_a_dictation_that_heard_nothing_arms_the_timer(self):
        c, sched, audio, _ = _make(
            config=_config(keep_warm=True, keep_warm_timeout=300),
            whisper_factory=FakeWhisper.factory(result=""),
        )
        _dictate(c, audio)
        assert [t[3] for t in sched.timers] == [300_000]
        assert c.model_loaded()

    def test_a_paste_landing_during_the_next_recording_keeps_its_context(self):
        # With keep-warm off a paste frees the context, unless a new recording that is
        # about to use it has started meanwhile.
        c, sched, audio, _ = _make(
            config=_config(keep_warm=False),
            type_text=lambda _t: InjectionResult.SUCCESS,
        )
        _dictate(c, audio)
        c.start()
        sched.fire_all()
        join_thread(c, "_inject_threads")
        assert c._whisper is not None

    def test_warm_expiry_frees_context(self):
        c, sched, _, _ = _make(config=_config(keep_warm=True))
        c.ensure_context()
        c.apply_keep_warm()
        sched.fire_all()
        assert not c.model_loaded()

    def test_an_expiry_while_the_engine_is_in_use_waits_a_round(self):
        # Say a slow NPU compile replacing the context; a deferred free would lose it.
        c, sched, _, _ = _make(config=_config(keep_warm=True))
        c.ensure_context()
        c._schedule_warm_timeout()
        armed = c._warm_timeout_id
        with c._inference_lock:
            sched.fire_all()
        assert c.model_loaded()
        assert not c._free_pending
        # The same timer stays armed, so a recording start still cancels it.
        assert c._warm_timeout_id == armed
        assert [timer[0] for timer in sched.timers] == [armed]

        sched.fire_all()
        assert not c.model_loaded()
        assert sched.timers == []
        assert c._warm_timeout_id is None
        assert not c._inference_lock.locked()

    def test_reload_cancels_a_stale_warm_timer(self):
        # A keep-warm timer armed for a prior (model, device) must not survive a swap
        # and free the freshly-reloaded context.
        c, sched, _, _ = _make(
            config=_config(keep_warm=True, keep_warm_timeout=300),
            model_mgr=FakeModelManager(ggml=True),
            device="CPU",
        )
        c.ensure_context()
        c._schedule_warm_timeout()  # a keep-warm expiry is now pending
        assert len(sched.timers) == 1
        c.set_device("GPU")  # key now mismatches → next ensure reloads
        c.ensure_context()
        assert sched.timers == []  # stale timer cancelled; won't free the reloaded ctx
        assert c._loaded_key == (c._config.model, "GPU")  # and the reload happened


class _CancelThreads(DeferredScheduler):
    """A `DeferredScheduler` that notes the thread of every timer cancel."""

    def __init__(self):
        super().__init__()
        self.cancelled_on: list[threading.Thread] = []

    def cancel(self, source_id):
        self.cancelled_on.append(threading.current_thread())
        super().cancel(source_id)


def _paused_reconcile(c, monkeypatch):
    """Hold the decode worker after its epoch check, before it reconciles the context.

    Return the events for "reached" and "go on".
    """
    reached, go = threading.Event(), threading.Event()
    reconcile = c._ensure_context

    def paused(desired=None):
        reached.set()
        go.wait(2)
        return reconcile(desired)

    monkeypatch.setattr(c, "_ensure_context", paused)
    return reached, go


class TestKeepWarmThreads:
    """The main thread alone arms and cancels the keep-warm timer."""

    @pytest.mark.parametrize("work", ["load", "decode"])
    def test_busy_work_stops_the_timer_from_the_main_thread(self, work):
        # A pending expiry would free the context the work is about to use.
        scheduler = _CancelThreads()
        release = threading.Event()

        class Blocks(FakeWhisper):
            def __init__(self, *args, **kwargs):
                release.wait(2)
                super().__init__(*args, **kwargs)

        c, _, audio, _ = _make(
            config=_config(keep_warm=True, keep_warm_timeout=300),
            scheduler=scheduler,
            whisper_factory=Blocks,
        )
        c._schedule_warm_timeout()
        if work == "load":
            c.ensure_context_async("small", "CPU", lambda *_result: None)
            worker = "_context_threads"
        else:
            assert c.begin_take()
            audio.on_audio_buffer(np.ones(1600, dtype=np.float32))
            c.end_take("CPU", lambda _text: None)
            worker = "_transcribe_thread"
        assert scheduler.timers == []  # none can expire while the work runs
        release.set()
        join_thread(c, worker)
        scheduler.fire_dispatches()
        assert c.model_loaded()
        assert set(scheduler.cancelled_on) == {threading.main_thread()}

    @pytest.mark.parametrize("take", [False, True])
    def test_a_cancel_during_the_decode_leaves_the_timer_armed(self, monkeypatch, take):
        # The cancel lands after the worker's epoch check; the discarded result arms
        # nothing, so the timer the cancel armed must outlive the worker.
        scheduler = DeferredScheduler()
        c, _, audio, _ = _make(
            config=_config(keep_warm=True, keep_warm_timeout=300),
            scheduler=scheduler,
        )
        c.ensure_context()  # loaded, so the start spawns no load of its own
        reached, go = _paused_reconcile(c, monkeypatch)
        if take:
            assert c.begin_take()
            audio.on_audio_buffer(np.ones(1600, dtype=np.float32))
            c.end_take("CPU", lambda _text: None)
        else:
            c.start()
            audio.on_audio_buffer(np.ones(1600, dtype=np.float32))
            c.stop()
        assert reached.wait(2)
        c.cancel()
        go.set()
        join_thread(c, "_transcribe_thread")
        scheduler.fire_dispatches()
        assert len(scheduler.timers) == 1
        scheduler.fire_all()
        assert not c.model_loaded()

    @pytest.mark.parametrize("ending", ["end", "cancel"])
    def test_a_take_stops_the_timer_until_it_ends(self, ending):
        c, sched, audio, _ = _make(
            config=_config(keep_warm=True, keep_warm_timeout=300)
        )
        c.ensure_context()
        c._schedule_warm_timeout()
        assert c.begin_take()
        assert sched.timers == []  # no expiry frees the context mid-take
        audio.on_audio_buffer(np.ones(1600, dtype=np.float32))
        if ending == "end":
            c.end_take("CPU", lambda _text: None)
            join_thread(c, "_transcribe_thread")
        else:
            c.cancel()
        assert [timer[3] for timer in sched.timers] == [300_000]

    def test_a_timer_armed_before_a_load_cannot_free_it(self):
        # The load cancels the timer before starting, so no expiry frees what it builds.
        scheduler = DeferredScheduler()
        c, _, _, _ = _make(
            config=_config(keep_warm=True, keep_warm_timeout=300),
            scheduler=scheduler,
        )
        c._schedule_warm_timeout()
        c.ensure_context_async("small", "CPU", lambda *_result: None)
        join_thread(c, "_context_threads")
        scheduler.fire_all()
        scheduler.fire_dispatches()
        assert c.model_loaded()


class TestShutdown:
    def test_shutdown_releases_audio_and_context(self):
        c, _, audio, _ = _make()
        c.start()  # creates the audio pipeline and loads the model
        join_thread(c, "_context_threads")
        whisper = c._whisper
        c.shutdown()
        assert audio.destroyed == 1
        assert whisper.closed is True
        assert c._whisper is None

    def test_shutdown_joins_a_running_worker(self):
        # A worker still running at shutdown is joined (not killed mid-call): shutdown
        # blocks until it finishes, well within the grace budget.
        gate = threading.Event()
        c, _, audio, _ = _make(whisper_factory=FakeWhisper.factory(gate=gate))
        c.start()
        audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
        threading.Timer(0.15, gate.set).start()  # the worker stays mid-call that long
        c.stop()  # spawns the transcribe worker (still blocked when shutdown runs)

        c.shutdown()
        assert not c._transcribe_thread.is_alive()

    def test_shutdown_does_not_hang_on_a_wedged_worker(self, monkeypatch, caplog):
        # A wedged native call must not hang shutdown: it returns within the bounded
        # budget and warns that a worker is still running.
        monkeypatch.setattr(recording, "SHUTDOWN_JOIN_TIMEOUT_S", 0.1)
        entered, gate = threading.Event(), threading.Event()
        factory = FakeWhisper.factory(gate=gate, entered=entered)
        c, _, audio, _ = _make(whisper_factory=factory)
        c.start()
        audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
        c.stop()
        assert entered.wait(2)  # the worker is blocked in transcribe_buffer

        caplog.set_level("WARNING")
        started = time.monotonic()
        c.shutdown()
        assert time.monotonic() - started < 2  # did not block on the wedged worker
        assert any("still running at shutdown" in m for m in caplog.messages)

        gate.set()  # unblock so the daemon thread can exit cleanly
        join_thread(c, "_transcribe_thread")
        assert c._transcribe_thread.daemon  # nor would it hold the process open at exit


class TestBufferThreadSafety:
    def test_a_chunk_waits_for_the_buffer_lock(self):
        c, _, audio, _ = _make()
        c.prepare_audio()
        writer = threading.Thread(
            target=audio.on_audio_buffer, args=(np.ones(1, dtype=np.float32),)
        )
        with c._buffer_lock:
            writer.start()
            writer.join(0.2)  # a chunk appended without the lock would land by now
            assert writer.is_alive()
        writer.join(2)
        assert not writer.is_alive()
        assert len(c._buffer) == 1

    def test_a_stop_waits_for_the_buffer_lock(self):
        # The main thread drains what the GStreamer thread appends, so both hold it.
        c, _, audio, _ = _make()
        c.start()
        audio.on_audio_buffer(np.ones(1600, dtype=np.float32))
        stopper = threading.Thread(target=c.stop)
        with c._buffer_lock:
            stopper.start()
            stopper.join(0.2)  # a drain without the lock would be done by now
            assert stopper.is_alive()
        stopper.join(2)
        assert not stopper.is_alive()
        join_thread(c, "_transcribe_thread")
        assert c._buffer == []


class TestDeviceEncoderGate:
    """Only NPU requests the OpenVINO encoder; GPU (Vulkan) and CPU don't."""

    @pytest.mark.parametrize(
        ("device", "encoder"),
        [
            ("NPU", "/models/ggml-small-encoder-openvino.xml"),
            ("GPU", None),
            ("CPU", None),
        ],
    )
    def test_only_the_npu_requests_the_encoder(self, device, encoder):
        c, _, _, _ = _make(
            device=device, model_mgr=FakeModelManager(), whisper_factory=OffloadWhisper
        )
        c.ensure_context()
        assert c._whisper.init_kwargs.get("openvino_model_path") == encoder


class TestEnsureContextAsync:
    """`ensure_context_async` loads off the main thread and always fires `on_ready`."""

    def test_loads_model_and_fires_on_ready(self):
        c, _, _, _ = _make(config=_config(keep_warm=True))
        done = threading.Event()
        ready_calls = []

        def on_ready(model, device, outcome):
            ready_calls.append((model, device, outcome))
            done.set()

        c.ensure_context_async(c._config.model, c._device, on_ready)
        assert done.wait(2)
        assert ready_calls == [("small", "CPU", ContextLoadOutcome.LOADED)]
        assert c._whisper is not None

    @pytest.mark.parametrize("keep_warm", [True, False])
    def test_an_idle_preload_is_kept_only_as_keep_warm_allows(self, keep_warm):
        scheduler = DeferredScheduler()
        c, _, _, _ = _make(
            config=_config(keep_warm=keep_warm, keep_warm_timeout=300),
            scheduler=scheduler,
        )
        # What `on_ready` finds: keep-warm has already applied, so a load it starts
        # is not freed by it.
        seen = []
        c.ensure_context_async(
            "small",
            "CPU",
            lambda *_result: seen.append((c.model_loaded(), len(scheduler.timers))),
        )
        join_thread(c, "_context_threads")
        scheduler.fire_dispatches()

        if keep_warm:
            assert seen == [(True, 1)]
            assert scheduler.timers[0][3] == 300_000
            scheduler.fire_all()
        else:
            assert seen == [(False, 0)]
        assert not c.model_loaded()

    def test_a_failed_load_arms_no_timer(self):
        scheduler = DeferredScheduler()
        c, _, _, _ = _make(
            config=_config(keep_warm=True),
            scheduler=scheduler,
            whisper_factory=FakeWhisper.factory(raise_on_init=RuntimeError("boom")),
        )
        c.ensure_context_async("small", "CPU", lambda *_result: None)
        join_thread(c, "_context_threads")
        scheduler.fire_dispatches()
        assert scheduler.timers == []

    def test_a_load_finishing_under_a_newer_one_arms_no_timer(self):
        scheduler = DeferredScheduler()
        opening, release = threading.Event(), threading.Event()
        built = []

        class SecondBlocks(FakeWhisper):
            def __init__(self, *args, **kwargs):
                built.append(self)
                if len(built) == 2:
                    opening.set()
                    release.wait(5)
                super().__init__(*args, **kwargs)

        c, _, _, _ = _make(
            config=_config(keep_warm=True, keep_warm_timeout=30),
            scheduler=scheduler,
            whisper_factory=SecondBlocks,
        )
        c.ensure_context_async("small", "CPU", lambda *_result: None)
        join_thread(c, "_context_threads")
        c.ensure_context_async("base", "CPU", lambda *_result: None)
        assert opening.wait(2)
        # The first load's completion runs after the second, say an NPU compile that
        # outlasts the timeout, has closed the first load's context.
        scheduler.fire_dispatches()
        assert scheduler.timers == []

        release.set()
        join_thread(c, "_context_threads")
        scheduler.fire_dispatches()
        assert c._loaded_key == ("base", "CPU")
        assert len(scheduler.timers) == 1

    def test_a_dictation_started_during_a_preload_keeps_its_context(self):
        scheduler = DeferredScheduler()
        c, _, _, _ = _make(config=_config(keep_warm=False), scheduler=scheduler)
        c.ensure_context_async("small", "CPU", lambda *_result: None)
        join_thread(c, "_context_threads")
        assert c.start() is True

        scheduler.fire_dispatches()
        assert c.model_loaded()

    def test_a_recording_loads_a_freed_model_while_the_user_speaks(self):
        c, _, _, _ = _make(config=_config(keep_warm=True))
        assert not c.model_loaded()
        assert c.start() is True
        join_thread(c, "_context_threads")
        assert c.model_loaded()
        # No throwaway decode: it would hold up a short dictation.
        assert c._whisper.warmups == 0

    def test_no_second_load_starts_while_the_engine_is_busy(self):
        c, _, _, _ = _make(config=_config(keep_warm=True))
        with c._inference_lock:
            assert c.start() is True
        assert c._context_threads == []

    def test_a_cancelled_recording_keeps_its_load_only_as_keep_warm_allows(self):
        c, sched, _, _ = _make(config=_config(keep_warm=True, keep_warm_timeout=300))
        c.start()
        join_thread(c, "_context_threads")
        c.cancel()
        assert c.model_loaded()
        assert [t[3] for t in sched.timers] == [300_000]
        sched.fire_all()
        assert not c.model_loaded()

    def test_a_load_that_fails_on_press_is_reported_once(self):
        c, _, audio, events = _make(
            whisper_factory=FakeWhisper.factory(raise_on_init=RuntimeError("bad model"))
        )
        _dictate(c, audio)
        join_thread(c, "_context_threads")
        assert len(events["error_calls"]) == 1

    def test_loads_requested_pair_when_config_changes_before_worker_runs(self):
        config = _config(model="small", keep_warm=True)
        c, _, _, _ = _make(config=config, device="CPU")
        done = threading.Event()
        results = []

        c._inference_lock.acquire()
        try:
            c.ensure_context_async(
                "base",
                "CPU",
                lambda *result: (results.append(result), done.set()),
            )
            config.model = "large-v3-turbo"
            c.set_device("GPU")
        finally:
            c._inference_lock.release()

        assert done.wait(2)
        assert c._loaded_key == ("base", "CPU")
        assert results == [("base", "CPU", ContextLoadOutcome.LOADED)]

    def test_ready_result_survives_context_replacement_before_callback(self):
        scheduler = DeferredScheduler()
        config = _config(model="small")
        results = []
        c, _, _, _ = _make(
            config=config,
            device="NPU",
            scheduler=scheduler,
            model_mgr=FakeModelManager(encoder=True),
            whisper_factory=OffloadWhisper,
        )

        c.ensure_context_async("small", "NPU", lambda *result: results.append(result))
        join_thread(c, "_context_threads")
        assert scheduler.dispatches and results == []

        config.model = "base"
        c.set_device("CPU")
        c.ensure_context()
        assert c._loaded_key == ("base", "CPU")
        assert c.context_status.outcome is ContextLoadOutcome.LOADED

        scheduler.fire_dispatches()
        assert results == [("small", "NPU", ContextLoadOutcome.OPENVINO)]

    def test_warms_up_inference_before_ready(self):
        """One throwaway inference runs during the async load.

        It keeps OpenVINO and kernel first-run costs off the first real transcription.
        """
        c, _, _, _ = _make(
            config=_config(keep_warm=True),
            whisper_factory=FakeWhisper.factory(result=""),
        )
        done = threading.Event()
        c.ensure_context_async(c._config.model, c._device, lambda *_result: done.set())
        assert done.wait(2)
        assert c._whisper.warmups == 1
        assert c._whisper.calls == 0  # warmup is not a user dictation

    def test_an_interrupted_warmup_retires_the_context_without_a_traceback(
        self, caplog
    ):
        interrupted = WarmupAborted("Whisper warmup aborted")
        c, _, _, _ = _make(
            config=_config(keep_warm=True),
            device="NPU",
            model_mgr=FakeModelManager(encoder=True),
            whisper_factory=FakeWhisper.factory(raise_on_warm_up=interrupted),
        )
        done = threading.Event()
        results = []
        c.ensure_context_async(
            c._config.model,
            c._device,
            lambda *result: (results.append(result), done.set()),
        )
        assert done.wait(2)
        assert results == [("small", "NPU", ContextLoadOutcome.FAILED)]
        assert c._whisper is None
        assert not any(record.exc_info for record in caplog.records)

    @pytest.mark.parametrize("device", ["CPU", "GPU", "NPU"])
    def test_warmup_failure_closes_context_and_reports_failure(self, device):
        contexts = []

        class BrokenWarmup(FakeWhisper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                contexts.append(self)

            def warm_up(self):
                super().warm_up()
                raise RuntimeError("warmup boom")

        c, _, _, events = _make(
            config=_config(keep_warm=True),
            device=device,
            model_mgr=FakeModelManager(encoder=True),
            whisper_factory=BrokenWarmup,
        )
        done = threading.Event()
        results = []
        c.ensure_context_async(
            c._config.model,
            c._device,
            lambda *result: (results.append(result), done.set()),
        )
        assert done.wait(2)
        assert results == [("small", device, ContextLoadOutcome.FAILED)]
        assert c._whisper is None
        assert c.context_status is None
        assert contexts[0].closed
        assert events["context"][-1] is None

    def test_cancelling_a_dictation_queued_behind_the_load_keeps_the_load(self):
        entered = threading.Event()
        release = threading.Event()

        class CancellableWarmup(FakeWhisper):
            def warm_up(self):
                super().warm_up()
                entered.set()
                release.wait(5)
                if self.cancelled:
                    raise WarmupAborted("Whisper warmup aborted")

        # Keep-warm on, so the cancelled dictation leaves the context loaded.
        c, _, audio, _ = _make(
            config=_config(keep_warm=True), whisper_factory=CancellableWarmup
        )
        done = threading.Event()
        results = []
        c.ensure_context_async(
            "small", "CPU", lambda *result: (results.append(result), done.set())
        )
        assert entered.wait(2)
        assert c.start() is True
        audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
        c.stop()
        c.cancel()
        release.set()
        assert done.wait(2)
        join_thread(c, "_transcribe_thread")
        assert results == [("small", "CPU", ContextLoadOutcome.LOADED)]
        assert c.model_loaded()

    def test_cache_hit_warms_again_without_duplicate_status(self):
        c, _, _, events = _make(config=_config(keep_warm=True))
        first_done = threading.Event()
        c.ensure_context_async("small", "CPU", lambda *_: first_done.set())
        assert first_done.wait(2)
        context = c._whisper

        second_done = threading.Event()
        c.ensure_context_async("small", "CPU", lambda *_: second_done.set())
        assert second_done.wait(2)

        assert c._whisper is context
        assert context.warmups == 2
        assert events["context"] == [
            ContextStatus("small", "CPU", ContextLoadOutcome.LOADED, "CPU")
        ]

    def test_teardown_requested_during_warmup_clears_finished_context(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingWarmup(FakeWhisper):
            def warm_up(self):
                super().warm_up()
                entered.set()
                release.wait(5)

        c, _, _, events = _make(
            config=_config(keep_warm=True), whisper_factory=BlockingWarmup
        )
        done = threading.Event()
        results = []
        c.ensure_context_async(
            "small",
            "CPU",
            lambda *result: (results.append(result), done.set()),
        )
        assert entered.wait(2)

        c.free_context()
        assert c._free_pending is True
        release.set()
        assert done.wait(2)

        assert results == [("small", "CPU", ContextLoadOutcome.LOADED)]
        assert c._whisper is None
        assert c.context_status is None
        assert c._free_pending is False
        assert events["context"][-1] is None

    def test_fires_on_ready_even_when_model_unavailable(self):
        """No downloaded model: on_ready must still fire (no hang)."""
        c, _, _, _ = _make(model_mgr=FakeModelManager(ggml=False))
        done = threading.Event()

        results = []
        c.ensure_context_async(
            c._config.model,
            c._device,
            lambda *result: (results.append(result), done.set()),
        )
        assert done.wait(2)
        assert c._whisper is None  # not loaded, but on_ready still fired
        assert results == [("small", "CPU", ContextLoadOutcome.MISSING)]

    def test_fires_on_ready_even_when_factory_raises(self):
        """A crashing whisper factory must not swallow on_ready."""
        c, _, _, _ = _make(
            whisper_factory=FakeWhisper.factory(
                raise_on_init=RuntimeError("factory exploded")
            )
        )
        done = threading.Event()

        results = []
        c.ensure_context_async(
            c._config.model,
            c._device,
            lambda *result: (results.append(result), done.set()),
        )
        assert done.wait(2)
        assert c._whisper is None
        assert results == [("small", "CPU", ContextLoadOutcome.FAILED)]

    def test_unexpected_worker_exception_is_logged_once_and_reports_failure(
        self, caplog
    ):
        caplog.set_level("ERROR")
        c, _, _, _ = _make()
        c._ensure_context = MagicMock(side_effect=RuntimeError("unexpected preload"))
        done = threading.Event()
        results = []
        c.ensure_context_async(
            "small",
            "CPU",
            lambda *result: (results.append(result), done.set()),
        )
        assert done.wait(2)

        assert results == [("small", "CPU", ContextLoadOutcome.FAILED)]
        assert sum("Async context preload failed" in m for m in caplog.messages) == 1

    def test_context_thread_is_tracked_and_joined_on_shutdown(self):
        # The load worker runs native OpenVINO compiles, so shutdown must track and join
        # it rather than let interpreter exit kill it mid-compile.
        compiled = threading.Event()

        class SlowWarmup(FakeWhisper):
            def warm_up(self):
                super().warm_up()
                compiled.wait(2)

        c, _, _, _ = _make(whisper_factory=SlowWarmup)
        c.ensure_context_async(c._config.model, c._device, lambda *_result: None)
        assert c._context_threads
        threading.Timer(0.15, compiled.set).start()  # the compile runs that long
        c.shutdown()
        assert not any(t.is_alive() for t in c._context_threads)


class TestContextStatus:
    def test_lazy_load_publishes_initialized_npu_offload(self):
        c, _, _, events = _make(
            device="NPU",
            model_mgr=FakeModelManager(encoder=True),
            whisper_factory=OffloadWhisper,
        )
        c.ensure_context()

        expected = ContextStatus("small", "NPU", ContextLoadOutcome.OPENVINO, "NPU")
        assert c.context_status == expected
        assert events["context"] == [expected]

    def test_npu_init_fallback_is_loaded_on_cpu_path(self):
        c, _, _, events = _make(
            config=_config(keep_warm=True),
            device="NPU",
            model_mgr=FakeModelManager(encoder=True),
        )
        done = threading.Event()
        results = []
        c.ensure_context_async(
            "small",
            "NPU",
            lambda *result: (results.append(result), done.set()),
        )
        assert done.wait(2)

        expected = ContextStatus("small", "NPU", ContextLoadOutcome.LOADED, "CPU")
        assert results == [("small", "NPU", ContextLoadOutcome.LOADED)]
        assert c.context_status == expected
        assert expected.fallback
        assert events["context"] == [expected]

    def test_a_status_is_a_fallback_only_off_the_requested_device(self):
        loaded = ContextLoadOutcome.LOADED
        assert not ContextStatus("small", "GPU", loaded, "GPU").fallback
        assert ContextStatus("small", "GPU", loaded, "CPU").fallback

    @pytest.mark.parametrize("outcome", list(ContextLoadOutcome))
    def test_only_a_missing_or_failed_load_is_unloaded(self, outcome):
        assert outcome.loaded is (
            outcome not in {ContextLoadOutcome.MISSING, ContextLoadOutcome.FAILED}
        )

    def test_free_and_expiry_publish_context_clear(self):
        config = _config(keep_warm=True)
        c, sched, _, events = _make(config=config)
        c.ensure_context()
        expected = ContextStatus("small", "CPU", ContextLoadOutcome.LOADED, "CPU")

        c.free_context()
        assert events["context"] == [expected, None]

        c.ensure_context()
        c.apply_keep_warm()
        sched.fire_all()
        assert events["context"] == [expected, None, expected, None]

    def test_stale_queued_status_is_not_delivered(self):
        scheduler = DeferredScheduler()
        config = _config(model="small")
        c, _, _, events = _make(config=config, scheduler=scheduler)
        c.ensure_context()
        config.model = "base"
        c.ensure_context()
        assert events["context"] == []

        scheduler.fire_dispatches()
        assert events["context"] == [
            ContextStatus("base", "CPU", ContextLoadOutcome.LOADED, "CPU")
        ]


def _refusing(devices, seen):
    """A factory that fails to build a context on `devices`, as ggml refuses a GPU."""

    def factory(*args, device, **kwargs):
        seen.append(device)
        if device in devices:
            raise ModelLoadFailed(f"Unsupported device {device}")
        return FakeWhisper(*args, device=device, **kwargs)

    return factory


class TestGpuRefusal:
    def test_a_refused_gpu_runs_the_model_on_cpu_and_is_not_retried(self):
        seen = []
        c, _, _, _ = _make(device="GPU", whisper_factory=_refusing({"GPU"}, seen))
        c.ensure_context()
        status = ContextStatus("small", "GPU", ContextLoadOutcome.LOADED, "CPU")
        assert c.context_status == status
        assert seen == ["GPU", "CPU"]

        c.free_context()
        c.ensure_context()
        assert seen == ["GPU", "CPU", "CPU"]
        # Another model may still fit the GPU.
        c._config.model = "base"
        c.ensure_context()
        assert seen == ["GPU", "CPU", "CPU", "GPU", "CPU"]

    def test_a_refusal_logs_one_plain_line_per_layer(self, caplog):
        # Anticipated and handled: no traceback, so no crash report.
        c, _, _, _ = _make(device="GPU", whisper_factory=_refusing({"GPU"}, []))
        c.ensure_context()
        refusal = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert [(r.name, r.levelname) for r in refusal] == [
            ("quoth.whisper", "ERROR"),
            ("quoth.recording", "WARNING"),
        ]
        assert refusal[1].getMessage() == (
            "GPU refused model 'small'; running it on the CPU instead"
        )
        assert not any(r.exc_info for r in caplog.records)

    def test_an_engine_that_fails_everywhere_is_not_remembered(self):
        seen = []
        refusing = _refusing({"GPU", "CPU"}, seen)
        c, _, _, _ = _make(device="GPU", whisper_factory=refusing)
        c.ensure_context()
        assert c.context_status is None
        c.ensure_context()
        assert seen == ["GPU", "CPU", "GPU", "CPU"]

    def test_a_missing_model_is_not_retried_on_cpu(self):
        seen = []
        c, _, _, _ = _make(
            device="GPU",
            model_mgr=FakeModelManager(ggml=False),
            whisper_factory=_refusing({"GPU"}, seen),
        )
        c.ensure_context()
        assert c.context_status is None
        assert seen == []

    def test_a_guided_take_on_a_refused_gpu_still_transcribes(self):
        seen = []
        c, _, audio, _ = _make(whisper_factory=_refusing({"GPU"}, seen))
        texts = []
        assert c.begin_take() is True
        audio.on_audio_buffer(np.ones(1600, dtype=np.float32))
        c.end_take("GPU", texts.append)
        join_thread(c, "_transcribe_thread")
        assert texts == ["hello world"]
        assert seen == ["GPU", "CPU"]


class TestNpuRefusal:
    """An NPU whose encoder fails to compile hands the model to the GPU, else CPU."""

    @pytest.fixture(autouse=True)
    def _all_devices(self, detected_devices):
        detected_devices.extend(["NPU", "GPU"])

    def _controller(self, refuse, seen):
        built = []

        def factory(*args, **kwargs):
            ctx = _refusing(refuse, seen)(*args, **kwargs)
            built.append(ctx)
            return ctx

        c, _, _, _ = _make(
            device="NPU",
            model_mgr=FakeModelManager(encoder=True),
            whisper_factory=factory,
        )
        return c, built

    def test_a_failed_npu_runs_the_model_on_the_gpu_and_is_not_retried(self):
        seen = []
        c, built = self._controller(set(), seen)
        c.ensure_context()
        loaded = ContextLoadOutcome.LOADED
        assert c.context_status == ContextStatus("small", "NPU", loaded, "GPU")
        assert seen == ["NPU", "GPU"]
        assert built[0].closed

        c.free_context()
        c.ensure_context()
        assert seen == ["NPU", "GPU", "GPU"]

    def test_a_gpu_that_also_refuses_leaves_the_cpu(self):
        seen = []
        c, _ = self._controller({"GPU"}, seen)
        c.ensure_context()
        loaded = ContextLoadOutcome.LOADED
        assert c.context_status == ContextStatus("small", "NPU", loaded, "CPU")
        assert seen == ["NPU", "GPU", "CPU"]

        c.free_context()
        c.ensure_context()
        assert seen == ["NPU", "GPU", "CPU", "CPU"]

    @pytest.mark.parametrize(
        ("compiled", "retried"), [(True, ["NPU", "GPU"]), (False, ["GPU"])]
    )
    def test_a_benchmark_that_runs_a_refused_device_retries_it(self, compiled, retried):
        seen = []
        c, _ = self._controller(set(), seen)
        c.ensure_context()
        done = threading.Event()
        _calibrate(
            c, ["small"], ["NPU"], lambda *a: CellResult(compiled), lambda r: done.set()
        )
        join_thread(c, "_calibrate_thread")
        assert done.is_set()
        c.ensure_context()
        assert seen == ["NPU", "GPU", *retried]


class TestInferenceAndEpoch:
    """Covers cooperative cancellation, inference serialisation, and stale results."""

    def test_queued_workers_keep_their_own_dictation_requests(self):
        gate = threading.Event()
        first_entered = threading.Event()
        calls = []

        class CapturingWhisper(FakeWhisper):
            def __init__(self, model_path, **kwargs):
                super().__init__(model_path, **kwargs)
                self._model_path = model_path

            def transcribe_buffer(self, audio, language="auto", translate=False):
                calls.append(
                    (self._model_path, self.init_kwargs["device"], language, translate)
                )
                if len(calls) == 1:
                    first_entered.set()
                    gate.wait(5)
                return f"{language} text"

        config = _config(
            model="small",
            language="de",
            translate=True,
            keep_warm=True,
        )
        typed = []
        c, sched, audio, _ = _make(
            config=config,
            device="CPU",
            whisper_factory=CapturingWhisper,
            type_text=lambda text: typed.append(text) or InjectionResult.SUCCESS,
        )
        c.start()
        audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
        c.stop()
        first_thread = c._transcribe_thread
        assert first_entered.wait(1)

        c.cancel()
        config.model = "large-v3-turbo"
        config.language = "it"
        config.translate = False
        c.set_device("GPU")
        c.start()
        audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
        c.stop()
        second_thread = c._transcribe_thread

        gate.set()
        first_thread.join(2)
        second_thread.join(2)
        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        assert calls == [
            ("/models/ggml-small.bin", "CPU", "de", True),
            ("/models/ggml-large-v3-turbo.bin", "GPU", "it", False),
        ]
        # The cancelled dictation's late result never lands in the next one.
        sched.fire_all()
        join_thread(c, "_inject_threads")
        assert typed == ["it text "]

    def test_slow_decode_waits_for_result_without_fixed_timeout(self):
        gate = threading.Event()
        typed = []
        c, sched, audio, _ = _make(
            whisper_factory=FakeWhisper.factory(gate=gate),
            type_text=lambda text: typed.append(text) or InjectionResult.SUCCESS,
        )
        c.start()
        audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
        c.stop()
        assert c.state is RecordingState.TRANSCRIBING
        assert sched.timers == []
        gate.set()
        join_thread(c, "_transcribe_thread")
        assert c.state is RecordingState.IDLE
        sched.fire_all()
        join_thread(c, "_inject_threads")
        assert typed == ["hello world "]

    def test_cancel_bumps_epoch_and_discards_result(self):
        gate = threading.Event()
        typed = []
        c, sched, audio, _ = _make(
            whisper_factory=FakeWhisper.factory(gate=gate),
            type_text=lambda t: typed.append(t) or InjectionResult.SUCCESS,
        )
        c.start()
        audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
        c.stop()
        c.cancel()
        assert c.state is RecordingState.IDLE

        gate.set()  # unblock worker; its _finish should be a no-op (epoch mismatch)
        join_thread(c, "_transcribe_thread")
        # Fire whatever the late result armed: a kept result would be typed here.
        sched.fire_all()
        join_thread(c, "_inject_threads")
        assert typed == []

    def test_finish_stale_epoch_is_noop(self):
        gate = threading.Event()
        c, _, audio, _ = _make(whisper_factory=FakeWhisper.factory(gate=gate))
        c.start()
        audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
        c.stop()
        # State is TRANSCRIBING, epoch is 1. Simulate a late dispatch from a
        # hypothetical previous cycle (epoch 0).
        c._finish("stale text", c._transcription_epoch - 1)
        assert (
            c.state is RecordingState.TRANSCRIBING
        )  # unchanged: stale result discarded
        gate.set()
        join_thread(c, "_transcribe_thread")
        assert c.state is RecordingState.IDLE

    def test_inference_lock_serializes_workers(self):
        # Hold the inference lock from the test thread (simulating an in-flight
        # inference) and verify the worker blocks until it is released.
        entered = {}  # transcribe_buffer records into it on entry
        c, _, audio, _ = _make(whisper_factory=FakeWhisper.factory(captured=entered))

        c._inference_lock.acquire()
        c.start()
        audio.on_audio_buffer(np.ones(16000, dtype=np.float32))
        c.stop()  # spawns worker; worker blocks at with self._inference_lock

        # The lock is still held, so transcribe_buffer cannot have been entered. The
        # join gives an unserialised worker time to get there.
        c._transcribe_thread.join(0.2)
        assert not entered

        c._inference_lock.release()  # worker unblocks, runs inference
        join_thread(c, "_transcribe_thread")

        assert entered
        assert c.state is RecordingState.IDLE


class TestCalibrate:
    """The benchmark run owns the engine; an injected measurement keeps it headless."""

    @staticmethod
    def _measure(record_calls: list | None = None):
        """A fast fake measurement: (model, device, model_mgr) -> a warm cell."""

        def measure(model, device, model_mgr):
            if record_calls is not None:
                record_calls.append((model, device))
            return CellResult(compiled=True, ms=10.0)

        return measure

    def test_runs_every_cell_and_fires_on_done(self):
        c, _, _, _ = _make(model_mgr=FakeModelManager(ggml=True))
        calls, done = [], []
        started = _calibrate(
            c, ["small", "base"], ["CPU", "GPU"], self._measure(calls), done.append
        )
        assert started is True
        join_thread(c, "_calibrate_thread")
        # Every models x devices cell measured, in row-major order.
        assert calls == [
            ("small", "CPU"),
            ("small", "GPU"),
            ("base", "CPU"),
            ("base", "GPU"),
        ]
        assert len(done) == 1
        assert done[0].fingerprint == "FP"
        assert set(done[0].cells) == {"small|CPU", "small|GPU", "base|CPU", "base|GPU"}

    def test_progress_marshaled_through_scheduler(self):
        scheduler = DeferredScheduler()
        c, _, _, _ = _make(scheduler=scheduler)
        progress = []
        c.calibrate(
            ["small"],
            ["CPU"],
            "FP",
            on_progress=lambda done, total, m, d, r: progress.append(
                (done, total, m, d, r)
            ),
            on_done=lambda r: None,
            measure_fn=self._measure(),
        )
        join_thread(c, "_calibrate_thread")
        assert progress == []  # the benchmark thread must not touch the UI
        scheduler.fire_dispatches()
        assert progress == [(1, 1, "small", "CPU", CellResult(compiled=True, ms=10.0))]

    def test_is_idle_false_and_recording_blocked_during_run(self):
        c, _, _, _ = _make()
        gate = threading.Event()

        def measure(model, device, model_mgr):
            gate.wait(5)
            return CellResult(compiled=True, ms=1.0)

        started = _calibrate(c, ["small"], ["CPU"], measure)
        assert started is True
        assert c.is_idle is False  # engine claimed, swaps wait
        assert c.start() is False  # recording refused while calibrating
        assert c.state is RecordingState.IDLE  # never entered RECORDING
        gate.set()
        join_thread(c, "_calibrate_thread")
        assert c.is_idle is True  # released after the run

    def test_refused_while_recording(self):
        c, _, _, _ = _make()
        c.start()  # RECORDING
        calls = []
        started = _calibrate(c, ["small"], ["CPU"], self._measure(calls))
        assert started is False
        assert calls == []  # never touched the engine

    def test_refused_while_already_calibrating(self):
        c, _, _, _ = _make()
        gate = threading.Event()

        def measure(model, device, model_mgr):
            gate.wait(5)
            return CellResult(compiled=True, ms=1.0)

        first = _calibrate(c, ["small"], ["CPU"], measure)
        second = _calibrate(c, ["base"], ["CPU"], measure)
        assert first is True and second is False
        gate.set()
        join_thread(c, "_calibrate_thread")

    def test_frees_active_context_and_suspends_keep_warm(self):
        cfg = _config(keep_warm=True, keep_warm_timeout=300)
        c, sched, _, _ = _make(config=cfg, model_mgr=FakeModelManager(ggml=True))
        c.ensure_context()
        c._schedule_warm_timeout()
        assert c.model_loaded() and len(sched.timers) == 1
        loaded = c._whisper
        _calibrate(c, ["small"], ["CPU"], self._measure())
        join_thread(c, "_calibrate_thread")
        assert loaded.closed is True  # active model's RAM released before the sweep
        assert c._whisper is None  # left cold: next dictation reloads (cache hit)
        assert sched.timers == []  # keep-warm timer suspended for the run

    def test_cancel_stops_at_next_cell_boundary(self):
        c, _, _, _ = _make()
        seen, done = [], []

        def measure(model, device, model_mgr):
            seen.append((model, device))
            c.cancel_calibration()  # request cancel after the first cell
            return CellResult(compiled=True, ms=1.0)

        _calibrate(c, ["small", "base", "large"], ["CPU"], measure, done.append)
        join_thread(c, "_calibrate_thread")
        assert seen == [("small", "CPU")]  # stopped before the second cell
        assert len(done) == 1  # on_done still fires, with the partial report

    def test_shutdown_stops_the_run_and_joins_it(self):
        c, _, _, _ = _make()
        seen = []
        measuring, measured = threading.Event(), threading.Event()

        def measure(model, device, model_mgr):
            seen.append((model, device))
            measuring.set()
            measured.wait(2)
            return CellResult(compiled=True, ms=1.0)

        _calibrate(c, ["small", "base"], ["CPU"], measure)
        assert measuring.wait(2)
        threading.Timer(0.15, measured.set).start()  # the first cell runs that long
        c.shutdown()
        assert not c._calibrate_thread.is_alive()
        assert seen == [("small", "CPU")]  # stopped before the second cell
