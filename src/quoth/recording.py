# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Recording lifecycle: the IDLE -> RECORDING -> TRANSCRIBING state machine.

The controller knows no GTK or concrete engine types: it reaches the main thread
through an injected `Scheduler`, reports to the UI through `RecordingCallbacks`, and
builds audio capture and transcription through factories, so tests can use fakes.
"""

import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from loguru import logger

from quoth import calibration
from quoth.accel import backend_for
from quoth.audio import SAMPLE_RATE, AudioBackend, AudioPipeline, AudioUnavailable
from quoth.hardware import available_devices
from quoth.models import MODELS_BY_NAME
from quoth.scheduler import Scheduler
from quoth.typer import InjectionResult
from quoth.typer import type_text as _default_type_text
from quoth.whisper import (
    TranscriptionBackend,
    WarmupAborted,
    WhisperContext,
    open_context,
)

TYPE_DELAY_MS = 50  # without it injected keystrokes can be dropped
NO_INPUT_TIMEOUT_S = 2.5  # warn if a recording captures zero audio (dead/wrong source)
SHUTDOWN_JOIN_TIMEOUT_S = 3  # grace for in-flight workers to finish before exit


class RecordingState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"


class ContextLoadOutcome(Enum):
    """Result of loading one requested model/device pair."""

    MISSING = "missing"
    FAILED = "failed"
    LOADED = "loaded"
    OPENVINO = "openvino"

    @property
    def loaded(self) -> bool:
        return self not in {ContextLoadOutcome.MISSING, ContextLoadOutcome.FAILED}

    @property
    def openvino_active(self) -> bool:
        return self is ContextLoadOutcome.OPENVINO


@dataclass(frozen=True)
class ContextStatus:
    """The backend currently installed in the controller."""

    model: str
    requested_device: str
    outcome: ContextLoadOutcome
    # The device running the model: the requested one unless it could not start.
    runs_on: str

    @property
    def fallback(self) -> bool:
        return self.runs_on != self.requested_device


_LEGAL_TRANSITIONS = {
    RecordingState.IDLE: {RecordingState.RECORDING},
    RecordingState.RECORDING: {RecordingState.IDLE, RecordingState.TRANSCRIBING},
    RecordingState.TRANSCRIBING: {RecordingState.IDLE},
}


@dataclass
class RecordingCallbacks:
    """Hooks the app wires to the overlay pill and tray.

    The app decides every UI reaction, so the state machine stays UI-agnostic.
    """

    on_state_change: Callable[[RecordingState], None]
    on_level: Callable[[float], None]
    # (message, detail, essential, guidance): a terse title for the pill, an optional
    # fuller notification body, whether it must notify beside the pill, and an optional
    # second line for the pill.
    on_error: Callable[[str, str | None, bool, str | None], None]
    # Fired when a transcription couldn't be pasted. The app decides the UI from whether
    # the transcript remains on the clipboard.
    on_typing_failed: Callable[[InjectionResult], None]
    # Fired when a recording captures no audio within NO_INPUT_TIMEOUT_S, which
    # usually means a dead or misrouted source.
    on_no_input: Callable[[], None]
    # Fired once a transcription has pasted; the app counts it in the local usage stats.
    on_dictation_success: Callable[[], None]
    # Fired when a context is installed or removed, scheduled onto the main thread.
    on_context_changed: Callable[[ContextStatus | None], None]


def model_error(*, downloaded: bool) -> tuple[str, str, bool, str]:
    """Return the `on_error` arguments for a model that cannot serve a dictation.

    A model on disk that fails to load is most often a damaged download, which
    re-downloading repairs; a missing one needs downloading first.
    """
    if downloaded:
        return (
            "Speech model failed to load",
            "The speech model couldn't load. Delete and re-download the model in "
            "Settings. If it still fails, reinstall Quoth, then restart it.",
            True,
            "Re-download in Settings",
        )
    return (
        "No model installed",
        "No model installed. Open Settings to download one.",
        True,
        "Download in Settings",
    )


@dataclass(frozen=True)
class _DictationRequest:
    """Inference settings fixed when audio capture begins."""

    model: str
    device: str
    language: str
    translate: bool


class RecordingController:
    """The owner of recording state, the audio buffer and the whisper lifecycle.

    Recording is driven only through `start`, `stop` and `cancel`; every state move goes
    through `_transition_to`, and main-thread work through the injected scheduler.
    """

    def __init__(
        self,
        config,
        model_mgr,
        device: str,
        scheduler: Scheduler,
        callbacks: RecordingCallbacks,
        *,
        audio_factory: Callable[..., AudioBackend] = AudioPipeline,
        whisper_factory: Callable[..., TranscriptionBackend] = WhisperContext,
        type_text: Callable[[str], InjectionResult] = _default_type_text,
    ):
        self._config = config
        self._model_mgr = model_mgr
        self._device = device
        self._scheduler = scheduler
        self._cb = callbacks
        self._audio_factory = audio_factory
        self._whisper_factory = whisper_factory
        self._type_text = type_text

        self._state = RecordingState.IDLE
        self._audio: AudioBackend | None = None
        self._whisper: TranscriptionBackend | None = None
        # The (model, device) the loaded context was built from, or None.
        self._loaded_key: tuple[str, str] | None = None
        self._context_status: ContextStatus | None = None
        # Pairs whose device refused the model; its fallbacks run it until restart, or
        # until a benchmark runs it there.
        self._refused: set[tuple[str, str]] = set()
        self._context_status_generation = 0
        # A teardown deferred while inference holds _inference_lock, applied once it is
        # released: freeing mid-whisper_full would use-after-free the context.
        self._free_pending = False
        # The GStreamer thread appends captured chunks and the main thread drains them,
        # so every access holds _buffer_lock.
        self._buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._warm_timeout_id: int | None = None
        self._transcription_epoch = 0
        self._dictation_request: _DictationRequest | None = None
        self._transcribe_thread: threading.Thread | None = None
        # The context a decode is running on, the only engine work a cancel stops.
        self._decoding: TranscriptionBackend | None = None
        # One per dictation. typer serialises them, so one may wait behind another;
        # all are kept so shutdown joins every pending paste, not only the last.
        self._inject_threads: list[threading.Thread] = []
        # The last transcript, kept in memory only so it can be copied again when a
        # paste did not land.
        self._last_text: str | None = None
        # The ensure_context_async workers, joined at shutdown like the others.
        self._context_threads: list[threading.Thread] = []
        # A benchmark run holds the engine: recording is refused and model or device
        # swaps wait. It runs within IDLE, outside the state machine, so needs a flag.
        self._calibrating = False
        self._calibration_cancelled = False
        self._calibrate_thread: threading.Thread | None = None
        # Onboarding's guided take: a capture and decode whose caller shows its own
        # feedback, so it runs within IDLE and holds the engine, like a benchmark.
        self._taking = False
        # Work that must not share the engine with a dictation, take or benchmark.
        self._idle_waiters: list[Callable[[], None]] = []
        # Whoever is showing the input level (set_input_monitoring).
        self._input_watchers: set[object] = set()
        # The GStreamer thread only sets `_got_audio`, which the main-thread no-input
        # timer reads, so that thread never has to cancel the timer.
        self._no_input_timeout_id: int | None = None
        self._got_audio = False

    @property
    def state(self) -> RecordingState:
        return self._state

    @property
    def last_text(self) -> str | None:
        return self._last_text

    @property
    def is_idle(self) -> bool:
        """Whether no dictation, guided take or calibration run holds the engine.

        Each needs its context to stay put until it ends, so work that would swap or
        free the context waits for this through `when_idle`.
        """
        return (
            self._state is RecordingState.IDLE
            and not self._calibrating
            and not self._taking
        )

    def when_idle(self, callback: Callable[[], None]) -> None:
        """Run `callback` now, or once no dictation, take or benchmark holds the engine.

        A callback already waiting is not queued twice.
        """
        if self.is_idle:
            callback()
        elif callback not in self._idle_waiters:
            self._idle_waiters.append(callback)

    def _run_idle_waiters(self) -> None:
        while self._idle_waiters and self.is_idle:
            self._idle_waiters.pop(0)()

    @property
    def context_status(self) -> ContextStatus | None:
        return self._context_status

    def _transition_to(self, new: RecordingState) -> bool:
        """Move to `new` and notify, or log and return False when the move is illegal.

        This is the only place the state changes; reaching IDLE frees the engine for
        whatever waited on it.
        """
        if new not in _LEGAL_TRANSITIONS[self._state]:
            logger.error("Illegal state transition {} -> {}", self._state, new)
            return False
        logger.debug("State: {} -> {}", self._state, new)
        self._state = new
        self._cb.on_state_change(new)
        if new is RecordingState.IDLE:
            self._on_engine_free()
        return True

    def _on_engine_free(self) -> None:
        """Apply what waited for the engine to come free.

        That is a deferred teardown (safe now: the worker released the lock before
        dispatching its completion), the saved input, and the idle waiters.
        """
        if self._free_pending:
            self.free_context()
        self.apply_mic_source()
        self._run_idle_waiters()

    # --- recording lifecycle (main thread) ---

    def _ensure_audio(self) -> None:
        """Create the audio pipeline once, lazily, and prime it.

        The pipeline is reused across recordings; priming keeps its first capture from
        being lost to pipewiresrc's cold-start race.
        """
        if self._audio is None:
            self._audio = self._audio_factory(
                mic_source=self._config.mic_source,
                on_audio_buffer=self._on_audio_data,
                on_level=self._on_audio_level,
            )
            self._audio.warm_up()

    def prepare_audio(self) -> None:
        """Bring the audio pipeline up at activation, unless onboarding already did.

        A failure here must not stop activation: the first recording retries and tells
        the user.
        """
        try:
            self._ensure_audio()
        except Exception:
            logger.exception("Audio warm-up failed")

    def list_microphones(self) -> list[dict]:
        """Discover devices through the same source that captures their audio."""
        self._ensure_audio()
        return self._audio.list_microphones()

    def set_input_monitoring(self, watcher: object, enabled: bool) -> None:
        """Let `watcher` see input levels through `on_level` without recording.

        The one shared pipeline keeps metering while any watcher remains; no audio
        reaches the buffer.
        """
        self._ensure_audio()
        if enabled:
            self._input_watchers.add(watcher)
        else:
            self._input_watchers.discard(watcher)
        self._audio.set_monitoring(bool(self._input_watchers))

    def start(self) -> bool:
        """Begin recording; return whether it started."""
        if self._calibrating or self._taking:
            # A benchmark or take holds the engine; recording would contend for it.
            logger.debug("Refusing to record: the engine is busy")
            return False
        if not self.model_available():
            # Don't record with no model to transcribe it. The worker still reports a
            # model that disappears before transcription.
            logger.warning(
                "Refusing to record: model '{}' not installed", self._config.model
            )
            self._cb.on_error(*model_error(downloaded=False))
            return False
        model = MODELS_BY_NAME[self._config.model]
        if self._config.translate and not model.supports_translation:
            logger.warning(
                "Refusing to record: model '{}' does not support translation",
                model.name,
            )
            self._cb.on_error(
                "Translation unavailable",
                f"{model.label} cannot translate to English. "
                "Turn off translation or choose a different model.",
                True,
                "Change model or translation",
            )
            return False
        if not self.apply_mic_source():
            return False
        if not self._transition_to(RecordingState.RECORDING):
            return False
        self._dictation_request = _DictationRequest(
            self._config.model,
            self._device,
            self._config.language,
            self._config.translate,
        )
        with self._buffer_lock:
            self._buffer.clear()
        self._got_audio = False
        self._cancel_warm_timeout()
        try:
            self._ensure_audio()
            self._audio.start()
            logger.info("Recording started")
        except Exception as e:
            # Audio already said why an unavailable microphone failed.
            if not isinstance(e, AudioUnavailable):
                logger.exception("Failed to start audio")
            self._dictation_request = None
            self._transition_to(RecordingState.IDLE)
            # Essential: every dictation fails this way until the input is fixed.
            self._cb.on_error(
                "Microphone unavailable", None, True, "Check microphone in Settings"
            )
            self._schedule_warm_timeout()  # cancelled above; the context must not stay pinned
            return False
        self._cancel_no_input_timeout()
        # after_ms because after_s would truncate the fractional timeout.
        self._no_input_timeout_id = self._scheduler.after_ms(
            int(NO_INPUT_TIMEOUT_S * 1000), self._on_no_input_timeout
        )
        if not self.model_loaded() and not self._inference_lock.locked():
            # Keep-warm freed the model and nothing is loading it: load it while the
            # user speaks, not after release. A warm-up would hold up a short dictation.
            request = self._dictation_request
            self.ensure_context_async(
                request.model, request.device, lambda *_result: None, warm=False
            )
        return True

    def stop(self) -> bool:
        """Stop recording and kick off transcription."""
        if self._state != RecordingState.RECORDING:
            logger.debug("Stop ignored: not recording (state={})", self._state)
            return False
        self._cancel_no_input_timeout()
        self._audio.stop()
        self._transition_to(RecordingState.TRANSCRIBING)
        request, self._dictation_request = self._dictation_request, None
        assert request is not None
        self._decode(request, self._finish)
        return True

    def cancel(self) -> None:
        """Discard the current recording or guided take, stopping a running decode."""
        self._cancel_no_input_timeout()
        if self._audio:
            self._audio.stop()
        with self._buffer_lock:
            self._buffer.clear()
        self._dictation_request = None
        self._transcription_epoch += 1
        # Stop a running decode, or it keeps the inference lock and the next dictation
        # waits; a load's warm-up runs on, since this dictation does not own it.
        decoding = self._decoding
        if decoding is not None:
            decoding.cancel()
        if self._state in (RecordingState.RECORDING, RecordingState.TRANSCRIBING):
            # start() stopped the keep-warm timer; a discarded dictation must not pin
            # the context. Re-arm before the idle waiters, so it never frees their load.
            self._schedule_warm_timeout()
            self._transition_to(RecordingState.IDLE)
            logger.info("Recording cancelled")
        elif self._taking:
            self._end_take()
            logger.debug("Take cancelled")
        else:
            logger.debug("Cancel requested while idle (no-op)")

    def _on_audio_data(self, audio_data: np.ndarray) -> None:
        """Append a captured chunk under the buffer lock, on the GStreamer thread."""
        self._got_audio = True  # atomic bool; read by the main-thread no-input watchdog
        with self._buffer_lock:
            self._buffer.append(audio_data)

    def _on_audio_level(self, amplitude: float) -> None:
        """Forward a waveform amplitude, which the bus watch delivers on the main loop.

        It goes through the scheduler like every other callback.
        """
        self._scheduler.dispatch(self._cb.on_level, amplitude)

    def _on_no_input_timeout(self) -> bool:
        """Fire `on_no_input` once if the recording has still captured no buffer."""
        self._no_input_timeout_id = None
        if self._state is RecordingState.RECORDING and not self._got_audio:
            logger.warning(
                "No audio captured within {}s of recording start", NO_INPUT_TIMEOUT_S
            )
            self._cb.on_no_input()
        return False

    def _cancel_no_input_timeout(self) -> None:
        if self._no_input_timeout_id is not None:
            self._scheduler.cancel(self._no_input_timeout_id)
            self._no_input_timeout_id = None

    def _dispatch_context_error(self, model: str) -> None:
        """Surface a failed model or engine context load from the worker thread."""
        downloaded = self._model_mgr.is_downloaded(model)
        self._scheduler.dispatch(self._cb.on_error, *model_error(downloaded=downloaded))

    def _decode(
        self,
        request: _DictationRequest,
        deliver: Callable[[str, int], bool],
    ) -> None:
        """Transcribe the captured buffer off the main thread, then call `deliver`.

        `deliver(text, epoch)` runs on the main thread, with "" when nothing was
        captured or the engine failed, and must drop a result whose epoch is stale.
        """
        self._transcription_epoch += 1
        epoch = self._transcription_epoch
        # A pending expiry would free the context this decode is about to use.
        self._cancel_warm_timeout()
        with self._buffer_lock:
            chunks, self._buffer = self._buffer, []
        if not chunks:
            logger.warning("No audio captured")
            deliver("", epoch)
            return
        audio = np.concatenate(chunks)
        logger.info("Transcribing {:.1f}s of audio...", len(audio) / SAMPLE_RATE)
        # The level alone tells a silent or wrong microphone from quiet speech.
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        logger.debug(
            "Captured peak {:.3f}, RMS {:.1f} dBFS",
            float(np.max(np.abs(audio), initial=0.0)),
            20 * np.log10(max(rms, 1e-10)),
        )

        def _worker():
            text = self._locked_decode(request, audio, epoch)
            # A free deferred while this decode held the lock: a cancel may already
            # have returned the engine to idle, so no later transition would apply it.
            if self._free_pending:
                self.free_context()
            if text is not None:
                self._scheduler.dispatch(deliver, text, epoch)

        self._transcribe_thread = threading.Thread(
            target=_worker, name="decode", daemon=True
        )
        self._transcribe_thread.start()

    def _locked_decode(
        self, request: _DictationRequest, audio: np.ndarray, epoch: int
    ) -> str | None:
        """Decode `audio` under `_inference_lock`, on the worker thread.

        Return the text, "" once a failure is reported, or None when a cancel
        superseded `epoch` before the decode began.
        """
        # All inference shares one non-reentrant context.
        with self._inference_lock:
            if epoch != self._transcription_epoch:
                # Cancelled while queued behind an earlier decode: nothing to do, and
                # the flag reset in the backend must not swallow a newer cancel.
                logger.debug("Skipping decode for a superseded cycle")
                return None
            try:
                self._ensure_context((request.model, request.device))
                if epoch != self._transcription_epoch:
                    # Cancelled during a slow model load: the backend would clear its
                    # abort flag on entry, so do not enter.
                    logger.debug("Skipping decode for a superseded cycle")
                    return None
                if not self._whisper:
                    # open_context has already logged why.
                    logger.debug("No whisper context available")
                    self._dispatch_context_error(request.model)
                    return ""
                t0 = time.perf_counter()
                self._decoding = self._whisper
                try:
                    text = self._whisper.transcribe_buffer(
                        audio,
                        language=request.language,
                        translate=request.translate,
                    )
                finally:
                    self._decoding = None
                logger.debug(
                    "Transcribed {:.1f}s in {:.0f}ms on {} ({} chars)",
                    len(audio) / SAMPLE_RATE,
                    (time.perf_counter() - t0) * 1000,
                    request.device,
                    len(text),
                )
                # Transcript text only at TRACE, which never reaches the log file.
                logger.trace("Transcription text: '{}'", text)
                # A cancel also returns empty text, but moves the epoch on first.
                if not text and epoch == self._transcription_epoch:
                    logger.info("No speech detected")
                return text
            except Exception:
                logger.exception(
                    "Transcription failed (model={}, device={}, samples={})",
                    request.model,
                    request.device,
                    len(audio),
                )
                self._scheduler.dispatch(
                    self._cb.on_error,
                    "Transcription failed",
                    None,
                    False,
                    "Try dictating again",
                )
                return ""

    def _finish(self, text: str, epoch: int) -> bool:
        if (
            epoch != self._transcription_epoch
            or self._state != RecordingState.TRANSCRIBING
        ):
            logger.debug("Discarding stale transcription result (epoch {})", epoch)
            return False
        self._transition_to(RecordingState.IDLE)
        if text:
            logger.success("Transcription complete ({} chars)", len(text))
            self._last_text = text
            # The trailing space keeps consecutive dictations into one field apart,
            # except after wide (CJK) text, which is written without spaces.
            if unicodedata.east_asian_width(text[-1]) not in "WF":
                text += " "
            self._scheduler.after_ms(TYPE_DELAY_MS, self._inject, text)
        else:
            self._schedule_warm_timeout()
        return False

    def _inject(self, text: str) -> bool:
        # Injection shells out to wl-copy and a chord sender, so it runs off the main
        # thread: a stalled spawn must not freeze the UI or block the next recording.
        def _worker():
            try:
                result = self._type_text(text)
            except Exception:
                logger.exception("Text injection raised")
                result = InjectionResult.FAILED
            self._scheduler.dispatch(self._after_inject, result)

        self._inject_threads = [t for t in self._inject_threads if t.is_alive()]
        thread = threading.Thread(target=_worker, name="inject", daemon=True)
        self._inject_threads.append(thread)
        thread.start()
        return False

    def _after_inject(self, result: InjectionResult) -> bool:
        if result is InjectionResult.SUCCESS:
            self._cb.on_dictation_success()
        else:
            logger.warning("Paste did not land ({})", result.value)
            self._cb.on_typing_failed(result)
        # Only manage keep-warm if still idle: a new recording may have started while
        # injection ran, and we must not free a context it's about to use.
        if self._state == RecordingState.IDLE:
            self._schedule_warm_timeout()
        return False

    # --- whisper context + keep-warm ---

    def model_available(self) -> bool:
        """Whether a dictation could produce text right now.

        That needs a loaded context or the active model's weights on disk. Unlike
        `ModelSwitcher.assets_ready` it ignores the OpenVINO encoder, since without one
        another device runs the model.
        """
        if self.model_loaded():
            return True
        return self._model_mgr.is_downloaded(self._config.model)

    def model_loaded(self) -> bool:
        """Whether the whisper context is loaded in memory and ready to infer."""
        return bool(self._whisper and self._whisper.is_loaded)

    def ensure_context(self) -> None:
        """Preload the model synchronously; production uses `ensure_context_async`."""
        # A pending expiry would free the context this load is about to build.
        self._cancel_warm_timeout()
        with self._inference_lock:
            self._ensure_context()

    def ensure_context_async(
        self,
        model: str,
        device: str,
        on_ready: Callable[[str, str, ContextLoadOutcome], None],
        warm: bool = True,
    ) -> None:
        """Load one requested pair off the main thread and report its result.

        The result is captured while the inference lock still protects that context,
        so a later load cannot change what the queued main-thread callback observes.
        Always fires `on_ready`, even on load failure. If the engine is idle when the
        load completes, what it loaded is kept only as long as keep-warm keeps a
        dictation's context. `warm` runs one throwaway inference so the outcome proves
        the device; without it the outcome is unproven, and the load only spares the
        next dictation the wait.
        """

        def _worker() -> None:
            outcome = ContextLoadOutcome.FAILED
            try:
                with self._inference_lock:
                    try:
                        self._ensure_context((model, device))
                        if self.model_loaded() and self._loaded_key == (model, device):
                            try:
                                if warm:
                                    t0 = time.perf_counter()
                                    self._whisper.warm_up()
                                    logger.debug(
                                        "Inference warmup complete in {:.0f}ms on {}",
                                        (time.perf_counter() - t0) * 1000,
                                        device,
                                    )
                            except Exception as e:
                                # An unproven context never marks the device prepared.
                                if isinstance(e, WarmupAborted):
                                    logger.debug(
                                        "Inference warmup interrupted on {}", device
                                    )
                                else:
                                    logger.exception(
                                        "Inference warmup failed (model={}, device={})",
                                        model,
                                        device,
                                    )
                                self._close_whisper()
                            else:
                                outcome = self._context_status.outcome
                        elif not self._model_mgr.is_downloaded(model):
                            outcome = ContextLoadOutcome.MISSING
                        if self._free_pending:
                            self._close_whisper()
                    except Exception:
                        self._close_whisper()
                        raise
            except Exception:
                logger.exception(
                    "Async context preload failed (model={}, device={})", model, device
                )
            finally:
                if self._free_pending:
                    self.free_context()
                self._scheduler.dispatch(
                    self._after_load, on_ready, model, device, outcome
                )

        # A pending expiry would free the context this load is about to build.
        self._cancel_warm_timeout()
        # Keep every loader that may still run so shutdown joins them all.
        self._context_threads = [t for t in self._context_threads if t.is_alive()]
        thread = threading.Thread(target=_worker, name="context-load", daemon=True)
        self._context_threads.append(thread)
        thread.start()

    def _after_load(
        self,
        on_ready: Callable[[str, str, ContextLoadOutcome], None],
        model: str,
        device: str,
        outcome: ContextLoadOutcome,
    ) -> bool:
        # Nothing to time out after a failed load, or while a later load replaces this
        # one. Armed before `on_ready`, so a load `on_ready` starts cancels the timer
        # rather than being freed by it; a dictation waiting on this load arms its own.
        if self.is_idle and self.model_loaded():
            self._schedule_warm_timeout()
        on_ready(model, device, outcome)
        return False

    def _ensure_context(self, desired: tuple[str, str] | None = None) -> None:
        """Reconcile the loaded context to `desired`, by default the configured pair.

        This is the single load and invalidate point: a model or device change only
        records the new choice, and the next call here reloads. Call it holding
        `_inference_lock`, so closing the stale context never races inference.
        """
        desired = desired or (self._config.model, self._device)
        model, device = desired
        if self.model_loaded() and self._loaded_key == desired:
            return
        # Closing may release the GIL; keep any teardown requested meanwhile.
        self._close_whisper(clear_pending=False)
        ctx, runs_on = self._open_first(model, device)
        if ctx is None:
            return
        outcome = (
            ContextLoadOutcome.OPENVINO
            if ctx.openvino_active
            else ContextLoadOutcome.LOADED
        )
        self._whisper = ctx
        self._loaded_key = desired
        self._set_context_status(ContextStatus(model, device, outcome, runs_on))
        logger.info("Whisper model loaded: {} on {}", model, runs_on)

    def _open_first(
        self, model: str, device: str
    ) -> tuple[TranscriptionBackend | None, str]:
        """Open `model` on `device`, else on the first of its fallbacks that runs it.

        ggml can list a GPU it then refuses for a model, and an NPU can fail its encoder
        compile. A refusal is remembered only once a fallback runs the model, since an
        engine failing everywhere says nothing about the device.
        """
        refused: list[str] = []
        for runs_on in (device, *backend_for(device).fallbacks):
            if (model, runs_on) in self._refused:
                continue
            # The requested device was resolved as present.
            if runs_on != device and runs_on not in available_devices():
                continue
            ctx = self._open_context(model, runs_on)
            if ctx is None:
                if not self._model_mgr.is_downloaded(model):
                    return None, device
            elif ctx.openvino_active or not backend_for(runs_on).wants_openvino_encoder:
                break
            else:
                ctx.close()
            refused.append(runs_on)
        else:
            return None, device
        for failed in refused:
            logger.warning(
                "{} refused model '{}'; running it on the {} instead",
                failed,
                model,
                runs_on,
            )
            self._refused.add((model, failed))
        return ctx, runs_on

    def _open_context(self, model: str, device: str) -> TranscriptionBackend | None:
        return open_context(
            model,
            device,
            self._model_mgr,
            vad=self._config.vad,
            factory=self._whisper_factory,
        )

    def set_device(self, device: str) -> None:
        """Switch the inference device; the context reloads on its next use."""
        self._device = device

    def apply_mic_source(self) -> bool:
        """Reconcile the saved input while idle; False means reconfiguration failed.

        Busy changes remain in config until the engine is free. With no pipeline yet,
        its constructor will use the latest saved selection.
        """
        if self._audio is None or not self.is_idle:
            return True
        try:
            if self._audio.set_mic_source(self._config.mic_source):
                self._audio.warm_up()
        except Exception as e:
            # Audio already said why an unavailable microphone failed.
            if not isinstance(e, AudioUnavailable):
                logger.exception("Failed to change microphone")
            self._cb.on_error(
                "Microphone change failed", None, True, "Check microphone in Settings"
            )
            return False
        return True

    def free_context(self) -> None:
        """Tear down the loaded context; safe from any caller, in any state.

        While inference is in flight this defers until the lock is released: closing
        the native context from under whisper_full would use-after-free it.
        """
        if not self._inference_lock.acquire(blocking=False):
            # This atomic flag lets the lock owner reconcile teardown before and after
            # release; the second acquire covers a release racing the flag write.
            self._free_pending = True
            if not self._inference_lock.acquire(blocking=False):
                return
        try:
            self._close_whisper()
        finally:
            self._inference_lock.release()

    def _close_whisper(self, *, clear_pending: bool = True) -> None:
        """Tear down the raw context; call it only while holding `_inference_lock`."""
        try:
            if self._whisper:
                self._whisper.close()
        finally:
            self._whisper = None
            self._loaded_key = None
            if clear_pending:
                self._free_pending = False
            self._set_context_status(None)

    def _set_context_status(self, status: ContextStatus | None) -> None:
        if status == self._context_status:
            return
        self._context_status = status
        self._context_status_generation += 1
        self._scheduler.dispatch(
            self._deliver_context_status, status, self._context_status_generation
        )

    def _deliver_context_status(
        self, status: ContextStatus | None, generation: int
    ) -> bool:
        if (
            generation == self._context_status_generation
            and status is self._context_status
        ):
            self._cb.on_context_changed(status)
        return False

    def apply_keep_warm(self) -> None:
        """React to a keep_warm toggle change."""
        if self._config.keep_warm:
            self._schedule_warm_timeout()
        else:
            self.free_context()
            self._cancel_warm_timeout()

    def reschedule_warm_timeout(self) -> None:
        """React to a keep_warm_timeout change, only if a timer is pending."""
        if self._warm_timeout_id is not None:
            self._cancel_warm_timeout()
            self._schedule_warm_timeout()

    def _schedule_warm_timeout(self) -> None:
        if not self._config.keep_warm:
            self.free_context()
            return
        self._cancel_warm_timeout()
        self._warm_timeout_id = self._scheduler.after_s(
            self._config.keep_warm_timeout,
            self._on_warm_expired,
        )

    def _cancel_warm_timeout(self) -> None:
        if self._warm_timeout_id is not None:
            self._scheduler.cancel(self._warm_timeout_id)
            self._warm_timeout_id = None

    def _on_warm_expired(self) -> bool:
        # A load, decode or calibration holds the lock, and a deferred free would lose
        # its work: try again a timeout later, unless the holder's end re-arms it.
        if not self._inference_lock.acquire(blocking=False):
            return True
        self._warm_timeout_id = None
        try:
            if self.model_loaded():
                logger.info("Freeing whisper model (idle timeout)")
            self._close_whisper()
        finally:
            self._inference_lock.release()
        return False

    # --- calibration (the benchmark run) ---

    def calibrate(
        self,
        models: list[str],
        devices: list[str],
        fingerprint: str,
        *,
        on_progress: calibration.ProgressFn,
        on_done: calibration.DoneFn,
        measure_fn: calibration.MeasureFn = calibration.measure,
        path: Path | None = None,
    ) -> bool:
        """Benchmark every `models` x `devices` cell off the main thread.

        Return False unless the engine is idle. Otherwise the run frees the active
        context and holds the engine until it ends. `on_progress` and `on_done` run on
        the main thread; `on_done` fires once, with the report (partial on cancel) or
        None if the run raised.
        """
        if not self.is_idle:
            return False
        self._calibrating = True
        self._calibration_cancelled = False
        self._cancel_warm_timeout()

        def _progress(done: int, total: int, model: str, device: str, result) -> None:
            # A device the sweep ran the model on is no longer refused. This runs inside
            # calibration.run, under _inference_lock.
            if result.compiled:
                self._refused.discard((model, device))
            self._scheduler.dispatch(on_progress, done, total, model, device, result)

        def _worker() -> None:
            try:
                with self._inference_lock:
                    # Drop the active model before the sweep, or it would double model
                    # RAM; it reloads lazily (a warm cache hit) on the next dictation.
                    self._close_whisper()
                    report = calibration.run(
                        models,
                        devices,
                        fingerprint,
                        self._model_mgr,
                        path=path,
                        measure_fn=measure_fn,
                        on_progress=_progress,
                        should_cancel=lambda: self._calibration_cancelled,
                    )
            except Exception:
                logger.exception("Calibration run failed")
                report = None
            self._scheduler.dispatch(self._finish_calibration, report, on_done)

        self._calibrate_thread = threading.Thread(
            target=_worker, name="calibrate", daemon=True
        )
        self._calibrate_thread.start()
        return True

    def cancel_calibration(self) -> None:
        """Request an early stop; safe to call when not calibrating.

        The run ends at the next cell boundary and keeps the cells already measured.
        """
        self._calibration_cancelled = True

    def _finish_calibration(
        self,
        report: calibration.Report | None,
        on_done: calibration.DoneFn,
    ) -> bool:
        """Release the engine and hand back the report, on the main thread.

        The context stays cold, so there is no keep-warm timer to re-arm; the next
        dictation reloads it.
        """
        self._calibrating = False
        on_done(report)
        self._on_engine_free()
        return False

    # --- the guided take (onboarding) ---

    def begin_take(self) -> bool:
        """Capture one utterance for a caller that shows its own feedback.

        The caller is onboarding's Try it step, so no state or UI callback fires. Return
        False when the engine is busy or the microphone cannot start.
        """
        if not self.is_idle or not self.apply_mic_source():
            return False
        with self._buffer_lock:
            self._buffer.clear()
        try:
            self._ensure_audio()
            self._audio.start()
        except Exception as e:
            # Audio already said why an unavailable microphone failed.
            if not isinstance(e, AudioUnavailable):
                logger.exception("Failed to start audio")
            return False
        # A pending expiry would free the context this take's decode is about to use.
        self._cancel_warm_timeout()
        self._taking = True
        return True

    def end_take(self, device: str, on_text: Callable[[str], None]) -> None:
        """Close the take and transcribe it on `device`.

        It uses the saved model and language. `on_text` gets the transcript on the main
        thread, "" when nothing was captured or understood or the engine failed; a
        `cancel` first drops it. An engine failure is reported as a dictation's is.
        """
        self._audio.stop()
        request = _DictationRequest(
            self._config.model, device, self._config.language, False
        )
        self._decode(
            request, lambda text, epoch: self._finish_take(text, epoch, on_text)
        )

    def _finish_take(
        self, text: str, epoch: int, on_text: Callable[[str], None]
    ) -> bool:
        if epoch != self._transcription_epoch:
            logger.debug("Discarding stale take (epoch {})", epoch)
            return False
        self._end_take()
        on_text(text)
        return False

    def _end_take(self) -> None:
        self._taking = False
        # The take stopped the keep-warm timer; however it ends, it must not pin the
        # context. Re-arm before the waiters, so it never frees their new load.
        self._schedule_warm_timeout()
        self._on_engine_free()

    def _join_workers(self, timeout: float) -> None:
        """Join every worker thread within one shared `timeout` budget."""
        workers = (
            self._transcribe_thread,
            *self._inject_threads,
            self._calibrate_thread,
            *self._context_threads,
        )
        deadline = time.monotonic() + timeout
        for worker in workers:
            if worker is not None and worker.is_alive():
                worker.join(timeout=max(0.0, deadline - time.monotonic()))
        stragglers = sum(1 for w in workers if w is not None and w.is_alive())
        if stragglers:
            logger.warning(
                "{} worker(s) still running at shutdown; exiting anyway", stragglers
            )

    def shutdown(self) -> None:
        """Tear down audio and whisper resources on app exit.

        A bounded join lets an in-flight native call finish and release the lock before
        teardown, while a wedged call cannot hang the exit.
        """
        self._calibration_cancelled = True  # an in-flight sweep stops at the next cell
        # Whatever runs on the engine, a warm-up too, would hold the join budget; a
        # dictation cut short here is discarded, as a cancel's is.
        self._transcription_epoch += 1
        if self._whisper is not None:
            self._whisper.cancel()
        self._join_workers(SHUTDOWN_JOIN_TIMEOUT_S)
        if self._audio:
            self._audio.destroy()
            self._audio = None
        self.free_context()
        self._cancel_warm_timeout()
        self._cancel_no_input_timeout()
