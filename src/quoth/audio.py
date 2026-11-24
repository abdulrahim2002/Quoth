# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""GStreamer audio capture for PulseAudio and PipeWire.

One long-lived pipeline captures the microphone through a capability-selected source
(PulseAudio, then PipeWire, then auto), converts it to the 16 kHz mono float32 that
whisper.cpp takes, delivers buffers through an appsink callback and posts amplitude for
the waveform through a level element. `RecordingController` drives it through the
`AudioBackend` protocol.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import gi
import numpy as np
from loguru import logger

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

Gst.init(None)


class AudioUnavailable(RuntimeError):
    """The capture pipeline could not be started: no microphone, or no sound server."""


SAMPLE_RATE = 16000
CHANNELS = 1
LEVEL_INTERVAL = 33_333_333  # ~30 Hz in nanoseconds

# The meter: speech RMS in dBFS across this window, bent by the curve so syllables
# stand out. Calibrated on a laptop mic, where room tone sits near -55 and normal
# dictation near -36.
LEVEL_FLOOR_DB = -50.0
LEVEL_CEILING_DB = -20.0
LEVEL_CURVE = 0.7
# Fans and rumble sit below this; voices barely do.
METER_HIGHPASS_HZ = 120

# Cold-start warmup: pipewiresrc reports PLAYING before the PipeWire
# stream actually delivers buffers, so the first recording can be lost. Prime
# the stream at startup and verify real buffer flow, rebuilding on a stall.
WARMUP_WINDOW_MS = 1000  # observation window per attempt
WARMUP_MIN_BUFFERS = 25  # sustained buffers proving the stream is fully settled
WARMUP_MAX_CYCLES = 3  # rebuild attempts before giving up gracefully


def level_from_db(db: float) -> float:
    """Map a meter RMS in dBFS onto 0..1.

    GStreamer reports digital silence as -700 dB and a NaN sample poisons the whole
    interval; both read as silence.
    """
    if not math.isfinite(db):
        return 0.0
    span = (db - LEVEL_FLOOR_DB) / (LEVEL_CEILING_DB - LEVEL_FLOOR_DB)
    return min(1.0, max(0.0, span)) ** LEVEL_CURVE


@dataclass(frozen=True)
class _AudioSource:
    """A GStreamer capture source and how to drive it."""

    element: str
    device_prop: str | None  # property for explicit device selection, if any
    extra_props: str  # element launch props (e.g. pipewiresrc always-copy)
    stays_playing: bool  # pipewiresrc: never stopped, as it resumes unreliably


# The name and icon the microphone indicator shows; the icon name mirrors quoth.APP_ID,
# since this leaf imports nothing from quoth. No media.role: on PulseAudio and
# WirePlumber 0.4 a call role switches Bluetooth headphones to the headset profile.
# pipewiresrc sets no node.latency: it stays PLAYING, so a latency request would
# lower the graph's quantum for every app.
_PIPEWIRE_PROPS = (
    'client-name=Quoth stream-properties="props,'
    'application.name=Quoth,application.icon-name=dev.quoth.Quoth"'
)
_PULSE_PROPS = "client-name=Quoth"

_SOURCES = (
    # The Pulse protocol works with both PulseAudio and PipeWire's Pulse server.
    _AudioSource("pulsesrc", "device", _PULSE_PROPS, False),
    # always-copy=true recycles pipewiresrc's finite buffer pool immediately,
    # else ~half the audio is dropped under load.
    _AudioSource(
        "pipewiresrc", "target-object", f"always-copy=true {_PIPEWIRE_PROPS}", True
    ),
    _AudioSource("autoaudiosrc", None, "", False),
)


def _pulse_devices() -> list | None:
    """Return Pulse inputs, or None when the provider cannot reach a sound server."""
    provider = Gst.DeviceProviderFactory.get_by_name("pulsedeviceprovider")
    if provider is None or not provider.start():
        return None
    try:
        return [
            dev
            for dev in provider.get_devices()
            if dev.get_device_class() == "Audio/Source"
            and dev.get_properties().get_string("device.class") != "monitor"
        ]
    finally:
        provider.stop()


def _select_source() -> _AudioSource:
    """Return the first available capture source: PulseAudio, PipeWire, then auto."""
    for source in _SOURCES:
        if Gst.ElementFactory.find(source.element):
            # A Pulse plugin can be installed without a running Pulse server.
            if source.element == "pulsesrc" and _pulse_devices() is None:
                continue
            logger.info("Audio source: {}", source.element)
            return source
    logger.warning("No known audio source element found; using autoaudiosrc")
    return _SOURCES[-1]


def list_microphones(source: _AudioSource | None = None) -> list[dict]:
    """Return the inputs, named by identifiers the capture source understands.

    Each dict has `name` (the identifier to configure), `display_name` and
    `device_class`. The list is empty when the source cannot select a device or finds
    none.
    """
    source = source or _select_source()
    if source.element == "pulsesrc":
        # DeviceMonitor may hide Pulse devices behind PipeWire's provider.
        return [
            {
                "name": dev.get_property("internal-name"),
                "display_name": dev.get_display_name(),
                "device_class": dev.get_device_class(),
            }
            for dev in _pulse_devices() or []
        ]
    if source.device_prop is None:
        return []
    monitor = Gst.DeviceMonitor.new()
    monitor.add_filter("Audio/Source", None)
    monitor.start()
    devices = monitor.get_devices()

    results = []
    for dev in devices:
        props = dev.get_properties()
        results.append(
            {
                "name": props.get_string("node.name") or dev.get_display_name(),
                "display_name": dev.get_display_name(),
                "device_class": props.get_string("media.class") or "Audio/Source",
            }
        )

    monitor.stop()
    return results


@runtime_checkable
class AudioBackend(Protocol):
    """The audio-capture seam `RecordingController` drives across a recording.

    The injected `audio_factory` builds one from `mic_source`, `on_audio_buffer`
    (float32 16 kHz mono chunks, called on the GStreamer streaming thread) and
    `on_level` (0..1 amplitude, called on the main loop by the bus watch). It is created
    once and reused; `start` and `stop` gate delivery, and with a Pulse or auto source
    also open and close the microphone.
    """

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def set_monitoring(self, enabled: bool) -> None: ...

    def list_microphones(self) -> list[dict]: ...

    def warm_up(self) -> None: ...

    def set_mic_source(self, source: str | None) -> bool: ...

    def destroy(self) -> None: ...


class AudioPipeline:
    """The GStreamer `AudioBackend`, with a whisper output and a waveform output.

    Graph, in launch syntax:
        src ! audioconvert ! audioresample ! capsfilter(F32LE, 16 kHz, mono) ! tee
        tee ! queue ! appsink: float32 buffers for whisper
        tee ! leaky queue ! high-pass ! level ! fakesink: amplitude for the meter

    Recordings never create or destroy the pipeline; only an idle mic switch and
    `warm_up()`, at startup and after such a switch, rebuild it. A Pulse or auto source
    plays only while a recording or a monitor wants audio and is stopped in between,
    which closes its stream and so the microphone. pipewiresrc resumes unreliably, so
    it stays PLAYING from its first use until `destroy()`. Either way the software gate
    (the `_recording` flag) decides whether captured audio reaches `on_audio_buffer`.
    """

    def __init__(
        self,
        mic_source: str | None = None,
        on_audio_buffer: Callable[[np.ndarray], None] | None = None,
        on_level: Callable[[float], None] | None = None,
    ):
        self._on_audio_buffer = on_audio_buffer
        self._on_level = on_level
        self._requested_mic_source = mic_source or "auto"
        self._pipeline = None
        self._appsink = None
        self._level = None
        self._chunk_count = 0
        self._recording = False
        self._monitoring = False  # an input meter wants levels without a recording
        self._pipeline_playing = False
        self._total_samples_seen = 0
        self._discarded_while_not_recording = 0
        self._warmup_timer_id: int | None = None
        self._warmup_cycle = 0
        self._baseline_samples = 0
        self._default_device: str | None = None
        self._source = _select_source()
        resolved_source = self._resolve_mic_source(self._requested_mic_source)
        self._pipeline, self._appsink, self._level = self._build_pipeline(
            resolved_source
        )
        self._applied_mic_source = resolved_source
        self._watch_pipeline(self._pipeline)

    def _resolve_mic_source(self, requested_source: str, *, warn: bool = True) -> str:
        if requested_source == "auto":
            return "auto"
        if self._source.device_prop is None:
            if warn:
                logger.warning(
                    "Audio source '{}' cannot select configured mic '{}'; using default device",
                    self._source.element,
                    requested_source,
                )
            return "auto"
        if requested_source in {
            microphone["name"] for microphone in list_microphones(self._source)
        }:
            return requested_source
        if warn:
            logger.warning(
                "Configured mic '{}' not available; using default device",
                requested_source,
            )
        return "auto"

    def _build_pipeline(self, resolved_source: str):
        src_decl = f"{self._source.element} name=src"
        if self._source.extra_props:
            src_decl += f" {self._source.extra_props}"
        pipeline_str = (
            f"{src_decl} "
            "! audioconvert "
            "! audioresample "
            f"! audio/x-raw,format=F32LE,rate={SAMPLE_RATE},channels={CHANNELS} "
            "! tee name=t "
            "t. ! queue max-size-time=30000000000 ! appsink name=sink emit-signals=true sync=false max-buffers=0 drop=false "
            # The meter branch is disposable: its leaky queue drops buffers rather than
            # back-pressure the tee, which cannot advance until every branch accepts a
            # buffer and would stall the appsink (speech) branch too.
            "t. ! queue leaky=downstream max-size-buffers=2 "
            f"! audiocheblimit mode=high-pass cutoff={METER_HIGHPASS_HZ} poles=4 "
            f"! level name=level interval={LEVEL_INTERVAL} post-messages=false ! fakesink sync=false async=false"
        )
        logger.debug("Audio pipeline: {}", pipeline_str)
        pipeline = Gst.parse_launch(pipeline_str)

        if resolved_source != "auto" and self._source.device_prop:
            pipeline.get_by_name("src").set_property(
                self._source.device_prop, resolved_source
            )
            logger.debug("Mic device applied: {}", resolved_source)

        appsink = pipeline.get_by_name("sink")
        appsink.connect("new-sample", self._on_new_sample)
        # post-messages starts off and is toggled on only while a recording or a
        # monitor (the Settings input meter) wants levels, never as idle bus traffic.
        level = pipeline.get_by_name("level")
        return pipeline, appsink, level

    def _watch_pipeline(self, pipeline) -> None:
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::element", self._on_bus_message)
        bus.connect("message::error", self._on_bus_error)
        bus.connect("message::warning", self._on_bus_warning)

    def _on_new_sample(self, appsink) -> Gst.FlowReturn:
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        self._total_samples_seen += 1

        if not self._recording:
            self._discarded_while_not_recording += 1
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if success:
            audio_data = np.frombuffer(bytes(map_info.data), dtype=np.float32).copy()
            buf.unmap(map_info)
            self._chunk_count += 1
            if self._chunk_count == 1:
                logger.debug("First audio chunk received ({} samples)", len(audio_data))
            if self._on_audio_buffer:
                self._on_audio_buffer(audio_data)

        return Gst.FlowReturn.OK

    def _on_bus_message(self, bus, message) -> None:
        """Forward the level element's RMS to `on_level` as a 0..1 amplitude."""
        struct = message.get_structure()
        if struct is None or struct.get_name() != "level":
            return

        rms = struct.get_value("rms")
        if rms and len(rms) > 0:
            # Match the post-messages gate: a monitor wants levels with no recording.
            if not (self._recording or self._monitoring):
                return
            if self._on_level:
                self._on_level(level_from_db(rms[0]))

    def _on_bus_error(self, bus, message) -> None:
        err, debug = message.parse_error()
        logger.error("GStreamer error: {} ({})", err.message, debug)
        # A pipeline in error produces nothing more. Reset it to NULL so the next start
        # or meter tries again, once per request.
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
        self._pipeline_playing = False
        # PulseAudio refuses a stream to a vanished device rather than moving it to the
        # default, so the next request looks a named device up again.
        if self._applied_mic_source != "auto":
            self._applied_mic_source = None

    def _try_playing(self, pipeline, mic: str | None) -> bool:
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            logger.warning(
                "Audio pipeline refused to start ({}, mic {})",
                self._source.element,
                mic,
            )
            pipeline.set_state(Gst.State.NULL)
            return False
        return True

    def _ensure_playing(self) -> bool:
        """Set the pipeline PLAYING unless it already is; return False when it refuses.

        SUCCESS counts, and so do ASYNC and NO_PREROLL, which a live source returns
        before its data flows: a failure after that arrives on the bus instead (see
        `_on_bus_error`). FAILURE puts the pipeline back to NULL so the next call
        tries again.
        """
        if self._pipeline_playing:
            return True
        if not self._try_playing(self._pipeline, self._applied_mic_source):
            return False
        self._pipeline_playing = True
        return True

    def _on_bus_warning(self, bus, message) -> None:
        warn, debug = message.parse_warning()
        logger.warning("GStreamer warning: {} ({})", warn.message, debug)

    def start(self):
        """Open the software gate to `on_audio_buffer`, and the microphone if shut.

        Sets the pipeline PLAYING unless it already is (pipewiresrc after `warm_up()`,
        or a running meter). Raises `AudioUnavailable`, with the gate closed again, when
        the pipeline refuses.
        """
        self._chunk_count = 0
        self._recording = True
        self._sync_level_posting()
        if not self._ensure_playing():
            # Close the gate again, or warm-up and the meter would think a recording
            # is still running.
            self._recording = False
            self._sync_level_posting()
            raise AudioUnavailable("the audio pipeline refused to start")
        logger.debug(
            "Audio pipeline started (total_seen={}, discarded={})",
            self._total_samples_seen,
            self._discarded_while_not_recording,
        )

    def set_monitoring(self, enabled: bool) -> None:
        """Post amplitude without capturing audio, for a live input meter.

        The level element sits on its own tee branch, so the capture gate stays as it
        is and nothing reaches the transcription buffer. Enabling sets the pipeline
        PLAYING; disabling stops a Pulse or auto source again unless a recording runs.
        """
        self._monitoring = enabled
        if enabled and self._pipeline is not None:
            self._ensure_playing()
        self._sync_level_posting()
        self._release_when_unused()

    def _sync_level_posting(self):
        """Let the meter post while a recording or a monitor wants it.

        Off otherwise, so it never wakes the main loop at ~30 Hz for nothing.
        """
        if self._level is not None:
            self._level.set_property(
                "post-messages", self._recording or self._monitoring
            )

    def _release_when_unused(self) -> None:
        """Stop a Pulse or auto source when no recording or meter needs it.

        NULL releases the microphone and the server connection, so the next start
        connects afresh even after a sound-server restart; PAUSED would keep both and
        replay its last few milliseconds into the next recording.
        """
        if self._recording or self._monitoring or self._source.stays_playing:
            return
        if self._pipeline_playing:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline_playing = False

    def stop(self):
        """Close the software gate, and stop a Pulse or auto source unless metering."""
        self._recording = False
        self._sync_level_posting()
        self._log_default_device()
        self._release_when_unused()
        logger.debug(
            "Audio pipeline stopped (gate closed, captured={}, total_seen={}, discarded={})",
            self._chunk_count,
            self._total_samples_seen,
            self._discarded_while_not_recording,
        )

    def _log_default_device(self) -> None:
        """Name the device `auto` captured from, once per change.

        Read before the source stops, while its stream still exists; only pulsesrc
        reports it.
        """
        if self._applied_mic_source != "auto" or self._pipeline is None:
            return
        src = self._pipeline.get_by_name("src")
        if src is None or src.find_property("current-device") is None:
            return
        device = src.get_property("current-device")
        if device and device != self._default_device:
            self._default_device = device
            logger.debug("Default input device: {}", device)

    def list_microphones(self) -> list[dict]:
        return list_microphones(self._source)

    def set_mic_source(self, source: str | None) -> bool:
        """Switch the capture device; return whether the pipeline was rebuilt.

        The rebuild connects the source to the new device. The caller must ensure
        capture is idle; `AudioUnavailable` means a monitored replacement
        refused to start and the old pipeline stays.
        """
        source = source or "auto"
        request_changed = source != self._requested_mic_source
        self._requested_mic_source = source
        if source == self._applied_mic_source:
            return False
        # Also warn when falling back after a bus error forgot the applied device.
        warn = request_changed or self._applied_mic_source is None
        resolved_source = self._resolve_mic_source(source, warn=warn)
        if resolved_source == self._applied_mic_source:
            return False
        logger.debug(
            "Mic source change: {} -> {}", self._applied_mic_source, resolved_source
        )
        self._rebuild_pipeline(resolved_source)
        if self._warmup_timer_id is not None:
            GLib.source_remove(self._warmup_timer_id)
            self._warmup_timer_id = None
        self._warmup_cycle = 0
        return True

    def warm_up(self):
        """Prime the PipeWire stream so the next recording is not lost.

        It runs at startup and again after an idle microphone switch. It sets the
        pipeline PLAYING with the gate closed, checks for real buffer flow after a short
        window and rebuilds the cold pipeline on a stall, a bounded number of times,
        while no recording runs. Non-blocking; a no-op for sources other than
        pipewiresrc and once a recording has started.
        """
        if self._pipeline is None or self._recording or not self._source.stays_playing:
            return
        if self._warmup_timer_id is not None:
            GLib.source_remove(self._warmup_timer_id)
            self._warmup_timer_id = None
        self._warmup_cycle += 1
        self._baseline_samples = self._total_samples_seen
        # A refusal is left to the check below: no buffers means a rebuild, within
        # the same bounded cycles as a stall.
        self._ensure_playing()
        self._warmup_timer_id = GLib.timeout_add(WARMUP_WINDOW_MS, self._check_warmup)

    def _check_warmup(self) -> bool:
        """Confirm buffer flow once the window elapses, else rebuild and retry."""
        self._warmup_timer_id = None
        if self._recording:  # user already started, leave the live pipeline alone
            return False
        delivered = self._total_samples_seen - self._baseline_samples
        if delivered >= WARMUP_MIN_BUFFERS:
            logger.info(
                "Audio pipeline warm ({} buffers in {}ms)", delivered, WARMUP_WINDOW_MS
            )
            return False
        if self._warmup_cycle >= WARMUP_MAX_CYCLES:
            logger.warning(
                "Audio warmup gave up after {} cycles ({} buffers): "
                "first recording may be delayed",
                self._warmup_cycle,
                delivered,
            )
            return False
        logger.debug(
            "Audio warmup stalled ({} buffers), rebuilding pipeline", delivered
        )
        try:
            resolved_source = self._resolve_mic_source(
                self._requested_mic_source, warn=False
            )
            self._rebuild_pipeline(resolved_source)
        except AudioUnavailable:
            return False  # _try_playing already said why
        except Exception:
            logger.exception("Audio warmup recovery failed")
            return False
        self.warm_up()
        return False

    def _rebuild_pipeline(self, resolved_source: str):
        """Replace the pipeline with a new one for a fresh PipeWire negotiation.

        When a recording or a monitor needs it, the replacement goes PLAYING before the
        old pipeline is dropped; a refusal raises `AudioUnavailable` and keeps the old.
        """
        candidate, appsink, level = self._build_pipeline(resolved_source)
        candidate_playing = False
        if self._recording or self._monitoring:
            if not self._try_playing(candidate, resolved_source):
                raise AudioUnavailable(
                    "the replacement audio pipeline refused to start"
                )
            candidate_playing = True

        old_pipeline = self._pipeline
        if old_pipeline is not None:
            old_pipeline.get_bus().remove_signal_watch()
            old_pipeline.set_state(Gst.State.NULL)
        self._pipeline = candidate
        self._appsink = appsink
        self._level = level
        self._applied_mic_source = resolved_source
        self._pipeline_playing = candidate_playing
        self._watch_pipeline(candidate)
        # A monitored replacement is already live; an unmonitored warmup rebuild starts
        # idle and re-primes through warm_up().
        self._sync_level_posting()

    def destroy(self):
        """Tear down the pipeline and wait for PipeWire to release its resources.

        Blocks up to 2 seconds for the NULL state to be confirmed. Called only on app
        shutdown, never between recordings.
        """
        self._recording = False
        if self._warmup_timer_id is not None:
            GLib.source_remove(self._warmup_timer_id)
            self._warmup_timer_id = None
        if self._pipeline is None:
            return

        bus = self._pipeline.get_bus()
        if bus:
            bus.remove_signal_watch()

        self._pipeline.set_state(Gst.State.NULL)
        ret, state, pending = self._pipeline.get_state(2 * Gst.SECOND)
        if ret == Gst.StateChangeReturn.SUCCESS:
            logger.debug("Audio pipeline destroyed (NULL confirmed)")
        else:
            logger.warning(
                "Audio pipeline destroy: get_state returned {} (state={}, pending={})",
                ret,
                state,
                pending,
            )

        self._pipeline = None
        self._appsink = None
        self._pipeline_playing = False
