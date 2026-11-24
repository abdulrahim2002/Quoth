# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""The onboarding "try it" step: one guided dictation, end to end.

Drives a guided take on the app's engine (`RecordingController.begin_take` and
`end_take`), which shares the dictation's pipeline, inference lock and shutdown
join. A `Step` says where the take is, `_VISIBLE` says which elements that step
shows, and everything else follows from those two.

The page owns nothing about navigation; onboarding wraps `content` in its chrome
and drives `start`/`stop` from the navigation page's shown/hidden signals, so the
meter and any take live exactly as long as the step is visible.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, GLib, GObject, Gtk
from loguru import logger

from quoth import brand, pill_view
from quoth.animation import AnimationView
from quoth.config import Config
from quoth.download_format import EtaEstimator, format_download_progress
from quoth.downloads import DownloadCoordinator, Subscription
from quoth.models import ModelManager
from quoth.pill_view import PillView
from quoth.recording import RecordingController
from quoth.resolution import provision_for
from quoth.theme import DEFAULT_THEME

# The pill is the focus of this step, so it rides larger than the live one (48).
_PILL_HEIGHT = 84
# Below this loudest meter level (0..1) during a take, about -48 dBFS RMS, the take
# is treated as silence: above room tone, below even soft speech.
_QUIET_PEAK = 0.12
# Monitoring pushes levels continuously, silence included, so hearing nothing at all
# within this window means the capture path is dead rather than the room being quiet.
_MIC_TIMEOUT_MS = 2000
# Long enough to read as motion, short enough not to sit between the user and their
# next tap.
_REVEAL_MS = 250
_GAP = 12  # vertical rhythm between elements
_LOADER_PX = 64  # the wait's visual anchor, sized against the body text
# The frame the loader rests on, part-drawn, so a stopped loader reads as working
# rather than as the finished mark sitting there.
_LOADER_STILL = 0.08
# Wrap caps, in characters, so every text element breaks at roughly the pill's width.
# A size request is only a minimum, so a long line would widen the column instead.
_TEXT_CHARS = 38
_TRANSCRIPT_CHARS = 34  # fewer, because the transcript is set at display size
# Must be a slide: crossfade and NONE keep the child's height while hidden, leaving a
# gap for every retired element.
_REVEAL_TRANSITION = Gtk.RevealerTransitionType.SLIDE_UP


class Step(Enum):
    """Where the guided take has got to.

    Each step is a resting state except TRANSCRIBING, which resolves into SUCCESS, QUIET
    or UNCLEAR unless the step is left mid-take.
    """

    PREPARING = "preparing"
    MODEL_MISSING = "model_missing"
    READY = "ready"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    SUCCESS = "success"
    QUIET = "quiet"
    UNCLEAR = "unclear"
    MIC_FAILED = "mic_failed"


# Which elements each step shows. This is the whole layout rule: anything absent here
# is retired, so the page cannot silently accumulate widgets as the take progresses.
# The prompt stays up through RECORDING because the user is reading it aloud.
_VISIBLE: dict[Step, frozenset[str]] = {
    # Nothing to operate until the model lands: the loader carries the wait, which
    # can run for minutes, and the status line reports it.
    Step.PREPARING: frozenset({"loader", "status"}),
    Step.MODEL_MISSING: frozenset({"button"}),
    Step.READY: frozenset({"pill", "prompt", "button"}),
    Step.RECORDING: frozenset({"pill", "prompt", "button"}),
    Step.TRANSCRIBING: frozenset({"pill", "status"}),
    # No pill: beside the transcript, a live meter would only animate room noise.
    Step.SUCCESS: frozenset({"field", "praise", "button"}),
    Step.QUIET: frozenset({"pill", "prompt", "status", "button"}),
    Step.UNCLEAR: frozenset({"pill", "prompt", "status", "button"}),
    Step.MIC_FAILED: frozenset(),
}

_BUTTON_LABEL: dict[Step, str] = {
    Step.MODEL_MISSING: "Retry download",
    Step.READY: "Tap to speak",
    Step.RECORDING: "◉  Recording, tap to stop",
    Step.SUCCESS: "Say something else",
    Step.QUIET: "Try again",
    Step.UNCLEAR: "Try again",
}

# The button's weight, so only one primary action is ever on screen: after a good
# take that is the chrome's Continue.
_BUTTON_STYLE: dict[Step, str] = {
    Step.MODEL_MISSING: "suggested-action",
    Step.READY: "suggested-action",
    Step.RECORDING: "destructive-action",
    Step.SUCCESS: "",
    Step.QUIET: "suggested-action",
    Step.UNCLEAR: "suggested-action",
}

_STATUS_TEXT: dict[Step, str] = {
    # Replaced with live percent-done as the download reports it.
    Step.PREPARING: "Getting the speech model ready…",
    Step.TRANSCRIBING: "Transcribing…",
    Step.QUIET: "It's a little quiet in here.",
    Step.UNCLEAR: "Didn't quite catch that.",
}

# The page heading and lead, per step. They retire once the transcript is up, so they
# never outrank the result.
_TITLE = "Try it out"
_LEAD = "Tap to speak, say a few words, then tap again. They never leave your machine."
_HEADER: dict[Step, tuple[str, str]] = {
    # The lead promises no duration: on a slow connection this takes minutes.
    Step.PREPARING: (
        _TITLE,
        "Downloading the speech model. This only happens once.",
    ),
    Step.MODEL_MISSING: (
        "Couldn't download the speech model",
        "Quoth needs a speech model before you can try dictation. Retry the download, "
        "or finish setup and download it later in Settings.",
    ),
    Step.READY: (_TITLE, _LEAD),
    Step.RECORDING: (_TITLE, _LEAD),
    Step.TRANSCRIBING: (_TITLE, _LEAD),
    Step.QUIET: (_TITLE, _LEAD),
    Step.UNCLEAR: (_TITLE, _LEAD),
    Step.MIC_FAILED: (
        "Couldn't get microphone audio",
        "Quoth couldn't receive audio for this trial. You can finish setup "
        "and check your microphone in Settings.",
    ),
    Step.SUCCESS: ("", ""),
}

_PROMPT = "Try saying: I love private, local, and open source transcription."
_PRAISE = "Nice, that's you dictating, fully offline."
_ATTRIBUTION = "– You"


def demo_device(available: list[str]) -> str:
    """Return the device the demo transcribes on: GPU when present, else CPU.

    The GPU is compile-free and shows real acceleration. Never the NPU, whose one-time
    encoder compile would stall the first transcription; that compile stays owned by the
    calibration step.
    """
    return "GPU" if "GPU" in available else "CPU"


def outcome(text: str, peak: float) -> Step:
    """Classify a finished take as SUCCESS, QUIET or UNCLEAR.

    An empty transcript means either that nothing was said or that nothing was
    understood; `peak`, the take's loudest level, separates them so the two do not read
    as the same unhelpful failure.
    """
    if text:
        return Step.SUCCESS
    return Step.QUIET if peak < _QUIET_PEAK else Step.UNCLEAR


class TryItPage:
    """View and choreography for the guided first dictation."""

    def __init__(
        self,
        config: Config,
        model_mgr: ModelManager,
        devices: list[str],
        *,
        engine: RecordingController,
        downloads: DownloadCoordinator,
        motion_enabled: bool = True,
        on_success: Callable[[], None] | None = None,
    ):
        """Build the page; `on_success` fires each time a take comes back with words.

        It lets the page's chrome turn its skip into a continue.
        """
        self._config = config
        self._on_success = on_success
        self._model_mgr = model_mgr
        self._devices = devices
        self._engine = engine
        self._downloads = downloads
        self._motion_enabled = motion_enabled

        # The step is showing: the meter is on and the engine's levels reach the page.
        self._live = False
        self._step = Step.READY
        # Loudest level of the take, to tell "said nothing" from "said something".
        self._peak = 0.0
        self._heard = False
        self._watchdog: int | None = None
        self._eta = EtaEstimator()
        # This page's place on the model download; one at a time, however often
        # the step is entered or retried.
        self._download: Subscription | None = None

        self.content = self._build()
        self._enter(Step.READY)

    # --- lifecycle (driven by the navigation page) ---

    def start(self) -> None:
        """Bring up the live level meter. No-op while live or after a successful take.

        A dead capture path lands on MIC_FAILED and a missing model on PREPARING, so
        the step always stays passable.
        """
        if self._live:
            return
        if self._step is Step.SUCCESS:
            return
        self._peak = 0.0
        self._heard = False
        self._enter(Step.READY if self._model_ready() else Step.PREPARING)
        if self._step is Step.PREPARING:
            self._await_model()
        self._live = True
        try:
            self._engine.set_input_monitoring(self, True)
        except Exception:
            logger.exception("Demo capture unavailable")
            self._live = False
            self._enter(Step.MIC_FAILED)
            return
        self._arm_watchdog()

    def _model_ready(self) -> bool:
        return self._model_mgr.is_downloaded(self._config.model)

    def _await_model(self) -> None:
        """Join or start the model's download and follow it to READY or MODEL_MISSING.

        Onboarding starts the download when its window opens, so this usually joins it.
        """
        self._eta = EtaEstimator()  # a retry re-estimates from scratch
        # Join before leaving the old place, so the transfer never sees nobody
        # waiting and cancels.
        previous = self._download
        model = self._config.model
        self._download = self._downloads.start_or_join(
            model,
            provision_for(model, self._config.device, self._model_mgr),
            on_progress=self._on_model_progress,
            on_done=self._on_model_done,
            on_error=lambda _msg: self._on_model_failed(),
        )
        if previous is not None:
            previous.cancel()

    def _on_model_progress(self, progress) -> None:
        if self._step is not Step.PREPARING:
            return
        self._status.set_label(format_download_progress(progress))
        # The lead carries the estimate once there is a trustworthy one.
        eta = self._eta.update(progress)
        if eta is not None:
            self._description.set_label(
                f"Downloading the speech model. {eta.capitalize()}."
            )

    def _on_model_done(self) -> None:
        # Only claim readiness if the step is still waiting: the user may have left,
        # or a retry may have already moved things on.
        if self._step is not Step.PREPARING:
            return
        self._enter(Step.READY)
        # The one-shot mic watchdog may have fired unheeded during the wait. Re-arm it,
        # or a dead mic would read as a quiet room.
        self._arm_watchdog()

    def _on_model_failed(self) -> None:
        if self._step is Step.PREPARING:
            logger.warning("Demo model unavailable; the step stays passable")
            self._enter(Step.MODEL_MISSING)

    def stop(self) -> None:
        """Drop any take in flight, the model wait and the meter on leaving the step."""
        self._cancel_watchdog()
        if self._download is not None:
            self._download.cancel()
            self._download = None
        if self._step in (Step.RECORDING, Step.TRANSCRIBING):
            self._engine.cancel()
        if self._live:
            self._live = False
            self._engine.set_input_monitoring(self, False)
        if self._step is not Step.SUCCESS:
            self._enter(Step.READY)

    def _arm_watchdog(self) -> None:
        if not self._heard and self._watchdog is None:
            self._watchdog = GLib.timeout_add(_MIC_TIMEOUT_MS, self._on_mic_silent)

    def _cancel_watchdog(self) -> None:
        if self._watchdog is not None:
            GLib.source_remove(self._watchdog)
            self._watchdog = None

    def _on_mic_silent(self) -> bool:
        self._watchdog = None
        # Only a step that expected to be usable can fail on the mic; while the model
        # is still arriving there is nothing for a dead mic to interrupt.
        if not self._heard and self._step is Step.READY:
            logger.warning("Demo mic delivered no levels in {}ms", _MIC_TIMEOUT_MS)
            self._enter(Step.MIC_FAILED)
        return False

    def set_motion_enabled(self, enabled: bool) -> None:
        """Apply the reduced-motion preference to the pill, loader and reveals."""
        if enabled == self._motion_enabled:
            return
        self._motion_enabled = enabled
        self._pill.set_motion_enabled(enabled)
        self._loader.set_motion_enabled(enabled)
        for revealer in self._revealers.values():
            revealer.set_transition_duration(_REVEAL_MS if enabled else 0)

    # --- view ---

    def _build(self) -> Gtk.Widget:
        column = brand.status_column(None, *_HEADER[Step.READY])
        self._heading = column.get_first_child()
        self._description = column.get_last_child()
        # No box spacing: it would apply between collapsed revealers too, leaving a
        # gap per retired element. The gap lives on the children instead, so it
        # collapses with them.
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        box.set_halign(Gtk.Align.CENTER)
        box.set_margin_top(12)

        self._pill = PillView(
            DEFAULT_THEME,
            height=_PILL_HEIGHT,
            animated=True,
            scripted=False,
            timer=True,
            times_takes=True,  # the clock rests at 0:00 until a take
        )
        self._pill.set_motion_enabled(self._motion_enabled)
        self._pill.set_halign(Gtk.Align.CENTER)

        self._prompt = Gtk.Label(label=_PROMPT)
        self._prompt.add_css_class("dim-label")
        self._prompt.set_wrap(True)
        self._prompt.set_justify(Gtk.Justification.CENTER)
        self._prompt.set_max_width_chars(_TEXT_CHARS)

        self._status = Gtk.Label()
        self._status.add_css_class("dim-label")
        self._status.set_wrap(True)
        self._status.set_justify(Gtk.Justification.CENTER)
        self._status.set_max_width_chars(_TEXT_CHARS)

        # The transcript is set as display type, not in an entry, which would read as a
        # disabled form control.
        self._field = Gtk.Label()
        self._field.add_css_class("title-2")
        self._field.set_wrap(True)
        self._field.set_justify(Gtk.Justification.CENTER)
        self._field.set_max_width_chars(_TRANSCRIPT_CHARS)
        self._field.set_halign(Gtk.Align.CENTER)

        # The attribution frames the transcript as something the user said.
        attribution = Gtk.Label(label=_ATTRIBUTION)
        attribution.add_css_class("dim-label")
        attribution.set_halign(Gtk.Align.END)
        self._quote = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._quote.set_halign(Gtk.Align.CENTER)
        self._quote.append(self._field)
        self._quote.append(attribution)

        self._praise = Gtk.Label(
            label=_PRAISE, wrap=True, justify=Gtk.Justification.CENTER
        )
        self._praise.add_css_class("dim-label")

        # Tap to start recording, tap again to stop and transcribe.
        self._button = Gtk.Button()
        self._button.add_css_class("pill")
        self._button.add_css_class("suggested-action")
        self._button.set_halign(Gtk.Align.CENTER)
        brand.pointer_cursors(self._button)
        # The keyboard focus ring of a filled button sits 3px outside it, and the
        # revealer below clips at its edge; the margin keeps the ring in frame.
        self._button.set_margin_top(3)
        self._button.set_margin_bottom(3)
        self._button.connect("clicked", self._toggle)

        self._loader = AnimationView(
            "loader",
            loop=True,
            still_fraction=_LOADER_STILL,
            motion_enabled=self._motion_enabled,
        )
        self._loader.set_size_request(_LOADER_PX, _LOADER_PX)

        self._revealers: dict[str, Gtk.Revealer] = {}
        for key, child in (
            ("pill", self._pill),
            ("loader", self._loader),
            ("prompt", self._prompt),
            ("status", self._status),
            ("field", self._quote),
            ("praise", self._praise),
            ("button", self._button),
        ):
            if key != "pill":
                child.set_margin_top(_GAP)
            revealer = Gtk.Revealer(child=child)
            revealer.set_transition_type(_REVEAL_TRANSITION)
            # Revealer honours only gtk-enable-animations; zeroing the duration also
            # follows the reduced-motion portal.
            revealer.set_transition_duration(_REVEAL_MS if self._motion_enabled else 0)
            revealer.set_halign(Gtk.Align.CENTER)
            self._revealers[key] = revealer
            box.append(revealer)

        column.append(box)
        compact = Adw.Breakpoint.new(Adw.BreakpointCondition.parse("max-width: 480sp"))
        compact.add_setter(self._pill, "content-height", GObject.Value(int, 56))
        compact.add_setter(
            self._pill,
            "content-width",
            GObject.Value(int, pill_view.pill_width(DEFAULT_THEME, 56)),
        )
        self.breakpoints = (compact,)
        return Adw.Clamp(maximum_size=brand.STATUS_WIDTH, child=column)

    def _enter(self, step: Step) -> None:
        """Move to `step`: set its copy, then reveal exactly its elements."""
        logger.debug("Try it: {}", step.name)
        self._step = step
        if step is Step.SUCCESS and self._on_success is not None:
            self._on_success()
        title, lead = _HEADER[step]
        self._heading.set_label(title)
        self._description.set_label(lead)
        self._button.set_label(_BUTTON_LABEL.get(step, ""))
        self._status.set_label(_STATUS_TEXT.get(step, ""))
        for style in ("suggested-action", "destructive-action"):
            self._button.remove_css_class(style)
        if wanted := _BUTTON_STYLE.get(step, ""):
            self._button.add_css_class(wanted)
        self._pill.set_state(
            pill_view.OverlayState.TRANSCRIBING
            if step is Step.TRANSCRIBING
            else pill_view.OverlayState.RECORDING
        )
        # The timer runs for the take itself and fades out with the transcribing
        # transition, as on the live pill; the resting steps show none.
        if step is Step.RECORDING:
            self._pill.start_timer()
        else:
            self._pill.stop_timer(keep=step is Step.TRANSCRIBING)
        visible = _VISIBLE[step]
        for key, revealer in self._revealers.items():
            revealer.set_reveal_child(key in visible)
        # Retiring the focused button moves the focus up to the page around it, or
        # nowhere; it returns with the button.
        root = self.content.get_root()
        if "button" in visible and root is not None:
            focus = root.get_focus()
            if focus is None or self._button.is_ancestor(focus):
                self._button.grab_focus()
        if step in (Step.MODEL_MISSING, Step.MIC_FAILED):
            self.content.announce(
                f"{title}. {lead}", Gtk.AccessibleAnnouncementPriority.MEDIUM
            )
        # Only the visible loader ticks: a hidden one would redraw for every step
        # that follows the wait.
        if "loader" in visible:
            self._loader.start()
        else:
            self._loader.stop()

    # --- capture ---

    def push_level(self, amplitude: float) -> None:
        """Take one input level from the engine; ignored while the step is hidden."""
        if not self._live:
            return
        if not self._heard:
            self._heard = True
            self._cancel_watchdog()
            # A mic that starts after the watchdog gave up on it works after all.
            if self._step is Step.MIC_FAILED:
                self._enter(Step.READY)
        if self._step is Step.RECORDING:
            self._peak = max(self._peak, amplitude)
        self._pill.push_level(amplitude)

    def _toggle(self, _button) -> None:
        # A retired button still takes clicks while it slides out.
        if "button" not in _VISIBLE[self._step]:
            return
        if self._step is Step.MODEL_MISSING:
            self._enter(Step.PREPARING)
            self._await_model()
            return
        if self._step is Step.SUCCESS and not self._live:
            self._enter(Step.READY)
            self.start()
            if self._step is not Step.READY:
                return
        if not self._live:
            return
        if self._step is Step.RECORDING:
            self._enter(Step.TRANSCRIBING)
            self._engine.end_take(demo_device(self._devices), self._on_text)
        else:
            self._peak = 0.0
            self._field.set_text("")
            began = self._engine.begin_take()
            if not began:
                logger.warning(
                    "Demo take did not start ({})",
                    "microphone unavailable" if self._engine.is_idle else "engine busy",
                )
            self._enter(Step.RECORDING if began else Step.MIC_FAILED)

    def _on_text(self, text: str) -> None:
        if self._step is not Step.TRANSCRIBING:
            return
        step = outcome(text, self._peak)
        if step is Step.SUCCESS:
            self._field.set_text(f"\u201c{text}\u201d")
        self._enter(step)
        message = (
            f"Dictation complete: {text}"
            if step is Step.SUCCESS
            else _STATUS_TEXT[step]
        )
        self.content.announce(message, Gtk.AccessibleAnnouncementPriority.MEDIUM)
