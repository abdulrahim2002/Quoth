# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""The theme-aware view layer, where Theme meets the pure leaf renderers.

The leaf renderers (`pill_render`, `orb`) never see a Theme, so everything that
binds the two lives here. The live overlay and every preview draw through
`draw_pill`, so the pill has exactly one appearance.

`PillView` is that pill as a widget, for Settings and onboarding. `OrbView` draws a
palette as the orb, a drifting convection cell, and `OrbDirector` maps recording state
and voice level onto the orb.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from enum import StrEnum

import gi
import numpy as np

gi.require_version("Gtk", "4.0")

from gi.repository import Gtk

from quoth import orb
from quoth.pill_render import (
    BARS_START_X,
    DOT_RADIUS,
    DOT_RIPPLE,
    DOT_X,
    LABEL_LEFT_X,
    LABEL_PAD_X,
    NUM_BARS,
    PHASE_RATE,
    TRANSITION_DECAY_S,
    TRANSITION_EXPAND_S,
    TRANSITION_TOTAL_S,
    WAVE_PAD,
    WAVE_PEAK,
    LevelTrace,
    draw_background,
    draw_dot,
    draw_label,
    draw_waveform,
    format_timer,
    ink_extents,
    label_layout,
    lerp,
    lerp_color,
    tabular_advance,
    transcribing_amplitudes,
    wave_end_x,
)
from quoth.scheduler import guarded
from quoth.theme import Theme


class OverlayState(StrEnum):
    """What the overlay shows; the previews and the orb director use the same names."""

    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    ERROR = "error"
    HINT = "hint"


_SCRIPT_INTERVAL_S = 1.0 / 60.0  # scripted-wave push cadence (the trace scroll speed)
_ORB_PHYSICS_INTERVAL_S = 1.0 / 30.0  # row orbs integrate at 30 Hz; plenty for ambience
_DT_MAX_S = 0.1  # clamp on a frame delta, so a stall cannot fast-forward a view
_TIMER_WRAP_S = 3600  # a view's timer restarts hourly rather than counting forever


def animations_on(theme: Theme, motion_enabled: bool) -> bool:
    """Whether decorative motion runs: the system preference and the theme allow it."""
    return motion_enabled and theme.animation.enabled


def frame_dt(last_us: int | None, now_us: int) -> float:
    """Return the seconds between two frame-clock times: 0 at first, then clamped."""
    return 0.0 if last_us is None else min((now_us - last_us) / 1e6, _DT_MAX_S)


@dataclass(frozen=True)
class PillFrame:
    """What the pill is doing right now; the Theme says how that looks."""

    state: OverlayState
    amplitudes: list[float]
    phase: float = 0.0
    timer: str | None = None  # None = draw no timer (nothing is being timed)
    # Transcribing only: the wave frozen at the moment recording stopped, decayed
    # away over the transition, plus how far into that transition we are.
    frozen: list[float] = field(default_factory=list)
    transition_elapsed: float = 0.0
    duration_scale: float = 1.0
    motion_enabled: bool = True


def pill_width(theme: Theme, height: int) -> int:
    """Return the width that keeps `theme`'s pill at its own aspect for `height`."""
    return round(theme.geometry.width * height / theme.geometry.height)


def fit_height(theme: Theme, max_width: int) -> int:
    """Return the height at which `theme`'s pill fits `max_width`, never magnified.

    Width is the only bound needed because the theme schema keeps width >= height.
    """
    g = theme.geometry
    return max(1, round(g.height * min(1.0, max_width / g.width)))


def draw_pill(
    cr, *, theme: Theme, frame: PillFrame, width: float, height: float
) -> None:
    """Draw the pill of `theme` in `frame`'s state, filling width x height.

    Scale is derived from the theme's own height, so this is that pill at a
    different size rather than a second drawing of it.
    """
    scale = height / theme.geometry.height
    pal = theme.palette
    draw_background(
        cr,
        width=width,
        height=height,
        radius=height / 2,
        bg=pal.background,
        border=pal.border,
        border_width=theme.geometry.border_width * scale,
    )
    if frame.state == OverlayState.TRANSCRIBING:
        _draw_transcribing(cr, theme, frame, width, height, scale)
    else:
        _draw_recording(cr, theme, frame, width, height, scale)


def _draw_recording(cr, theme: Theme, frame: PillFrame, width, height, scale) -> None:
    draw_status_dot(
        cr,
        theme.palette.dot_recording,
        frame.phase,
        width,
        height,
        scale,
        ripple=frame.motion_enabled,
    )
    wave_end = _recording_wave_end(cr, theme, frame.timer, width, height, scale)
    _draw_wave(
        cr,
        theme,
        frame.amplitudes,
        BARS_START_X * scale,
        wave_end,
        theme.palette.accent,
        height,
        scale,
        gated=True,
    )
    _draw_timer(cr, theme, frame.timer, width, height, scale)


def _draw_transcribing(
    cr, theme: Theme, frame: PillFrame, width, height, scale
) -> None:
    """Draw the transcribing state: decay -> expand -> wave.

    Recording's dot, timer and bars fade out and collapse, then the bar region grows
    to fill the pill as the wave takes over, paced by `animation.duration_scale`.
    """
    pal = theme.palette
    if not frame.motion_enabled:
        _draw_wave(
            cr,
            theme,
            transcribing_amplitudes(frame.phase),
            WAVE_PAD * scale,
            width - WAVE_PAD * scale,
            pal.dot_transcribing,
            height,
            scale,
            gated=False,
        )
        return
    elapsed = frame.transition_elapsed
    decay_s = TRANSITION_DECAY_S * frame.duration_scale
    expand_s = TRANSITION_EXPAND_S * frame.duration_scale

    t_decay = min(1.0, elapsed / decay_s)
    if t_decay < 1.0:
        fade = 1.0 - t_decay
        draw_status_dot(
            cr,
            pal.dot_recording,
            frame.phase,
            width,
            height,
            scale,
            opacity=fade,
            ripple=frame.motion_enabled,
        )
        _draw_timer(cr, theme, frame.timer, width, height, scale, opacity=fade)
        _draw_wave(
            cr,
            theme,
            [a * fade for a in frame.frozen],
            BARS_START_X * scale,
            _recording_wave_end(cr, theme, frame.timer, width, height, scale),
            pal.accent,
            height,
            scale,
            gated=False,  # already fading out; the gate would only re-floor it
        )
        return

    t_expand = min(1.0, (elapsed - decay_s) / expand_s)
    line_width = theme.geometry.line_width * scale
    min_x = line_width / 2
    max_x = max(min_x, width - min_x)
    start_x = min(
        max(lerp(BARS_START_X * scale, WAVE_PAD * scale, t_expand), min_x),
        max_x,
    )
    end_x = min(
        max(
            lerp(
                wave_end_x(theme.geometry.bar_gap) * scale,
                width - WAVE_PAD * scale,
                t_expand,
            ),
            min_x,
        ),
        max_x,
    )
    _draw_wave(
        cr,
        theme,
        [a * t_expand for a in transcribing_amplitudes(frame.phase)],
        start_x,
        end_x,
        lerp_color(pal.accent, pal.dot_transcribing, t_expand),
        height,
        scale,
        gated=False,
    )


def draw_status_dot(
    cr, color, phase, width, height, scale, opacity=1.0, *, ripple=True
) -> None:
    """Draw the status dot at its inset for a pill `scale` times the theme's size."""
    radius = DOT_RADIUS * scale
    x = (
        width / 2
        if width < radius * 2
        else min(max(radius, DOT_X * scale), width - radius)
    )
    draw_dot(
        cr,
        x=x,
        y=height / 2,
        color=color,
        phase=phase,
        radius=radius,
        ripple_reach=DOT_RIPPLE * scale,
        opacity=opacity,
        ripple=ripple,
        line_width=scale,
    )


def _draw_wave(cr, theme, amplitudes, start_x, end_x, color, height, scale, *, gated):
    draw_waveform(
        cr,
        height=height,
        start_x=start_x,
        end_x=end_x,
        color=color,
        amplitudes=amplitudes,
        line_width=theme.geometry.line_width * scale,
        peak=WAVE_PEAK * scale,
        gated=gated,
    )


def _draw_timer(cr, theme, timer, width, height, scale, opacity=1.0) -> None:
    if timer is None:
        return
    draw_label(
        cr,
        timer,
        width=width,
        height=height,
        color=theme.palette.text,
        font_family=theme.font.family,
        font_size=theme.font.size * scale,
        left_x=LABEL_LEFT_X * scale,
        pad_x=LABEL_PAD_X * scale,
        align="right",
        opacity=opacity * 0.778,
    )


def _recording_wave_end(cr, theme, timer, width, height, scale) -> float:
    line_width = theme.geometry.line_width * scale
    end_x = min(
        wave_end_x(theme.geometry.bar_gap) * scale,
        width - line_width / 2,
    )
    timer_left = _timer_left_x(cr, theme, timer, width, height, scale)
    if timer_left is not None:
        end_x = min(end_x, timer_left - max(4 * scale, line_width / 2))
    return end_x


def _timer_left_x(cr, theme, timer, width, height, scale) -> float | None:
    """Return the timer ink's left edge, or None when the authored text cannot fit."""
    if timer is None:
        return None
    pad_x = LABEL_PAD_X * scale
    layout = label_layout(
        cr,
        timer,
        font_family=theme.font.family,
        font_size=theme.font.size * scale,
        max_width=width - LABEL_LEFT_X * scale - pad_x,
        tabular=True,
    )
    if layout is None:
        return None
    ink_x, _ink_y, _ink_width, ink_height = ink_extents(layout)
    if ink_height > height:
        return None
    return width - tabular_advance(layout) - pad_x + ink_x


class PillView(Gtk.DrawingArea):
    """The pill of `theme`, drawn at `height` in one of its two visible states.

    `animated` runs the dot ripple and the wave, which the system reduced-motion
    preference overrides via `set_motion_enabled`. `scripted` generates the wave
    from a clock for views with no microphone; others feed `push_level`. `timer`
    shows a clock counting the view's age, or with `times_takes` only the take that
    `start_timer` opens.
    """

    def __init__(
        self,
        theme: Theme,
        *,
        height: int,
        state: OverlayState = OverlayState.RECORDING,
        animated: bool = False,
        scripted: bool = True,
        timer: bool = False,
        times_takes: bool = False,
    ) -> None:
        super().__init__()
        self._theme = theme
        self._state = state
        self._animated = animated
        self._scripted = scripted
        self._timer = timer
        self._motion = True
        self._phase = 0.0
        self._last_frame_us: int | None = None
        self._script_acc = 0.0
        self._trace = LevelTrace()
        self._started = time.monotonic()
        # A view timing real takes (the Try-it pill) rather than its own age.
        self._timing_since: float | None = None
        self._timer_final: str | None = None
        self._times_takes = times_takes
        self._transition_start = 0.0
        self._frozen: list[float] = []
        self._tick_id: int | None = None
        self.set_content_height(height)
        self.set_content_width(pill_width(theme, height))
        self.set_draw_func(self._draw)
        if scripted:
            # A static view never ticks; priming keeps its wave from reading as silence.
            self._prime()
        self.connect("map", lambda _w: self._sync_tick())
        self.connect("unmap", lambda _w: self._sync_tick())

    def set_theme(self, theme: Theme, *, height: int | None = None) -> None:
        self._theme = theme
        h = height if height is not None else self.get_content_height()
        self.set_content_height(h)
        self.set_content_width(pill_width(theme, h))
        self._reconcile_motion()
        self._sync_tick()
        self.queue_draw()

    def set_state(self, state: OverlayState) -> None:
        """Enter `state`, animating into it as the live pill does.

        Entering TRANSCRIBING freezes the wave and starts the transition clock, so the
        bars decay from what was really there. With animations off the transition
        starts complete.
        """
        if state == self._state:
            return
        if state == OverlayState.TRANSCRIBING:
            self._frozen = self._trace.values()
            self._transition_start = time.monotonic()
        self._state = state
        self._reconcile_motion()

    def push_level(self, amplitude: float) -> None:
        """Feed one real microphone level (0..1)."""
        self._trace.push(amplitude)
        self.queue_draw()

    def start_timer(self) -> None:
        """Time a take from now, as the live pill times a recording."""
        self._times_takes = True
        self._timing_since = time.monotonic()
        self._timer_final = None
        self.queue_draw()

    def stop_timer(self, *, keep: bool = True) -> None:
        """End the take; `keep` leaves its final time up for the transcribing fade.

        Without `keep` the clock returns to its at-rest reading, as before a take.
        """
        if self._timing_since is not None and keep:
            self._timer_final = format_timer(time.monotonic() - self._timing_since)
        elif not keep:
            self._timer_final = None
        self._timing_since = None
        self.queue_draw()

    def set_motion_enabled(self, enabled: bool) -> None:
        if enabled != self._motion:
            self._motion = enabled
            self._reconcile_motion()
            self._sync_tick()

    def _animations_on(self) -> bool:
        return animations_on(self._theme, self._motion)

    def _reconcile_motion(self) -> None:
        if not self._animations_on() and self._state == OverlayState.TRANSCRIBING:
            self._transition_start = (
                time.monotonic()
                - TRANSITION_TOTAL_S * self._theme.animation.duration_scale
            )
        self.queue_draw()

    def _prime(self) -> None:
        for i in range(NUM_BARS):
            self._trace.push(_scripted_level(i * PHASE_RATE * _SCRIPT_INTERVAL_S))

    def _sync_tick(self) -> None:
        """Hold a frame-clock callback only while mapped and actually animating.

        A registered callback keeps the frame clock running, even on a hidden page,
        so a paused or unmapped view must drop it rather than return early from it.
        """
        want = (
            self._animated
            and self.get_mapped()
            and (self._animations_on() or self._timer)
        )
        if want and self._tick_id is None:
            self._last_frame_us = None  # a fresh delta baseline, not a resume jump
            self._tick_id = self.add_tick_callback(
                guarded(self._tick, on_error=self._forget_tick)
            )
        elif not want and self._tick_id is not None:
            self.remove_tick_callback(self._tick_id)
            self._tick_id = None
            self.queue_draw()

    def _forget_tick(self) -> None:
        self._tick_id = None

    def _tick(self, _widget, clock) -> bool:
        now_us = clock.get_frame_time()
        dt = frame_dt(self._last_frame_us, now_us)
        self._last_frame_us = now_us
        decorative_dt = dt / self._theme.animation.duration_scale
        if self._animations_on():
            self._phase += PHASE_RATE * decorative_dt
        if self._scripted and self._animations_on():
            # Fixed-cadence pushes keep the trace scrolling at one speed on any
            # refresh rate; the accumulator carries the remainder between frames.
            self._script_acc += decorative_dt
            while self._script_acc >= _SCRIPT_INTERVAL_S:
                self._script_acc -= _SCRIPT_INTERVAL_S
                self._trace.push(_scripted_level(self._phase))
        self.queue_draw()
        return True

    def _draw(self, _area, cr, width: int, height: int) -> None:
        draw_pill(
            cr,
            theme=self._theme,
            frame=PillFrame(
                state=self._state,
                amplitudes=self._trace.values(),
                phase=self._phase,
                timer=self._timer_text(),
                frozen=self._frozen,
                transition_elapsed=time.monotonic() - self._transition_start,
                duration_scale=self._theme.animation.duration_scale,
                motion_enabled=self._animations_on(),
            ),
            width=width,
            height=height,
        )

    def _timer_text(self) -> str | None:
        """Return the clock text, or None when the view shows no timer.

        A view that does not time takes counts its own age, wrapping hourly.
        """
        if not self._timer:
            return None
        if self._times_takes:
            if self._timing_since is not None:
                return format_timer(time.monotonic() - self._timing_since)
            # At rest the clock reads zero, so only a take makes it move.
            return self._timer_final or format_timer(0)
        return format_timer((time.monotonic() - self._started) % _TIMER_WRAP_S)


def _scripted_level(t: float) -> float:
    """Return a level at time `t` for views with no microphone.

    Two detuned sines let the wave breathe without an obvious beat.
    """
    return max(0.0, min(1.0, 0.45 + 0.3 * math.sin(t * 3.0) + 0.15 * math.sin(t * 1.7)))


# --- Orb --------------------------------------------------------------------

# Each palette role's weight in the orb, by its part in the pill, so every theme is
# weighed the same way. Background is the base fill and border the rim, so neither
# is a blob.
_ORB_BLOBS: list[tuple[str, float]] = [
    ("accent", 3.0),  # the waveform
    ("dot_transcribing", 2.5),  # colours the whole wave while transcribing
    ("text", 1.5),  # the timer
    ("dot_recording", 0.35),  # faint until recording emphasises it
    ("dot_error", 0.35),  # faint until an error emphasises it
]


def orb_blobs(theme: Theme) -> list[tuple[tuple, float]]:
    """Return the palette as weighted blobs for the orb, in pill-anatomy proportion."""
    pal = theme.palette
    return [(getattr(pal, role), weight) for role, weight in _ORB_BLOBS]


def orb_seed(theme: Theme) -> int:
    """Return a stable identity for a palette, seeding its orb's private motion.

    Hashed from the palette's colours alone with sha256 (`hash()` is salted per
    process), so the same palette is the same orb in every session. The payload
    lists the fields in a fixed order; changing it reshuffles every orb.
    """
    pal = theme.palette
    payload = repr(
        [
            pal.background,
            pal.border,
            pal.accent,
            pal.text,
            pal.dot_recording,
            pal.dot_transcribing,
            pal.dot_error,
        ]
    )
    return int.from_bytes(hashlib.sha256(payload.encode()).digest(), "big")


def make_orb(theme: Theme, *, warmed: bool = True) -> orb.Orb:
    """Build a theme's orb: colours by role, glow in the accent, motion from its seed.

    Every view builds its orb here, so a palette is always the same orb. `warmed`
    starts it mid-life rather than at its rest frame, so fresh orbs do not all open
    on the same layout; the randomness lives here so `Orb` stays deterministic.
    """
    pal = theme.palette
    o = orb.Orb(
        background=pal.background,
        blobs=orb_blobs(theme),
        rim=pal.border,
        glow=pal.accent,
        seed=orb_seed(theme),
    )
    if warmed:
        # Seek to a random instant, then advance briefly so the marbling has a head
        # start.
        o.seek(random.uniform(0.0, 600.0))
        for _ in range(int(random.uniform(10, 20))):
            o.advance(0.1)
    return o


# --- Orb director -------------------------------------------------------------

_DIRECTOR_LERP_S = 0.3  # seconds to morph between states, before duration_scale
# Voice-level smoothing: rise fast enough to ride speech, fall gently enough not
# to flicker between syllables.
_LEVEL_ATTACK = 0.5
_LEVEL_RELEASE = 0.08


def _emphasis(rest: float = 1.0, **roles: float) -> tuple[float, ...]:
    """Return an emphasis vector over the orb's blobs.

    Each palette role named in `roles` gets its factor; every other blob gets `rest`.
    """
    return tuple(roles.get(role, rest) for role, _ in _ORB_BLOBS)


class OrbDirector:
    """A mapper from recording state and voice level to `OrbParams`, always lerped.

    The orb morphs between states rather than switching, and itself knows nothing of
    recording states or palette roles. Neutral is the exact ambient look, so an idle
    director changes nothing.
    """

    def __init__(self, duration_scale: float = 1.0) -> None:
        self._duration_scale = duration_scale
        self._level = 0.0
        self._t = 0.0
        self._current = self._vector(orb.OrbParams())

    @staticmethod
    def _vector(p: orb.OrbParams) -> np.ndarray:
        emphasis = p.emphasis or _emphasis()
        return np.array(
            [p.time_scale, p.drift_scale, p.swell, p.glow, p.scale, *emphasis]
        )

    def set_duration_scale(self, duration_scale: float) -> None:
        self._duration_scale = duration_scale

    def reset(self) -> None:
        """Reset the voice level and pulse clock for a new recording.

        The params keep morphing from wherever they are, so the orb flows into the
        new state.
        """
        self._level = 0.0
        self._t = 0.0

    def push_level(self, amplitude: float) -> None:
        """Feed one microphone level (0..1), attack/release smoothed."""
        w = _LEVEL_ATTACK if amplitude > self._level else _LEVEL_RELEASE
        self._level += (amplitude - self._level) * w

    def update(
        self, dt: float, state: OverlayState, *, motion_enabled: bool = True
    ) -> orb.OrbParams:
        """Advance the morph by `dt` seconds toward `state`'s look; return the params.

        With motion off the params jump straight to the target.
        """
        if motion_enabled:
            self._t += max(dt, 0.0) / self._duration_scale
        target = self._vector(self._target(state, pulse=motion_enabled))
        if not motion_enabled:
            self._current = target
            return self._params()
        lerp_s = _DIRECTOR_LERP_S * self._duration_scale
        frac = min(1.0, max(dt, 0.0) / lerp_s)
        self._current = self._current + (target - self._current) * frac
        return self._params()

    def _params(self) -> orb.OrbParams:
        c = self._current
        return orb.OrbParams(
            time_scale=float(c[0]),
            drift_scale=float(c[1]),
            swell=float(c[2]),
            glow=float(c[3]),
            scale=float(c[4]),
            emphasis=tuple(float(v) for v in c[5:]),
        )

    def _target(self, state: OverlayState, *, pulse: bool = True) -> orb.OrbParams:
        # The floor keeps room noise reading as calm.
        level = max(0.0, (self._level - 0.12) / 0.88)
        if state == OverlayState.RECORDING:
            # The voice grows the orb and speeds its slow ambient flow several-fold.
            return orb.OrbParams(
                time_scale=1.0 + 4.0 * level,
                drift_scale=1.0 + 3.0 * level,
                swell=1.0 + 2.0 * level,
                glow=1.0 + 2.5 * level,
                scale=0.85 + 0.15 * level,
                emphasis=_emphasis(dot_recording=1.0 + 4.0 * level),
            )
        if state == OverlayState.TRANSCRIBING:
            # No voice to ride, so thinking is a fast swirl with a slow size pulse.
            return orb.OrbParams(
                time_scale=4.5,
                drift_scale=3.0,
                swell=0.8,
                glow=1.1,
                scale=0.88 + (0.05 * math.sin(self._t * 4.5) if pulse else 0.0),
                emphasis=_emphasis(rest=0.85, dot_transcribing=2.0),
            )
        if state == OverlayState.ERROR:
            return orb.OrbParams(
                time_scale=0.7,
                drift_scale=0.7,
                swell=0.5,
                glow=1.2,
                scale=0.82,
                emphasis=_emphasis(rest=0.45, dot_error=8.0),
            )
        if state == OverlayState.HINT:
            return orb.OrbParams(
                time_scale=0.7,
                drift_scale=0.7,
                swell=0.5,
                glow=0.5,
                scale=0.82,
                emphasis=_emphasis(rest=0.6),
            )
        return orb.OrbParams()


class OrbView(Gtk.DrawingArea):
    """A theme's palette as the orb, a drifting convection cell, for the theme list.

    `animated` runs the flow, which the system reduced-motion preference freezes at
    its rest frame via `set_motion_enabled`. The flow advances by frame-clock time,
    so its pace does not depend on the refresh rate.
    """

    def __init__(
        self,
        theme: Theme,
        *,
        size: int,
        animated: bool = True,
        orb_instance: orb.Orb | None = None,
        live: bool = False,
    ) -> None:
        super().__init__()
        self._theme = theme
        # A caller may hand in a running orb so a row rebuild continues its flow.
        self._orb = orb_instance if orb_instance is not None else make_orb(theme)
        # A live view maps state and the real microphone level onto the orb as the
        # overlay does; ambient views (the theme rows) have no director.
        self._director = OrbDirector(theme.animation.duration_scale) if live else None
        self._state = OverlayState.RECORDING
        self._size = size
        self._animated = animated
        self._motion = True
        self._viewports: tuple[Gtk.Viewport, ...] = ()
        self._expanders: tuple[Gtk.Widget, ...] = ()
        self._in_viewport = True
        self._last_frame_us: int | None = None
        self._physics_acc = 0.0
        self._tick_id: int | None = None
        self.set_content_width(size)
        self.set_content_height(size)
        self.set_draw_func(self._draw)
        self.connect("map", self._on_map)
        self.connect("unmap", self._on_unmap)

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        self._orb = make_orb(theme)
        if self._director is not None:
            self._director.set_duration_scale(theme.animation.duration_scale)
        self._reconcile_motion()
        self._sync_tick()
        self.queue_draw()

    def set_state(self, state: OverlayState) -> None:
        """Enter `state` (live views only; ambient views have no states)."""
        if self._director is not None and state != self._state:
            self._state = state
            self.queue_draw()

    def push_level(self, amplitude: float) -> None:
        """Feed one real microphone level (live views only)."""
        if self._director is not None:
            self._director.push_level(amplitude)

    def set_motion_enabled(self, enabled: bool) -> None:
        if enabled != self._motion:
            self._motion = enabled
            self._reconcile_motion()
            self._sync_tick()

    def _animations_on(self) -> bool:
        return animations_on(self._theme, self._motion)

    def _reconcile_motion(self) -> None:
        if not self._animations_on() and self._director is not None:
            self._orb.set_params(
                self._director.update(0.0, self._state, motion_enabled=False)
            )
        self.queue_draw()

    def _on_map(self, *_args) -> None:
        ancestors = []
        expanders = []
        parent = self.get_parent()
        while parent is not None:
            if isinstance(parent, Gtk.Viewport):
                ancestors.append(parent)
            if parent.find_property("expanded") is not None:
                expanders.append(parent)
            parent = parent.get_parent()
        self._viewports = tuple(ancestors)
        self._expanders = tuple(expanders)
        self._in_viewport = self._intersects_viewports()
        self._last_frame_us = None
        self._physics_acc = 0.0
        self._sync_tick()

    def _on_unmap(self, *_args) -> None:
        self._viewports = ()
        self._expanders = ()
        self._last_frame_us = None
        self._physics_acc = 0.0
        self._sync_tick()

    def _intersects_viewports(self) -> bool:
        if any(not widget.get_property("expanded") for widget in self._expanders):
            return False
        for viewport in self._viewports:
            ok, bounds = self.compute_bounds(viewport)
            if (
                not ok
                or bounds.get_x() + bounds.get_width() <= 0
                or bounds.get_y() + bounds.get_height() <= 0
                or bounds.get_x() >= viewport.get_width()
                or bounds.get_y() >= viewport.get_height()
            ):
                return False
        return True

    def _sync_tick(self) -> None:
        want = (
            self._animated
            and self.get_mapped()
            and (self._animations_on() or self._director is not None)
        )
        if want and self._tick_id is None:
            self._last_frame_us = None  # a fresh delta baseline, not a resume jump
            self._tick_id = self.add_tick_callback(
                guarded(self._tick, on_error=self._forget_tick)
            )
        elif not want and self._tick_id is not None:
            self.remove_tick_callback(self._tick_id)
            self._tick_id = None
            self._last_frame_us = None
            self._physics_acc = 0.0
            self.queue_draw()

    def _forget_tick(self) -> None:
        self._tick_id = None

    def _tick(self, _widget, clock) -> bool:
        now_us = clock.get_frame_time()
        # Integrating at ~30 Hz halves the dye transport cost and looks the same at
        # these speeds.
        self._physics_acc += frame_dt(self._last_frame_us, now_us)
        self._last_frame_us = now_us
        if self._physics_acc >= _ORB_PHYSICS_INTERVAL_S:
            self._in_viewport = self._intersects_viewports()
            if self._in_viewport:
                if self._director is not None:
                    self._orb.set_params(
                        self._director.update(
                            self._physics_acc,
                            self._state,
                            motion_enabled=self._animations_on(),
                        )
                    )
                if self._animations_on():
                    self._orb.advance(
                        self._physics_acc / self._theme.animation.duration_scale
                    )
                self.queue_draw()
            self._physics_acc = 0.0
        return True

    def _draw(self, _area, cr, _width: int, _height: int) -> None:
        self._orb.draw(cr, size=self._size, motion=self._animations_on())
