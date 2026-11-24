# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Onboarding's animation widgets: Lottie playback and the Cairo-drawn page pictures.

`AnimationView` plays a Lottie document into a DrawingArea, once (the intro) or on a
loop (the loader), driving a frame from the widget's frame clock and rendering it
through ThorVG. When libthorvg is absent the view draws nothing and `play` reports done
at once.

`Ripple`, `Halo`, `Breakwater` and `DotField` draw with Cairo directly. Under reduced
motion each holds a still frame or draws nothing.
"""

from __future__ import annotations

import math
from pathlib import Path

import cairo
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, GLib, Gtk

from quoth import lottie
from quoth.easing import ease
from quoth.scheduler import guarded

_ANIM_DIR = Path(__file__).parent / "animations"


class AnimationView(Gtk.DrawingArea):
    """A Lottie animation as a widget, played once or on a loop.

    One-shot (the intro): `play(on_done)` runs it once and fires `on_done`; `skip()`
    jumps to the end. Loop (the loader): `start()`/`stop()`, holding the frame at
    `still_fraction` when stopped. `loop` picks the shape at construction.
    """

    def __init__(
        self,
        name: str,
        *,
        loop: bool = False,
        still_fraction: float = 0.0,
        end_fraction: float = 1.0,
        motion_enabled: bool = True,
    ) -> None:
        """Load the animation `name`; a one-shot rests at `end_fraction` of the clip.

        That lets a clip that draws and then undraws play up to its drawn moment and
        hold there.
        """
        super().__init__(accessible_role=Gtk.AccessibleRole.PRESENTATION)
        self.add_css_class("quoth-animation")
        self._anim = lottie.load(str(_ANIM_DIR / f"{name}.json"))
        self._loop = loop
        self._still = still_fraction
        self._end = end_fraction
        self._motion = motion_enabled
        self._loop_requested = False
        self._frame = self._still_frame()
        self._on_done = None
        self._tick_id: int | None = None
        self._start_us: int | None = None
        self._color: tuple[float, float, float, float] | None = None
        self.set_draw_func(self._draw)
        self.connect("unrealize", lambda _w: self._teardown())

    # --- geometry ---

    def _total(self) -> float:
        return self._anim.total_frames if self._anim else 0.0

    def _still_frame(self) -> float:
        return self._still * self._total()

    def _end_frame(self) -> float:
        return self._end * self._total()

    # --- one-shot (intro) ---

    def play(self, on_done=None) -> None:
        """Run the clip once and fire `on_done` when it ends.

        When motion is off, or libthorvg is absent, it jumps to the end and fires
        `on_done` at once (still deferred to the main loop), so a caller can sequence on
        it either way.
        """
        self._on_done = on_done
        if self._anim is None or not self._motion:
            self._frame = self._end_frame()
            self.queue_draw()
            self._fire_done()
            return
        self._run()

    def skip(self) -> None:
        """Jump to the finished frame and fire `on_done`."""
        self._stop_tick()
        self._frame = self._end_frame()
        self.queue_draw()
        self._fire_done()

    # --- loop (loader) ---

    def start(self) -> None:
        """Begin looping. Idempotent.

        When motion is off, or libthorvg is absent, it only holds the still frame, so a
        reduced-motion loader costs nothing.
        """
        self._loop_requested = True
        if self._anim is None or not self._motion:
            self._frame = self._still_frame()
            self.queue_draw()
            return
        if self._tick_id is None:
            self._run()

    def stop(self) -> None:
        """Halt and hold the still frame."""
        self._loop_requested = False
        self._stop_tick()
        self._frame = self._still_frame()
        self.queue_draw()

    # --- shared ---

    def set_motion_enabled(self, enabled: bool) -> None:
        """Settle active motion now; requested loops resume when motion returns."""
        if enabled == self._motion:
            return
        self._motion = enabled
        if self._loop:
            if enabled and self._loop_requested and self._anim is not None:
                self._run()
            elif not enabled:
                self._stop_tick()
                self._frame = self._still_frame()
                self.queue_draw()
            return
        if not enabled:
            self._stop_tick()
            self._frame = self._end_frame()
            self.queue_draw()
            self._fire_done()

    def _run(self) -> None:
        # Drive from the widget's frame clock, capturing the start on the first tick
        # (in `_advance`) rather than here: at play the clock's frame time can be
        # stale, and an animation started against it would jump straight to its end.
        self._stop_tick()
        self._start_us = None
        self._tick_id = self.add_tick_callback(
            guarded(self._advance, on_error=self._forget_tick)
        )

    def _advance(self, _widget, clock) -> bool:
        now = clock.get_frame_time()  # microseconds, monotonic
        if self._start_us is None:
            self._start_us = now
        duration_ms = self._anim.duration_s * 1000.0
        elapsed_ms = (now - self._start_us) / 1000.0
        total = self._total()
        if self._loop:
            # ThorVG carries the keyframe easing, so time maps linearly onto frame.
            phase = (elapsed_ms % duration_ms) / duration_ms if duration_ms else 0.0
            self._frame = phase * total
            self.queue_draw()
            return GLib.SOURCE_CONTINUE
        duration_ms *= self._end  # a one-shot ends where it is told to
        if elapsed_ms >= duration_ms:
            self._frame = self._end_frame()
            self.queue_draw()
            # Returning REMOVE drops the tick; clear the id so _stop_tick won't re-drop.
            self._tick_id = None
            self._fire_done()
            return GLib.SOURCE_REMOVE
        self._frame = (
            (elapsed_ms / duration_ms) * self._end_frame()
            if duration_ms
            else self._end_frame()
        )
        self.queue_draw()
        return GLib.SOURCE_CONTINUE

    def _forget_tick(self) -> None:
        self._tick_id = None

    def _stop_tick(self) -> None:
        if self._tick_id is not None:
            self.remove_tick_callback(self._tick_id)
            self._tick_id = None

    def _fire_done(self) -> None:
        # Deferred: under reduced motion play() fires done synchronously, and a caller's
        # on_done (revealing the page) must not re-enter that call.
        if self._on_done is not None:
            callback, self._on_done = self._on_done, None

            def complete() -> bool:
                callback()
                return GLib.SOURCE_REMOVE

            GLib.idle_add(guarded(complete))

    def _teardown(self) -> None:
        self._stop_tick()
        if self._anim is not None:
            self._anim.close()
            self._anim = None

    def _draw(self, _area, cr: cairo.Context, width: int, height: int) -> None:
        if self._anim is None or width <= 0 or height <= 0:
            return
        foreground = self.get_color()
        color = (foreground.red, foreground.green, foreground.blue, foreground.alpha)
        if color != self._color:
            self._color = color
            self._anim.apply_ink(color)

        # ThorVG stretches its render to the target, so fit the document's own aspect
        # inside the allocation, centred, and letterbox the rest.
        scale = min(width / self._anim.width, height / self._anim.height)
        box_w = self._anim.width * scale
        box_h = self._anim.height * scale
        offset_x = (width - box_w) / 2.0
        offset_y = (height - box_h) / 2.0

        # Render at device resolution so the raster stays crisp on HiDPI, then scale the
        # blit back down to the logical letterbox.
        factor = self.get_scale_factor()
        px_w = max(1, round(box_w * factor))
        px_h = max(1, round(box_h * factor))
        surface = self._anim.render(px_w, px_h, self._frame)
        if surface is None:
            return

        cr.save()
        cr.translate(offset_x, offset_y)
        cr.scale(box_w / px_w, box_h / px_h)
        pattern = cairo.SurfacePattern(surface)
        pattern.set_filter(cairo.FILTER_GOOD)
        cr.set_source(pattern)
        cr.paint()
        cr.restore()


# The ripple: a single hairline ring that leaves the mark and fades as it goes.
_RIPPLE_MS = 800
_RIPPLE_REACH = 1.5  # final radius, as a multiple of the mark's


class Ripple(Gtk.DrawingArea):
    """One ring sent out from the centre on `play()`: the pill's recording pulse, once.

    Sized larger than what it surrounds, so the ring has room; it never takes input.
    Nothing draws before `play()` or when motion is off.
    """

    def __init__(self, mark_px: int, *, motion_enabled: bool = True) -> None:
        super().__init__(accessible_role=Gtk.AccessibleRole.PRESENTATION)
        self.add_css_class("quoth-animation")
        self._mark_radius = mark_px / 2
        self._motion = motion_enabled
        self._progress: float | None = None
        self._animation: Adw.TimedAnimation | None = None
        size = round(mark_px * _RIPPLE_REACH) + 2
        self.set_size_request(size, size)
        self.set_can_target(False)
        self.set_draw_func(self._draw)

    def play(self) -> None:
        if not self._motion:
            return

        def step(value: float | None) -> None:
            self._progress = value
            self.queue_draw()

        animation = Adw.TimedAnimation.new(
            self, 0.0, 1.0, _RIPPLE_MS, Adw.CallbackAnimationTarget.new(step)
        )
        animation.set_easing(Adw.Easing.EASE_OUT_CUBIC)

        def done(_animation) -> None:
            step(None)
            self._animation = None

        animation.connect("done", done)
        self._animation = animation  # kept so reduced motion can skip it
        animation.play()

    def set_motion_enabled(self, enabled: bool) -> None:
        if enabled == self._motion:
            return
        self._motion = enabled
        if not enabled and self._animation is not None:
            self._animation.skip()
        if not enabled:
            self._progress = None
            self.queue_draw()

    def _draw(self, _area, cr: cairo.Context, width: int, height: int) -> None:
        if self._progress is None:
            return
        progress = self._progress
        color = self.get_color()
        cr.set_source_rgba(
            color.red,
            color.green,
            color.blue,
            color.alpha * (1.0 - progress) ** 1.5,
        )
        cr.set_line_width(1.0)
        radius = self._mark_radius * (1.0 + (_RIPPLE_REACH - 1.0) * progress)
        cr.arc(width / 2, height / 2, radius, 0, 2 * math.pi)
        cr.stroke()


# The halo: a listening pulse shaped to a key, Tailwind's ping slowed down. Each ring
# is gone by three quarters of the period, and fades in so it never reads as the key's
# own border flashing.
HALO_REACH_PX = 16
_HALO_PERIOD_S = 1.6
_HALO_TRAVEL = 0.75  # of the period; ping's keyframes end at 75%
_HALO_FADE_IN = 0.12  # of the travel
_HALO_ALPHA = 0.5
_HALO_STILL = 0.35  # the travel shown, unmoving, while on under reduced motion


class Halo(Gtk.DrawingArea):
    """A pulse around a rounded rectangle inset `HALO_REACH_PX` inside the allocation.

    Whatever it surrounds sits inset by that much. `start()` sends rings out
    continuously; `stop()` lets the ring in flight finish and sends no more. Starting
    again while it finishes carries the loop on, so quick toggles never restart, stack
    or cut a ring. Under reduced motion it holds one still ring while started.
    """

    def __init__(self, corner_px: float, *, motion_enabled: bool = True) -> None:
        super().__init__(accessible_role=Gtk.AccessibleRole.PRESENTATION)
        self.add_css_class("quoth-animation")
        self._corner = corner_px
        self._motion = motion_enabled
        self._on = False
        self._born_us: int | None = None  # when the ring in flight left the edge
        self._progress: float | None = None  # that ring's travel, 0..1; None if none
        self._tick_id: int | None = None
        self.set_can_target(False)
        self.set_draw_func(self._draw)
        self.connect("unmap", lambda _w: self._rest())

    def start(self) -> None:
        self._on = True
        if not self._motion:
            self.queue_draw()
            return
        if self._tick_id is None:
            self._born_us = None
            self._tick_id = self.add_tick_callback(
                guarded(self._advance, on_error=self._forget_tick)
            )

    def stop(self) -> None:
        self._on = False
        if not self._motion:
            self.queue_draw()

    def set_motion_enabled(self, enabled: bool) -> None:
        if enabled == self._motion:
            return
        self._motion = enabled
        if not enabled:
            if self._tick_id is not None:
                self.remove_tick_callback(self._tick_id)
                self._tick_id = None
            self._born_us = None
            self._progress = _HALO_STILL if self._on else None
            self.queue_draw()
        elif self._on:
            self.start()

    def _forget_tick(self) -> None:
        self._tick_id = None

    def _rest(self) -> None:
        self._on = False
        self._progress = None
        if self._tick_id is not None:
            self.remove_tick_callback(self._tick_id)
            self._tick_id = None

    def _advance(self, _widget, clock) -> bool:
        now = clock.get_frame_time()
        if self._born_us is None:
            self._born_us = now
        elapsed = (now - self._born_us) / (_HALO_PERIOD_S * 1e6)
        if elapsed >= _HALO_TRAVEL and not self._on:
            # Stopped: the ring in flight has finished, so send no more.
            self._tick_id = None
            self._progress = None
            self.queue_draw()
            return GLib.SOURCE_REMOVE
        if elapsed >= 1.0:
            self._born_us = now
            elapsed = 0.0
        self._progress = elapsed / _HALO_TRAVEL if elapsed < _HALO_TRAVEL else None
        self.queue_draw()
        return GLib.SOURCE_CONTINUE

    def _draw(self, _area, cr: cairo.Context, width: int, height: int) -> None:
        if not self._motion:
            progress = _HALO_STILL if self._on else None
        else:
            progress = self._progress
        if progress is None:
            return
        grow = HALO_REACH_PX * ease("ease-out-cubic", progress)
        inset = HALO_REACH_PX - grow
        alpha = min(1.0, progress / _HALO_FADE_IN) * (1.0 - progress) ** 1.4
        color = self.get_color()
        cr.set_source_rgba(
            color.red, color.green, color.blue, color.alpha * _HALO_ALPHA * alpha
        )
        cr.set_line_width(1.0)
        _rounded_rect(
            cr,
            inset + 0.5,
            inset + 0.5,
            width - 2 * inset - 1,
            height - 2 * inset - 1,
            self._corner + grow,
        )
        cr.stroke()


def _rounded_rect(
    cr: cairo.Context, x: float, y: float, w: float, h: float, radius: float
) -> None:
    radius = min(radius, w / 2, h / 2)
    cr.new_sub_path()
    cr.arc(x + w - radius, y + radius, radius, -math.pi / 2, 0)
    cr.arc(x + w - radius, y + h - radius, radius, 0, math.pi / 2)
    cr.arc(x + radius, y + h - radius, radius, math.pi / 2, math.pi)
    cr.arc(x + radius, y + radius, radius, math.pi, 3 * math.pi / 2)
    cr.close_path()


# The breakwater, the site's "Private" showcase: waves leave the centre and break
# against a ring of fins; nothing crosses. Geometry and timing match the site's
# 240-unit square.
_BREAKWATER_VIEW = 240.0
_BREAKWATER_PERIOD = 4.2  # seconds between two waves leaving the source
_BREAKWATER_TRAVEL = 2.15  # seconds from the source to the ring
_BREAKWATER_BREAK = 1.35  # seconds from contact until the crests die
_BREAKWATER_CONTACT = 92.0
_BREAKWATER_FINS = 48  # also the number of crests a wave breaks into
_BREAKWATER_WAVES = 2
_GOLDEN_ANGLE = 2.39996  # per-index variation that never lines up into a pattern
# Under reduced motion: two fronts on their way out, the fins at rest.
_BREAKWATER_STILL = ([(36.0, 0.45), (72.0, 0.22)], [], 0.0, 0.0)


def _elastic_out(p: float) -> float:
    """Ease like GSAP's elastic.out(1, 0.6): the fins' recoil after an impact."""
    if p >= 1.0:
        return 1.0
    return 2.0 ** (-10.0 * p) * math.sin((p - 0.15) * math.tau / 0.6) + 1.0


def _polar(radius: float, angle: float) -> tuple[float, float]:
    return math.cos(angle) * radius, math.sin(angle) * radius


def _crests(breaking: float) -> list:
    """Return a broken wave's crests at `breaking`, its progress through the break.

    The crests gather along the face, then shear sideways as they recede. Each is a
    quadratic curve (start, control, end).
    """
    crests = []
    for sector in range(_BREAKWATER_FINS):
        variation = (math.sin(sector * _GOLDEN_ANGLE) + 1) / 2
        decay = min(1.0, max(0.0, breaking - 0.1) / 0.9 * (0.85 + 0.3 * variation))
        release = ease("ease-out-quad", decay)
        centre = sector * math.tau / _BREAKWATER_FINS + release * (
            0.025 + 0.055 * variation
        )
        half = math.pi / _BREAKWATER_FINS * (1 - release)
        radius = _BREAKWATER_CONTACT - decay * (5 + 8 * variation)
        curl = math.sin(decay * math.pi) * (3 + 5 * variation)
        crests.append(
            (
                _polar(radius, centre - half),
                _polar(radius - curl, centre),
                _polar(radius, centre + half),
            )
        )
    return crests


def _breakwater_frame(t: float):
    """Return the scene at `t` seconds as `(fronts, breaking_waves, impact, push)`.

    Fronts are travelling rings (radius, alpha), breaking waves are (alpha, crests), and
    `impact` and `push` say how hard the fins are hit and pushed.
    """
    fronts, breaking_waves, impact, push = [], [], 0.0, 0.0
    for index in range(_BREAKWATER_WAVES):
        age = (t + index * _BREAKWATER_PERIOD / _BREAKWATER_WAVES) % _BREAKWATER_PERIOD
        if age < _BREAKWATER_TRAVEL:
            progress = age / _BREAKWATER_TRAVEL
            # Near-linear travel makes the arrival read as contact, not a fade-out.
            radius = 8 + (_BREAKWATER_CONTACT - 8) * progress
            fronts.append((radius, min(1.0, age / 0.16) * (0.5 - 0.18 * progress)))
            continue
        breaking = (age - _BREAKWATER_TRAVEL) / _BREAKWATER_BREAK
        if breaking >= 1:
            continue
        arrival = math.sin(min(1.0, breaking / 0.12) * math.pi / 2)
        impact = max(impact, arrival * (1 - breaking) ** 2)
        recoil = 1 - _elastic_out((breaking - 0.12) / 0.88)
        push = max(push, 0.8 * (arrival if breaking < 0.12 else recoil))
        alpha = (0.32 + 0.24 * arrival) * (1 - breaking) ** 1.4
        breaking_waves.append((alpha, _crests(breaking)))
    return fronts, breaking_waves, impact, push


def _quad_to(cr: cairo.Context, start, control, end) -> None:
    """Add a quadratic Bezier to the path as the equivalent cubic, which cairo draws."""
    (x0, y0), (qx, qy), (x2, y2) = start, control, end
    cr.move_to(x0, y0)
    cr.curve_to(
        x0 + 2 / 3 * (qx - x0),
        y0 + 2 / 3 * (qy - y0),
        x2 + 2 / 3 * (qx - x2),
        y2 + 2 / 3 * (qy - y2),
        x2,
        y2,
    )


class Breakwater(Gtk.DrawingArea):
    """The privacy picture: ripples from a source break on a ring and never cross it.

    Runs while mapped and motion is on; holds a still frame otherwise.
    """

    def __init__(self, size_px: int, *, motion_enabled: bool = True) -> None:
        super().__init__(accessible_role=Gtk.AccessibleRole.PRESENTATION)
        self.add_css_class("quoth-animation")
        self._motion = motion_enabled
        self._start_us: int | None = None
        self._t: float | None = None  # seconds since the first frame; None at rest
        self._tick_id: int | None = None
        self.set_size_request(size_px, size_px)
        self.set_draw_func(self._draw)
        self.connect("map", lambda _w: self._start())
        self.connect("unmap", lambda _w: self._stop())

    def _start(self) -> None:
        if not self._motion or self._tick_id is not None:
            return
        self._start_us = None
        self._tick_id = self.add_tick_callback(
            guarded(self._advance, on_error=self._forget_tick)
        )

    def _forget_tick(self) -> None:
        self._tick_id = None

    def _stop(self) -> None:
        if self._tick_id is not None:
            self.remove_tick_callback(self._tick_id)
            self._tick_id = None

    def set_motion_enabled(self, enabled: bool) -> None:
        if enabled == self._motion:
            return
        self._motion = enabled
        if enabled:
            if self.get_mapped():
                self._start()
        else:
            self._stop()
            self._t = None
            self.queue_draw()

    def _advance(self, _widget, clock) -> bool:
        now = clock.get_frame_time()
        if self._start_us is None:
            self._start_us = now
        self._t = (now - self._start_us) / 1e6
        self.queue_draw()
        return GLib.SOURCE_CONTINUE

    def _draw(self, _area, cr: cairo.Context, width: int, height: int) -> None:
        frame = _BREAKWATER_STILL if self._t is None else _breakwater_frame(self._t)
        fronts, breaking_waves, impact, push = frame
        color = self.get_color()
        r, g, b, color_alpha = color.red, color.green, color.blue, color.alpha
        scale = min(width, height) / _BREAKWATER_VIEW
        cr.translate(width / 2, height / 2)
        cr.scale(scale, scale)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)

        def stroke(alpha: float, line_width: float) -> None:
            cr.set_source_rgba(r, g, b, color_alpha * alpha)
            cr.set_line_width(line_width)
            cr.stroke()

        cr.arc(0, 0, 104, 0, math.tau)
        stroke(0.14, 0.75)
        for fin in range(_BREAKWATER_FINS):
            variation = math.sin(fin * _GOLDEN_ANGLE)
            depth = 97 + 2 * variation
            cr.save()
            cr.rotate(fin * math.tau / _BREAKWATER_FINS)
            cr.translate(push * (0.8 + 0.2 * variation), 0)
            _quad_to(cr, (92, -3), (depth, -2), (depth, 3))
            cr.restore()
            stroke(0.32 + 0.27 * impact * (0.75 + 0.25 * variation), 1.25)
        for radius, alpha in fronts:
            cr.arc(0, 0, radius, 0, math.tau)
            stroke(alpha, 1.0)
        for alpha, crests in breaking_waves:
            for crest in crests:
                _quad_to(cr, *crest)
            stroke(alpha, 1.0)
        cr.set_source_rgba(r, g, b, color_alpha)
        cr.arc(0, 0, 4, 0, math.tau)
        cr.fill()


# The dot field, the site's "Instant" showcase: idle cells glow at random, busy cells
# flicker, and settled, the chosen device's wave launches from its origin (CPU from the
# left, GPU from a corner, NPU from the centre). The waves use the site's maths.
_FIELD_ROWS = 10
_FIELD_HARDWARE = {"CPU": 0, "GPU": 1, "NPU": 2}
_FIELD_STILL_T = 3.6  # the frame shown under reduced motion, as on the site
_FIELD_SPEED = 1.75
_FIELD_BUSY_HZ = 6.0  # flicker steps per second while busy
_FIELD_RISE = 2.5  # per second: how fast the flicker builds after Optimize
_FIELD_DRAIN = 6.0  # per second: how fast it drains once the answer is in
_FIELD_LAUNCH = 1.1  # seconds the winning wave takes to reach the far edge
_FIELD_IDLE_PEAK = 0.55  # residual charge never reaches full brightness
_FIELD_HALO = 0.35  # a glowing cell lights its neighbours this much
_FIELD_ASPECT = 3.0  # width to height, as the site's canvas
_FIELD_MIN_WIDTH = 280
_FIELD_MIN_HEIGHT = 90
_FIELD_MAX_HEIGHT = 240


def _smooth(value: float) -> float:
    bounded = max(0.0, min(1.0, value))
    return bounded * bounded * (3.0 - 2.0 * bounded)


def _noise(*seeds: int) -> float:
    """Return a stable pseudo-random value in [0, 1) for a cell (and a time step)."""
    h = 2166136261
    for seed in seeds:
        h = ((h ^ (seed & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return ((h >> 8) & 0xFFFFFF) / 0x1000000


def _idle_glow(column: int, row: int, t: float) -> float:
    """Return a cell's residual charge: its own slow rhythm, lit only near the crest."""
    rate = 0.25 + 0.55 * _noise(column, row, 1)
    phase = _noise(column, row, 2) * math.tau
    return _smooth((math.sin(t * rate + phase) - 0.88) / 0.12)


def _wave(index: int, x: float, y: float, t: float) -> tuple[float, float]:
    """Return the site's wave for device `index` at a cell, and the cell's distance.

    The distance is from the wave's origin (0..1), which the launch envelope uses.
    """
    if index == 0:
        value = math.sin(x * 9 - t * _FIELD_SPEED + math.sin(y * math.pi) * 0.4)
        distance = x
    elif index == 1:
        value = math.sin(x * 8 + y * 5 - t * _FIELD_SPEED)
        distance = (x + y) / 2
    else:
        radius = math.hypot((x - 0.5) * 1.7, y - 0.5)
        value = math.sin(radius * 12 - t * _FIELD_SPEED)
        distance = radius / 0.98
    return _smooth(((value + 1) / 2 - 0.3) / 0.7), distance


class DotField(Gtk.DrawingArea):
    """The animated dot grid.

    Starts idle; `set_busy()` while work runs; `set_hardware("NPU")` settles it into
    that device's wave, launched from its origin. Runs while mapped and motion is on;
    holds one still frame of the current state otherwise.
    """

    def __init__(self, *, motion_enabled: bool = True) -> None:
        super().__init__(accessible_role=Gtk.AccessibleRole.PRESENTATION)
        self.add_css_class("quoth-animation")
        self.set_hexpand(True)
        self._motion = motion_enabled
        self._state = "idle"
        self._hardware = 2
        self._t = _FIELD_STILL_T
        self._busy_level = 0.0  # eased: rises after set_busy, drains on settle
        self._idle_level = 1.0  # eased: drains on settle
        self._wave_t0: float | None = None  # when the winning wave launched
        self._last_us: int | None = None
        self._tick_id: int | None = None
        self.set_draw_func(self._draw)
        self.connect("map", lambda _w: self._start())
        self.connect("unmap", lambda _w: self._stop())

    # The field fills the width it is given and keeps the site's proportion, so it
    # grows with the window instead of sitting at one size in a large one.
    def do_get_request_mode(self) -> Gtk.SizeRequestMode:
        return Gtk.SizeRequestMode.HEIGHT_FOR_WIDTH

    def do_measure(self, orientation, for_size: int):
        if orientation == Gtk.Orientation.HORIZONTAL:
            return _FIELD_MIN_WIDTH, _FIELD_MIN_WIDTH, -1, -1
        width = for_size if for_size > 0 else _FIELD_MIN_WIDTH
        height = int(
            min(_FIELD_MAX_HEIGHT, max(_FIELD_MIN_HEIGHT, width / _FIELD_ASPECT))
        )
        return height, height, -1, -1

    @property
    def state(self) -> str:
        return self._state

    def set_idle(self) -> None:
        self._state = "idle"
        self._wave_t0 = None
        if not self._motion:
            self._busy_level, self._idle_level = 0.0, 1.0
        self.queue_draw()

    def set_busy(self) -> None:
        self._state = "busy"
        self._wave_t0 = None
        if not self._motion:
            self._busy_level, self._idle_level = 1.0, 1.0
        self.queue_draw()

    def set_hardware(self, device: str | None) -> None:
        """Settle into `device`'s wave; an unknown device leaves the state as it is."""
        index = _FIELD_HARDWARE.get((device or "").upper())
        if index is None:
            return
        self._state = "settled"
        self._hardware = index
        # Launched now: with motion the wave spreads from its origin; without,
        # the still frame shows it fully arrived.
        self._wave_t0 = self._t if self._motion else self._t - 10.0
        if not self._motion:
            self._busy_level, self._idle_level = 0.0, 0.0
        self.queue_draw()

    def _start(self) -> None:
        if not self._motion or self._tick_id is not None:
            return
        self._last_us = None
        self._tick_id = self.add_tick_callback(
            guarded(self._advance, on_error=self._forget_tick)
        )

    def _forget_tick(self) -> None:
        self._tick_id = None

    def _stop(self) -> None:
        if self._tick_id is not None:
            self.remove_tick_callback(self._tick_id)
            self._tick_id = None

    def set_motion_enabled(self, enabled: bool) -> None:
        if enabled == self._motion:
            return
        self._motion = enabled
        if enabled:
            if self.get_mapped():
                self._start()
            return
        self._stop()
        self._t = _FIELD_STILL_T
        self._last_us = None
        self._wave_t0 = None
        if self._state == "idle":
            self._busy_level, self._idle_level = 0.0, 1.0
        elif self._state == "busy":
            self._busy_level, self._idle_level = 1.0, 1.0
        else:
            self._busy_level, self._idle_level = 0.0, 0.0
            self._wave_t0 = self._t - 10.0
        self.queue_draw()

    def _advance(self, _widget, clock) -> bool:
        now = clock.get_frame_time()
        delta = 0.0 if self._last_us is None else min((now - self._last_us) / 1e6, 0.05)
        self._last_us = now
        self.step(delta)
        return GLib.SOURCE_CONTINUE

    def step(self, delta: float) -> None:
        """Advance the field by `delta` seconds (the frame clock calls this)."""
        self._t += delta
        busy_target = 1.0 if self._state == "busy" else 0.0
        idle_target = 0.0 if self._state == "settled" else 1.0
        rise = 1.0 - math.exp(-delta * _FIELD_RISE)
        drain = 1.0 - math.exp(-delta * _FIELD_DRAIN)
        self._busy_level += (busy_target - self._busy_level) * (
            rise if busy_target > self._busy_level else drain
        )
        self._idle_level += (idle_target - self._idle_level) * drain
        self.queue_draw()

    def _activity(self, column: int, row: int, x: float, y: float) -> float:
        """Return how lit a cell is (0..1): charge, flicker and wave at their levels."""
        t = self._t
        total = 0.0
        if self._idle_level > 0.001:
            glow = _idle_glow(column, row, t)
            halo = max(
                _idle_glow(column + dc, row + dr, t)
                for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
            total += max(glow, halo * _FIELD_HALO) * _FIELD_IDLE_PEAK * self._idle_level
        if self._busy_level > 0.001:
            # Each cell joins the flicker a little later than the last, so it spreads
            # rather than switching on; a fresh value per step, eased between steps.
            onset = _smooth((self._busy_level - 0.4 * _noise(column, row, 3)) / 0.6)
            step = t * _FIELD_BUSY_HZ
            k = int(step)
            a, b = _noise(column, row, k), _noise(column, row, k + 1)
            value = a + (b - a) * _smooth(step - k)
            total += _smooth((value - 0.6) / 0.4) * onset
        if self._wave_t0 is not None:
            since = t - self._wave_t0
            pulse, distance = _wave(self._hardware, x, y, since)
            front = _smooth((since / _FIELD_LAUNCH - distance) / 0.12)
            total += pulse * front
        return min(1.0, total)

    def _draw(self, _area, cr: cairo.Context, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return
        rows = _FIELD_ROWS
        target_spacing = max(1.0, (height - 16) / (rows - 1))
        columns = max(10, min(44, int((width - 20) / target_spacing) + 1))
        spacing = max(
            1.0, min((width - 20) / (columns - 1), (height - 16) / (rows - 1))
        )
        cell = spacing * 0.46
        left = (width - (columns - 1) * spacing) / 2
        top = (height - (rows - 1) * spacing) / 2
        color = self.get_color()
        for row in range(rows):
            y = row / (rows - 1)
            for column in range(columns):
                x = column / (columns - 1)
                pulse = self._activity(column, row, x, y)
                pattern = ((column * 7 + row * 11) % 13) / 13
                presence = 0.28 + 0.72 * _smooth((0.5 - pattern) * 4)
                edge = 0.65 + math.sin(x * math.pi) * math.sin(y * math.pi) * 0.35
                cr.set_source_rgba(
                    color.red,
                    color.green,
                    color.blue,
                    color.alpha * (0.055 + pulse * 0.69 * presence) * edge,
                )
                cr.rectangle(
                    left + column * spacing - cell / 2,
                    top + row * spacing - cell / 2,
                    cell,
                    cell,
                )
                cr.fill()
