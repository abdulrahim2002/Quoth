# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Pill drawing primitives.

The parts every pill is made of: the status dot with its recording-pulse ripple,
the continuous edge-faded waveform, the background and the themed label. Also the
two pure pieces of pill behaviour every view needs identically: `LevelTrace` (the
scrolling amplitude history) and the transcribing wave.

Pure Cairo and Pango, with no GTK and no Theme object: callers pass the few values
they own. The theme-aware `pill_view.draw_pill` arranges these parts into a pill.
"""

import math
import string
from collections import deque

import cairo
import gi

gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Pango, PangoCairo

# The live pill's element geometry (240x48), which draw_pill scales to each view.
NUM_BARS = 24  # waveform bar count (visual density)
BARS_START_X = 46  # waveform's left edge inset from the pill's left edge
BAR_WIDTH = 2  # one bar's width; with bar_gap it sets where the wave ends
WAVE_PEAK = 24  # peak-to-peak span (per-point offset from the centre line is peak/2)
WAVE_PAD = 16  # horizontal padding when the wave fills the pill (transcribing)
DOT_X = 24  # status-dot inset from the pill's left edge
DOT_RADIUS = 4
DOT_RIPPLE = 10  # max ripple expansion beyond the dot

_NOISE_FLOOR = 0.15  # below this an amplitude reads as silence (recording gate)

# Transcribing transition timing (seconds; callers scale by theme duration_scale).
TRANSITION_DECAY_S = 0.25  # dot/timer fade + bars decay to zero
TRANSITION_EXPAND_S = 0.20  # bar region expands to fill pill
TRANSITION_TOTAL_S = TRANSITION_DECAY_S + TRANSITION_EXPAND_S

# Transcribing wave animation.
WAVE_SPEED = 4.0  # wave radians per unit of decorative phase
WAVE_BAR_OFFSET = 0.3  # phase offset between adjacent bars

# Label geometry: left inset (clears the status dot) and right padding.
LABEL_LEFT_X = 42
LABEL_PAD_X = 24

# Attack/release smoothing. Attack is fast and release slow so a syllable reads as a
# peak rather than a spike, and the wave settles instead of snapping back to silence.
_ATTACK = 0.8
_RELEASE = 0.3


def lerp(a: float, b: float, t: float) -> float:
    """Interpolate linearly from `a` to `b`, with `t` clamped to [0, 1]."""
    return a + (b - a) * max(0.0, min(1.0, t))


def lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    """Interpolate two RGBA colour tuples, with `t` clamped to [0, 1]."""
    t = max(0.0, min(1.0, t))
    return tuple(c1[i] + (c2[i] - c1[i]) * t for i in range(len(c1)))


def format_timer(seconds: float) -> str:
    """Format elapsed `seconds` as the pill's `m:ss`.

    Minutes are not capped; previews wrap their own elapsed time.
    """
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


# Pace of the decorative phase (dot ripple, transcribing wave), per second of
# frame-clock time, so the speed does not depend on the refresh rate.
PHASE_RATE = 3.0


def transcribing_amplitudes(phase: float) -> list[float]:
    """Return the travelling sine the pill animates while transcribing.

    No microphone feeds this state, so the wave is generated rather than measured.
    """
    amps = []
    for i in range(NUM_BARS):
        p = phase * WAVE_SPEED + i * WAVE_BAR_OFFSET
        amps.append(0.3 + 0.2 * math.sin(p) + 0.1 * math.sin(p * 2.3))
    return amps


class LevelTrace:
    """The scrolling amplitude history behind the waveform.

    A fixed-width window of the most recent levels, attack/release smoothed. The
    wave is this history drawn left to right, so it scrolls as levels arrive.
    """

    def __init__(self) -> None:
        self._values = deque([0.0] * NUM_BARS, maxlen=NUM_BARS)
        self._smoothed = 0.0

    def push(self, amplitude: float) -> None:
        weight = _ATTACK if amplitude > self._smoothed else _RELEASE
        self._smoothed = self._smoothed * (1.0 - weight) + amplitude * weight
        self._values.append(self._smoothed)

    def reset(self) -> None:
        """Reset to silence, so a new recording does not inherit the last one's tail."""
        self._values.clear()
        self._values.extend([0.0] * NUM_BARS)
        self._smoothed = 0.0

    def values(self) -> list[float]:
        return list(self._values)


def wave_end_x(bar_gap: int) -> float:
    """Return where the recording waveform ends, at the live pill's 240x48 scale.

    Callers scale the result. The gap sits between bars, never after the last one,
    hence the trailing subtraction.
    """
    return BARS_START_X + NUM_BARS * (BAR_WIDTH + bar_gap) - bar_gap


def draw_dot(
    cr,
    *,
    x,
    y,
    color,
    phase,
    radius=DOT_RADIUS,
    ripple_reach=DOT_RIPPLE,
    opacity=1.0,
    ripple=True,
    line_width=1.0,
):
    """Draw the status dot and its expanding ripple, the recording pulse.

    `phase` advances the ripple (pass the caller's animation phase); `opacity` fades
    the whole dot, as the transcribing transition does.
    """
    if opacity <= 0:
        return
    r, g, b, a = color

    # Solid centre dot.
    cr.set_source_rgba(r, g, b, a * opacity)
    cr.arc(x, y, radius, 0, 2 * math.pi)
    cr.fill()

    if not ripple:
        return

    # Expanding ripple; the 2.5pi loop runs 20% slower than a 2pi one.
    loop = math.pi * 2.5
    progress = (phase % loop) / loop
    cr.set_source_rgba(r, g, b, a * (1.0 - progress**1.5) * opacity)
    cr.set_line_width(line_width)
    cr.arc(x, y, radius + progress * ripple_reach, 0, 2 * math.pi)
    cr.stroke()


def draw_waveform(
    cr,
    *,
    height: float,
    start_x: float,
    end_x: float,
    color: tuple,
    amplitudes,
    line_width: float,
    peak: float = WAVE_PEAK,
    gated: bool = True,
) -> None:
    """Draw `amplitudes` (each 0..1) as one continuous Bézier string with faded ends.

    The string is centred vertically in `height`. `peak` is its peak-to-peak span
    (each point sits at most `peak / 2` from the centre line). `gated` applies the
    recording noise floor; pass False for a raw wave, as the transcribing decay does
    with its already faded values.
    """
    n = len(amplitudes)
    if n == 0 or end_x <= start_x:
        return

    dx = (end_x - start_x) / max(n - 1, 1)
    mid_y = height / 2
    peak = min(peak, max(0.0, height - line_width))

    # Points alternate above/below the centre line to form the wave.
    points = []
    for i, amp in enumerate(amplitudes):
        if gated:
            amp = max(0.0, amp - _NOISE_FLOOR) / (1.0 - _NOISE_FLOOR)
        sign = 1 if i % 2 == 0 else -1
        points.append((start_x + i * dx, mid_y + amp * (peak / 2) * sign))

    # Linear gradient fades the string's opacity at both ends.
    pat = cairo.LinearGradient(start_x, 0, end_x, 0)
    r, g, b, a = color
    pat.add_color_stop_rgba(0.0, r, g, b, 0.0)
    pat.add_color_stop_rgba(0.15, r, g, b, a)
    pat.add_color_stop_rgba(0.85, r, g, b, a)
    pat.add_color_stop_rgba(1.0, r, g, b, 0.0)
    cr.set_source(pat)
    cr.set_line_width(line_width)
    cr.set_line_cap(cairo.LINE_CAP_ROUND)
    cr.set_line_join(cairo.LINE_JOIN_ROUND)

    # Continuous spline; control points halfway between samples give clean S-curves.
    cr.move_to(*points[0])
    for i in range(n - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        cp_x = x0 + dx / 2
        cr.curve_to(cp_x, y0, cp_x, y1, x1, y1)
    cr.stroke()


def draw_label(
    cr,
    text: str,
    *,
    width: float,
    height: float,
    color: tuple,
    font_family: str,
    font_size: float,
    left_x: float = LABEL_LEFT_X,
    pad_x: float = LABEL_PAD_X,
    align: str = "left",
    opacity: float = 1.0,
    font_weight: Pango.Weight = Pango.Weight.NORMAL,
    center_y: float | None = None,
) -> bool:
    """Draw a pill label in the one typography the timer, the error and the hint share.

    Ellipsizes to the space between the insets, so a long string degrades gracefully
    instead of overflowing the fixed-size pill. Right-aligned text is a count, set with
    tabular digits. `opacity` fades it (the transcribing transition). Returns False
    when nothing was drawn: faded out, no room, or the text is taller than `height`.
    """
    if opacity <= 0:
        return False
    r, g, b, a = color
    cr.set_source_rgba(r, g, b, a * opacity)
    layout = label_layout(
        cr,
        text,
        font_family=font_family,
        font_size=font_size,
        max_width=width - left_x - pad_x,
        font_weight=font_weight,
        tabular=align == "right",
    )
    if layout is None:
        return False
    _ink_x, ink_y, _ink_width, ink_height = ink_extents(layout)
    x = width - tabular_advance(layout) - pad_x if align == "right" else left_x
    center_y = height / 2 if center_y is None else center_y
    if center_y - ink_height / 2 < 0 or center_y + ink_height / 2 > height:
        return False
    cr.move_to(x, center_y - ink_y - ink_height / 2)
    PangoCairo.show_layout(cr, layout)
    return True


def label_layout(
    cr,
    text: str,
    *,
    font_family: str,
    font_size: float,
    max_width: float,
    font_weight: Pango.Weight = Pango.Weight.NORMAL,
    tabular: bool = False,
) -> Pango.Layout | None:
    """Lay `text` out as one line, ellipsized to `max_width`; sizes are Cairo px.

    `tabular` sets digits as tabular figures without kerning, for counts. Returns
    None when there is nothing to draw: no text, or not even the ellipsis fits.
    """
    if not text or max_width <= 0:
        return None
    font = Pango.FontDescription()
    font.set_family(font_family)
    font.set_absolute_size(font_size * Pango.SCALE)
    font.set_weight(font_weight)
    layout = PangoCairo.create_layout(cr)
    layout.set_font_description(font)
    if tabular:
        attrs = Pango.AttrList()
        attrs.insert(Pango.AttrFontFeatures.new("tnum, -kern"))
        layout.set_attributes(attrs)
    layout.set_single_paragraph_mode(True)
    layout.set_width(round(max_width * Pango.SCALE))
    layout.set_ellipsize(Pango.EllipsizeMode.END)
    layout.set_text(text, -1)
    # Pango keeps a lone ellipsis even where it overflows the width.
    if layout.get_extents()[1].width > layout.get_width():
        return None
    return layout


def ink_extents(layout) -> tuple[float, float, float, float]:
    """Return the layout's ink rectangle as (x, y, width, height) from its origin."""
    ink, _logical = layout.get_extents()
    return tuple(value / Pango.SCALE for value in (ink.x, ink.y, ink.width, ink.height))


def tabular_advance(layout) -> float:
    """Return the line's advance with every digit as wide as the widest one.

    Right-aligned by it, a count keeps its left edge as its digits change, even in a
    font with proportional digits and no tabular figures for `tnum` to select.
    """
    digits = layout.copy()
    digits.set_width(-1)
    digits.set_text(string.digits, -1)
    advances = [digits.index_to_pos(i).width for i in range(len(string.digits))]
    widest = string.digits[advances.index(max(advances))]
    tabular = layout.copy()
    tabular.set_text(
        layout.get_text().translate(str.maketrans(string.digits, widest * 10)), -1
    )
    return tabular.get_extents()[1].width / Pango.SCALE


def _rounded_rect(cr, x, y, w, h, r):
    """Trace a rounded-rectangle path (stadium when r == h / 2)."""
    cr.new_path()
    cr.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
    cr.arc(x + w - r, y + r, r, 1.5 * math.pi, 2 * math.pi)
    cr.arc(x + w - r, y + h - r, r, 0, 0.5 * math.pi)
    cr.arc(x + r, y + h - r, r, 0.5 * math.pi, math.pi)
    cr.close_path()


def draw_background(cr, *, width, height, radius, bg, border, border_width):
    """Fill the flat pill background and stroke its border inside the fill.

    The border is inset by half its stroke; `border_width` <= 0 skips it.
    """
    _rounded_rect(cr, 0, 0, width, height, radius)
    cr.set_source_rgba(*bg)
    cr.fill()

    if border_width > 0:
        inset = border_width / 2
        _rounded_rect(
            cr,
            inset,
            inset,
            width - border_width,
            height - border_width,
            radius - inset,
        )
        cr.set_source_rgba(*border)
        cr.set_line_width(border_width)
        cr.stroke()
