# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.pill_render`, the shared pill drawing primitives.

The look is checked by eye; these pin branch behaviour and geometry so the renderers
shared by the live pill and onboarding cannot regress into a crash.
"""

import string
from unittest.mock import MagicMock

import cairo
import pytest

from quoth.pill_render import (
    NUM_BARS,
    WAVE_PEAK,
    LevelTrace,
    draw_background,
    draw_dot,
    draw_label,
    draw_waveform,
    format_timer,
    ink_extents,
    label_layout,
    tabular_advance,
    transcribing_amplitudes,
)


class TestLevelTrace:
    """The scrolling wave history the live pill and the Settings preview share."""

    def test_starts_silent(self):
        assert LevelTrace().values() == [0.0] * NUM_BARS

    def test_is_always_the_bar_count_wide(self):
        trace = LevelTrace()
        for _ in range(NUM_BARS * 3):
            trace.push(0.5)
        assert len(trace.values()) == NUM_BARS

    def test_scrolls_oldest_out(self):
        trace = LevelTrace()
        for _ in range(NUM_BARS):
            trace.push(1.0)
        assert min(trace.values()) > 0.0  # the initial silence has scrolled away

    def test_attacks_faster_than_it_releases(self):
        # A syllable should read as a peak and settle, not spike and snap back.
        rising = LevelTrace()
        rising.push(1.0)
        falling = LevelTrace()
        for _ in range(50):
            falling.push(1.0)
        before = falling.values()[-1]
        falling.push(0.0)
        attack_moved = rising.values()[-1]
        release_moved = before - falling.values()[-1]
        assert attack_moved > release_moved

    def test_reset_returns_to_silence(self):
        trace = LevelTrace()
        for _ in range(NUM_BARS):
            trace.push(1.0)
        trace.reset()
        # A new recording must not inherit the last one's tail.
        assert trace.values() == [0.0] * NUM_BARS

    def test_silence_stays_flat(self):
        # No input draws no wave.
        trace = LevelTrace()
        for _ in range(NUM_BARS * 2):
            trace.push(0.0)
        assert trace.values() == [0.0] * NUM_BARS


class TestFormatTimer:
    @pytest.mark.parametrize(
        ("seconds", "text"),
        [
            (7, "0:07"),
            (75, "1:15"),
            # Truncated: rounding up would show a minute that hasn't elapsed.
            (59.9, "0:59"),
        ],
    )
    def test_text(self, seconds, text):
        assert format_timer(seconds) == text


class TestTranscribingAmplitudes:
    """The generated transcribing wave: one definition, as no microphone feeds it."""

    def test_is_the_bar_count_wide(self):
        assert len(transcribing_amplitudes(0.0)) == NUM_BARS

    def test_stays_in_range(self):
        for step in range(50):
            assert all(0.0 <= a <= 1.0 for a in transcribing_amplitudes(step * 0.1))

    def test_travels_with_phase(self):
        assert transcribing_amplitudes(0.0) != transcribing_amplitudes(0.5)


def _ctx():
    return cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 240, 48))


def _pixel(surface, x, y):
    """Return the (r, g, b, a) bytes at x, y of a little-endian ARGB32 surface."""
    surface.flush()
    i = y * surface.get_stride() + x * 4
    b, g, r, a = surface.get_data()[i : i + 4]
    return r, g, b, a


@pytest.mark.parametrize("text", ["No model installed", "Preparing model", "0:07"])
@pytest.mark.parametrize("family", ["sans-serif", "monospace"])
@pytest.mark.parametrize("scale", [1, 2])
def test_label_ink_is_vertically_centered(text, family, scale):
    surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
    cr = cairo.Context(surface)
    cr.scale(scale, scale)
    draw_label(
        cr,
        text,
        width=240,
        height=48,
        color=(1, 1, 1, 1),
        font_family=family,
        font_size=12,
    )
    _, top, _, height = surface.ink_extents()
    assert abs(top + height / 2 - 24 * scale) <= 1


@pytest.mark.parametrize("family", ["sans-serif", "monospace"])
@pytest.mark.parametrize("scale", [1, 2])
def test_timer_prefix_stays_put_when_seconds_change(family, scale):
    prefixes = []
    for digit in "0123456789":
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 240 * scale, 48 * scale)
        cr = cairo.Context(surface)
        cr.scale(scale, scale)
        draw_label(
            cr,
            "0:0" + digit,
            width=240,
            height=48,
            color=(1, 1, 1, 1),
            font_family=family,
            font_size=12,
            align="right",
        )
        # The minute and colon must not move when the seconds change.
        seconds = label_layout(
            cr, "00", font_family=family, font_size=12, max_width=174, tabular=True
        )
        end = round((216 - tabular_advance(seconds)) * scale) * 4
        surface.flush()
        pixels = bytes(surface.get_data())
        stride = surface.get_stride()
        prefixes.append(
            b"".join(pixels[y * stride : y * stride + end] for y in range(48 * scale))
        )
    assert any(prefixes[0])
    assert all(prefix == prefixes[0] for prefix in prefixes)


def test_inter_has_proportional_digits():
    # test_tabular_advance_is_the_same_for_every_digit reaches the widest-digit rule
    # only through a font whose digits differ in width: Inter (`fonts-inter` on
    # Debian and Ubuntu). Without it fontconfig substitutes another font and this fails.
    cr = _ctx()
    widths = {
        label_layout(cr, digit, font_family="Inter", font_size=12, max_width=174)
        .get_extents()[1]
        .width
        for digit in string.digits
    }
    assert len(widths) > 1


@pytest.mark.parametrize("family", ["Inter", "sans-serif", "serif", "monospace"])
@pytest.mark.parametrize("tabular", [True, False])
def test_tabular_advance_is_the_same_for_every_digit(family, tabular):
    # Without tnum, Inter keeps its proportional digits, so this holds through the
    # widest-digit rule alone.
    cr = _ctx()
    advances = set()
    for text in ["0:00", "0:11", "1:11", "8:58", "8:88"]:
        layout = label_layout(
            cr, text, font_family=family, font_size=12, max_width=174, tabular=tabular
        )
        ink_x, _ink_y, ink_width, _ink_height = ink_extents(layout)
        assert ink_x + ink_width <= tabular_advance(layout)
        advances.add(tabular_advance(layout))
    assert len(advances) == 1


def test_long_label_ellipsizes_inside_the_insets():
    surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
    assert draw_label(
        cairo.Context(surface),
        "Could not load the speech model from the selected folder",
        width=240,
        height=48,
        color=(1, 1, 1, 1),
        font_family="sans-serif",
        font_size=12,
    )
    left, _top, width, _height = surface.ink_extents()
    assert left >= 41
    assert left + width <= 217
    layout = label_layout(
        _ctx(),
        "Could not load the speech model from the selected folder",
        font_family="sans-serif",
        font_size=12,
        max_width=174,
    )
    assert layout.is_ellipsized()
    assert layout.get_extents()[1].width <= layout.get_width()


@pytest.mark.parametrize(
    ("amplitudes", "options", "span"),
    [
        ([], {}, (0, 0)),
        # One point is a move_to with no curves, which strokes nothing.
        ([0.5], {"gated": False}, (0, 0)),
        # 46..216, widened by each round cap.
        ([0.2, 0.6, 0.9, 0.4, 0.7, 0.3], {"peak": WAVE_PEAK, "gated": True}, (45, 172)),
    ],
    ids=["no-points", "one-point", "many-points"],
)
def test_waveform_draws_any_number_of_points(amplitudes, options, span):
    surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
    draw_waveform(
        cairo.Context(surface),
        height=48,
        start_x=46,
        end_x=216,
        color=(0.5, 0.7, 1.0, 1.0),
        amplitudes=amplitudes,
        line_width=1.5,
        **options,
    )
    left, _top, width, _height = surface.ink_extents()
    assert (left, width) == span


@pytest.mark.parametrize(("gated", "height"), [(True, 2), (False, 6)])
def test_gate_reads_the_noise_floor_as_silence(gated, height):
    # At the floor the gated wave is a flat line one stroke tall; ungated it zigzags.
    surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
    draw_waveform(
        cairo.Context(surface),
        height=48,
        start_x=46,
        end_x=216,
        color=(1, 1, 1, 1),
        amplitudes=[0.15] * 6,
        line_width=1.5,
        gated=gated,
    )
    assert surface.ink_extents()[3] == height


# Zero opacity returns before drawing anything.
@pytest.mark.parametrize(
    ("opacity", "centre"), [(1.0, (255, 76, 76, 255)), (0.0, (0, 0, 0, 0))]
)
def test_dot_draws_at_any_opacity(opacity, centre):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 240, 48)
    draw_dot(
        cairo.Context(surface),
        x=24,
        y=24,
        color=(1.0, 0.3, 0.3, 1.0),
        phase=1.2,
        opacity=opacity,
    )
    assert _pixel(surface, 24, 24) == pytest.approx(centre, abs=1)


def test_dot_can_hold_the_solid_mark_without_a_ripple():
    cr = MagicMock()
    draw_dot(
        cr,
        x=24,
        y=24,
        color=(1, 1, 1, 1),
        phase=1.0,
        ripple=False,
        line_width=2.0,
    )
    cr.fill.assert_called_once_with()
    cr.stroke.assert_not_called()


def test_waveform_skips_a_nonpositive_span():
    cr = MagicMock()
    draw_waveform(
        cr,
        height=24,
        start_x=12,
        end_x=12,
        color=(1, 1, 1, 1),
        amplitudes=[1.0, 1.0],
        line_width=1.0,
    )
    cr.move_to.assert_not_called()


def test_waveform_peak_keeps_the_stroke_inside_the_height():
    surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
    cr = cairo.Context(surface)
    draw_waveform(
        cr,
        height=24,
        start_x=2,
        end_x=22,
        color=(1, 1, 1, 1),
        amplitudes=[1.0, 1.0],
        line_width=20,
        peak=100,
        gated=False,
    )
    _left, top, _width, height = surface.ink_extents()
    assert top >= 0
    assert top + height <= 24
    assert height == 24  # drawn, and clamped only as far as the height requires


def test_label_is_omitted_when_even_the_ellipsis_does_not_fit():
    surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
    assert not draw_label(
        cairo.Context(surface),
        "too wide",
        width=10,
        height=24,
        color=(1, 1, 1, 1),
        font_family="sans-serif",
        font_size=12,
        left_x=4,
        pad_x=4,
    )
    assert surface.ink_extents()[2:] == (0, 0)
    assert (
        label_layout(
            _ctx(), "too wide", font_family="sans-serif", font_size=12, max_width=2
        )
        is None
    )


def test_background_renders_with_and_without_border():
    bordered = cairo.ImageSurface(cairo.FORMAT_ARGB32, 240, 48)
    draw_background(
        cairo.Context(bordered),
        width=240,
        height=48,
        radius=24,
        bg=(0.1, 0.1, 0.1, 1.0),
        border=(1, 1, 1, 0.2),
        border_width=1.0,
    )
    assert _pixel(bordered, 120, 24) == pytest.approx((25, 25, 25, 255), abs=1)
    # The top edge: 20% white over the fill.
    assert _pixel(bordered, 120, 0) == pytest.approx((71, 71, 71, 255), abs=1)
    assert _pixel(bordered, 0, 0) == (0, 0, 0, 0)  # outside the rounded end
    # border_width <= 0 skips the stroke; must still fill cleanly.
    plain = cairo.ImageSurface(cairo.FORMAT_ARGB32, 240, 48)
    draw_background(
        cairo.Context(plain),
        width=240,
        height=48,
        radius=24,
        bg=(0.1, 0.1, 0.1, 1.0),
        border=(1, 1, 1, 0.2),
        border_width=0,
    )
    assert _pixel(plain, 120, 0) == pytest.approx((25, 25, 25, 255), abs=1)


def test_label_opacity_only_multiplies_the_theme_alpha():
    def ink(alpha, opacity):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 240, 48)
        draw_label(
            cairo.Context(surface),
            "Download in Settings",
            width=240,
            height=48,
            color=(1, 1, 1, alpha),
            font_family="sans-serif",
            font_size=10,
            opacity=opacity,
        )
        surface.flush()
        return sum(surface.get_data()[3::4])

    full = ink(1.0, 1.0)
    assert full > 0
    assert ink(0.9, 0.8) == pytest.approx(full * 0.72, rel=0.05)
