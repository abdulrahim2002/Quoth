# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.overlay` messages, rendered and replaced without a compositor.

A real pill shows them on a fake layer shell; the drawing tests run the message painter
unbound on a stand-in, so they need no display.
"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import cairo
import pytest

from quoth.overlay import WaveformPill
from quoth.pill_render import draw_label, label_layout
from quoth.theme import DEFAULT_THEME


def test_new_error_replaces_guidance_and_restarts_timeout(layer_shell):
    pill = WaveformPill()
    pill.set_motion_enabled(False)
    with (
        patch("quoth.overlay.GLib.timeout_add", side_effect=[11, 12]) as timeout,
        patch("quoth.overlay.GLib.source_remove") as remove,
    ):
        pill.show_error("No model installed", guidance="Download in Settings")
        assert pill._error_guidance == "Download in Settings"
        timeout.assert_called_with(5000, pill._auto_hide_error)
        pill.show_error("Another error")
        assert pill._error_message == "Another error"
        assert pill._error_guidance is None
        remove.assert_called_once_with(11)
        assert timeout.call_count == 2
        timeout.assert_called_with(5000, pill._auto_hide_error)
    assert pill._auto_hide_error() is False
    assert pill._hide_timeout_id is None
    assert pill._state == "idle"


def test_new_hint_does_not_inherit_previous_guidance(layer_shell):
    pill = WaveformPill()
    pill.show_recording()
    pill.show_hint("No input detected", guidance="Check your microphone")
    assert pill._hint_guidance == "Check your microphone"
    pill.push_amplitude(0.5)
    assert pill._state == "recording"
    pill.show_hint("Another hint")
    assert pill._hint_message == "Another hint"
    assert pill._hint_guidance is None


def test_reduced_motion_idle_error_invalidates_and_settles_the_orb(layer_shell):
    from quoth.pill_view import _ORB_BLOBS

    pill = WaveformPill(style="orb")
    pill.set_motion_enabled(False)
    with (
        patch("quoth.overlay.GLib.timeout_add", return_value=11),
        patch.object(pill._canvas, "queue_draw") as queue_draw,
        patch.object(pill._orb, "set_params") as set_params,
    ):
        pill.show_error("boom")
    queue_draw.assert_called_once_with()
    params = set_params.call_args.args[0]
    roles = [role for role, _weight in _ORB_BLOBS]
    assert params.emphasis[roles.index("dot_error")] > 1.0


@pytest.mark.parametrize("state", ["error", "hint"])
@pytest.mark.parametrize("family", ["sans-serif", "monospace"])
@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize(
    ("height", "font_size", "lines"), [(48, 12, 2), (24, 12, 1), (48, 30, 1)]
)
def test_message_ink_fits_and_is_centered(
    state, family, scale, height, font_size, lines
):
    theme = replace(
        DEFAULT_THEME, font=replace(DEFAULT_THEME.font, family=family, size=font_size)
    )
    pill = SimpleNamespace(
        _theme=theme,
        _state=state,
        _dot_phase=0,
        _error_message="Microphone change failed",
        _error_guidance="Check microphone in Settings",
        _hint_message="No input detected",
        _hint_guidance="Check your microphone",
        _animations_on=lambda: True,
    )
    surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
    cr = cairo.Context(surface)
    cr.scale(scale, scale)
    with (
        patch("quoth.overlay.draw_background"),
        patch("quoth.overlay.draw_status_dot"),
        patch("quoth.overlay.draw_label", wraps=draw_label) as label,
    ):
        WaveformPill._draw_message(pill, cr, 240, height)
    assert label.call_count == lines
    left, top, width, ink_height = surface.ink_extents()
    assert left >= 41 * scale
    assert left + width <= 217 * scale
    assert top >= 0
    assert top + ink_height <= height * scale
    assert abs(top + ink_height / 2 - height * scale / 2) <= 1


def test_both_message_lines_ellipsize_independently():
    pill = SimpleNamespace(
        _theme=DEFAULT_THEME,
        _state="error",
        _dot_phase=0,
        _error_message="A" * 80 + "g",
        _error_guidance="A" * 80 + "g",
        _animations_on=lambda: True,
    )
    cr = cairo.Context(cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None))
    layouts = []

    def record(*args, **kwargs):
        layouts.append(label_layout(*args, **kwargs))
        return layouts[-1]

    with (
        patch("quoth.overlay.draw_label", wraps=draw_label) as label,
        patch("quoth.pill_render.label_layout", side_effect=record),
    ):
        WaveformPill._draw_message(pill, cr, 240, 48)
    assert label.call_count == 2
    assert len(layouts) == 2
    assert all(layout.is_ellipsized() for layout in layouts)
    bounds = []
    for call in label.call_args_list:
        surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
        draw_label(cairo.Context(surface), call.args[1], **call.kwargs)
        left, top, width, height = surface.ink_extents()
        assert left + width <= 217
        bounds.append((top, top + height))
    assert bounds[1][0] - bounds[0][1] == pytest.approx(4, abs=1)
    assert (bounds[0][0] + bounds[1][1]) / 2 == pytest.approx(24, abs=1)
