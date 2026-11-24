# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.overlay`: the pill's state machine, drawing and placement."""

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import cairo
import pytest

# The pill's state machine, placement and input region act only on a layer surface.
pytestmark = pytest.mark.usefixtures("layer_shell")
FIXTURES = Path(__file__).with_name("fixtures")


def motion_theme(name):
    from quoth.theme import read_theme

    return read_theme(FIXTURES / name)


class TestOverlayState:
    def _make_pill(self):
        from quoth.overlay import WaveformPill

        return WaveformPill()

    def test_initial_state_is_idle(self):
        pill = self._make_pill()
        assert pill._state == "idle"

    def test_show_recording_sets_state(self):
        pill = self._make_pill()
        pill.show_recording()
        assert pill._state == "recording"

    def test_show_transcribing_sets_state(self):
        pill = self._make_pill()
        pill.show_recording()
        pill.show_transcribing()
        assert pill._state == "transcribing"

    def test_hide_pill_sets_idle_from_recording(self):
        # Motion off: no exit animation to play, so the hide lands immediately.
        pill = self._make_pill()
        pill.set_motion_enabled(False)
        pill.show_recording()
        pill.hide_pill()
        assert pill._state == "idle"

    def test_hide_pill_from_recording_plays_the_exit(self):
        # Motion on: the pill animates out, so idle arrives when the exit finishes.
        pill = self._make_pill()
        pill.show_recording()
        pill.hide_pill()
        assert pill._is_exiting is True
        assert pill._state == "recording"

    def test_show_error_sets_state(self):
        pill = self._make_pill()
        pill.show_error("Test error")
        assert pill._state == "error"
        assert pill._error_message == "Test error"

    def test_push_amplitude_appends(self):
        # Smoothed on the way in (attack 0.8), so the trace holds the eased value, not
        # the raw one: the wave follows a voice rather than jittering with every sample.
        pill = self._make_pill()
        pill.push_amplitude(0.5)
        assert pill._trace.values()[-1] == pytest.approx(0.4)

    def test_show_hint_sets_state_during_recording(self):
        pill = self._make_pill()
        pill.show_recording()
        pill.show_hint("No input detected")
        assert pill._state == "hint"
        assert pill._hint_message == "No input detected"

    def test_show_hint_ignored_when_not_recording(self):
        pill = self._make_pill()
        pill.show_hint("No input detected")  # idle
        assert pill._state == "idle"

    def test_push_amplitude_clears_hint(self):
        pill = self._make_pill()
        pill.show_recording()
        pill.show_hint("No input detected")
        pill.push_amplitude(0.5)
        assert pill._state == "recording"

    def test_hide_deferred_during_transition(self):
        pill = self._make_pill()
        pill.show_recording()
        pill.show_transcribing()
        pill.hide_pill()
        assert pill._state == "transcribing"
        assert pill._hide_pending is True

    def test_hide_immediate_when_not_transcribing(self):
        # Only the transcribing wave earns a deferral (it has a minimum display time);
        # recording is hidden at once, even when it began inside the last wave's window.
        pill = self._make_pill()
        pill.show_recording()
        pill.show_transcribing()
        pill.show_recording()
        pill.hide_pill()
        assert pill._hide_pending is False
        assert pill._is_exiting is True

    def test_frozen_amplitudes_snapshot(self):
        # The transition decays from the wave as it really stood, so the freeze is the
        # smoothed trace: 0.7 attacks to 0.56, then 0.3 releases to 0.482.
        pill = self._make_pill()
        pill.show_recording()
        pill.push_amplitude(0.7)
        pill.push_amplitude(0.3)
        pill.show_transcribing()
        assert pill._frozen_amplitudes[-1] == pytest.approx(0.482)
        assert pill._frozen_amplitudes[-2] == pytest.approx(0.56)

    def test_recording_resets_the_trace(self):
        # A new recording must not open with the last one's tail still on screen.
        pill = self._make_pill()
        pill.show_recording()
        pill.push_amplitude(1.0)
        pill.show_recording()
        assert pill._trace.values() == [0.0] * len(pill._trace.values())

    def test_do_hide_clears_pending(self):
        pill = self._make_pill()
        pill.show_recording()
        pill.show_transcribing()
        pill._hide_pending = True
        pill._do_hide()
        assert pill._state == "idle"
        assert pill._hide_pending is False

    def test_prewarm_maps_surface_without_leaving_idle(self):
        # Pre-mapping keeps the pill IDLE (drawn transparent) with no animation, while
        # the window turns visible so the first show is instant.
        pill = self._make_pill()
        with patch.object(pill, "set_visible") as set_visible:
            pill.prewarm()
            set_visible.assert_called_once_with(True)
        assert pill._state == "idle"
        assert pill._tick_id is None

    def test_show_recording_after_prewarm_still_records(self):
        pill = self._make_pill()
        pill.prewarm()
        pill.show_recording()
        assert pill._state == "recording"

    def test_idle_surface_is_click_through(self):
        # The always-mapped IDLE surface must pass pointer events through (empty input
        # region) so it never silently eats clicks at its anchored position.
        pill = self._make_pill()
        surface = MagicMock()
        with patch.object(pill, "get_surface", return_value=surface):
            pill._apply_input_region()  # state is idle
        region = surface.set_input_region.call_args.args[0]
        assert region.is_empty()

    def test_mapping_keeps_the_idle_surface_click_through(self):
        # A newly mapped surface takes clicks over its whole rectangle, so the map
        # handler must set the empty region again while IDLE.
        pill = self._make_pill()
        surface = MagicMock()
        with patch.object(pill, "get_surface", return_value=surface):
            pill.set_visible(True)
        region = surface.set_input_region.call_args.args[0]
        assert region.is_empty()

    def test_visible_pill_captures_clicks(self):
        pill = self._make_pill()
        surface = MagicMock()
        pill._state = "recording"
        with patch.object(pill, "get_surface", return_value=surface):
            pill._apply_input_region()
        region = surface.set_input_region.call_args.args[0]
        rect = cairo.RectangleInt(0, 0, *pill._surface_size())
        assert region.contains_rectangle(rect) == cairo.REGION_OVERLAP_IN

    def test_large_back_input_region_covers_the_overshoot_surface(self):
        from quoth.overlay import WaveformPill

        pill = WaveformPill(theme=motion_theme("motion-2-large-back.toml"))
        surface = MagicMock()
        pill._state = "recording"
        with patch.object(pill, "get_surface", return_value=surface):
            pill._apply_input_region()
        region = surface.set_input_region.call_args.args[0]
        rect = cairo.RectangleInt(0, 0, *pill._surface_size())
        assert region.contains_rectangle(rect) == cairo.REGION_OVERLAP_IN

    def test_disabling_motion_finishes_entry(self):
        pill = self._make_pill()
        pill.show_recording()
        assert pill._is_entering
        pill.set_motion_enabled(False)
        assert not pill._is_entering
        assert pill._state == "recording"

    def test_disabling_motion_finishes_transcribing(self):
        from quoth.pill_render import TRANSITION_TOTAL_S

        pill = self._make_pill()
        pill.show_recording()
        pill.show_transcribing()
        pill.set_motion_enabled(False)
        assert not pill._frame().motion_enabled
        assert pill._frame().transition_elapsed >= pill._scaled(TRANSITION_TOTAL_S)

    def test_disabling_motion_completes_a_pending_hide_without_unmapping(self):
        pill = self._make_pill()
        pill.show_recording()
        pill.show_transcribing()
        pill.hide_pill()
        with patch.object(pill, "set_visible") as set_visible:
            pill.set_motion_enabled(False)
        assert pill._state == "idle"
        set_visible.assert_not_called()
        assert pill.get_visible()

    def test_disabling_motion_completes_an_exit(self):
        pill = self._make_pill()
        pill.show_recording()
        pill.hide_pill()
        assert pill._is_exiting
        pill.set_motion_enabled(False)
        assert pill._state == "idle"

    def test_hot_theme_motion_flag_uses_the_same_reconciliation(self):
        from dataclasses import replace

        pill = self._make_pill()
        pill.show_recording()
        assert pill._is_entering
        static = replace(
            pill._theme,
            animation=replace(pill._theme.animation, enabled=False),
        )
        pill.apply_theme(static)
        assert not pill._is_entering
        assert not pill._frame().motion_enabled


class TestInertWithoutLayerShell:
    """No layer-shell, no mapping: a toplevel would steal focus and break the paste."""

    def _make_inert_pill(self):
        from quoth.overlay import WaveformPill

        with patch("quoth.overlay.presentation.overlay_available", return_value=False):
            return WaveformPill()

    def test_show_recording_is_noop(self, caplog):
        pill = self._make_inert_pill()
        pill.show_recording()
        assert pill._state == "idle"
        assert not pill.get_visible()
        [record] = [r for r in caplog.records if r.levelname == "WARNING"]
        assert record.getMessage().startswith("Overlay unavailable (")

    def test_show_error_is_noop(self):
        pill = self._make_inert_pill()
        pill.show_error("boom")
        assert pill._state == "idle"
        assert not pill.get_visible()

    def test_prewarm_is_noop(self):
        pill = self._make_inert_pill()
        pill.prewarm()
        assert not pill.get_visible()

    def test_resize_is_noop(self):
        from quoth import overlay

        pill = self._make_inert_pill()
        with (
            patch.object(overlay, "Gtk4LayerShell") as shell,
            patch.object(pill, "get_surface") as get_surface,
        ):
            pill._on_resize(pill._canvas, 500, 100)
        shell.set_margin.assert_not_called()
        get_surface.assert_not_called()

    def test_no_code_path_unmaps_the_overlay(self):
        # Unmapping a layer-shell surface crashes COSMIC and an inert pill is never
        # shown, so the overlay only calls set_visible(True) and nothing that hides.
        import ast
        import inspect

        from quoth import overlay

        calls = [
            node
            for node in ast.walk(ast.parse(inspect.getsource(overlay)))
            if isinstance(node, ast.Call)
        ]
        args = [
            ast.literal_eval(node.args[0])
            for node in calls
            if getattr(node.func, "attr", None) == "set_visible"
        ]
        assert args and all(arg is True for arg in args)
        assert not {getattr(node.func, "attr", None) for node in calls} & {
            "hide",
            "close",
            "unmap",
        }


class TestOverlayDraw:
    """Draw every state on a real surface; each state takes its own drawing path."""

    def _draw(self, state, **attrs):
        from quoth.overlay import WaveformPill

        pill = WaveformPill()
        pill._state = state
        for name, value in attrs.items():
            setattr(pill, name, value)
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 240, 48)
        pill._draw(None, cairo.Context(surface), 240, 48)
        return surface

    @pytest.mark.parametrize(
        ("state", "attrs"),
        [
            ("recording", {}),
            ("transcribing", {}),
            ("error", {"_error_message": "boom"}),
            ("hint", {"_hint_message": "say something"}),
        ],
    )
    def test_every_state_draws(self, state, attrs):
        assert any(self._draw(state, **attrs).get_data())

    def test_idle_draws_nothing(self):
        # The hidden overlay stays mapped, so idle must paint a fully transparent frame.
        assert not any(self._draw("idle").get_data())

    def test_entry_starts_transparent(self):
        # The alpha/scale wrapper is overlay-only and wraps whatever draws inside it;
        # an entry fades in from nothing.
        with patch("quoth.overlay.time.monotonic", return_value=100.0):
            surface = self._draw("recording", _is_entering=True, _entry_start=100.0)
        assert not any(surface.get_data())

    def test_a_failing_draw_logs_once_and_goes_idle(self, caplog):
        # Every tick queues a draw, so a defect must not repeat once per frame.
        from quoth.overlay import WaveformPill

        pill = WaveformPill()
        pill._state = "recording"
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 240, 48)
        with (
            patch("quoth.overlay.draw_pill", side_effect=ValueError("bad frame")),
            patch("quoth.overlay.GLib.idle_add"),  # the hide, left to its own test
        ):
            for _ in range(3):
                pill._draw(None, cairo.Context(surface), 240, 48)
        assert pill._state == "idle"
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert [r.getMessage() for r in errors] == [
            "Overlay draw failed; hiding the overlay"
        ]

    def _fail_a_draw(self, pill):
        """Fail one recording draw and run the idle work it queues."""
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 240, 48)
        with (
            patch("quoth.overlay.draw_pill", side_effect=ValueError("bad frame")),
            patch("quoth.overlay.GLib.idle_add") as idle_add,
        ):
            pill._draw(None, cairo.Context(surface), 240, 48)
        for call in idle_add.call_args_list:
            call.args[0]()

    def test_a_failing_draw_ends_as_a_normal_hide_does(self):
        # An idle surface left with a full input region would swallow clicks unseen.
        from quoth.overlay import WaveformPill

        pill = WaveformPill()
        surface = MagicMock()
        with patch.object(pill, "get_surface", return_value=surface):
            pill.show_recording()
            self._fail_a_draw(pill)
        assert surface.set_input_region.call_args.args[0].is_empty()
        assert pill._tick_id is None
        assert (pill._state, pill._is_entering, pill._is_exiting) == (
            "idle",
            False,
            False,
        )

    def test_the_stop_after_a_failed_draw_leaves_the_overlay_idle(self):
        # No tick runs after the failure to serve a deferred hide, so none may be left.
        from quoth.overlay import WaveformPill

        pill = WaveformPill()
        pill.show_recording()
        self._fail_a_draw(pill)
        pill.show_transcribing()
        pill.hide_pill()
        assert (pill._state, pill._hide_pending) == ("idle", False)

    def test_large_back_draw_keeps_the_authored_geometry_centered(self):
        from quoth.overlay import WaveformPill

        pill = WaveformPill(theme=motion_theme("motion-2-large-back.toml"))
        pill._state = "recording"
        assert pill._surface_size() == (426, 82)
        width, height = 501, 101
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        context = cairo.Context(surface)
        with patch("quoth.overlay.draw_pill") as draw:
            pill._draw(None, context, width, height)
        assert draw.call_args.kwargs["width"] == 420
        assert draw.call_args.kwargs["height"] == 80
        matrix = context.get_matrix()
        assert (matrix.x0, matrix.y0) == (40.5, 10.5)


class TestOrbStyle:
    """The orb style: same window, state machine and lifecycle, different drawing."""

    def _make_orb_pill(self):
        from quoth.overlay import WaveformPill

        return WaveformPill(style="orb")

    def test_pill_style_shows_text_and_orb_does_not(self):
        from quoth.overlay import WaveformPill

        assert WaveformPill().shows_text is True
        assert self._make_orb_pill().shows_text is False

    def test_surface_is_the_orb_square(self):
        from quoth.overlay import ORB_SURFACE

        pill = self._make_orb_pill()
        assert pill._surface_size() == (ORB_SURFACE, ORB_SURFACE)
        assert pill._canvas.get_content_width() == ORB_SURFACE

    def test_state_machine_is_unchanged(self):
        pill = self._make_orb_pill()
        assert pill._state == "idle"
        pill.show_recording()
        assert pill._state == "recording"
        pill.show_transcribing()
        assert pill._state == "transcribing"

    def test_amplitude_feeds_the_director(self):
        pill = self._make_orb_pill()
        pill.show_recording()
        assert pill._director._level == 0.0
        pill.push_amplitude(0.8)
        assert pill._director._level > 0.0

    def test_a_new_recording_resets_the_voice_level(self):
        pill = self._make_orb_pill()
        pill.show_recording()
        pill.push_amplitude(0.9)
        pill.show_transcribing()
        pill.show_recording()
        assert pill._director._level == 0.0

    def test_tick_directs_the_orb(self):
        # Two ticks with a real delta: the orb's params must be morphing toward
        # the state's look and its flow must have advanced.
        pill = self._make_orb_pill()
        pill.show_recording()
        pill.show_transcribing()
        clock = MagicMock()
        clock.get_frame_time.side_effect = [0, 50_000, 100_000]
        flow = pill._orb._t  # constructing an orb warms its simulation
        for _ in range(3):
            pill._tick(None, clock)
        assert pill._orb._params.time_scale > 1.0
        assert pill._orb._t > flow

    def test_theme_swap_rebuilds_the_orb(self):
        from quoth.theme import preset_theme

        pill = self._make_orb_pill()
        before = pill._orb
        pill.apply_theme(preset_theme("nord"))
        assert pill._orb is not before

    def test_pill_style_has_no_orb(self):
        from quoth.overlay import WaveformPill

        pill = WaveformPill()
        assert pill._orb is None and pill._director is None

    def test_hide_keeps_orb_geometry(self):
        from quoth.overlay import ORB_SURFACE, WaveformPill

        pill = WaveformPill(style="orb")
        pill.set_motion_enabled(False)
        pill.show_recording()
        pill.hide_pill()
        assert pill._canvas.get_content_width() == ORB_SURFACE
        assert pill._canvas.get_content_height() == ORB_SURFACE


class TestSetStyle:
    """The orb switch's hot-apply: an in-place swap, never a window rebuild."""

    def test_pill_to_orb_builds_the_orb(self):
        from quoth.overlay import ORB_SURFACE, WaveformPill

        pill = WaveformPill()
        pill.set_style("orb")
        assert pill._orb is not None and pill._director is not None
        assert pill.shows_text is False
        assert pill._surface_size() == (ORB_SURFACE, ORB_SURFACE)
        assert pill._canvas.get_content_width() == ORB_SURFACE

    def test_orb_back_to_pill_drops_it(self):
        from quoth.overlay import WaveformPill

        pill = WaveformPill(style="orb")
        pill.set_style("pill")
        assert pill._orb is None and pill._director is None
        assert pill.shows_text is True
        g = pill._theme.geometry
        assert pill._surface_size() == (g.width, g.height)

    def test_same_style_is_a_no_op(self):
        pill = self._orb_pill()
        orb_before = pill._orb
        pill.set_style("orb")
        assert pill._orb is orb_before  # no needless reseed mid-flight

    def test_state_survives_the_switch(self):
        from quoth.overlay import WaveformPill

        pill = WaveformPill()
        pill.show_recording()
        pill.set_style("orb")
        assert pill._state == "recording"

    def test_large_back_padding_tracks_style_and_motion_changes(self):
        from quoth.overlay import ORB_SURFACE, WaveformPill

        pill = WaveformPill(theme=motion_theme("motion-2-large-back.toml"))
        assert pill._surface_size() == (426, 82)
        pill.set_style("orb")
        assert pill._surface_size() == (ORB_SURFACE + 2, ORB_SURFACE + 2)
        pill.set_motion_enabled(False)
        assert pill._surface_size() == (ORB_SURFACE, ORB_SURFACE)
        pill.set_style("pill")
        assert pill._surface_size() == (420, 80)

    def test_theme_changes_share_the_surface_sizing_path(self):
        from quoth.overlay import WaveformPill

        pill = WaveformPill(theme=motion_theme("motion-1-slow-linear.toml"))
        pill.apply_theme(motion_theme("motion-2-large-back.toml"))
        assert pill._surface_size() == (426, 82)
        assert pill._canvas.get_content_width() == 426
        assert pill._canvas.get_content_height() == 82

    def test_non_back_and_static_fixtures_keep_nominal_size(self):
        from quoth.overlay import WaveformPill

        for name in (
            "motion-1-slow-linear.toml",
            "motion-3-compact-fast.toml",
            "motion-4-still-control.toml",
        ):
            theme = motion_theme(name)
            assert WaveformPill(theme=theme)._surface_size() == (
                theme.geometry.width,
                theme.geometry.height,
            )

    @pytest.mark.parametrize("state", ["recording", "error"])
    def test_switch_back_to_pill_is_clickable_before_the_relayout(self, state):
        # The surface keeps the orb's size until it relays out, and no resize signal
        # has arrived yet: the whole pill must still take the cancel click.
        from quoth.overlay import ORB_SURFACE

        pill = self._orb_pill()
        surface = MagicMock()
        surface.get_width.return_value = ORB_SURFACE
        surface.get_height.return_value = ORB_SURFACE
        with patch.object(pill, "get_surface", return_value=surface):
            pill.set_style("pill")
            assert surface.set_input_region.call_args.args[0].is_empty()  # idle
            if state == "recording":
                pill.show_recording()
            else:
                pill.show_error("Test error")
        region = surface.set_input_region.call_args.args[0]
        g = pill._theme.geometry
        rect = cairo.RectangleInt(0, 0, g.width, g.height)
        assert region.contains_rectangle(rect) == cairo.REGION_OVERLAP_IN

    def _orb_pill(self):
        from quoth.overlay import WaveformPill

        return WaveformPill(style="orb")


class TestBackEasingPosition:
    def test_visible_bottom_margin_stays_authored(self):
        from quoth import overlay

        pill = overlay.WaveformPill(theme=motion_theme("motion-2-large-back.toml"))
        with patch.object(overlay, "Gtk4LayerShell") as shell:
            shell.Edge.TOP = "top"
            shell.Edge.RIGHT = "right"
            shell.Edge.BOTTOM = "bottom"
            shell.Edge.LEFT = "left"
            pill._apply_position()
        shell.set_margin.assert_any_call(pill, "bottom", 83)

    def test_padding_adjustment_clamps_at_signed_int_minimum(self):
        from quoth import overlay

        theme = motion_theme("motion-2-large-back.toml")
        theme = dataclasses.replace(
            theme,
            position=dataclasses.replace(
                theme.position,
                anchor="top-left",
                margin=(-(2**31), 0, 0, -(2**31)),
            ),
        )
        pill = overlay.WaveformPill(theme=theme)
        with patch.object(overlay, "Gtk4LayerShell") as shell:
            shell.Edge.TOP = "top"
            shell.Edge.RIGHT = "right"
            shell.Edge.BOTTOM = "bottom"
            shell.Edge.LEFT = "left"
            pill._apply_position()
        shell.set_margin.assert_any_call(pill, "top", -(2**31))
        shell.set_margin.assert_any_call(pill, "left", -(2**31))


class TestOverlayAllocation:
    @staticmethod
    def _name_edges(shell):
        shell.Edge.TOP = "top"
        shell.Edge.RIGHT = "right"
        shell.Edge.BOTTOM = "bottom"
        shell.Edge.LEFT = "left"

    @pytest.mark.parametrize(
        ("width", "height", "bottom_margin"),
        [(500, 100, 34), (501, 101, 34), (200, 40, 60)],
    )
    def test_resize_uses_allocation_for_position(self, width, height, bottom_margin):
        from quoth import overlay

        pill = overlay.WaveformPill()
        pill._state = "recording"
        with patch.object(overlay, "Gtk4LayerShell") as shell:
            self._name_edges(shell)
            pill._canvas.emit("resize", width, height)

        shell.set_margin.assert_any_call(pill, "bottom", bottom_margin)

    @pytest.mark.parametrize(
        ("anchor", "anchored"),
        [
            ("top-left", {"top", "left"}),
            ("top-center", {"top"}),
            ("top-right", {"top", "right"}),
            ("center-left", {"left"}),
            ("center", set()),
            ("center-right", {"right"}),
            ("bottom-left", {"bottom", "left"}),
            ("bottom-center", {"bottom"}),
            ("bottom-right", {"bottom", "right"}),
        ],
    )
    def test_all_anchors_offset_only_their_anchored_edges(self, anchor, anchored):
        from quoth import overlay

        theme = dataclasses.replace(
            overlay.DEFAULT_THEME,
            position=dataclasses.replace(
                overlay.DEFAULT_THEME.position,
                anchor=anchor,
                margin=(11, 22, 33, 44),
            ),
        )
        pill = overlay.WaveformPill(theme=theme)
        with patch.object(overlay, "Gtk4LayerShell") as shell:
            self._name_edges(shell)
            pill._apply_position((500, 100))

        raw_margins = {"top": -15, "right": -108, "bottom": 7, "left": -86}
        margins = {
            call.args[1]: call.args[2] for call in shell.set_margin.call_args_list
        }
        assert margins == {
            edge: raw_margins[edge] if edge in anchored else 0 for edge in raw_margins
        }

    def test_theme_request_does_not_replace_the_current_allocation(self):
        from quoth import overlay

        pill = overlay.WaveformPill()
        pill._state = "recording"
        pill._canvas = MagicMock()
        surface = MagicMock()
        with (
            patch.object(overlay, "Gtk4LayerShell") as shell,
            patch.object(pill, "get_surface", return_value=surface),
            patch.object(pill, "get_width", return_value=500),
            patch.object(pill, "get_height", return_value=100),
        ):
            self._name_edges(shell)
            pill.apply_theme(motion_theme("motion-2-large-back.toml"))

        pill._canvas.set_content_width.assert_called_once_with(426)
        pill._canvas.set_content_height.assert_called_once_with(82)
        shell.set_margin.assert_any_call(pill, "bottom", 74)
        region = surface.set_input_region.call_args.args[0]
        rect = cairo.RectangleInt(0, 0, 500, 100)
        assert region.contains_rectangle(rect) == cairo.REGION_OVERLAP_IN

    def test_growth_shrink_style_and_motion_follow_the_live_allocation(self):
        from quoth import overlay

        pill = overlay.WaveformPill(theme=motion_theme("motion-2-large-back.toml"))
        pill._state = "recording"
        pill._canvas = MagicMock()
        surface = MagicMock()
        with (
            patch.object(overlay, "Gtk4LayerShell") as shell,
            patch.object(pill, "get_surface", return_value=surface),
            patch.object(pill, "get_width", return_value=500),
            patch.object(pill, "get_height", return_value=100),
        ):
            self._name_edges(shell)
            pill.set_motion_enabled(False)
            shell.set_margin.assert_any_call(pill, "bottom", 74)

            shell.reset_mock()
            pill.set_style("orb")
            shell.set_margin.assert_any_call(pill, "bottom", 82)

            shell.reset_mock()
            pill._on_resize(pill._canvas, 90, 90)
            shell.set_margin.assert_any_call(pill, "bottom", 84)

            shell.reset_mock()
            pill._on_resize(pill._canvas, 501, 101)
            shell.set_margin.assert_any_call(pill, "bottom", 82)

        region = surface.set_input_region.call_args.args[0]
        rect = cairo.RectangleInt(0, 0, 501, 101)
        assert region.contains_rectangle(rect) == cairo.REGION_OVERLAP_IN


class TestSurfaceIdentity:
    def test_layer_surface_is_named_quoth(self, layer_shell):
        from quoth import overlay

        overlay.WaveformPill()
        layer_shell.set_namespace.assert_called_once()
        assert layer_shell.set_namespace.call_args.args[1] == "quoth"

    def test_layer_surface_is_a_no_focus_overlay(self, layer_shell):
        # A surface that took keyboard focus would take the paste from the editor.
        from quoth import overlay

        pill = overlay.WaveformPill()
        layer_shell.init_for_window.assert_called_once_with(pill)
        layer_shell.set_layer.assert_called_once_with(pill, layer_shell.Layer.OVERLAY)
        layer_shell.set_keyboard_mode.assert_called_once_with(
            pill, layer_shell.KeyboardMode.NONE
        )

    @pytest.mark.parametrize("motion", [False, True])
    def test_hide_keeps_the_layer_surface_mapped(self, motion):
        # Unmapping a layer-shell surface crashes COSMIC, so a hide only redraws.
        from quoth import overlay
        from tests.support import wait_until

        pill = overlay.WaveformPill()
        pill.set_motion_enabled(motion)
        pill.show_recording()
        pill.hide_pill()
        wait_until(lambda: pill._state == "idle")
        assert pill.get_mapped()

    def test_a_modal_window_does_not_grab_the_overlay(self):
        # A modal window's grab takes every pointer event in its window group, so the
        # overlay sharing one with Settings would lose its cancel click.
        from gi.repository import Gtk

        from quoth.overlay import WaveformPill

        pill = WaveformPill()
        modal = Gtk.Window(modal=True)
        assert pill.get_group() is not modal.get_group()

    def test_pill_shows_a_hand_cursor(self):
        from quoth.overlay import WaveformPill

        pill = WaveformPill()
        cursor = pill._canvas.get_cursor()
        assert cursor is not None and cursor.get_name() == "pointer"
