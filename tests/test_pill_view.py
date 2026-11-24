# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.pill_view`, the theme-aware pill and orb view layer.

`draw_pill` runs over every state and the whole transcribing transition on a real Cairo
surface; the widget tests need a display.
"""

import dataclasses
from pathlib import Path

import cairo
import numpy as np
import pytest

from quoth.pill_render import DOT_X, NUM_BARS, TRANSITION_TOTAL_S
from quoth.pill_view import (
    OverlayState,
    PillFrame,
    draw_pill,
    fit_height,
    orb_seed,
    pill_width,
)
from quoth.theme import DEFAULT_THEME, list_presets, preset_theme

FIXTURES = Path(__file__).with_name("fixtures")


def _draw(frame, theme=DEFAULT_THEME):
    """Render one frame onto a surface the size of the theme's pill."""
    surface = cairo.ImageSurface(
        cairo.FORMAT_ARGB32, theme.geometry.width, theme.geometry.height
    )
    draw_pill(
        cairo.Context(surface),
        theme=theme,
        frame=frame,
        width=theme.geometry.width,
        height=theme.geometry.height,
    )
    surface.flush()
    return surface


def _rgba(surface, x, y):
    """Return the pixel at x, y as straight (unpremultiplied) RGBA floats."""
    i = y * surface.get_stride() + x * 4
    b, g, r, a = surface.get_data()[i : i + 4]
    # A blank pixel reads as (0, 0, 0, 0).
    return r / max(a, 1), g / max(a, 1), b / max(a, 1), a / 255


class TestDrawPill:
    @pytest.mark.parametrize(
        "frame",
        [
            PillFrame(
                state=OverlayState.RECORDING, amplitudes=[0.5] * NUM_BARS, timer="0:07"
            ),
            PillFrame(
                state=OverlayState.RECORDING, amplitudes=[0.5] * NUM_BARS, timer=None
            ),
            PillFrame(state=OverlayState.RECORDING, amplitudes=[0.0] * NUM_BARS),
            # Entering transcribing without ever having recorded a level.
            PillFrame(
                state=OverlayState.TRANSCRIBING,
                amplitudes=[],
                frozen=[],
                transition_elapsed=0.1,
            ),
        ],
        ids=["recording", "without-a-timer", "silence", "from-an-empty-freeze"],
    )
    def test_draws(self, frame):
        # A spot clear of the dot, wave and timer shows the theme's background.
        assert _rgba(_draw(frame), 120, 6) == pytest.approx(
            DEFAULT_THEME.palette.background, abs=0.01
        )

    def test_transition_start_to_finish(self):
        frame = PillFrame(
            state=OverlayState.TRANSCRIBING,
            amplitudes=[0.5] * NUM_BARS,
            frozen=[0.5] * NUM_BARS,
        )
        # Every point of decay -> expand -> wave, including both boundaries.
        frames = [
            _draw(
                dataclasses.replace(
                    frame, transition_elapsed=TRANSITION_TOTAL_S * i / 20
                )
            )
            for i in range(41)
        ]
        # It opens on the recording dot at full strength and lands on the static
        # transcribing look, which it then holds.
        assert _rgba(frames[0], 24, 24) == pytest.approx(
            DEFAULT_THEME.palette.dot_recording, abs=0.01
        )
        static = _draw(dataclasses.replace(frame, motion_enabled=False))
        for landed in (frames[20], frames[40]):
            # Within one byte: a lerp's last ulp must not decide a colour rounding.
            gap = np.frombuffer(landed.get_data(), np.uint8).astype(int)
            gap -= np.frombuffer(static.get_data(), np.uint8)
            assert abs(gap).max() <= 1

    def test_every_preset_draws(self):
        # Each preset in its own colours: the one draw path takes them from the theme.
        for name in list_presets():
            theme = preset_theme(name)
            surface = _draw(
                PillFrame(
                    state=OverlayState.RECORDING,
                    amplitudes=[0.5] * NUM_BARS,
                    timer="0:07",
                ),
                theme=theme,
            )
            geometry = theme.geometry
            assert _rgba(surface, geometry.width // 2, 6) == pytest.approx(
                theme.palette.background, abs=0.01
            )
            assert _rgba(surface, DOT_X, geometry.height // 2) == pytest.approx(
                theme.palette.dot_recording, abs=0.01
            )

    def test_timer_draws_between_the_wave_and_the_right_inset(self):
        import numpy as np

        timed = PillFrame(
            state=OverlayState.RECORDING, amplitudes=[0.5] * NUM_BARS, timer="0:07"
        )
        untimed = dataclasses.replace(timed, timer=None)
        changed = np.frombuffer(_draw(timed).get_data(), np.uint8) != np.frombuffer(
            _draw(untimed).get_data(), np.uint8
        )
        columns = np.flatnonzero(changed.reshape(48, 240, 4).any(axis=(0, 2)))
        # The timer is all that differs: right of the stock wave's end (x=163) and
        # left of the 24 px right inset.
        assert columns.size > 0
        assert 163 < columns.min() and columns.max() < 216

    def test_compact_timer_reserves_space_before_the_wave(self):
        from unittest.mock import patch

        compact = dataclasses.replace(
            DEFAULT_THEME,
            geometry=dataclasses.replace(
                DEFAULT_THEME.geometry, width=120, height=32, bar_gap=1
            ),
            font=dataclasses.replace(DEFAULT_THEME.font, size=10),
        )
        with patch("quoth.pill_view.draw_waveform") as wave:
            _draw(
                PillFrame(
                    state=OverlayState.RECORDING,
                    amplitudes=[0.5] * NUM_BARS,
                    timer="0:07",
                ),
                compact,
            )
        from quoth.pill_render import ink_extents, label_layout, tabular_advance
        from quoth.pill_view import LABEL_LEFT_X, LABEL_PAD_X

        cr = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 120, 32))
        layout = label_layout(
            cr,
            "0:07",
            font_family=compact.font.family,
            font_size=compact.font.size,
            max_width=120 - LABEL_LEFT_X - LABEL_PAD_X,
            tabular=True,
        )
        timer_left = (
            120 - LABEL_PAD_X - tabular_advance(layout) + ink_extents(layout)[0]
        )
        assert wave.call_args.kwargs["end_x"] + 4 <= timer_left

    def test_motion_off_draws_the_full_static_transcribing_state(self):
        from unittest.mock import patch

        frame = PillFrame(
            state=OverlayState.TRANSCRIBING,
            amplitudes=[0.5] * NUM_BARS,
            frozen=[0.5] * NUM_BARS,
            transition_elapsed=0.0,
            motion_enabled=False,
        )
        with (
            patch("quoth.pill_view.draw_dot") as dot,
            patch("quoth.pill_view.draw_waveform") as wave,
        ):
            _draw(frame)
        dot.assert_not_called()
        assert wave.call_args.kwargs["start_x"] == 16
        assert wave.call_args.kwargs["end_x"] == 224

    def test_motion_off_holds_the_recording_dot_still(self):
        # Without motion the ripple does not run, so the phase changes nothing.
        frame = PillFrame(
            state=OverlayState.RECORDING,
            amplitudes=[0.5] * NUM_BARS,
            motion_enabled=False,
        )
        early = _draw(dataclasses.replace(frame, phase=0.5))
        late = _draw(dataclasses.replace(frame, phase=2.0))
        assert bytes(early.get_data()) == bytes(late.get_data())

    def test_extreme_transcribing_expansion_stays_inside_the_pill(self):
        from unittest.mock import patch

        compact = dataclasses.replace(
            DEFAULT_THEME,
            geometry=dataclasses.replace(
                DEFAULT_THEME.geometry, width=24, height=24, bar_gap=50
            ),
        )
        frame = PillFrame(
            state=OverlayState.TRANSCRIBING,
            amplitudes=[0.5] * NUM_BARS,
            frozen=[0.5] * NUM_BARS,
            transition_elapsed=0.35,
        )
        with patch("quoth.pill_view.draw_waveform") as wave:
            _draw(frame, compact)
        start_x = wave.call_args.kwargs["start_x"]
        end_x = wave.call_args.kwargs["end_x"]
        half_stroke = compact.geometry.line_width / 2
        assert half_stroke <= start_x <= compact.geometry.width - half_stroke
        assert half_stroke <= end_x <= compact.geometry.width - half_stroke


class TestPillWidth:
    """The width follows the theme's aspect, so each view is the same pill, resized."""

    def test_swatch_is_half_the_live_pill(self):
        assert pill_width(DEFAULT_THEME, 24) == 120

    def test_full_size_is_the_live_pill(self):
        assert pill_width(DEFAULT_THEME, 48) == DEFAULT_THEME.geometry.width

    def test_hero_reproduces_the_demo_it_replaced(self):
        # Any height keeps the stock 5:1 aspect; a change to the rule would silently
        # resize every view built on it.
        assert pill_width(DEFAULT_THEME, 60) == 300

    def test_every_preset_shares_the_stock_aspect(self):
        # Palette-only by convention, so all four swatches line up at one width.
        assert all(pill_width(preset_theme(n), 24) == 120 for n in list_presets())

    def test_aspect_follows_a_themed_geometry(self):
        # A wide pill previews wide instead of being squashed into a fixed box.
        wide = dataclasses.replace(
            DEFAULT_THEME,
            geometry=dataclasses.replace(DEFAULT_THEME.geometry, width=900),
        )
        assert pill_width(wide, 24) == 450

    def test_a_tall_theme_previews_narrower(self):
        tall = dataclasses.replace(
            DEFAULT_THEME,
            geometry=dataclasses.replace(DEFAULT_THEME.geometry, height=96),
        )
        assert pill_width(tall, 24) == 60


class TestFitHeight:
    """The preview shows the pill at its themed size and only ever shrinks it."""

    def test_stock_pill_is_untouched(self):
        assert fit_height(DEFAULT_THEME, 420) == DEFAULT_THEME.geometry.height

    def test_a_themed_height_is_honoured_at_1_to_1(self):
        # A fixed preview box would draw this at 48 and squash the font with it, hiding
        # the very key a ricer changes.
        tall = dataclasses.replace(
            DEFAULT_THEME,
            geometry=dataclasses.replace(DEFAULT_THEME.geometry, height=80, width=400),
        )
        assert fit_height(tall, 420) == 80

    def test_never_magnifies_a_small_pill(self):
        small = dataclasses.replace(
            DEFAULT_THEME,
            geometry=dataclasses.replace(DEFAULT_THEME.geometry, height=24, width=60),
        )
        assert fit_height(small, 420) == 24

    def test_too_wide_a_pill_scales_down_keeping_aspect(self):
        wide = dataclasses.replace(
            DEFAULT_THEME,
            geometry=dataclasses.replace(DEFAULT_THEME.geometry, width=840, height=48),
        )
        assert fit_height(wide, 420) == 24  # half scale
        assert pill_width(wide, fit_height(wide, 420)) == 420  # exactly fills

    def test_the_widest_themeable_pill_still_fits(self):
        # geometry clamps width at 2000; the preview must stay inside its box anyway.
        widest = dataclasses.replace(
            DEFAULT_THEME,
            geometry=dataclasses.replace(
                DEFAULT_THEME.geometry, width=2000, height=200
            ),
        )
        assert pill_width(widest, fit_height(widest, 420)) <= 420


class TestOrbDirector:
    """State and level map onto orb params through lerps, never jumps."""

    def _settled(self, director, state, steps=200, dt=0.05):
        params = None
        for _ in range(steps):
            params = director.update(dt, state)
        return params

    def test_idle_settles_at_neutral(self):
        from quoth.pill_view import OrbDirector, OverlayState

        director = OrbDirector()
        # Away from neutral, so idle must return.
        self._settled(director, OverlayState.TRANSCRIBING)
        p = self._settled(director, OverlayState.IDLE)
        assert p.time_scale == pytest.approx(1.0)
        assert p.glow == pytest.approx(1.0)
        assert all(e == pytest.approx(1.0) for e in p.emphasis)

    def test_voice_speeds_and_brightens_listening(self):
        from quoth.pill_view import OrbDirector, OverlayState

        quiet, loud = OrbDirector(), OrbDirector()
        for _ in range(50):
            loud.push_level(1.0)
        p_quiet = self._settled(quiet, OverlayState.RECORDING)
        p_loud = self._settled(loud, OverlayState.RECORDING)
        assert p_loud.time_scale > p_quiet.time_scale
        assert p_loud.glow > p_quiet.glow
        assert p_loud.swell > p_quiet.swell

    def test_level_attack_beats_release(self):
        from quoth.pill_view import OrbDirector

        d = OrbDirector()
        d.push_level(1.0)
        risen = d._level
        d.push_level(0.0)
        assert risen > 0.3  # one loud sample already registers
        assert d._level > risen * 0.5  # but one quiet sample decays gently

    def test_transcribing_leans_into_its_role(self):
        from quoth.pill_view import _ORB_BLOBS, OrbDirector, OverlayState

        p = self._settled(OrbDirector(), OverlayState.TRANSCRIBING)
        roles = [role for role, _ in _ORB_BLOBS]
        favoured = p.emphasis[roles.index("dot_transcribing")]
        assert favoured > max(
            e for i, e in enumerate(p.emphasis) if roles[i] != "dot_transcribing"
        )
        assert p.time_scale > 1.5  # thinking flows faster than ambient

    def test_error_leans_into_the_error_dot_and_hint_dims(self):
        from quoth.pill_view import _ORB_BLOBS, OrbDirector

        roles = [role for role, _ in _ORB_BLOBS]
        p_err = self._settled(OrbDirector(), "error")
        assert p_err.emphasis[roles.index("dot_error")] > 1.5
        p_hint = self._settled(OrbDirector(), "hint")
        assert p_hint.glow < 1.0
        assert all(e < 1.0 for e in p_hint.emphasis)

    def test_states_morph_rather_than_switch(self):
        from quoth.pill_view import OrbDirector, OverlayState

        d = OrbDirector()
        first = d.update(0.016, OverlayState.TRANSCRIBING)
        settled = self._settled(d, OverlayState.TRANSCRIBING)
        # One frame in, the params are on their way but nowhere near the target.
        assert 1.0 < first.time_scale < settled.time_scale

    def test_voice_grows_the_orb_toward_the_full_box(self):
        from quoth.pill_view import OrbDirector, OverlayState

        quiet, loud = OrbDirector(), OrbDirector()
        for _ in range(50):
            loud.push_level(1.0)
        p_quiet = self._settled(quiet, OverlayState.RECORDING)
        p_loud = self._settled(loud, OverlayState.RECORDING)
        assert p_loud.scale > p_quiet.scale
        assert p_quiet.scale < 1.0  # rests small, so growth has somewhere to go
        assert p_loud.scale <= 1.0  # never past the box: the surface would clip it

    def test_thinking_pulses_its_size(self):
        from quoth.pill_view import OrbDirector, OverlayState

        d = OrbDirector()
        scales = [d.update(1 / 30, OverlayState.TRANSCRIBING).scale for _ in range(600)]
        settled = scales[300:]
        assert max(settled) - min(settled) > 0.02  # a visible breath, not a hold
        assert max(settled) <= 1.0

    def test_voice_summons_the_live_dot(self):
        # The green dot is the pill's "mic hot" signal; the orb glints it in
        # proportion to the voice rather than hosting it permanently.
        from quoth.pill_view import _ORB_BLOBS, OrbDirector, OverlayState

        roles = [role for role, _ in _ORB_BLOBS]
        quiet, loud = OrbDirector(), OrbDirector()
        for _ in range(50):
            loud.push_level(1.0)
        idx = roles.index("dot_recording")
        assert (
            self._settled(loud, OverlayState.RECORDING).emphasis[idx]
            > self._settled(quiet, OverlayState.RECORDING).emphasis[idx]
        )

    def test_room_noise_reads_as_calm(self):
        # The metering floor: a quiet room's residual level must not excite the orb.
        from quoth.pill_view import OrbDirector, OverlayState

        noisy = OrbDirector()
        for _ in range(50):
            noisy.push_level(0.1)
        p = self._settled(noisy, OverlayState.RECORDING)
        assert p.time_scale == pytest.approx(1.0, abs=0.01)

    def test_duration_scale_slows_the_morph(self):
        from quoth.pill_view import OrbDirector, OverlayState

        fast, slow = OrbDirector(duration_scale=1.0), OrbDirector(duration_scale=4.0)
        p_fast = fast.update(0.05, OverlayState.TRANSCRIBING)
        p_slow = slow.update(0.05, OverlayState.TRANSCRIBING)
        assert p_fast.time_scale > p_slow.time_scale

    def test_duration_scale_slows_the_transcribing_pulse_clock(self):
        from quoth.pill_view import OrbDirector, OverlayState

        fast, slow = OrbDirector(duration_scale=1.0), OrbDirector(duration_scale=4.0)
        fast.update(0.4, OverlayState.TRANSCRIBING)
        slow.update(0.4, OverlayState.TRANSCRIBING)
        assert fast._t == pytest.approx(0.4)
        assert slow._t == pytest.approx(0.1)

    def test_motion_off_settles_without_pulsing_but_keeps_voice_response(self):
        from quoth.pill_view import OrbDirector, OverlayState

        director = OrbDirector()
        thinking = director.update(1.0, OverlayState.TRANSCRIBING, motion_enabled=False)
        assert thinking.scale == pytest.approx(0.88)
        assert director._t == 0.0
        director.push_level(1.0)
        listening = director.update(1.0, OverlayState.RECORDING, motion_enabled=False)
        assert listening.time_scale > 1.0


class TestMakeOrb:
    """Construction catches the orb mid-life; identity stays fully seeded."""

    def test_warmed_orbs_are_caught_at_different_moments(self):
        from quoth.pill_view import make_orb

        a = make_orb(DEFAULT_THEME)
        b = make_orb(DEFAULT_THEME)
        # Same seeded identity, different point on its timeline: without this,
        # every Settings open replays the same opening on the same layout.
        assert a._t != b._t

    def test_cold_construction_is_deterministic(self):
        import numpy as np

        from quoth.pill_view import make_orb

        a = make_orb(DEFAULT_THEME, warmed=False)
        b = make_orb(DEFAULT_THEME, warmed=False)
        assert a._t == 0.0
        assert np.array_equal(a._dye, b._dye)

    def test_warm_up_moves_the_centres_off_the_shared_layout(self):
        import numpy as np

        from quoth.pill_view import make_orb

        warmed = make_orb(DEFAULT_THEME)
        assert not np.array_equal(warmed._centres, warmed._homes)


class TestOrbSeed:
    """An orb's identity is its palette: same colours, same orb, forever."""

    def test_stable_for_the_same_theme(self):
        assert orb_seed(DEFAULT_THEME) == orb_seed(DEFAULT_THEME)

    def test_each_preset_gets_its_own(self):
        seeds = [orb_seed(preset_theme(n)) for n in list_presets()]
        assert len(set(seeds)) == len(seeds)

    def test_a_palette_edit_changes_it(self):
        recoloured = dataclasses.replace(
            DEFAULT_THEME,
            palette=dataclasses.replace(
                DEFAULT_THEME.palette, accent=(0.9, 0.1, 0.5, 1.0)
            ),
        )
        assert orb_seed(recoloured) != orb_seed(DEFAULT_THEME)

    def test_geometry_edits_do_not_reshuffle_it(self):
        # Identity follows the colours alone; resizing the pill keeps the orb.
        resized = dataclasses.replace(
            DEFAULT_THEME,
            geometry=dataclasses.replace(DEFAULT_THEME.geometry, width=500),
        )
        assert orb_seed(resized) == orb_seed(DEFAULT_THEME)

    @pytest.mark.parametrize(
        ("theme", "expected"),
        [
            (
                DEFAULT_THEME,
                73863214341422448447910049978176966609746762949131795457522764884341017742269,
            ),
            (
                preset_theme("default"),
                52297426625225105295424355166736856942951826203727574814717818200495683171264,
            ),
            (
                preset_theme("catppuccin-mocha"),
                91541422170991887366690304176510045114449895756491093154789348302303469167672,
            ),
            (
                preset_theme("dracula"),
                39617742949090097620369777149168648002782825363561305438263171517659057460193,
            ),
            (
                preset_theme("nord"),
                78618995670471994313371068748871391216187994120745223860555320013271017089881,
            ),
        ],
    )
    def test_historical_palette_seeds_are_frozen(self, theme, expected):
        assert orb_seed(theme) == expected


def test_takes_timer_rests_at_zero_until_a_take(display):
    """The Try-it pill must not look live while idle.

    Its clock reads 0:00, and only a take makes it move.
    """
    from unittest.mock import patch

    from quoth.pill_view import PillView
    from quoth.theme import DEFAULT_THEME

    view = PillView(DEFAULT_THEME, height=40, timer=True, times_takes=True)
    assert view._timer_text() == "0:00"
    with patch("quoth.pill_view.time.monotonic", return_value=100.0):
        view.start_timer()
    with patch("quoth.pill_view.time.monotonic", return_value=107.0):
        assert view._timer_text() == "0:07"
    view.stop_timer(keep=False)
    assert view._timer_text() == "0:00"


@pytest.mark.parametrize("kind", ["pill", "orb"])
def test_views_tick_only_while_mapped(display, kind):
    # A tick keeps the window's frame clock running, so a view on a hidden page
    # must drop it.
    from gi.repository import Gtk

    from quoth.pill_view import OrbView, PillView

    if kind == "pill":
        view = PillView(DEFAULT_THEME, height=48, animated=True, timer=True)
    else:
        view = OrbView(DEFAULT_THEME, size=32)
    stack = Gtk.Stack()
    stack.add_named(view, "view")
    stack.add_named(Gtk.Label(), "other")
    window = Gtk.Window(child=stack)
    assert view._tick_id is None
    window.present()
    try:
        assert view._tick_id is not None
        stack.set_visible_child_name("other")
        assert not view.get_mapped()
        assert view._tick_id is None
        stack.set_visible_child_name("view")
        assert view._tick_id is not None
    finally:
        window.destroy()


def test_theme_flag_disables_pill_and_orb_decoration(display):
    from gi.repository import Gtk

    from quoth.pill_view import OrbView, PillView

    static = dataclasses.replace(
        DEFAULT_THEME,
        animation=dataclasses.replace(DEFAULT_THEME.animation, enabled=False),
    )
    pill = PillView(static, height=48, animated=True)
    orb = OrbView(static, size=32)
    box = Gtk.Box()
    box.append(pill)
    box.append(orb)
    window = Gtk.Window(child=box)
    window.present()
    try:
        assert pill.get_mapped() and orb.get_mapped()
        assert pill._tick_id is None
        assert orb._tick_id is None
    finally:
        window.destroy()


def test_static_fixture_draws_even_when_cached_offscreen(display):
    from unittest.mock import patch

    from quoth.pill_view import OrbView
    from quoth.theme import read_theme

    view = OrbView(read_theme(FIXTURES / "motion-4-still-control.toml"), size=32)
    view._in_viewport = False
    cr = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 32, 32))
    with patch.object(view._orb, "draw") as draw:
        view._draw(None, cr, 32, 32)
    draw.assert_called_once()


def test_orb_view_skips_offscreen_physics_but_keeps_frame_time(display):
    from unittest.mock import MagicMock, patch

    from quoth.pill_view import OrbView

    view = OrbView(DEFAULT_THEME, size=32)
    view._last_frame_us = 0
    clock = MagicMock()
    clock.get_frame_time.side_effect = [40_000, 80_000]
    with (
        patch.object(view, "_intersects_viewports", side_effect=[False, True]),
        patch.object(view._orb, "advance") as advance,
    ):
        view._tick(None, clock)
        advance.assert_not_called()
        assert view._last_frame_us == 40_000
        view._tick(None, clock)
        advance.assert_called_once_with(pytest.approx(0.04))


def test_frame_dt_starts_at_zero_and_clamps_a_stall():
    from quoth.pill_view import frame_dt

    assert frame_dt(None, 5_000_000) == 0.0
    assert frame_dt(0, 16_000) == pytest.approx(0.016)
    assert frame_dt(0, 5_000_000) == pytest.approx(0.1)


def test_a_stalled_frame_clock_cannot_fast_forward_the_live_orb(display):
    from unittest.mock import MagicMock, patch

    from quoth.pill_view import OrbView

    view = OrbView(DEFAULT_THEME, size=32, live=True)
    view._last_frame_us = 0
    clock = MagicMock()
    clock.get_frame_time.return_value = 5_000_000  # a five-second stall
    director = view._director
    with (
        patch.object(view, "_intersects_viewports", return_value=True),
        patch.object(director, "update", wraps=director.update) as update,
    ):
        view._tick(None, clock)
    assert update.call_args.args[0] == pytest.approx(0.1)


def test_orb_view_treats_a_collapsed_expander_as_offscreen(display):
    from unittest.mock import MagicMock

    from quoth.pill_view import OrbView

    view = OrbView(DEFAULT_THEME, size=32)
    collapsed = MagicMock()
    collapsed.get_property.return_value = False
    view._expanders = (collapsed,)
    assert not view._intersects_viewports()


@pytest.mark.parametrize("fade", [0.25, 0.5])
def test_timer_keeps_its_dimmed_alpha_during_fades(fade):
    from quoth.pill_view import _draw_timer

    def ink(opacity):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 240, 48)
        _draw_timer(
            cairo.Context(surface), DEFAULT_THEME, "0:07", 240, 48, 1, opacity=opacity
        )
        surface.flush()
        return sum(surface.get_data()[3::4])

    full = ink(1.0)
    assert full > 0
    assert ink(fade) == pytest.approx(full * fade, rel=0.05)
