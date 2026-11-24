# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `animation`."""

import pytest
from gi.repository import Gtk

# Building a widget needs a display; the scenes' pure maths does not.
needs_display = pytest.mark.usefixtures("display")


@needs_display
def test_lottie_ink_follows_widget_color_and_only_updates_on_change(monkeypatch):
    from unittest.mock import Mock, call

    import cairo
    from gi.repository import Gdk

    from quoth import lottie
    from quoth.animation import AnimationView

    animation = Mock(
        spec_set=lottie.LottieAnimation, total_frames=10.0, width=100, height=100
    )
    animation.render.return_value = None
    monkeypatch.setattr(lottie, "load", lambda _: animation)
    view = AnimationView("intro", motion_enabled=False)
    foreground = Gdk.RGBA()
    foreground.parse("white")
    monkeypatch.setattr(view, "get_color", lambda: foreground)
    context = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 100, 100))

    view._draw(None, context, 100, 100)
    view._draw(None, context, 100, 100)
    foreground.parse("black")
    view._draw(None, context, 100, 100)

    assert view.has_css_class("quoth-animation")
    assert animation.apply_ink.call_args_list == [
        call((1.0, 1.0, 1.0, 1.0)),
        call((0.0, 0.0, 0.0, 1.0)),
    ]
    view._teardown()


@needs_display
def test_procedural_drawings_use_inherited_colour_and_alpha(monkeypatch):
    from unittest.mock import Mock

    from gi.repository import Gdk

    from quoth.animation import Ripple

    ripple = Ripple(20, motion_enabled=False)
    ripple._progress = 0.5
    foreground = Gdk.RGBA()
    foreground.parse("rgba(18,52,86,0.4)")
    monkeypatch.setattr(ripple, "get_color", lambda: foreground)
    context = Mock()

    ripple._draw(None, context, 100, 100)

    assert ripple.has_css_class("quoth-animation")
    assert context.set_source_rgba.call_args.args == pytest.approx(
        (foreground.red, foreground.green, foreground.blue, 0.4 * 0.5**1.5)
    )


@needs_display
def test_every_procedural_drawing_is_tagged_for_skin_repaints():
    from quoth.animation import Breakwater, DotField, Halo, Ripple

    drawings = (Ripple(20), Halo(14), Breakwater(120), DotField())
    assert all(drawing.has_css_class("quoth-animation") for drawing in drawings)


@needs_display
def test_disabling_one_shot_settles_and_completes_once(monkeypatch):
    from unittest.mock import Mock

    from quoth import animation, lottie

    clip = Mock(spec_set=lottie.LottieAnimation, total_frames=20.0, duration_s=1.0)
    monkeypatch.setattr(lottie, "load", lambda _path: clip)
    monkeypatch.setattr(animation.GLib, "idle_add", lambda callback: callback())
    view = animation.AnimationView("intro")
    ticks = []
    view.add_tick_callback = lambda callback: ticks.append(callback) or 7
    view.remove_tick_callback = lambda _tick: ticks.clear()
    completed = []

    view.play(lambda: completed.append(True))
    view.set_motion_enabled(False)
    view.set_motion_enabled(False)

    assert view._frame == 20.0
    assert ticks == []
    assert completed == [True]


@needs_display
def test_without_thorvg_a_loop_holds_still_when_motion_returns(monkeypatch):
    from quoth import animation, lottie

    monkeypatch.setattr(lottie, "load", lambda _path: None)
    view = animation.AnimationView("loader", loop=True, motion_enabled=False)
    ticks = []
    view.add_tick_callback = lambda callback: ticks.append(callback) or 7

    view.start()
    view.set_motion_enabled(True)

    assert ticks == []


@needs_display
def test_one_shot_completion_exception_is_logged_and_removed(
    monkeypatch, caplog, callback_raises
):
    from unittest.mock import Mock

    from quoth import animation, lottie

    monkeypatch.setattr(
        lottie,
        "load",
        lambda _path: Mock(
            spec_set=lottie.LottieAnimation, total_frames=20.0, duration_s=1.0
        ),
    )
    monkeypatch.setattr(animation.GLib, "idle_add", lambda callback: callback())
    caplog.set_level("ERROR")
    view = animation.AnimationView("intro", motion_enabled=False)
    completed = []

    def fail():
        completed.append(True)
        raise RuntimeError("completion failed")

    view.play(fail)

    assert completed == [True]
    assert view._on_done is None
    assert [str(record.exc_info[1]) for record in caplog.records] == [
        "completion failed"
    ]


@needs_display
def test_requested_loop_resumes_but_stopped_loop_does_not(monkeypatch):
    from unittest.mock import Mock

    from quoth import lottie
    from quoth.animation import AnimationView

    monkeypatch.setattr(
        lottie,
        "load",
        lambda _path: Mock(
            spec_set=lottie.LottieAnimation, total_frames=20.0, duration_s=1.0
        ),
    )
    view = AnimationView("loader", loop=True, still_fraction=0.25)
    ticks = []
    view.add_tick_callback = lambda callback: ticks.append(callback) or len(ticks) + 1
    view.remove_tick_callback = lambda _tick: ticks.clear()

    view.start()
    view.set_motion_enabled(False)
    assert ticks == [] and view._frame == 5.0
    view.set_motion_enabled(True)
    assert len(ticks) == 1
    view.stop()
    view.set_motion_enabled(False)
    view.set_motion_enabled(True)
    assert ticks == []


@needs_display
def test_procedural_views_settle_and_remove_ticks_when_motion_stops():
    from quoth.animation import _FIELD_STILL_T, Breakwater, DotField, Halo

    views = (Halo(14), Breakwater(120), DotField())
    halo, breakwater, field = views
    # Mid-motion: a stopped ring still out, a wave on its way, a flicker draining.
    halo._progress = 0.5
    breakwater._t = 1.0
    field.set_busy()
    field.step(1.0)
    field.set_idle()
    for view in views:
        view._tick_id = 9
        removed = []
        view.remove_tick_callback = removed.append
        view.set_motion_enabled(False)
        assert removed == [9]
        assert view._tick_id is None

    assert halo._progress is None
    assert breakwater._t is None
    assert field._t == _FIELD_STILL_T and field._busy_level == 0.0


@needs_display
class TestDotField:
    """The site's dot field: a wave per device, a still frame under reduced motion."""

    def _field(self, motion=False):
        from quoth.animation import DotField

        return DotField(motion_enabled=motion)

    def _alpha_sum(self, field):
        import cairo

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 200, 80)
        field._draw(None, cairo.Context(surface), 200, 80)
        surface.flush()
        return sum(surface.get_data()[3::4])

    def test_draws_something_at_rest(self):
        assert self._alpha_sum(self._field()) > 0

    def test_starts_idle_and_settles_on_a_device(self):
        field = self._field()
        assert field.state == "idle"
        field.set_busy()
        assert field.state == "busy"
        field.set_hardware("NPU")
        assert field.state == "settled" and field._hardware == 2
        field.set_hardware("TPU")  # unknown: keeps the last
        assert field._hardware == 2

    def test_states_look_different(self):
        idle, busy, npu = self._field(), self._field(), self._field()
        busy.set_busy()
        npu.set_hardware("NPU")
        sums = {self._alpha_sum(f) for f in (idle, busy, npu)}
        assert len(sums) == 3

    def test_idle_is_sparse_and_soft(self):
        idle, cpu = self._field(), self._field()
        cpu.set_hardware("CPU")
        assert self._alpha_sum(idle) < self._alpha_sum(cpu)

    def test_with_motion_the_flicker_builds_and_the_wave_launches(self):
        field = self._field(motion=True)
        field.set_busy()
        assert field._busy_level == 0.0
        for _ in range(30):
            field.step(1 / 30)
        assert 0.5 < field._busy_level <= 1.0
        field.set_hardware("NPU")
        launched = self._alpha_sum(field)  # the wave has not spread yet
        for _ in range(60):
            field.step(1 / 30)
        assert field._busy_level < 0.05  # drained
        assert self._alpha_sum(field) > launched  # the wave has arrived

    def test_keeps_the_site_proportion_within_bounds(self):
        field = self._field()
        assert field.do_measure(Gtk.Orientation.VERTICAL, 300)[0] == 100
        assert field.do_measure(Gtk.Orientation.VERTICAL, 900)[0] == 240  # capped
        assert field.do_measure(Gtk.Orientation.VERTICAL, 100)[0] == 90  # floor


class TestBreakwater:
    """The site's privacy picture: waves travel out, break on the ring, never cross."""

    @needs_display
    def test_draws_a_still_frame_without_motion(self):
        import cairo

        from quoth.animation import Breakwater

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 120, 120)
        Breakwater(120, motion_enabled=False)._draw(
            None, cairo.Context(surface), 120, 120
        )
        surface.flush()
        assert sum(surface.get_data()[3::4]) > 0

    def test_a_wave_travels_then_breaks_inside_the_ring(self):
        from quoth.animation import _BREAKWATER_CONTACT, _breakwater_frame

        fronts, breaking, _impact, _push = _breakwater_frame(1.0)
        assert fronts and all(radius < _BREAKWATER_CONTACT for radius, _a in fronts)
        _fronts, breaking, impact, push = _breakwater_frame(2.3)
        assert breaking and impact > 0 and push > 0
        points = [
            point for _a, crests in breaking for crest in crests for point in crest
        ]
        assert all(x * x + y * y <= _BREAKWATER_CONTACT**2 + 1e-6 for x, y in points)


class _Clock:
    def __init__(self):
        self.us = 0

    def get_frame_time(self):
        return self.us


@needs_display
class TestHalo:
    """A listening pulse: the last ring finishes, and a quick restart carries it on."""

    def _halo(self):
        from quoth.animation import Halo

        halo = Halo(14)
        ticks = []
        halo.add_tick_callback = lambda callback: ticks.append(callback) or 1
        halo.remove_tick_callback = lambda _id: ticks.clear()
        clock = _Clock()

        def at(ms: int) -> float | None:
            clock.us = ms * 1000
            if ticks and not ticks[0](halo, clock):
                ticks.clear()
                halo._tick_id = None
            return halo._progress

        return halo, at

    def test_a_stopped_ring_finishes_and_no_other_follows(self):
        halo, at = self._halo()
        halo.start()
        at(0)
        assert at(400) > 0
        halo.stop()
        assert at(800) > at(400)  # the ring out keeps travelling
        assert at(1300) is None  # then it is gone
        assert at(2000) is None  # and none follows

    def test_a_quick_restart_does_not_reset_the_ring(self):
        halo, at = self._halo()
        halo.start()
        at(0)
        travelled = at(500)
        halo.stop()
        halo.start()
        assert at(520) > travelled
