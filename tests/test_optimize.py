# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `optimize`, the onboarding benchmark step; they need a display."""

from unittest.mock import Mock

from quoth.config import Config
from tests.support import FakeBenchmark, accessible_mismatch, shown_texts


def _optimize_page(*, devices=("CPU",), start=None, cancel=None, events=None):
    from quoth.optimize import OptimizePage

    config = Config()
    config.model = "small"
    events = [] if events is None else events
    return OptimizePage(
        config,
        list(devices),
        start=start or FakeBenchmark(),
        cancel=cancel or (lambda: None),
        on_advance=lambda: events.append("advance"),
        on_lock=lambda locked: events.append(f"lock {locked}"),
        motion_enabled=False,
    )


def test_optimize_offer_describes_the_selected_model_and_available_devices(display):
    page = _optimize_page(devices=["CPU", "NPU"])

    assert (
        "Quoth can benchmark Small on your CPU and NPU so Automatic can use the "
        "fastest measured device. First-time NPU preparation can take longer. "
        "You can skip and benchmark later in Settings."
    ) in shown_texts(page.content)


def test_optimize_progress_moves_on_to_the_next_device(display):
    from quoth import calibration

    benchmark = FakeBenchmark()
    page = _optimize_page(devices=["CPU", "NPU"], start=benchmark)
    page._start()
    assert page._current.get_label() == "Measuring Small on CPU…"

    benchmark.on_progress(1, 2, "small", "CPU", calibration.CellResult(True, 120.0))

    assert page._current.get_label() == "Measuring Small on NPU…"
    # Past the finished device, but never claiming the run is done.
    assert 0.5 < page._bar.get_fraction() < 1.0


def test_the_progress_bar_describes_what_is_being_measured(display):
    # Focusing the bar must say what the visible stage line says.
    from gi.repository import Gtk

    from quoth import calibration

    benchmark = FakeBenchmark()
    page = _optimize_page(devices=["CPU", "NPU"], start=benchmark)
    description = Gtk.AccessibleProperty.DESCRIPTION
    page._start()
    assert (
        accessible_mismatch(page._bar, description, "Measuring Small on CPU…") is None
    )

    benchmark.on_progress(1, 2, "small", "CPU", calibration.CellResult(True, 120.0))

    assert (
        accessible_mismatch(page._bar, description, "Measuring Small on NPU…") is None
    )


def test_a_skip_stops_the_run_and_unlocks_when_it_lands(display):
    benchmark = FakeBenchmark()
    cancelled = []
    events = []
    page = _optimize_page(
        devices=["CPU", "NPU"],
        start=benchmark,
        cancel=lambda: cancelled.append(True),
        events=events,
    )
    page._start()
    reached = page._bar.get_fraction()
    stop_progress = Mock(wraps=page._bar.glide_to)
    page._bar.glide_to = stop_progress
    page._skip()  # the measurement in flight cannot be interrupted
    assert cancelled == [True]
    assert events == ["lock True", "advance"]
    assert page._field.state == "idle"
    stop_progress.assert_called_once_with(reached)

    benchmark.finish(None)  # the run lands
    assert events == ["lock True", "advance", "lock False"]


def test_a_busy_engine_leaves_the_step_passable(display, caplog):
    events = []
    page = _optimize_page(start=FakeBenchmark(accept=False), events=events)

    page._start()

    assert events == []
    assert page._stack.get_visible_child_name() == "result"
    assert page._result_title.get_label() == "Quoth is busy"
    assert page._result_description.get_label() == (
        "You can continue and benchmark later in Settings."
    )
    assert "Onboarding benchmark refused: the engine is busy" in caplog.messages


def test_the_optimize_result_is_read_out(display):
    # Showing it moves the focus to Continue, past the result text.
    from gi.repository import Gtk

    page = _optimize_page(start=FakeBenchmark(accept=False))
    page._result_title.announce = Mock()
    page._start()
    page._result_title.announce.assert_called_once_with(
        "Quoth is busy. You can continue and benchmark later in Settings.",
        Gtk.AccessibleAnnouncementPriority.MEDIUM,
    )


def test_optimize_failure_shows_result_and_waits_for_continue(display):
    benchmark = FakeBenchmark()
    events = []
    page = _optimize_page(start=benchmark, events=events)

    page._start()
    reached = page._bar.get_fraction()
    settle_progress = Mock(wraps=page._bar.glide_to)
    page._bar.glide_to = settle_progress
    benchmark.finish(None)

    assert events == ["lock True", "lock False"]
    assert page._stack.get_visible_child_name() == "result"
    assert page._result_title.get_label() == "Benchmark failed"
    assert page._result_description.get_label() == (
        "You can continue and benchmark later in Settings."
    )
    assert page._field.state == "idle"
    settle_progress.assert_called_once_with(reached)


def test_optimize_failed_cells_are_distinct_from_a_whole_run_failure(display):
    from quoth import calibration

    benchmark = FakeBenchmark()
    page = _optimize_page(start=benchmark)

    page._start()
    settle_progress = Mock(wraps=page._bar.glide_to)
    page._bar.glide_to = settle_progress
    benchmark.finish(
        calibration.Report(
            "fingerprint", {"small|CPU": calibration.CellResult(False, None)}
        )
    )

    assert page._result_title.get_label() == (
        "Quoth couldn't measure your hardware and will use a safe default."
    )
    assert page._result_description.get_label() == (
        "You can re-run this any time in Settings."
    )
    assert page._field.state == "idle"
    settle_progress.assert_called_once_with(1.0)
