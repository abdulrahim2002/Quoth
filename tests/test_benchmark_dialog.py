# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `benchmark_dialog`.

Every test builds the real dialog, so each needs a display.
"""

from unittest.mock import Mock

import pytest
from gi.repository import Adw, Gdk, Gtk

from quoth import brand
from quoth.calibration import CellResult, Report, speed_label
from tests.support import descendants, join_thread, settle, wait_until

_INTRO = (
    "Quoth benchmarks your downloaded models on available hardware. Automatic mode uses the "
    "fastest measured device for each model. Larger models can take longer."
)
_NPU_NOTE = " First-time NPU preparation can take longer."
_BUSY = "Quoth is busy. Wait for the current operation to finish, then try again."
_RESULT_HEADING = "Benchmark results"


def _dialog(
    *,
    models=None,
    devices=None,
    start=None,
    cancel=None,
    motion_enabled=True,
):
    from quoth.benchmark_dialog import BenchmarkDialog

    return BenchmarkDialog(
        models or ["base"],
        devices or ["CPU"],
        start=start or (lambda *_args: False),
        cancel=cancel or (lambda: None),
        motion_enabled=motion_enabled,
    )


@pytest.fixture
def standard_text_dpi(display):
    settings = Gtk.Settings.get_default()
    previous_dpi = settings.get_property("gtk-xft-dpi")
    settings.set_property("gtk-xft-dpi", 96 * 1024)
    settle()
    try:
        yield
    finally:
        settings.set_property("gtk-xft-dpi", previous_dpi)
        settle()


def _assert_action_reachable(dialog, action_row):
    adjustment = dialog._scroller.get_vadjustment()
    assert adjustment.get_page_size() > 0
    assert adjustment.get_upper() >= adjustment.get_page_size()

    def at_bottom():
        target = adjustment.get_upper() - adjustment.get_page_size()
        adjustment.set_value(target)
        return (
            adjustment.get_value() == pytest.approx(target)
            and action_row.compute_bounds(dialog._scroller)[1].get_y()
            < dialog._scroller.get_height()
        )

    wait_until(at_bottom)
    success, bounds = action_row.compute_bounds(dialog._scroller)
    assert success
    assert bounds.get_height() > 0
    assert bounds.get_y() < dialog._scroller.get_height()
    assert bounds.get_y() + bounds.get_height() > 0


def _assert_horizontally_contained(widget, dialog):
    success, bounds = widget.compute_bounds(dialog)
    assert success
    assert bounds.get_width() > 0
    assert bounds.get_x() >= 0
    assert bounds.get_x() + bounds.get_width() <= dialog.get_width()


def test_disabled_at_construction_has_no_stack_transition(display):
    dialog = _dialog(motion_enabled=False)

    assert dialog._stack.get_transition_duration() == 0
    assert not dialog._field._motion


def test_live_change_updates_stack_and_field(display):
    dialog = _dialog()
    duration = dialog._stack.get_transition_duration()

    dialog.set_motion_enabled(False)
    assert dialog._stack.get_transition_duration() == 0
    assert not dialog._field._motion

    dialog.set_motion_enabled(True)
    assert dialog._stack.get_transition_duration() == duration
    assert dialog._field._motion


def test_stages_share_one_clamped_scroll_layout(display):
    dialog = _dialog(models=["base", "small"], devices=["CPU", "GPU"])
    widgets = [dialog.get_child(), *descendants(dialog.get_child())]
    clamps = [widget for widget in widgets if isinstance(widget, Adw.Clamp)]
    scrollers = [widget for widget in widgets if isinstance(widget, Gtk.ScrolledWindow)]

    assert len(clamps) == 1
    assert clamps[0].get_maximum_size() == brand.STATUS_WIDTH
    assert scrollers == [dialog._scroller]
    assert dialog._scroller.get_policy() == (
        Gtk.PolicyType.NEVER,
        Gtk.PolicyType.AUTOMATIC,
    )
    assert not dialog._stack.get_hhomogeneous()
    assert not dialog._stack.get_vhomogeneous()
    assert dialog.get_follows_content_size()


@pytest.mark.parametrize(
    ("parent_width", "font_size", "narrow"),
    [(900, None, False), (420, None, True), (510, "24pt", True)],
    ids=("wide", "narrow", "narrow-large-font"),
)
def test_every_stage_stays_contained_and_reachable(
    display, standard_text_dpi, parent_width, font_size, narrow
):
    provider = Gtk.CssProvider()
    screen = Gdk.Display.get_default()
    window = Adw.Window()
    window.set_default_size(parent_width, 600)
    dialog = _dialog(
        models=["base", "small"],
        devices=["NPU", "GPU", "CPU"],
        start=lambda *_args: True,
        motion_enabled=False,
    )
    if font_size:
        provider.load_from_string(f".benchmark-test * {{ font-size: {font_size}; }}")
        Gtk.StyleContext.add_provider_for_display(
            screen, provider, Gtk.STYLE_PROVIDER_PRIORITY_USER + 20
        )
        dialog.add_css_class("benchmark-test")

    try:
        window.present()
        dialog.present(window)
        expected_orientation = (
            Gtk.Orientation.VERTICAL if narrow else Gtk.Orientation.HORIZONTAL
        )
        wait_until(
            lambda: (
                dialog.get_mapped()
                and dialog.get_width() > 0
                and dialog.get_child().get_width() <= dialog.get_width()
                and (
                    dialog._content.get_width() < brand.STATUS_WIDTH
                    if narrow
                    else dialog._content.get_width() == brand.STATUS_WIDTH
                )
                and all(
                    row.get_orientation() == expected_orientation
                    for row in dialog._action_rows
                )
            )
        )
        toolbar = dialog.get_child()
        clamp = dialog._content.get_parent()
        assert isinstance(toolbar, Adw.ToolbarView)
        assert isinstance(clamp, Adw.Clamp)
        assert dialog._content.get_width() == (
            brand.STATUS_WIDTH if not narrow else clamp.get_width()
        )

        report = Report(
            "fp",
            {
                "base|NPU": CellResult(True, 22.0),
                "base|GPU": CellResult(True, 18.0),
                "base|CPU": CellResult(True, 40.0),
                "small|NPU": CellResult(True, 32.0),
                "small|GPU": CellResult(True, 25.0),
                "small|CPU": CellResult(True, 55.0),
            },
        )
        stages = (
            ("prepare", lambda: dialog._stack.set_visible_child_name("prepare")),
            ("measure", dialog._on_run),
            ("result", lambda: dialog.finish(report)),
        )
        for index, (stage, show_stage) in enumerate(stages):
            show_stage()
            wait_until(
                lambda stage=stage: (
                    dialog._stack.get_visible_child_name() == stage
                    and dialog._stack.get_visible_child().get_mapped()
                    and dialog._stack.get_visible_child().get_width() > 0
                )
            )
            visible = dialog._stack.get_visible_child()
            assert visible is dialog._stack.get_child_by_name(stage)
            assert visible.get_width() == dialog._stack.get_width()
            assert visible.get_height() == dialog._stack.get_height()
            for widget in (toolbar, dialog._scroller, clamp, dialog._content):
                _assert_horizontally_contained(widget, dialog)
            _assert_horizontally_contained(visible, dialog)
            _assert_horizontally_contained(dialog._action_rows[index], dialog)
            if stage == "measure":
                dialog.advance(1, 6, "base", "NPU", CellResult(True, None))
                dialog.advance(2, 6, "base", "GPU", CellResult(True, 10.0))
                settle()  # lay out the new labels
                for row_index, status in enumerate(dialog._cell_status.values()):
                    row = dialog._cell_list.get_row_at_index(row_index)
                    assert row is not None
                    assert status.get_wrap()
                    _assert_horizontally_contained(row, dialog)
            elif stage == "result":
                for row_index in range(2):
                    row = dialog._result_list.get_row_at_index(row_index)
                    assert row is not None
                    _assert_horizontally_contained(row, dialog)
                speeds = [
                    widget
                    for widget in descendants(dialog._result_list)
                    if isinstance(widget, Gtk.Label)
                    and widget.get_label() in {speed_label(18.0), speed_label(25.0)}
                ]
                assert len(speeds) == 2
                for speed in speeds:
                    assert speed.get_wrap()
            # Every line of text the stage shows stays inside the dialog.
            for label in descendants(visible):
                if isinstance(label, Gtk.Label) and label.get_label():
                    _assert_horizontally_contained(label, dialog)
            _assert_action_reachable(dialog, dialog._action_rows[index])
            dialog._scroller.get_vadjustment().set_value(0)
    finally:
        dialog.force_close()
        window.destroy()
        if font_size:
            Gtk.StyleContext.remove_provider_for_display(screen, provider)


def test_dialog_tracks_the_visible_stages_natural_height(display, standard_text_dpi):
    window = Adw.Window()
    window.set_default_size(900, 800)
    dialog = _dialog(
        models=["base", "small"],
        devices=["NPU", "GPU", "CPU"],
        start=lambda *_args: True,
        motion_enabled=False,
    )
    report = Report(
        "fp",
        {
            "base|GPU": CellResult(True, 20.0),
            "small|GPU": CellResult(True, 30.0),
        },
    )

    try:
        window.present()
        dialog.present(window)
        wait_until(lambda: dialog.get_child().get_height() > 0)
        prepare_height = dialog.get_child().get_height()

        dialog._on_run()
        wait_until(lambda: dialog.get_child().get_height() > prepare_height)
        measure_height = dialog.get_child().get_height()

        dialog.finish(report)
        wait_until(lambda: dialog.get_child().get_height() < measure_height)
        result_height = dialog.get_child().get_height()

        assert measure_height > prepare_height
        assert result_height < measure_height
    finally:
        dialog.force_close()
        window.destroy()


@pytest.mark.parametrize(
    ("devices", "intro"),
    [(["CPU", "GPU"], _INTRO), (["CPU", "NPU"], _INTRO + _NPU_NOTE)],
)
def test_intro_and_busy_copy_are_accurate_and_non_blocking(display, devices, intro):
    dialog = _dialog(devices=devices)
    labels = [
        widget.get_label()
        for widget in descendants(dialog.get_child())
        if isinstance(widget, Gtk.Label)
    ]

    assert intro in labels
    assert "You can run the benchmark anyway." in dialog._note.get_label()
    dialog._on_run()
    assert dialog._note.get_label() == _BUSY
    assert dialog._stack.get_visible_child_name() == "prepare"


def test_a_cell_skipped_for_missing_files_says_so(display):
    dialog = _dialog(models=["base"], devices=["NPU", "CPU"], start=lambda *_a: True)
    dialog._on_run()
    dialog.advance(1, 2, "base", "NPU", CellResult(False, skipped=True))
    assert dialog._cell_status[("base", "NPU")].get_label() == "Not downloaded"


def test_completed_untimed_cell_differs_from_unattempted_cell(display):
    cancelled = Mock()
    dialog = _dialog(
        models=["base"],
        devices=["NPU", "GPU", "CPU"],
        start=lambda *_args: True,
        cancel=cancelled,
    )

    dialog._on_run()
    assert dialog._cell_status[("base", "NPU")].get_label() == "Measuring…"
    dialog.advance(1, 3, "base", "NPU", CellResult(True, None))
    assert dialog._cell_status[("base", "GPU")].get_label() == "Measuring…"
    dialog.advance(2, 3, "base", "GPU", CellResult(True, 10.0))
    assert dialog._cell_status[("base", "CPU")].get_label() == "Measuring…"
    dialog._on_cancel()
    dialog.finish(
        Report(
            "fp",
            {
                "base|NPU": CellResult(True, None),
                "base|GPU": CellResult(True, 10.0),
            },
        )
    )

    cancelled.assert_called_once_with()
    assert dialog._cell_status[("base", "NPU")].get_label() == "No result"
    assert dialog._cell_status[("base", "GPU")].get_label() == speed_label(10.0)
    assert dialog._cell_status[("base", "CPU")].get_label() == "Not measured"
    assert dialog._verdict.get_label().startswith("Based on 2 of 3 measurements. ")
    assert _RESULT_HEADING in [
        widget.get_label()
        for widget in descendants(dialog._stack.get_child_by_name("result"))
        if isinstance(widget, Gtk.Label)
    ]
    assert dialog._stack.get_visible_child_name() == "result"


@pytest.mark.parametrize("width", [640, 360])
def test_short_cell_statuses_stay_on_one_line(display, width):
    # A wrapping label asks only for its narrowest width unless told otherwise, so
    # a status that grows after layout broke onto a second line in Adwaita's rows.
    Adw.init()
    window = Adw.Window()
    window.set_default_size(width, 600)
    dialog = _dialog(
        models=["tiny", "base", "small", "large-v3-turbo"],
        devices=["NPU", "GPU", "CPU"],
        start=lambda *_args: True,
        motion_enabled=False,
    )
    statuses = [dialog._cell_status[("tiny", d)] for d in ("NPU", "GPU", "CPU")]
    try:
        window.present()
        dialog.present(window)
        wait_until(lambda: dialog.get_mapped() and dialog.get_width() > 0)
        Gtk.test_widget_wait_for_draw(window)
        dialog._on_run()
        wait_until(lambda: statuses[0].get_width() > 0)
        Gtk.test_widget_wait_for_draw(window)
        dialog.advance(1, 12, "tiny", "NPU", CellResult(False, None, skipped=True))
        dialog.advance(2, 12, "tiny", "GPU", CellResult(True, 357.0))

        assert [s.get_label() for s in statuses] == [
            "Not downloaded",
            "14× realtime",
            "Measuring…",
        ]
        wait_until(
            lambda: [s.get_layout().get_line_count() for s in statuses] == [1, 1, 1]
        )
    finally:
        dialog.force_close()
        window.destroy()


def test_cancel_says_the_run_stops_after_the_cell_in_flight(display, caplog):
    dialog = _dialog(devices=["NPU", "GPU", "CPU"], start=lambda *_args: True)
    dialog._on_run()
    dialog._on_cancel()
    # A hang after Cancel must be told apart from one before it.
    assert "Benchmark cancel requested after 0/3 cells" in caplog.messages

    assert dialog._current.get_label() == "Stopping after this measurement…"
    assert not dialog._stop.get_sensitive()
    dialog.advance(1, 3, "base", "NPU", CellResult(True, 10.0))
    assert dialog._current.get_label() == "Stopping after this measurement…"
    assert dialog._cell_status[("base", "GPU")].get_label() == "Waiting"
    dialog.finish(Report("fp", {"base|NPU": CellResult(True, 10.0)}))
    dialog._on_run()
    assert dialog._stop.get_sensitive()


def test_a_refused_rerun_restores_the_dialog_title(display):
    runs = iter([True, False])
    dialog = _dialog(start=lambda *_args: next(runs))
    dialog._on_run()
    dialog.finish(Report("fp", {"base|CPU": CellResult(True, 10.0)}))
    assert dialog.get_title() == "Benchmark complete"

    dialog._on_run()

    assert dialog.get_title() == "Hardware benchmark"
    assert dialog._stack.get_visible_child_name() == "prepare"


def test_zero_winner_result_uses_neutral_heading(display):
    dialog = _dialog()

    dialog.finish(Report("fp", {"base|CPU": CellResult(False, None)}))

    labels = [
        widget.get_label()
        for widget in descendants(dialog._stack.get_child_by_name("result"))
        if isinstance(widget, Gtk.Label)
    ]
    assert _RESULT_HEADING in labels
    assert "You're all set" not in labels
    assert dialog._verdict.get_label() == (
        "The benchmark didn't finish. Quoth will use a safe default."
    )
    assert dialog._field.state == "idle"
    assert dialog._result_list.get_first_child() is None


def test_cancelled_zero_winner_result_is_not_reported_as_a_worker_failure(display):
    dialog = _dialog(start=lambda *_args: True)

    dialog._on_run()
    dialog._on_cancel()
    dialog.finish(Report("fp", {}))

    assert dialog._verdict.get_label() == (
        "The benchmark didn't finish. Quoth will use a safe default."
    )
    assert dialog.get_title() == "Benchmark complete"
    assert dialog._field.state == "idle"


def test_result_rows_name_each_models_fastest_device_and_speed(display):
    dialog = _dialog(models=["base", "small"], devices=["NPU", "GPU", "CPU"])

    dialog.finish(
        Report(
            "fp",
            {
                "base|NPU": CellResult(True, 20.0),
                "base|GPU": CellResult(True, 25.0),
                "small|NPU": CellResult(True, 50.0),
                "small|GPU": CellResult(True, 40.0),
            },
        )
    )

    rows = [
        (row.get_title(), row.get_subtitle())
        for row in descendants(dialog._result_list)
        if isinstance(row, Adw.ActionRow)
    ]
    speeds = [
        widget.get_label()
        for widget in descendants(dialog._result_list)
        if isinstance(widget, Gtk.Label) and widget.get_label().endswith("realtime")
    ]
    assert rows == [("Base", "Fastest on your NPU"), ("Small", "Fastest on your GPU")]
    assert speeds == ["250× realtime", "125× realtime"]


def test_whole_run_failure_has_a_terminal_result_and_close_action(display):
    dialog = _dialog()

    dialog.finish(None)

    labels = [
        widget.get_label()
        for widget in descendants(dialog._stack.get_child_by_name("result"))
        if isinstance(widget, Gtk.Label)
    ]
    assert "Benchmark failed" in labels
    assert dialog._verdict.get_label() == (
        "The benchmark couldn't finish. You can try again or close this window."
    )
    assert dialog._stack.get_visible_child_name() == "result"
    assert dialog.get_title() == "Benchmark failed"
    assert dialog._field.state == "idle"
    assert dialog._done.get_label() == "Close"
    assert dialog._result_list.get_first_child() is None


def test_controller_worker_failure_reaches_dialog_through_glib_and_app_bridge(
    display, tmp_path
):
    from dataclasses import fields

    from quoth.app import QuothApp
    from quoth.config import Config
    from quoth.recording import RecordingCallbacks, RecordingController
    from quoth.scheduler import GLibScheduler
    from tests.support import FakeAudio, FakeModelManager, FakeWhisper

    app = type(
        "AppBridge",
        (),
        {"_apply_effective_device": Mock(), "withdraw_notification": Mock()},
    )()
    callbacks = RecordingCallbacks(
        **{f.name: Mock() for f in fields(RecordingCallbacks)}
    )
    controller = RecordingController(
        Config(),
        FakeModelManager(),
        "CPU",
        GLibScheduler(),
        callbacks,
        audio_factory=lambda **_: FakeAudio(),
        whisper_factory=FakeWhisper,
    )

    def broken_measure(*_args):
        raise RuntimeError("the engine failed mid-run")

    def start(models, devices, on_progress, on_done):
        return controller.calibrate(
            models,
            devices,
            "fp",
            on_progress=on_progress,
            on_done=lambda report: QuothApp._benchmark_done(app, report, on_done),
            measure_fn=broken_measure,
            path=tmp_path / "calibration.json",
        )

    dialog = _dialog(start=start, motion_enabled=False)
    dialog._on_run()

    wait_until(lambda: dialog._stack.get_visible_child_name() == "result")
    join_thread(controller, "_calibrate_thread")
    assert controller.is_idle
    assert dialog.get_title() == "Benchmark failed"
    assert dialog._done.get_label() == "Close"
    app._apply_effective_device.assert_called_once_with()


def test_progress_exposes_name_and_current_measurement(display, monkeypatch):
    properties = []
    update_property = Gtk.ProgressBar.update_property

    def record_property(bar, names, values):
        properties.append((bar, list(names), list(values)))
        return update_property(bar, names, values)

    monkeypatch.setattr(Gtk.ProgressBar, "update_property", record_property)
    dialog = _dialog(
        models=["base"],
        devices=["NPU", "CPU"],
        start=lambda *_args: True,
    )
    dialog._on_run()
    first = dialog._current.get_label()
    dialog.advance(1, 2, "base", "NPU", CellResult(True, 10.0))
    second = dialog._current.get_label()

    assert any(
        bar is dialog._bar
        and names == [Gtk.AccessibleProperty.LABEL]
        and values == ["Benchmark progress"]
        for bar, names, values in properties
    )
    descriptions = [
        values[0]
        for bar, names, values in properties
        if bar is dialog._bar and names == [Gtk.AccessibleProperty.DESCRIPTION]
    ]
    assert descriptions == [first, second]
    assert Gtk.test_accessible_has_property(dialog._bar, Gtk.AccessibleProperty.LABEL)
    assert Gtk.test_accessible_has_property(
        dialog._bar, Gtk.AccessibleProperty.DESCRIPTION
    )


def test_buttons_receive_pointer_cursors_and_result_rebuild_reapplies_policy(
    display, monkeypatch
):
    dialog = _dialog()
    buttons = [
        widget
        for widget in descendants(dialog.get_child())
        if isinstance(widget, Gtk.Button)
    ]
    assert buttons
    assert all(
        button.get_cursor().get_name() == "pointer"
        for button in buttons
        if button.get_sensitive()
    )

    calls = []
    monkeypatch.setattr(brand, "pointer_cursors", calls.append)
    dialog.finish(Report("fp", {"base|CPU": CellResult(True, 10.0)}))
    assert calls == [dialog._result_list]


def test_close_marks_closed_before_cancel_and_ignores_late_callbacks(display):
    dialog = None
    closed_during_cancel = []

    def cancel():
        closed_during_cancel.append(dialog._closed)

    dialog = _dialog(start=lambda *_args: True, cancel=cancel)
    dialog._stack.set_visible_child_name("measure")
    dialog._on_closed()
    dialog.advance(1, 1, "base", "CPU", CellResult(True, 10.0))
    dialog.finish(Report("fp", {"base|CPU": CellResult(True, 10.0)}))
    dialog.finish(None)

    assert closed_during_cancel == [True]
    assert dialog._done_count == 0
    assert dialog.get_title() == "Hardware benchmark"


def test_finish_announces_the_verdict_once(display):
    dialog = _dialog()
    dialog._verdict.announce = Mock()
    report = Report("fp", {"base|CPU": CellResult(True, 10.0)})

    dialog.finish(report)
    dialog.finish(report)

    dialog._verdict.announce.assert_called_once_with(
        dialog._verdict.get_label(),
        Gtk.AccessibleAnnouncementPriority.MEDIUM,
    )
