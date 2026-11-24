# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `scheduler`: the GLib scheduling seam and its error guard."""

from unittest.mock import patch

from gi.repository import GLib

import quoth.scheduler as scheduler
from quoth.scheduler import GLibScheduler, Scheduler, guarded


def test_glib_scheduler_conforms_to_scheduler_protocol():
    # Drift guard: a rename of any dispatch/timer method on GLibScheduler breaks this.
    assert issubclass(GLibScheduler, Scheduler)


def test_guarded_passes_through_return_value():
    assert guarded(lambda: "ok")() == "ok"


def test_guarded_forwards_args():
    captured = []
    guarded(lambda *a: captured.append(a))(1, 2, 3)
    assert captured == [(1, 2, 3)]


def test_guarded_logs_and_returns_false_on_exception(caplog, callback_raises):
    def boom():
        raise RuntimeError("kaboom")

    assert guarded(boom)() is False
    [record] = caplog.records
    assert record.getMessage() == "Callback boom raised"
    assert record.exc_info[0] is RuntimeError


def test_guarded_calls_on_error_then_returns_false(callback_raises):
    def boom():
        raise RuntimeError("kaboom")

    forgotten = []
    assert guarded(boom, on_error=lambda: forgotten.append(True))() is False
    assert forgotten == [True]


def test_guarded_skips_on_error_on_success():
    forgotten = []
    guarded(lambda: None, on_error=lambda: forgotten.append(True))()
    assert forgotten == []


def test_scheduler_methods_wrap_callbacks_with_guard(callback_raises):
    """`dispatch`, `after_ms` and `after_s` must hand GLib a `guarded` callable."""

    def fn(*_args):
        raise RuntimeError("kaboom")

    sched = scheduler.GLibScheduler()
    with patch.object(scheduler.GLib, "idle_add") as idle:
        sched.dispatch(fn, "a")
        assert idle.call_args.args[0]("a") is False
        assert idle.call_args.args[1:] == ("a",)
    with patch.object(scheduler.GLib, "timeout_add") as tadd:
        sched.after_ms(50, fn)
        assert tadd.call_args.args[0] == 50
        assert tadd.call_args.args[1]() is False
    with patch.object(scheduler.GLib, "timeout_add_seconds") as tsec:
        sched.after_s(2, fn)
        assert tsec.call_args.args[0] == 2
        assert tsec.call_args.args[1]() is False
    assert callback_raises == ["Callback fn raised"] * 3


def test_cancel_removes_the_source():
    sched = GLibScheduler()
    source_id = sched.after_ms(60_000, lambda: False)
    sched.cancel(source_id)
    assert GLib.MainContext.default().find_source_by_id(source_id) is None
