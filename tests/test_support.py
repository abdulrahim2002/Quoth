# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `tests.support`, so the shared fakes cannot drift from the real ones."""

import inspect
import threading
import types

import pytest

from quoth.audio import AudioBackend, AudioPipeline
from quoth.models import ModelManager
from quoth.portal_shortcuts import PortalShortcuts
from quoth.scheduler import Scheduler
from quoth.whisper import TranscriptionBackend, WhisperContext
from tests.support import (
    DeferredScheduler,
    FakeAudio,
    FakeModelManager,
    FakePortalShortcuts,
    FakeScheduler,
    FakeWhisper,
    join_thread,
    wait_until,
)


class TestFakeScheduler:
    def test_satisfies_the_scheduler_protocol(self):
        """Mirror `test_scheduler`'s guard on `GLibScheduler` for the fake.

        A new `Scheduler` method must fail here too, not silently leave the fake behind.
        """
        assert isinstance(FakeScheduler(), Scheduler)

    def test_dispatch_runs_inline(self):
        seen = []
        FakeScheduler().dispatch(seen.append, 1)
        assert seen == [1]

    def test_timers_are_recorded_not_run(self):
        s = FakeScheduler()
        seen = []
        s.after_ms(50, seen.append, "a")
        s.after_s(1, seen.append, "b")
        assert seen == []
        s.fire_all()
        assert seen == ["a", "b"]

    def test_cancel_removes_only_its_own_timer(self):
        s = FakeScheduler()
        seen = []
        keep = s.after_ms(10, seen.append, "keep")
        drop = s.after_ms(10, seen.append, "drop")
        s.cancel(drop)
        s.fire_all()
        assert seen == ["keep"] and keep != drop

    def test_a_dispatch_that_returns_true_fails(self):
        # GLib would run it again on every idle.
        with pytest.raises(AssertionError, match="rerun it forever"):
            FakeScheduler().dispatch(lambda: True)

    def test_cancelling_a_timer_that_is_not_armed_fails(self):
        s = FakeScheduler()
        fired = s.after_ms(10, lambda: False)
        s.fire_all()
        with pytest.raises(AssertionError, match="no armed timer"):
            s.cancel(fired)

    def test_a_deferred_dispatch_waits_for_the_main_loop(self):
        s, seen = DeferredScheduler(), []
        s.dispatch(seen.append, 1)
        assert seen == []
        s.fire_dispatches()
        assert seen == [1]


def _parameters(fn):
    """What a caller may pass: each parameter's name, kind and default."""
    return [
        (p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()
    ]


@pytest.mark.parametrize(
    ("fake", "real", "surface"),
    [
        (FakeWhisper, WhisperContext, TranscriptionBackend.__protocol_attrs__),
        (FakeAudio, AudioPipeline, AudioBackend.__protocol_attrs__),
        (
            FakeModelManager,
            ModelManager,
            (
                "models_dir",
                "cache_dir",
                "is_downloaded",
                "model_path",
                "openvino_path",
                "has_openvino",
                "vad_model_path",
            ),
        ),
        (
            FakePortalShortcuts,
            PortalShortcuts,
            ("status", "binding", "version", "subscribe", "configure"),
        ),
    ],
    ids=["whisper", "audio", "models", "portal"],
)
def test_a_fake_keeps_the_surface_of_the_class_it_stands_in_for(fake, real, surface):
    # A member the real class renamed or re-signed would leave every test on the fake
    # green while the code under test fails against the real one.
    for name in surface:
        assert hasattr(real, name) and hasattr(fake(), name), name
        if inspect.isfunction(member := getattr(fake, name, None)):
            assert _parameters(member) == _parameters(getattr(real, name)), name


class _Real:
    """`GLibScheduler` on the default main context, run until a condition holds."""

    def __init__(self):
        from quoth.scheduler import GLibScheduler

        self.scheduler = GLibScheduler()

    def run(self, until):
        wait_until(until)


class _Fake:
    """`FakeScheduler`, each `fire_all` standing for one round of due timers."""

    def __init__(self):
        self.scheduler = FakeScheduler()

    def run(self, until):
        for _ in range(3):
            if until():
                return
            self.scheduler.fire_all()
        assert until()


class TestSchedulerContract:
    """The rules of GLib's sources that `FakeScheduler` keeps, on both schedulers."""

    @pytest.fixture(params=[_Real, _Fake], ids=["glib", "fake"])
    def driver(self, request):
        return request.param()

    def test_dispatch_and_timers_pass_their_arguments(self, driver):
        s, seen = driver.scheduler, []
        s.dispatch(seen.append, "idle")
        s.after_ms(1, seen.append, "ms")
        s.after_s(0, seen.append, "s")
        driver.run(lambda: len(seen) == 3)
        assert sorted(seen) == ["idle", "ms", "s"]

    def test_a_timer_that_returns_true_runs_again_until_cancelled(self, driver):
        s, ticks, done = driver.scheduler, [], []
        source = s.after_ms(1, lambda: ticks.append(1) or True)
        driver.run(lambda: len(ticks) >= 2)
        s.cancel(source)
        cancelled_at = len(ticks)
        s.after_ms(5, done.append, True)
        driver.run(lambda: done)
        assert len(ticks) == cancelled_at

    def test_a_timer_may_cancel_itself_and_a_pending_one(self, driver):
        s, ids, ran = driver.scheduler, {}, []

        def first():
            ran.append("first")
            s.cancel(ids["first"])
            s.cancel(ids["second"])
            return False

        ids["first"] = s.after_ms(1, first)
        ids["second"] = s.after_ms(1, ran.append, "second")
        driver.run(lambda: ran)
        s.after_ms(30, ran.append, "after")
        driver.run(lambda: "after" in ran)
        assert ran == ["first", "after"]


class TestJoinThread:
    def test_a_thread_never_started_is_a_no_op(self):
        join_thread(types.SimpleNamespace(t=None), "t")

    def test_a_name_the_owner_lacks_fails(self):
        with pytest.raises(AttributeError, match="nothing"):
            join_thread(object(), "nothing")

    def test_waits_for_a_live_thread(self):
        done = threading.Event()
        owner = type("O", (), {})()
        # Still running when join_thread is called, so only the join sees it finish.
        owner.t = threading.Timer(0.05, done.set)
        owner.t.start()
        join_thread(owner, "t")
        assert done.is_set()

    def test_fails_on_a_thread_that_never_finishes(self):
        stop = threading.Event()
        owner = type("O", (), {})()
        owner.t = threading.Thread(target=stop.wait, daemon=True)
        owner.t.start()
        try:
            with pytest.raises(AssertionError, match="did not finish"):
                join_thread(owner, "t", timeout=0.05)
        finally:
            stop.set()
