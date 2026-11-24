# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `update_notifier`, the update-check shell over the pure `updates` core.

The GTK effects, main-thread hop, network check and worker thread are injected, so it
all runs inline and headless; state persists in a real temp file.
"""

from quoth import updates
from quoth.update_notifier import UpdateNotifier
from tests.support import FakeScheduler


class _Recorder:
    """Records the `push` and `refresh` calls."""

    def __init__(self):
        self.pushed: list[str] = []
        self.refreshes = 0

    def push(self, version: str) -> None:
        self.pushed.append(version)

    def refresh(self) -> None:
        self.refreshes += 1


def _make(tmp_path, *, enabled=True, result=None, check_fn=None, spawn=None):
    rec = _Recorder()
    if check_fn is None:
        result = result or updates.UpdateResult(version="0.2.0", reached=True)
        check_fn = lambda _v: result  # noqa: E731
    notifier = UpdateNotifier(
        "0.1.0",
        tmp_path / "update_check.json",
        is_enabled=lambda: enabled,
        push=rec.push,
        refresh=rec.refresh,
        scheduler=FakeScheduler(),
        check_fn=check_fn,
        spawn=spawn or (lambda work: work()),  # inline by default, no real thread
    )
    return notifier, rec


class TestMaybeCheck:
    def test_disabled_never_touches_network(self, tmp_path):
        called = []
        notifier, rec = _make(
            tmp_path, enabled=False, check_fn=lambda v: called.append(v)
        )
        notifier.maybe_check()
        assert called == []  # opt-in gate: no fetch
        assert rec.refreshes == 0

    def test_force_does_not_bypass_the_opt_in_gate(self, tmp_path):
        called = []
        notifier, _ = _make(tmp_path, enabled=False, check_fn=called.append)
        notifier.maybe_check(force=True)
        assert called == []

    def test_new_version_sets_state_and_pushes(self, tmp_path):
        notifier, rec = _make(tmp_path)
        notifier.maybe_check()
        assert notifier.available == "0.2.0"
        assert notifier.checked_once is True
        assert notifier.last_reached is True
        assert notifier.is_checking is False
        assert rec.pushed == ["0.2.0"]  # background push fired
        assert rec.refreshes == 2  # once at start (spinner), once at finish

    def test_throttle_blocks_second_check(self, tmp_path):
        calls = []
        notifier, _ = _make(
            tmp_path,
            check_fn=lambda v: (
                calls.append(v) or updates.UpdateResult(version="0.2.0", reached=True)
            ),
        )
        notifier.maybe_check()
        notifier.maybe_check()  # within the daily interval, so throttled
        assert len(calls) == 1

    def test_force_bypasses_throttle(self, tmp_path):
        calls = []
        notifier, _ = _make(
            tmp_path,
            check_fn=lambda v: (
                calls.append(v) or updates.UpdateResult(version="0.2.0", reached=True)
            ),
        )
        notifier.maybe_check()
        notifier.maybe_check(force=True)
        assert len(calls) == 2

    def test_a_check_in_flight_is_not_doubled(self, tmp_path):
        # Startup's check and enabling the toggle can overlap; one release, one push.
        workers = []  # both run only after both were asked
        notifier, rec = _make(tmp_path, spawn=workers.append)
        notifier.maybe_check()
        notifier.maybe_check(force=True)
        for work in workers:
            work()
        assert len(workers) == 1
        assert rec.pushed == ["0.2.0"]

    def test_a_release_already_announced_is_not_pushed_again(self, tmp_path):
        notifier, rec = _make(
            tmp_path, result=updates.UpdateResult(version="0.2.0", reached=True)
        )
        notifier.maybe_check()
        notifier.maybe_check(force=True)  # reaches GitHub again: the same release
        assert rec.pushed == ["0.2.0"]

    def test_unreached_check_does_not_persist_or_wedge(self, tmp_path):
        calls = []
        notifier, rec = _make(
            tmp_path,
            check_fn=lambda v: (
                calls.append(v) or updates.UpdateResult(version=None, reached=False)
            ),
        )
        notifier.maybe_check()
        assert notifier.checked_once is False
        assert notifier.last_reached is False
        assert rec.pushed == []
        # An unreached check records nothing, so it is still due and a retry runs.
        notifier.maybe_check()
        assert len(calls) == 2


class TestCheckNow:
    def test_manual_check_ignores_toggle_and_never_pushes(self, tmp_path):
        notifier, rec = _make(tmp_path, enabled=False)  # opt-in off
        notifier.check_now()
        assert notifier.available == "0.2.0"  # ran despite the toggle being off
        assert rec.pushed == []  # manual path reports inline, never a desktop push

    def test_concurrent_manual_check_guarded(self, tmp_path):
        # A check_now() fired re-entrantly from check_fn is a no-op while in flight.
        calls = []

        def reentrant(v):
            calls.append(v)
            notifier.check_now()  # should hit the _checking guard and return
            return updates.UpdateResult(version="0.2.0", reached=True)

        notifier, _ = _make(tmp_path, check_fn=reentrant)
        notifier.check_now()
        assert len(calls) == 1

    def test_a_check_that_cannot_start_clears_in_flight_state(self, tmp_path):
        def unstartable(_work):
            raise RuntimeError("can't start new thread")

        notifier, rec = _make(tmp_path, spawn=unstartable)
        notifier.check_now()
        assert notifier.is_checking is False
        assert rec.refreshes == 2

    def test_a_known_update_survives_a_failed_recheck(self, tmp_path):
        results = [
            updates.UpdateResult(version="0.2.0", reached=True),
            updates.UpdateResult(version=None, reached=False),
        ]
        notifier, _ = _make(tmp_path, check_fn=lambda _v: results.pop(0))
        notifier.check_now()
        notifier.check_now()
        assert notifier.available == "0.2.0"

    def test_worker_exception_clears_in_flight_state(self, tmp_path):
        def boom(_v):
            raise RuntimeError("network blew up")

        notifier, rec = _make(tmp_path, check_fn=boom)
        notifier.check_now()
        assert notifier.is_checking is False  # spinner cleared despite the crash
        assert notifier.available is None
        assert rec.refreshes == 2
