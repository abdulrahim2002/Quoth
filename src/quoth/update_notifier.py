# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Opt-in update checking: the imperative shell around `quoth.updates`.

`updates.py` is the check itself. This module is the thin stateful shell: it holds
run-time state (newest known version, whether a check is in flight, whether a reached
check has completed this run), spawns the fetch off the main thread, and marshals the
result back.

GTK-free: the notification push, the UI refresh and the main-thread scheduler are
injected, so this stays headlessly testable.
"""

import threading
import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from quoth import updates
from quoth.scheduler import Scheduler


def _spawn_thread(work: Callable[[], None]) -> None:
    """Run `work` on a fire-and-forget daemon thread: the default `spawn`."""
    threading.Thread(target=work, name="update-check", daemon=True).start()


class UpdateNotifier:
    """The driver of opt-in update checks, owning the newest-known-release state."""

    def __init__(
        self,
        current_version: str,
        state_path: Path,
        *,
        is_enabled: Callable[[], bool],
        push: Callable[[str], None],
        refresh: Callable[[], None],
        scheduler: Scheduler,
        check_fn: Callable[[str], updates.UpdateResult] = updates.check_for_update,
        spawn: Callable[[Callable[[], None]], None] = _spawn_thread,
    ):
        self._version = current_version
        self._state_path = state_path
        self._is_enabled = is_enabled
        self._push = push
        self._refresh = refresh
        self._scheduler = scheduler
        self._check_fn = check_fn
        self._spawn = spawn

        self._available: str | None = None  # newest known release, or None
        self._checking = False  # guards concurrent checks
        self._checked_once = False  # a reached check has completed this run
        self._last_reached: bool | None = None  # outcome of the last completed check

    # --- State the composition root reflects onto the UI ---

    @property
    def available(self) -> str | None:
        """The newest known release above the running version, or None."""
        return self._available

    @property
    def is_checking(self) -> bool:
        """Whether a check is in flight (drives the settings spinner)."""
        return self._checking

    @property
    def checked_once(self) -> bool:
        """Whether a check has actually reached GitHub this run."""
        return self._checked_once

    @property
    def last_reached(self) -> bool | None:
        """Whether the last completed check reached GitHub (None before the first)."""
        return self._last_reached

    # --- Entry points ---

    def maybe_check(self, force: bool = False) -> None:
        """Run the background auto-check, gated by the opt-in toggle and daily throttle.

        On a newer version it pushes a desktop notification; it fails silent. `force`
        skips only the throttle (used when the toggle is first enabled), never the
        opt-in gate.
        """
        if not self._is_enabled():
            logger.debug("Update check skipped: notify_updates disabled")
            return
        state = updates.load_state(self._state_path)
        now = time.time()
        if not force and not updates.due_for_check(state, now):
            logger.debug("Update check skipped: not due yet")
            return
        self._run(state, now, notify=True)

    def check_now(self) -> None:
        """Run a manual, user-initiated check (the in-app 'Check now').

        It bypasses the toggle and the throttle (the click is explicit consent) and
        reports inline through the refresh callback rather than a desktop notification.
        """
        self._run(updates.load_state(self._state_path), time.time(), notify=False)

    # --- Worker + result glue ---

    def _run(self, state: dict, now: float, *, notify: bool) -> None:
        """Fetch off the main thread and marshal the result back.

        `notify` distinguishes the background push (True) from the inline manual check
        (False). A check already in flight makes this a no-op, or two could push the
        same release.
        """
        if self._checking:
            return
        self._checking = True
        self._refresh()

        def _work():
            try:
                result = self._check_fn(self._version)
            except Exception:
                # Never let a worker die without clearing the in-flight state: a
                # wedged spinner would block future manual checks.
                logger.exception("Update check thread failed")
                result = updates.UpdateResult(version=None, reached=False)
            self._scheduler.dispatch(self._on_result, result, state, now, notify)

        try:
            self._spawn(_work)
        except Exception:
            logger.exception("Could not start the update check")
            self._on_result(
                updates.UpdateResult(version=None, reached=False), state, now, notify
            )

    def _on_result(
        self, result: updates.UpdateResult, state: dict, checked_at: float, notify: bool
    ) -> bool:
        """Persist the state, push a new release (background only), refresh the view.

        Runs on the main thread.
        """
        self._checking = False
        self._last_reached = result.reached
        if result.reached:
            self._checked_once = True
        is_new = updates.register_result(state, result, checked_at)
        logger.debug(
            "Update check result: reached={} version={} is_new={} notify={}",
            result.reached,
            result.version,
            is_new,
            notify,
        )
        updates.save_state(self._state_path, state)
        if result.version:  # only ever set to something strictly newer than we run
            self._available = result.version
        if result.version and is_new and notify:
            self._push(result.version)
        self._refresh()
        return False  # one-shot idle source
