# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""The main-thread scheduling seam and its GLib implementation.

`RecordingController` must not import GTK directly, so it reaches the main thread and
timers through `Scheduler`. Production passes `GLibScheduler`, which keeps every UI
update on `GLib.idle_add` and GLib timers; tests inject a fake that runs callbacks
synchronously. `guarded` logs a raise in any detached callback.
"""

import functools
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from gi.repository import GLib
from loguru import logger


@runtime_checkable
class Scheduler(Protocol):
    """The main-thread dispatch seam `RecordingController` depends on.

    Background threads (evdev, GStreamer, whisper) never touch GTK directly; they
    reach the main loop and timers through this. `dispatch` runs a callback on the
    main thread; `after_ms`/`after_s` schedule a timer and return a source id for
    `cancel`.
    """

    def dispatch(self, fn: Callable, *args) -> None: ...

    def after_ms(self, ms: int, fn: Callable, *args) -> int: ...

    def after_s(self, seconds: int, fn: Callable, *args) -> int: ...

    def cancel(self, source_id: int) -> None: ...


def guarded(fn: Callable, *, on_error: Callable[[], None] | None = None) -> Callable:
    """Wrap a detached callback so its exceptions are logged, not swallowed.

    A raise is logged and the wrapper returns False, so GLib drops the source;
    otherwise the callback's own return value passes through. `on_error` runs after
    the log line, so an owner that keeps the source id can forget it.
    """

    @functools.wraps(fn)
    def wrapper(*args):
        try:
            return fn(*args)
        except Exception:
            logger.exception("Callback {} raised", getattr(fn, "__name__", fn))
            if on_error is not None:
                on_error()
            return False

    return wrapper


class GLibScheduler:
    """The production `Scheduler`: GLib main-loop sources around `guarded` callbacks."""

    def dispatch(self, fn: Callable, *args) -> None:
        GLib.idle_add(guarded(fn), *args)

    def after_ms(self, ms: int, fn: Callable, *args) -> int:
        return GLib.timeout_add(ms, guarded(fn), *args)

    def after_s(self, seconds: int, fn: Callable, *args) -> int:
        return GLib.timeout_add_seconds(seconds, guarded(fn), *args)

    def cancel(self, source_id: int) -> None:
        GLib.source_remove(source_id)
