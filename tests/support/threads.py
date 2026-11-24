# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Deterministic waits on the background threads the app spawns."""

from __future__ import annotations


def join_thread(owner, attr: str, timeout: float = 2.0) -> None:
    """Wait for `owner.attr` to finish, and fail if it does not.

    `attr` may name one thread or a list of them, None while none has started. A name
    the owner lacks fails, so a renamed attribute cannot quietly skip the wait. Joining
    rather than polling state waits for the whole worker chain, not just the part a
    flag reflects. It asserts the thread finished, since a bare join returns silently
    on a wedged thread and the test that follows would read as passing.
    """
    found = getattr(owner, attr)
    threads = found if isinstance(found, list) else [found]
    for thread in threads:
        if thread is None:
            continue
        thread.join(timeout)
        assert not thread.is_alive(), f"{attr} did not finish within {timeout}s"
