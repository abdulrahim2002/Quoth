# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Local, private usage stats.

A tiny record of how much Quoth has been used, kept only on this machine to power the
one-time support nudge and the read-only usage summary in Settings. It never holds
transcript or audio content or identifying data, and never leaves the device; it is
plain JSON under XDG state so the user can inspect or delete it.
"""

import json
from pathlib import Path

from loguru import logger

from quoth import atomic_write, paths

# Successful dictations before the one-time support nudge fires, high enough that the
# ask only reaches someone who has had real, repeated value from the app.
NUDGE_THRESHOLD = 100

# Schema version, bumped only on a breaking layout change.
_VERSION = 1

_DEFAULTS = {
    "version": _VERSION,
    "dictations_total": 0,
    "first_seen": None,  # ISO date of the first recorded dictation
    "nudge_shown": False,
}


def default_path() -> Path:
    """Return the stats file in the state directory: runtime state, not user config."""
    return paths.state_dir() / "stats.json"


def load(path: Path) -> dict:
    """Read the stats, filling missing keys from the defaults.

    Return a fresh default set when the file is absent, unreadable or malformed, so a
    corrupt file never breaks startup.
    """
    stored = atomic_write.read_state(path, "Usage stats")
    return dict(_DEFAULTS) if stored is None else _coerce({**_DEFAULTS, **stored})


def _coerce(state: dict) -> dict:
    """Force the logic-bearing fields to their expected types.

    A wrong type reverts to its default, so a hand-edited file cannot crash a dictation.
    bool is excluded from the count because it is an int subclass.
    """
    total = state["dictations_total"]
    if not isinstance(total, int) or isinstance(total, bool):
        state["dictations_total"] = 0
    if not isinstance(state["nudge_shown"], bool):
        state["nudge_shown"] = False
    if not isinstance(state["first_seen"], (str, type(None))):
        state["first_seen"] = None
    return state


def save(path: Path, state: dict) -> None:
    """Persist the stats atomically with owner-only permissions.

    A write failure is logged and swallowed: stats are best-effort and must never
    interrupt dictation.
    """
    try:
        atomic_write.write_text(path, json.dumps(state, indent=2) + "\n", mode=0o600)
    except OSError as e:
        logger.warning("Could not persist usage stats: {}", e)


def record_dictation(state: dict, *, today: str) -> dict:
    """Return the state after one more successful dictation.

    The total goes up and the first-seen date is stamped once. `today` is passed in so
    the logic stays pure and testable.
    """
    return {
        **state,
        "dictations_total": state["dictations_total"] + 1,
        "first_seen": state["first_seen"] or today,
    }


def due_for_nudge(state: dict) -> bool:
    """Whether the one-time support nudge should fire.

    It fires once there are enough dictations to have earned the ask and it has not been
    shown yet.
    """
    return state["dictations_total"] >= NUDGE_THRESHOLD and not state["nudge_shown"]


def mark_nudged(state: dict) -> dict:
    """Return the state after the support nudge has shown, so it never fires again."""
    return {**state, "nudge_shown": True}
