# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""What a hotkey press or release does to a dictation, whichever source reported it."""

from typing import Literal

# The `mode` setting's values, in the order Settings offers them.
TAP_TO_TOGGLE = "tap-to-toggle"
PUSH_TO_HOLD = "push-to-hold"
MODES = (TAP_TO_TOGGLE, PUSH_TO_HOLD)


def hotkey_action(
    mode: str, active: bool, pressed: bool
) -> Literal["start", "stop"] | None:
    """Return what a hotkey press or release does, or None when it does nothing.

    Tap starts or stops on each press; hold starts on the press and stops on the
    release. Decided from whether a dictation is active, so a dictation started by
    any other trigger is stopped the same way.
    """
    if mode == PUSH_TO_HOLD:
        if pressed:
            return None if active else "start"
        return "stop" if active else None
    if not pressed:
        return None
    return "stop" if active else "start"
