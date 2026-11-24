# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `trigger`: what a hotkey press or release does to a dictation."""

import pytest

from quoth.trigger import PUSH_TO_HOLD, TAP_TO_TOGGLE, hotkey_action


@pytest.mark.parametrize(
    ("mode", "active", "pressed", "action"),
    [
        (TAP_TO_TOGGLE, False, True, "start"),
        (TAP_TO_TOGGLE, True, True, "stop"),
        (TAP_TO_TOGGLE, False, False, None),
        (TAP_TO_TOGGLE, True, False, None),
        (PUSH_TO_HOLD, False, True, "start"),
        (PUSH_TO_HOLD, True, True, None),
        (PUSH_TO_HOLD, False, False, None),
        (PUSH_TO_HOLD, True, False, "stop"),
    ],
)
def test_hotkey_action(mode, active, pressed, action):
    assert hotkey_action(mode, active, pressed) == action
