# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.easing`: the theme's named curves evaluated through `Adw.Easing`."""

import pytest

from quoth.easing import ease
from quoth.theme import VALID_EASINGS


def test_every_theme_easing_is_an_adw_curve_from_0_to_1():
    for name in VALID_EASINGS:
        assert ease(name, 0.0) == pytest.approx(0.0, abs=1e-6), name
        assert ease(name, 1.0) == pytest.approx(1.0, abs=1e-6), name


def test_cubic_curves_match_their_formulas():
    assert ease("ease-out-cubic", 0.5) == pytest.approx(0.875)
    assert ease("ease-in-cubic", 0.5) == pytest.approx(0.125)


def test_back_curves_leave_unit_range():
    peak = max(ease("ease-out-back", t / 100) for t in range(101))
    dip = min(ease("ease-in-back", t / 100) for t in range(101))
    assert peak > 1.0
    assert dip < 0.0


def test_linear_is_linear():
    # Adw.Easing.LINEAR is enum value 0, which a truthiness check would lose.
    assert ease("linear", 0.5) == 0.5
