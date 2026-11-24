# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `color`."""

import numpy as np

from quoth import color


def test_oklab_round_trip_preserves_srgb():
    rng = np.random.default_rng(0)
    rgb = rng.random((200, 3))

    back = color.oklab_to_srgb(color.srgb_to_oklab(rgb))

    assert np.allclose(rgb, back, atol=1e-5)


def test_srgb_red_matches_the_reference_oklab_value():
    # The reference value from Björn Ottosson's OKLab definition.
    lab = color.srgb_to_oklab(np.array([1.0, 0.0, 0.0]))

    assert np.allclose(lab, [0.627955, 0.224863, 0.125846], atol=1e-5)


def test_neutral_srgb_has_no_oklab_chroma():
    lab = color.srgb_to_oklab(np.array([0.5, 0.5, 0.5]))

    assert abs(lab[1]) < 1e-6
    assert abs(lab[2]) < 1e-6


def test_oklab_to_linear_keeps_out_of_gamut_channels_unclipped():
    linear = color.oklab_to_linear(np.array([0.85, 0.4, 0.1]))

    assert np.any((linear < 0.0) | (linear > 1.0))
