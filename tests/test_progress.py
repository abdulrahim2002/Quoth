# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.progress`, the smooth progress bar.

The trickle target is pure arithmetic; the motion tests need a display for the widget.
"""

from quoth.progress import _TRICKLE_REACH, _trickle_target


def test_trickle_aims_partway_to_the_ceiling():
    # From 0 toward the first of two milestones (0.5): 90% of the way is 0.45.
    assert _trickle_target(0.0, 0.5) == 0.45


def test_trickle_never_reaches_the_ceiling():
    # It must stay short of the milestone so it can't claim unseen progress.
    assert _trickle_target(0.5, 1.0, _TRICKLE_REACH) < 1.0


def test_trickle_from_a_partial_position():
    # Halfway, aiming at the top: covers 90% of the remaining half.
    assert _trickle_target(0.5, 1.0) == 0.95


def test_trickle_clamps_into_range():
    assert _trickle_target(0.0, 2.0) == 1.0
    assert _trickle_target(0.0, -1.0) == 0.0


def test_reduced_motion_sets_the_target_without_starting_animation(display):
    from unittest.mock import Mock

    from quoth.progress import SmoothProgressBar

    bar = SmoothProgressBar(motion_enabled=False)
    bar._anim = Mock()
    bar.glide_to(0.75)

    assert bar.get_fraction() == 0.75
    bar._anim.play.assert_not_called()


def test_motion_eases_from_the_current_fraction(display):
    from unittest.mock import Mock

    from quoth.progress import SmoothProgressBar

    bar = SmoothProgressBar()
    bar._anim = Mock()
    bar.set_fraction(0.25)
    bar.glide_to(0.75)

    # The animation carries the bar from where it stands; nothing snaps.
    bar._anim.set_value_from.assert_called_once_with(0.25)
    bar._anim.set_value_to.assert_called_once_with(0.75)
    bar._anim.play.assert_called_once_with()
    assert bar.get_fraction() == 0.25


def test_reduced_motion_trickle_stops_short_of_the_ceiling(display):
    from quoth.progress import SmoothProgressBar

    bar = SmoothProgressBar(motion_enabled=False)
    bar.set_fraction(0.5)
    bar.trickle_to(1.0)

    # 90% of the remaining way from where the bar stands, never the milestone.
    assert bar.get_fraction() == 0.95


def test_disabling_motion_skips_an_active_transition(display):
    from unittest.mock import Mock

    from quoth.progress import SmoothProgressBar

    bar = SmoothProgressBar()
    bar._anim = Mock()
    bar.set_motion_enabled(False)

    bar._anim.skip.assert_called_once_with()
