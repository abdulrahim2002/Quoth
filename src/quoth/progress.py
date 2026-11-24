# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""A progress bar that eases to each value instead of snapping to it.

Two verbs over one `Adw.TimedAnimation` on the bar's `fraction` property: `glide_to`
eases to a known value, and `trickle_to` creeps toward an unknown-duration target so a
long wait never sits frozen and never claims to be finished. libadwaita drives the
tween and the easing, so there is no timer loop here. With motion off each value is
set at once.

Used by onboarding's Optimize step, whose benchmark reports progress in bursts.
"""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gtk

_GLIDE_MS = 350  # ease to a known value
_TRICKLE_MS = 20_000  # a long, decelerating creep toward an unknown-duration target
# How much of the gap to the next milestone a trickle covers before it holds: enough to
# read as steady progress, short of the milestone so it never claims unseen work.
_TRICKLE_REACH = 0.9


def _trickle_target(
    current: float, ceiling: float, reach: float = _TRICKLE_REACH
) -> float:
    """Return where a trickle aims: `reach` of the way from `current` to `ceiling`.

    The result is clamped to 0..1.
    """
    return max(0.0, min(1.0, current + (ceiling - current) * reach))


class SmoothProgressBar(Gtk.ProgressBar):
    """A `Gtk.ProgressBar` that animates between fractions.

    Use `glide_to` for a value you know and `trickle_to` while waiting on one you don't.
    """

    def __init__(self, *, motion_enabled: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self._motion_enabled = motion_enabled
        self._anim = Adw.TimedAnimation.new(
            self, 0.0, 0.0, _GLIDE_MS, Adw.PropertyAnimationTarget.new(self, "fraction")
        )

    def glide_to(self, fraction: float) -> None:
        """Ease to a known fraction (0..1)."""
        self._run(fraction, _GLIDE_MS)

    def trickle_to(self, ceiling: float) -> None:
        """Creep toward, but not to, `ceiling` on a decelerating curve.

        An unknown wait then keeps moving without ever claiming to have reached the
        milestone.
        """
        self._run(_trickle_target(self.get_fraction(), ceiling), _TRICKLE_MS)

    def _run(self, to: float, duration_ms: int) -> None:
        to = max(0.0, min(1.0, to))
        if not self._motion_enabled:
            self.set_fraction(to)
            return
        self._anim.set_value_from(self.get_fraction())
        self._anim.set_value_to(to)
        self._anim.set_duration(duration_ms)
        self._anim.set_easing(Adw.Easing.EASE_OUT_CUBIC)
        self._anim.play()

    def set_motion_enabled(self, enabled: bool) -> None:
        if enabled == self._motion_enabled:
            return
        self._motion_enabled = enabled
        if not enabled:
            self._anim.skip()
