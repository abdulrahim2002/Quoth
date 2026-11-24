# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""The theme's named easing curves, evaluated by libadwaita's AdwEasing."""

import gi

gi.require_version("Adw", "1")
from gi.repository import Adw  # noqa: E402


def ease(name: str, t: float) -> float:
    """Return the named curve's value at progress `t` (0..1).

    The back family leaves [0, 1]: it dips below 0 or overshoots 1.
    """
    return Adw.Easing.ease(getattr(Adw.Easing, name.upper().replace("-", "_")), t)
