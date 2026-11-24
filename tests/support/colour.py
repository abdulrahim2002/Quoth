# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Colour arithmetic for assertions about what reads."""


def contrast(first: str, second: str) -> float:
    """Return the WCAG 2 contrast ratio of two `#rrggbb` colours."""

    def luminance(value: str) -> float:
        channels = [int(value[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        red, green, blue = (
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        )
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)
