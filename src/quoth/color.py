# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Shared sRGB and OKLab conversion primitives."""

import numpy as np

# Single precision is sufficient for 8-bit output and keeps the orb's image fields
# compact; plain float constants preserve a float32 input through each operation.
_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float32,
)
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float32,
)
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    """Convert sRGB components in 0..1 to linear light."""
    return np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    """Convert linear light to sRGB, clipping to 0..1 first."""
    value = np.clip(value, 0.0, 1.0)
    return np.where(
        value <= 0.0031308,
        value * 12.92,
        1.055 * value ** (1 / 2.4) - 0.055,
    )


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB triples on the last axis to OKLab."""
    lms = np.cbrt(srgb_to_linear(rgb) @ _M1.T)
    return lms @ _M2.T


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    """Convert OKLab to unclipped linear sRGB."""
    lms = (lab @ _M2_INV.T) ** 3
    return lms @ _M1_INV.T


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert OKLab to sRGB clipped to 0..1."""
    return linear_to_srgb(oklab_to_linear(lab))
