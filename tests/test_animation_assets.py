# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for the shipped Lottie assets in `quoth/animations`.

They check the committed JSON directly rather than against a reference render;
`test_lottie.py` covers the rendering path.
"""

import json
from pathlib import Path

import pytest

from quoth import lottie

_ANIM_DIR = Path(__file__).resolve().parents[1] / "src" / "quoth" / "animations"
# The committed assets' frame counts, so a bad regeneration or hand-edit is caught here.
_ASSETS = {"intro.json": 138, "loader.json": 191}


def _fills(node) -> list[dict]:
    """Every fill (ty == 'fl') anywhere in the document tree."""
    out = []
    if isinstance(node, dict):
        if node.get("ty") == "fl":
            out.append(node)
        for value in node.values():
            out.extend(_fills(value))
    elif isinstance(node, list):
        for item in node:
            out.extend(_fills(item))
    return out


@pytest.mark.parametrize(("name", "frames"), _ASSETS.items())
class TestShippedAssets:
    def _doc(self, name: str) -> dict:
        return json.loads((_ANIM_DIR / name).read_text())

    def test_is_a_lottie_document(self, name, frames):
        doc = self._doc(name)
        assert doc["v"] and doc["fr"] == 60 and doc["layers"]

    def test_frame_count_matches_the_committed_asset(self, name, frames):
        assert self._doc(name)["op"] == frames

    def test_every_fill_carries_the_ink_slot(self, name, frames):
        # The theme recolours through the `ink` slot; a fill without it would silently
        # ship a frozen colour that ignores light/dark.
        fills = _fills(self._doc(name))
        assert fills, "no fills found"
        assert all(fill["c"].get("sid") == "ink" for fill in fills)

    def test_renders_a_non_empty_frame(self, name, frames):
        if lottie._LIB is None:
            pytest.skip("libthorvg not built")
        anim = lottie.load(str(_ANIM_DIR / name))
        surface = anim.render(120, 120, frames * 0.5)
        assert any(surface.get_data()[3::4])
