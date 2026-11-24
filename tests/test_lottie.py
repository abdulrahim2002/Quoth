# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.lottie`, the ThorVG Lottie renderer.

The render tests skip without libthorvg, which a plain `uv run pytest` does not build
(no `make vendor-thorvg`); the availability and degradation tests always run.
"""

import json

import pytest

from quoth import lottie

# A minimal one-layer Lottie: an 80x80 fill tagged `sid: "ink"`, so both rendering and
# the recolour slot are exercised without depending on the shipped animation assets.
_MINIMAL = {
    "v": "5.7.0",
    "fr": 30,
    "ip": 0,
    "op": 10,
    "w": 100,
    "h": 100,
    "nm": "t",
    "ddd": 0,
    "assets": [],
    "layers": [
        {
            "ddd": 0,
            "ind": 1,
            "ty": 4,
            "nm": "r",
            "sr": 1,
            "ip": 0,
            "op": 10,
            "st": 0,
            "bm": 0,
            "ks": {
                "o": {"a": 0, "k": 100},
                "r": {"a": 0, "k": 0},
                "p": {"a": 0, "k": [50, 50, 0]},
                "a": {"a": 0, "k": [0, 0, 0]},
                "s": {"a": 0, "k": [100, 100, 100]},
            },
            "shapes": [
                {
                    "ty": "gr",
                    "it": [
                        {
                            "ty": "rc",
                            "p": {"a": 0, "k": [0, 0]},
                            "s": {"a": 0, "k": [80, 80]},
                            "r": {"a": 0, "k": 0},
                        },
                        {
                            "ty": "fl",
                            "c": {"a": 0, "k": [0, 0, 0, 1], "sid": "ink"},
                            "o": {"a": 0, "k": 100},
                        },
                        {
                            "ty": "tr",
                            "p": {"a": 0, "k": [0, 0]},
                            "a": {"a": 0, "k": [0, 0]},
                            "s": {"a": 0, "k": [100, 100]},
                            "r": {"a": 0, "k": 0},
                            "o": {"a": 0, "k": 100},
                        },
                    ],
                }
            ],
        }
    ],
}

needs_thorvg = pytest.mark.skipif(
    lottie._LIB is None, reason="libthorvg not built (no make vendor-thorvg)"
)


def _asset(tmp_path) -> str:
    p = tmp_path / "min.json"
    p.write_text(json.dumps(_MINIMAL))
    return str(p)


def _ink(surface) -> int:
    """Total alpha, so 'did it draw anything' is a number. 0 for a None surface."""
    return sum(surface.get_data()[3::4]) if surface is not None else 0


class TestAvailability:
    def test_a_missing_file_loads_to_none_not_a_crash(self):
        # Whether or not the library is present, a bad path must degrade to None.
        assert lottie.load("/no/such/file.json") is None


@needs_thorvg
class TestRender:
    def test_loads_and_reports_frame_count(self, tmp_path):
        anim = lottie.load(_asset(tmp_path))
        assert anim is not None
        assert anim.total_frames == 10.0
        assert anim.duration_s > 0

    def test_reports_the_document_size(self, tmp_path):
        anim = lottie.load(_asset(tmp_path))
        assert anim.width == 100 and anim.height == 100

    def test_renders_ink(self, tmp_path):
        anim = lottie.load(_asset(tmp_path))
        assert _ink(anim.render(100, 100, 0)) > 0

    def test_renders_at_a_second_size(self, tmp_path):
        anim = lottie.load(_asset(tmp_path))
        assert _ink(anim.render(100, 100, 0)) > 0
        assert _ink(anim.render(40, 40, 0)) > 0

    def test_apply_ink_still_renders(self, tmp_path):
        anim = lottie.load(_asset(tmp_path))
        anim.apply_ink((0.2, 0.33, 0.82, 1.0))
        assert _ink(anim.render(100, 100, 0)) > 0

    def test_close_is_idempotent_and_render_after_close_is_none(self, tmp_path):
        anim = lottie.load(_asset(tmp_path))
        anim.close()
        anim.close()
        assert anim.render(100, 100, 0) is None  # must not touch freed memory

    def test_repeated_load_and_close_does_not_crash(self, tmp_path):
        # The teardown contract: the picture is animation-owned and reference-counted
        # with the canvas, so many create/destroy cycles free it once, with no leak.
        path = _asset(tmp_path)
        for i in range(50):
            anim = lottie.load(path)
            anim.render(64, 64, i % 10)
            anim.close()
