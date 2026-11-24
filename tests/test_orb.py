# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.orb`: the theme orb's colour maths, flow and rendering.

The look is tuned by eye, so these pin behaviour, not pixels: any palette stays in
gamut, a seed fixes one unique orb, and the flow stays divergence-free inside the rim.
"""

import cairo
import numpy as np
import pytest

from quoth import color, orb
from quoth.pill_view import orb_blobs, orb_seed
from quoth.theme import list_presets, preset_theme

_BG = (0.10, 0.11, 0.13, 1.0)
_BLOBS = [
    ((0.36, 0.51, 0.67, 1.0), 3.0),
    ((0.53, 0.75, 0.82, 1.0), 2.5),
    ((0.85, 0.87, 0.91, 1.0), 1.5),
    ((0.75, 0.38, 0.42, 1.0), 1.0),
    ((0.92, 0.80, 0.55, 1.0), 1.0),
]


def _orb(background=_BG, blobs=_BLOBS, rim=(1.0, 1.0, 1.0, 0.2), seed=7):
    return orb.Orb(
        background=background,
        blobs=blobs,
        rim=rim,
        glow=(0.5, 0.6, 0.9, 1.0),
        seed=seed,
    )


def _rest_frame(o):
    """Return the bytes `o` draws with motion off, the reduced-motion view."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 48, 48)
    o.draw(cairo.Context(surface), size=48, motion=False)
    surface.flush()
    return bytes(surface.get_data())


class TestOKLabMixing:
    def test_midpoint_is_not_muddy(self):
        # OKLab mixing: the sRGB mean of blue and yellow is a dead grey, while the
        # perceptual mean keeps real chroma.
        blue = np.array([0.0, 0.0, 1.0])
        yellow = np.array([1.0, 1.0, 0.0])
        srgb_mid = (blue + yellow) / 2
        ok_mid = color.oklab_to_srgb(
            (color.srgb_to_oklab(blue) + color.srgb_to_oklab(yellow)) / 2
        )
        srgb_chroma = srgb_mid.max() - srgb_mid.min()
        ok_chroma = ok_mid.max() - ok_mid.min()
        assert ok_chroma > srgb_chroma


class TestGamut:
    def test_in_gamut_colours_pass_through_untouched(self):
        rng = np.random.default_rng(3)
        lab = color.srgb_to_oklab(rng.random((100, 3)))
        assert np.allclose(orb._gamut_project(lab), lab)

    def test_out_of_gamut_lands_on_the_boundary(self):
        # A heavily boosted chroma must come back displayable.
        lab = np.array([[0.85, 0.4, 0.1], [0.3, -0.5, 0.3], [0.95, 0.0, 0.5]])
        fixed = orb._gamut_project(lab)
        rgb = color.oklab_to_linear(fixed)
        assert np.all(rgb >= -1e-3) and np.all(rgb <= 1.0 + 1e-3)
        # On the boundary, not merely inside it: each colour has a channel at a limit.
        assert np.all((rgb.min(axis=-1) < 1e-3) | (rgb.max(axis=-1) > 1.0 - 1e-3))

    def test_projection_preserves_hue(self):
        # Unlike a per-channel clip, the projection keeps the hue angle.
        lab = np.array([[0.7, 0.35, 0.2]])
        fixed = orb._gamut_project(lab)[0]
        assert np.arctan2(lab[0, 2], lab[0, 1]) == pytest.approx(
            np.arctan2(fixed[2], fixed[1]), abs=1e-6
        )


class TestField:
    def test_finite_and_in_gamut_for_arbitrary_palettes(self):
        # Any theme is possible, not just the bundled four: random RGBA blobs on a
        # random background must never produce a NaN or an out-of-range channel.
        rng = np.random.default_rng(1)
        for trial in range(25):
            blobs = [(tuple(rng.random(4)), float(rng.random() * 3)) for _ in range(5)]
            o = _orb(background=tuple(rng.random(4)), blobs=blobs, seed=trial)
            for _ in range(3):
                o.advance(0.5)
            assert np.all(np.isfinite(o._dye))
            field = o._paint_srgb(o._centres, o._t)
            assert np.all(np.isfinite(field))
            assert field.min() >= 0.0 and field.max() <= 1.0

    def test_fully_transparent_blob_contributes_nothing(self):
        # A blob's alpha folds into its weight, so an invisible colour must leave the
        # field as pure background.
        transparent = _orb(blobs=[((1.0, 0.0, 0.0, 0.0), 3.0)], seed=5)
        empty = _orb(blobs=[], seed=5)
        a = transparent._paint_srgb(transparent._centres, 0.0)
        b = empty._paint_srgb(empty._centres, 0.0)
        assert np.allclose(a, b, atol=1e-6)

    def test_blob_alpha_scales_material_weight(self):
        # Half the alpha paints exactly as half the weight does.
        faint = _orb(
            blobs=[
                ((1.0, 0.0, 0.0, 0.5), 4.0),
                ((0.0, 0.0, 1.0, 1.0), 3.0),
            ]
        )
        lighter = _orb(
            blobs=[
                ((1.0, 0.0, 0.0, 1.0), 2.0),
                ((0.0, 0.0, 1.0, 1.0), 3.0),
            ]
        )

        assert np.array_equal(faint._dye, lighter._dye)

    def test_background_alpha_does_not_change_the_opaque_material(self):
        transparent = _orb(background=(0.5, 0.2, 0.7, 0.0), seed=6)
        opaque = _orb(background=(0.5, 0.2, 0.7, 1.0), seed=6)

        assert np.array_equal(transparent._dye, opaque._dye)

    def test_no_blobs_is_the_background(self):
        o = _orb(background=(0.5, 0.2, 0.7, 1.0), blobs=[])
        field = o._paint_srgb(o._centres, 0.0)
        centre = field[orb._FIELD // 2, orb._FIELD // 2]
        assert centre == pytest.approx(
            color.oklab_to_srgb(color.srgb_to_oklab(np.array([0.5, 0.2, 0.7]))),
            abs=1e-5,
        )

    def test_an_owned_pixel_is_the_exact_palette_colour(self):
        # The faithfulness contract: where one colour fully owns a pixel, the theme
        # author's exact colour shows, boosts and gates notwithstanding.
        colour = (0.75, 0.30, 0.45, 1.0)
        o = _orb(blobs=[(colour, 500.0)])
        o._warp[:] = 0.0  # undisplaced coordinates, so the home lands where aimed
        home = o._homes[0]
        field = o._paint_srgb(o._homes, 0.0)
        col = int((home[0] + 0.5) * orb._FIELD)
        row = int((home[1] + 0.5) * orb._FIELD)
        assert field[row, col] == pytest.approx(np.array(colour[:3]), abs=0.02)

    def test_grey_palette_stays_neutral(self):
        # Chroma restoration scales the a/b resultant; on an all-grey palette that
        # resultant is zero, so nothing may invent colour.
        greys = [((v, v, v, 1.0), w) for v, w in ((0.8, 3.0), (0.5, 2.0), (0.3, 1.0))]
        o = _orb(background=(0.1, 0.1, 0.1, 1.0), blobs=greys)
        o.advance(2.0)
        field = o._paint_srgb(o._centres, o._t)
        assert np.allclose(field[..., 0], field[..., 1], atol=2e-3)
        assert np.allclose(field[..., 1], field[..., 2], atol=2e-3)
        # The transported dye is grey too: advection of grey is grey.
        srgb_dye = color.oklab_to_srgb(orb._gamut_project(o._dye))
        assert np.allclose(srgb_dye[..., 0], srgb_dye[..., 1], atol=2e-3)


class TestParams:
    def test_neutral_params_change_nothing(self):
        # A caller that never touches params must get the ambient orb unchanged.
        plain, tuned = _orb(seed=3), _orb(seed=3)
        tuned.set_params(orb.OrbParams())
        for o in (plain, tuned):
            o.advance(0.5)
        assert np.array_equal(plain._dye, tuned._dye)

    def test_seek_jumps_the_clock_but_not_the_centres(self):
        o = _orb(seed=16)
        homes = o._centres.copy()
        o.seek(120.0)
        assert o._t == 120.0
        assert np.array_equal(o._centres, homes)
        # A jumped clock has no transport history: the dye is the fresh painting.
        assert np.array_equal(o._dye, o._paint(o._centres, o._t))
        # And it stays deterministic: the same seek lands on the same dye.
        other = _orb(seed=16)
        other.seek(120.0)
        assert np.array_equal(o._dye, other._dye)

    def test_time_scale_paces_the_flow(self):
        fast, slow = _orb(seed=4), _orb(seed=4)
        fast.set_params(orb.OrbParams(time_scale=2.0))
        for o in (fast, slow):
            o.advance(0.1)
        assert fast._t == pytest.approx(2 * slow._t)
        assert not np.array_equal(fast._centres, slow._centres)

    def test_zero_emphasis_silences_the_palette(self):
        # Emphasis multiplies weights, so all-zero must leave pure background.
        o = _orb(seed=6)
        o.set_params(orb.OrbParams(emphasis=(0.0,) * len(_BLOBS)))
        muted = o._paint_srgb(o._centres, 0.0)
        empty = _orb(blobs=[], seed=6)
        assert np.allclose(muted, empty._paint_srgb(empty._centres, 0.0), atol=1e-6)

    def test_emphasis_retilts_the_field(self):
        o = _orb(seed=8)
        neutral = o._paint_srgb(o._centres, 0.0)
        o.set_params(orb.OrbParams(emphasis=(3.0, 0.2, 0.2, 0.2, 0.2)))
        assert not np.array_equal(neutral, o._paint_srgb(o._centres, 0.0))

    def test_deep_swell_keeps_the_field_in_gamut(self):
        # The breathing floor: a hard swell must not swing weights negative. Ten
        # seconds in, this swell would take two of the five weights below -1.5.
        o = _orb(seed=10)
        o.set_params(orb.OrbParams(swell=6.0))
        o.seek(10.0)
        lightness = o._paint(o._centres, o._t)[..., 0]
        # A blend of the palette is never darker than its darkest colour, the
        # background, nor lighter than white.
        darkest = color.srgb_to_oklab(np.array(_BG[:3]))[0]
        assert lightness.min() >= darkest - 1e-3
        assert lightness.max() <= 1.0

    def test_scale_shrinks_the_drawing(self):
        # The size channel must actually change pixels, and full scale must fill
        # the box exactly as an unparameterised orb does.
        def rendered(scale):
            o = _orb(seed=14)
            o.set_params(orb.OrbParams(scale=scale))
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 48, 48)
            o.draw(cairo.Context(surface), size=48)
            surface.flush()
            return bytes(surface.get_data())

        assert rendered(1.0) != rendered(0.8)
        plain = _orb(seed=14)
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 48, 48)
        plain.draw(cairo.Context(surface), size=48)
        surface.flush()
        assert rendered(1.0) == bytes(surface.get_data())

    def test_glow_lights_the_drawing(self):
        # The glow layers only add light, so a stronger glow draws a brighter orb.
        def brightness(glow):
            o = _orb(seed=14)
            o.set_params(orb.OrbParams(glow=glow))
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 48, 48)
            o.draw(cairo.Context(surface), size=48)
            surface.flush()
            return sum(surface.get_data())

        assert brightness(2.0) > brightness(1.0) > brightness(0.0)

    def test_params_apply_to_the_rest_frame(self):
        # Reduced motion keeps the causal response: emphasis shows at rest too.
        o = _orb(seed=12)
        rest_neutral = _rest_frame(o)
        o.set_params(orb.OrbParams(emphasis=(2.0, 0.5, 0.5, 0.5, 0.5)))
        assert rest_neutral != _rest_frame(o)


class TestDisplay:
    def test_display_table_matches_the_reference_curve(self):
        # The table is a quantiser, not an approximation: within one 8-bit step
        # of the exact transfer curve everywhere.
        linear = np.random.default_rng(31).random(500).astype(np.float32)
        exact = np.round(
            np.clip(color.linear_to_srgb(linear.astype(np.float64)), 0.0, 1.0) * 255.0
        )
        table = orb._ENCODE_LUT.take((linear * 4095.0 + 0.5).astype(np.intp))
        assert np.abs(table.astype(np.float64) - exact).max() <= 1

    def test_pipeline_stays_single_precision(self):
        # One stray float64 scalar would silently upcast the whole chain.
        o = _orb(seed=33)
        o.advance(0.3)
        assert o._dye.dtype == np.float32
        assert o._paint(o._centres, o._t).dtype == np.float32


class TestDye:
    """The dye: colour the flow transports, fed by sources, else the background."""

    def test_dye_is_born_as_the_painting(self):
        o = _orb(seed=22)
        assert np.array_equal(o._dye, o._paint(o._centres, o._t))

    def test_a_source_keeps_its_centre_coloured(self):
        # Injection carries the faithfulness invariant into the transported world:
        # wherever the emitter wanders, the dye at its centre stays its colour.
        colour = (0.75, 0.30, 0.45, 1.0)
        o = _orb(blobs=[(colour, 500.0)], seed=23)
        o._warp[:] = 0.0
        for _ in range(120):
            o.advance(0.05)
        srgb = color.oklab_to_srgb(orb._gamut_project(o._dye))
        col = int(np.clip((o._centres[0, 0] + 0.5) * orb._FIELD, 0, orb._FIELD - 1))
        row = int(np.clip((o._centres[0, 1] + 0.5) * orb._FIELD, 0, orb._FIELD - 1))
        assert srgb[row, col] == pytest.approx(np.array(colour[:3]), abs=0.06)

    def test_unfed_dye_settles_to_the_background(self):
        # Silenced sources feed nothing, so the colour they painted fades out.
        o = _orb(seed=24)
        o.set_params(orb.OrbParams(emphasis=(0.0,) * len(_BLOBS)))
        for _ in range(60):
            o.advance(0.1)
        background = _orb(blobs=[], seed=24)._dye
        assert np.allclose(o._dye, background, atol=1e-3)


class TestIdentity:
    def test_same_seed_is_the_same_orb(self):
        a, b = _orb(seed=42), _orb(seed=42)
        for o in (a, b):
            for _ in range(5):
                o.advance(0.25)
        assert np.array_equal(a._dye, b._dye)

    def test_different_seeds_are_different_orbs(self):
        a, b = _orb(seed=1), _orb(seed=2)
        assert not np.array_equal(a._dye, b._dye)

    def test_rest_frame_ignores_the_flow(self):
        # Reduced motion shows the designed t=0 frame however long the orb has run.
        o = _orb(seed=9)
        rest_before = _rest_frame(o)
        for _ in range(20):
            o.advance(0.1)
        assert _rest_frame(o) == rest_before


class TestMotion:
    def test_advancing_changes_the_field(self):
        o = _orb()
        before = o._dye.copy()
        o.advance(2.0)
        assert not np.array_equal(before, o._dye)

    def test_centres_never_escape_the_rim(self):
        o = _orb(seed=11)
        for _ in range(1200):
            o.advance(0.1)
        radii = np.hypot(o._centres[:, 0], o._centres[:, 1])
        assert np.all(radii <= orb._RIM_BOUND + 1e-9)

    def test_flow_is_divergence_free(self):
        # The fluid look rests on the field having no sinks; the rim confinement
        # multiplies the potential, so it must not break this either.
        o = _orb(seed=13)
        eps = 1e-4  # wide enough that single-precision gradients don't drown the diff
        pts = np.array([[0.05, -0.1], [0.2, 0.15], [-0.3, 0.2], [0.05, 0.36]])
        for p in pts:

            def vel(q):
                return o._velocity(1.0, orb._flow_stencil(q[None]))[0]

            dx, dy = np.array([eps, 0.0]), np.array([0.0, eps])
            dvx = (vel(p + dx)[0] - vel(p - dx)[0]) / (2 * eps)
            dvy = (vel(p + dy)[1] - vel(p - dy)[1]) / (2 * eps)
            assert abs(dvx + dvy) < 1e-3

    def test_a_stalled_frame_cannot_teleport(self):
        # One clamped step, however huge the reported delta.
        a, b = _orb(seed=21), _orb(seed=21)
        a.advance(3600.0)
        b.advance(orb._DT_MAX)
        assert np.array_equal(a._centres, b._centres)

    def test_empty_orb_advances_without_error(self):
        # An emitterless orb's dye must hold the background, not drift.
        o = _orb(blobs=[])
        before = o._dye.copy()
        o.advance(1.0)
        assert np.array_equal(o._dye, before)


class TestDraw:
    def _draw(self, theme, *, motion, size=48):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
        o = orb.Orb(
            background=theme.palette.background,
            blobs=orb_blobs(theme),
            rim=theme.palette.border,
            glow=theme.palette.accent,
            seed=orb_seed(theme),
        )
        o.advance(1.0)
        o.draw(cairo.Context(surface), size=size, motion=motion)
        return surface

    def test_every_preset_draws(self):
        # A disc: opaque at the centre, clipped away at the corners (alpha is byte 3).
        for name in list_presets():
            for motion in (False, True):
                surface = self._draw(preset_theme(name), motion=motion)
                surface.flush()
                data, stride = surface.get_data(), surface.get_stride()
                assert data[24 * stride + 24 * 4 + 3] == 255
                assert data[3] == 0

    def test_byte_order_is_correct(self):
        # ARGB32 is native-endian premultiplied, BGRA on little-endian. A red orb must
        # come out red, not blue: a channel swap draws without any error.
        o = orb.Orb(
            background=(0.9, 0.05, 0.05, 1.0),
            blobs=[],
            rim=(1.0, 1.0, 1.0, 0.2),
            glow=(0.9, 0.05, 0.05, 1.0),
            seed=0,
        )
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 40, 40)
        o.draw(cairo.Context(surface), size=40)
        surface.flush()
        buf = surface.get_data()
        stride = surface.get_stride()
        # The centre pixel, BGRA in memory.
        i = 20 * stride + 20 * 4
        b, r = buf[i], buf[i + 2]
        assert r > 150 and b < 80  # reads red, not blue

    def test_display_material_is_opaque(self):
        lab = np.zeros((2, 3, 3), dtype=np.float32)

        pixels = np.frombuffer(orb._display_bytes(lab), dtype=np.uint8).reshape(2, 3, 4)

        assert np.all(pixels[..., 3] == 255)
