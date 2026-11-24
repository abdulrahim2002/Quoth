# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Theme orb: a palette drawn as a live little convection cell.

Pure numpy and Cairo, with no Theme and no GTK: callers pass the colours, weights
and seed they own, and `pill_view` maps a Theme onto them. Blob centres ride a
seeded, divergence-free curl-noise flow, so they never clump, and paint through a
breathing domain warp with metaball falloffs, so approaching regions merge.
Colours blend in OKLab, where a numeric average is a perceptual one. What shows is
a persistent dye field the flow transports each step, so colour stretches into
filaments; the rest frame (reduced motion, t=0) is the static painting `_paint`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import cairo
import numpy as np

from quoth.color import (
    linear_to_srgb,
    oklab_to_linear,
    oklab_to_srgb,
    srgb_to_oklab,
)

Color = tuple[float, float, float, float]


@dataclass(frozen=True)
class OrbParams:
    """Live modulation of an orb, as dimensionless multiples of the tuned look.

    Neutral (all 1.0, no emphasis) is exactly the ambient orb.

    `emphasis` multiplies each blob's weight, one entry per blob in caller order,
    so a state can favour a colour role without the orb knowing role names.
    """

    time_scale: float = 1.0  # pace of the flow: field evolution, warp, breathing
    drift_scale: float = 1.0  # extra factor on centre advection speed
    swell: float = 1.0  # scales the weight-breathing depth
    glow: float = 1.0  # scales the glow layer opacities
    # Diameter as a fraction of the drawing box, never above 1.0, so a director can
    # rest the orb small and swell it with the voice.
    scale: float = 1.0
    emphasis: tuple[float, ...] | None = None


# Linear light straight to 8-bit sRGB through one table lookup: exact to within
# one 8-bit step, where the piecewise transfer curve costs a pow over the field.
# `linear_to_srgb` stays the reference; a test pins the table to it.
_ENCODE_LUT = (
    np.clip(linear_to_srgb(np.linspace(0.0, 1.0, 4096)), 0.0, 1.0) * 255.0 + 0.5
).astype(np.uint8)


# --- gamut ------------------------------------------------------------------
# Ottosson's adaptive-L0 projection: an out-of-gamut colour travels toward a grey
# whose lightness leans toward 0.5 as chroma grows, so chroma compresses at constant
# hue. The boundary crossing is found by bisection.

_GAMUT_ALPHA = 0.05
_GAMUT_STEPS = 12
_GAMUT_EPS = 1e-4


def _in_gamut(lab: np.ndarray) -> np.ndarray:
    rgb = oklab_to_linear(lab)
    return np.all((rgb >= -_GAMUT_EPS) & (rgb <= 1.0 + _GAMUT_EPS), axis=-1)


def _gamut_project(lab: np.ndarray) -> np.ndarray:
    """Map every out-of-gamut OKLab colour to the sRGB boundary at constant hue.

    Returns the input array itself, not a copy, when everything already fits.
    """
    flat = lab.reshape(-1, 3)
    outside = ~_in_gamut(flat)
    if not outside.any():
        return lab
    bad = flat[outside]
    ld = bad[:, 0] - 0.5
    chroma = np.hypot(bad[:, 1], bad[:, 2])
    e1 = 0.5 + np.abs(ld) + _GAMUT_ALPHA * chroma
    l0 = 0.5 * (
        1.0 + np.sign(ld) * (e1 - np.sqrt(np.maximum(e1 * e1 - 2 * np.abs(ld), 0.0)))
    )
    anchor = np.zeros_like(bad)
    anchor[:, 0] = l0
    span = bad - anchor
    lo = np.zeros(len(bad), dtype=lab.dtype)
    hi = np.ones(len(bad), dtype=lab.dtype)
    for _ in range(_GAMUT_STEPS):
        mid = (lo + hi) / 2.0
        inside = _in_gamut(anchor + mid[:, None] * span)
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    fixed = flat.copy()
    fixed[outside] = anchor + lo[:, None] * span
    return fixed.reshape(lab.shape)


def _restore_chroma(
    blob: np.ndarray, target: np.ndarray, own: np.ndarray
) -> np.ndarray:
    """Pull the mix's chroma toward its parents', gated and in proportion.

    Averaging opposing hues shrinks the a/b resultant toward grey; scaling it
    back to the weighted parent chroma keeps mixes vivid. The gate trusts the
    resultant's hue only while it is strong (cancelled hues settle to neutral
    instead of flickering), and the boost applies only where colours actually
    mix, so a pixel one colour owns shows exactly that colour.
    """
    resultant = np.hypot(blob[..., 1], blob[..., 2])
    gate = resultant / (_GATE_SOFT + resultant)
    mixedness = 1.0 - own
    chroma = (resultant + gate * (target - resultant)) * (1.0 + _MIX_BOOST * mixedness)
    scale = (chroma / np.maximum(resultant, 1e-9))[..., None]
    return np.concatenate([blob[..., :1], blob[..., 1:] * scale], axis=-1)


# --- look constants ---------------------------------------------------------
_FIELD = 48  # px the colour field is computed at, scaled by Cairo to the orb size
# Homes are seeded along the golden angle, the most irrational turn and so the
# evenest spread on a disc: the rest frame, and where every orb's flow starts.
_GOLDEN = math.pi * (3.0 - math.sqrt(5.0))
_HOME_RADIUS = 0.30  # home ring radius, as a fraction of the orb
_BLOB_RADIUS = 0.46  # falloff support: past this distance a blob paints nothing
# How strongly the background resists the blobs: high enough that the theme's dark
# fill shows as the orb's base, or a light-on-dark palette reads as washed fog.
_BG_WEIGHT = 2.2
_MIX_BOOST = 0.35  # extra chroma in proportion to how mixed a pixel is
_GATE_SOFT = 0.03  # resultant chroma below which cancelling hues settle to neutral

# --- motion constants -------------------------------------------------------
_LATTICE = 4  # curl-noise lattice cells across the orb; vortices span ~1/_LATTICE
_ROT_RATE = 0.4  # rad/s scale of per-cell gradient rotation (how fast the flow evolves)
_DRIFT_SPEED = 0.018  # velocity scale, orb fractions per second
_RIM_BOUND = 0.40  # centres stay inside this radius
_RAMP_WIDTH = 0.16  # band inside the bound over which the flow eases to rim-tangent
_DT_MAX = 0.1  # clamp on a frame delta so a stalled frame clock cannot teleport blobs
_WARP_RES = 2  # warp Perlin lattice cells across the field
_WARP_AMP = 0.16  # warp displacement, orb fractions
_BREATHE_AMP = 0.25  # weight breathing depth

# --- dye constants ----------------------------------------------------------
_DYE_SPEED = 0.10  # advection gain: the flow paints faster than the emitters amble
_INJECT = 3.5  # per second: how hard a source pulls the dye toward its own colour
_FADE = 1.5  # per second: how fast unfed dye settles back to the background


class _FlowStencil(NamedTuple):
    """The position-derived half of a flow evaluation.

    Cell fractions, fade weights and their derivatives, flat corner indices into the
    gradient lattice, and the rim confinement terms. Positions that never move (the
    pixel grid) build theirs once; only the lattice rotation is per-frame.
    """

    fx: np.ndarray
    fy: np.ndarray
    fx1: np.ndarray
    fy1: np.ndarray
    u: np.ndarray
    v: np.ndarray
    du: np.ndarray
    dv: np.ndarray
    corners: np.ndarray  # (4, N) flat indices: cells 00, 10, 01, 11
    ramp: np.ndarray
    confine_x: np.ndarray  # psi coefficients of the tangent-steering term
    confine_y: np.ndarray


def _flow_stencil(pts: np.ndarray) -> _FlowStencil:
    g = _LATTICE
    p = (pts + 0.5) * g
    pi = np.floor(p).astype(np.intp)
    # Keep the points' dtype; integer operands would promote the stencil to float64.
    frac = (p - pi).astype(p.dtype)
    fx, fy = frac[:, 0], frac[:, 1]
    i0, j0 = pi[:, 0] % g, pi[:, 1] % g
    i1, j1 = (i0 + 1) % g, (j0 + 1) % g
    corners = np.stack([i0 * g + j0, i1 * g + j0, i0 * g + j1, i1 * g + j1])
    r = np.hypot(pts[:, 0], pts[:, 1])
    x = np.clip((_RIM_BOUND - r) / _RAMP_WIDTH, 0.0, 1.0)
    dramp = np.where(
        r > 1e-9,
        _fade_deriv(x) * (-1.0 / _RAMP_WIDTH) / np.maximum(r, 1e-9),
        0.0,
    )
    return _FlowStencil(
        fx=fx,
        fy=fy,
        fx1=fx - 1.0,
        fy1=fy - 1.0,
        u=_fade(fx),
        v=_fade(fy),
        du=_fade_deriv(fx),
        dv=_fade_deriv(fy),
        corners=corners,
        ramp=_fade(x),
        confine_x=dramp * pts[:, 0],
        confine_y=dramp * pts[:, 1],
    )


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _fade_deriv(t: np.ndarray) -> np.ndarray:
    """d/dt of _fade: 30t^2(t-1)^2."""
    return 30.0 * t * t * (t - 1.0) ** 2


def _perlin_grid(rng: np.random.Generator, size: int, res: int) -> np.ndarray:
    """Return a (size, size) Perlin field of res x res cells, values ~[-1, 1]."""
    d = size // res
    grid = np.mgrid[0:size, 0:size].astype(np.float64).transpose(1, 2, 0) / d % 1.0
    angles = 2 * np.pi * rng.random((res + 1, res + 1))
    grads = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = grads[:-1, :-1].repeat(d, 0).repeat(d, 1)
    g10 = grads[1:, :-1].repeat(d, 0).repeat(d, 1)
    g01 = grads[:-1, 1:].repeat(d, 0).repeat(d, 1)
    g11 = grads[1:, 1:].repeat(d, 0).repeat(d, 1)
    gx, gy = grid[..., 0], grid[..., 1]
    n00 = gx * g00[..., 0] + gy * g00[..., 1]
    n10 = (gx - 1) * g10[..., 0] + gy * g10[..., 1]
    n01 = gx * g01[..., 0] + (gy - 1) * g01[..., 1]
    n11 = (gx - 1) * g11[..., 0] + (gy - 1) * g11[..., 1]
    u, v = _fade(gx), _fade(gy)
    n0 = n00 + u * (n10 - n00)
    n1 = n01 + u * (n11 - n01)
    return math.sqrt(2) * (n0 + v * (n1 - n0))


class Orb:
    """One palette as a drifting, breathing, seeded convection cell.

    `advance` integrates the flow with real elapsed seconds; `draw` renders the
    current frame, or the deterministic t=0 rest frame when motion is off. The
    same seed and colours always reproduce the same orb.
    """

    def __init__(
        self,
        *,
        background: Color,
        blobs: list[tuple[Color, float]],
        rim: Color,
        glow: Color,
        seed: int,
    ) -> None:
        self._rim = rim
        self._glow = glow
        self._bg_lab = srgb_to_oklab(np.array(background[:3], dtype=np.float32))
        n = len(blobs)
        # A blob's own alpha folds into its weight, so a faint colour washes in
        # faintly.
        self._weights = np.array([w * rgba[3] for rgba, w in blobs], dtype=np.float32)
        self._labs = np.array(
            [srgb_to_oklab(np.array(rgba[:3], dtype=np.float32)) for rgba, _ in blobs],
            dtype=np.float32,
        ).reshape(n, 3)
        self._chromas = np.hypot(self._labs[:, 1], self._labs[:, 2])

        ang = np.arange(n) * _GOLDEN
        self._homes = (
            _HOME_RADIUS * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        ).astype(np.float32)
        self._centres = self._homes.copy()
        self._t = 0.0
        self._params = OrbParams()

        flow, warp, breathe = (
            np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
        )
        self._grad_angle = (2 * np.pi * flow.random((_LATTICE, _LATTICE))).astype(
            np.float32
        )
        # Zero-mean per-cell rotation rates: the flow evolves with no net spin, and
        # a continuous distribution makes the rates pairwise irrational, so the
        # field is quasi-periodic and never visibly loops.
        self._grad_rate = (
            _ROT_RATE * flow.normal(0.0, 1.0, (_LATTICE, _LATTICE))
        ).astype(np.float32)

        self._warp = np.stack(
            [_perlin_grid(warp, _FIELD, _WARP_RES) * _WARP_AMP for _ in range(4)]
        ).astype(np.float32)
        self._warp_freq = warp.uniform(0.06, 0.16, 2)
        self._breathe_freq = breathe.uniform(0.10, 0.30, n).astype(np.float32)
        self._breathe_phase = breathe.uniform(0.0, 2 * np.pi, n).astype(np.float32)

        # Pixel centres in orb fractions, origin at the middle; the flat list is
        # the same grid as the flow sees it, for transporting the dye.
        span = ((np.arange(_FIELD) + 0.5) / _FIELD - 0.5).astype(np.float32)
        self._xs, self._ys = np.meshgrid(span, span)
        self._grid_pts = np.stack([self._xs.ravel(), self._ys.ravel()], axis=1)
        self._grid_stencil = _flow_stencil(self._grid_pts)
        self._dye = self._paint(self._centres, self._t)

    # --- motion ---------------------------------------------------------------

    def set_params(self, params: OrbParams) -> None:
        """Adopt the modulation for subsequent advances and draws."""
        self._params = params

    def seek(self, t: float) -> None:
        """Jump the flow's clock to `t` without integrating.

        Everything driven purely by time (warp crossfade, breathing, lattice
        rotation) lands exactly where it would have; centre positions are a path
        integral and stay put, and the dye resets to the fresh painting at the new
        instant, since a jumped clock has no transport history.
        """
        self._t = max(t, 0.0)
        self._dye = self._paint(self._centres, self._t)

    def advance(self, dt: float) -> None:
        """Integrate the flow by `dt` seconds, clamped so a stall cannot teleport.

        The emitters ride the flow, then the dye is transported, fed and settled.
        """
        dt = min(max(dt, 0.0), _DT_MAX) * self._params.time_scale
        self._t += dt
        if not len(self._centres):
            return
        v = self._velocity(self._t, _flow_stencil(self._centres))
        centres = self._centres + v * (dt * _DRIFT_SPEED * self._params.drift_scale)
        # The rim ramp already turns the flow tangent; this clamp only catches the
        # residual of a finite Euler step crossing the line.
        r = np.hypot(centres[:, 0], centres[:, 1])
        over = r > _RIM_BOUND
        if over.any():
            centres[over] *= (_RIM_BOUND / r[over])[:, None]
        self._centres = centres
        self._transport(dt)

    def _velocity(self, t: float, st: _FlowStencil) -> np.ndarray:
        """Return the curl of the rim-confined stream function at the stencil's points.

        v = curl(A * psi) = A * curl(psi) + psi * curl(A): confinement multiplies
        the potential, not the velocity, which keeps the field divergence-free
        (fluid, sink-free) while easing it tangent to the rim. Trig runs on the
        16-cell lattice and is gathered from there, never per point.
        """
        ang = self._grad_angle + t * self._grad_rate
        gx = np.cos(ang).ravel().take(st.corners)
        gy = np.sin(ang).ravel().take(st.corners)

        n00 = gx[0] * st.fx + gy[0] * st.fy
        n10 = gx[1] * st.fx1 + gy[1] * st.fy
        n01 = gx[2] * st.fx + gy[2] * st.fy1
        n11 = gx[3] * st.fx1 + gy[3] * st.fy1

        nx0 = n00 + st.u * (n10 - n00)
        nx1 = n01 + st.u * (n11 - n01)
        psi = nx0 + st.v * (nx1 - nx0)
        dnx0 = gx[0] + st.u * (gx[1] - gx[0]) + st.du * (n10 - n00)
        dnx1 = gx[2] + st.u * (gx[3] - gx[2]) + st.du * (n11 - n01)
        dpsidx = (dnx0 + st.v * (dnx1 - dnx0)) * _LATTICE
        dgy0 = gy[0] + st.u * (gy[1] - gy[0])
        dgy1 = gy[2] + st.u * (gy[3] - gy[2])
        dpsidy = (dgy0 + st.v * (dgy1 - dgy0) + st.dv * (nx1 - nx0)) * _LATTICE

        vx = st.ramp * dpsidy + psi * st.confine_y
        vy = -st.ramp * dpsidx - psi * st.confine_x
        return np.stack([vx, vy], axis=1)

    def _transport(self, dt: float) -> None:
        """Step the dye: advect along the flow, feed from the sources, settle the unfed.

        Every term is dt-proportional, so the marbling rate is frame-rate independent
        and rides `time_scale` like the rest of the physics.
        """
        vel = self._velocity(self._t, self._grid_stencil)
        self._dye = self._sample(self._grid_pts - vel * (dt * _DYE_SPEED))
        colour, presence = self._sources(self._centres, self._t)
        pull = np.clip(dt * _INJECT, 0.0, 1.0) * presence
        self._dye += pull * (colour - self._dye)
        settle = np.clip(dt * _FADE, 0.0, 1.0) * (1.0 - presence)
        self._dye += settle * (self._bg_lab - self._dye)

    def _sample(self, pts: np.ndarray) -> np.ndarray:
        """Sample the dye bilinearly at centred coordinates, clamped to the field.

        The confined flow rarely backtraces past the rim; the clamp catches the rest.
        Corners are gathered with flat takes on the flattened field, much cheaper than
        advanced indexing on the cube.
        """
        col = np.clip((pts[:, 0] + 0.5) * _FIELD - 0.5, 0.0, _FIELD - 1.0)
        row = np.clip((pts[:, 1] + 0.5) * _FIELD - 0.5, 0.0, _FIELD - 1.0)
        c0 = np.minimum(col.astype(np.intp), _FIELD - 2)
        r0 = np.minimum(row.astype(np.intp), _FIELD - 2)
        tc = (col - c0).astype(col.dtype)[:, None]
        tr = (row - r0).astype(row.dtype)[:, None]
        flat = self._dye.reshape(-1, 3)
        i00 = r0 * _FIELD + c0
        top = flat.take(i00, axis=0) * (1.0 - tc) + flat.take(i00 + 1, axis=0) * tc
        i10 = i00 + _FIELD
        bot = flat.take(i10, axis=0) * (1.0 - tc) + flat.take(i10 + 1, axis=0) * tc
        return (top * (1.0 - tr) + bot * tr).reshape(_FIELD, _FIELD, 3)

    # --- rendering --------------------------------------------------------------

    def _sources(self, centres: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the palette's current per-pixel source colour (OKLab) and presence.

        Presence is measured against the background. The painting and the dye
        injection both draw from here, so they can never disagree.
        """
        # The warp crossfades two seeded displacement fields on incommensurate
        # frequencies, so region boundaries writhe without visibly looping.
        c1, s2 = math.cos(self._warp_freq[0] * t), math.sin(self._warp_freq[1] * t)
        xw = self._xs + c1 * self._warp[0] + s2 * self._warp[2]
        yw = self._ys + c1 * self._warp[1] + s2 * self._warp[3]

        n = len(centres)

        # Breathing is anchored so t=0 is exactly the base weights, with the random
        # phase desynchronising the blobs. The floor keeps a deep swell from swinging
        # a weight negative.
        p = self._params
        phase = self._breathe_freq * t + self._breathe_phase
        amp = _BREATHE_AMP * p.swell
        breathe = 1.0 + amp * (np.sin(phase) - np.sin(self._breathe_phase))
        breathe = np.maximum(breathe, 0.05)
        if p.emphasis is not None:
            breathe = breathe * np.asarray(p.emphasis, dtype=breathe.dtype)
        acc = np.zeros((_FIELD, _FIELD, 3), dtype=np.float32)
        csum = np.zeros((_FIELD, _FIELD), dtype=np.float32)
        # Seeded tiny, so bare pixels divide safely.
        fpsum = np.full((_FIELD, _FIELD), 1e-9, dtype=np.float32)
        fpmax = np.zeros((_FIELD, _FIELD), dtype=np.float32)
        total = np.zeros((_FIELD, _FIELD), dtype=np.float32)
        r2 = _BLOB_RADIUS * _BLOB_RADIUS
        for i in range(n):
            d2 = (xw - centres[i, 0]) ** 2 + (yw - centres[i, 1]) ** 2
            # Wyvill-style compact falloff: smooth, zero past the support radius,
            # and additive across blobs so approaching regions neck and merge.
            # Cubed twice over, as multiplies: once for the smooth falloff, once
            # more for colour-ownership sharpening (tighter zones).
            w = np.clip(1.0 - d2 / r2, 0.0, None)
            f = self._weights[i] * breathe[i] * (w * w * w)
            fp = f * f * f
            acc += fp[..., None] * self._labs[i]
            csum += fp * self._chromas[i]
            fpsum += fp
            fpmax = np.maximum(fpmax, fp)
            total += f

        blob = acc / fpsum[..., None]
        blob = _restore_chroma(blob, csum / fpsum, fpmax / fpsum)
        presence = (total / (total + _BG_WEIGHT))[..., None]
        return blob, presence

    def _paint(self, centres: np.ndarray, t: float) -> np.ndarray:
        """Return the static picture the sources would paint from scratch, in OKLab.

        It is the rest frame and the dye's no-history state.
        """
        colour, presence = self._sources(centres, t)
        return self._bg_lab * (1.0 - presence) + colour * presence

    def _paint_srgb(self, centres: np.ndarray, t: float) -> np.ndarray:
        """Return the painting through the exact transfer curve, as sRGB.

        Only tests call it, to check the painted colours through that curve.
        """
        return oklab_to_srgb(_gamut_project(self._paint(centres, t)))

    def draw(self, cr, *, size: float, motion: bool = True) -> None:
        """Draw the orb into a `size` x `size` box at the current origin.

        With motion off this is the deterministic t=0 rest frame. Params apply either
        way, since they carry the voice and state response that reduced motion keeps.
        """
        # The lower clamp keeps the orb a visible disc even with degenerate params.
        eff = size * min(max(self._params.scale, 0.3), 1.0)
        lab = self._dye if motion else self._paint(self._homes, 0.0)
        # Cairo holds a raw pointer to the buffer, not a reference, so it must stay
        # alive through the paint.
        buf = _display_bytes(_gamut_project(lab))
        surface = cairo.ImageSurface.create_for_data(
            buf, cairo.FORMAT_ARGB32, _FIELD, _FIELD
        )
        radius = eff / 2.0

        cr.save()
        cr.translate((size - eff) / 2.0, (size - eff) / 2.0)
        cr.save()
        cr.arc(radius, radius, radius, 0, 2 * math.pi)
        cr.clip()
        cr.scale(eff / _FIELD, eff / _FIELD)
        pattern = cairo.SurfacePattern(surface)
        pattern.set_filter(cairo.FILTER_GOOD)  # bilinear: the field is the gradient
        cr.set_source(pattern)
        cr.paint()
        cr.restore()

        _draw_glow(cr, eff, self._rim, self._glow, self._params.glow)
        cr.restore()


def _display_bytes(lab: np.ndarray) -> bytearray:
    """Return an OKLab field as opaque Cairo ARGB32 bytes in BGRA order.

    BGRA is ARGB32's native-endian layout on little-endian machines. Alpha is
    always opaque because the circular clip shapes the edge.
    """
    linear = np.clip(oklab_to_linear(lab), 0.0, 1.0)
    rgb = _ENCODE_LUT.take((linear * 4095.0 + 0.5).astype(np.intp))
    argb = np.empty((lab.shape[0], lab.shape[1], 4), dtype=np.uint8)
    argb[..., 0] = rgb[..., 2]
    argb[..., 1] = rgb[..., 1]
    argb[..., 2] = rgb[..., 0]
    argb[..., 3] = 255
    return bytearray(argb.tobytes())


def _draw_glow(cr, size: float, rim: Color, glow: Color, strength: float) -> None:
    """Light the orb rather than shade it, with a Fresnel rim and a core bloom.

    The rim brightens toward the edge, screen-blended so it reads as emitted light,
    and the off-centre bloom is `glow`, the theme's accent, so the orb shines in its
    own hue. `strength` scales both, clamped to stay opacities; the hairline is `rim`.
    """
    r = size / 2.0
    rim_a = min(1.0, 0.14 * strength)
    core_a = min(1.0, 0.30 * strength)

    cr.save()
    cr.arc(r, r, r, 0, 2 * math.pi)
    cr.clip()

    rim_glow = cairo.RadialGradient(r, r, r * 0.55, r, r, r)
    rim_glow.add_color_stop_rgba(0.0, 1, 1, 1, 0.0)
    rim_glow.add_color_stop_rgba(0.82, 1, 1, 1, 0.0)
    rim_glow.add_color_stop_rgba(1.0, 1, 1, 1, rim_a)
    cr.set_operator(cairo.OPERATOR_SCREEN)  # add light, never darken
    cr.set_source(rim_glow)
    cr.paint()

    core = cairo.RadialGradient(r * 0.7, r * 0.62, 0, r * 0.7, r * 0.62, r * 0.7)
    core.add_color_stop_rgba(0.0, *glow[:3], core_a)
    core.add_color_stop_rgba(1.0, *glow[:3], 0.0)
    cr.set_source(core)
    cr.paint()
    cr.restore()

    cr.arc(r, r, r - 0.5, 0, 2 * math.pi)
    cr.set_source_rgba(*rim)
    cr.set_line_width(1.0)
    cr.stroke()
