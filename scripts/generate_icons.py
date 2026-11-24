# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Generate the Quoth icon pack from the master mark SVG.

Single source of truth: `data/icons/src/dev.quoth.Quoth.svg`, the bare Q-raven mark in
solid `#000000` on transparent (load-bearing: the recolouring rewrites that literal).
The transcribing frames come from `data/icons/src/quoth-loader.svg`. Everything else
is derived, so re-run this after changing the master and commit the result.

Writes the freedesktop hicolor layout under `data/icons/hicolor/` that the desktop
entry and metainfo reference by app id. The app icon is the white mark, inset to
`GLYPH_FRAC` of the canvas, on an ink (#0E0E11) rounded-square tile that is lit from
the top and inset to `TILE_FRAC`, with no baked-in shadow. The tray states are one Q
at one size, so the panel never sees it resize or shift: idle (`-symbolic`) and
transcribing (`-transcribing-symbolic`, then `-transcribing-pulse-N-symbolic` loader
frames) are recolourable symbolics; recording (`-recording` and its pulse frames)
floods the ring's counter red. It is the one coloured state, since the host would
recolour a symbolic flat.

Run: `uv run --group icons python scripts/generate_icons.py`
"""

import io
import itertools
import math
import re
from pathlib import Path

import cairosvg
from PIL import Image, ImageChops

APP_ID = "dev.quoth.Quoth"
INK = "#0E0E11"  # brand tile (app/dock/store), shared with quoth.dev
# The tile's top edge. Shading the front from a lighter top keeps the dark tile's edge
# visible on a dark dock, where a flat ink tile disappears.
INK_TOP = "#2A2A31"
# Transcribing frames sampled across the loader's whole loop, spaced by equal motion
# (see _even_motion_phases). 16 at the tray's 200ms tick covers the 3180ms loop, so the
# cycle plays the whole gesture: both beats, and the peel of each wing in turn.
TRANSCRIBE_FRAMES = 16
TRANSCRIBE_PHASE = 0.08  # mid-unfurl, in loop fractions
RECORD_RED = "#E5484D"  # recording-state tray tile: the one coloured status
# Neutral grey for hosts that render symbolic icons literally;
# recolouring hosts (COSMIC, GTK) override the fill anyway.
SYMBOLIC_FILL = "#bebebe"

RASTER_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
# The system tray only ever displays small sizes; recording-pulse frames (a tray-only
# animation) are emitted at just these to keep the committed asset count down.
TRAY_SIZES = (16, 24, 32, 48, 64)
# The app tile's side, as a fraction of the canvas: the square of Flathub's icon grid,
# so the icon sits at the same size as its neighbours in a store or dock.
TILE_FRAC = 0.805
CORNER_FRAC = 0.225  # rounded-square corner radius, as a fraction of the tile
GLYPH_FRAC = 0.50  # app-tile mark, as a fraction of the canvas
# Symbolic marks fill their canvas edge to edge: a tray icon sits beside stock
# symbolics that do the same, and any inset here reads as Quoth's icon being smaller
# than its neighbours rather than as breathing room. The tray applies its own padding.
SYMBOLIC_FRAC = 1.0
COUNTER_OVERLAP = 1.03  # recording fill, as a fraction of the ring's inner radius
_RENDER_PX = 1024  # resolution the mark is rendered at to measure its ink bbox

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MASTER = _REPO_ROOT / "data" / "icons" / "src" / f"{APP_ID}.svg"
_LOADER = _REPO_ROOT / "data" / "icons" / "src" / "quoth-loader.svg"
_HICOLOR = _REPO_ROOT / "data" / "icons" / "hicolor"


def _cubic_bezier(x1: float, y1: float, x2: float, y2: float):
    """Return a CSS cubic-bezier easing: solve the curve's x for t, then read y."""

    def bez(a: float, b: float, t: float) -> float:
        u = 1.0 - t
        return 3 * u * u * t * a + 3 * u * t * t * b + t * t * t

    def ease(t: float) -> float:
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if bez(x1, x2, mid) < t:
                lo = mid
            else:
                hi = mid
        return bez(y1, y2, (lo + hi) / 2)

    return ease


def _keyframes(css: str, name: str) -> list[tuple[float, float, object]]:
    """Read one dash-offset keyframe track as (stop, offset, easing-into-next)."""
    block = re.search(r"@keyframes\s+" + name + r"\s*\{(.*?)\n\s*\}", css, re.S)
    if block is None:
        raise ValueError(f"loader asset has no @keyframes {name}")
    stops = []
    for m in re.finditer(
        r"([\d.]+)%\s*\{\s*stroke-dashoffset:\s*(-?[\d.]+);"
        r"\s*animation-timing-function:\s*([^;]+);",
        block.group(1),
    ):
        fn = m.group(3).strip()
        if fn == "linear":
            ease = lambda t: t  # noqa: E731
        else:
            ease = _cubic_bezier(*(float(v) for v in re.findall(r"-?[\d.]+", fn)))
        stops.append((float(m.group(1)) / 100, float(m.group(2)), ease))
    if len(stops) < 2:
        raise ValueError(f"loader keyframes {name} did not parse ({len(stops)} stops)")
    return stops


def _dash_offset(stops, t: float) -> float:
    for (p0, v0, ease), (p1, v1, _) in itertools.pairwise(stops):
        if p0 <= t <= p1:
            span = p1 - p0
            return v0 + (v1 - v0) * ease(0.0 if span <= 0 else (t - p0) / span)
    return stops[-1][1]


def _loader_frame(svg: str, tracks: dict, t: float, transform: str = "") -> str:
    """Return the loader with time `t`'s dash offsets baked in, as a symbolic icon.

    The wings are strokes rather than fills. Symbolic recolouring masks the host's
    colour by the icon's alpha, so stroked ink recolours the same as filled ink.
    Without a `transform` this keeps the asset's own viewBox, which is what the frame
    extents are measured in.
    """
    frame = re.sub(r"<style.*?</style>", "", svg, flags=re.S)
    for cls, track in (("wa", "alt_a"), ("wb", "alt_b")):
        offset = _dash_offset(tracks[track], t)
        frame = frame.replace(
            f'class="{cls}"', f'style="stroke-dashoffset:{offset:.3f}"'
        )
    # Supply the colour currentColor took from the stylesheet dropped above.
    if not transform:
        return frame.replace("<svg ", f'<svg color="{SYMBOLIC_FILL}" ', 1)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" '
        f'width="16" height="16" color="{SYMBOLIC_FILL}">'
        f'<g transform="{transform}">{_inner(frame)}</g></svg>\n'
    )


def _frame_alpha(svg: str, tracks: dict, t: float, px: int = 96) -> Image.Image:
    """Return one loader frame's alpha coverage, for comparing frames to each other."""
    png = cairosvg.svg2png(
        bytestring=_loader_frame(svg, tracks, t).encode(), output_width=px
    )
    return Image.open(io.BytesIO(png)).convert("RGBA").getchannel("A")


def _even_motion_phases(svg: str, tracks: dict, count: int, res: int = 256) -> list:
    """Return `count` loop phases spaced by equal motion rather than equal time.

    The phases are rotated so frame 0 sits at or after `TRANSCRIBE_PHASE`: frame 0
    doubles as the static icon, and the finished Q the gesture settles on differs from
    idle by stroke weight alone. Rotating relabels the cycle without changing any step.
    Raises ValueError for an odd `count`.
    """
    # The loader is heavily eased, so equal time steps would stall on its plateaus and
    # read as a hung app; frames are spaced by how much each one changes instead. The
    # loop is one gesture twice with the wings swapped, so one beat is sampled and
    # mirrored, keeping both peels equally smooth.
    if count % 2:
        raise ValueError(f"frame count must be even to mirror a beat, got {count}")
    half = count // 2
    xs = [i * 0.5 / res for i in range(res + 1)]
    frames = [_frame_alpha(svg, tracks, x) for x in xs]
    cumulative = [0.0]
    for i in range(res):
        histogram = ImageChops.difference(frames[i], frames[i + 1]).histogram()
        cumulative.append(cumulative[-1] + sum(i * n for i, n in enumerate(histogram)))
    total = cumulative[-1]
    beat = []
    for j in range(half):
        target = total * j / half
        i = 0
        while i < res and cumulative[i + 1] < target:
            i += 1
        span = cumulative[i + 1] - cumulative[i]
        share = 0.0 if span <= 0 else (target - cumulative[i]) / span
        beat.append(xs[i] + (xs[i + 1] - xs[i]) * share)
    phases = beat + [(p + 0.5) % 1.0 for p in beat]
    start = next(i for i, p in enumerate(phases) if p >= TRANSCRIBE_PHASE)
    return phases[start:] + phases[:start]


def _loader_extent(svg: str, tracks: dict, phases) -> tuple[float, float, float, float]:
    """Return the union of every frame's ink, in the loader's own user units.

    One extent for the whole loop, because each frame is drawn through the same
    transform: fitting frames to their own bounds would rescale the mark as the ring
    sweeps away, so it would breathe in size rather than draw itself.
    """
    vb = re.search(r'viewBox="([\d.\s-]+)"', svg).group(1).split()
    vx, vy, vw, _vh = (float(v) for v in vb)
    box = None
    for t in phases:
        png = cairosvg.svg2png(
            bytestring=_loader_frame(svg, tracks, t).encode(), output_width=_RENDER_PX
        )
        found = Image.open(io.BytesIO(png)).convert("RGBA").getbbox()
        if found is None:
            continue
        box = (
            found
            if box is None
            else (
                min(box[0], found[0]),
                min(box[1], found[1]),
                max(box[2], found[2]),
                max(box[3], found[3]),
            )
        )
    u = vw / _RENDER_PX
    return (
        vx + box[0] * u,
        vy + box[1] * u,
        (box[2] - box[0]) * u,
        (box[3] - box[1]) * u,
    )


def _viewbox(svg: str) -> tuple[float, float]:
    m = re.search(r'viewBox="([\d.\s-]+)"', svg)
    _, _, w, h = (float(v) for v in m.group(1).split())
    return w, h


def _inner(svg: str) -> str:
    """Return an SVG's drawable body (defs and paths) without its `<svg>` wrapper."""
    open_end = svg.index(">", svg.index("<svg")) + 1
    return svg[open_end : svg.rindex("</svg>")]


def _render_master(svg_bytes: bytes) -> Image.Image:
    png = cairosvg.svg2png(bytestring=svg_bytes, output_width=_RENDER_PX)
    return Image.open(io.BytesIO(png)).convert("RGBA")


def _ring_counter(svg: str, vb_w: float) -> tuple[float, float, float]:
    """Return the centre and radius of the ring's counter, in the master's user units.

    Measured off the ring path rather than written down, so the recording fill keeps
    following the mark if it is ever redrawn. Rays are cast from the ring's centre and
    the first ink along each one is its inner edge; the Q's tail gap leaves a handful of
    rays running far past the rest, so the median is taken and not the maximum.
    """
    if 'id="ring"' not in svg:
        raise ValueError("master mark has no ring path to measure the counter from")
    ring_only = re.sub(r'<path id="bird".*?/>', "", svg, flags=re.S)
    n = _RENDER_PX
    png = cairosvg.svg2png(
        bytestring=ring_only.encode(), output_width=n, output_height=n
    )
    alpha = Image.open(io.BytesIO(png)).convert("RGBA").getchannel("A").load()
    bbox = Image.open(io.BytesIO(png)).convert("RGBA").getbbox()
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    radii = []
    for step in range(120):
        theta = step * math.tau / 120
        for r in range(n // 2):
            x, y = int(cx + r * math.cos(theta)), int(cy + r * math.sin(theta))
            if 0 <= x < n and 0 <= y < n and alpha[x, y] > 128:
                radii.append(r)
                break
    radii.sort()
    u = vb_w / n
    # Overlapped into the ring: meeting it exactly leaves an antialiased hairline of
    # background between the two. The ring is drawn over the fill, so the overlap costs
    # nothing.
    return cx * u, cy * u, radii[len(radii) // 2] * u * COUNTER_OVERLAP


def _fit_transform(bbox_u, canvas: float, frac: float) -> str:
    """Return the SVG transform that centres the mark bbox in `frac` of the canvas."""
    bx, by, bw, bh = bbox_u
    s = (frac * canvas) / max(bw, bh)
    tx = canvas / 2 - s * (bx + bw / 2)
    ty = canvas / 2 - s * (by + bh / 2)
    return f"translate({tx:.4f},{ty:.4f}) scale({s:.6f})"


def _raster(svg: str, size: int) -> Image.Image:
    """Rasterise a colour icon SVG at `size`.

    The PNGs are rendered from the same SVG they ship beside, rather than composited
    separately, so the two can't disagree. It also keeps the tile edge antialiased:
    drawing the disc or squircle straight onto a 16-24px bitmap gives a hard binary
    edge, which reads as a staircase against the panel.
    """
    png = cairosvg.svg2png(
        bytestring=svg.encode(), output_width=size, output_height=size
    )
    return Image.open(io.BytesIO(png)).convert("RGBA")


def _write(path: Path, data: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(path, mode) as f:
        f.write(data)
    print(f"  {path.relative_to(_REPO_ROOT)}")


def _colour_svg(
    inner: str,
    bbox_u,
    canvas: float,
    tile_color: str,
    shape: str = "squircle",
    glyph_frac: float = GLYPH_FRAC,
    opacity: float = 1.0,
    counter=None,
) -> str:
    """Return a colour icon SVG: the mark on a `tile_color` tile, or counter-filled.

    `counter` picks the second form: the mark keeps its normal size and the ring's
    counter is flooded, inside the mark's transform (so it tracks the ring) and under
    the paths (so the ring and bird stay crisp). The tile form draws the mark white,
    which reads on the dark tile. The counter form draws it in the symbolic base: it
    sits on the panel beside the symbolic states, and desktop foregrounds are a muted
    grey near that value, so white would make recording the brightest of the three.
    """
    mark = SYMBOLIC_FILL if counter is not None else "#ffffff"
    body = inner.replace('fill="#000000"', f'fill="{mark}"')
    fill = ""
    bg = ""
    fade = "" if opacity >= 1.0 else f' opacity="{opacity:g}"'
    if counter is not None:
        cx, cy, r = counter
        # The breath dims the fill alone: the mark is the logo and holds still, while
        # the red behind it is the indicator that pulses. Fading the pair together
        # would read as the whole icon blinking out.
        fill = f'<circle cx="{cx:g}" cy="{cy:g}" r="{r:g}" fill="{tile_color}"{fade}/>'
        fade = ""
    elif shape == "circle":
        bg = (
            f'<circle cx="{canvas / 2:g}" cy="{canvas / 2:g}" r="{canvas / 2:g}" '
            f'fill="{tile_color}"/>'
        )
    else:
        side = canvas * TILE_FRAC
        inset = (canvas - side) / 2
        radius = side * CORNER_FRAC
        bg = (
            '<defs><linearGradient id="tile" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{INK_TOP}"/>'
            f'<stop offset="1" stop-color="{tile_color}"/>'
            "</linearGradient></defs>"
            f'<rect x="{inset:g}" y="{inset:g}" width="{side:g}" height="{side:g}" '
            f'rx="{radius:g}" ry="{radius:g}" fill="url(#tile)"/>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {canvas:g} {canvas:g}" width="{canvas:g}" height="{canvas:g}">'
        f"<g{fade}>{bg}"
        f'<g transform="{_fit_transform(bbox_u, canvas, glyph_frac)}">{fill}{body}</g>'
        f"</g></svg>\n"
    )


def _emit_colour_pack(
    name: str,
    inner: str,
    bbox_u,
    canvas: float,
    tile_color: str,
    shape: str = "squircle",
    glyph_frac: float = GLYPH_FRAC,
    opacity: float = 1.0,
    sizes: tuple[int, ...] = RASTER_SIZES,
    counter=None,
) -> None:
    """Write a hicolor colour pack (scalable SVG and a PNG per size) for icon `name`.

    `opacity` fades the whole tile form, or only the counter fill in the counter form
    the recording-pulse frames use. Every state gets the scalable form: a host that
    resolves one state to vector and the next to a bitmap shows the icon sharpen and
    soften as the state changes.
    """
    print(f"Colour pack: {name} ({tile_color}, {shape}, opacity={opacity:g}):")
    svg = _colour_svg(
        inner, bbox_u, canvas, tile_color, shape, glyph_frac, opacity, counter
    )
    _write(_HICOLOR / "scalable" / "apps" / f"{name}.svg", svg)
    for size in sizes:
        out = _HICOLOR / f"{size}x{size}" / "apps" / f"{name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        _raster(svg, size).save(out)
        print(f"  {out.relative_to(_REPO_ROOT)}")


def main() -> None:
    svg = _MASTER.read_text()
    vb_w, vb_h = _viewbox(svg)
    inner = _inner(svg)
    master = _render_master(svg.encode())

    # Mark bbox in SVG user units, converted from render px by the viewBox scale.
    bx, by, bx2, by2 = master.getbbox()
    u = vb_w / _RENDER_PX
    bbox_u = (bx * u, by * u, (bx2 - bx) * u, (by2 - by) * u)

    canvas = max(vb_w, vb_h)

    # App tile (ink squircle): app/dock/store.
    _emit_colour_pack(APP_ID, inner, bbox_u, canvas, INK)
    # Recording: the mark with the ring's counter flooded red, sized like the symbolic
    # states so the tray does not resize when recording starts.
    counter = _ring_counter(svg, vb_w)
    _emit_colour_pack(
        f"{APP_ID}-recording",
        inner,
        bbox_u,
        canvas,
        RECORD_RED,
        glyph_frac=SYMBOLIC_FRAC,
        counter=counter,
    )
    # Recording-pulse frames: the same fill at reduced opacity, cycled by the tray while
    # recording. Only the tray shows them, so the raster sizes stop there; the base
    # -recording icon is frame 0 and the static fallback.
    for frame, alpha in ((1, 0.70), (2, 0.45)):
        _emit_colour_pack(
            f"{APP_ID}-recording-pulse-{frame}",
            inner,
            bbox_u,
            canvas,
            RECORD_RED,
            glyph_frac=SYMBOLIC_FRAC,
            opacity=alpha,
            sizes=TRAY_SIZES,
            counter=counter,
        )

    # The idle symbolic is the bare Q, drawn in the freedesktop symbolic base (#bebebe)
    # rather than the master's #000000 (see SYMBOLIC_FILL). The -recording icon stays
    # coloured (RECORD_RED).
    print("Symbolic SVGs:")
    sym_body = inner.replace('fill="#000000"', f'fill="{SYMBOLIC_FILL}"')
    sym_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 16 16" width="16" height="16">'
        f'<g transform="{_fit_transform(bbox_u, 16, SYMBOLIC_FRAC)}">{sym_body}</g>'
        f"</svg>\n"
    )
    _write(_HICOLOR / "symbolic" / "apps" / f"{APP_ID}-symbolic.svg", sym_svg)

    # Transcribing: frames of the loader, where the wings unfurl into the Q and peel
    # back off. Symbolic SVGs rather than raster frames, so they stay recolourable and
    # cost one small file each. Frame 0 doubles as the static icon.
    loader = _LOADER.read_text()
    css = re.search(r"<style.*?</style>", loader, re.S).group(0)
    tracks = {name: _keyframes(css, name) for name in ("alt_a", "alt_b")}
    phases = _even_motion_phases(loader, tracks, TRANSCRIBE_FRAMES)
    # Fitted like the idle mark, and to the same fraction, so the tray icon does not
    # change size when transcription starts.
    loader_fit = _fit_transform(
        _loader_extent(loader, tracks, phases), 16, SYMBOLIC_FRAC
    )
    for i, t in enumerate(phases):
        stem = (
            f"{APP_ID}-transcribing-symbolic"
            if i == 0
            else f"{APP_ID}-transcribing-pulse-{i}-symbolic"
        )
        _write(
            _HICOLOR / "symbolic" / "apps" / f"{stem}.svg",
            _loader_frame(loader, tracks, t, loader_fit),
        )

    print("Done.")


if __name__ == "__main__":
    main()
