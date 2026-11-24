# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Lottie rendering through libthorvg into Cairo surfaces.

The ctypes seam to libthorvg, which draws the onboarding animations from Lottie JSON.
It takes an RGBA tuple for the recolourable ink rather than a Theme. The caller owns
the frame clock and blits the returned surface.

Lifecycle, as of libthorvg 1.1.1: the engine is reference-counted, so init is
idempotent and it is never torn down. Paints are reference-counted too, so
`animation_del` then `canvas_destroy` frees the picture exactly once.
"""

from __future__ import annotations

import ctypes

import cairo
from loguru import logger

from quoth import _native

# TVG_COLORSPACE_ARGB8888: alpha-premultiplied ARGB, byte-identical to Cairo's
# FORMAT_ARGB32, so ThorVG renders straight into the surface buffer with no conversion.
_ARGB8888 = 1
# TVG_ENGINE_OPTION_DEFAULT.
_ENGINE_DEFAULT = 1
_SUCCESS = 0


def _load_library() -> ctypes.CDLL | None:
    """Return the libthorvg handle with its signatures bound, or None if absent.

    A source tree that never ran `make vendor-thorvg` has no library. Absence is not an
    error here: `load` then returns None and callers degrade.
    """
    for path in _native.thorvg_lib_paths():
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        _bind(lib)
        logger.debug("libthorvg loaded: {}", path)
        return lib
    logger.debug("libthorvg not found; Lottie rendering unavailable")
    return None


def _bind(lib: ctypes.CDLL) -> None:
    """Set argtypes and restypes for the subset of the ThorVG C API used here.

    ThorVG hands back opaque pointers (canvas, animation, picture) as void*; results are
    int enums.
    """
    lib.tvg_engine_init.argtypes = [ctypes.c_uint]
    lib.tvg_swcanvas_create.argtypes = [ctypes.c_int]
    lib.tvg_swcanvas_create.restype = ctypes.c_void_p
    lib.tvg_swcanvas_set_target.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_int,
    ]
    lib.tvg_canvas_update.argtypes = [ctypes.c_void_p]
    lib.tvg_canvas_draw.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    lib.tvg_canvas_sync.argtypes = [ctypes.c_void_p]
    lib.tvg_canvas_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.tvg_canvas_destroy.argtypes = [ctypes.c_void_p]
    lib.tvg_lottie_animation_new.restype = ctypes.c_void_p
    lib.tvg_animation_del.argtypes = [ctypes.c_void_p]
    lib.tvg_animation_get_picture.argtypes = [ctypes.c_void_p]
    lib.tvg_animation_get_picture.restype = ctypes.c_void_p
    lib.tvg_animation_set_frame.argtypes = [ctypes.c_void_p, ctypes.c_float]
    lib.tvg_animation_get_total_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.tvg_animation_get_duration.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.tvg_picture_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.tvg_picture_get_size.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.tvg_picture_set_size.argtypes = [
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.tvg_lottie_animation_gen_slot.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.tvg_lottie_animation_gen_slot.restype = ctypes.c_uint32
    lib.tvg_lottie_animation_apply_slot.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.tvg_lottie_animation_del_slot.argtypes = [ctypes.c_void_p, ctypes.c_uint32]


_LIB: ctypes.CDLL | None = _load_library()
_engine_ready = False


def _ensure_engine() -> None:
    global _engine_ready
    if not _engine_ready:
        # 0 threads: render synchronously on the calling (main) thread.
        _LIB.tvg_engine_init(0)
        _engine_ready = True


def load(path: str) -> LottieAnimation | None:
    """Load the Lottie document at `path`, or return None if it cannot be rendered."""
    if _LIB is None:
        return None
    _ensure_engine()
    return LottieAnimation._load(_LIB, path)


class LottieAnimation:
    """One loaded Lottie document with its own canvas.

    `render` rasterises a frame and returns it as a fresh Cairo surface; `apply_ink`
    recolours the tagged ink. Not constructed directly: use `load()`, which returns None
    on failure.
    """

    def __init__(self, lib: ctypes.CDLL, anim: int, canvas: int, picture: int):
        self._lib = lib
        self._anim: int | None = anim
        self._canvas = canvas
        self._picture = picture
        self._slot = 0  # the applied ink slot id, 0 = none
        # A persistent render buffer, re-pointed only on a size change. Each render
        # wraps it in a fresh Cairo surface rather than reusing one: a surface used as a
        # paint source is snapshotted by Cairo, and writing into it again then asserts.
        self._buffer: bytearray | None = None
        self._buffer_wh: tuple[int, int] | None = None
        cnt = ctypes.c_float()
        lib.tvg_animation_get_total_frame(anim, ctypes.byref(cnt))
        self._total_frames = cnt.value
        dur = ctypes.c_float()
        lib.tvg_animation_get_duration(anim, ctypes.byref(dur))
        self._duration_s = dur.value
        # The document's own size, read before any render re-points it to a surface.
        # ThorVG stretches to the target, so the caller needs this aspect to letterbox.
        w, h = ctypes.c_float(), ctypes.c_float()
        lib.tvg_picture_get_size(picture, ctypes.byref(w), ctypes.byref(h))
        self._width, self._height = w.value, h.value

    @classmethod
    def _load(cls, lib: ctypes.CDLL, path: str) -> LottieAnimation | None:
        anim = lib.tvg_lottie_animation_new()
        picture = lib.tvg_animation_get_picture(anim)
        if lib.tvg_picture_load(picture, path.encode()) != _SUCCESS:
            lib.tvg_animation_del(anim)
            logger.warning("Lottie load failed: {}", path)
            return None
        canvas = lib.tvg_swcanvas_create(_ENGINE_DEFAULT)
        lib.tvg_canvas_add(canvas, picture)
        return cls(lib, anim, canvas, picture)

    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
        return self._height

    @property
    def total_frames(self) -> float:
        return self._total_frames

    @property
    def duration_s(self) -> float:
        return self._duration_s

    def render(self, w: int, h: int, frame: float) -> cairo.ImageSurface | None:
        """Rasterise `frame` at `w` x `h` and return it as a fresh Cairo surface.

        None once closed or for an empty size. The surface wraps this animation's own
        buffer, which stays alive on the instance; the caller must paint it before the
        next render overwrites it.
        """
        if self._anim is None or w <= 0 or h <= 0:
            return None
        if (w, h) != self._buffer_wh:
            self._retarget(w, h)
        self._lib.tvg_animation_set_frame(self._anim, ctypes.c_float(frame))
        self._lib.tvg_canvas_update(self._canvas)
        self._lib.tvg_canvas_draw(self._canvas, True)
        self._lib.tvg_canvas_sync(self._canvas)
        # FORMAT_ARGB32 stride is 4*w, which matches the target stride `_retarget` sets.
        return cairo.ImageSurface.create_for_data(
            self._buffer, cairo.FORMAT_ARGB32, w, h
        )

    def _retarget(self, w: int, h: int) -> None:
        self._buffer = bytearray(w * h * 4)
        self._buffer_wh = (w, h)
        ptr = ctypes.cast(
            (ctypes.c_char * len(self._buffer)).from_buffer(self._buffer),
            ctypes.POINTER(ctypes.c_uint32),
        )
        self._lib.tvg_swcanvas_set_target(self._canvas, ptr, w, w, h, _ARGB8888)
        self._lib.tvg_picture_set_size(self._picture, float(w), float(h))

    def apply_ink(self, rgba: tuple[float, float, float, float]) -> None:
        """Recolour every fill tagged `sid: "ink"` to `rgba` (each channel 0..1).

        The prior slot is released first, so repeated theme changes don't accumulate.
        """
        if self._anim is None:
            return
        if self._slot:
            self._lib.tvg_lottie_animation_del_slot(self._anim, self._slot)
        r, g, b, a = rgba
        spec = f'{{"ink":{{"p":{{"a":0,"k":[{r:g},{g:g},{b:g},{a:g}]}}}}}}'
        slot = self._lib.tvg_lottie_animation_gen_slot(self._anim, spec.encode())
        if slot:
            self._lib.tvg_lottie_animation_apply_slot(self._anim, slot)
        self._slot = slot

    def close(self) -> None:
        """Free the document and its canvas. Idempotent.

        The picture is owned by the animation and reference-counted with the canvas, so
        this order frees it once.
        """
        if self._anim is None:
            return
        self._lib.tvg_animation_del(self._anim)
        self._lib.tvg_canvas_destroy(self._canvas)
        self._anim = None
        self._buffer = None
