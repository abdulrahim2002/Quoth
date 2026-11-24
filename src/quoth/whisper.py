# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Whisper speech-to-text through ctypes and the C helper shim.

libwhisper.so serves only pointer-based calls, which ctypes makes without a copy of
any struct layout: GGML backend registration, OpenVINO encoder init, segment reading
and freeing the context.

libwhisper_helper.so loads the model and transcribes. Both wrap struct-by-value calls
(whisper_context_default_params for init, whisper_full_default_params and whisper_full
for transcription), so those structs are built in C against whisper.h: a ctypes copy of
their layout would go stale when a whisper.cpp release changes it.

C API reference: whisper.cpp/include/whisper.h
"""

import ctypes
import ctypes.util
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from loguru import logger

from quoth import _native
from quoth.accel import backend_for
from quoth.environment import usable_cpus

if TYPE_CHECKING:
    from collections.abc import Callable

    from quoth.models import ModelManager

# whisper.cpp input contract: 16 kHz mono float32 PCM. The audio pipeline
# (quoth/audio.py) owns the capture rate; this copy sizes logs and synthetic buffers.
SAMPLE_RATE = 16000

# The shortest speech segment VAD will keep. The user's key already delimits the
# utterance, so whisper.cpp's 250 ms default would only delete short words like "yes".
VAD_MIN_SPEECH_MS = 32

# Audio kept on each side of a VAD speech segment, in ms. whisper.cpp's 30 ms
# default loses short words and costs accuracy on sentences.
VAD_SPEECH_PAD_MS = 200

# Peak amplitude above which a buffer VAD called silent is retried untrimmed. It is
# clear of both sides: wrongly discarded speech peaks at 0.36 and up, empty audio at
# 0.02 and down.
VAD_FALLBACK_PEAK = 0.1


class WarmupAborted(RuntimeError):
    """A cancel or shutdown stopped the warm-up decode; the context itself is intact."""


class ModelLoadFailed(RuntimeError):
    """whisper.cpp returned no context for the model on the requested device.

    The weights are unreadable, or the device refused them: ggml-vulkan can list a GPU
    that it then cannot run a given model on.
    """


@runtime_checkable
class TranscriptionBackend(Protocol):
    """The engine seam: an audio buffer in, a transcript out.

    `WhisperContext` is the only implementation; the controller's injected
    `whisper_factory` builds one from a model path and a device.
    """

    def transcribe_buffer(
        self, audio: np.ndarray, language: str = "auto", translate: bool = False
    ) -> str: ...

    def warm_up(self) -> None:
        """Exercise the backend once so the next transcription has no cold-start cost.

        Raises `WarmupAborted` when `cancel()` stops it.
        """
        ...

    def cancel(self) -> None:
        """Stop a running `transcribe_buffer` early; it returns "" soon after."""
        ...

    def close(self) -> None: ...

    @property
    def is_loaded(self) -> bool: ...

    @property
    def openvino_active(self) -> bool:
        """Whether the NPU (OpenVINO) encoder offload actually engaged.

        False for engines without one, and False when an NPU offload was requested but
        its runtime init failed (whisper.cpp then silently runs the encoder on CPU).
        Consumers must not infer NPU-active from a merely loaded context.
        """
        ...


# --- Transcript text ---


def clean_text(text: str) -> str:
    """Normalise whitespace in a transcript.

    Whisper's non-speech cues ([Music], (applause), ...) are suppressed at decode time
    (suppress_nst in the shim), so nothing needs stripping here. This collapses
    whitespace runs and trims the leading space whisper emits.
    """
    return " ".join(text.split())


# --- Library discovery ---


def _find_lib(name: str) -> str | None:
    """Return the path of a shared library by name, or None when none is found.

    The bundled dirs `_native` knows come first, then, only from a source checkout, a
    developer's local whisper.cpp build. An installed package never looks in a home
    directory a user may have for other reasons.
    """
    candidates = [Path(d) / name for d in _native.bundled_lib_dirs()]
    if _native.running_from_source():
        candidates.append(Path.home() / "whisper.cpp" / "build" / "bin" / name)
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def find_libwhisper() -> str | None:
    """Return a loadable libwhisper reference, or None.

    That is `_find_lib`'s path, else the system library's name from `find_library`.
    """
    path = _find_lib("libwhisper.so")
    if path:
        return path
    system = ctypes.util.find_library("whisper")
    return system


def find_libwhisper_helper() -> str | None:
    """Return the helper shim's path, searched like libwhisper's, or None."""
    return _find_lib("libwhisper_helper.so")


# --- GGML backend registration ---

_BACKEND_INIT_LOCK = threading.Lock()
_BACKEND_INIT_ATTEMPTED_DIRS: set[str] = set()


def _ensure_cpu_backend(lib: ctypes.CDLL, lib_path: str) -> None:
    """Register the CPU backend before whisper creates a model context."""
    reg_by_name = lib.ggml_backend_reg_by_name
    reg_by_name.argtypes = [ctypes.c_char_p]
    reg_by_name.restype = ctypes.c_void_p
    load_all = lib.ggml_backend_load_all_from_path
    load_all.argtypes = [ctypes.c_char_p]
    load_all.restype = None

    with _BACKEND_INIT_LOCK:
        if reg_by_name(b"CPU"):
            logger.debug("GGML CPU backend already registered for {}", lib_path)
            return

        if not os.path.dirname(lib_path):
            raise RuntimeError(
                f"Whisper CPU backend is unavailable and {lib_path!r} does not "
                "identify its dynamic backend directory. Reinstall Quoth, then "
                "restart it."
            )

        backend_dir = os.fspath(Path(lib_path).resolve().parent)
        if backend_dir not in _BACKEND_INIT_ATTEMPTED_DIRS:
            _BACKEND_INIT_ATTEMPTED_DIRS.add(backend_dir)
            logger.debug("Loading GGML backends from {}", backend_dir)
            load_all(os.fsencode(backend_dir))

        if not reg_by_name(b"CPU"):
            raise RuntimeError(
                "Whisper CPU backend did not register from "
                f"{backend_dir}. Reinstall Quoth, then restart it."
            )


# --- Model context ---

whisper_context_p = ctypes.c_void_p


class WhisperContext:
    """A loaded whisper.cpp context: the local `TranscriptionBackend`.

    It uses two shared libraries:
    - libwhisper_helper.so: model loading and transcription, the struct-by-value calls
    - libwhisper.so: OpenVINO init and segment reading, which are pointer-based
    """

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        openvino_model_path: str | None = None,
        cache_dir: str | None = None,
        vad_model_path: str | None = None,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        lib_path = find_libwhisper()
        if lib_path is None:
            raise RuntimeError(
                "libwhisper.so not found. Reinstall Quoth, then restart it."
            )

        helper_path = find_libwhisper_helper()
        if helper_path is None:
            raise RuntimeError(
                "libwhisper_helper.so not found. Reinstall Quoth, then restart it."
            )

        self._lib = ctypes.CDLL(lib_path)
        _ensure_cpu_backend(self._lib, lib_path)
        self._helper = ctypes.CDLL(helper_path)
        # Read by the shim's abort callback after each encoder pass and each decoded
        # token; cancel() sets it.
        self._abort = ctypes.c_int(0)
        logger.debug(
            "Native libs resolved: whisper={}, helper={}", lib_path, helper_path
        )
        self._setup_functions()

        # use_gpu selects the GGML Vulkan backend; wants_openvino_encoder gates the
        # NPU encoder offload below.
        backend = backend_for(device)

        # Init goes through the helper because whisper_context_default_params returns
        # its struct by value.
        use_gpu = backend.use_gpu
        logger.debug("Loading model: {} (use_gpu={})", model_path, use_gpu)
        self._ctx = self._helper.whisper_helper_init(
            model_path.encode("utf-8"),
            ctypes.c_bool(use_gpu),
        )
        if not self._ctx:
            raise ModelLoadFailed(f"Failed to load whisper model: {model_path}")

        # The NPU offloads only the encoder; the decoder stays on the CPU. whisper.cpp
        # silently falls back to the CPU encoder on a failed init, so record whether
        # the offload engaged.
        self._openvino_active = False
        if backend.wants_openvino_encoder and openvino_model_path:
            logger.debug(
                "Initializing OpenVINO encoder: {} on {}", openvino_model_path, device
            )
            # open_context passes the encoder with its blob cache (ModelManager's).
            ret = self._lib.whisper_ctx_init_openvino_encoder(
                self._ctx,
                openvino_model_path.encode("utf-8"),
                device.encode("utf-8"),
                cache_dir.encode("utf-8"),
            )
            self._openvino_active = ret == 0
            if ret != 0:
                logger.warning(
                    "OpenVINO encoder init failed (code {}); the encoder runs on CPU",
                    ret,
                )
            else:
                logger.debug("OpenVINO encoder initialized on {}", device)

        self._device = device
        # Passed to the shim per transcription; None disables the built-in VAD.
        self._vad_model_path = vad_model_path
        # Threads for what GGML runs on the CPU, the decoder included except on Vulkan.
        self._n_threads = usable_cpus()
        logger.debug("CPU threads: {}", self._n_threads)

    def _setup_functions(self):
        """Set up ctypes function signatures for both libraries."""
        lib = self._lib
        helper = self._helper

        # --- libwhisper.so (pointer-based functions only) ---

        lib.whisper_full_n_segments.restype = ctypes.c_int
        lib.whisper_full_n_segments.argtypes = [whisper_context_p]

        lib.whisper_full_get_segment_text.restype = ctypes.c_char_p
        lib.whisper_full_get_segment_text.argtypes = [whisper_context_p, ctypes.c_int]

        lib.whisper_free.restype = None
        lib.whisper_free.argtypes = [whisper_context_p]

        lib.whisper_ctx_init_openvino_encoder.restype = ctypes.c_int
        lib.whisper_ctx_init_openvino_encoder.argtypes = [
            whisper_context_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]

        # --- libwhisper_helper.so (avoids struct-by-value) ---

        helper.whisper_helper_init.restype = whisper_context_p
        helper.whisper_helper_init.argtypes = [ctypes.c_char_p, ctypes.c_bool]

        helper.whisper_helper_transcribe.restype = ctypes.c_int
        helper.whisper_helper_transcribe.argtypes = [
            whisper_context_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_bool,
            ctypes.c_int,  # CPU thread count
            ctypes.c_char_p,  # VAD model path, or NULL to disable
            ctypes.c_int,  # VAD min speech segment ms, or 0 for whisper.cpp's default
            ctypes.c_int,  # VAD speech pad ms, or 0 for whisper.cpp's default
            ctypes.POINTER(ctypes.c_int),  # abort flag the shim polls, or NULL
        ]

    def _run(
        self,
        samples,
        n_samples: int,
        lang_bytes: bytes,
        translate: bool,
        vad_bytes: bytes | None,
    ) -> str:
        """Run one decode pass and return the joined segment text.

        `vad_bytes` of None transcribes the buffer untrimmed.
        """
        ret = self._helper.whisper_helper_transcribe(
            self._ctx,
            samples,
            ctypes.c_int(n_samples),
            lang_bytes,
            ctypes.c_bool(translate),
            ctypes.c_int(self._n_threads),
            vad_bytes,
            ctypes.c_int(VAD_MIN_SPEECH_MS),
            ctypes.c_int(VAD_SPEECH_PAD_MS),
            ctypes.byref(self._abort),
        )
        logger.debug("whisper_helper_transcribe returned: {}", ret)
        if ret != 0:
            if self._abort.value:
                logger.debug("Transcription aborted")
                return ""
            raise RuntimeError(f"whisper_helper_transcribe failed with code {ret}")

        # Read segments from libwhisper.so (pointer-based, safe)
        n_segments = self._lib.whisper_full_n_segments(self._ctx)
        logger.debug("Segments: {}", n_segments)
        parts = []
        for i in range(n_segments):
            text_bytes = self._lib.whisper_full_get_segment_text(self._ctx, i)
            if text_bytes:
                parts.append(text_bytes.decode("utf-8", errors="replace"))
        return " ".join(parts)

    def transcribe_buffer(
        self,
        audio: np.ndarray,
        language: str = "auto",
        translate: bool = False,
    ) -> str:
        """Transcribe a whole 16 kHz mono float32 buffer and return the cleaned text.

        `language` is an ISO 639-1 code or "auto". `translate` runs Whisper's translate
        task, which outputs English whatever the spoken language.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        self._abort.value = 0
        samples = audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        n_samples = len(audio)
        lang_bytes = language.encode("utf-8") if language else b"auto"
        vad_bytes = (
            self._vad_model_path.encode("utf-8") if self._vad_model_path else None
        )

        logger.debug(
            "Transcribing: {} samples ({:.1f}s), language={}, translate={}, vad={}",
            n_samples,
            n_samples / SAMPLE_RATE,
            language,
            translate,
            bool(vad_bytes),
        )

        raw_text = self._run(samples, n_samples, lang_bytes, translate, vad_bytes)

        # VAD found no speech. A loud buffer means it was wrong, so transcribe untrimmed
        # rather than drop a word the user said. A quiet one is the case VAD exists for
        # (the key was tapped in silence), so that answer stands.
        if vad_bytes and not raw_text.strip() and not self._abort.value:
            peak = float(np.max(np.abs(audio))) if n_samples else 0.0
            if peak >= VAD_FALLBACK_PEAK:
                logger.debug(
                    "VAD found no speech at peak {:.3f}; retrying untrimmed", peak
                )
                raw_text = self._run(samples, n_samples, lang_bytes, translate, None)
        cleaned = clean_text(raw_text)
        if raw_text != cleaned:
            # Transcript content only ever at TRACE (never persisted to disk);
            # DEBUG carries the length delta, not the text.
            logger.debug(
                "Cleaned transcript ({} → {} chars)", len(raw_text), len(cleaned)
            )
            logger.trace("Raw {!r} → cleaned {!r}", raw_text, cleaned)
        return cleaned

    def warm_up(self) -> None:
        """Run one VAD-free decode to initialise lazy backend work."""
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self._abort.value = 0
        samples = audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._run(samples, len(audio), b"auto", False, None)
        # _run returns empty text for a cancel, so check whether the lazy work finished.
        if self._abort.value:
            raise WarmupAborted("Whisper warmup aborted")

    def cancel(self) -> None:
        """Stop the running decode; a single int write, safe from any thread.

        whisper checks the flag after each encoder pass and decoded token, so
        `transcribe_buffer` returns "" soon after.
        """
        self._abort.value = 1

    def close(self):
        """Free whisper context and resources."""
        # __del__ may run even if __init__ raised before _ctx was set.
        ctx = getattr(self, "_ctx", None)
        if ctx:
            self._lib.whisper_free(ctx)
            self._ctx = None
            logger.debug(
                "Whisper context closed (device={})", getattr(self, "_device", "?")
            )

    @property
    def is_loaded(self) -> bool:
        return self._ctx is not None

    @property
    def openvino_active(self) -> bool:
        """Whether the OpenVINO (NPU) encoder offload actually engaged here."""
        return self._openvino_active

    def __del__(self):
        self.close()


def open_context(
    model: str,
    device: str,
    model_mgr: "ModelManager",
    *,
    vad: bool = False,
    factory: "Callable[..., TranscriptionBackend]" = WhisperContext,
) -> "TranscriptionBackend | None":
    """Build a transcription context for a `(model, device)` pair through `factory`.

    This is the one place that resolves the ggml weights and the NPU-only OpenVINO
    encoder path from `model_mgr`. Returns None, having logged, when the model is not
    downloaded or construction fails.
    `vad` enables whisper.cpp's built-in silence trimming; a missing VAD weights file
    leaves it off rather than failing.
    """
    model_path = model_mgr.model_path(model)
    if model_path is None:
        logger.warning("Model '{}' not downloaded", model)
        return None
    vad_path = None
    if vad:
        p = model_mgr.vad_model_path()
        if p:
            vad_path = str(p)
        else:
            logger.warning(
                "VAD enabled but its model is missing; transcribing without it"
            )
    ov_path = None
    if backend_for(device).wants_openvino_encoder:
        p = model_mgr.openvino_path(model)
        if p:
            ov_path = str(p)
        else:
            # Only a race reaches here (the encoder removed mid-flight). The context
            # opens without the offload, which its callers treat as the NPU refusing.
            logger.warning(
                "NPU device for '{}' but its OpenVINO encoder is missing; "
                "opening without it",
                model,
            )
    try:
        return factory(
            str(model_path),
            device=device,
            openvino_model_path=ov_path,
            cache_dir=str(model_mgr.cache_dir),
            vad_model_path=vad_path,
        )
    except ModelLoadFailed:
        logger.error(
            "Failed to load whisper model '{}' on {} from {}", model, device, model_path
        )
        return None
    except (OSError, RuntimeError) as error:
        # A missing file or library: the reason goes in the message, since the file
        # sink drops exception values; the traceback files a broken install's report.
        logger.opt(exception=True).error(
            "Failed to load whisper model '{}' on {}: {}", model, device, error
        )
        return None
    except Exception:
        logger.exception("Failed to load whisper model '{}' on {}", model, device)
        return None
