# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Whisper model catalog and on-disk resolution.

Pure catalog and path layer: no network. Downloading and integrity verification
live in `quoth.downloader`; this module knows what models exist, their filenames,
which are present on disk, their deletion and the OpenVINO compile cache.
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from quoth import paths
from quoth.accel import backend_for


@dataclass
class ModelInfo:
    """One catalog entry: a Whisper model and its downloadable assets."""

    name: str
    label: str  # the name the UI shows; `name` stays the file and log identifier
    ggml_size_mb: int  # decimal megabytes, the unit GLib.format_size shows
    has_openvino_encoder: bool  # whether an Intel-NPU OpenVINO encoder exists
    supports_translation: bool  # whether Whisper's translate-to-English task works
    description: str


# The OpenVINO encoder is the extra asset the NPU path needs. `large-v3-turbo` shares
# large-v3's encoder and is unusably slow on CPU alone.
AVAILABLE_MODELS = [
    ModelInfo("tiny", "Tiny", 44, True, True, "Fastest, lower accuracy"),
    ModelInfo("base", "Base", 82, True, True, "Balanced speed and accuracy"),
    ModelInfo("small", "Small", 264, True, True, "Higher accuracy, slower on CPU"),
    ModelInfo(
        "large-v3-turbo",
        "Turbo",
        874,
        True,
        False,
        "High accuracy; best with a GPU or NPU (large-v3-turbo)",
    ),
]

MODELS_BY_NAME = {m.name: m for m in AVAILABLE_MODELS}


def model_label(name: str) -> str:
    """Return the display name of model `name`, or `name` itself if it is unknown."""
    info = MODELS_BY_NAME.get(name)
    return info.label if info else name


# --- Filename convention (single source of truth) ---


# whisper.cpp's Silero VAD weights, shared by all models and small enough to ship in
# the package, so VAD works offline from the first launch. They live in weights/ since
# a models/ directory would shadow this module on import.
VAD_MODEL_FILE = "ggml-silero-v6.2.0.bin"
_BUNDLED_VAD = Path(__file__).parent / "weights" / VAD_MODEL_FILE


# Weights are whisper.cpp's q8_0 quantisation: half the download of the f16 files and a
# fifth faster on the CPU, with no measurable loss of accuracy.
def ggml_name(model: str) -> str:
    return f"ggml-{model}-q8_0.bin"


def encoder_xml_name(model: str) -> str:
    return f"ggml-{model}-encoder-openvino.xml"


def encoder_bin_name(model: str) -> str:
    return f"ggml-{model}-encoder-openvino.bin"


def partial_name(asset: str) -> str:
    """Return the name a download of `asset` writes and resumes until it is verified."""
    return f"{asset}.tmp"


def needs_openvino_encoder(device: str, model: str) -> bool:
    """Whether a `(device, model)` pair needs the OpenVINO encoder asset.

    True only when the backend offloads the encoder to OpenVINO (the NPU path) and the
    model publishes one. Call sites check this rather than a `device in (...)` test,
    so device meaning stays in `accel`.
    """
    return (
        backend_for(device).wants_openvino_encoder
        and MODELS_BY_NAME[model].has_openvino_encoder
    )


class ModelManager:
    """The tracker for whisper model files on disk: presence and path resolution."""

    def __init__(self, models_dir: Path | None = None):
        self._dir = models_dir or paths.models_dir()
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # Startup ends here, and the crash hook's file line drops the path.
            logger.error("Models directory {} unusable: {}", self._dir, e)
            raise
        logger.debug("Models directory: {}", self._dir)

    @property
    def models_dir(self) -> Path:
        return self._dir

    def list_downloaded(self) -> list[str]:
        """Return the names of the catalog models with a GGML file present."""
        return [
            m.name for m in AVAILABLE_MODELS if (self._dir / ggml_name(m.name)).exists()
        ]

    def is_downloaded(self, model: str) -> bool:
        return (self._dir / ggml_name(model)).exists()

    def has_openvino(self, model: str) -> bool:
        """Whether both OpenVINO encoder files are present for this model."""
        return (self._dir / encoder_xml_name(model)).exists() and (
            self._dir / encoder_bin_name(model)
        ).exists()

    def model_path(self, model: str) -> Path | None:
        path = self._dir / ggml_name(model)
        return path if path.exists() else None

    def openvino_path(self, model: str) -> Path | None:
        path = self._dir / encoder_xml_name(model)
        return path if path.exists() else None

    def vad_model_path(self) -> Path | None:
        """Return the shared Silero VAD weights, or None when neither copy exists.

        A copy in the models directory wins over the bundled one. None (a source tree
        without the asset) leaves VAD off rather than failing.
        """
        for path in (self._dir / VAD_MODEL_FILE, _BUNDLED_VAD):
            if path.exists():
                return path
        return None

    def delete_model(self, model: str):
        """Delete all files for a model, its download partials included."""
        logger.info("Deleting model '{}'", model)
        for asset in (
            ggml_name(model),
            encoder_xml_name(model),
            encoder_bin_name(model),
        ):
            for name in (asset, partial_name(asset)):
                path = self._dir / name
                if path.exists():
                    path.unlink()
                    logger.debug("Removed {}", path)

    @property
    def cache_dir(self) -> Path:
        """The OpenVINO compiled-blob cache, surfaced and cleared here."""
        return paths.openvino_cache_dir()

    def cache_size_bytes(self) -> int:
        """Return the total size of the compiled-blob cache on disk, 0 if absent.

        Files may disappear while OpenVINO replaces its cache. Other filesystem
        errors are surfaced so the UI never reports an unreadable cache as empty.
        """
        cache = self.cache_dir
        try:
            cache.stat()
        except FileNotFoundError:
            return 0
        total = 0

        def _walk_error(error: OSError) -> None:
            if not isinstance(error, FileNotFoundError):
                raise error

        for root, _dirs, files in os.walk(cache, onerror=_walk_error):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except FileNotFoundError:
                    continue
        return total

    def clear_cache(self) -> None:
        """Delete the compiled-blob cache; absent caches are already clear.

        Callers must first evict calibration warmth so a partial deletion cannot
        leave remaining state falsely marked warm.
        """
        cache = self.cache_dir
        try:
            cache.stat()
        except FileNotFoundError:
            return
        logger.info("Clearing acceleration cache: {}", cache)
        try:
            shutil.rmtree(cache)
        except FileNotFoundError:
            try:
                cache.stat()
            except FileNotFoundError:
                return
            raise
