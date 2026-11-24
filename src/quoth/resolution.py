# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Effective-backend resolution: the single composer.

`resolve_for` is the one app-facing entry point that turns a device preference into the
concrete backend to use, composing four lower layers:

- detection (`hardware.available_devices`): what can run here,
- rules (`accel.resolve_backend`): the pure preference -> device decision,
- catalog (`model_mgr.has_openvino`): whether the model's NPU encoder is on disk,
- measurement (`calibration.current_ranking`): `auto`'s per-model fastest device here.

Every consumer resolves through here, so the rules and the calibration ranking are
consulted in exactly one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from quoth import calibration, paths
from quoth.accel import BackendResolution, backend_for, resolve_backend
from quoth.hardware import available_devices, fingerprint

if TYPE_CHECKING:
    from quoth.models import ModelManager


def resolve_for(
    model: str,
    preference: str,
    model_mgr: ModelManager,
    *,
    assume_encoder: bool = False,
) -> BackendResolution:
    """Resolve a device preference to the effective backend from this machine's facts.

    The facts are the runnable devices, whether the model's OpenVINO encoder is on disk,
    and the calibrated per-model ranking (else the resolver's default order). An NPU or
    GPU request that cannot be honoured degrades explicitly, with a reason.

    `assume_encoder=True` resolves as if the encoder will be fetched, for provisioning
    a download.
    """
    encoder_on_disk = True if assume_encoder else model_mgr.has_openvino(model)
    ranking = calibration.current_ranking(fingerprint())
    res = resolve_backend(
        model,
        preference,
        available=available_devices(),
        encoder_on_disk=encoder_on_disk,
        ranking=ranking,
    )
    if res.degraded:
        logger.warning(
            "Backend resolve degraded: model={} preference={} -> {} ({})",
            model,
            preference,
            res.device,
            res.reason,
        )
    else:
        logger.debug(
            "Backend resolve: model={} preference={} -> {} (encoder_on_disk={}, ranked={})",
            model,
            preference,
            res.device,
            encoder_on_disk,
            ranking is not None,
        )
    return res


def provision_for(model: str, preference: str, model_mgr: ModelManager) -> str:
    """Return the device whose assets a download of `model` fetches for `preference`.

    It is the device the runtime would run once those assets are on disk, except that
    `auto` provisions the NPU whenever one is present: only then can the benchmark
    measure it and the ranking choose it.
    """
    if preference == "auto" and "NPU" in available_devices():
        return "NPU"
    return resolve_for(model, preference, model_mgr, assume_encoder=True).device


def expects_slow_compile(model: str, device: str) -> bool:
    """Whether `(model, device)` will trigger the one-time OpenVINO encoder compile.

    True for an NPU-bound model whose blob is not recorded warm, or whose cache has been
    emptied, so the app can warm it before the first dictation. Pass the effective
    device from `resolve_for`, not the raw preference.
    """
    if not backend_for(device).wants_openvino_encoder:
        return False
    pending = not (
        _cache_populated() and calibration.is_warm(model, device, fingerprint())
    )
    logger.debug("Slow compile expected for {} on {}: {}", model, device, pending)
    return pending


def _cache_populated() -> bool:
    # The cache lives where the user or a cleaner may delete it, leaving all blobs cold.
    try:
        return any(paths.openvino_cache_dir().iterdir())
    except OSError:
        return False


def current_ranking() -> dict[str, list[str]] | None:
    """Return the measured per-model device ranking under the current fingerprint.

    None means no benchmark has run here yet. It lives in the composer so the app never
    touches `calibration` directly.
    """
    ranking = calibration.current_ranking(fingerprint())
    logger.debug("Current calibration ranking: {}", ranking)
    return ranking


def worth_offering() -> bool:
    """Whether a first-run calibration offer is worthwhile on this machine.

    It is when there is an accelerator to rank and no fresh ranking yet.
    """
    worthwhile = calibration.worth_offering(available_devices(), fingerprint())
    logger.debug("Calibration offer worthwhile: {}", worthwhile)
    return worthwhile


def mark_compiled(model: str, device: str) -> None:
    """Record `(model, device)`'s encoder as compiled and warm under the fingerprint.

    A no-op for non-OpenVINO devices; any measured timing is kept.
    """
    if backend_for(device).wants_openvino_encoder:
        calibration.mark_warm(model, device, fingerprint())


def evict_warmth() -> None:
    """Mark the compiled OpenVINO blobs cold; call before clearing the on-disk cache.

    The warm gate then shows the preparing notice again instead of recompiling during
    a dictation.
    """
    calibration.evict_warmth(fingerprint())
