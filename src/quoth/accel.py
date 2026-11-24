# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Acceleration-backend model: the single source for what a device token means.

A concrete device (`CPU`, `GPU`, `NPU`) resolves to two orthogonal engine properties:

- `use_gpu`: run the GGML compute graph (decoder, and the encoder when there is no
  OpenVINO offload) on the GPU via Vulkan rather than the CPU.
- `wants_openvino_encoder`: offload the Whisper encoder to OpenVINO (the Intel NPU
  path; the decoder then stays on the GGML CPU backend).

Each also names its `fallbacks`, the devices that run a model it cannot.

Single-sourced here so no other module re-derives device meaning with `device in (...)`
checks: `hardware.py` detects a device, this module interprets it, everything else
derives from it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AccelBackend:
    """The engine knobs a device token maps onto, and where its work goes instead."""

    use_gpu: bool
    wants_openvino_encoder: bool
    # Tried in order when this device cannot run a model; CPU is the floor.
    fallbacks: tuple[str, ...] = ()

    @property
    def is_accelerated(self) -> bool:
        """Whether any work is offloaded off the CPU floor.

        True for the Vulkan GPU and the OpenVINO NPU encoder; False only for plain CPU.
        """
        return self.use_gpu or self.wants_openvino_encoder


# CPU: the whole GGML graph on CPU, the universal floor.
# GPU: Vulkan (any vendor: Intel/AMD/NVIDIA) runs encoder + decoder.
# NPU: OpenVINO encoder on the Intel NPU + GGML decoder on CPU, the low-power path.
_BACKENDS: dict[str, AccelBackend] = {
    "CPU": AccelBackend(use_gpu=False, wants_openvino_encoder=False),
    "GPU": AccelBackend(use_gpu=True, wants_openvino_encoder=False, fallbacks=("CPU",)),
    "NPU": AccelBackend(
        use_gpu=False, wants_openvino_encoder=True, fallbacks=("GPU", "CPU")
    ),
}

_CPU_BACKEND = _BACKENDS["CPU"]


def backend_for(device: str) -> AccelBackend:
    """Return the backend descriptor for a concrete device token.

    `auto` is resolved to a concrete device upstream by `resolve_backend`. An unknown
    token falls back to CPU, the always-safe floor.
    """
    return _BACKENDS.get(device, _CPU_BACKEND)


def has_accelerator(devices: list[str]) -> bool:
    """Whether any device in the list is a real accelerator, not the CPU floor.

    A CPU-only machine has nothing to calibrate. Call sites use this rather than
    testing `device == "CPU"`, so device meaning stays in this module.
    """
    return any(backend_for(device).is_accelerated for device in devices)


# --- Model-aware backend resolution -----------------------------------------
#
# Device is a preference, resolved per model to a concrete backend. The NPU counts
# only when the model's OpenVINO encoder IR is on disk, and a request that cannot be
# honoured degrades and says so. The caller passes the capability facts, so this stays
# pure and testable without hardware.

# Default `auto` order when uncalibrated: GPU (broadest, vendor-agnostic, full offload),
# then NPU (encoder-only, low-power), then CPU. Calibration overrides this per model.
_DEFAULT_AUTO_ORDER: tuple[str, ...] = ("GPU", "NPU", "CPU")


@dataclass(frozen=True)
class BackendResolution:
    """The outcome of resolving a (model, device-preference) pair to a real backend."""

    device: str  # effective concrete device: "CPU" / "GPU" / "NPU"
    degraded: bool  # True when an explicit GPU/NPU request could not be honoured
    reason: str | None  # human-readable why, when degraded (for the log + a UI note)


def _first_usable(
    order: tuple[str, ...], available: list[str], encoder_on_disk: bool
) -> str:
    """Return the first device in `order` that can actually run this model here.

    The NPU counts only when its encoder IR is present; CPU is the guaranteed floor.
    """
    for dev in order:
        if dev == "NPU":
            if "NPU" in available and encoder_on_disk:
                return "NPU"
        elif dev in available:
            return dev
    return "CPU"


def resolve_backend(
    model: str,
    preference: str,
    *,
    available: list[str],
    encoder_on_disk: bool,
    ranking: dict[str, list[str]] | None = None,
) -> BackendResolution:
    """Resolve a device preference to the effective concrete backend for `model`.

    `available` is the machine's runnable devices (`hardware.available_devices`),
    `encoder_on_disk` whether this model's OpenVINO encoder IR is present
    (`ModelManager.has_openvino`), and `ranking` the optional per-model measured order
    from calibration. An NPU or GPU request that cannot be honoured degrades
    explicitly, with a reason.
    """
    npu_ok = "NPU" in available and encoder_on_disk

    if preference == "NPU":
        if npu_ok:
            return BackendResolution("NPU", degraded=False, reason=None)
        dev = _first_usable(_BACKENDS["NPU"].fallbacks, available, encoder_on_disk)
        why = (
            "This model's NPU encoder isn't downloaded"
            if "NPU" in available
            else "NPU not available on this system"
        )
        return BackendResolution(
            dev, degraded=True, reason=f"{why}; using {dev} instead"
        )

    if preference == "GPU":
        if "GPU" in available:
            return BackendResolution("GPU", degraded=False, reason=None)
        dev = _first_usable(_BACKENDS["GPU"].fallbacks, available, encoder_on_disk)
        return BackendResolution(
            dev, degraded=True, reason=f"GPU not available; using {dev} instead"
        )

    if preference == "CPU":
        return BackendResolution("CPU", degraded=False, reason=None)

    # "auto" (and any unknown token): the best available device for this model, by the
    # measured ranking if calibration has one, else the default order. Never "degraded":
    # auto is choosing, not failing to honour a specific request.
    order = (
        tuple(ranking[model]) if ranking and model in ranking else _DEFAULT_AUTO_ORDER
    )
    dev = _first_usable(order, available, encoder_on_disk)
    return BackendResolution(dev, degraded=False, reason=None)
