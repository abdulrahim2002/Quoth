# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `accel`."""

import pytest

from quoth.accel import backend_for, has_accelerator, resolve_backend


class TestHasAccelerator:
    def test_cpu_only_has_no_accelerator(self):
        assert has_accelerator(["CPU"]) is False

    def test_gpu_or_npu_is_an_accelerator(self):
        assert has_accelerator(["CPU", "GPU"]) is True
        assert has_accelerator(["NPU", "CPU"]) is True

    def test_empty_list_has_no_accelerator(self):
        assert has_accelerator([]) is False


class TestBackendFor:
    @pytest.mark.parametrize(
        ("device", "use_gpu", "openvino_encoder", "fallbacks"),
        [
            ("CPU", False, False, ()),
            # Vulkan: GGML compute on the GPU, no OpenVINO encoder.
            ("GPU", True, False, ("CPU",)),
            # OpenVINO encoder offload; the decoder stays on the CPU.
            ("NPU", False, True, ("GPU", "CPU")),
        ],
    )
    def test_device_knobs(self, device, use_gpu, openvino_encoder, fallbacks):
        b = backend_for(device)
        assert b.use_gpu is use_gpu
        assert b.wants_openvino_encoder is openvino_encoder
        assert b.fallbacks == fallbacks

    def test_unknown_device_falls_back_to_cpu(self):
        assert backend_for("bogus") == backend_for("CPU")

    def test_is_accelerated_only_for_gpu_and_npu(self):
        # CPU is the floor; GPU (Vulkan) and NPU (encoder offload) accelerate.
        assert backend_for("CPU").is_accelerated is False
        assert backend_for("GPU").is_accelerated is True
        assert backend_for("NPU").is_accelerated is True


# --- resolve_backend -------------------------------------------------------

_ALL = ["NPU", "GPU", "CPU"]  # a fully-equipped machine (Lunar Lake)
_NO_NPU = ["GPU", "CPU"]
_CPU_ONLY = ["CPU"]


class TestResolveBackendNpuPreference:
    def test_npu_honoured_when_encoder_present(self):
        r = resolve_backend("small", "NPU", available=_ALL, encoder_on_disk=True)
        assert r.device == "NPU"
        assert r.degraded is False and r.reason is None

    def test_npu_without_encoder_degrades_to_gpu_not_cpu(self):
        r = resolve_backend(
            "large-v3-turbo", "NPU", available=_ALL, encoder_on_disk=False
        )
        assert r.device == "GPU"
        assert r.degraded is True
        assert (
            r.reason == "This model's NPU encoder isn't downloaded; using GPU instead"
        )

    def test_npu_without_encoder_and_no_gpu_degrades_to_cpu(self):
        r = resolve_backend(
            "small", "NPU", available=["NPU", "CPU"], encoder_on_disk=False
        )
        assert r.device == "CPU"
        assert r.degraded is True
        assert (
            r.reason == "This model's NPU encoder isn't downloaded; using CPU instead"
        )

    def test_npu_requested_but_hardware_absent(self):
        r = resolve_backend("small", "NPU", available=_NO_NPU, encoder_on_disk=True)
        assert r.device == "GPU"
        assert r.degraded is True
        assert "not available" in r.reason


class TestResolveBackendGpuCpu:
    def test_gpu_honoured(self):
        r = resolve_backend("small", "GPU", available=_ALL, encoder_on_disk=True)
        assert r.device == "GPU" and r.degraded is False

    def test_gpu_absent_degrades_to_cpu(self):
        r = resolve_backend("small", "GPU", available=_CPU_ONLY, encoder_on_disk=False)
        assert r.device == "CPU" and r.degraded is True
        assert r.reason == "GPU not available; using CPU instead"

    def test_cpu_always_cpu(self):
        r = resolve_backend("small", "CPU", available=_ALL, encoder_on_disk=True)
        assert r.device == "CPU" and r.degraded is False


class TestResolveBackendAuto:
    def test_auto_prefers_gpu_by_default(self):
        # The uncalibrated default order is GPU -> NPU -> CPU.
        r = resolve_backend("small", "auto", available=_ALL, encoder_on_disk=True)
        assert r.device == "GPU" and r.degraded is False

    def test_auto_uses_npu_when_no_gpu_and_encoder_present(self):
        r = resolve_backend(
            "small", "auto", available=["NPU", "CPU"], encoder_on_disk=True
        )
        assert r.device == "NPU"

    def test_auto_skips_npu_without_encoder(self):
        r = resolve_backend(
            "small", "auto", available=["NPU", "CPU"], encoder_on_disk=False
        )
        assert r.device == "CPU"

    def test_auto_respects_calibrated_ranking(self):
        # Calibration measured turbo fastest on the NPU here, so auto should pick it.
        ranking = {"large-v3-turbo": ["NPU", "GPU", "CPU"]}
        r = resolve_backend(
            "large-v3-turbo",
            "auto",
            available=_ALL,
            encoder_on_disk=True,
            ranking=ranking,
        )
        assert r.device == "NPU" and r.degraded is False

    def test_auto_falls_back_to_default_when_model_not_in_ranking(self):
        ranking = {"small": ["NPU", "GPU", "CPU"]}  # no turbo entry
        r = resolve_backend(
            "large-v3-turbo",
            "auto",
            available=_ALL,
            encoder_on_disk=True,
            ranking=ranking,
        )
        assert r.device == "GPU"  # default order

    def test_auto_ranking_skips_unusable_npu(self):
        # The ranking puts NPU first, but with no encoder auto takes the next usable.
        ranking = {"small": ["NPU", "GPU", "CPU"]}
        r = resolve_backend(
            "small", "auto", available=_ALL, encoder_on_disk=False, ranking=ranking
        )
        assert r.device == "GPU"
