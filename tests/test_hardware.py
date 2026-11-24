# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.hardware`.

A device is offered only when its backend can run: a Vulkan GPU must enumerate, and the
NPU needs an Intel NPU device node, the OpenVINO runtime and Intel's NPU driver. Each
signal is mocked independently.
"""

import ctypes
import os
from unittest.mock import call, patch

import pytest

from quoth.hardware import (
    _npu_reasons_logged,
    _npu_user_driver_present,
    _vulkan_gpu_present,
    available_devices,
    fingerprint,
)


@pytest.fixture(autouse=True)
def detected_devices():
    """Leave detection unpinned: it is what this file tests."""


@pytest.fixture(autouse=True)
def _fresh_probes():
    """Clear the memoised probes around each test, so no answer outlives its fakes."""
    _vulkan_gpu_present.cache_clear()
    _npu_user_driver_present.cache_clear()
    _npu_reasons_logged.clear()
    yield
    _vulkan_gpu_present.cache_clear()
    _npu_user_driver_present.cache_clear()


@pytest.fixture
def capabilities(tmp_path, monkeypatch):
    """Patch the accel node, OpenVINO core, NPU driver and Vulkan GPU signals.

    `npu_driver` names the kernel driver bound to a fake sysfs `accel0`; `dev_node`
    is whether its /dev node is reachable, as a sandbox without device access hides it.
    `user_driver` is whether Intel's NPU user-mode driver loads.
    """

    def patch_signals(
        *, npu_driver=None, dev_node=True, openvino=False, user_driver=True, gpu=False
    ):
        sysfs, dev = tmp_path / "class", tmp_path / "dev"
        sysfs.mkdir()
        dev.mkdir()
        if npu_driver is not None:
            (tmp_path / "drivers" / npu_driver).mkdir(parents=True)
            (sysfs / "accel0" / "device").mkdir(parents=True)
            (sysfs / "accel0" / "device" / "driver").symlink_to(
                tmp_path / "drivers" / npu_driver
            )
            if dev_node:
                (dev / "accel0").touch()
        monkeypatch.setattr("quoth.hardware._ACCEL_CLASS", str(sysfs))
        monkeypatch.setattr("quoth.hardware._ACCEL_DEV", str(dev))
        monkeypatch.setattr(
            "quoth.hardware.openvino_core_lib_paths",
            lambda: ["libopenvino.so"] if openvino else [],
        )
        monkeypatch.setattr(
            "quoth.hardware._npu_user_driver_present", lambda: user_driver
        )
        monkeypatch.setattr("quoth.hardware._vulkan_gpu_present", lambda: gpu)

    return patch_signals


class TestAvailableDevices:
    @pytest.mark.parametrize("driver", ["intel_vpu", "intel_npu"])
    def test_npu_when_node_and_openvino_present(self, capabilities, driver):
        capabilities(npu_driver=driver, openvino=True, gpu=True)
        assert available_devices() == ["NPU", "GPU", "CPU"]

    def test_no_npu_without_openvino_even_with_node(self, capabilities, caplog):
        # A build that ships no OpenVINO may still see the node, as a sandbox can
        # expose it, so it offers the GPU, not the NPU.
        capabilities(npu_driver="intel_vpu", openvino=False, gpu=True)
        assert available_devices() == ["GPU", "CPU"]
        assert "NPU accel0: OpenVINO core not found" in caplog.messages

    def test_no_npu_without_node_even_with_openvino(self, capabilities):
        capabilities(openvino=True, gpu=True)
        assert available_devices() == ["GPU", "CPU"]

    def test_no_npu_without_the_accel_class(self, capabilities, monkeypatch, caplog):
        # A kernel without the accel subsystem has no /sys/class/accel at all.
        capabilities(openvino=True, gpu=True)
        monkeypatch.setattr("quoth.hardware._ACCEL_CLASS", "/nonexistent/class/accel")
        assert available_devices() == ["GPU", "CPU"]
        assert any("no accel subsystem" in m for m in caplog.messages)

    def test_no_npu_on_an_accel_node_of_another_vendor(self, capabilities, caplog):
        # AMD's NPU driver makes the same /dev/accel node, which OpenVINO cannot drive.
        capabilities(npu_driver="amdxdna", openvino=True, gpu=True)
        assert available_devices() == ["GPU", "CPU"]
        assert "NPU accel0: driver amdxdna not supported" in caplog.messages

    def test_no_npu_when_the_sandbox_hides_the_node(self, capabilities):
        capabilities(npu_driver="intel_vpu", dev_node=False, openvino=True)
        assert available_devices() == ["CPU"]

    def test_no_npu_when_the_user_cannot_open_the_node(
        self, capabilities, monkeypatch, caplog
    ):
        capabilities(npu_driver="intel_vpu", openvino=True)
        # Readable only: the NPU needs the node writable too.
        monkeypatch.setattr(os, "access", lambda path, mode: not mode & os.W_OK)
        assert available_devices() == ["CPU"]
        # Every resolve re-probes; the reason is logged once.
        assert available_devices() == ["CPU"]
        rejected = [m for m in caplog.messages if "cannot read and write" in m]
        assert len(rejected) == 1

    def test_no_npu_without_intels_user_mode_driver(self, capabilities):
        # OpenVINO ships in the .deb, but Debian and Ubuntu do not package the driver.
        capabilities(npu_driver="intel_vpu", openvino=True, user_driver=False)
        assert available_devices() == ["CPU"]

    def test_gpu_then_cpu(self, capabilities):
        capabilities(gpu=True)
        assert available_devices() == ["GPU", "CPU"]

    def test_cpu_is_the_floor(self, capabilities):
        capabilities()
        assert available_devices() == ["CPU"]


class TestFingerprint:
    def test_combines_openvino_and_devices(self, capabilities, monkeypatch):
        capabilities(npu_driver="intel_vpu", openvino=True, gpu=True)
        monkeypatch.setattr(
            "quoth.hardware.openvino_core_lib_paths",
            lambda: ["/lib/libopenvino.so.2600"],
        )
        fp = fingerprint()
        assert fp == "engine=ggml-dispatch-1;ov=libopenvino.so.2600;dev=NPU/GPU/CPU"

    def test_none_when_no_openvino(self, capabilities):
        capabilities(gpu=True)  # openvino=False: no NPU, no ov lib
        assert fingerprint() == "engine=ggml-dispatch-1;ov=none;dev=GPU/CPU"


class _FakeVulkan:
    """A Vulkan loader enumerating one physical device per entry of `device_types`."""

    def __init__(self, device_types):
        self._count = len(device_types)
        self._types = iter(device_types)

    def vkCreateInstance(self, create_info, allocator, instance):
        return 0  # VK_SUCCESS

    def vkEnumeratePhysicalDevices(self, instance, count, devices):
        ctypes.cast(count, ctypes.POINTER(ctypes.c_uint32)).contents.value = self._count

    def vkGetPhysicalDeviceProperties(self, device, props):
        # deviceType is VkPhysicalDeviceProperties' fifth uint32.
        props[16:20] = next(self._types).to_bytes(4, "little")

    def vkDestroyInstance(self, instance, allocator):
        pass


class TestVulkanGpuPresent:
    def test_false_when_loader_missing(self):
        # Without a Vulkan loader the probe fails safe to False; auto-detect skips GPU.
        with patch("ctypes.CDLL", side_effect=OSError("libvulkan.so.1 not found")):
            assert _vulkan_gpu_present() is False

    @pytest.mark.parametrize(
        ("device_types", "present"),
        [
            # 0 is "other", what an unwritten properties struct reads as.
            ([0], False),
            # 1-3 are integrated, discrete and virtual GPUs.
            ([1], True),
            ([2], True),
            ([3], True),
            # 4 is a CPU renderer such as lavapipe; a GPU beside it still counts.
            ([4], False),
            ([4, 2], True),
            # A loader without a driver enumerates nothing.
            ([], False),
        ],
    )
    def test_only_a_real_gpu_counts(self, device_types, present):
        with patch("ctypes.CDLL", return_value=_FakeVulkan(device_types)):
            assert _vulkan_gpu_present() is present


class TestNpuUserDriverPresent:
    def test_needs_both_the_loader_and_the_driver(self):
        def load(soname):
            if soname == "libze_intel_npu.so.1":
                raise OSError(f"{soname} not found")

        with (
            patch("quoth.hardware.npu_driver_dir", return_value=None),
            patch("ctypes.CDLL", side_effect=load),
        ):
            assert _npu_user_driver_present() is False
        _npu_user_driver_present.cache_clear()
        with (
            patch("quoth.hardware.npu_driver_dir", return_value=None),
            patch("ctypes.CDLL") as cdll,
        ):
            assert _npu_user_driver_present() is True
        cdll.assert_has_calls(
            [call("libze_loader.so.1"), call("libze_intel_npu.so.1")], any_order=True
        )

    def test_loads_the_bundled_copies_where_the_package_ships_them(self):
        # They are off the system path, so loading them by soname would fail.
        with (
            patch("quoth.hardware.npu_driver_dir", return_value="/pkg/lib/npu"),
            patch("ctypes.CDLL") as cdll,
        ):
            assert _npu_user_driver_present() is True
        cdll.assert_has_calls(
            [
                call("/pkg/lib/npu/libze_loader.so.1"),
                call("/pkg/lib/npu/libze_intel_npu.so.1"),
            ],
            any_order=True,
        )
