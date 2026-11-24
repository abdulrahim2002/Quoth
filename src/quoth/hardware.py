# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Hardware detection for NPU, GPU and CPU inference devices.

`available_devices()` lists the devices runnable in this build, in presence order
rather than by speed; `resolution` and `accel` choose the runtime device. A device is
offered only if its backend can actually run here: a Vulkan GPU must enumerate, and the
NPU needs an Intel NPU device node, the shipped OpenVINO runtime and Intel's NPU driver
libraries. This is a leaf module (it imports only `_native`), so `resolution` can depend
on it.
"""

import ctypes
import os
from functools import lru_cache
from pathlib import Path

from loguru import logger

from quoth._native import npu_driver_dir, openvino_core_lib_paths

# VkPhysicalDeviceType values that denote a real GPU (integrated / discrete /
# virtual). 4 = CPU/software (e.g. lavapipe), which does not count as a GPU.
_VK_GPU_DEVICE_TYPES = frozenset({1, 2, 3})
_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
# deviceType is the 5th uint32 of VkPhysicalDeviceProperties (apiVersion,
# driverVersion, vendorID, deviceID, deviceType), at byte offset 16.
_VK_DEVICE_TYPE_OFFSET = 16
_VK_PROPERTIES_BUF_SIZE = 1024  # >= sizeof(VkPhysicalDeviceProperties) (~824 B)

# The kernel drivers Intel's Level Zero NPU driver accepts. Other NPUs, such as AMD's
# amdxdna, create the same /dev/accel nodes, and OpenVINO cannot drive them.
_NPU_DRIVERS = frozenset({"intel_vpu", "intel_npu"})
_ACCEL_CLASS = "/sys/class/accel"
_ACCEL_DEV = "/dev/accel"


@lru_cache(maxsize=1)
def _vulkan_gpu_present() -> bool:
    """Whether the Vulkan loader enumerates at least one real (non-CPU) GPU.

    A `/dev/dri` render node can exist for a GPU whose Vulkan driver cannot run it, so
    this enumerates devices rather than trust the node. Any failure returns False.
    Memoised, since the call loads the GPU drivers and the answer is constant per
    session.
    """
    try:
        vk = ctypes.CDLL("libvulkan.so.1")
    except OSError:
        logger.debug("Vulkan loader (libvulkan.so.1) not found: no GPU backend")
        return False

    # Minimal zeroed VkInstanceCreateInfo (64 B on LP64); only sType is required.
    create_info = (ctypes.c_size_t * 8)()
    create_info[0] = _VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
    instance = ctypes.c_void_p()
    try:
        if (
            vk.vkCreateInstance(ctypes.byref(create_info), None, ctypes.byref(instance))
            != 0
        ):
            logger.debug("vkCreateInstance failed: no usable Vulkan")
            return False
    except Exception as e:
        logger.debug("Vulkan instance creation raised ({}); treating as no GPU", e)
        return False

    try:
        count = ctypes.c_uint32(0)
        vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None)
        if count.value == 0:
            logger.debug("Vulkan enumerated 0 physical devices: no GPU backend")
            return False
        devices = (ctypes.c_void_p * count.value)()
        vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), devices)
        props = ctypes.create_string_buffer(_VK_PROPERTIES_BUF_SIZE)
        for dev in devices[: count.value]:
            vk.vkGetPhysicalDeviceProperties(ctypes.c_void_p(dev), props)
            device_type = int.from_bytes(
                props.raw[_VK_DEVICE_TYPE_OFFSET : _VK_DEVICE_TYPE_OFFSET + 4],
                "little",
            )
            if device_type in _VK_GPU_DEVICE_TYPES:
                logger.debug("Vulkan GPU present (device_type={})", device_type)
                return True
        logger.debug(
            "Vulkan enumerated {} device(s), none a real GPU (software/CPU only)",
            count.value,
        )
        return False
    except Exception as e:
        logger.debug("Vulkan device enumeration failed ({}); treating as no GPU", e)
        return False
    finally:
        try:
            vk.vkDestroyInstance(instance, None)
        except Exception as e:
            logger.debug("vkDestroyInstance failed: {}", e)


@lru_cache(maxsize=1)
def _npu_user_driver_present() -> bool:
    """Whether the Level Zero loader and Intel's NPU user-mode driver load.

    OpenVINO's NPU plugin loads the loader, which loads the driver: the package's
    bundled copies where it ships them (`quoth/__init__` points both there), else the
    system's by soname. Memoised: the answer is constant per session.
    """
    bundled = npu_driver_dir()
    for soname in ("libze_loader.so.1", "libze_intel_npu.so.1"):
        try:
            ctypes.CDLL(os.path.join(bundled, soname) if bundled else soname)
        except OSError as e:
            logger.debug("No Intel NPU driver: {}", e)
            return False
    return True


def _npu_present() -> bool:
    """Whether an Intel NPU is usable here.

    An accel node bound to Intel's NPU kernel driver must be readable and writable by
    this process, and the OpenVINO runtime and Intel's user-mode driver must be there to
    drive it: a sandbox can expose the node alone, and a native install can lack the
    driver.
    """
    try:
        nodes = os.listdir(_ACCEL_CLASS)
    except OSError:
        _npu_rejected("NPU: no accel subsystem ({} unreadable)", _ACCEL_CLASS)
        return False
    for node in nodes:
        driver = Path(_ACCEL_CLASS, node, "device", "driver").resolve().name
        device = Path(_ACCEL_DEV, node)
        if driver not in _NPU_DRIVERS:
            _npu_rejected("NPU {}: driver {} not supported", node, driver)
        elif not os.access(device, os.R_OK | os.W_OK):
            _npu_rejected("NPU {}: cannot read and write {}", node, device)
        elif not openvino_core_lib_paths():
            _npu_rejected("NPU {}: OpenVINO core not found", node)
            return False
        else:
            return _npu_user_driver_present()
    return False


# Every resolve re-runs the NPU probe, so each reason it is not offered logs only once.
_npu_reasons_logged: set[str] = set()


def _npu_rejected(message: str, *args: object) -> None:
    reason = message.format(*args)
    if reason not in _npu_reasons_logged:
        _npu_reasons_logged.add(reason)
        logger.debug(message, *args)


# Log-on-change guard for available_devices: the runnable set is constant per session,
# but every resolve re-evaluates it (via fingerprint), so log it once, not per call.
_last_logged_devices: list[str] | None = None


def available_devices() -> list[str]:
    """Return the inference devices runnable in this build, in capability order.

    The order is NPU, GPU, CPU by presence, not a speed ranking: `accel.resolve_backend`
    picks the runtime device, preferring GPU by default and following calibration.
    """
    global _last_logged_devices
    devices: list[str] = []
    if _npu_present():
        devices.append("NPU")
    if _vulkan_gpu_present():
        devices.append("GPU")
    devices.append("CPU")
    if devices != _last_logged_devices:
        logger.debug("Available devices (capability order): {}", devices)
        _last_logged_devices = devices
    return devices


def fingerprint() -> str:
    """Key calibration by inference policy, OpenVINO soname and runnable devices.

    This does not detect every hardware or driver change. OpenVINO validates its
    own compiled blobs independently of these coarse measurement and warm markers.
    """
    ov = (
        ",".join(sorted({os.path.basename(p) for p in openvino_core_lib_paths()}))
        or "none"
    )
    return f"engine=ggml-dispatch-1;ov={ov};dev={'/'.join(available_devices())}"
