# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Quoth: private voice dictation for Linux."""

__version__ = "0.1.0"
APP_ID = "dev.quoth.Quoth"
# Where the in-app support links send users who want to fund development.
SUPPORT_URL = "https://buymeacoffee.com/samuelbullard"
WEBSITE_URL = "https://quoth.dev"

# gtk4-layer-shell needs GI_TYPELIB_PATH to include its typelib directory, and its .so
# preloaded before libwayland-client, whose symbols it shims. Both must happen before
# any gi or GTK import, and app.py imports gi on load, so they happen here. See
# https://github.com/wmww/gtk4-layer-shell/blob/main/linking.md; quoth/_native.py
# derives the candidate paths from the platform.
import ctypes as _ctypes
import os as _os

from quoth._native import (
    layer_shell_lib_paths,
    layer_shell_typelib_dirs,
    npu_driver_dir,
    openvino_core_lib_paths,
)

_typelib_dir = next((d for d in layer_shell_typelib_dirs() if _os.path.isdir(d)), None)
if _typelib_dir:
    _current_typelib = _os.environ.get("GI_TYPELIB_PATH", "")
    if _typelib_dir not in _current_typelib.split(_os.pathsep):
        _os.environ["GI_TYPELIB_PATH"] = (
            f"{_typelib_dir}:{_current_typelib}" if _current_typelib else _typelib_dir
        )


# Libraries that exist but failed to load, with no candidate loading in their place,
# as (path, reason) for app startup to log.
LOAD_FAILURES: list[tuple[str, str]] = []


def _preload(paths):
    """Preload the first loadable library in `paths` and return its path, or None.

    RTLD_GLOBAL lets its symbols satisfy later-loaded libraries' NEEDED dependencies,
    which is how a dependency in a non-default location resolves: the dynamic loader
    does not honour a mid-process LD_LIBRARY_PATH change.
    """
    _failures = []
    for _p in paths:
        if _os.path.exists(_p):
            try:
                _ctypes.CDLL(_p, mode=_ctypes.RTLD_GLOBAL)
                return _p
            except OSError as _e:
                _failures.append((_p, str(_e)))
    LOAD_FAILURES.extend(_failures)
    return None


# Preload gtk4-layer-shell before anything loads libwayland-client.
_loaded_layer_shell = _preload(layer_shell_lib_paths())

# A libwhisper.so built with -DWHISPER_OPENVINO=1 needs OpenVINO core for the Intel-NPU
# encoder. Preloading it by absolute path resolves it without LD_LIBRARY_PATH; its
# siblings (TBB, device plugins) resolve through OpenVINO's own $ORIGIN. It is absent
# on CPU and Vulkan-only builds, which do not link it.
_loaded_openvino = _preload(openvino_core_lib_paths())

# A bundled NPU driver stack is off the system library path. OpenVINO loads the Level
# Zero loader by soname, which then finds this preloaded copy, and the loader takes
# drivers only from the system path unless ZE_ENABLE_ALT_DRIVERS names them.
_npu_driver_dir = npu_driver_dir()
if _npu_driver_dir:
    _preload([_os.path.join(_npu_driver_dir, "libze_loader.so.1")])
    _os.environ.setdefault(
        "ZE_ENABLE_ALT_DRIVERS", _os.path.join(_npu_driver_dir, "libze_intel_npu.so.1")
    )

# Make libgcc's unwinder global before GStreamer loads libunwind, which on Debian and
# Ubuntu exports the same _Unwind_* functions: a C++ exception inside whisper.cpp then
# unwinds through one of them instead of both, which crashes.
_ctypes.CDLL("libgcc_s.so.1", mode=_ctypes.RTLD_GLOBAL)

# What discovery found, for app startup to log once the sinks exist. Nothing
# is logged here: log.py's sinks and filters do not exist yet. Importing the package
# must stay cheap, as the keybinding's `quoth --toggle` pays for it on every press.
TYPELIB_DIR = _typelib_dir
LAYER_SHELL_LIB = _loaded_layer_shell
OPENVINO_LIB = _loaded_openvino
NPU_DRIVER_DIR = _npu_driver_dir
