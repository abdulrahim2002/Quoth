# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth._native`, the arch-generic native-library path discovery."""

import _ctypes
import fnmatch
import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import quoth
import quoth._native as native

_ROOT = Path(__file__).resolve().parents[1]


class TestLibDirs:
    def test_no_triplet_dirs_when_multiarch_unset(self):
        with patch("sysconfig.get_config_var", return_value=None):
            dirs = native._lib_dirs()
        assert dirs == ["/usr/local/lib", "/usr/lib", "/usr/lib64"]

    def test_multiarch_dirs_come_first(self):
        with patch("sysconfig.get_config_var", return_value="aarch64-linux-gnu"):
            dirs = native._lib_dirs()
        assert dirs == [
            "/usr/local/lib/aarch64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
            # The plain and lib64 fallbacks are always present, for Arch and Fedora.
            "/usr/local/lib",
            "/usr/lib",
            "/usr/lib64",
        ]


class TestTypelibAndLibPaths:
    def test_typelib_dirs_include_system_girepository(self):
        with patch("sysconfig.get_config_var", return_value="aarch64-linux-gnu"):
            dirs = native.layer_shell_typelib_dirs()
        assert "/usr/lib/aarch64-linux-gnu/girepository-1.0" in dirs

    @pytest.mark.parametrize(
        ("finder", "leaf"),
        [
            (native.layer_shell_lib_paths, "libgtk4-layer-shell.so"),
            (native.thorvg_lib_paths, "libthorvg-1.so"),
        ],
    )
    def test_lib_paths_target_the_so(self, finder, leaf):
        with patch("sysconfig.get_config_var", return_value="aarch64-linux-gnu"):
            paths = finder()
        assert f"/usr/lib/aarch64-linux-gnu/{leaf}" in paths
        assert all(p.endswith(f"/{leaf}") for p in paths)

    @pytest.mark.parametrize(
        ("finder", "leaf"),
        [
            (native.layer_shell_typelib_dirs, ""),
            (native.layer_shell_lib_paths, "/libgtk4-layer-shell.so"),
            (native.thorvg_lib_paths, "/libthorvg-1.so"),
            (native.openvino_lib_dirs, "/openvino"),
        ],
    )
    def test_bundled_dirs_come_first(self, finder, leaf):
        # The bundled copy, beside the installed package or in a dev checkout's
        # repo-root lib/, wins over any system or venv one: no distro ships thorvg,
        # and Ubuntu 24.04 and Pop!_OS ship no gtk4-layer-shell for a .deb to load.
        with (
            patch("sysconfig.get_config_var", return_value=None),
            patch("glob.glob", return_value=[]),
        ):
            found = finder()
        assert found[0].endswith(f"/quoth/lib{leaf}")  # installed
        assert found[1] == f"{_ROOT}/lib{leaf}"  # dev repo-root


class TestNpuDriverDir:
    def test_the_first_bundled_npu_dir_wins(self, tmp_path):
        installed, dev = tmp_path / "quoth" / "lib", tmp_path / "lib"
        (dev / "npu").mkdir(parents=True)
        with patch.object(native, "bundled_lib_dirs", return_value=[installed, dev]):
            assert native.npu_driver_dir() == str(dev / "npu")
            (installed / "npu").mkdir(parents=True)
            assert native.npu_driver_dir() == str(installed / "npu")

    def test_none_without_a_bundled_stack(self, tmp_path):
        # Every channel but the .deb uses the system's or its runtime's driver.
        with patch.object(native, "bundled_lib_dirs", return_value=[tmp_path]):
            assert native.npu_driver_dir() is None


class TestOpenvinoDirs:
    @pytest.mark.parametrize("from_source", [True, False])
    def test_globs_the_dev_venv_only_from_source(self, from_source):
        # An installed package must never search ~/openvino-env: a user could have
        # that directory unrelated to Quoth. The system openvino-* glob always runs.
        venv = "/home/u/openvino-env/lib/python3.12/site-packages/openvino/libs"

        def fake_glob(pattern):
            if "openvino-env" in pattern:
                assert "python3.*" in pattern  # version not pinned
                return [venv]
            if pattern.endswith("/openvino-*"):
                return ["/usr/lib/x86_64-linux-gnu/openvino-2026.0"]
            return []

        with (
            patch("sysconfig.get_config_var", return_value="x86_64-linux-gnu"),
            patch("glob.glob", side_effect=fake_glob),
            patch(
                "os.path.expanduser", side_effect=lambda p: p.replace("~", "/home/u")
            ),
            patch.object(native, "running_from_source", return_value=from_source),
        ):
            dirs = native.openvino_lib_dirs()

        assert [d for d in dirs if "openvino-env" in d] == (
            [venv] if from_source else []
        )
        assert "/usr/lib/x86_64-linux-gnu/openvino-2026.0" in dirs

    def test_core_lib_paths_target_libopenvino_so(self):
        # An unsorted listing, with OpenVINO's other libs beside the core.
        listing = [
            "/opt/ov/libs/libtbb.so.12",
            "/opt/ov/libs/libopenvino.so.2600",
            "/opt/ov/libs/libopenvino_c.so.2600",
            "/opt/ov/libs/libopenvino.so",
        ]
        with (
            patch.object(native, "openvino_lib_dirs", return_value=["/opt/ov/libs"]),
            patch("glob.glob", side_effect=lambda p: fnmatch.filter(listing, p)),
        ):
            paths = native.openvino_core_lib_paths()

        # Sorted, and every candidate is a libopenvino.so* file (for the preload).
        assert paths == [
            "/opt/ov/libs/libopenvino.so",
            "/opt/ov/libs/libopenvino.so.2600",
        ]


class TestTypelibBootstrap:
    def _run_with_typelib_path(self, current_path):
        typelib_dir = "/usr/lib/python3/dist-packages/quoth/lib"
        with (
            patch.dict(os.environ, {"GI_TYPELIB_PATH": current_path}),
            patch.object(
                native, "layer_shell_typelib_dirs", return_value=[typelib_dir]
            ),
            patch.object(native, "layer_shell_lib_paths", return_value=[]),
            patch.object(native, "openvino_core_lib_paths", return_value=[]),
            patch("os.path.isdir", return_value=True),
        ):
            runpy.run_path(quoth.__file__)
            result = os.environ["GI_TYPELIB_PATH"]
        return result

    def test_nested_girepository_path_does_not_match_parent(self):
        current = "/usr/lib/python3/dist-packages/quoth/lib/nested/girepository-1.0"
        result = self._run_with_typelib_path(current)

        assert result == f"/usr/lib/python3/dist-packages/quoth/lib:{current}"

    def test_exact_existing_path_is_not_duplicated(self):
        current = "/usr/lib/python3/dist-packages/quoth/lib"

        assert self._run_with_typelib_path(current) == current


class TestPreloadFailures:
    def _import_with_layer_shell_candidates(self, paths):
        with (
            patch.object(native, "layer_shell_lib_paths", return_value=paths),
            patch.object(native, "openvino_core_lib_paths", return_value=[]),
            patch.object(native, "npu_driver_dir", return_value=None),
        ):
            return runpy.run_path(quoth.__file__)

    def test_a_library_that_fails_to_load_is_reported_with_the_reason(self, tmp_path):
        broken = tmp_path / "libgtk4-layer-shell.so"
        broken.write_text("not a library")
        found = self._import_with_layer_shell_candidates([str(broken)])

        assert found["LAYER_SHELL_LIB"] is None
        [(path, reason)] = found["LOAD_FAILURES"]
        assert path == str(broken)
        assert reason

    def test_a_failure_another_candidate_replaces_is_not_reported(self, tmp_path):
        broken = tmp_path / "libgtk4-layer-shell.so"
        broken.write_text("not a library")
        loadable = _ctypes.__file__
        found = self._import_with_layer_shell_candidates([str(broken), loadable])

        assert found["LAYER_SHELL_LIB"] == loadable
        assert found["LOAD_FAILURES"] == []


class TestRunningFromSource:
    def test_true_in_the_checkout(self):
        # The suite runs from the source tree (pyproject.toml at the repo root).
        assert native.running_from_source() is True

    def test_false_without_pyproject(self):
        # An installed package: no pyproject.toml at the computed repo root.
        with patch("os.path.exists", return_value=False):
            assert native.running_from_source() is False


# numpy loads first, as in the app, so libstdc++ already throws through libgcc_s; every
# library loaded after GStreamer must then resolve _Unwind_* there too.
_UNWINDER_PROBE = """
import ctypes
import numpy
import quoth.audio

def resume(lib):
    return ctypes.cast(lib._Unwind_Resume, ctypes.c_void_p).value

print(resume(ctypes.CDLL(None)) == resume(ctypes.CDLL("libgcc_s.so.1")))
"""


def test_cxx_exceptions_keep_one_unwinder_once_gstreamer_loads():
    probe = subprocess.run(
        [sys.executable, "-c", _UNWINDER_PROBE],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout == "True\n"
