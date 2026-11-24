# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Arch-generic discovery of native libraries Quoth preloads at startup.

The gtk4-layer-shell typelib/`.so` and OpenVINO runtime libs live in distro- and
version-specific dirs, so candidate paths come from the platform: the multiarch
triplet (`sysconfig`) on Debian/Ubuntu, plain `lib`/`lib64` on Arch/Fedora, and
globs for the versioned OpenVINO dirs.

Pure path logic, stdlib only. Kept out of `__init__.py` (which must run before
any `gi` import) so it stays unit-testable.
"""

import glob
import os
import sysconfig


def _lib_dirs() -> list[str]:
    """Return this platform's standard shared-library dirs, most specific first."""
    multiarch = sysconfig.get_config_var("MULTIARCH")
    dirs: list[str] = []
    if multiarch:
        dirs += [f"/usr/local/lib/{multiarch}", f"/usr/lib/{multiarch}"]
    dirs += ["/usr/local/lib", "/usr/lib", "/usr/lib64"]
    return list(dict.fromkeys(dirs))  # dedupe, preserve order


def bundled_lib_dirs() -> list[str]:
    """Return the two vendored-lib dirs, the installed package's first.

    `quoth/lib` sits beside the installed package; the repo-root `lib/` of a dev
    checkout is where the Makefile stages bundled native libs.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(here, "lib"),  # installed: quoth/lib
        os.path.join(
            os.path.dirname(os.path.dirname(here)), "lib"
        ),  # dev: repo-root lib
    ]


def running_from_source() -> bool:
    """Whether Quoth runs from a source checkout rather than an installed package.

    A `pyproject.toml` at the repo root, two levels above this package, marks a
    checkout. This gates dev-only search paths, which a shipped install must never
    consult.
    """
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.exists(os.path.join(repo_root, "pyproject.toml"))


def layer_shell_typelib_dirs() -> list[str]:
    """Return candidate GI_TYPELIB_PATH dirs for `Gtk4LayerShell-1.0.typelib`.

    The bundled copy comes first, for distros lacking the package (e.g. Ubuntu 24.04,
    Pop!_OS), then the system girepository dirs.
    """
    return bundled_lib_dirs() + [f"{d}/girepository-1.0" for d in _lib_dirs()]


def layer_shell_lib_paths() -> list[str]:
    """Return candidate paths to preload `libgtk4-layer-shell.so`, bundled first."""
    return [f"{d}/libgtk4-layer-shell.so" for d in bundled_lib_dirs()] + [
        f"{d}/libgtk4-layer-shell.so" for d in _lib_dirs()
    ]


def thorvg_lib_paths() -> list[str]:
    """Return candidate paths to `libthorvg-1.so`, the ctypes-loaded Lottie renderer.

    The bundled copy comes first, then system installs. The `-1` is ThorVG's API
    generation, part of the soname. A system copy is only a courtesy for a source
    checkout; every channel bundles its own, because no distro ships one.
    """
    return [f"{d}/libthorvg-1.so" for d in bundled_lib_dirs()] + [
        f"{d}/libthorvg-1.so" for d in _lib_dirs()
    ]


def openvino_lib_dirs() -> list[str]:
    """Return the OpenVINO runtime-lib dirs, most-specific first.

    The bundled copies (`openvino/` under each `bundled_lib_dirs()` entry) come first,
    then, only from a source checkout, a dev build venv, then system installs. The
    dirs are version-globbed; callers glob within each and check existence, so a
    missing one costs nothing.
    """
    dirs = [os.path.join(d, "openvino") for d in bundled_lib_dirs()]
    if running_from_source():
        # Dev-only: the local OpenVINO build venv. Excluded from installed packages,
        # where a user might coincidentally have ~/openvino-env unrelated to Quoth.
        dirs += glob.glob(
            os.path.expanduser(
                "~/openvino-env/lib/python3.*/site-packages/openvino/libs"
            )
        )
    for d in _lib_dirs():
        dirs += glob.glob(f"{d}/openvino-*")
    return dirs


def npu_driver_dir() -> str | None:
    """Return the dir of a bundled Intel NPU driver stack, or None to use the system's.

    The .deb bundles the Level Zero loader and Intel's NPU user-mode driver with its
    compiler in `quoth/lib/npu`, since Debian and Ubuntu package no NPU driver.
    """
    dirs = (os.path.join(d, "npu") for d in bundled_lib_dirs())
    return next((d for d in dirs if os.path.isdir(d)), None)


def openvino_core_lib_paths() -> list[str]:
    """Return the paths to the OpenVINO core lib (`libopenvino.so*`), sorted per dir.

    The dirs are `openvino_lib_dirs()`. `libwhisper.so` (built `-DWHISPER_OPENVINO=1`)
    links this as a NEEDED dependency, so it must be preloaded before libwhisper: the
    dynamic loader does not honour a mid-process `LD_LIBRARY_PATH` change. The core's
    sibling libs (TBB, the device plugins) resolve via its own `$ORIGIN` rpath.
    """
    paths: list[str] = []
    for d in openvino_lib_dirs():
        paths += sorted(glob.glob(os.path.join(d, "libopenvino.so*")))
    return paths
