#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Exercise whisper.cpp's malformed-model boundary in bounded subprocesses."""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

_FTYPE_OFFSET = 44
_INVALID_FTYPE = 2**31 - 1
_TIMEOUT_SECONDS = 30


def _ensure_cpu_backend(native: ctypes.CDLL, library: Path) -> None:
    native.ggml_backend_reg_by_name.argtypes = [ctypes.c_char_p]
    native.ggml_backend_reg_by_name.restype = ctypes.c_void_p
    native.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
    native.ggml_backend_load_all_from_path.restype = None
    if native.ggml_backend_reg_by_name(b"CPU"):
        return
    if not os.path.dirname(os.fspath(library)):
        raise RuntimeError("library path does not identify its backend directory")
    native.ggml_backend_load_all_from_path(os.fsencode(library.resolve().parent))
    if not native.ggml_backend_reg_by_name(b"CPU"):
        raise RuntimeError("whisper CPU backend failed to register")


def _load(library: Path, model: Path) -> int:
    native = ctypes.CDLL(str(library))
    _ensure_cpu_backend(native, library)
    native.whisper_init_from_file.argtypes = [ctypes.c_char_p]
    native.whisper_init_from_file.restype = ctypes.c_void_p
    native.whisper_free.argtypes = [ctypes.c_void_p]
    native.whisper_free.restype = None
    context = native.whisper_init_from_file(os.fsencode(model))
    if context:
        native.whisper_free(context)
        print("loaded")
    else:
        print("rejected")
    return 0


def _run_loader(library: Path, model: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    lib_dirs = (library.parent, library.parent / "openvino")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [os.fspath(path) for path in lib_dirs]
        + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
    )
    return subprocess.run(
        [sys.executable, "-I", __file__, "--load-only", str(library), str(model)],
        env=env,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_SECONDS,
    )


def _check(library: Path, model: Path) -> int:
    if not library.is_file():
        raise SystemExit(f"native library not found: {library}")
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")

    with tempfile.TemporaryDirectory(prefix="quoth-whisper-boundary-") as temp:
        malformed = Path(temp) / model.name
        shutil.copyfile(model, malformed)
        with malformed.open("r+b") as stream:
            stream.seek(_FTYPE_OFFSET)
            stream.write(struct.pack("<i", _INVALID_FTYPE))

        valid = _run_loader(library, model)
        if valid.returncode != 0 or valid.stdout.strip() != "loaded":
            raise SystemExit(
                "valid model failed to load\n" + valid.stdout + valid.stderr
            )

        invalid = _run_loader(library, malformed)
        if invalid.returncode != 0 or invalid.stdout.strip() != "rejected":
            raise SystemExit(
                "malformed model did not return a clean rejection\n"
                + invalid.stdout
                + invalid.stderr
            )

    print("OK: valid model loads and malformed ftype is rejected")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("library", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    if args.load_only:
        return _load(args.library, args.model)
    return _check(args.library.resolve(), args.model.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
