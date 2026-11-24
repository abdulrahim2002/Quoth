# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `scripts/check_whisper_model_boundary.py`, the malformed-model checker.

`ctypes.CDLL` is faked, so no native library loads.
"""

import ctypes
import os
import struct
import subprocess
from pathlib import Path

import pytest

from scripts import check_whisper_model_boundary as checker


class _Function:
    def __init__(self, callback):
        self._callback = callback

    def __call__(self, *args):
        return self._callback(*args)


def test_load_initializes_cpu_from_library_directory_before_model(
    monkeypatch, tmp_path
):
    events = []
    registered = set()
    library = tmp_path / "Quøth libs/libwhisper.so"
    model = tmp_path / "model.bin"

    native = type("Native", (), {})()
    native.ggml_backend_reg_by_name = _Function(
        lambda name: 1 if name.decode() in registered else None
    )

    def load_backends(path):
        events.append(("backends", path))
        registered.add("CPU")

    native.ggml_backend_load_all_from_path = _Function(load_backends)

    def load_model(path):
        assert "CPU" in registered
        events.append(("model", path))
        return 1

    native.whisper_init_from_file = _Function(load_model)
    native.whisper_free = _Function(lambda _context: None)
    monkeypatch.setattr(ctypes, "CDLL", lambda path: native)

    assert checker._load(library, model) == 0
    assert events == [
        ("backends", os.fsencode(library.resolve().parent)),
        ("model", os.fsencode(model)),
    ]


def test_missing_cpu_stops_before_model_load(monkeypatch, tmp_path):
    model_calls = []
    native = type("Native", (), {})()
    native.ggml_backend_reg_by_name = _Function(lambda _name: None)
    native.ggml_backend_load_all_from_path = _Function(lambda _path: None)
    native.whisper_init_from_file = _Function(
        lambda path: model_calls.append(path) or 1
    )
    native.whisper_free = _Function(lambda _context: None)
    monkeypatch.setattr(ctypes, "CDLL", lambda path: native)

    with pytest.raises(RuntimeError, match="CPU backend failed to register"):
        checker._load(tmp_path / "libwhisper.so", tmp_path / "model.bin")

    assert model_calls == []


def test_bare_soname_does_not_scan_working_directory(monkeypatch, tmp_path):
    load_calls = []
    native = type("Native", (), {})()
    native.ggml_backend_reg_by_name = _Function(lambda _name: None)
    native.ggml_backend_load_all_from_path = _Function(load_calls.append)
    monkeypatch.setattr(ctypes, "CDLL", lambda path: native)

    with pytest.raises(RuntimeError, match="does not identify"):
        checker._load(Path("libwhisper.so"), tmp_path / "model.bin")

    assert load_calls == []


def test_check_passes_when_only_the_corrupted_copy_is_rejected(tmp_path, monkeypatch):
    library = tmp_path / "libwhisper.so"
    model = tmp_path / "model.bin"
    library.touch()
    model.write_bytes(bytes(64))
    invalid_ftype = struct.pack("<i", 2**31 - 1)

    def load(_library, path):
        # Like whisper.cpp, refuse only the unknown ftype at byte 44.
        verdict = "rejected" if path.read_bytes()[44:48] == invalid_ftype else "loaded"
        return subprocess.CompletedProcess([], 0, stdout=f"{verdict}\n", stderr="")

    monkeypatch.setattr(checker, "_run_loader", load)

    assert checker._check(library, model) == 0


def test_check_fails_when_the_corrupted_copy_still_loads(tmp_path, monkeypatch):
    library = tmp_path / "libwhisper.so"
    model = tmp_path / "model.bin"
    library.touch()
    model.write_bytes(bytes(64))
    loaded = subprocess.CompletedProcess([], 0, stdout="loaded\n", stderr="")
    monkeypatch.setattr(checker, "_run_loader", lambda _library, _model: loaded)

    with pytest.raises(SystemExit, match="malformed model did not return"):
        checker._check(library, model)


def test_whisper_model_boundary_finds_bundled_openvino(tmp_path, monkeypatch):
    library = tmp_path / "quoth/lib/libwhisper.so"
    previous = tmp_path / "existing-libs"
    captured = {}

    def capture_run(*args, **kwargs):
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setenv("LD_LIBRARY_PATH", os.fspath(previous))
    monkeypatch.setattr(subprocess, "run", capture_run)

    checker._run_loader(library, tmp_path / "model.bin")

    assert captured["env"]["LD_LIBRARY_PATH"].split(os.pathsep) == [
        os.fspath(library.parent),
        os.fspath(library.parent / "openvino"),
        os.fspath(previous),
    ]
