# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `scripts/check_install.py`.

`_fake_install` builds a throwaway venv laid out like one channel's installed package.
"""

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest

from scripts import check_install as checker
from tests.support import compile_commands

_ROOT = Path(__file__).resolve().parents[1]
_CHECK_INSTALL = _ROOT / "scripts" / "check_install.py"
_FAKE_GI = _ROOT / "tests" / "fixtures" / "fake-gi"
# The licence texts the Flatpak's NPU module installs, each one the check must require.
_FLATPAK_NPU_NOTICES = re.findall(
    r"install -m644 \S+ /app/share/licenses/(\S+)",
    (_ROOT / "packaging" / "flatpak" / "openvino-npu.yaml").read_text(),
)
_CANDIDATE_RELEASE = f"quoth@0.1.0+g{'0' * 40}"
_CPU_MODULES = (
    "libggml-cpu-x64.so",
    "libggml-cpu-sse42.so",
    "libggml-cpu-sandybridge.so",
    "libggml-cpu-ivybridge.so",
    "libggml-cpu-piledriver.so",
    "libggml-cpu-haswell.so",
    "libggml-cpu-skylakex.so",
    "libggml-cpu-cannonlake.so",
    "libggml-cpu-cascadelake.so",
    "libggml-cpu-icelake.so",
    "libggml-cpu-cooperlake.so",
    "libggml-cpu-zen4.so",
    "libggml-cpu-alderlake.so",
    "libggml-cpu-sapphirerapids.so",
)
_NATIVE_BUNDLE_FILES = (
    "libwhisper.so",
    "libwhisper_helper.so",
    "libggml.so",
    "libggml-base.so",
    "libggml-vulkan.so",
    "LICENSE.whisper-cpp",
    "libthorvg-1.so",
    "LICENSE.thorvg",
    *_CPU_MODULES,
)


_THORVG_NOTICE = (
    "MIT\n"
    "Copyright (C) 2015 THL A29 Limited, a Tencent company, and Milo Yip.\n"
    "This software is based in part on the work of the FreeType Team\n"
    "This Source Code Form is subject to the terms of the Mozilla Public\n"
    "License, v. 2.0. If a copy of the MPL was not distributed with this\n"
    "file, You can obtain one at http://mozilla.org/MPL/2.0/.\n"
)


_WHISPER_NOTICE = (
    "MIT\n"
    "Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng\n"
    "Copyright 2015-2025 The Khronos Group Inc.\n"
)


def _elf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x7fELF" + b"\0" * 12)


def _versioned_elf(path: Path, *versions: str) -> None:
    """Write a minimal ELF64 file whose version needs ask libc for `versions`."""
    strings = b"\0libc.so.6\0"
    names = []
    for version in versions:
        names.append(len(strings))
        strings += version.encode() + b"\0"
    strings += b"\0" * (-len(strings) % 8)
    aux = b"".join(
        struct.pack("<IHHII", 0, 0, 0, name, 16 if index < len(names) - 1 else 0)
        for index, name in enumerate(names)
    )
    needs = struct.pack("<HHIII", 1, len(names), 1, 16, 0) + aux
    strings_offset = 64
    needs_offset = strings_offset + len(strings)
    sections_offset = needs_offset + len(needs)
    sections = (
        bytes(64)
        + struct.pack(
            "<IIQQQQIIQQ", 0, 3, 0, 0, strings_offset, len(strings), 0, 0, 1, 0
        )
        + struct.pack(
            "<IIQQQQIIQQ", 0, 0x6FFFFFFE, 2, 0, needs_offset, len(needs), 1, 1, 8, 0
        )
    )
    header = b"\x7fELF\x02\x01\x01" + bytes(9)
    header += struct.pack(
        "<HHIQQQIHHHHHH", 3, 62, 1, 0, 0, sections_offset, 0, 64, 0, 0, 64, 3, 0
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + strings + needs + sections)


def _readelf(bin_dir: Path, output: str = "", returncode: int = 0) -> None:
    script = bin_dir / "readelf"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"sys.stdout.write({output!r})\n"
        f"sys.exit({returncode})\n"
    )
    script.chmod(0o755)


def _run(root: Path, bin_dir: Path, *, pythonpath: Path | None = None):
    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    if pythonpath:
        env["PYTHONPATH"] = str(pythonpath)
    return subprocess.run(
        [sys.executable, str(_CHECK_INSTALL), "--elf-root", str(root)],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _command(path: Path, body: str) -> None:
    _write(path, f"#!{sys.executable}\n{body}")
    path.chmod(0o755)


def _fake_install(
    tmp_path: Path,
    channel: str,
    release: str = _CANDIDATE_RELEASE,
    environment_name: str = "candidate",
) -> tuple[Path, Path]:
    environment = tmp_path / "environment"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        check=True,
        capture_output=True,
        text=True,
    )
    python = environment / "bin" / "python"
    site = (
        environment
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    package = site / "quoth"
    # A real shared object for ldd and readelf: this process's libc, wherever it lives.
    libc = next(
        Path(line.split()[-1])
        for line in Path("/proc/self/maps").read_text().splitlines()
        if line.endswith("/libc.so.6")
    )
    lib_dir = package / "lib"
    lib_dir.mkdir(parents=True)
    shutil.copy2(libc, lib_dir / "libwhisper.so")
    for name in (
        "libwhisper_helper.so",
        "libthorvg-1.so",
        "libggml.so",
        "libggml-base.so",
        "libggml-vulkan.so",
        *_CPU_MODULES,
    ):
        os.link(lib_dir / "libwhisper.so", lib_dir / name)
    _write(lib_dir / "LICENSE.whisper-cpp", _WHISPER_NOTICE)
    _write(lib_dir / "LICENSE.thorvg", _THORVG_NOTICE)
    _write(package / "__init__.py", f"NPU_DRIVER_DIR = {str(lib_dir / 'npu')!r}\n")
    _write(
        package / "whisper.py",
        f"def find_libwhisper(): return {str(lib_dir / 'libwhisper.so')!r}\n"
        f"def find_libwhisper_helper(): return {str(lib_dir / 'libwhisper_helper.so')!r}\n"
        "def _ensure_cpu_backend(lib, lib_path): pass\n",
    )
    _write(
        package / "_native.py",
        f"def thorvg_lib_paths(): return [{str(lib_dir / 'libthorvg-1.so')!r}]\n",
    )
    _write(
        package / "audio.py",
        "class AudioPipeline:\n    def destroy(self): pass\n",
    )
    _write(package / "hardware.py", "def _npu_user_driver_present(): return True\n")
    _write(
        package / "crash_reports.py",
        "import sys\n"
        "from pathlib import Path\n"
        "def _add_vendor_path():\n"
        "    vendor = Path(__file__).parent / '_vendor'\n"
        "    if vendor.is_dir(): sys.path.insert(0, str(vendor))\n",
    )
    _write(
        package / "lib" / "build-identity.json",
        json.dumps({"release": release, "environment": environment_name}),
    )
    if channel != "flatpak":
        stamp = {"debian": "deb", "arch": "aur"}[channel]
        _write(package / "_channel.py", f"CHANNEL = {stamp!r}\n")
    _write(
        environment / "share/applications/dev.quoth.Quoth.desktop", "[Desktop Entry]\n"
    )
    _write(environment / "share/metainfo/dev.quoth.Quoth.metainfo.xml", "<component/>")
    _write(
        environment / "share/icons/hicolor/scalable/apps/dev.quoth.Quoth.svg", "<svg/>"
    )
    for icon in ("gauge", "bug", "heart", "coffee", "github"):
        _write(
            environment
            / "share"
            / "icons"
            / "hicolor"
            / "symbolic"
            / "apps"
            / f"dev.quoth.Quoth-{icon}-symbolic.svg",
            "<svg/>",
        )
    _write(environment / "share/licenses/quoth/LICENSE.lucide-icons", "ISC\n")
    for notice in ("catppuccin", "dracula", "nord", "silero-vad", "loguru"):
        source = _ROOT / "data" / "licenses" / f"LICENSE.{notice}"
        destination = environment / "share" / "licenses" / "quoth" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if channel == "flatpak":
        for notice in (
            "openvino/LICENSE",
            "wl-clipboard/COPYING",
            "wl-clipboard/PROTOCOL-NOTICES",
            "libportal/COPYING",
            "wtype/LICENSE",
            "wtype/PROTOCOL-NOTICES",
            "ydotool/LICENSE",
            "pygobject/COPYING",
            "intel-npu-driver/LICENSE.md",
            "intel-npu-driver/third-party-programs.txt",
            "intel-npu-compiler/LICENSE",
            "intel-npu-compiler/LICENSE.llvm",
            "level-zero/LICENSE",
        ):
            _write(environment / "share/licenses" / notice, "license\n")
        # The NPU driver, its loader and its compiler, in the Flatpak's own lib dir.
        for name in (
            "libopenvino_intel_npu_compiler.so",
            "libze_intel_npu.so.1",
            "libze_loader.so.1",
        ):
            os.link(lib_dir / "libwhisper.so", environment / "lib" / name)
    sandboxed = channel == "flatpak"
    # The Flatpak's helpers run on the host through flatpak-spawn: `host` stands for
    # that view of /app/bin, whose ELF files the sandbox inspects.
    host_bin = environment / ("host" if sandboxed else "bin")
    if sandboxed:
        for tool in ("wl-copy", "wl-paste", "wtype", "ydotool", "ydotoold"):
            _versioned_elf(environment / "bin" / tool, "GLIBC_2.2.5", "GLIBC_2.34")
    _write(
        package / "typer.py",
        "import os\n"
        f"_PACKAGE_BIN = {str(package / 'lib' / 'bin')!r}\n"
        "class _ExecutionContext:\n"
        f"    sandboxed = {sandboxed!r}\n"
        f"    host_bin = {str(host_bin)!r}\n"
        "    @classmethod\n"
        "    def detect(cls): return cls()\n"
        "    def wrap(self, tool, args, env=None):\n"
        "        command = f'{self.host_bin}/{tool}' if self.sandboxed else tool\n"
        "        bundled = os.path.join(_PACKAGE_BIN, tool)\n"
        "        if not self.sandboxed and os.path.isfile(bundled):\n"
        "            command = bundled\n"
        "        return [command, *args]\n",
    )
    _write(package / "backgrounds" / "onboarding.webm", "vp9")
    _write(site / "certifi" / "__init__.py")
    _write(site / "urllib3" / "__init__.py", '__version__ = "2.8.0"\n')
    shutil.copytree(_FAKE_GI, site / "gi")
    if channel == "flatpak":
        # The check requires the Flatpak's own PyGObject, under /app.
        with (site / "gi" / "__init__.py").open("a") as stream:
            stream.write(
                "__file__ = '/app/lib/python3/site-packages/gi/__init__.py'\n"
                "class _gi: __file__ = '/app/lib/python3/site-packages/gi/_gi.so'\n"
            )
    sentry_root = package / "_vendor" if channel == "debian" else site
    sentry_version = "2.73.0" if channel == "arch" else "2.72.0"
    _write(
        sentry_root / "sentry_sdk" / "__init__.py",
        f'VERSION = "{sentry_version}"\n'
        "def init(*args, **kwargs): raise AssertionError('checker initialized Sentry')\n",
    )
    if channel == "debian":
        _write(sentry_root / "LICENSE.sentry-sdk", "MIT\n")
    # The .deb bundles wl-clipboard and ydotool beside its native libs; others use the
    # PATH ones.
    tool_bin = package / "lib" / "bin" if channel == "debian" else host_bin
    _command(tool_bin / "wl-copy", "print('wl-clipboard 2.3.0')\n")
    if channel == "debian":
        _write(
            package / "lib" / "LICENSE.wl-clipboard",
            "GPL-3.0-or-later\nCopyright © 2018 Simon Ser\n",
        )
        _write(package / "lib" / "LICENSE.ydotool", "AGPL-3.0-or-later\n")
        _command(tool_bin / "ydotoold", "")
    _command(
        tool_bin / "ydotool",
        "print('Usage: ydotool <cmd> <args>\\nAvailable commands:"
        "\\n  click\\n  debug\\n  key\\n  mousemove\\n  type')\n",
    )
    _command(host_bin / "wtype", "raise SystemExit(1)\n")
    return python, package


def _run_contract(
    python: Path,
    channel: str,
    release: str = _CANDIDATE_RELEASE,
    environment_name: str = "candidate",
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = f"{python.parent}:{env['PATH']}"
    return subprocess.run(
        [
            python,
            "-I",
            _CHECK_INSTALL,
            "--package-channel",
            channel,
            "--expected-release",
            release,
            "--expected-environment",
            environment_name,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )


def test_elf_root_accepts_needed_and_origin_runpaths(tmp_path):
    root = tmp_path / "root"
    _elf(root / "libone.so")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _readelf(
        bin_dir,
        " 0x1 (NEEDED) Shared library: [libc.so.6]\n"
        " 0x2 (RPATH) Library rpath: [$ORIGIN]\n"
        " 0x3 (RUNPATH) Library runpath: [${ORIGIN}/openvino]\n",
    )

    result = _run(root, bin_dir)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT: PASS" in result.stdout


@pytest.mark.parametrize("marker_kind", ["file", "dangling-symlink"])
def test_elf_root_rejects_partial_native_build(tmp_path, marker_kind):
    root = tmp_path / "root"
    _elf(root / "libgeneric.so")
    marker = root / "libggml.so"
    if marker_kind == "file":
        _elf(marker)
    else:
        marker.symlink_to("missing")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _readelf(bin_dir)

    result = _run(root, bin_dir)

    assert result.returncode == 1
    assert "FAIL missing libwhisper.so" in result.stdout
    assert "RESULT: FAIL" in result.stdout


def test_native_bundle_inventory_matches_pinned_whisper_module_family(tmp_path):
    assert checker._CPU_MODULES == _CPU_MODULES
    assert checker._NATIVE_BUNDLE_FILES == _NATIVE_BUNDLE_FILES
    for name in checker._NATIVE_BUNDLE_FILES:
        _write(tmp_path / name)

    assert checker._native_bundle_errors(tmp_path, "arch") == []


@pytest.mark.parametrize("missing", _NATIVE_BUNDLE_FILES)
def test_native_bundle_inventory_rejects_missing_file(tmp_path, missing):
    for name in checker._NATIVE_BUNDLE_FILES:
        _write(tmp_path / name)
    (tmp_path / missing).unlink()

    assert checker._native_bundle_errors(tmp_path, "arch") == [f"missing {missing}"]


def test_bundled_layer_shell_carries_its_notice(tmp_path):
    for name in checker._NATIVE_BUNDLE_FILES:
        _write(tmp_path / name)
    _write(tmp_path / "libgtk4-layer-shell.so")

    assert checker._native_bundle_errors(tmp_path, "debian") == [
        "missing LICENSE.gtk4-layer-shell"
    ]
    _write(tmp_path / "LICENSE.gtk4-layer-shell")
    assert checker._native_bundle_errors(tmp_path, "debian") == []


def test_bundled_wl_clipboard_carries_its_notice(tmp_path):
    for name in checker._NATIVE_BUNDLE_FILES:
        _write(tmp_path / name)
    _write(tmp_path / "bin" / "wl-copy")

    assert checker._native_bundle_errors(tmp_path, "debian") == [
        "missing LICENSE.wl-clipboard"
    ]
    _write(tmp_path / "LICENSE.wl-clipboard")
    assert checker._native_bundle_errors(tmp_path, "debian") == []


def test_bundled_ydotool_carries_its_notice_and_daemon(tmp_path):
    for name in checker._NATIVE_BUNDLE_FILES:
        _write(tmp_path / name)
    _write(tmp_path / "bin" / "ydotool")

    assert checker._native_bundle_errors(tmp_path, "debian") == [
        "missing bin/ydotoold",
        "missing LICENSE.ydotool",
    ]
    _write(tmp_path / "bin" / "ydotoold")
    _write(tmp_path / "LICENSE.ydotool")
    assert checker._native_bundle_errors(tmp_path, "debian") == []


def test_bundled_openvino_carries_its_notice(tmp_path):
    for name in checker._NATIVE_BUNDLE_FILES:
        _write(tmp_path / name)
    _write(tmp_path / "openvino" / "libopenvino.so")

    assert checker._native_bundle_errors(tmp_path, "arch") == [
        "missing openvino/licenses/LICENSE"
    ]
    # The Flatpak keeps the notice under share/licenses, checked with its modules'.
    assert checker._native_bundle_errors(tmp_path, "flatpak") == []
    _write(tmp_path / "openvino" / "licenses" / "LICENSE")
    assert checker._native_bundle_errors(tmp_path, "arch") == []


def test_bundled_npu_driver_carries_its_notices(tmp_path):
    for name in checker._NATIVE_BUNDLE_FILES:
        _write(tmp_path / name)
    _write(tmp_path / "npu" / "libze_intel_npu.so.1")
    notices = [
        "npu/licenses/intel-npu-driver/LICENSE.md",
        "npu/licenses/intel-npu-driver/third-party-programs.txt",
        "npu/licenses/intel-npu-compiler/LICENSE",
        "npu/licenses/intel-npu-compiler/LICENSE.llvm",
        "npu/licenses/level-zero/LICENSE",
    ]

    assert checker._native_bundle_errors(tmp_path, "debian") == [
        f"missing {notice}" for notice in notices
    ]
    for notice in notices:
        _write(tmp_path / notice)
    assert checker._native_bundle_errors(tmp_path, "debian") == []


def test_native_bundle_inventory_rejects_stale_monolithic_cpu(tmp_path):
    for name in checker._NATIVE_BUNDLE_FILES:
        _write(tmp_path / name)
    (tmp_path / "libggml-cpu.so").symlink_to("missing")

    assert checker._native_bundle_errors(tmp_path, "arch") == [
        "obsolete libggml-cpu.so is present"
    ]


@pytest.mark.parametrize(
    "name",
    ["libggml-cpu-unknown.so", "libggml-cpu-sse42.so.0", "libggml-cpu-old.so.bak"],
)
def test_native_bundle_inventory_rejects_unexpected_cpu_module(tmp_path, name):
    for required in checker._NATIVE_BUNDLE_FILES:
        _write(tmp_path / required)
    (tmp_path / name).symlink_to("missing")

    assert checker._native_bundle_errors(tmp_path, "arch") == [
        f"unexpected CPU module {name}"
    ]


def test_compile_commands_accept_pinned_variant_flags(tmp_path):
    path = tmp_path / "compile_commands.json"
    compile_commands(path)

    assert checker._compile_command_errors(path) == []


@pytest.mark.parametrize(
    ("target", "flag"),
    [
        ("ggml-base", "-march=native"),
        ("ggml-cpu-x64", "-mavx2"),
        ("ggml-cpu-haswell-feats", "-msse4.2"),
        ("ggml-cpu-x64-feats", "-mtune=native"),
        ("whisper", "-mcpu=native"),
        ("ggml-cpu-haswell", "-march=native"),
        ("ggml", "-mssse3"),
        ("ggml", "-mpopcnt"),
        ("ggml", "-mlzcnt"),
        ("ggml", "-maes"),
        ("ggml", "-mfuture-isa"),
    ],
)
def test_compile_commands_reject_unsafe_cpu_flags(tmp_path, target, flag):
    path = tmp_path / "compile_commands.json"
    rows = compile_commands(path)
    row = next(item for item in rows if f"CMakeFiles/{target}.dir/" in str(item))
    if "arguments" in row:
        row["arguments"].insert(1, flag)
    else:
        row["command"] = f"c++ {flag} " + row["command"].removeprefix("c++ ")
    path.write_text(json.dumps(rows))

    assert f"{target} uses unsafe CPU flag {flag}" in checker._compile_command_errors(
        path
    )


def test_compile_commands_require_every_baseline_scorer(tmp_path):
    path = tmp_path / "compile_commands.json"
    rows = compile_commands(path)
    rows = [
        row
        for row in rows
        if "CMakeFiles/ggml-cpu-alderlake-feats.dir/"
        not in " ".join(row.get("arguments", []))
    ]
    path.write_text(json.dumps(rows))

    assert "missing compile target ggml-cpu-alderlake-feats" in (
        checker._compile_command_errors(path)
    )


def test_compile_commands_require_no_lto_for_scorers(tmp_path):
    path = tmp_path / "compile_commands.json"
    rows = compile_commands(path)
    row = next(
        item
        for item in rows
        if "CMakeFiles/ggml-cpu-x64-feats.dir/" in " ".join(item.get("arguments", []))
    )
    row["arguments"].remove("-fno-lto")
    path.write_text(json.dumps(rows))

    assert "ggml-cpu-x64-feats scorer is missing -fno-lto" in (
        checker._compile_command_errors(path)
    )


def test_compile_commands_mode_fails_the_build_on_an_unsafe_flag(tmp_path):
    # build_whisper.sh stops on this exit status before compiling anything.
    path = tmp_path / "compile_commands.json"
    rows = compile_commands(path)
    row = next(item for item in rows if "CMakeFiles/ggml.dir/" in str(item))
    row["arguments"].insert(1, "-mavx2")
    path.write_text(json.dumps(rows))

    result = subprocess.run(
        [sys.executable, str(_CHECK_INSTALL), "--compile-commands", str(path)],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 1
    assert "FAIL ggml uses unsafe CPU flag -mavx2" in result.stdout


def test_native_bundle_inventory_rejects_external_notice_symlink(tmp_path):
    for name in checker._NATIVE_BUNDLE_FILES:
        _write(tmp_path / name)
    outside = tmp_path.parent / "LICENSE.whisper-cpp"
    _write(outside)
    (tmp_path / "LICENSE.whisper-cpp").unlink()
    (tmp_path / "LICENSE.whisper-cpp").symlink_to(outside)

    assert checker._native_bundle_errors(tmp_path, "arch") == [
        f"LICENSE.whisper-cpp resolves outside {tmp_path}"
    ]


def test_elf_root_does_not_import_quoth(tmp_path):
    root = tmp_path / "root"
    _elf(root / "libone.so")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _readelf(bin_dir)
    poison = tmp_path / "poison"
    package = poison / "quoth"
    package.mkdir(parents=True)
    marker = tmp_path / "imported"
    (package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    )

    result = _run(root, bin_dir, pythonpath=poison)

    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    "dynamic_entry",
    [
        "(RPATH) Library rpath: [/root/build]",
        "(RUNPATH) Library runpath: [/root/build]",
        "(RUNPATH) Library runpath: [/tmp/build]",
        "(RUNPATH) Library runpath: [relative/lib]",
        "(RUNPATH) Library runpath: [$ORIGINATED/lib]",
        "(RUNPATH) Library runpath: [$ORIGIN]:/tmp/build]",
    ],
)
def test_elf_root_rejects_unsafe_dynamic_paths(tmp_path, dynamic_entry):
    root = tmp_path / "root"
    _elf(root / "libbad.so")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _readelf(bin_dir, f" 0x1 {dynamic_entry}\n")

    result = _run(root, bin_dir)

    assert result.returncode == 1
    assert "RESULT: FAIL" in result.stdout
    assert "libbad.so" in result.stdout


def test_elf_root_fails_when_readelf_fails(tmp_path):
    root = tmp_path / "root"
    _elf(root / "libbad.so")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _readelf(bin_dir, returncode=7)

    result = _run(root, bin_dir)

    assert result.returncode == 1
    assert "readelf failed" in result.stdout
    assert "RESULT: FAIL" in result.stdout


def test_elf_root_fails_without_readelf(tmp_path):
    root = tmp_path / "root"
    _elf(root / "libone.so")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    result = _run(root, bin_dir)

    assert result.returncode == 1
    assert "readelf not found" in result.stdout
    assert "RESULT: FAIL" in result.stdout


@pytest.mark.parametrize("kind", ["missing", "empty", "directory"])
def test_elf_root_fails_without_an_elf_shared_object(tmp_path, kind):
    root = tmp_path / "root"
    if kind != "missing":
        root.mkdir()
    if kind == "directory":
        (root / "libfake.so").mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _readelf(bin_dir)

    result = _run(root, bin_dir)

    assert result.returncode == 1
    assert "RESULT: FAIL" in result.stdout


def test_installed_smoke_records_failed_readelf_as_failure(tmp_path):
    package_root = tmp_path / "packages"
    quoth = package_root / "quoth"
    quoth.mkdir(parents=True)
    native_lib = quoth / "libwhisper.so"
    _elf(native_lib)
    (quoth / "__init__.py").write_text("")
    (quoth / "whisper.py").write_text(
        f"def find_libwhisper(): return {str(native_lib)!r}\n"
        f"def find_libwhisper_helper(): return {str(native_lib)!r}\n"
    )
    (quoth / "_native.py").write_text("def thorvg_lib_paths(): return []\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _readelf(bin_dir, returncode=7)
    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["PYTHONPATH"] = str(package_root)

    result = subprocess.run(
        [sys.executable, str(_CHECK_INSTALL)],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert result.returncode == 1
    assert "FAIL inspect libwhisper dependencies: readelf failed" in result.stdout
    assert "Vulkan-only build" not in result.stdout


@pytest.mark.parametrize(
    "args",
    [
        ["--package-channel", "debian"],
        ["--expected-release", "quoth@0.1.0"],
        ["--expected-environment", "production"],
    ],
)
def test_package_contract_arguments_must_be_complete(args):
    result = subprocess.run(
        [sys.executable, str(_CHECK_INSTALL), *args],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "must be set together" in result.stderr


def test_package_contract_cannot_be_combined_with_elf_mode(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(_CHECK_INSTALL),
            "--elf-root",
            str(tmp_path),
            "--package-channel",
            "debian",
            "--expected-release",
            "quoth@0.1.0",
            "--expected-environment",
            "production",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "cannot be used with standalone checks" in result.stderr


@pytest.mark.parametrize(
    ("channel", "release", "environment_name"),
    [
        ("debian", "quoth@0.1.0", "production"),
        ("arch", _CANDIDATE_RELEASE, "development"),
        ("flatpak", _CANDIDATE_RELEASE, "candidate"),
    ],
)
def test_package_contract_accepts_official_channel_layouts(
    tmp_path, channel, release, environment_name
):
    python, _package = _fake_install(tmp_path, channel, release, environment_name)

    result = _run_contract(python, channel, release, environment_name)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT: PASS" in result.stdout


def test_package_contract_reports_every_cpu_module_defect(tmp_path):
    python, package = _fake_install(tmp_path, "arch")
    (package / "lib/libggml-cpu-x64.so").unlink()
    os.link(package / "lib/libwhisper.so", package / "lib/libggml-cpu.so")
    (package / "lib/libggml-cpu-old.so").symlink_to("missing")

    result = _run_contract(python, "arch")

    assert result.returncode == 1
    assert (
        "FAIL native bundle inventory: missing libggml-cpu-x64.so; "
        "obsolete libggml-cpu.so is present; "
        "unexpected CPU module libggml-cpu-old.so"
    ) in result.stdout


def test_package_contract_rejects_thorvg_notice_without_compiled_in_terms(tmp_path):
    python, package = _fake_install(tmp_path, "debian")
    _write(package / "lib/LICENSE.thorvg", "MIT\n")

    result = _run_contract(python, "debian")

    assert result.returncode == 1
    assert (
        "FAIL bundled notices carry their compiled-in terms: "
        "LICENSE.thorvg: THL A29 Limited, "
        "LICENSE.thorvg: based in part on the work of the FreeType Team, "
        "LICENSE.thorvg: mozilla.org/MPL/2.0"
    ) in result.stdout


def test_package_contract_rejects_notices_without_compiled_in_terms(tmp_path):
    python, package = _fake_install(tmp_path, "debian")
    _write(package / "lib/LICENSE.whisper-cpp", "MIT\n")
    _write(package / "lib/LICENSE.wl-clipboard", "GPL-3.0-or-later\n")
    os.link(package / "lib/libwhisper.so", package / "lib/libgtk4-layer-shell.so")
    _write(package / "lib/LICENSE.gtk4-layer-shell", "MIT\n")

    result = _run_contract(python, "debian")

    assert result.returncode == 1
    assert (
        "FAIL bundled notices carry their compiled-in terms: "
        "LICENSE.whisper-cpp: Jeffrey Quesnelle and Bowen Peng, "
        "LICENSE.whisper-cpp: The Khronos Group Inc., "
        "LICENSE.gtk4-layer-shell: Drew DeVault, "
        "LICENSE.wl-clipboard: Simon Ser"
    ) in result.stdout


def test_package_contract_rejects_gpu_only_native_bundle(tmp_path):
    python, package = _fake_install(tmp_path, "arch")
    with (package / "whisper.py").open("a") as stream:
        stream.write(
            "def _ensure_cpu_backend(lib, lib_path):\n"
            "    raise RuntimeError('Whisper CPU backend did not register')\n"
        )

    result = _run_contract(python, "arch")

    assert result.returncode == 1
    assert (
        "FAIL CPU backend registration: Whisper CPU backend did not register"
        in result.stdout
    )


def test_package_contract_uses_packaged_openvino_without_readelf(tmp_path):
    python, package = _fake_install(tmp_path, "flatpak")
    core = package / "lib/openvino/libopenvino.so.2600"
    core.parent.mkdir()
    shutil.copy2(package / "lib/libwhisper.so", core)
    with (package / "__init__.py").open("a") as stream:
        stream.write(f"OPENVINO_LIB = {os.fspath(core)!r}\n")
    _command(
        python.parent / "ldd",
        "import os\n"
        f"expected = {os.fspath(core)!r}\n"
        "if os.environ.get('LD_PRELOAD') != expected:\n"
        "    print('libopenvino.so.2600 => not found')\n",
    )

    result = subprocess.run(
        [
            python,
            "-I",
            _CHECK_INSTALL,
            "--package-channel",
            "flatpak",
            "--expected-release",
            _CANDIDATE_RELEASE,
            "--expected-environment",
            "candidate",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": os.fspath(python.parent)},
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "SKIP OpenVINO check (readelf unavailable)" in result.stdout
    assert "OK   packaged shared-object closure" in result.stdout


def test_debian_requires_the_bundled_ydotool(tmp_path):
    python, package = _fake_install(tmp_path, "debian")
    (package / "lib" / "bin" / "ydotool").unlink()

    result = _run_contract(python, "debian")

    assert result.returncode == 1
    assert "FAIL ydotool help probe" in result.stdout


def test_debian_requires_a_working_wtype(tmp_path):
    python, _package = _fake_install(tmp_path, "debian")
    _command(python.parent / "wtype", "")

    result = _run_contract(python, "debian")

    assert result.returncode == 1
    assert "FAIL wtype usage probe" in result.stdout


def test_flatpak_requires_every_bundled_helper(tmp_path):
    python, _package = _fake_install(tmp_path, "flatpak")
    (python.parents[1] / "host" / "ydotool").unlink()

    result = _run_contract(python, "flatpak")

    assert result.returncode == 1
    assert "FAIL ydotool help probe" in result.stdout


def test_package_contract_rejects_extra_private_debian_content(tmp_path):
    python, package = _fake_install(tmp_path, "debian")
    _write(package / "_vendor" / "unexpected.py")

    result = _run_contract(python, "debian")

    assert result.returncode == 1
    assert "FAIL private Debian vendor is narrow" in result.stdout


@pytest.mark.parametrize(("bundled", "loads"), [(True, False), (False, True)])
def test_debian_contract_requires_the_bundled_npu_driver_to_load(
    tmp_path, bundled, loads
):
    python, package = _fake_install(tmp_path, "debian")
    if not bundled:
        _write(package / "__init__.py")
    _write(package / "hardware.py", f"def _npu_user_driver_present(): return {loads}\n")

    result = _run_contract(python, "debian")

    assert result.returncode == 1
    assert "FAIL bundled NPU driver loads" in result.stdout


def test_package_contract_rejects_wrong_install_channel(tmp_path):
    python, package = _fake_install(tmp_path, "arch")
    (package / "_channel.py").write_text("CHANNEL = 'deb'\n")

    result = _run_contract(python, "arch")

    assert result.returncode == 1
    assert "FAIL install channel" in result.stdout


def test_package_contract_rejects_extra_identity_keys(tmp_path):
    python, package = _fake_install(tmp_path, "flatpak")
    (package / "lib" / "build-identity.json").write_text(
        json.dumps(
            {
                "release": _CANDIDATE_RELEASE,
                "environment": "candidate",
                "source_commit": "0" * 40,
            }
        )
    )

    result = _run_contract(python, "flatpak")

    assert result.returncode == 1
    assert "FAIL build identity" in result.stdout


@pytest.mark.parametrize("module", ["gi", "_gi"])
def test_flatpak_rejects_runtime_pygobject(tmp_path, module):
    python, package = _fake_install(tmp_path, "flatpak")
    gi_init = package.parent / "gi" / "__init__.py"
    with gi_init.open("a") as stream:
        target = "__file__" if module == "gi" else "_gi.__file__"
        stream.write(f"{target} = '/usr/lib/python3/site-packages/gi/runtime.so'\n")

    result = _run_contract(python, "flatpak")

    assert result.returncode == 1
    assert "FAIL PyGObject uses bundled ownership fix" in result.stdout


@pytest.mark.parametrize(
    ("stdout", "stderr", "returncode", "expected"),
    [
        ("libmissing.so => not found\n", "", 0, "missing dependency"),
        ("", "ldd: malformed ELF", 1, "ldd failed"),
        (
            "",
            "libfoo.so: version `FOO_2.0' not found",
            1,
            "version `FOO_2.0' not found",
        ),
    ],
)
def test_ldd_failures_are_reported(
    monkeypatch, tmp_path, stdout, stderr, returncode, expected
):
    monkeypatch.setattr(
        checker.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], returncode, stdout=stdout, stderr=stderr
        ),
    )

    errors = checker._ldd_errors(tmp_path / "libquoth.so")

    assert any(expected in error for error in errors)


def test_ldd_timeout_is_reported(monkeypatch, tmp_path):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(checker.subprocess, "run", timeout)

    assert "timed out" in checker._ldd_errors(tmp_path / "libquoth.so")[0]


def test_helper_probe_has_no_input_and_times_out(monkeypatch):
    captured = {}

    def timeout(argv, **kwargs):
        captured.update(kwargs)
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(checker.subprocess, "run", timeout)

    ok, detail = checker._run_probe(["wl-copy", "--version"])

    assert not ok
    assert "timed out" in detail
    assert captured["stdin"] is subprocess.DEVNULL


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "detail"),
    [
        (
            0,
            "Usage: ydotool <cmd> <args>\nAvailable commands:\n"
            "  click\n  debug\n  key\n  mousemove\n  type\n",
            "",
            "modern interface",
        ),
        (
            1,
            "",
            "Usage: ydotool <cmd> <args>\nAvailable commands:\n"
            "  click\n  key\n  mousemove\n  recorder\n  type\n",
            "legacy interface",
        ),
    ],
)
def test_ydotool_probe_accepts_supported_interfaces(
    monkeypatch, returncode, stdout, stderr, detail
):
    monkeypatch.setattr(
        checker.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], returncode, stdout=stdout, stderr=stderr
        ),
    )

    assert checker._ydotool_help_probe(["ydotool", "--help"]) == (True, detail)


def test_ydotool_probe_rejects_unrecognized_help(monkeypatch):
    monkeypatch.setattr(
        checker.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="unknown tool\n", stderr=""
        ),
    )

    ok, detail = checker._ydotool_help_probe(["ydotool", "--help"])

    assert not ok
    assert detail == "unsupported help output (exit 0)"


def test_dynamic_closure_preloads_only_discovered_openvino_for_engine(
    monkeypatch, tmp_path
):
    package = tmp_path / "quoth"
    engine = package / "lib" / "libwhisper.so"
    sibling = package / "lib" / "libthorvg-1.so"
    core = package / "lib" / "openvino" / "libopenvino.so.2600"
    for path in (engine, sibling, core):
        _elf(path)
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs.get("env")))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/poison")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/poison.so")
    monkeypatch.setattr(checker.subprocess, "run", run)

    assert not checker._dynamic_closure_errors(
        package, engine=engine, openvino_core=core
    )
    by_path = {Path(argv[-1]): env for argv, env in calls}
    assert by_path[engine]["LD_PRELOAD"] == str(core)
    assert "LD_LIBRARY_PATH" not in by_path[engine]
    assert by_path[sibling]["LD_PRELOAD"] == str(core)
    assert by_path[core]["LD_PRELOAD"] == str(core)


def test_dynamic_closure_preloads_the_cores_private_libraries(monkeypatch, tmp_path):
    # The bundled NPU compiler takes the TBB that the core loads from its own dir.
    package = tmp_path / "quoth"
    engine = package / "lib" / "libwhisper.so"
    core = package / "lib" / "openvino" / "libopenvino.so.2600"
    tbb = core.with_name("libtbb.so.12")
    compiler = package / "lib" / "npu" / "libopenvino_intel_npu_compiler.so"
    for path in (engine, core, tbb, compiler):
        _elf(path)
    preloads = {}

    def run(argv, **kwargs):
        preloads[Path(argv[-1])] = kwargs["env"]["LD_PRELOAD"]
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(checker.subprocess, "run", run)

    assert not checker._dynamic_closure_errors(
        package, engine=engine, openvino_core=core
    )
    assert preloads[compiler] == f"{core} {tbb}"


def test_dynamic_closure_checks_the_flatpak_npu_objects(monkeypatch, tmp_path):
    # The Flatpak's compiler sits outside the package with no RPATH, and takes the TBB
    # the preloaded core loaded.
    package = tmp_path / "app/lib/python3/site-packages/quoth"
    engine = package / "lib" / "libwhisper.so"
    core = package / "lib" / "openvino" / "libopenvino.so.2600"
    tbb = core.with_name("libtbb.so.12")
    app_lib = tmp_path / "app/lib"
    compiler = app_lib / "libopenvino_intel_npu_compiler.so"
    driver = app_lib / "libze_intel_npu.so.1.38.0"
    loader = app_lib / "libze_loader.so.1.32.0"
    for path in (engine, core, tbb, compiler, driver, loader, app_lib / "libother.so"):
        _elf(path)
    (app_lib / "libze_intel_npu.so.1").symlink_to(driver.name)
    preloads = {}

    def run(argv, **kwargs):
        preloads[Path(argv[-1])] = kwargs["env"]["LD_PRELOAD"]
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(checker.subprocess, "run", run)
    objects = checker._flatpak_npu_objects(app_lib)

    assert objects == [compiler, driver, loader]
    assert not checker._dynamic_closure_errors(
        package, engine=engine, openvino_core=core, driver_objects=tuple(objects)
    )
    for path in objects:
        assert preloads[path] == f"{core} {tbb}"


def test_package_contract_requires_the_flatpak_npu_objects(tmp_path):
    python, _package = _fake_install(tmp_path, "flatpak")
    for path in (tmp_path / "environment/lib").glob("lib*.so*"):
        path.unlink()

    result = _run_contract(python, "flatpak")

    assert result.returncode == 1
    assert "FAIL Flatpak NPU driver objects found: 0 object(s)" in result.stdout


def test_ldd_without_controlled_preload_removes_ambient_loader_paths(
    monkeypatch, tmp_path
):
    captured = {}

    def run(argv, **kwargs):
        captured.update(kwargs["env"])
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/poison")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/poison.so")
    monkeypatch.setattr(checker.subprocess, "run", run)

    assert not checker._ldd_errors(tmp_path / "libquoth.so")
    assert "LD_LIBRARY_PATH" not in captured
    assert "LD_PRELOAD" not in captured


def test_dynamic_closure_accepts_packaged_openvino_soname_symlink(
    monkeypatch, tmp_path
):
    package = tmp_path / "quoth"
    engine = package / "lib" / "libwhisper.so"
    target = package / "lib" / "openvino" / "libopenvino.so.2600"
    core = target.with_name("libopenvino.so")
    _elf(engine)
    _elf(target)
    core.symlink_to(target.name)
    preloads = []

    def run(argv, **kwargs):
        preloads.append(kwargs["env"].get("LD_PRELOAD"))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(checker.subprocess, "run", run)

    assert not checker._dynamic_closure_errors(
        package, engine=engine, openvino_core=core
    )
    assert preloads and all(preload == str(core) for preload in preloads)


@pytest.mark.parametrize("kind", ["missing", "outside", "directory"])
def test_dynamic_closure_rejects_untrusted_openvino_core(monkeypatch, tmp_path, kind):
    package = tmp_path / "quoth"
    engine = package / "lib" / "libwhisper.so"
    _elf(engine)
    if kind == "missing":
        core = package / "lib" / "openvino" / "libopenvino.so"
    elif kind == "outside":
        outside = tmp_path / "system" / "libopenvino.so"
        _elf(outside)
        core = package / "lib" / "openvino" / "libopenvino.so"
        core.parent.mkdir(parents=True)
        core.symlink_to(outside)
    else:
        core = package / "lib" / "openvino"
        core.mkdir()
    calls = []

    def run(argv, **kwargs):
        calls.append(kwargs["env"])
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setenv("LD_PRELOAD", "/tmp/poison.so")
    monkeypatch.setattr(checker.subprocess, "run", run)

    errors = checker._dynamic_closure_errors(package, engine=engine, openvino_core=core)

    assert errors
    assert all("LD_PRELOAD" not in env for env in calls)


def test_dynamic_closure_never_resolves_host_objects(monkeypatch, tmp_path):
    package = tmp_path / "quoth"
    packaged = package / "lib" / "libwhisper.so"
    _elf(packaged)
    outside = tmp_path / "host.so"
    _elf(outside)
    (package / "lib" / "libhost.so").symlink_to(outside)
    calls = []

    def run(argv, **kwargs):
        calls.append(Path(argv[-1]))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(checker.subprocess, "run", run)

    errors = checker._dynamic_closure_errors(
        package,
        engine=Path("/usr/lib/libwhisper.so"),
        openvino_core=None,
    )

    assert calls == [packaged]
    assert "resolves outside" in errors[0]


def _media_fakes(mode):
    class Media:
        def __init__(self):
            self.callbacks = []
            self.error = None
            self.prepared = False

        def connect(self, _signal, callback):
            self.callbacks.append(callback)

        def set_filename(self, _path):
            assert len(self.callbacks) == 2
            if mode == "immediate-error":
                self.error = types.SimpleNamespace(message="decoder missing")
                self.callbacks[0](self, None)

        def get_error(self):
            return self.error

        def is_prepared(self):
            return self.prepared

        def has_video(self):
            return True

    media = Media()

    class Loop:
        running = False

        def is_running(self):
            return self.running

        def run(self):
            self.running = True
            if mode == "delayed":
                media.prepared = True
                media.callbacks[1](media, None)
            else:
                GLib.timeout()
            self.running = False

        def quit(self):
            self.running = False

    class GLib:
        SOURCE_REMOVE = False
        timeout = None
        MainLoop = Loop

        @classmethod
        def timeout_add(cls, _milliseconds, callback):
            cls.timeout = callback
            return 1

        @staticmethod
        def source_remove(_source):
            pass

    Gtk = types.SimpleNamespace(MediaFile=types.SimpleNamespace(new=lambda: media))
    return Gtk, GLib


def test_media_check_observes_immediate_errors(tmp_path):
    Gtk, GLib = _media_fakes("immediate-error")

    ok, detail = checker._media_file_check(Gtk, GLib, tmp_path / "clip.webm")

    assert not ok
    assert detail == "decoder missing"


def test_media_check_waits_for_delayed_prepare(tmp_path):
    Gtk, GLib = _media_fakes("delayed")

    ok, detail = checker._media_file_check(Gtk, GLib, tmp_path / "clip.webm")

    assert ok
    assert "rendering not verified" in detail


def test_media_check_times_out(tmp_path):
    Gtk, GLib = _media_fakes("timeout")

    ok, detail = checker._media_file_check(Gtk, GLib, tmp_path / "clip.webm")

    assert not ok
    assert "timed out" in detail


@pytest.fixture
def execution_context(monkeypatch):
    """A fake `quoth.typer._ExecutionContext`, unsandboxed unless a test sets it."""

    class Context:
        sandboxed = False
        host_bin = None

        @classmethod
        def detect(cls):
            return cls()

        def wrap(self, tool, args):
            return [tool, *args]

    monkeypatch.setitem(sys.modules, "quoth", types.ModuleType("quoth"))
    typer = types.ModuleType("quoth.typer")
    typer._ExecutionContext = Context
    typer._PACKAGE_BIN = "/usr/lib/python3/dist-packages/quoth/lib/bin"
    monkeypatch.setitem(sys.modules, "quoth.typer", typer)
    return Context


def test_flatpak_helper_probe_rejects_direct_fallback(execution_context):
    results = list(checker._helper_probe_results("flatpak"))

    assert len(results) == 1
    assert results[0][0] == "Flatpak helper execution stays on host"
    assert results[0][1] is False


def test_flatpak_helper_probe_rejects_malformed_host_wrap(execution_context):
    execution_context.sandboxed = True
    execution_context.host_bin = "/host/app/bin"

    results = list(checker._helper_probe_results("flatpak"))

    assert results[0][1] is True
    assert [label for label, _, _ in results[1:]] == [
        "wl-copy version probe",
        "ydotool help probe",
        "wtype usage probe",
    ]
    assert all(not ok and "host path absent" in detail for _, ok, detail in results[1:])


@pytest.mark.parametrize(
    ("tool", "label"),
    [("wl-copy", "wl-copy version probe"), ("ydotool", "ydotool help probe")],
)
def test_debian_helper_probe_requires_the_bundled_tool(execution_context, tool, label):
    results = {
        name: (ok, detail)
        for name, ok, detail in checker._helper_probe_results("debian")
    }
    ok, detail = results[label]

    assert not ok
    assert f"bundled {tool} absent" in detail


def test_package_contract_rejects_missing_xdp_session_method(tmp_path):
    python, package = _fake_install(tmp_path, "flatpak")
    repository = package.parent / "gi" / "repository" / "__init__.py"
    repository.write_text(
        repository.read_text().replace(
            "configure_shortcuts = configure_shortcuts_finish = close = object()",
            "configure_shortcuts = configure_shortcuts_finish = object()",
        )
    )

    result = _run_contract(python, "flatpak")

    assert result.returncode == 1
    assert "FAIL Xdp portal API imports" in result.stdout
    assert "GlobalShortcutsSession.close" in result.stdout


def test_package_contract_rejects_unresolved_packaged_dependency(tmp_path):
    python, _package = _fake_install(tmp_path, "debian")
    _command(
        python.parent / "ldd",
        "print('libmissing.so => not found')\n",
    )

    result = _run_contract(python, "debian")

    assert result.returncode == 1
    assert "FAIL packaged shared-object closure" in result.stdout
    assert "libmissing.so => not found" in result.stdout


_SITE_PACKAGES = (
    f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
)


@pytest.mark.parametrize(
    ("channel", "relative_path", "label"),
    [
        (
            "debian",
            "share/icons/hicolor/symbolic/apps/dev.quoth.Quoth-heart-symbolic.svg",
            "desktop entry, metainfo and icons installed",
        ),
        ("arch", f"{_SITE_PACKAGES}/quoth/lib/libthorvg-1.so", "thorvg resolves"),
        (
            "debian",
            f"{_SITE_PACKAGES}/quoth/lib/build-identity.json",
            "build identity loads",
        ),
        (
            "flatpak",
            "share/metainfo/dev.quoth.Quoth.metainfo.xml",
            "desktop entry, metainfo and icons installed",
        ),
        (
            "arch",
            "share/licenses/quoth/LICENSE.lucide-icons",
            "full Lucide icon notice installed",
        ),
        (
            "flatpak",
            "share/licenses/libportal/COPYING",
            "Flatpak module notices installed",
        ),
        *(
            ("flatpak", f"share/licenses/{notice}", "Flatpak module notices installed")
            for notice in _FLATPAK_NPU_NOTICES
        ),
        ("flatpak", "share/licenses/quoth/LICENSE.loguru", "Loguru notice"),
    ],
)
def test_package_contract_rejects_missing_installed_resource(
    tmp_path, channel, relative_path, label
):
    python, _package = _fake_install(tmp_path, channel)
    (python.parents[1] / relative_path).unlink()

    result = _run_contract(python, channel)

    assert result.returncode == 1
    assert f"FAIL {label}" in result.stdout


@pytest.mark.parametrize(
    ("channel", "relative_path", "content", "label"),
    [
        (
            "arch",
            "sentry_sdk/__init__.py",
            'VERSION = "2.71.0"\n',
            "Sentry SDK version",
        ),
        # The .deb and the Flatpak ship exactly the pinned SDK.
        (
            "debian",
            "quoth/_vendor/sentry_sdk/__init__.py",
            'VERSION = "2.73.0"\n',
            "Sentry SDK version",
        ),
        (
            "arch",
            "quoth/_vendor/sentry_sdk/__init__.py",
            'VERSION = "2.72.0"\n',
            "no private Sentry SDK copy",
        ),
        (
            "flatpak",
            "quoth/_channel.py",
            "CHANNEL = 'deb'\n",
            "install channel stamp absent",
        ),
        (
            "arch",
            "quoth/audio.py",
            "raise ImportError('Gst')\n",
            "audio pipeline builds",
        ),
    ],
    ids=[
        "sentry-below-floor",
        "sentry-off-pin",
        "private-sentry-on-arch",
        "flatpak-channel-stamp",
        "broken-audio",
    ],
)
def test_package_contract_rejects_a_defective_package_file(
    tmp_path, channel, relative_path, content, label
):
    python, package = _fake_install(tmp_path, channel)
    _write(package.parent / relative_path, content)

    result = _run_contract(python, channel)

    assert result.returncode == 1
    assert f"FAIL {label}" in result.stdout


def test_package_contract_rejects_an_uncompiled_settings_schema(tmp_path):
    python, package = _fake_install(tmp_path, "arch")
    repository = package.parent / "gi" / "repository" / "__init__.py"
    repository.write_text(
        repository.read_text().replace('schema_id == "dev.quoth.Quoth"', "False")
    )

    result = _run_contract(python, "arch")

    assert result.returncode == 1
    assert "FAIL settings schema installed and compiled" in result.stdout


def test_package_contract_requires_the_openvino_that_libwhisper_links(tmp_path):
    python, _package = _fake_install(tmp_path, "arch")
    _readelf(python.parent, " 0x1 (NEEDED) Shared library: [libopenvino.so.2600]\n")

    result = _run_contract(python, "arch")

    assert result.returncode == 1
    assert "FAIL bundled OpenVINO present" in result.stdout
    assert "FAIL OpenVINO core was preloaded" in result.stdout


def test_arch_requires_a_working_chord_sender(tmp_path):
    # Arch needs only one sender, so both are broken.
    python, _package = _fake_install(tmp_path, "arch")
    _command(python.parent / "ydotool", "print('unknown tool')\n")
    _command(python.parent / "wtype", "")

    result = _run_contract(python, "arch")

    assert result.returncode == 1
    assert "FAIL supported chord-sender probe" in result.stdout


_UPSTREAM_NOTICES = [
    ("catppuccin", "Catppuccin palette"),
    ("dracula", "Dracula palette"),
    ("nord", "Nord palette"),
    ("silero-vad", "Silero VAD model"),
    ("loguru", "Loguru"),
]


def test_package_contract_rejects_missing_upstream_notices(tmp_path):
    python, _package = _fake_install(tmp_path, "arch")
    for notice, _asset in _UPSTREAM_NOTICES:
        (python.parents[1] / f"share/licenses/quoth/LICENSE.{notice}").unlink()

    result = _run_contract(python, "arch")

    assert result.returncode == 1
    lines = result.stdout.splitlines()
    for notice, asset in _UPSTREAM_NOTICES:
        # The error names the file, so each label is checked against its own notice.
        (line,) = [line for line in lines if f"FAIL {asset} notice:" in line]
        assert line.endswith(f"/LICENSE.{notice}'")


def test_package_contract_rejects_modified_upstream_notices(tmp_path):
    python, _package = _fake_install(tmp_path, "arch")
    digests = {}
    for notice, _asset in _UPSTREAM_NOTICES:
        installed = python.parents[1] / f"share/licenses/quoth/LICENSE.{notice}"
        installed.write_text(installed.read_text() + "modified\n")
        digests[notice] = hashlib.sha256(installed.read_bytes()).hexdigest()

    result = _run_contract(python, "arch")

    assert result.returncode == 1
    for notice, asset in _UPSTREAM_NOTICES:
        assert f"FAIL {asset} notice: {digests[notice]}" in result.stdout


def test_host_tools_within_the_glibc_ceiling_pass(tmp_path):
    for tool in checker._FLATPAK_HOST_TOOLS:
        _versioned_elf(tmp_path / tool, "GLIBC_2.2.5", "GLIBC_2.35", "V_1.0.0")

    assert checker._host_glibc_errors(tmp_path) == []


def test_host_tool_past_the_glibc_ceiling_is_named(tmp_path):
    for tool in checker._FLATPAK_HOST_TOOLS:
        _versioned_elf(tmp_path / tool, "GLIBC_2.34")
    _versioned_elf(tmp_path / "wtype", "GLIBC_2.4", "GLIBC_2.38", "GLIBC_2.35")

    assert checker._host_glibc_errors(tmp_path) == [
        f"{tmp_path / 'wtype'} needs GLIBC_2.38, past GLIBC_2.35"
    ]


def test_host_tool_needing_an_unnumbered_glibc_version_is_named(tmp_path):
    for tool in checker._FLATPAK_HOST_TOOLS:
        _versioned_elf(tmp_path / tool, "GLIBC_2.34")
    _versioned_elf(tmp_path / "ydotool", "GLIBC_ABI_DT_RELR", "GLIBC_2.34")

    assert checker._host_glibc_errors(tmp_path) == [
        f"{tmp_path / 'ydotool'} needs GLIBC_ABI_DT_RELR, not a version up to GLIBC_2.35"
    ]


def test_host_tool_without_section_headers_is_reported(tmp_path):
    for tool in checker._FLATPAK_HOST_TOOLS:
        _versioned_elf(tmp_path / tool, "GLIBC_2.34")
    data = bytearray((tmp_path / "wtype").read_bytes())
    struct.pack_into("<H", data, 0x3C, 0)
    (tmp_path / "wtype").write_bytes(data)

    assert checker._host_glibc_errors(tmp_path) == [
        f"{tmp_path / 'wtype'}: cannot read its symbol versions: no section headers"
    ]


def test_missing_or_unreadable_host_tool_is_reported(tmp_path):
    for tool in checker._FLATPAK_HOST_TOOLS:
        _versioned_elf(tmp_path / tool, "GLIBC_2.34")
    (tmp_path / "ydotoold").unlink()
    _write(tmp_path / "wl-paste", "#!/bin/sh\n")

    assert checker._host_glibc_errors(tmp_path) == [
        f"{tmp_path / 'wl-paste'}: cannot read its symbol versions: "
        "not a 64-bit little-endian ELF file",
        f"missing {tmp_path / 'ydotoold'}",
    ]


def test_flatpak_contract_rejects_a_host_tool_past_the_glibc_ceiling(tmp_path):
    python, _package = _fake_install(tmp_path, "flatpak")
    _versioned_elf(python.parent / "wl-copy", "GLIBC_2.38")

    result = _run_contract(python, "flatpak")

    assert result.returncode == 1
    assert (
        "FAIL host tools load on the oldest supported glibc: "
        f"{python.parent / 'wl-copy'} needs GLIBC_2.38, past GLIBC_2.35"
    ) in result.stdout
