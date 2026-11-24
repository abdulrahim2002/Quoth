#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Installed Quoth smoke test and standalone native-build policy checks.

The default mode verifies an installed Quoth and may skip ELF inspection when the
user has no `readelf`. `--elf-root` imports no Quoth code and fails closed so
build and packaging gates can share one relocatability policy. `--compile-commands`
checks a CMake compile database for CPU flags beyond the portable baseline.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import os
import re
import shlex
import struct
import subprocess
import sys
from pathlib import Path

_DYNAMIC_ENTRY = re.compile(r"\((NEEDED|RPATH|RUNPATH)\)[^[]*\[(.*)\]\s*$")
_APP_ID = "dev.quoth.Quoth"
_SENTRY_VERSION = "2.72.0"
_URLLIB3_FLOOR = "1.26.11"
_PROBE_TIMEOUT_S = 5
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
_NATIVE_BUILD_FILES = (
    "libwhisper.so",
    "libggml.so",
    "libggml-base.so",
    "libggml-vulkan.so",
)
_CPU_TARGETS = frozenset(
    name.removeprefix("lib").removesuffix(".so") for name in _CPU_MODULES
)
_SCORER_TARGETS = frozenset(f"{target}-feats" for target in _CPU_TARGETS)
_REQUIRED_BASELINE_TARGETS = frozenset({"ggml", "ggml-base", "whisper", "ggml-cpu-x64"})
_BASELINE_M_FLAGS = frozenset(
    {
        "-m64",
        "-march=x86-64",
        "-mindirect-branch=thunk",
        "-mmmx",
        "-msse",
        "-msse2",
        "-mtune=generic",
    }
)
_SYMBOLIC_ICONS = ("gauge", "bug", "heart", "coffee", "github")
_UPSTREAM_NOTICES = (
    (
        "Catppuccin palette",
        "LICENSE.catppuccin",
        "814096d2c34cc216c624738a49356f32b7237733b4f7edb0685f4e50ef5074ba",
    ),
    (
        "Dracula palette",
        "LICENSE.dracula",
        "38b806a7e799cca4e9821e6f8d65cf0d8bfaaddbe454b41f12e5baad52d310f1",
    ),
    (
        "Nord palette",
        "LICENSE.nord",
        "25ac8188d670bd2ad2ce2f4f55ab88573010ee9f7a4502543cb1eea1e2274f8a",
    ),
    (
        "Silero VAD model",
        "LICENSE.silero-vad",
        "2e63e9a38b6e8fc0c7bc37ce174caca1862870856c6daf5697cfb785e925520b",
    ),
    (
        "Loguru",
        "LICENSE.loguru",
        "b35d026cc7aca9d5859a02eb87ddf7a386a24c986838651bd1f283f94e003327",
    ),
)
_FLATPAK_NOTICES = (
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
)
# A bundled library's notice must carry the terms of what it compiles in; a row
# applies where its library is bundled.
_COMPILED_IN_TERMS = (
    (
        "libwhisper.so",
        "LICENSE.whisper-cpp",
        ("Jeffrey Quesnelle and Bowen Peng", "The Khronos Group Inc."),
    ),
    (
        "libthorvg-1.so",
        "LICENSE.thorvg",
        (
            "THL A29 Limited",
            "based in part on the work of the FreeType Team",
            "mozilla.org/MPL/2.0",
        ),
    ),
    ("libgtk4-layer-shell.so", "LICENSE.gtk4-layer-shell", ("Drew DeVault",)),
    ("bin/wl-copy", "LICENSE.wl-clipboard", ("Simon Ser",)),
)
_NPU_DRIVER_NOTICES = (
    "npu/licenses/intel-npu-driver/LICENSE.md",
    "npu/licenses/intel-npu-driver/third-party-programs.txt",
    "npu/licenses/intel-npu-compiler/LICENSE",
    "npu/licenses/intel-npu-compiler/LICENSE.llvm",
    "npu/licenses/level-zero/LICENSE",
)
# The Flatpak runs these on the host through flatpak-spawn, against the host's glibc,
# so they must load on Ubuntu 22.04's 2.35.
_FLATPAK_HOST_TOOLS = ("wl-copy", "wl-paste", "wtype", "ydotool", "ydotoold")
_HOST_GLIBC_CEILING = (2, 35)
_SHT_GNU_VERNEED = 0x6FFFFFFE
_YDOTOOL_HELP_HEADER = ("Usage: ydotool <cmd> <args>", "Available commands:")
_YDOTOOL_LEGACY_COMMANDS = {"click", "key", "mousemove", "recorder", "type"}
_YDOTOOL_MODERN_COMMANDS = {"click", "debug", "key", "mousemove", "type"}


def _read_dynamic(path: str | os.PathLike[str]) -> dict[str, list[str]]:
    result = subprocess.run(
        ["readelf", "-d", os.fspath(path)],
        capture_output=True,
        text=True,
        check=True,
    )

    dynamic = {"NEEDED": [], "RPATH": [], "RUNPATH": []}
    for line in result.stdout.splitlines():
        match = _DYNAMIC_ENTRY.search(line)
        if match:
            dynamic[match.group(1)].append(match.group(2))
    return dynamic


def _elf_version_needs(path: Path) -> set[str]:
    """Return the symbol versions a 64-bit little-endian ELF file needs to load.

    Read from its GNU version-needs section in Python, since the Flatpak's Platform
    runtime, where the installed check runs, has no readelf.
    """
    data = path.read_bytes()
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("not a 64-bit little-endian ELF file")
    (section_offset,) = struct.unpack_from("<Q", data, 0x28)
    entry_size, count = struct.unpack_from("<HH", data, 0x3A)
    if not count:
        raise ValueError("no section headers")
    sections = [
        struct.unpack_from("<IIQQQQII", data, section_offset + index * entry_size)
        for index in range(count)
    ]
    needs = set()
    for _name, kind, _flags, _addr, offset, _size, link, entries in sections:
        if kind != _SHT_GNU_VERNEED:
            continue
        strings = sections[link][4]
        for _ in range(entries):
            _version, aux_count, _file, aux, next_entry = struct.unpack_from(
                "<HHIII", data, offset
            )
            aux_offset = offset + aux
            for _ in range(aux_count):
                _hash, _aux_flags, _other, name, next_aux = struct.unpack_from(
                    "<IHHII", data, aux_offset
                )
                start = strings + name
                needs.add(data[start : data.index(b"\0", start)].decode())
                aux_offset += next_aux
            offset += next_entry
    return needs


def _host_glibc_errors(bin_dir: Path) -> list[str]:
    ceiling = "GLIBC_" + ".".join(map(str, _HOST_GLIBC_CEILING))
    errors = []
    for tool in _FLATPAK_HOST_TOOLS:
        path = bin_dir / tool
        if not path.is_file():
            errors.append(f"missing {path}")
            continue
        try:
            needs = _elf_version_needs(path)
        except (OSError, ValueError, IndexError, struct.error) as error:
            errors.append(f"{path}: cannot read its symbol versions: {error}")
            continue
        # An unnumbered need fails too: GLIBC_ABI_DT_RELR, which linking with
        # `-z pack-relative-relocs` adds, first exists in glibc 2.36, and
        # GLIBC_PRIVATE ties a tool to the glibc build it was linked against.
        glibc = {}
        for name in needs:
            if match := re.fullmatch(r"GLIBC_(\d+(?:\.\d+)+)", name):
                glibc[tuple(map(int, match.group(1).split(".")))] = name
            elif name.startswith("GLIBC_"):
                errors.append(f"{path} needs {name}, not a version up to {ceiling}")
        errors.extend(
            f"{path} needs {glibc[version]}, past {ceiling}"
            for version in sorted(glibc)
            if version > _HOST_GLIBC_CEILING
        )
    return errors


def _readelf_failure(error: subprocess.CalledProcessError) -> str:
    detail = error.stderr.strip() or f"exit {error.returncode}"
    return f"readelf failed for {error.cmd[-1]}: {detail}"


def _shared_objects(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*.so*") if path.is_file())


def _required_file_errors(lib_root: Path, names: tuple[str, ...]) -> list[str]:
    errors = []
    resolved_root = lib_root.resolve()
    for name in names:
        path = lib_root / name
        if not path.is_file():
            errors.append(f"missing {name}")
        elif not path.resolve().is_relative_to(resolved_root):
            errors.append(f"{name} resolves outside {lib_root}")
    return errors


def _cpu_module_errors(lib_root: Path) -> list[str]:
    expected = set(_CPU_MODULES)
    errors = _required_file_errors(lib_root, _CPU_MODULES)
    if not lib_root.is_dir():
        return errors
    actual = {
        path.name
        for path in lib_root.iterdir()
        if path.name.startswith("libggml-cpu") and ".so" in path.name
    }
    stale_cpu = lib_root / "libggml-cpu.so"
    if stale_cpu.exists() or stale_cpu.is_symlink():
        errors.append("obsolete libggml-cpu.so is present")
        actual.discard(stale_cpu.name)
    errors.extend(f"unexpected CPU module {name}" for name in sorted(actual - expected))
    return errors


def _native_bundle_errors(lib_root: Path, package_channel: str) -> list[str]:
    non_cpu = tuple(name for name in _NATIVE_BUNDLE_FILES if name not in _CPU_MODULES)
    errors = _required_file_errors(lib_root, non_cpu) + _cpu_module_errors(lib_root)
    # A build may omit gtk4-layer-shell, wl-clipboard, ydotool, OpenVINO or the NPU
    # driver stack, but one that bundles any carries its notice beside it, and ydotool
    # its daemon; the Flatpak's OpenVINO module installs its own under share/licenses.
    if (lib_root / "libgtk4-layer-shell.so").exists():
        errors += _required_file_errors(lib_root, ("LICENSE.gtk4-layer-shell",))
    if (lib_root / "bin" / "wl-copy").exists():
        errors += _required_file_errors(lib_root, ("LICENSE.wl-clipboard",))
    if (lib_root / "bin" / "ydotool").exists():
        errors += _required_file_errors(lib_root, ("bin/ydotoold", "LICENSE.ydotool"))
    if package_channel != "flatpak" and (lib_root / "openvino").is_dir():
        errors += _required_file_errors(lib_root, ("openvino/licenses/LICENSE",))
    if (lib_root / "npu").is_dir():
        errors += _required_file_errors(lib_root, _NPU_DRIVER_NOTICES)
    return errors


def _native_build_errors(lib_root: Path) -> list[str]:
    errors = _required_file_errors(lib_root, _NATIVE_BUILD_FILES)
    errors.extend(_cpu_module_errors(lib_root))
    return errors


def _compile_arguments(row: object) -> list[str] | None:
    if not isinstance(row, dict):
        return None
    arguments = row.get("arguments")
    if isinstance(arguments, list) and all(isinstance(arg, str) for arg in arguments):
        return arguments
    command = row.get("command")
    if isinstance(command, str):
        try:
            return shlex.split(command)
        except ValueError:
            return None
    return None


def _cmake_target(arguments: list[str]) -> str | None:
    try:
        output = arguments[arguments.index("-o") + 1]
    except (ValueError, IndexError):
        return None
    return next(
        (
            part.removesuffix(".dir")
            for part in Path(output).parts
            if part.endswith(".dir")
        ),
        None,
    )


def _unsafe_cpu_flags(
    arguments: list[str], *, selectors_only: bool = False
) -> list[str]:
    unsafe = []
    for index, argument in enumerate(arguments):
        if argument in {"-march", "-mcpu", "-mtune"}:
            value = arguments[index + 1] if index + 1 < len(arguments) else "<missing>"
            combined = f"{argument}={value}"
            if combined not in _BASELINE_M_FLAGS:
                unsafe.append(combined)
        elif argument.startswith(("-march=", "-mcpu=", "-mtune=")):
            if argument not in _BASELINE_M_FLAGS:
                unsafe.append(argument)
        elif (
            not selectors_only
            and argument.startswith("-m")
            and argument not in _BASELINE_M_FLAGS
            and not argument.startswith("-mno-")
        ):
            unsafe.append(argument)
    return unsafe


def _compile_command_errors(path: Path) -> list[str]:
    try:
        rows = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return [f"cannot read compile commands: {error}"]
    if not isinstance(rows, list):
        return ["compile commands root is not a list"]

    errors = []
    seen = set()
    for index, row in enumerate(rows):
        arguments = _compile_arguments(row)
        if arguments is None:
            errors.append(f"compile command {index} has no parseable arguments")
            continue
        target = _cmake_target(arguments)
        if target is None:
            errors.append(f"compile command {index} has no CMake target output")
            continue
        relevant = target in _REQUIRED_BASELINE_TARGETS | _SCORER_TARGETS | _CPU_TARGETS
        if not relevant:
            continue
        seen.add(target)
        optimized_kernel = target in _CPU_TARGETS and target != "ggml-cpu-x64"
        for flag in _unsafe_cpu_flags(arguments, selectors_only=optimized_kernel):
            errors.append(f"{target} uses unsafe CPU flag {flag}")
        if target in _SCORER_TARGETS and "-fno-lto" not in arguments:
            errors.append(f"{target} scorer is missing -fno-lto")

    required = _REQUIRED_BASELINE_TARGETS | _SCORER_TARGETS
    errors.extend(
        f"missing compile target {target}" for target in sorted(required - seen)
    )
    return errors


def _check_compile_commands(path: Path) -> int:
    print(f"check_install: compile commands {path}")
    failures = _compile_command_errors(path)
    for failure in failures:
        print(f"  FAIL {failure}")
    if not failures:
        print("  OK   baseline, x64 and CPU scorer objects are ISA-safe")
    print("RESULT:", "FAIL" if failures else "PASS")
    return 1 if failures else 0


def _path_errors(path: Path, dynamic: dict[str, list[str]]) -> list[str]:
    errors = []
    for tag in ("RPATH", "RUNPATH"):
        for value in dynamic[tag]:
            for entry in value.split(":"):
                if not (
                    entry in {"$ORIGIN", "${ORIGIN}"}
                    or entry.startswith("$ORIGIN/")
                    or entry.startswith("${ORIGIN}/")
                ):
                    errors.append(f"{path}: unsafe {tag} entry {entry!r}")
    return errors


def _check_elf_root(root: Path) -> int:
    print(f"check_install: ELF root {root}")
    if not root.is_dir():
        print(f"  FAIL ELF root is not a directory: {root}")
        print("RESULT: FAIL")
        return 1

    shared_objects = _shared_objects(root)
    if not shared_objects:
        print("  FAIL ELF root contains no shared objects")
        print("RESULT: FAIL")
        return 1

    failures = []
    native_build = any(
        path.name.startswith(("libwhisper.so", "libggml")) for path in root.iterdir()
    )
    if native_build:
        failures.extend(_native_build_errors(root))
    try:
        for path in shared_objects:
            failures.extend(_path_errors(path, _read_dynamic(path)))
    except FileNotFoundError:
        failures.append("readelf not found")
    except subprocess.CalledProcessError as error:
        failures.append(_readelf_failure(error))

    for failure in failures:
        print(f"  FAIL {failure}")
    if not failures:
        print(f"  OK   {len(shared_objects)} shared object(s) use safe dynamic paths")
    print("RESULT:", "FAIL" if failures else "PASS")
    return 1 if failures else 0


def _libwhisper_needs_openvino(lw: str | None) -> bool | None:
    """Whether libwhisper links OpenVINO for NPU encoder support.

    None when `lw` names no library.
    """
    if not lw:
        return None
    return any("libopenvino" in needed for needed in _read_dynamic(lw)["NEEDED"])


def _version_tuple(version: str) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:[.+-].*)?$", version)
    return tuple(map(int, match.groups())) if match else None


def _run_probe(
    argv: list[str],
    *,
    expected_returncode: int = 0,
) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except OSError as error:
        return False, str(error)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {_PROBE_TIMEOUT_S}s"
    detail = (result.stderr.strip() or result.stdout.strip()).splitlines()
    summary = detail[-1] if detail else f"exit {result.returncode}"
    return result.returncode == expected_returncode, summary


def _ydotool_help_probe(argv: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except OSError as error:
        return False, str(error)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {_PROBE_TIMEOUT_S}s"

    def _commands(output: str) -> set[str]:
        lines = output.strip().splitlines()
        if tuple(lines[:2]) != _YDOTOOL_HELP_HEADER:
            return set()
        return {line[2:] for line in lines[2:] if line.startswith("  ")}

    legacy = (
        result.returncode == 1
        and not result.stdout
        and _YDOTOOL_LEGACY_COMMANDS <= _commands(result.stderr)
    )
    modern = (
        result.returncode == 0
        and not result.stderr
        and _YDOTOOL_MODERN_COMMANDS <= _commands(result.stdout)
    )
    if legacy:
        return True, "legacy interface"
    if modern:
        return True, "modern interface"
    return False, f"unsupported help output (exit {result.returncode})"


def _ldd_errors(path: Path, *, preload: tuple[Path, ...] = ()) -> list[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"LD_LIBRARY_PATH", "LD_PRELOAD"}
    }
    if preload:
        env["LD_PRELOAD"] = " ".join(map(os.fspath, preload))
    try:
        result = subprocess.run(
            ["ldd", os.fspath(path)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=env,
            timeout=_PROBE_TIMEOUT_S,
        )
    except OSError as error:
        return [f"{path}: ldd failed to start: {error}"]
    except subprocess.TimeoutExpired:
        return [f"{path}: ldd timed out after {_PROBE_TIMEOUT_S}s"]

    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    errors = [
        f"{path}: missing dependency: {line.strip()}"
        for line in output.splitlines()
        if "not found" in line
    ]
    if result.returncode:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or f"exit {result.returncode}"
        )
        errors.append(f"{path}: ldd failed: {detail}")
    return errors


def _flatpak_npu_objects(lib: Path) -> list[Path]:
    paths = [*lib.glob("libopenvino_intel_npu_compiler*.so"), *lib.glob("libze_*.so*")]
    return sorted({path.resolve() for path in paths if path.is_file()})


def _dynamic_closure_errors(
    package_root: Path,
    *,
    engine: Path | None,
    openvino_core: Path | None,
    driver_objects: tuple[Path, ...] = (),
) -> list[str]:
    errors = []
    lib_root = (package_root / "lib").resolve()
    preload: tuple[Path, ...] = ()
    if openvino_core is not None:
        try:
            resolved_core = openvino_core.resolve(strict=True)
        except OSError as error:
            errors.append(f"{openvino_core}: cannot resolve packaged OpenVINO: {error}")
        else:
            if not resolved_core.is_file():
                errors.append(
                    f"{openvino_core}: packaged OpenVINO target is not a regular file"
                )
            elif not resolved_core.is_relative_to(lib_root):
                errors.append(
                    f"{openvino_core}: packaged OpenVINO resolves outside {lib_root}"
                )
            else:
                # The core loads the TBB beside it, which the bundled NPU compiler also
                # takes, so each child loads both first, as the installed process does.
                tbb = sorted(resolved_core.parent.glob("libtbb.so.*"))
                preload = (openvino_core, *tbb)
    for path in _shared_objects(package_root / "lib"):
        try:
            trusted = path.resolve(strict=True).is_relative_to(lib_root)
        except OSError as error:
            errors.append(f"{path}: cannot resolve packaged object: {error}")
            continue
        if not trusted:
            errors.append(f"{path}: packaged object resolves outside {lib_root}")
            continue
        # The installed process preloads this core before any native library. Mirror
        # that exact loader state in each child without widening its search path.
        errors.extend(_ldd_errors(path, preload=preload if engine is not None else ()))
    # The compiler has no RPATH and takes the TBB the core loaded, as in the process.
    for path in driver_objects:
        errors.extend(_ldd_errors(path, preload=preload if engine is not None else ()))
    return errors


def _media_file_check(Gtk, GLib, path: Path) -> tuple[bool, str]:
    loop = GLib.MainLoop()
    state = {"done": False, "timed_out": False}
    media = Gtk.MediaFile.new()

    def _updated(*_args) -> None:
        if media.get_error() is not None or media.is_prepared():
            state["done"] = True
            if loop.is_running():
                loop.quit()

    media.connect("notify::error", _updated)
    media.connect("notify::prepared", _updated)
    media.set_filename(os.fspath(path))
    _updated()

    timeout_id = None
    if not state["done"]:

        def _timeout() -> bool:
            state["timed_out"] = True
            loop.quit()
            return GLib.SOURCE_REMOVE

        timeout_id = GLib.timeout_add(_PROBE_TIMEOUT_S * 1000, _timeout)
        loop.run()
    if timeout_id is not None and not state["timed_out"]:
        GLib.source_remove(timeout_id)

    error = media.get_error()
    if error is not None:
        return False, error.message
    if state["timed_out"]:
        return False, f"timed out after {_PROBE_TIMEOUT_S}s"
    if not media.is_prepared():
        return False, "media did not prepare"
    if not media.has_video():
        return False, "media prepared without video"
    return True, "prepared with video and no decoder error (rendering not verified)"


def _helper_probe_results(package_channel: str):
    import quoth.typer as typer

    context = typer._ExecutionContext.detect()
    if package_channel == "flatpak":
        valid_context = bool(
            context.sandboxed
            and context.host_bin
            and Path(context.host_bin).is_absolute()
        )
        yield "Flatpak helper execution stays on host", valid_context, repr(context)
        if not valid_context:
            return

    def _probe(label, tool, args, returncode):
        argv = context.wrap(tool, args)
        if package_channel == "flatpak":
            host_tool = os.fspath(Path(context.host_bin) / tool)
            if host_tool not in argv:
                return label, False, f"host path absent from {argv!r}"
        # The .deb bundles wl-clipboard (its distros ship one without --sensitive) and
        # ydotool (Debian 13 ships it only in backports).
        if package_channel == "debian" and tool in ("wl-copy", "ydotool"):
            if argv[0] != os.path.join(typer._PACKAGE_BIN, tool):
                return label, False, f"bundled {tool} absent from {argv!r}"
        if returncode is None:
            ok, detail = _ydotool_help_probe(argv)
        else:
            ok, detail = _run_probe(argv, expected_returncode=returncode)
        return label, ok, detail

    yield _probe("wl-copy version probe", "wl-copy", ["--version"], 0)
    senders = (
        _probe("ydotool help probe", "ydotool", ["--help"], None),
        _probe("wtype usage probe", "wtype", [], 1),
    )
    # The Flatpak and the .deb install both senders; elsewhere either one will do.
    if package_channel in ("flatpak", "debian"):
        yield from senders
    else:
        detail = "; ".join(f"{label}: {result}" for label, _, result in senders)
        yield "supported chord-sender probe", any(ok for _, ok, _ in senders), detail


def _installed_smoke(
    package_channel: str | None = None,
    expected_release: str | None = None,
    expected_environment: str | None = None,
) -> int:
    import quoth  # runs __init__: preloads gtk4-layer-shell + the OpenVINO core when present
    import quoth.whisper as w
    from quoth import _native as native

    failures: list[str] = []

    def _check(label: str, cond: bool, detail: str = "") -> None:
        print(
            f"  {'OK  ' if cond else 'FAIL'} {label}"
            + (f": {detail}" if detail else "")
        )
        if not cond:
            failures.append(label)

    pkgdir = os.path.dirname(quoth.__file__)
    print(f"check_install: quoth at {pkgdir}")

    if package_channel:
        package_root = Path(pkgdir).resolve()
        vendor_root = package_root / "_vendor"
        identity_path = package_root / "lib" / "build-identity.json"
        channel_path = package_root / "_channel.py"
        expected_channel = {"debian": "deb", "arch": "aur", "flatpak": None}[
            package_channel
        ]
        if expected_channel is None:
            _check("install channel stamp absent", not channel_path.exists())
        else:
            try:
                from quoth._channel import CHANNEL
            except ImportError as error:
                _check("install channel", False, str(error))
            else:
                _check("install channel", CHANNEL == expected_channel, repr(CHANNEL))
        try:
            identity = json.loads(identity_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            _check("build identity loads", False, str(error))
        else:
            expected_identity = {
                "release": expected_release,
                "environment": expected_environment,
            }
            _check(
                "build identity",
                identity == expected_identity,
                repr(identity),
            )
        from gi.repository import Gio

        schemas = Gio.SettingsSchemaSource.get_default()
        _check(
            "settings schema installed and compiled",
            schemas is not None and schemas.lookup(_APP_ID, True) is not None,
        )

        package_root = Path(pkgdir).resolve()
        prefix = (
            Path("/app")
            if package_channel == "flatpak" and package_root.is_relative_to("/app")
            else Path(sys.prefix)
        )
        share = prefix / "share"
        icons = share / "icons" / "hicolor"
        missing_data = [
            path.name
            for path in (
                share / "applications" / f"{_APP_ID}.desktop",
                share / "metainfo" / f"{_APP_ID}.metainfo.xml",
                icons / "scalable" / "apps" / f"{_APP_ID}.svg",
                *(
                    icons / "symbolic" / "apps" / f"{_APP_ID}-{icon}-symbolic.svg"
                    for icon in _SYMBOLIC_ICONS
                ),
            )
            if not path.is_file()
        ]
        _check(
            "desktop entry, metainfo and icons installed",
            not missing_data,
            ", ".join(missing_data),
        )
        license_root = prefix / "share" / "licenses"
        lucide_notice = license_root / "quoth" / "LICENSE.lucide-icons"
        _check(
            "full Lucide icon notice installed",
            lucide_notice.is_file(),
            os.fspath(lucide_notice),
        )
        for asset, notice, expected_digest in _UPSTREAM_NOTICES:
            notice_path = license_root / "quoth" / notice
            try:
                actual_digest = hashlib.sha256(notice_path.read_bytes()).hexdigest()
            except OSError as error:
                _check(f"{asset} notice", False, str(error))
            else:
                _check(
                    f"{asset} notice",
                    actual_digest == expected_digest,
                    actual_digest,
                )
        if package_channel == "flatpak":
            missing_notices = [
                notice
                for notice in _FLATPAK_NOTICES
                if not (license_root / notice).is_file()
            ]
            _check(
                "Flatpak module notices installed",
                not missing_notices,
                ", ".join(missing_notices),
            )
            glibc_errors = _host_glibc_errors(prefix / "bin")
            _check(
                "host tools load on the oldest supported glibc",
                not glibc_errors,
                "; ".join(glibc_errors),
            )

        from quoth import crash_reports

        crash_reports._add_vendor_path()
        try:
            import certifi
            import sentry_sdk
            import urllib3
        except ImportError as error:
            _check("crash-report dependencies import", False, str(error))
        else:
            sdk_path = Path(sentry_sdk.__file__).resolve()
            certifi_path = Path(certifi.__file__).resolve()
            urllib3_path = Path(urllib3.__file__).resolve()
            sdk_version = sentry_sdk.VERSION
            sdk_tuple = _version_tuple(sdk_version)
            floor_tuple = _version_tuple(_SENTRY_VERSION)
            urllib3_tuple = _version_tuple(urllib3.__version__)
            urllib3_floor_tuple = _version_tuple(_URLLIB3_FLOOR)
            _check(
                "Sentry SDK version",
                sdk_version == _SENTRY_VERSION
                if package_channel in {"debian", "flatpak"}
                else bool(sdk_tuple and floor_tuple and sdk_tuple >= floor_tuple),
                sdk_version,
            )
            _check(
                "urllib3 version floor",
                bool(
                    urllib3_tuple
                    and urllib3_floor_tuple
                    and urllib3_tuple >= urllib3_floor_tuple
                ),
                urllib3.__version__,
            )
            _check(
                "certifi is external to private vendor",
                not certifi_path.is_relative_to(vendor_root),
                os.fspath(certifi_path),
            )
            _check(
                "urllib3 is external to private vendor",
                not urllib3_path.is_relative_to(vendor_root),
                os.fspath(urllib3_path),
            )
            if package_channel == "debian":
                _check(
                    "Sentry SDK uses private Debian vendor",
                    sdk_path.is_relative_to(vendor_root / "sentry_sdk"),
                    os.fspath(sdk_path),
                )
                vendor_entries = (
                    sorted(path.name for path in vendor_root.iterdir())
                    if vendor_root.is_dir()
                    else []
                )
                _check(
                    "private Debian vendor is narrow",
                    vendor_entries == ["LICENSE.sentry-sdk", "sentry_sdk"],
                    repr(vendor_entries),
                )
            else:
                _check(
                    "Sentry SDK uses channel package",
                    not sdk_path.is_relative_to(vendor_root),
                    os.fspath(sdk_path),
                )
                _check(
                    "no private Sentry SDK copy",
                    not (vendor_root / "sentry_sdk").exists(),
                )

    lw, helper = w.find_libwhisper(), w.find_libwhisper_helper()
    _check("native libs resolve", bool(lw and helper), lw or "libwhisper not found")

    # Source builds can omit OpenVINO; require it whenever libwhisper links it.
    ov = glob.glob(os.path.join(pkgdir, "lib", "openvino", "libopenvino.so*"))
    needs_ov = None
    try:
        needs_ov = _libwhisper_needs_openvino(lw)
    except FileNotFoundError:
        print("  SKIP OpenVINO check (readelf unavailable)")
    except subprocess.CalledProcessError as error:
        _check("inspect libwhisper dependencies", False, _readelf_failure(error))
    else:
        if needs_ov is True:
            _check("bundled OpenVINO present", bool(ov), ov[0] if ov else "missing")
            loaded_openvino = getattr(quoth, "OPENVINO_LIB", None)
            loaded_openvino_error = None
            if loaded_openvino and package_channel:
                try:
                    resolved_openvino = Path(loaded_openvino).resolve(strict=True)
                except OSError as error:
                    loaded_openvino_error = str(error)
                else:
                    package_lib = (Path(pkgdir) / "lib").resolve()
                    if not resolved_openvino.is_file():
                        loaded_openvino_error = "target is not a regular file"
                    elif not resolved_openvino.is_relative_to(package_lib):
                        loaded_openvino_error = f"resolves outside {package_lib}"
            _check(
                "OpenVINO core was preloaded",
                bool(
                    loaded_openvino
                    and os.path.exists(loaded_openvino)
                    and loaded_openvino_error is None
                ),
                loaded_openvino_error or loaded_openvino or "no loaded OpenVINO core",
            )
        elif needs_ov is False:
            print("  OK   Vulkan-only build: no OpenVINO expected")

    # Loading libwhisper proves its NEEDED set resolves, including the bundled
    # OpenVINO core (preloaded by __init__) on a WITH_OPENVINO build.
    whisper_lib = None
    try:
        if lw:
            whisper_lib = ctypes.CDLL(lw)
        _check("CDLL libwhisper (NEEDED resolves)", whisper_lib is not None)
    except OSError as e:
        _check("CDLL libwhisper", False, str(e))
    try:
        if whisper_lib is not None and lw:
            w._ensure_cpu_backend(whisper_lib, lw)
        _check("CPU backend registration", whisper_lib is not None)
    except (AttributeError, OSError, RuntimeError) as e:
        _check("CPU backend registration", False, str(e))

    # ThorVG renders the onboarding animations. Every channel bundles it (no distro
    # ships one), so unlike OpenVINO no build may lack it: this is what catches a
    # recipe that forgot `make vendor-thorvg`.
    tvg = next((p for p in native.thorvg_lib_paths() if os.path.exists(p)), None)
    _check("thorvg resolves", bool(tvg), tvg or "libthorvg-1.so not found")
    try:
        if tvg:
            ctypes.CDLL(tvg)
        _check("CDLL libthorvg", bool(tvg))
    except OSError as e:
        _check("CDLL libthorvg", False, str(e))

    # The .deb bundles Intel's NPU driver stack: it must ship and load on this distro.
    if package_channel == "debian":
        from quoth import hardware

        npu_dir = getattr(quoth, "NPU_DRIVER_DIR", None)
        _check(
            "bundled NPU driver loads",
            bool(npu_dir) and hardware._npu_user_driver_present(),
            npu_dir or "not bundled",
        )

    if package_channel:
        package_lib = Path(pkgdir) / "lib"
        packaged_objects = _shared_objects(package_lib)
        engine = Path(os.path.abspath(lw)) if lw else None
        if engine is not None and not engine.is_relative_to(package_lib):
            engine = None
        loaded_openvino = getattr(quoth, "OPENVINO_LIB", None)
        openvino_core = Path(loaded_openvino) if loaded_openvino else None
        _check(
            "native engine is packaged",
            engine is not None,
            lw or "libwhisper not found",
        )
        bundle_errors = _native_bundle_errors(package_lib, package_channel)
        _check(
            "native bundle inventory",
            not bundle_errors,
            "; ".join(bundle_errors),
        )
        missing_terms = []
        for bundled, notice, terms in _COMPILED_IN_TERMS:
            if not (package_lib / bundled).exists():
                continue
            try:
                text = (package_lib / notice).read_text(errors="replace")
            except OSError:
                text = ""
            missing_terms += [f"{notice}: {t}" for t in terms if t not in text]
        _check(
            "bundled notices carry their compiled-in terms",
            not missing_terms,
            ", ".join(missing_terms),
        )
        _check(
            "packaged shared objects found",
            bool(packaged_objects),
            f"{len(packaged_objects)} object(s)",
        )
        # The Flatpak installs the NPU driver, its loader and its compiler in its own
        # lib dir, where the driver looks for its compiler, outside the package.
        driver_objects: tuple[Path, ...] = ()
        if package_channel == "flatpak":
            driver_objects = tuple(_flatpak_npu_objects(prefix / "lib"))
            _check(
                "Flatpak NPU driver objects found",
                bool(driver_objects),
                f"{len(driver_objects)} object(s) in {prefix / 'lib'}",
            )
        closure_errors = _dynamic_closure_errors(
            Path(pkgdir),
            engine=engine,
            openvino_core=openvino_core,
            driver_objects=driver_objects,
        )
        _check(
            "packaged shared-object closure",
            not closure_errors,
            "; ".join(closure_errors),
        )

    # gtk4-layer-shell via GI: system on Arch, bundled on .deb/Flatpak (the __init__
    # preload + GI_TYPELIB_PATH wiring makes either resolve).
    try:
        import gi

        if package_channel == "flatpak":
            from gi import _gi

            _check(
                "PyGObject uses bundled ownership fix",
                all(
                    Path(path).resolve().is_relative_to("/app")
                    for path in (gi.__file__, _gi.__file__)
                ),
                f"{gi.__file__}; {_gi.__file__}",
            )

        gi.require_foreign("cairo")
        gi.require_version("Gtk", "4.0")
        gi.require_version("Adw", "1")
        gi.require_version("Gtk4LayerShell", "1.0")
        from gi.repository import Adw, GLib, Gtk, Gtk4LayerShell

        _check(
            "GTK4 and Adwaita load with Cairo bridge",
            bool(Gtk.Window and Adw.Application),
        )
        _check(
            "gtk4-layer-shell loads via GI", hasattr(Gtk4LayerShell, "init_for_window")
        )
    except (ImportError, ValueError) as e:
        _check("gtk4-layer-shell loads via GI", False, str(e))
    else:
        if package_channel:
            try:
                media_ok, media_detail = _media_file_check(
                    Gtk, GLib, Path(pkgdir) / "backgrounds" / "onboarding.webm"
                )
            except Exception as error:
                _check("packaged VP9 WebM prepares", False, str(error))
            else:
                _check("packaged VP9 WebM prepares", media_ok, media_detail)

        if package_channel == "flatpak":
            try:
                gi.require_version("Xdp", "1.0")
                from gi.repository import Xdp
            except (ImportError, ValueError) as error:
                _check("Xdp portal API imports", False, str(error))
            else:
                xdp_methods = {
                    "Portal": (
                        "initable_new",
                        "create_global_shortcuts_session",
                        "create_global_shortcuts_session_finish",
                        "request_background",
                        "request_background_finish",
                    ),
                    "GlobalShortcut": ("new",),
                    "GlobalShortcutAssigned": (
                        "get_shortcut_id",
                        "get_trigger_description",
                    ),
                    "GlobalShortcutsSession": (
                        "get_session",
                        "bind_shortcuts",
                        "bind_shortcuts_finish",
                        "configure_shortcuts",
                        "configure_shortcuts_finish",
                        "close",
                    ),
                    "Session": ("close",),
                }
                missing_xdp = [
                    f"{class_name}.{method}"
                    for class_name, methods in xdp_methods.items()
                    for method in methods
                    if not hasattr(getattr(Xdp, class_name, None), method)
                ]
                _check(
                    "Xdp portal API imports",
                    not missing_xdp,
                    ", ".join(missing_xdp)
                    or "GI symbols present; portal access not tested",
                )

    # The capture stack: the Gst typelib and every element of the real pipeline (source,
    # convert, resample, tee, level, appsink). Built and torn down, never started, so it
    # needs no sound server, but a missing typelib or plugin package fails here.
    try:
        from quoth.audio import AudioPipeline

        AudioPipeline().destroy()
        _check("audio pipeline builds", True)
    except Exception as e:
        _check("audio pipeline builds", False, str(e))

    if package_channel:
        try:
            for label, ok, detail in _helper_probe_results(package_channel):
                _check(label, ok, detail)
        except Exception as error:
            _check("text-injection helper probes", False, str(error))

    # Build and packaging scripts use this same policy through --elf-root.
    try:
        path_errors = []
        for so in _shared_objects(Path(pkgdir) / "lib"):
            path_errors.extend(_path_errors(so, _read_dynamic(so)))
        _check("relocatable ELF paths", not path_errors, "; ".join(path_errors))
    except FileNotFoundError:
        print("  SKIP relocatable check (readelf unavailable)")
    except subprocess.CalledProcessError as error:
        _check("relocatable ELF paths", False, _readelf_failure(error))

    print("RESULT:", "PASS" if not failures else "FAIL")
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elf-root",
        type=Path,
        metavar="DIRECTORY",
        help="check bundled shared objects without importing Quoth",
    )
    parser.add_argument(
        "--compile-commands",
        type=Path,
        metavar="FILE",
        help="check native compile commands for baseline CPU safety",
    )
    parser.add_argument(
        "--package-channel",
        choices=("debian", "arch", "flatpak"),
        help="verify the channel-specific identity and Sentry SDK contract",
    )
    parser.add_argument("--expected-release")
    parser.add_argument(
        "--expected-environment",
        choices=("development", "candidate", "production"),
    )
    args = parser.parse_args(argv)
    contract_args = (
        args.package_channel,
        args.expected_release,
        args.expected_environment,
    )
    if any(contract_args) and not all(contract_args):
        parser.error(
            "--package-channel, --expected-release and --expected-environment "
            "must be set together"
        )
    standalone_modes = (args.elf_root, args.compile_commands)
    if sum(mode is not None for mode in standalone_modes) > 1:
        parser.error("--elf-root and --compile-commands cannot be combined")
    if any(mode is not None for mode in standalone_modes):
        if any(contract_args):
            parser.error(
                "package contract options cannot be used with standalone checks"
            )
    if args.elf_root is not None:
        return _check_elf_root(args.elf_root)
    if args.compile_commands is not None:
        return _check_compile_commands(args.compile_commands)
    return _installed_smoke(*contract_args)


if __name__ == "__main__":
    sys.exit(main())
