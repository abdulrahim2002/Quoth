# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for the packaging harnesses, workflows, recipes and the scripts they run."""

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tomllib
from pathlib import Path

import pytest
import yaml

from scripts import check_transcription
from tests.support import compile_commands, whisper_source_tree

_ROOT = Path(__file__).resolve().parents[1]


def _identity_step(name):
    """A step of package.yml's identity job, by name."""
    workflow = yaml.safe_load((_ROOT / ".github/workflows/package.yml").read_text())
    steps = workflow["jobs"]["identity"]["steps"]
    (step,) = (step for step in steps if step.get("name") == name)
    return step


def _git(cwd, *args):
    """Run git in a scratch repository, blind to the host's own git configuration."""
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.mark.parametrize(
    ("target", "ref", "accepted"),
    [
        ("staging", "refs/heads/topic", True),
        ("production", "refs/heads/topic", False),
        ("production", "refs/tags/v0.2.0", False),
        ("production", "refs/tags/v0.1.0", True),
    ],
)
def test_workflow_release_identity(tmp_path, target, ref, accepted):
    script = _identity_step("Resolve identity")["run"]
    (tmp_path / "pyproject.toml").write_text('version = "0.1.0"\n')
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/sync_pins.sh").write_text("exit 0\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    source = "a" * 40
    _command(bin_dir, "git", f"echo {source}\n")
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "TARGET": target,
            "GITHUB_REF": ref,
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == accepted, result.stderr
    if accepted:
        expected = "quoth@0.1.0" + (f"+g{source}" if target == "staging" else "")
        environment = {"staging": "candidate", "production": "production"}[target]
        assert f"release={expected}\n" in output.read_text()
        assert f"environment={environment}\n" in output.read_text()
        assert f"source_commit={source}\n" in output.read_text()
    else:
        assert not output.exists()


@pytest.mark.parametrize(
    ("changed", "base", "packaging"),
    [
        ("src/quoth/app.py", "parent", "false"),
        ("src/quoth/backgrounds/dawn.svg", "parent", "true"),
        ("src/quoth/typer.py", "parent", "true"),
        ("src/quoth/hardware.py", "parent", "true"),
        ("packaging/pins.sh", "parent", "true"),
        (".github/workflows/package.yml", "parent", "true"),
        (".github/workflows/ci.yml", "parent", "false"),
        ("src/quoth/app.py", "", "true"),
        ("src/quoth/app.py", "0" * 40, "true"),
    ],
)
def test_workflow_builds_every_channel_unless_only_the_app_changed(
    tmp_path, changed, base, packaging
):
    step = _identity_step("Detect package input changes")
    # Only a pull request has a base: a push's own diff misses a run it cancelled.
    assert step["env"]["BASE"] == "${{ github.event.pull_request.base.sha }}"
    script = step["run"]

    _git(tmp_path, "init", "-q")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "base")
    parent = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / changed).parent.mkdir(parents=True)
    (tmp_path / changed).write_text("change\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "change")
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        cwd=tmp_path,
        env={
            **os.environ,
            "BASE": parent if base == "parent" else base,
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == f"packaging={packaging}\n"


def test_package_inputs_name_every_module_the_installed_check_imports():
    # A pull request changing only such a module must build the channel checking it.
    tree = ast.parse((_ROOT / "scripts/check_install.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module)
            imported |= {f"{node.module}.{alias.name}" for alias in node.names}
    sources = set()
    for name in imported:
        parts = name.split(".")
        if parts[0] != "quoth":
            continue
        sources.add("src/quoth/__init__.py")
        if len(parts) > 1:
            sources.add(f"src/quoth/{parts[1]}.py")
    # _channel.py is written at install time from the Makefile, itself an input.
    sources = {path for path in sources if (_ROOT / path).exists()}
    listed = set(_identity_step("Detect package input changes")["run"].split())

    assert "src/quoth/hardware.py" in sources
    assert sources <= listed, sorted(sources - listed)


def _copy_harness(repo, relative_path):
    script = repo / relative_path
    script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_ROOT / relative_path, script)
    return script


def _command(bin_dir, name, body):
    command = bin_dir / name
    command.write_text("#!/bin/bash\nset -eu\n" + body)
    command.chmod(0o755)


def _copy_whisper_patch_harness(repo):
    script = _copy_harness(repo, "scripts/build_whisper.sh")
    helper = _copy_harness(repo, "scripts/apply_whisper_patches.sh")
    _copy_harness(repo, "scripts/check_install.py")
    helper.chmod(0o755)
    shutil.copytree(_ROOT / "packaging/patches", repo / "packaging/patches")
    return script, helper


def _whisper_checkout(tmp_path, *, patched=True):
    """The build harness beside a fake whisper.cpp tree, and a bin dir faking glslc."""
    script, helper = _copy_whisper_patch_harness(tmp_path / "repo")
    whisper_dir = tmp_path / "whisper.cpp"
    whisper_source_tree(whisper_dir)
    source = whisper_dir / "src/whisper.cpp"
    if patched:
        subprocess.run(
            [helper, "--apply", whisper_dir],
            check=True,
            capture_output=True,
            text=True,
        )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _command(bin_dir, "glslc", "exit 0\n")
    return script, source, bin_dir


def _configure(script, tmp_path, bin_dir, **env):
    """Run the build's configure step on the checkout `_whisper_checkout` made."""
    return subprocess.run(
        ["bash", script, "--configure-only"],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "WHISPER_DIR": os.fspath(tmp_path / "whisper.cpp"),
            "BUILD_DIR": os.fspath(tmp_path / "build"),
            **env,
        },
        capture_output=True,
        text=True,
        timeout=10,
    )


def _install_data_tree(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(_ROOT / "data", repo / "data")
    shutil.copyfile(_ROOT / "Makefile", repo / "Makefile")
    lib = repo / "lib"
    lib.mkdir()
    for name in (
        "libwhisper.so",
        "libggml.so",
        "libwhisper_helper.so",
        "LICENSE.whisper-cpp",
    ):
        (lib / name).touch()
    return repo, tmp_path / "stage"


def _copy_packaging_version_tree(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(_ROOT / "packaging", repo / "packaging")
    shutil.copytree(_ROOT / "data", repo / "data")
    (repo / "scripts").mkdir()
    shutil.copyfile(_ROOT / "scripts/sync_pins.sh", repo / "scripts/sync_pins.sh")
    shutil.copyfile(
        _ROOT / "scripts/check_install.py", repo / "scripts/check_install.py"
    )
    shutil.copyfile(_ROOT / "pyproject.toml", repo / "pyproject.toml")
    (repo / "src/quoth").mkdir(parents=True)
    shutil.copyfile(_ROOT / "src/quoth/__init__.py", repo / "src/quoth/__init__.py")
    return repo


@pytest.mark.parametrize(
    ("relative_path", "old", "new", "label"),
    [
        (
            "src/quoth/__init__.py",
            '__version__ = "{version}"',
            '__version__ = "0.0.0-drift"',
            "quoth.__version__",
        ),
        (
            "data/dev.quoth.Quoth.metainfo.xml",
            '<release version="{version}"',
            '<release version="0.0.0-drift"',
            "latest AppStream release",
        ),
        (
            "packaging/flatpak/dev.quoth.Quoth.yaml",
            "/v{version}/quoth-{version}.tar.gz",
            "/v0.0.0-drift/quoth-0.0.0-drift.tar.gz",
            "flatpak manifest quoth url",
        ),
    ],
)
def test_packaging_version_check_rejects_app_metadata_drift(
    tmp_path, relative_path, old, new, label
):
    repo = _copy_packaging_version_tree(tmp_path)
    path = repo / relative_path
    version = tomllib.loads((repo / "pyproject.toml").read_text())["project"]["version"]
    path.write_text(path.read_text().replace(old.format(version=version), new, 1))

    result = subprocess.run(
        ["bash", "scripts/sync_pins.sh", "--check"],
        cwd=repo,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    # The drifted value itself: a pattern that matches nothing also reports drift.
    assert f"VERSION DRIFT: {label} = '0.0.0-drift'" in result.stdout


@pytest.mark.parametrize(
    ("channel_arg", "expected"),
    [(None, "source"), ("deb", "deb"), ("aur", "aur"), ("", None)],
)
def test_install_data_writes_channel_stamp(tmp_path, channel_arg, expected):
    repo, stage = _install_data_tree(tmp_path)
    command = [
        "make",
        "install-data",
        f"DESTDIR={stage}",
        "PREFIX=/usr",
        "PYSITE=/site-packages",
    ]
    if channel_arg is not None:
        command.append(f"QUOTH_CHANNEL={channel_arg}")

    result = subprocess.run(command, cwd=repo, capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
    stamp = stage / "site-packages" / "quoth" / "_channel.py"
    if expected is None:
        assert not stamp.exists()
    else:
        assert stamp.read_text() == f'CHANNEL = "{expected}"\n'
    for name in (
        "lucide-icons",
        "catppuccin",
        "dracula",
        "nord",
        "silero-vad",
        "loguru",
    ):
        source = repo / "data/licenses" / f"LICENSE.{name}"
        installed = stage / "usr/share/licenses/quoth" / f"LICENSE.{name}"
        assert installed.read_bytes() == source.read_bytes()
    for name in ("coffee", "github", "gauge", "bug", "heart"):
        assert (
            stage
            / "usr/share/icons/hicolor/symbolic/apps"
            / f"dev.quoth.Quoth-{name}-symbolic.svg"
        ).is_file()
    # The rules install with any channel, source by default; the Flatpak passes none.
    for name in ("60-quoth-uinput.rules", "60-quoth-accel.rules"):
        installed = stage / "usr/lib/udev/rules.d" / name
        assert installed.is_file() == (expected is not None)


def _stage_libs(tmp_path, *, vulkan_hpp):
    """Run stage_libs.sh on a whisper.cpp build, with or without Vulkan-Hpp's header."""
    script = _copy_harness(tmp_path / "repo", "scripts/stage_libs.sh")
    whisper = tmp_path / "whisper.cpp"
    (whisper / "build/bin").mkdir(parents=True)
    (whisper / "build/bin/libwhisper.so").touch()
    (whisper / "build/bin/libggml.so").touch()
    (whisper / "LICENSE").write_text(
        "MIT License\n\nCopyright (c) 2023-2026 The ggml authors\n"
    )
    (whisper / "ggml/src/ggml-cpu").mkdir(parents=True)
    (whisper / "ggml/src/ggml-cpu/ops.cpp").write_text(
        "// MIT License\n// Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng\n"
    )
    include = tmp_path / "include"
    (whisper / "build/CMakeCache.txt").write_text(
        f"Vulkan_INCLUDE_DIR:PATH={include}\n"
    )
    if vulkan_hpp:
        (include / "vulkan").mkdir(parents=True)
        (include / "vulkan/vulkan.hpp").write_text(
            "// Copyright 2015-2025 The Khronos Group Inc.\n"
            "//\n// SPDX-License-Identifier: Apache-2.0 OR MIT\n"
        )

    return subprocess.run(
        ["bash", script, whisper / "build", tmp_path / "lib"],
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_stage_libs_appends_the_compiled_in_terms(tmp_path):
    result = _stage_libs(tmp_path, vulkan_hpp=True)

    assert result.returncode == 0, result.stdout + result.stderr
    notice = (tmp_path / "lib/LICENSE.whisper-cpp").read_text()
    assert notice.startswith(
        "MIT License\n\nCopyright (c) 2023-2026 The ggml authors\n"
    )
    assert "Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng" in notice
    assert "Copyright 2015-2025 The Khronos Group Inc." in notice
    assert notice.count("Permission is hereby granted") == 2


def test_stage_libs_refuses_a_build_without_vulkan_hpp(tmp_path):
    result = _stage_libs(tmp_path, vulkan_hpp=False)

    assert result.returncode == 1
    assert "vulkan.hpp not found" in result.stderr


def _pinned_checkouts(tmp_path, *, attributes=""):
    """The .deb's copyleft checkouts, the harness that packs them, and their commits."""
    script = _copy_harness(tmp_path / "repo", "scripts/deb_sources.sh")
    (tmp_path / "repo/pyproject.toml").write_text('version = "0.1.0"\n')
    pins = []
    commits = {}
    for name, var in (
        ("wl-clipboard", "WL_CLIPBOARD"),
        ("ydotool", "YDOTOOL"),
        ("thorvg", "THORVG"),
    ):
        checkout = tmp_path / "src" / name
        checkout.mkdir(parents=True)
        (checkout / "COPYING").write_text(f"{name} terms\n")
        (checkout / ".gitattributes").write_text(attributes)
        _git(checkout, "init", "-q")
        _git(checkout, "add", ".")
        _git(checkout, "commit", "-q", "-m", "release")
        commits[name] = _git(checkout, "rev-parse", "HEAD")
        pins += [f'QUOTH_{var}_TAG="v1.0"', f'QUOTH_{var}_COMMIT="{commits[name]}"']
    (tmp_path / "repo/packaging").mkdir()
    (tmp_path / "repo/packaging/pins.sh").write_text("\n".join(pins) + "\n")
    # The build distribution's wayland-protocols, as pkg-config and dpkg report it.
    protocols = tmp_path / "share/wayland-protocols"
    for xml in (*_WL_CLIPBOARD_PROTOCOLS, "staging/unused/unused-v1.xml"):
        (protocols / xml).parent.mkdir(parents=True, exist_ok=True)
        (protocols / xml).write_text(f"<protocol name='{Path(xml).stem}'/>\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _command(bin_dir, "pkg-config", f"echo /{protocols}\n")
    _command(bin_dir, "dpkg-query", "printf 1.45-1\n")
    return script, commits


# The files wl-clipboard's meson.build reads from wayland-protocols.
_WL_CLIPBOARD_PROTOCOLS = (
    "stable/xdg-shell/xdg-shell.xml",
    "unstable/primary-selection/primary-selection-unstable-v1.xml",
    "staging/ext-data-control/ext-data-control-v1.xml",
    "staging/xdg-activation/xdg-activation-v1.xml",
)


def _deb_sources(script, tmp_path, name, umask="022"):
    command = (script, tmp_path / "src", tmp_path / name)
    return subprocess.run(
        ["bash", "-c", f'umask {umask}; exec bash "$@"', "bash", *command],
        env={**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_deb_sources_packs_each_pinned_tree_reproducibly(tmp_path):
    script, commits = _pinned_checkouts(tmp_path)

    first = _deb_sources(script, tmp_path, "first.tar.gz")
    # Modes must not depend on the umask of whoever packs it.
    second = _deb_sources(script, tmp_path, "second.tar.gz", umask="077")

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    packed = (tmp_path / "first.tar.gz").read_bytes()
    assert packed == (tmp_path / "second.tar.gz").read_bytes()
    with tarfile.open(tmp_path / "first.tar.gz") as archive:
        names = archive.getnames()
        modes = {member.mode for member in archive.getmembers()}
        readme = archive.extractfile("quoth-deb-sources/README").read().decode()
    assert modes == {0o644, 0o755}
    protocols = "quoth-deb-sources/wayland-protocols-1.45-1"
    assert sorted(n for n in names if n.endswith(".xml")) == sorted(
        f"{protocols}/{xml}" for xml in _WL_CLIPBOARD_PROTOCOLS
    )
    assert "version 1.45-1." in readme
    assert "MIT and HPND-sell-variant" in readme
    assert f"under {tmp_path / 'share/wayland-protocols'}." in readme
    for name, commit in commits.items():
        assert f"quoth-deb-sources/{name}-v1.0/COPYING" in names
        assert commit in readme
    assert "quoth_0.1.0_amd64.deb" in readme
    assert "quoth-0.1.0.tar.gz" in readme


@pytest.mark.parametrize(
    ("attributes", "later_commit", "expected"),
    [
        pytest.param("", True, "ydotool is at", id="commit"),
        pytest.param(
            "COPYING export-ignore\n",
            False,
            "changes what git archive",
            id="export-ignore",
        ),
    ],
)
def test_deb_sources_refuses_a_tree_other_than_the_pinned_one(
    tmp_path, attributes, later_commit, expected
):
    script, _commits = _pinned_checkouts(tmp_path, attributes=attributes)
    if later_commit:
        _git(tmp_path / "src/ydotool", "commit", "-q", "--allow-empty", "-m", "later")

    result = _deb_sources(script, tmp_path, "out.tar.gz")

    assert result.returncode == 1
    assert expected in result.stderr


def test_install_data_rejects_unknown_channel(tmp_path):
    repo, stage = _install_data_tree(tmp_path)

    result = subprocess.run(
        [
            "make",
            "install-data",
            f"DESTDIR={stage}",
            "PREFIX=/usr",
            "PYSITE=/site-packages",
            "QUOTH_CHANNEL=snap",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "invalid QUOTH_CHANNEL: snap" in result.stderr


@pytest.mark.parametrize("already_patched", [False, True])
def test_openvino_patch_preflight_never_mutates_source(tmp_path, already_patched):
    script, source, bin_dir = _whisper_checkout(tmp_path, patched=already_patched)
    original = source.read_text()
    compile_commands(tmp_path / "build/compile_commands.json")
    _command(bin_dir, "cmake", "exit 0\n")

    result = _configure(script, tmp_path, bin_dir, WITH_OPENVINO="ON")

    assert source.read_text() == original
    assert result.returncode == (0 if already_patched else 1)
    if already_patched:
        assert "configure succeeded" in result.stdout
    else:
        assert "Apply the full patch set:" in result.stderr


@pytest.mark.parametrize(
    "patch",
    sorted(p.name for p in (_ROOT / "packaging/patches").glob("whisper-*.patch")),
)
def test_whisper_build_refuses_a_checkout_missing_any_one_patch(tmp_path, patch):
    script, _source, bin_dir = _whisper_checkout(tmp_path)
    subprocess.run(
        [
            "patch",
            "--reverse",
            "--silent",
            "-p1",
            "-i",
            _ROOT / "packaging/patches" / patch,
        ],
        cwd=tmp_path / "whisper.cpp",
        check=True,
        timeout=10,
    )
    _command(bin_dir, "cmake", "exit 0\n")

    result = _configure(script, tmp_path, bin_dir)

    assert result.returncode == 1
    assert f"whisper.cpp is missing packaging/patches/{patch}" in result.stderr


def test_patch_set_refuses_source_that_has_drifted(tmp_path):
    # A changed context line means the pin moved under the patch, which needs a look.
    _script, source, _bin_dir = _whisper_checkout(tmp_path, patched=False)
    source.write_text("// drifted\n" + source.read_text().split("\n", 1)[1])

    result = subprocess.run(
        [
            tmp_path / "repo/scripts/apply_whisper_patches.sh",
            "--apply",
            tmp_path / "whisper.cpp",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0


def test_openvino_patch_preflight_reports_missing_patch_tool(tmp_path):
    script, _source, bin_dir = _whisper_checkout(tmp_path, patched=False)
    # Only these commands are on PATH, so `patch` is not.
    _command(bin_dir, "bash", 'exec /bin/bash "$@"\n')
    _command(bin_dir, "dirname", 'printf "%s\\n" "${1%/*}"\n')
    _command(bin_dir, "nproc", "echo 1\n")
    _command(bin_dir, "cmake", "exit 0\n")

    result = _configure(
        script, tmp_path, bin_dir, PATH=os.fspath(bin_dir), WITH_OPENVINO="ON"
    )

    assert result.returncode == 1
    assert "ERROR: patch missing" in result.stderr


@pytest.mark.parametrize("with_openvino", ["ON", "OFF", None])
def test_whisper_build_configures_runtime_cpu_dispatch(
    tmp_path, monkeypatch, with_openvino
):
    script, _source, bin_dir = _whisper_checkout(tmp_path)
    args_log = tmp_path / "cmake-args"
    compile_commands(tmp_path / "build/compile_commands.json")
    _command(bin_dir, "cmake", 'printf \'%s\\n\' "$@" > "$CMAKE_ARGS_LOG"\n')
    monkeypatch.delenv("WITH_OPENVINO", raising=False)
    env = {"CMAKE_ARGS_LOG": os.fspath(args_log)}
    if with_openvino is not None:
        env["WITH_OPENVINO"] = with_openvino
    openvino = with_openvino or "OFF"

    result = _configure(script, tmp_path, bin_dir, **env)

    assert result.returncode == 0, result.stdout + result.stderr
    args = args_log.read_text().splitlines()
    for expected in (
        "-DBUILD_SHARED_LIBS=ON",
        "-DGGML_VULKAN=ON",
        "-DGGML_NATIVE=OFF",
        "-DGGML_BACKEND_DL=ON",
        "-DGGML_CPU_ALL_VARIANTS=ON",
        f"-DWHISPER_OPENVINO={openvino}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ):
        assert expected in args
    assert any(arg.startswith("-DOpenVINO_DIR=") for arg in args) == (openvino == "ON")
    assert "baseline, x64 and CPU scorer objects are ISA-safe" in result.stdout


def test_the_shim_build_passes_the_distro_build_flags(tmp_path):
    script = _copy_harness(tmp_path / "repo", "lib/build_helper.sh")
    args_log = tmp_path / "gcc-args"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _command(bin_dir, "gcc", 'printf \'%s\\n\' "$@" > "$GCC_ARGS_LOG"\n')

    result = subprocess.run(
        ["bash", script],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PYTHON": "true",
            "CPPFLAGS": "-D_FORTIFY_SOURCE=3",
            "CFLAGS": "-O2 -fstack-protector-strong",
            "LDFLAGS": "-Wl,-z,relro -Wl,-z,now",
            "GCC_ARGS_LOG": os.fspath(args_log),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    args = args_log.read_text().splitlines()
    for flag in (
        "-D_FORTIFY_SOURCE=3",
        "-O2",
        "-fstack-protector-strong",
        "-Wl,-z,relro",
        "-Wl,-z,now",
    ):
        assert flag in args


def test_whisper_build_rejects_unsafe_compile_commands(tmp_path):
    script, _source, bin_dir = _whisper_checkout(tmp_path)
    path = tmp_path / "build/compile_commands.json"
    rows = compile_commands(path)
    row = next(row for row in rows if "CMakeFiles/ggml-cpu-x64.dir/" in str(row))
    row["arguments"].append("-mavx2")
    path.write_text(json.dumps(rows))
    _command(bin_dir, "cmake", "exit 0\n")

    result = _configure(script, tmp_path, bin_dir)

    assert result.returncode == 1
    assert "ggml-cpu-x64 uses unsafe CPU flag -mavx2" in result.stdout
    assert "configure succeeded" not in result.stdout


@pytest.mark.parametrize("stale_kind", ["file", "dangling-symlink"])
def test_whisper_build_rejects_stale_monolithic_cpu_artifact(tmp_path, stale_kind):
    script, _source, bin_dir = _whisper_checkout(tmp_path)
    stale = tmp_path / "build/bin/libggml-cpu.so"
    stale.parent.mkdir(parents=True)
    if stale_kind == "file":
        stale.touch()
    else:
        stale.symlink_to("missing")
    _command(bin_dir, "cmake", "raise SystemExit('cmake must not run')\n")

    result = _configure(script, tmp_path, bin_dir)

    assert result.returncode == 1
    assert "stale monolithic CPU library" in result.stderr
    assert "Use a clean BUILD_DIR" in result.stderr


def test_transcription_evidence_requires_one_package_owned_cpu_module(
    tmp_path, monkeypatch
):
    package_lib = tmp_path / "installed package/quoth/lib"
    package_lib.mkdir(parents=True)
    selected = package_lib / "libggml-cpu-alderlake.so"
    selected.touch()
    maps = tmp_path / "maps"
    maps.write_text(
        f"7f00-7f10 r-xp 00000000 00:00 1 {selected}\n"
        "7f20-7f30 r-xp 00000000 00:00 2 /tmp/libggml-vulkan.so\n"
    )

    assert check_transcription._mapped_cpu_modules(maps) == [selected]
    monkeypatch.setattr(check_transcription, "_mapped_cpu_modules", lambda: [selected])
    assert check_transcription._selected_cpu_module(package_lib) == selected

    outside = tmp_path / "libggml-cpu-x64.so"
    outside.touch()
    monkeypatch.setattr(check_transcription, "_mapped_cpu_modules", lambda: [outside])
    with pytest.raises(RuntimeError, match="outside the package"):
        check_transcription._selected_cpu_module(package_lib)


def test_every_native_recipe_applies_the_shared_whisper_patch_set():
    assert '--check "$WHISPER_DIR"' in (_ROOT / "scripts/build_whisper.sh").read_text()

    applications = {
        "debian": (_ROOT / "packaging/debian/.dockervalidate/run.sh").read_text(),
        "aur": (_ROOT / "packaging/aur/PKGBUILD").read_text(),
        "aur-git": (_ROOT / "packaging/aur/quoth-git/PKGBUILD").read_text(),
        "flatpak": (_ROOT / "packaging/flatpak/dev.quoth.Quoth.yaml").read_text(),
    }
    for recipe in applications.values():
        assert "apply_whisper_patches.sh" in recipe
        assert "--apply" in recipe


def test_deb_installs_disable_recommends():
    workflow = (_ROOT / ".github/workflows/package.yml").read_text()
    local = (_ROOT / "packaging/debian/.dockervalidate/run.sh").read_text()
    matrix = (_ROOT / "packaging/debian/.dockervalidate/matrix.sh").read_text()

    assert "apt-get install -y --no-install-recommends /package/*.deb" in workflow
    assert 'apt-get install -y --no-install-recommends "$DEB"' in local
    assert "apt-get install -y -qq --no-install-recommends /deb/$DEBNAME" in matrix
    assert "apt-get install -y --no-install-recommends /deb/$DEBNAME" in matrix


def test_debian_harness_lints_the_package_before_installing_it():
    harness = (_ROOT / "packaging/debian/.dockervalidate/run.sh").read_text()

    build_package = harness.index("dpkg-buildpackage -b -us -uc")
    lint = harness.index('lintian --fail-on error,warning --allow-root "$DEB"')
    install = harness.index('apt-get install -y --no-install-recommends "$DEB"')
    assert build_package < lint < install


def test_deb_build_publishes_the_sources_of_its_components(tmp_path):
    harness = (_ROOT / "packaging/debian/.dockervalidate/run.sh").read_text()
    verified = harness.index('YDT_ACTUAL="$(git -C ydotool rev-parse HEAD)"')
    packed = harness.index('"$SRC/scripts/deb_sources.sh" . "$BUILD/quoth-deb-sources')
    assert verified < packed < harness.index("dpkg-buildpackage -b -us -uc")
    # It reads the wayland-protocols the build dependencies install.
    assert harness.index("apt-get build-dep -y ./") < packed
    assert 'cp "$DEB" "$BUILD/quoth-deb-sources.tar.gz" "$OUTPUT_DIR/"' in harness

    workflow = yaml.safe_load((_ROOT / ".github/workflows/package.yml").read_text())
    steps = workflow["jobs"]["build"]["steps"]
    (build,) = (step for step in steps if step.get("name") == "Build and export")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # The container writes the package and the sources to the /out mount.
    _command(
        bin_dir,
        "docker",
        "out=\"$(printf '%s\\n' \"$@\" | sed -n 's#:/out$##p')\"\n"
        '[ -z "$out" ] || { echo deb > "$out/quoth_0.1.0_amd64.deb"; '
        'echo src > "$out/quoth-deb-sources.tar.gz"; }\n',
    )
    runner = tmp_path / "runner"
    runner.mkdir()
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", build["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "RUNNER_TEMP": os.fspath(runner),
            "SOURCE_COMMIT": "0" * 40,
            "RELEASE_METADATA": "{}",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    sidecars = {"SHA256SUMS", "source-commit.txt", "release-metadata.json"}
    package = runner / "package"
    sources = runner / "package-sources"
    assert {p.name for p in package.iterdir()} == sidecars | {"quoth_0.1.0_amd64.deb"}
    assert {p.name for p in sources.iterdir()} == sidecars | {
        "quoth-deb-sources.tar.gz"
    }
    for directory in (package, sources):
        subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=directory, check=True)
    assert _attested(workflow) >= {
        "${{ runner.temp }}/package-sources/quoth-deb-sources.tar.gz"
    }
    uploads = {
        s["with"]["name"]: s["with"]["path"]
        for s in steps
        if s.get("uses", "").startswith("actions/upload-artifact@")
    }
    assert (
        uploads["deb-sources-${{ github.run_id }}-${{ github.run_attempt }}"]
        == "${{ runner.temp }}/package-sources/"
    )


_ARCH_LIB = "usr/lib/python3.14/site-packages/quoth/lib"


@pytest.mark.parametrize(
    ("report", "accepted"),
    [
        ("", True),
        ("quoth W: dependency-not-needed wtype", True),
        (f"quoth W: elffile-without-relro {_ARCH_LIB}/openvino/libhwloc.so.15", True),
        ("quoth W: dependency-not-needed python-numpy", False),
        (f"quoth W: elffile-without-relro {_ARCH_LIB}/libwhisper.so.1.9.5", False),
        ("quoth W: dependency-not-needed wtype\nquoth E: missing-license", False),
    ],
)
def test_aur_harness_fails_on_namcap_reports_it_does_not_accept(
    tmp_path, report, accepted
):
    script = _copy_harness(tmp_path / "repo", "packaging/aur/.dockervalidate/run.sh")
    src = tmp_path / "src"
    (src / "packaging/aur").mkdir(parents=True)
    (src / "pyproject.toml").write_text('version = "0.1.0"\n')
    (src / "packaging/aur/PKGBUILD").touch()
    (src / "packaging/aur/quoth.install").touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _command(bin_dir, "sudo", "")
    _command(
        bin_dir,
        "makepkg",
        '[ "$1" != --packagelist ] || echo "$PWD/quoth-0.1.0-1-x86_64.pkg.tar.zst"\n',
    )
    _command(
        bin_dir, "namcap", 'echo "$@" > "$NAMCAP_ARGS"\nprintf "%s" "$NAMCAP_REPORT"\n'
    )
    namcap_args = tmp_path / "namcap-args"

    result = subprocess.run(
        ["bash", script],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "HOME": os.fspath(tmp_path / "home"),
            "SRC": os.fspath(src),
            "SOURCE_COMMIT": "0" * 40,
            "NAMCAP_ARGS": os.fspath(namcap_args),
            "NAMCAP_REPORT": report,
        },
        capture_output=True,
        text=True,
        timeout=10,
    )

    package = tmp_path / "home/build/quoth-0.1.0-1-x86_64.pkg.tar.zst"
    assert namcap_args.read_text().split() == ["-m", "PKGBUILD", os.fspath(package)]
    assert ("==> smoke test" in result.stdout) == accepted
    if not accepted:
        assert result.returncode != 0
        assert report.splitlines()[-1] in result.stdout


def test_package_workflow_runs_read_only_weekly_staging_canary():
    workflow = (_ROOT / ".github/workflows/package.yml").read_text()

    assert "schedule:\n    - cron: '17 5 * * 1'" in workflow
    assert workflow.count("'src/quoth/**'") == 2
    assert "permissions:\n  contents: read" in workflow
    assert "TARGET: ${{ inputs.target || 'staging' }}" in workflow


def test_package_workflow_starts_only_on_its_own_workflow_file():
    # No package job runs another workflow, so an edit to one alone builds nothing.
    workflow = (_ROOT / ".github/workflows/package.yml").read_text()

    assert workflow.count("- '.github/workflows/package.yml'") == 2
    assert "'.github/workflows/**'" not in workflow


@pytest.mark.parametrize(
    "job",
    yaml.safe_load((_ROOT / ".github/workflows/package.yml").read_text())[
        "jobs"
    ].values(),
    ids=lambda job: job["name"],
)
def test_a_failed_package_job_summarises_the_logs_it_tees(tmp_path, job):
    # It shows the end of each log the job tees, and a missing log cannot fail it.
    scripts = "\n".join(step.get("run", "") for step in job["steps"])
    teed = set(re.findall(r'tee "\$RUNNER_TEMP/([\w.-]+)"', scripts))
    summaries = [
        step for step in job["steps"] if step.get("name") == "Summarise the failure"
    ]
    if not teed:
        assert not summaries
        return
    (step,) = summaries
    assert job["steps"][-1] is step
    assert step["if"] == "failure()"
    assert step["continue-on-error"] is True
    logs = re.search(r"tail -v -n 40 (.+) \|\| true", step["run"]).group(1).split()
    assert set(logs) <= teed

    (tmp_path / logs[0]).write_text("".join(f"line {n}\n" for n in range(100)))
    summary = tmp_path / "summary.md"
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]],
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    text = summary.read_text()
    assert text.startswith("```\n==> ") and text.endswith("```\n")
    assert "line 99\n" in text and "line 59\n" not in text


def test_flatpak_install_inspects_its_elf_files_from_the_host():
    # The Platform has no readelf, so the sandboxed check cannot inspect ELF files.
    workflow = (_ROOT / ".github/workflows/package.yml").read_text()

    assert (
        'python3 -I "$GITHUB_WORKSPACE/scripts/check_install.py" --elf-root' in workflow
    )


_ATTEST_ACTION = "actions/attest@1e69f48acb82d1966a394da916b4c1698aa569d6"


def _attested(workflow):
    """Each provenance subject the attest job signs, as the path its build job wrote.

    A subject is downloaded from a run artifact; mapping it back through that
    artifact's upload names the file the build produced.
    """
    uploads = {
        step["with"]["name"]: step["with"]["path"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    }
    steps = workflow["jobs"]["attest"]["steps"]
    downloads = {
        step["with"]["path"]: uploads[step["with"]["name"]]
        for step in steps
        if step.get("uses", "").startswith("actions/download-artifact@")
    }
    attested = set()
    for step in steps:
        if step.get("uses", "").startswith("actions/attest@") and set(step["with"]) == {
            "subject-path"
        }:
            for subject in step["with"]["subject-path"].splitlines():
                directory, name = subject.rsplit("/", 1)
                attested.add(downloads[directory + "/"] + name)
    return attested


def test_only_the_attest_job_holds_the_signing_token():
    workflow = yaml.safe_load((_ROOT / ".github/workflows/package.yml").read_text())
    assert workflow["permissions"] == {"contents": "read"}
    signing = {
        key
        for key, job in workflow["jobs"].items()
        if set(job.get("permissions", {})) - {"contents"}
    }
    assert signing == {"attest"}
    attest = workflow["jobs"]["attest"]
    assert attest["permissions"] == {
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
    }
    assert set(attest["needs"]) >= {"identity", "build", "flatpak-build"}
    # The documented verification needs a hosted runner wherever the builds ran.
    assert attest["runs-on"] == workflow["jobs"]["build"]["runs-on"]
    attesting = [
        (key, step)
        for key, job in workflow["jobs"].items()
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/attest@")
    ]
    assert attesting
    for key, step in attesting:
        assert key == "attest"
        assert step["uses"] == _ATTEST_ACTION


def test_the_attest_job_signs_the_four_release_packages_it_checked():
    workflow = yaml.safe_load((_ROOT / ".github/workflows/package.yml").read_text())
    # The files the build jobs attested themselves before this job existed.
    assert _attested(workflow) == {
        "${{ runner.temp }}/package/*.deb",
        "${{ runner.temp }}/package-sources/quoth-deb-sources.tar.gz",
        "${{ runner.temp }}/package/quoth.flatpak",
        "${{ runner.temp }}/package-sources/quoth-sources.flatpak",
    }
    steps = workflow["jobs"]["attest"]["steps"]
    downloaded = [
        step["with"]
        for step in steps
        if step.get("uses", "").startswith("actions/download-artifact@")
        and "-sbom-" not in step["with"]["name"]
    ]
    assert sorted(d["name"] for d in downloaded) == sorted(
        f"{name}-${{{{ github.run_id }}}}-${{{{ github.run_attempt }}}}"
        for name in ("deb", "deb-sources", "flatpak", "flatpak-sources")
    )
    (check,) = (
        s for s in steps if s.get("name") == "Verify package and source identity"
    )
    first_attestation = next(
        i
        for i, s in enumerate(steps)
        if s.get("uses", "").startswith("actions/attest@")
    )
    assert steps.index(check) < first_attestation
    # Every downloaded directory goes through the same check as the install jobs.
    checked = re.fullmatch(
        r"for artifact in ([\w -]+); do\n"
        r'  bash scripts/check_package_identity.sh "\$RUNNER_TEMP/\$artifact"\n'
        r"done\n",
        check["run"],
    )
    assert sorted(checked.group(1).split()) == sorted(
        d["path"].removeprefix("${{ runner.temp }}/").rstrip("/") for d in downloaded
    )


@pytest.mark.parametrize(
    ("channel", "job", "package"),
    [
        ("deb", "build", "${{ runner.temp }}/deb/*.deb"),
        ("flatpak", "flatpak-build", "${{ runner.temp }}/flatpak/quoth.flatpak"),
    ],
)
def test_the_attest_job_attests_each_package_with_its_build_jobs_bill(
    channel, job, package
):
    workflow = yaml.safe_load((_ROOT / ".github/workflows/package.yml").read_text())
    build = workflow["jobs"][job]["steps"]
    (written,) = (s for s in build if s.get("name") == "Write the bill of materials")
    output = f"$RUNNER_TEMP/quoth-{channel}.cdx.json"
    assert written["run"].split() == [
        "python3",
        "-I",
        "scripts/package_sbom.py",
        channel,
        "--commit",
        '"$SOURCE_COMMIT"',
        "\\",
        "--output",
        f'"{output}"',
    ]
    artifact = f"{channel}-sbom-${{{{ github.run_id }}}}-${{{{ github.run_attempt }}}}"
    (upload,) = (
        s["with"]
        for s in build
        if s.get("uses", "").startswith("actions/upload-artifact@")
        and s["with"]["name"] == artifact
    )
    assert upload["path"] == output.replace("$RUNNER_TEMP", "${{ runner.temp }}")
    assert upload["if-no-files-found"] == "error"
    steps = workflow["jobs"]["attest"]["steps"]
    (download,) = (
        s["with"]
        for s in steps
        if s.get("uses", "").startswith("actions/download-artifact@")
        and s["with"]["name"] == artifact
    )
    (attestation,) = (
        s["with"]
        for s in steps
        if s.get("uses") == _ATTEST_ACTION
        and "sbom-path" in s["with"]
        and s["with"]["subject-path"] == package
    )
    assert set(attestation) == {"subject-path", "sbom-path"}
    assert attestation["sbom-path"] == download["path"] + f"quoth-{channel}.cdx.json"


def test_install_jobs_fetch_fixtures_into_the_cached_directory():
    # The cache restores the model where the script looks for it, keyed by its hash.
    workflow = yaml.safe_load((_ROOT / ".github/workflows/package.yml").read_text())
    fetching = 0
    for job in workflow["jobs"].values():
        names = [step.get("name") for step in job["steps"]]
        if "Fetch and verify public fixtures" not in names:
            continue
        fetching += 1
        fetch = job["steps"][names.index("Fetch and verify public fixtures")]
        cache = job["steps"][names.index("Fetch and verify public fixtures") - 1]
        assert cache["uses"].startswith("actions/cache@")
        assert cache["with"] == {
            "path": "${{ runner.temp }}/fixtures/ggml-base-q8_0.bin",
            "key": "transcription-model-${{ env.MODEL_SHA256 }}",
        }
        assert fetch["run"] == 'bash scripts/fetch_fixtures.sh "$RUNNER_TEMP/fixtures"'
    assert fetching == 3


@pytest.mark.parametrize(
    ("cached", "audio", "downloads", "accepted"),
    [
        pytest.param(None, "jfk", ["ggml-base-q8_0.bin", "jfk.wav"], True, id="fresh"),
        pytest.param("model", "jfk", ["jfk.wav"], True, id="cached"),
        pytest.param("stale", "jfk", ["jfk.wav"], False, id="stale-cache"),
        pytest.param(
            None, "other", ["ggml-base-q8_0.bin", "jfk.wav"], False, id="audio"
        ),
    ],
)
def test_fixture_fetch_keeps_a_cached_model_and_checks_both_hashes(
    tmp_path, cached, audio, downloads, accepted
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _command(
        bin_dir,
        "curl",
        'while [ "$1" != --output ]; do shift; done\n'
        'echo "$2" >> "$CURL_LOG"\n'
        'if [ "$2" = jfk.wav ]; then printf %s "$AUDIO"; else printf model; fi > "$2"\n',
    )
    fixtures = tmp_path / "fixtures"
    if cached is not None:
        fixtures.mkdir()
        (fixtures / "ggml-base-q8_0.bin").write_text(cached)
    log = tmp_path / "curl.log"
    log.touch()

    result = subprocess.run(
        ["bash", _ROOT / "scripts/fetch_fixtures.sh", fixtures],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "CURL_LOG": os.fspath(log),
            "AUDIO": audio,
            "MODEL_URL": "https://example.invalid/model",
            "MODEL_SHA256": hashlib.sha256(b"model").hexdigest(),
            "AUDIO_URL": "https://example.invalid/jfk.wav",
            "AUDIO_SHA256": hashlib.sha256(b"jfk").hexdigest(),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert log.read_text().split() == downloads
    assert (result.returncode == 0) == accepted, result.stdout + result.stderr
    if accepted:
        assert result.stdout == "ggml-base-q8_0.bin: OK\njfk.wav: OK\n"


@pytest.mark.parametrize("tamper", [None, "package", "commit", "metadata"])
def test_package_identity_check_refuses_another_package_commit_or_release(
    tmp_path, tamper
):
    repo = tmp_path / "repo"
    script = _copy_harness(repo, "scripts/check_package_identity.sh")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "release")
    head = _git(repo, "rev-parse", "HEAD")
    package = tmp_path / "package"
    package.mkdir()
    (package / "quoth.deb").write_text("deb\n")
    sums = subprocess.run(
        ["sha256sum", "quoth.deb"], cwd=package, check=True, capture_output=True
    )
    (package / "SHA256SUMS").write_bytes(sums.stdout)
    (package / "source-commit.txt").write_text(
        ("0" * 40 if tamper == "commit" else head) + "\n"
    )
    (package / "release-metadata.json").write_text('{"target":"staging"}\n')
    if tamper == "package":
        (package / "quoth.deb").write_text("other\n")
    metadata = '{"target":"production"}' if tamper == "metadata" else None

    result = subprocess.run(
        ["bash", script, package],
        cwd=tmp_path,
        env={**os.environ, "RELEASE_METADATA": metadata or '{"target":"staging"}'},
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert (result.returncode == 0) == (tamper is None), result.stderr
    if tamper is None:
        assert result.stdout == "quoth.deb: OK\n"
    # A refusal says which check failed and shows both values.
    elif tamper == "commit":
        assert result.stderr == (
            f"ERROR: {package} was built from {'0' * 40}, but this checkout is at {head}\n"
        )
    elif tamper == "metadata":
        assert result.stderr == (
            f"ERROR: {package}/release-metadata.json is\n"
            '  {"target":"staging"}\n'
            "but RELEASE_METADATA is\n"
            '  {"target":"production"}\n'
        )


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("failing", [None, "gpg", "remote-add", "remote-info"])
def test_repository_signature_check_verifies_apt_then_flatpak(
    tmp_path, failing, relative
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _command(bin_dir, "gpg", 'echo "gpg $*" >> "$LOG"\n[ "$FAIL" != gpg ]\n')
    _command(
        bin_dir,
        "flatpak",
        'echo "flatpak $FLATPAK_USER_DIR $*" >> "$LOG"\n[ "$FAIL" != "$1" ]\n',
    )
    log = tmp_path / "commands.log"
    log.touch()
    site = tmp_path / "site"
    site.mkdir()
    client = tmp_path / "client"
    # The remote's file:// URL must be absolute whichever way the paths are given.
    arguments = ["site", "client"] if relative else [site, client]

    result = subprocess.run(
        ["bash", _ROOT / "scripts/check_repository_signatures.sh", *arguments],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "LOG": os.fspath(log),
            "FAIL": failing or "",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )

    commands = [
        f"gpg --batch --verify {site}/apt/dists/stable/InRelease",
        f"flatpak {client} remote-add --user --gpg-import={site}/quoth.gpg "
        f"quoth-test file://{site}/flatpak/repo",
        f"flatpak {client} remote-info --user quoth-test dev.quoth.Quoth",
    ]
    ran = {None: 3, "gpg": 1, "remote-add": 2, "remote-info": 3}[failing]
    assert log.read_text().splitlines() == commands[:ran]
    assert (result.returncode == 0) == (failing is None)


def _install_debian_matrix_fakes(bin_dir):
    _command(
        bin_dir,
        "docker",
        'image=""\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  case "$1" in\n'
        '    debian:*|ubuntu:*) image="$1" ;;\n'
        "    bash)\n"
        '      shift; [ "$1" = -c ]; shift\n'
        '      export FAKE_IMAGE="$image"\n'
        '      log="$TMPDIR/inst-${image//[:.]/_}.log"\n'
        '      command="${1//\\/tmp\\/inst.log/$log}"\n'
        '      exec bash -c "$command"\n'
        "      ;;\n"
        "  esac\n"
        "  shift\n"
        "done\n"
        "exit 2\n",
    )
    _command(
        bin_dir,
        "apt-get",
        'echo "$FAKE_IMAGE LC_ALL=${LC_ALL:-} $*" >> "$CALL_LOG"\n'
        'case "$1:$FAKE_IMAGE:$REFUSAL_MODE" in\n'
        "  update:ubuntu\\:22.04:mirror-failure) exit 100 ;;\n"
        "  update:ubuntu\\:22.04:partial-mirror-failure)\n"
        '    case "$*" in *"APT::Update::Error-Mode=any"*) exit 100 ;; esac ;;\n'
        "  install:ubuntu\\:22.04:unrelated-failure)\n"
        '    echo "E: Unable to locate package unrelated" >&2; exit 100 ;;\n'
        "esac\n"
        'case "$1:$FAKE_IMAGE" in\n'
        "  install:ubuntu\\:22.04|install:debian\\:12)\n"
        '    echo "quoth : Depends: gir1.2-glib-2.0 (>= 2.80) but 1.72.0-1 is to be installed" >&2\n'
        '    echo "        Depends: gir1.2-adw-1 (>= 1.5) but it is not going to be installed" >&2\n'
        '    if [ "$FAKE_IMAGE" = "ubuntu:22.04" ]; then\n'
        '      echo "        Depends: python3-urllib3 (>= 1.26.11) but it is not going to be installed" >&2\n'
        "    fi\n"
        '    if [ "$FAKE_IMAGE:$REFUSAL_MODE" = "ubuntu:22.04:mixed-dependency-failure" ]; then\n'
        '      echo "        Depends: unrelated-package but it is not installable" >&2\n'
        "    fi\n"
        '    if [ "$FAKE_IMAGE:$REFUSAL_MODE" = "ubuntu:22.04:undeclared-floor" ]; then\n'
        '      echo "        Depends: libundeclared1 (>= 2.0) but 1.0 is to be installed" >&2\n'
        "    fi\n"
        '    echo "E: Unable to correct problems, you have held broken packages." >&2\n'
        "    exit 100\n"
        "    ;;\n"
        "esac\n",
    )
    _command(
        bin_dir,
        "apt-cache",
        'case "$*:$REFUSAL_MODE" in\n'
        "  *gir1.2-adw-1*:current-adw) candidate=1.6.0 ;;\n"
        "  *python3-urllib3*:current-adw) candidate=1.26.20 ;;\n"
        "  *python3-urllib3*)\n"
        '    [ "$FAKE_IMAGE" = "ubuntu:22.04" ] && candidate=1.26.5 || candidate=1.26.12 ;;\n'
        "  *gir1.2-glib-2.0*)\n"
        '    [ "$FAKE_IMAGE" = "ubuntu:22.04" ] && candidate=1.72.0-1 || candidate=1.74.0-3 ;;\n'
        "  *) candidate=1.2.2 ;;\n"
        "esac\n"
        'printf "  Candidate: %s\\n" "$candidate"\n',
    )
    _command(
        bin_dir,
        "dpkg-deb",
        'echo "python3, gir1.2-glib-2.0 (>= 2.80), gir1.2-adw-1 (>= 1.5), '
        'python3-urllib3 (>= 1.26.11), libadwaita-1-0"\n',
    )
    _command(bin_dir, "python3", 'echo "Python 3.13.0"\n')
    _command(
        bin_dir,
        "xvfb-run",
        'while [ "$#" -gt 0 ]; do\n'
        '  case "$1" in -a) shift ;; -s) shift 2 ;; *) exec "$@" ;; esac\n'
        "done\n",
    )


@pytest.mark.parametrize(
    "failed_image",
    ["", "debian:trixie", "ubuntu:26.04", "ubuntu:22.04", "debian:12"],
)
def test_debian_matrix_propagates_docker_failure(tmp_path, monkeypatch, failed_image):
    repo = tmp_path / "repo"
    script = _copy_harness(repo, "packaging/debian/.dockervalidate/matrix.sh")
    deb = tmp_path / "quoth.deb"
    deb.touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _command(
        bin_dir,
        "docker",
        'for arg in "$@"; do\n'
        '  if [ "$arg" = "$FAILED_IMAGE" ]; then\n'
        '    echo "FAIL: simulated Docker failure"; exit 97\n'
        "  fi\n"
        "done\n"
        'echo "RESULT: simulated success"\n',
    )
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("FAILED_IMAGE", failed_image)

    result = subprocess.run(
        ["bash", str(script), str(deb), f"quoth@0.1.0+g{'0' * 40}", "candidate"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    if failed_image:
        assert result.returncode != 0, result.stdout
        assert "MATRIX_RESULT=FAIL" in result.stdout
        assert "MATRIX_RESULT=PASS" not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert "MATRIX_RESULT=PASS" in result.stdout


@pytest.mark.parametrize(
    ("refusal_mode", "succeeds"),
    [
        pytest.param("adw-refusal", True, id="genuine-adw-refusal"),
        pytest.param("mirror-failure", False, id="apt-update-failure"),
        pytest.param("partial-mirror-failure", False, id="partial-apt-update-failure"),
        pytest.param("unrelated-failure", False, id="unrelated-install-failure"),
        pytest.param("mixed-dependency-failure", False, id="mixed-dependency-failure"),
        pytest.param("undeclared-floor", False, id="undeclared-floor"),
        pytest.param("current-adw", False, id="candidate-meets-floor"),
    ],
)
def test_debian_matrix_verifies_old_adw_refusal(
    tmp_path, monkeypatch, refusal_mode, succeeds
):
    repo = tmp_path / "repo"
    script = _copy_harness(repo, "packaging/debian/.dockervalidate/matrix.sh")
    (repo / "scripts").mkdir()
    deb = tmp_path / "quoth.deb"
    deb.touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install_debian_matrix_fakes(bin_dir)
    call_log = tmp_path / "calls.log"
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("CALL_LOG", str(call_log))
    monkeypatch.setenv("REFUSAL_MODE", refusal_mode)
    monkeypatch.setenv("LC_ALL", "C.UTF-8")

    result = subprocess.run(
        ["bash", str(script), str(deb), f"quoth@0.1.0+g{'0' * 40}", "candidate"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == (0 if succeeds else 1), result.stdout + result.stderr
    assert ("MATRIX_RESULT=PASS" in result.stdout) is succeeds
    calls = call_log.read_text().splitlines()
    updates = [call for call in calls if " update " in f" {call} "]
    assert len(updates) == 4
    assert all(" LC_ALL=C " in call for call in calls)
    assert all("-o APT::Update::Error-Mode=any" in call for call in updates)
    if succeeds:
        assert any(call.startswith("debian:12 LC_ALL=C install ") for call in calls)


@pytest.mark.parametrize(
    "failed_step", ["", "build", "help", "smoke", "bundle", "sources"]
)
def test_flatpak_stops_on_failed_validation(tmp_path, monkeypatch, failed_step):
    repo = tmp_path / "repo"
    script = _copy_harness(repo, "packaging/flatpak/.flatpakvalidate/run.sh")
    flatpak_dir = repo / "packaging/flatpak"
    (flatpak_dir / "dev.quoth.Quoth.yaml").touch()
    (flatpak_dir / "openvino-npu.yaml").touch()
    (flatpak_dir / "python3-deps.yaml").touch()
    (flatpak_dir / "pygobject-transfer-full.patch").touch()
    (repo / "scripts").mkdir()
    (repo / "scripts/check_install.py").touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Stands in for `git -C <repo> archive --format=tar HEAD`.
    _command(bin_dir, "git", 'tar -C "$2" -cf - .\n')
    _command(bin_dir, "python3", 'cat >/dev/null\n: > "$3"\n')
    _command(
        bin_dir,
        "flatpak",
        'case "$*" in\n'
        "  *--command=quoth*) step=help ;;\n"
        "  *--command=python3*) step=smoke ;;\n"
        '  "build-bundle --runtime "*) step=sources ;;\n'
        '  "build-bundle "*) step=bundle ;;\n'
        # The corresponding source is part of the build, not an optional extra.
        "  *--bundle-sources*) step=build ;;\n"
        "  *) step=build-without-sources ;;\n"
        "esac\n"
        'echo "$step" >> "$CALL_LOG"\n'
        'if [ "$step" = "$FAILED_STEP" ]; then exit 97; fi\n'
        'if [ "$step" = bundle ]; then touch "$3"; fi\n'
        'if [ "$step" = sources ]; then touch "$4"; fi\n',
    )
    _command(
        bin_dir,
        "xvfb-run",
        'while [ "$#" -gt 0 ]; do\n'
        '  case "$1" in -a) shift ;; -s) shift 2 ;; *) exec "$@" ;; esac\n'
        "done\n",
    )
    call_log = tmp_path / "calls.log"
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("WORK", str(tmp_path / "work"))
    monkeypatch.setenv("CALL_LOG", str(call_log))
    monkeypatch.setenv("FAILED_STEP", failed_step)
    monkeypatch.setenv("QUOTH_RELEASE", f"quoth@0.1.0+g{'0' * 40}")
    monkeypatch.setenv("QUOTH_ENVIRONMENT", "candidate")

    result = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, timeout=10
    )

    steps = ["build", "help", "smoke", "bundle", "sources"]
    if failed_step:
        assert result.returncode == 97, result.stdout + result.stderr
        assert (
            call_log.read_text().splitlines() == steps[: steps.index(failed_step) + 1]
        )
        assert "OK: Flatpak validated" not in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert call_log.read_text().splitlines() == steps
        assert "OK: Flatpak validated" in result.stdout


@pytest.mark.parametrize("status", [0, 97])
def test_flatpak_lint_checks_only_the_manifest(tmp_path, monkeypatch, status):
    repo = tmp_path / "repo"
    script = _copy_harness(repo, "packaging/flatpak/.flatpakvalidate/lint.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _command(bin_dir, "flatpak", 'echo "$*" >> "$CALL_LOG"\nexit "$STATUS"\n')
    call_log = tmp_path / "calls.log"
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("CALL_LOG", str(call_log))
    monkeypatch.setenv("STATUS", str(status))

    result = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, timeout=10
    )

    assert result.returncode == status, result.stdout + result.stderr
    flatpak = repo / "packaging/flatpak"
    assert call_log.read_text().splitlines() == [
        "run --command=flatpak-builder-lint org.flatpak.Builder --exceptions"
        f" --user-exceptions {flatpak}/.flatpakvalidate/exceptions.json"
        f" manifest {flatpak}/dev.quoth.Quoth.yaml"
    ]


def test_flatpak_exceptions_are_narrow():
    exceptions = _ROOT / "packaging/flatpak/.flatpakvalidate/exceptions.json"

    assert json.loads(exceptions.read_text()) == {
        "dev.quoth.Quoth": [
            "finish-args-flatpak-spawn-access",
            "finish-args-only-wayland",
        ]
    }
