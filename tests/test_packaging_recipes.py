# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for the packaging recipes: each publishes its runtime and build obligations."""

import os
import pty
import re
import select
import shutil
import socket
import subprocess
import termios
import tty
import zipfile
from pathlib import Path

import pytest
import yaml

from tests.support import whisper_source_tree

_ROOT = Path(__file__).resolve().parents[1]
_AUR = _ROOT / "packaging" / "aur"
_RECIPES = [_AUR / "PKGBUILD", _AUR / "quoth-git" / "PKGBUILD"]
_INSTALLS = [_AUR / "quoth.install", _AUR / "quoth-git" / "quoth.install"]
_DEBIAN = _ROOT / "packaging" / "debian"
_FLATPAK = _ROOT / "packaging" / "flatpak" / "dev.quoth.Quoth.yaml"
_PATCHES = _ROOT / "packaging" / "patches"


def _recipe_values(recipe, field):
    result = subprocess.run(
        [
            "bash",
            "-e",
            "-c",
            'source "$1"; declare -n values="$2"; printf "%s\\n" "${values[@]}"',
            "bash",
            str(recipe),
            field,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return result.stdout.splitlines()


def _flatpak_module(name):
    (module,) = (
        module
        for module in yaml.safe_load(_FLATPAK.read_text())["modules"]
        if isinstance(module, dict) and module["name"] == name
    )
    return module


@pytest.mark.parametrize("recipe", _RECIPES, ids=["stable", "vcs"])
def test_prepare_stages_native_sources(tmp_path, recipe):
    whisper_tag = _recipe_values(recipe, "_whispertag")[0].removeprefix("v")
    thorvg_tag = _recipe_values(recipe, "_thorvgtag")[0].removeprefix("v")
    for directory, marker in [
        (f"whisper.cpp-{whisper_tag}", "CMakeLists.txt"),
        (f"thorvg-{thorvg_tag}", "meson.build"),
    ]:
        source = tmp_path / directory
        source.mkdir()
        (source / marker).write_text("native source")
    whisper_source_tree(tmp_path / f"whisper.cpp-{whisper_tag}")
    quoth_source = (
        tmp_path / "quoth"
        if recipe.parent.name == "quoth-git"
        else tmp_path / _recipe_values(recipe, "_srcname")[0]
    )
    shutil.copytree(_PATCHES, quoth_source / "packaging/patches")
    scripts = quoth_source / "scripts"
    scripts.mkdir()
    helper = scripts / "apply_whisper_patches.sh"
    shutil.copyfile(_ROOT / "scripts/apply_whisper_patches.sh", helper)
    helper.chmod(0o755)
    with zipfile.ZipFile(
        tmp_path / _recipe_values(recipe, "_ovwheel")[0], "w"
    ) as wheel:
        wheel.writestr("openvino/cmake/OpenVINOConfig.cmake", "runtime source")

    result = subprocess.run(
        [
            "bash",
            "-e",
            "-c",
            'source "$1"; srcdir="$2"; prepare',
            "bash",
            str(recipe),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "whisper.cpp/CMakeLists.txt").read_text() == "native source"
    for patch in _PATCHES.glob("whisper-*.patch"):
        # Reverses exactly only if prepare() applied it: fuzz lets a pure deletion
        # reverse onto the unpatched source, and --force never prompts.
        subprocess.run(
            [
                "patch",
                "--force",
                "--dry-run",
                "--reverse",
                "--fuzz=0",
                "-p1",
                "-i",
                patch,
            ],
            cwd=tmp_path / "whisper.cpp",
            check=True,
            timeout=10,
        )
    assert (tmp_path / "thorvg/meson.build").read_text() == "native source"
    assert (
        tmp_path / "openvino-wheel/openvino/cmake/OpenVINOConfig.cmake"
    ).read_text() == "runtime source"


@pytest.mark.parametrize("recipe", _RECIPES, ids=["stable", "vcs"])
@pytest.mark.parametrize("field", ["license", "depends", "makedepends", "optdepends"])
def test_srcinfo_matches_recipe_metadata(recipe, field):
    prefix = f"{field} = "
    published = [
        line.strip().removeprefix(prefix)
        for line in recipe.with_name(".SRCINFO").read_text().splitlines()
        if line.strip().startswith(prefix)
    ]

    assert published == _recipe_values(recipe, field)


def _debian_fields(paragraph):
    """The fields of debian/control's source (0) or binary (1) paragraph."""
    control = (_DEBIAN / "control").read_text()
    fields = {}
    current = None
    for line in control.split("\n\n")[paragraph].splitlines():
        if not line or line.startswith("#"):
            continue
        if line[0].isspace():
            fields[current] += " " + line.strip()
            continue
        current, value = line.split(":", 1)
        fields[current] = value.strip()
    return fields


def _relations(fields, name):
    return {relation.strip() for relation in fields[name].replace("\n", " ").split(",")}


def _debian_build_calls():
    """The make calls in debian/rules' build step, one per line."""
    # Under -n, make still runs the $(MAKE) lines, so each call is echoed, not run.
    result = subprocess.run(
        [
            "make",
            "-n",
            "-f",
            str(_DEBIAN / "rules"),
            "execute_after_dh_auto_build",
            "MAKE=echo",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return result.stdout.splitlines()


def test_debian_depends_on_wtype_and_bundles_ydotool():
    fields = _debian_fields(1)

    assert "wtype" in _relations(fields, "Depends")
    assert not any("ydotool" in value for value in fields.values())
    assert (
        f"vendor-ydotool YDOTOOL_SRC={_DEBIAN}/.build-src/ydotool"
        in _debian_build_calls()
    )


def test_debian_bundles_wl_clipboard_built_with_ext_data_control():
    source, fields = _debian_fields(0), _debian_fields(1)

    assert "wl-clipboard" not in _relations(fields, "Depends")
    assert "libwayland-client0" in _relations(fields, "Depends")
    assert "wayland-protocols (>= 1.39)" in _relations(source, "Build-Depends")
    assert (
        f"vendor-wl-clipboard WL_CLIPBOARD_SRC={_DEBIAN}/.build-src/wl-clipboard"
        in _debian_build_calls()
    )


def test_debian_installs_media_backend_and_suggests_gnome_tray():
    fields = _debian_fields(1)

    assert "libgtk-4-media-gstreamer" in _relations(fields, "Depends")
    assert "gnome-shell-extension-appindicator" in _relations(fields, "Suggests")


def test_debian_depends_on_the_cxx_runtime():
    # dh_shlibdeps skips the bundled libraries, which link libstdc++ and libgcc_s.
    fields = _debian_fields(1)

    assert {"libgcc-s1", "libstdc++6"} <= _relations(fields, "Depends")


def test_flatpak_may_run_the_bundled_injection_tools_on_the_host():
    manifest = yaml.safe_load(_FLATPAK.read_text())

    assert "--talk-name=org.freedesktop.Flatpak" in manifest["finish-args"]


def test_flatpak_builds_and_exposes_npu_backend():
    manifest = yaml.safe_load(_FLATPAK.read_text())
    build = " ".join(_flatpak_module("quoth")["build-commands"])

    assert "openvino-npu.yaml" in manifest["modules"]
    assert "WITH_OPENVINO=ON" in build
    assert "WITH_OPENVINO=OFF" not in build
    assert "--device=all" in manifest["finish-args"]
    assert "--device=dri" not in manifest["finish-args"]
    assert "scripts/apply_whisper_patches.sh --apply whisper.cpp" in build


def test_flatpak_takes_openvino_from_the_debs_wheel():
    pins = _ROOT / "packaging/pins.sh"
    npu_text = (_FLATPAK.parent / "openvino-npu.yaml").read_text()
    wheel = yaml.safe_load(npu_text)["sources"][0]
    build = " ".join(_flatpak_module("quoth")["build-commands"])

    assert wheel["url"] == _recipe_values(pins, "QUOTH_OPENVINO_URL")[0]
    assert wheel["sha256"] == _recipe_values(pins, "QUOTH_OPENVINO_SHA256")[0]
    assert "OpenVINO_DIR=/app/opt/openvino/openvino/cmake" in build
    assert "make vendor-openvino OPENVINO_WHEEL_DIR=/app/opt/openvino" in build
    assert "/app/opt/openvino/openvino-*.dist-info/licenses/." in npu_text
    # Intel's toolkit archive puts its runtime under the OpenVINO Distribution License.
    for text in (npu_text, _FLATPAK.read_text()):
        for archive_only in (
            "storage.openvinotoolkit.org",
            "docs/licensing",
            "vm_runtime",
        ):
            assert archive_only not in text


def test_flatpak_installs_bundled_tool_licenses():
    manifest = _FLATPAK.read_text()

    for path in (
        "/app/share/licenses/wl-clipboard/COPYING",
        "/app/share/licenses/libportal/COPYING",
        "/app/share/licenses/wtype/LICENSE",
        "/app/share/licenses/ydotool/LICENSE",
        "/app/share/licenses/pygobject/COPYING",
    ):
        assert path in manifest


def test_flatpak_bundles_the_source_of_numpys_gcc_runtime():
    deps = yaml.safe_load((_FLATPAK.parent / "python3-deps.yaml").read_text())
    (numpy,) = (m for m in deps["modules"] if m["name"] == "python3-numpy")
    (wheel,) = (s for s in numpy["sources"] if s["only-arches"] == ["x86_64"])
    runtime = _flatpak_module("numpy-gcc-runtime-source")

    # The source packages were matched to this wheel's libgfortran and libquadmath by
    # Build ID; a new wheel needs them matched again.
    assert wheel["sha256"] == (
        "a7830bab239b79cda9c08c2da014761cafb48da6150e1da17ac06283f43b6089"
    )
    assert [source["url"].rsplit("/", 1)[1] for source in runtime["sources"]] == [
        "gcc-libraries-8.3.1-2.1.1.el7.src.rpm",
        "gcc-4.8.5-44.el7.src.rpm",
    ]
    for source in runtime["sources"]:
        assert source["type"] == "file"
        assert len(source["sha256"]) == 64
        assert source["only-arches"] == ["x86_64"]
        # One archive host down must not stop every build.
        name = source["url"].rsplit("/", 1)[1]
        assert source["url"].startswith("https://vault.centos.org/")
        assert source["mirror-urls"] == [
            f"https://archive.kernel.org/centos-vault/7.9.2009/os/Source/SPackages/{name}"
        ]


def test_npu_compiler_llvm_notice_matches_the_pinned_compiler():
    notice = (_ROOT / "data/licenses/LICENSE.npu-compiler-llvm").read_text()
    pins = _ROOT / "packaging/pins.sh"
    compiler = _recipe_values(pins, "QUOTH_NPU_COMPILER_LICENSE_URL")[0]
    driver = _recipe_values(pins, "QUOTH_NPU_LICENSE_URL")[0]

    # It was taken for this compiler revision and driver release; a new pin needs a
    # fresh copy from the npu-compiler-llvm revision that compiler builds against.
    assert compiler.split("/")[-2] in notice
    assert f"linux-npu-driver {driver.split('/')[-2]}" in notice
    for term in (
        "Apache License v2.0 with LLVM Exceptions",
        "University of Illinois/NCSA",
        "Henry Spencer",
        "Yann Collet",
        "Unicode, Inc.",
    ):
        assert term in notice
    installed = "data/licenses/LICENSE.npu-compiler-llvm"
    assert (
        f"install -Dm644 {installed} /app/share/licenses/intel-npu-compiler/LICENSE.llvm"
        in _flatpak_module("quoth")["build-commands"]
    )
    vendor = (_ROOT / "scripts/vendor_npu_driver.sh").read_text().replace("\\\n", " ")
    assert (
        f'"$REPO/{installed}" "$DEST/licenses/intel-npu-compiler/LICENSE.llvm"'
        in " ".join(vendor.split())
    )


def test_licence_records_name_the_npu_compilers_llvm_terms():
    listing = (_ROOT / "THIRD-PARTY-LICENSES.md").read_text().splitlines()
    row = next(line for line in listing if "[Intel NPU compiler]" in line)
    copyright_ = " ".join((_DEBIAN / "copyright").read_text().split())
    clause = copyright_.split("its compiler,")[1].split(";")[0]

    # One name per set of terms in data/licenses/LICENSE.npu-compiler-llvm.
    for name in (
        "Apache-2.0 WITH LLVM-exception",
        "NCSA",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "Spencer",
        "Unicode",
    ):
        assert name in row
        assert name in clause


def _scanner_build(build):
    """A build tree with wayland-scanner output, with and without a protocol notice.

    xdg-shell is generated for two targets, and wlr-data-control's notice starts on
    the line that opens its comment, as when its <copyright> text starts on the tag's.
    """
    xdg_shell = (
        "/* Generated by wayland-scanner 1.22.0 */\n\n/*\n"
        " * Copyright © 2008-2013 Kristian Høgsberg\n *\n"
        " * Permission is hereby granted.\n */\n\n#include <stdlib.h>\n"
    )
    for target in ("src/protocol", "preload/preload.p"):
        (build / target).mkdir(parents=True)
        (build / target / "xdg-shell.c").write_text(xdg_shell)
    (build / "src/protocol/wlr-data-control.c").write_text(
        "/* Generated by wayland-scanner 1.22.0 */\n\n"
        "/* Copyright © 2018 Simon Ser\n *\n"
        " * Permission to use, copy, modify, distribute, and sell.\n */\n"
    )
    (build / "src/protocol/gtk-shell.c").write_text(
        "/* Generated by wayland-scanner 1.22.0 */\n\n#include <stdlib.h>\n"
    )
    (build / "src/main.c").write_text("/*\n * Not generated.\n */\n")
    return build


def test_protocol_notices_prints_each_compiled_in_protocol(tmp_path):
    script = _ROOT / "scripts/protocol_notices.sh"
    build = _scanner_build(tmp_path / "build")

    found = subprocess.run(
        ["bash", script, build], capture_output=True, text=True, timeout=10
    )
    (tmp_path / "plain").mkdir()
    none = subprocess.run(
        ["bash", script, tmp_path / "plain"], capture_output=True, text=True, timeout=10
    )

    assert found.returncode == 0, found.stderr
    assert found.stdout == (
        "## xdg-shell\n\nCopyright © 2008-2013 Kristian Høgsberg\n\n"
        "Permission is hereby granted.\n\n"
        "## wlr-data-control\n\nCopyright © 2018 Simon Ser\n\n"
        "Permission to use, copy, modify, distribute, and sell.\n\n"
    )
    assert none.returncode == 1
    assert "no wayland-scanner code" in none.stderr


def test_flatpak_tool_notices_match_the_protocol_notices_helper(tmp_path):
    build = _scanner_build(tmp_path / "build")
    expected = subprocess.run(
        ["bash", _ROOT / "scripts/protocol_notices.sh", build],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout
    for name in ("wl-clipboard", "wtype"):
        licenses = tmp_path / "licenses"
        (licenses / name).mkdir(parents=True)
        for command in _flatpak_module(name)["post-install"]:
            if "PROTOCOL-NOTICES" in command:
                subprocess.run(
                    ["sh", "-c", command.replace("/app/share/licenses", str(licenses))],
                    env={"PATH": "/usr/bin:/bin", "FLATPAK_BUILDER_BUILDDIR": build},
                    check=True,
                    timeout=10,
                )
        assert (licenses / name / "PROTOCOL-NOTICES").read_text() == expected


def test_lucide_icons_are_namespaced_and_attributed():
    icon_dir = _ROOT / "data/icons/hicolor/symbolic/apps"
    for name in ("coffee", "github", "gauge", "bug", "heart"):
        icon = (icon_dir / f"dev.quoth.Quoth-{name}-symbolic.svg").read_text()
        assert 'stroke="#bebebe"' in icon
        assert "Copyright (c) 2026 Lucide Icons and Contributors" in icon

    notice = (_ROOT / "data/licenses/LICENSE.lucide-icons").read_text()
    assert "ISC License" in notice
    assert "The MIT License (MIT)" in notice


def test_the_udev_rule_grants_only_active_seat_uinput_access():
    text = (_ROOT / "data/60-quoth-uinput.rules").read_text()
    rules = [line for line in text.splitlines() if not line.startswith("#")]

    assert rules == [
        'KERNEL=="uinput", SUBSYSTEM=="misc", TAG+="uaccess", '
        'OPTIONS+="static_node=uinput"'
    ]
    # Its own comment names the one override, the same on every channel.
    assert "/etc/udev/rules.d/60-quoth-uinput.rules" in text


def test_the_accel_rule_tags_npu_nodes_as_systemd_does():
    text = (_ROOT / "data/60-quoth-accel.rules").read_text()
    rules = [line for line in text.splitlines() if not line.startswith("#")]

    assert rules == ['SUBSYSTEM=="accel", KERNEL=="accel*", TAG+="uaccess"']


@pytest.mark.parametrize("install", _INSTALLS, ids=["stable", "vcs"])
def test_arch_install_names_the_shared_rule(install):
    assert "/etc/udev/rules.d/60-quoth-uinput.rules" in install.read_text()


def test_arch_install_hooks_are_identical():
    # Each AUR package base needs its own copy of the one install hook.
    assert _INSTALLS[1].read_bytes() == _INSTALLS[0].read_bytes()


# What a first install runs to apply the udev rules without a reboot.
_APPLY_UDEV_RULES = [
    "udevadm control --reload-rules",
    "modprobe uinput",
    "udevadm trigger --sysname-match=uinput",
    "udevadm trigger --subsystem-match=accel",
]


def _stub_tools(tmp_path, version="0.1.0", charmap="UTF-8"):
    """Stub udevadm and modprobe (failing, logged), dpkg-query and locale."""
    log = tmp_path / "calls"
    for tool in ("udevadm", "modprobe"):
        stub = tmp_path / tool
        stub.write_text(f'#!/bin/sh\necho "{tool} $*" >> "{log}"\nexit 1\n')
        stub.chmod(0o755)
    # dpkg-query answers only the postinst's own query; None reads as no answer.
    query = "-W -f=${Version} quoth"
    answer = f"printf %s '{version}'" if version is not None else "exit 1"
    stubs = {
        "dpkg-query": f"[ \"$*\" = '{query}' ] || exit 2\n{answer}\n",
        "locale": f'[ "$*" = charmap ] || exit 2\necho "{charmap}"\n',
    }
    for tool, body in stubs.items():
        stub = tmp_path / tool
        stub.write_text(f"#!/bin/sh\n{body}")
        stub.chmod(0o755)
    return log


def _hook_env(tmp_path, env):
    return {"PATH": f"{tmp_path}:/usr/bin:/bin", **(env or {})}


def _run_install_hook(
    tmp_path, argv, stdout=subprocess.PIPE, env=None, version="0.1.0", charmap="UTF-8"
):
    """Run an install hook with its tools stubbed; return output, calls."""
    log = _stub_tools(tmp_path, version, charmap)
    result = subprocess.run(
        argv,
        stdout=stdout,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
        timeout=10,
        env=_hook_env(tmp_path, env),
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return result.stdout, calls


def _run_on_terminal(
    tmp_path, argv, cols=80, env=None, version="0.1.0", charmap="UTF-8", gone=False
):
    """Run an install hook on a pseudo-terminal `cols` wide; return the raw bytes.

    An `env` value of None unsets that variable. `gone` closes the terminal's other
    end first, as a closed window does, and stubs stty to still report 24 by 80.
    """
    _stub_tools(tmp_path, version, charmap)
    if gone:
        stty = tmp_path / "stty"
        stty.write_text("#!/bin/sh\necho 24 80\n")
        stty.chmod(0o755)
    env = {"TERM": "xterm-256color", **(env or {})}
    env = {name: value for name, value in env.items() if value is not None}
    primary, secondary = pty.openpty()
    tty.setraw(secondary)
    termios.tcsetwinsize(secondary, (24, cols))
    if gone:
        os.close(primary)
    try:
        hook = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=secondary,
            stderr=subprocess.DEVNULL,
            env=_hook_env(tmp_path, env),
        )
    finally:
        os.close(secondary)
    output = b""
    while not gone and select.select([primary], [], [], 10)[0]:
        try:
            chunk = os.read(primary, 65536)
        except OSError:  # EIO once the hook has closed the terminal
            break
        if not chunk:
            break
        output += chunk
    if not gone:
        os.close(primary)
    assert hook.wait(timeout=10) == 0
    return output


def _run_on_socket(tmp_path, argv):
    """Run an install hook with stdout and stderr on a socket, as libalpm runs one.

    Everything else asks for the art: a UTF-8 charmap and a terminal's TERM.
    """
    _stub_tools(tmp_path)
    ours, theirs = socket.socketpair()
    with ours:
        with theirs:
            hook = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=theirs,
                stderr=theirs,
                env=_hook_env(tmp_path, {"TERM": "xterm-256color"}),
            )
        output = b""
        while chunk := ours.recv(65536):
            output += chunk
    assert hook.wait(timeout=10) == 0
    return output


def _assert_reboot_only_after_masking(output):
    # The rules apply at once; only masking needs a reboot, to drop the granted access.
    mask = (
        "To turn it off, run this, then reboot: "
        "sudo ln -sf /dev/null /etc/udev/rules.d/60-quoth-uinput.rules"
    )
    assert mask in output
    assert output.count("reboot") == 1


# The reference art: the wordmark in quarter blocks, then its mirror image.
_ART = [
    "     ▄▟██████▄▖                                  ▄▄▄       ",
    "   ▄██▛▀▀▀▀▀▀██▙▖                          ██▙   ███       ",
    "  ▐██▘        ▜██▖ ▄▄▄    ▗▄▄▖  ▗▄▄▄▄▄▄  ▗▄███▄▄ ███▗▄▄▄▄▄ ",
    "  ██▌  ▗▄▄▄    ██▌ ███    ▐██▌ ▟██▀▀▀▜██▄▝▀███▀▀ ███▛▀▀▀██▙",
    "  ██▙   ▝██▙   ██▌ ███    ▐██▌▐██▌    ▜██▖ ███   ███    ███",
    "  ▝██▄   ███▖ ▟█▛  ███    ▐██▌▐██▖    ▐██▌ ███   ███    ▐██",
    "   ▝▜██▙▄▝███▙▞▀   ▜██▄  ▗███▌▝██▙▖  ▗██▛  ███   ███    ▐██",
    "     ▝▀▜██▄▝▀▀▜█▙▖  ▀████▛▐██▘  ▀█████▛▀   ▀████ ██▛    ▐██",
    "    ▄▄▟█▛▜▙▄████▘  ▗██▛▀▀▜▟██▌ ▄▟█▀▀▀▜█▙▄  ▟██▀▀ ███    ▐██",
    "  ⣴⣿⡿⠉  ⢀⣿⣿⡿⠁ ⠙⣿⣷⡄ ⣿⣿⣿    ⢸⣿⣿⡇⢸⣿⣿⡅    ⣸⣿⣿⠇ ⣿⣿⣿   ⣿⣿⣿    ⣼⣿⣿",
    "  ⢑⢕⢅   ⠁⠁⠁   ⢀⢕⢕⠅ ⠕⠕⠕    ⠐⠕⠕⠅ ⠑⠑⠕⠔⠔⠔⠔⠕⠑⠁⠐⠔⢕⢕⢕⠔⠔ ⢕⢕⢕⠑⠔⠔⠔⠕⠕⠁",
    "   ⠁⠁⠁⠁      ⠁⠁⠁⠁                          ⠁⠁⠁   ⠁⠁⠁       ",
]
_BOLD, _FAINT, _RESET = "\x1b[1m", "\x1b[2m", "\x1b[0m"
_MAINTAINER = "Samuel Ruairí Bullard"
# The lines under the name line, as runs of text and whether each is bold or faint.
_TAIL = [
    [
        (
            "  Offline voice dictation. Press a key, speak, the text lands at your "
            "cursor.",
            _FAINT,
        )
    ],
    [],
    [
        ("  Start     ", _FAINT),
        ("open Quoth from your app launcher, or run ", ""),
        ("quoth", _BOLD),
    ],
    [("  Website   ", _FAINT), ("quoth.dev", "")],
    [("  Source    ", _FAINT), ("github.com/abullard1/Quoth", "")],
    [("  Made by   ", _FAINT), (_MAINTAINER, "")],
    [],
    [
        (
            "  On a computer shared with people you do not trust: Quoth's udev rule "
            "lets",
            _FAINT,
        )
    ],
    [
        (
            "  whoever is signed in at this computer's screen send keystrokes, even into a",
            _FAINT,
        )
    ],
    [("  later session. To turn it off, run this, then reboot:", _FAINT)],
    [("  sudo ln -sf /dev/null /etc/udev/rules.d/60-quoth-uinput.rules", _FAINT)],
    # Both forms end with a blank line, apart from what follows.
    [],
]


def _line(runs, styled):
    return "".join(
        f"{style}{text}{_RESET}" if styled and style else text for text, style in runs
    )


def _rich(version):
    """The message as a terminal that draws the art receives it, byte for byte."""
    name = [("  Quoth", _BOLD)]
    if version:
        name.append((f" {version}", _FAINT))
    name += [("  ·  ", _FAINT), ("Local. Private. Yours.", "")]
    art = [f"{_BOLD if i < 8 else _FAINT}{line}{_RESET}" for i, line in enumerate(_ART)]
    tail = [_line(runs, styled=True) for runs in _TAIL]
    lines = ["", *art, "", _line(name, styled=True), *tail]
    return "\n".join(lines).encode() + b"\n"


def _plain(version):
    """The message anywhere else: the same lines without the art or any styling."""
    name = f"  Quoth{f' {version}' if version else ''}  -  Local. Private. Yours."
    tail = [_line(runs, styled=False) for runs in _TAIL]
    return "\n".join(["", name, *tail]) + "\n"


_UPDATED = (
    "==> Quoth installed. If it is running, quit and reopen it to use this version.\n"
)

# Each hook's first-install and upgrade paths, called as dpkg and pacman call them.
_HOOK_PATHS = {
    "debian-install": ["sh", str(_DEBIAN / "quoth.postinst"), "configure"],
    "debian-upgrade": ["sh", str(_DEBIAN / "quoth.postinst"), "configure", "0.1.0"],
    **{
        f"arch-{base}-{path}": ["bash", "-c", f'. "$0" && {call}', str(install)]
        for base, install in zip(["stable", "vcs"], _INSTALLS, strict=True)
        for path, call in [
            ("install", "post_install 0.1.0-1"),
            ("upgrade", "post_upgrade 0.2.0-1 0.1.0-1"),
        ]
    },
}
_FIRST_INSTALLS = {key: argv for key, argv in _HOOK_PATHS.items() if "install" in key}
# Only dpkg runs a hook on a terminal; pacman gives a scriptlet a socket.
_DEBIAN_INSTALL = _HOOK_PATHS["debian-install"]
_DEBIAN_PATHS = {key: argv for key, argv in _HOOK_PATHS.items() if "debian" in key}


@pytest.mark.parametrize("install", _INSTALLS, ids=["stable", "vcs"])
def test_arch_install_applies_the_rules_and_survives_without_udev(tmp_path, install):
    # The stubs fail, as udevadm does in a container; the install goes on regardless.
    stdout, calls = _run_install_hook(
        tmp_path, ["bash", "-c", f'. "{install}" && post_install 0.1.0-1']
    )

    assert calls == _APPLY_UDEV_RULES
    assert stdout == _plain("0.1.0")
    output = " ".join(stdout.split())
    _assert_reboot_only_after_masking(output)
    assert "systemctl" not in output


def test_debian_postinst_applies_the_rules_and_needs_no_service(tmp_path):
    postinst = _DEBIAN / "quoth.postinst"
    stdout, calls = _run_install_hook(tmp_path, ["sh", str(postinst), "configure"])

    assert calls == _APPLY_UDEV_RULES
    assert stdout == _plain("0.1.0")
    output = " ".join(stdout.split())
    _assert_reboot_only_after_masking(output)
    assert "systemctl" not in output


def test_debian_postinst_prints_one_line_on_upgrade(tmp_path):
    # Nothing restarts a running Quoth, so an upgrade says to quit and reopen it.
    stdout, calls = _run_install_hook(
        tmp_path, ["sh", str(_DEBIAN / "quoth.postinst"), "configure", "0.1.0"]
    )

    assert stdout == _UPDATED
    assert calls == []


@pytest.mark.parametrize("install", _INSTALLS, ids=["stable", "vcs"])
def test_arch_install_prints_one_line_on_upgrade(tmp_path, install):
    stdout, calls = _run_install_hook(
        tmp_path, ["bash", "-c", f'. "{install}" && post_upgrade 0.2.0-1 0.1.0-1']
    )

    assert stdout == _UPDATED
    assert calls == []


def test_debian_first_install_draws_the_art_on_a_wide_utf8_terminal(tmp_path):
    assert _run_on_terminal(tmp_path, _DEBIAN_INSTALL) == _rich("0.1.0")


def test_the_art_needs_only_its_own_width(tmp_path):
    # The widest art line is the limit; the text lines below may wrap as they would
    # in the plain form.
    width = max(len(line) for line in _ART)

    assert _run_on_terminal(tmp_path, _DEBIAN_INSTALL, cols=width) == _rich("0.1.0")


@pytest.mark.parametrize("install", _INSTALLS, ids=["stable", "vcs"])
def test_arch_prints_the_plain_welcome_on_pacmans_socket(tmp_path, install):
    argv = ["bash", "-c", f'. "{install}" && post_install 0.1.0-1']

    assert _run_on_socket(tmp_path, argv) == _plain("0.1.0").encode()


# Each condition alone that turns a terminal's message plain.
_PLAIN_GATES = {
    "term-dumb": {"env": {"TERM": "dumb"}},
    # The text console has no glyphs for most of the art.
    "term-linux": {"env": {"TERM": "linux"}},
    "term-unset": {"env": {"TERM": None}},
    "term-empty": {"env": {"TERM": ""}},
    "ascii-charmap": {"charmap": "ANSI_X3.4-1968"},
    "too-narrow": {"cols": max(len(line) for line in _ART) - 1},
    # apt runs dpkg on a terminal of its own sized 0 by 0 when its output is piped.
    "unsized": {"cols": 0},
}


@pytest.mark.parametrize("gate", _PLAIN_GATES.values(), ids=_PLAIN_GATES.keys())
def test_each_gate_alone_turns_the_message_plain(tmp_path, gate):
    output = _run_on_terminal(tmp_path, _DEBIAN_INSTALL, **gate)

    assert output == _plain("0.1.0").encode()


@pytest.mark.parametrize("argv", _FIRST_INSTALLS.values(), ids=_FIRST_INSTALLS.keys())
def test_a_pipe_gets_the_plain_message(tmp_path, argv):
    stdout, _ = _run_install_hook(tmp_path, argv, env={"TERM": "xterm-256color"})

    assert stdout == _plain("0.1.0")


@pytest.mark.parametrize("value", ["", "1"])
def test_no_color_still_draws_the_art(tmp_path, value):
    # The welcome adds no colour, only bold and faint, which NO_COLOR does not cover.
    output = _run_on_terminal(tmp_path, _DEBIAN_INSTALL, env={"NO_COLOR": value})

    assert output == _rich("0.1.0")


@pytest.mark.parametrize("argv", _FIRST_INSTALLS.values(), ids=_FIRST_INSTALLS.keys())
def test_the_plain_message_is_ascii_but_for_the_maintainers_name(tmp_path, argv):
    stdout, _ = _run_install_hook(tmp_path, argv)

    assert "\x1b" not in stdout
    assert stdout.replace(_MAINTAINER, "").isascii()
    assert "  Quoth 0.1.0  -  Local. Private. Yours.\n" in stdout


@pytest.mark.parametrize(
    ("recorded", "shown"),
    [
        ("0.1.0", "0.1.0"),
        ("0.1.0-1", "0.1.0"),
        ("1:0.2.0~rc1-2", "0.2.0~rc1"),
        (None, ""),
    ],
)
def test_debian_shows_the_upstream_version_dpkg_records(tmp_path, recorded, shown):
    argv = _DEBIAN_INSTALL

    stdout, _ = _run_install_hook(tmp_path, argv, version=recorded)
    assert stdout == _plain(shown)
    assert _run_on_terminal(tmp_path, argv, version=recorded) == _rich(shown)


@pytest.mark.parametrize(
    ("new", "shown"),
    [
        ("0.1.0-1", "0.1.0"),
        ("2:0.1.0-3", "0.1.0"),
        ("0.1.0.r12.gabc1234-1", "0.1.0.r12.gabc1234"),
        ("", ""),
    ],
)
@pytest.mark.parametrize("install", _INSTALLS, ids=["stable", "vcs"])
def test_arch_shows_the_pkgver_pacman_passes(tmp_path, install, new, shown):
    argv = ["bash", "-c", f'. "$0" && post_install {new}', str(install)]

    stdout, _ = _run_install_hook(tmp_path, argv)

    assert stdout == _plain(shown)


def test_upgrade_prints_its_one_line_even_on_a_terminal(tmp_path):
    output = _run_on_terminal(tmp_path, _HOOK_PATHS["debian-upgrade"])

    assert output == _UPDATED.encode()


@pytest.mark.parametrize("argv", _HOOK_PATHS.values(), ids=_HOOK_PATHS.keys())
def test_install_hooks_succeed_when_stdout_is_closed(tmp_path, argv):
    # dpkg fails the configuration and pacman reports an error when a print fails.
    _run_install_hook(tmp_path, ["sh", "-c", '"$@" >&-', "sh", *argv])


@pytest.mark.parametrize("argv", _HOOK_PATHS.values(), ids=_HOOK_PATHS.keys())
def test_install_hooks_survive_a_reader_that_has_gone(tmp_path, argv):
    # A reader that has closed, as `apt-get install | head` leaves, makes a write raise
    # SIGPIPE, which kills a shell printing with a builtin.
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        _run_install_hook(tmp_path, argv, stdout=write_end)
    finally:
        os.close(write_end)


@pytest.mark.parametrize("argv", _DEBIAN_PATHS.values(), ids=_DEBIAN_PATHS.keys())
def test_debian_hook_survives_a_terminal_that_has_gone(tmp_path, argv):
    # A closed terminal fails each write with EIO. stty still reports a width, as it
    # does when the window closes mid-install, so the first install takes the art path.
    _run_on_terminal(tmp_path, argv, gone=True)


def _here_documents(script):
    return re.findall(r"<<EOF( \|\| true)?\n(.*?)\nEOF\n", script.read_text(), re.S)


def test_install_message_is_shared_and_its_art_is_the_reference(tmp_path):
    # Every native channel prints the same plain message; only the .deb draws art.
    scripts = [_DEBIAN / "quoth.postinst", *_INSTALLS]
    plain = {_run_install_hook(tmp_path, argv)[0] for argv in _FIRST_INSTALLS.values()}
    assert plain == {_plain("0.1.0")}

    art = _here_documents(scripts[0])[0][1].split("\n")[1 : len(_ART) + 1]
    assert art == [
        f"${{{'b' if i < 8 else 'f'}}}{line}${{r}}" for i, line in enumerate(_ART)
    ]
    # Non-ASCII text lives only in the here-documents.
    for script in scripts:
        text = script.read_text()
        for _, document in _here_documents(script):
            text = text.replace(document, "")
        assert text.isascii()
