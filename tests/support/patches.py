# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Fake whisper.cpp build inputs for the packaging tests."""

import json
from pathlib import Path

_PATCHES = Path(__file__).resolve().parents[2] / "packaging" / "patches"


def preimage(patch: Path) -> str:
    """The lines a single-file patch expects, so a fake source tree accepts it."""
    hunks = patch.read_text().split("\n@@", 1)[1]
    return "".join(
        line[1:]
        for line in hunks.splitlines(keepends=True)[1:]
        if line[:1] in (" ", "-")
    )


def whisper_source_tree(root: Path) -> None:
    """A whisper.cpp tree holding just the code the packaging patches change."""
    sources: dict[Path, str] = {}
    for patch in sorted(_PATCHES.glob("whisper-*.patch")):
        target = root / patch.read_text().split("\n+++ b/", 1)[1].split("\n", 1)[0]
        # Several patches may edit one file; its fake source holds every preimage.
        sources[target] = sources.get(target, "") + preimage(patch)
    for target, text in sources.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)


def compile_commands(path: Path) -> list[dict[str, object]]:
    """Write ISA-safe compile commands for every target the installed check requires."""
    # Loaded on use, so a broken script fails only the tests that build these.
    from scripts import check_install

    targets = {
        "ggml",
        "ggml-base",
        "whisper",
        "ggml-cpu-x64",
    } | check_install._SCORER_TARGETS
    rows = [
        {
            "file": "/src/source.cpp",
            "arguments": [
                "c++",
                "-march=x86-64",
                "-mtune=generic",
                "-msse",
                "-msse2",
                "-mindirect-branch=thunk",
                *(["-fno-lto"] if target.endswith("-feats") else []),
                "-o",
                f"CMakeFiles/{target}.dir/source.cpp.o",
                "-c",
                "/src/source.cpp",
            ],
        }
        for target in sorted(targets)
    ]
    rows.append(
        {
            "file": "/src/kernel.cpp",
            "command": (
                "c++ -msse4.2 -mavx -mavx2 -mfma -mf16c -mbmi2 "
                "-o CMakeFiles/ggml-cpu-haswell.dir/kernel.cpp.o "
                "-c /src/kernel.cpp"
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return rows
