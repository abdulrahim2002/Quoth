#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Write a CycloneDX bill of materials for the components a package bundles.

    scripts/package_sbom.py {deb,flatpak} --commit SHA --output FILE

Both packages bundle the components `packaging/pins.sh` names; the Flatpak also
bundles the modules its manifest alone pins. The output depends only on those files
and the commit, so the same commit always gives the same bytes.
"""

import argparse
import json
import re
import tomllib
import uuid
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import unquote

_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _ROOT / "packaging/flatpak/dev.quoth.Quoth.yaml"

# Pins that name no bundled component: build-time headers, licence and notice texts,
# the AUR's ThorVG tarball (the packages build its tagged commit), wheel file names and
# the NPU driver's inner .deb files, which the driver tarball's hash already pins.
UNLISTED_PINS = frozenset(
    {
        "QUOTH_NPU_UMD_SHA256",
        "QUOTH_NPU_COMPILER_SHA256",
        "QUOTH_SPIRV_HEADERS_TAG",
        "QUOTH_SPIRV_HEADERS_COMMIT",
        "QUOTH_NPU_LICENSE_URL",
        "QUOTH_NPU_LICENSE_SHA256",
        "QUOTH_NPU_NOTICES_URL",
        "QUOTH_NPU_NOTICES_SHA256",
        "QUOTH_NPU_COMPILER_LICENSE_URL",
        "QUOTH_NPU_COMPILER_LICENSE_SHA256",
        "QUOTH_LEVEL_ZERO_LICENSE_URL",
        "QUOTH_LEVEL_ZERO_LICENSE_SHA256",
        "QUOTH_THORVG_SHA256",
        "QUOTH_SENTRY_SDK_WHEEL",
        "QUOTH_OPENVINO_WHEEL",
    }
)

# The repositories the packages clone at the pinned commits; pins.sh names no URL.
GIT_SOURCES = {
    "gtk4-layer-shell": "https://github.com/wmww/gtk4-layer-shell.git",
    "thorvg": "https://github.com/thorvg/thorvg.git",
    "wl-clipboard": "https://github.com/bugaevc/wl-clipboard.git",
    "ydotool": "https://github.com/ReimuNotMoe/ydotool.git",
}

# Manifest modules the Flatpak bundles whose pins live only in the manifest.
FLATPAK_ONLY = (
    "wtype",
    "pygobject",
    "libportal",
    "python3-loguru",
    "python3-numpy",
    "python3-certifi",
    "python3-urllib3",
)

_TOOLS = {"wl-clipboard", "ydotool", "wtype"}


def read_pins(path: Path) -> dict[str, str]:
    """Return the `QUOTH_*="value"` assignments of a pins file."""
    return dict(re.findall(r'^(QUOTH_\w+)="([^"]*)"', path.read_text(), re.MULTILINE))


def _component(
    name: str,
    version: str,
    url: str,
    *,
    sha256: str | None = None,
    commit: str | None = None,
    purl: str | None = None,
) -> dict:
    """Return one component, from `url` (whose content is `sha256`) or a git commit."""
    component = {
        "type": "application" if name in _TOOLS else "library",
        "name": name,
        "version": version.removeprefix("v"),
    }
    if commit:
        owner, repo = re.fullmatch(r"https://github\.com/(.+)/(.+)\.git", url).groups()
        # The purl github type takes owner and repository in lower case.
        component["purl"] = f"pkg:github/{owner.lower()}/{repo.lower()}@{commit}"
    elif purl:
        component["purl"] = purl
    reference = {"type": "vcs" if commit else "distribution", "url": url}
    if sha256:
        reference["hashes"] = [{"alg": "SHA-256", "content": sha256}]
    component["externalReferences"] = [reference]
    return component


def _pypi(name: str, version: str) -> str:
    """Return the purl of a PyPI release, its name normalised as purl requires."""
    return f"pkg:pypi/{re.sub(r'[-_.]+', '-', name).lower()}@{version}"


def whisper_patches() -> list[str]:
    """Return the patches both recipes apply to whisper.cpp, by file name."""
    return sorted(path.name for path in (_ROOT / "packaging/patches").glob("*.patch"))


def pinned_components(pins: Mapping[str, str]) -> list[dict]:
    """Return the components both packages bundle from `packaging/pins.sh`."""

    def deb_version(filename: str) -> str:
        return unquote(filename).split("_")[1]

    whisper = pins["QUOTH_WHISPER_TAG"]
    patched = _component(
        "whisper.cpp",
        whisper,
        f"https://github.com/ggml-org/whisper.cpp/archive/refs/tags/{whisper}.tar.gz",
        sha256=pins["QUOTH_WHISPER_SHA256"],
    )
    patched["pedigree"] = {
        "notes": "Built with these patches from packaging/patches/ in Quoth's source "
        "applied: " + ", ".join(whisper_patches()) + "."
    }
    components = [
        patched,
        _component(
            "openvino",
            pins["QUOTH_OPENVINO_VERSION"],
            pins["QUOTH_OPENVINO_URL"],
            sha256=pins["QUOTH_OPENVINO_SHA256"],
            purl=_pypi("openvino", pins["QUOTH_OPENVINO_VERSION"]),
        ),
    ]
    for name, deb in (
        ("intel-level-zero-npu", "QUOTH_NPU_UMD_DEB"),
        ("intel-driver-compiler-npu", "QUOTH_NPU_COMPILER_DEB"),
    ):
        components.append(
            _component(
                name,
                deb_version(pins[deb]),
                pins["QUOTH_NPU_DRIVER_URL"],
                sha256=pins["QUOTH_NPU_DRIVER_SHA256"],
            )
        )
    level_zero = pins["QUOTH_LEVEL_ZERO_URL"]
    components.append(
        _component(
            "level-zero",
            deb_version(level_zero.rsplit("/", 1)[1]),
            level_zero,
            sha256=pins["QUOTH_LEVEL_ZERO_SHA256"],
        )
    )
    for name, prefix in (
        ("gtk4-layer-shell", "QUOTH_LAYER_SHELL"),
        ("thorvg", "QUOTH_THORVG"),
        ("wl-clipboard", "QUOTH_WL_CLIPBOARD"),
        ("ydotool", "QUOTH_YDOTOOL"),
    ):
        components.append(
            _component(
                name,
                pins[f"{prefix}_TAG"],
                GIT_SOURCES[name],
                commit=pins[f"{prefix}_COMMIT"],
            )
        )
    sentry = pins["QUOTH_SENTRY_SDK_VERSION"]
    components.append(
        _component(
            "sentry-sdk",
            sentry,
            pins["QUOTH_SENTRY_SDK_URL"],
            sha256=pins["QUOTH_SENTRY_SDK_SHA256"],
            purl=_pypi("sentry-sdk", sentry),
        )
    )
    return components


def manifest_modules(path: Path) -> dict[str, dict]:
    """Return every module of a Flatpak manifest by name, resolving included files."""
    import yaml  # the Flatpak build installs PyYAML; the .deb's SBOM needs none

    modules = {}

    def walk(entries: list, directory: Path) -> None:
        for entry in entries:
            here = directory
            if isinstance(entry, str):
                here = (directory / entry).parent
                entry = yaml.safe_load((directory / entry).read_text())
            modules[entry["name"]] = entry
            walk(entry.get("modules", []), here)

    walk(yaml.safe_load(path.read_text())["modules"], path.parent)
    return modules


def module_component(module: dict) -> dict:
    """Return what a Flatpak-only module bundles, from its one x86_64 source."""
    (source,) = (
        s
        for s in module["sources"]
        if s["type"] in {"git", "archive", "file"}
        and "x86_64" in s.get("only-arches", ["x86_64"])
    )
    if source["type"] == "git":
        return _component(
            module["name"], source["tag"], source["url"], commit=source["commit"]
        )
    filename = source["url"].rsplit("/", 1)[1]
    if filename.endswith(".whl"):
        name, version = filename.split("-")[:2]
        return _component(
            name,
            version,
            source["url"],
            sha256=source["sha256"],
            purl=_pypi(name, version),
        )
    version = re.fullmatch(
        rf"{re.escape(module['name'])}-(.+)\.tar\.\w+", filename
    ).group(1)
    return _component(module["name"], version, source["url"], sha256=source["sha256"])


def sbom(channel: str, commit: str) -> dict:
    """Return the CycloneDX document for one package of the commit."""
    components = pinned_components(read_pins(_ROOT / "packaging/pins.sh"))
    if channel == "flatpak":
        modules = manifest_modules(_MANIFEST)
        components += [module_component(modules[name]) for name in FLATPAK_ONLY]
    version = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"][
        "version"
    ]
    # CycloneDX names each document by a UUID; this one is the commit's and channel's.
    serial = uuid.uuid5(uuid.NAMESPACE_URL, f"quoth-{channel}-sbom:{commit}")
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "component": {"type": "application", "name": "quoth", "version": version}
        },
        "components": sorted(components, key=lambda c: c["name"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("channel", choices=("deb", "flatpak"))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("--commit takes a full commit hash")
    document = sbom(args.channel, args.commit)
    args.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
