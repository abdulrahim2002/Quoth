# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Render an auditable uv project from the Flatpak wheel manifests."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote, urlsplit

import yaml
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import Version


class ManifestError(ValueError):
    """Flatpak manifests that do not describe one exact, hashed wheel graph."""


class WheelSource(NamedTuple):
    """One wheel file a manifest pins, with its URL, hash and architectures."""

    manifest: str
    filename: str
    url: str
    sha256: str
    only_arches: tuple[str, ...]


class WheelPin(NamedTuple):
    """One package at its single exact version, with every wheel that provides it."""

    name: str
    version: Version
    sources: tuple[WheelSource, ...]


_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")

# The Python of the GNOME runtime that runtime-version in dev.quoth.Quoth.yaml names.
# The audit workflow reads it back from the rendered `.python-version`.
_PYTHON = "3.13"


def _source_groups(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "sources":
                if not isinstance(child, list):
                    raise ManifestError("manifest sources must be a list")
                yield child
            else:
                yield from _source_groups(child)
    elif isinstance(value, list):
        for child in value:
            yield from _source_groups(child)


def _load_manifest(path: Path) -> object:
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ManifestError(f"{path} is not valid YAML: {error}") from error
    if not isinstance(data, dict):
        raise ManifestError(f"{path} must contain a YAML mapping")
    return data


def _parse_source(path: Path, source: object) -> tuple[str, Version, WheelSource]:
    if not isinstance(source, dict):
        raise ManifestError(f"{path} contains a non-mapping source")
    if source.get("type") != "file":
        raise ManifestError(f"{path} contains unsupported source type")

    url = source.get("url")
    if not isinstance(url, str):
        raise ManifestError(f"{path} contains a source without a URL")
    filename = unquote(Path(urlsplit(url).path).name)
    try:
        name, version, _build, _tags = parse_wheel_filename(filename)
    except InvalidWheelFilename as error:
        raise ManifestError(
            f"{path} URL is not a valid wheel filename: {filename}"
        ) from error

    sha256 = source.get("sha256")
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        raise ManifestError(f"{path} wheel {filename} needs one valid sha256")

    arches = source.get("only-arches", [])
    if not isinstance(arches, list) or not all(
        isinstance(arch, str) for arch in arches
    ):
        raise ManifestError(f"{path} wheel {filename} has invalid only-arches")

    wheel_source = WheelSource(
        manifest=path.name,
        filename=filename,
        url=url,
        sha256=sha256.lower(),
        only_arches=tuple(sorted(arches)),
    )
    return str(name), version, wheel_source


def collect_wheel_pins(manifest_paths: list[Path]) -> list[WheelPin]:
    """Collect one exact pin per package, keeping every architecture's wheel.

    `manifest_paths` must name exactly two distinct manifests. Raises ManifestError
    when a source is malformed or unhashed, or when a package's versions disagree.
    """
    if len(manifest_paths) != 2:
        raise ManifestError("exactly two Flatpak wheel manifests are required")
    if len({path.resolve() for path in manifest_paths}) != 2:
        raise ManifestError("the two Flatpak wheel manifests must be distinct")

    versions: dict[str, Version] = {}
    sources: dict[str, list[WheelSource]] = {}
    for path in manifest_paths:
        groups = list(_source_groups(_load_manifest(path)))
        if not groups:
            raise ManifestError(f"{path} does not contain any wheel sources")
        parsed_wheels = 0
        for group in groups:
            for source in group:
                name, version, wheel_source = _parse_source(path, source)
                parsed_wheels += 1
                current = versions.setdefault(name, version)
                if current != version:
                    raise ManifestError(
                        f"{name} has conflicting versions {current} and {version}"
                    )
                sources.setdefault(name, []).append(wheel_source)
        if parsed_wheels == 0:
            raise ManifestError(f"{path} does not contain any wheel sources")

    return [
        WheelPin(name, versions[name], tuple(sorted(sources[name])))
        for name in sorted(versions)
    ]


def _project_text(pins: list[WheelPin]) -> str:
    requirements = "".join(f'    "{pin.name}=={pin.version}",\n' for pin in pins)
    major, minor = _PYTHON.split(".")
    return (
        "[project]\n"
        'name = "quoth-flatpak-python-audit"\n'
        'version = "0"\n'
        f'requires-python = ">={_PYTHON},<{major}.{int(minor) + 1}"\n'
        "dependencies = [\n"
        f"{requirements}"
        "]\n"
        "\n"
        "[tool.uv]\n"
        "environments = [\"sys_platform == 'linux'\"]\n"
    )


def _input_record(manifest_paths: list[Path], pins: list[WheelPin]) -> dict:
    manifests = [
        {
            "name": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(manifest_paths, key=lambda item: item.name)
    ]
    wheels = [
        {
            "filename": source.filename,
            "manifest": source.manifest,
            "name": pin.name,
            "only_arches": list(source.only_arches),
            "sha256": source.sha256,
            "url": source.url,
            "version": str(pin.version),
        }
        for pin in pins
        for source in pin.sources
    ]
    return {
        "manifests": manifests,
        "platform": "linux",
        "python": _PYTHON,
        "wheels": wheels,
    }


def render_audit_project(
    manifest_paths: list[Path], output_dir: Path
) -> list[WheelPin]:
    """Write the uv audit project for the manifests' wheels into `output_dir`.

    Writes `pyproject.toml`, `.python-version` and `flatpak-wheel-inputs.json`, and
    returns the pins; raises ManifestError as `collect_wheel_pins` does.
    """
    pins = collect_wheel_pins(manifest_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pyproject.toml").write_text(_project_text(pins))
    (output_dir / ".python-version").write_text(f"{_PYTHON}\n")
    (output_dir / "flatpak-wheel-inputs.json").write_text(
        json.dumps(_input_record(manifest_paths, pins), indent=2, sort_keys=True) + "\n"
    )
    return pins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        pins = render_audit_project(args.manifests, args.output_dir)
    except ManifestError as error:
        parser.error(str(error))
    print(f"Rendered {len(pins)} exact Flatpak wheel pins in {args.output_dir}")


if __name__ == "__main__":
    main()
