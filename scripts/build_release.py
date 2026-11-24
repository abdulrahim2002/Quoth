# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Build (and optionally publish) the per-release model asset manifest.

Model assets (ggml weights, and OpenVINO encoder files for NPU acceleration) ship as
GitHub release assets from a single origin. At download time `quoth.downloader`
fetches `manifest.json` from the release and verifies each file's SHA-256 against it.
This script produces that manifest from a directory of built assets, can check the
ggml weights against the upstream files they mirror, and can upload the assets and
manifest with the `gh` CLI. Filenames and the encoder set come from `quoth.models`
and hashing reuses `quoth.downloader._sha256`, so the manifest is exactly what the
downloader looks up and verifies. Release-time tooling; not shipped.

Usage:
    # Write manifest.json into dist/release-assets/ (no network):
    uv run python scripts/build_release.py --source ~/whisper.cpp/models
    # Also check that the ggml weights are bit-identical to upstream:
    uv run python scripts/build_release.py --verify-upstream
    # Also upload the assets and manifest to the default tag's draft release:
    uv run python scripts/build_release.py --source ~/whisper.cpp/models --upload
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import quoth.downloader as dl
from quoth.models import (
    AVAILABLE_MODELS,
    encoder_bin_name,
    encoder_xml_name,
    ggml_name,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SOURCE = Path.home() / "whisper.cpp" / "models"
_DEFAULT_OUT = _REPO_ROOT / "dist" / "release-assets"

# The upstream the ggml weights mirror. package.yml's CI fixture downloads its base
# model from the same revision, so CI transcribes the bytes the release ships; a
# workflow cannot import this constant.
UPSTREAM_REPO = "ggerganov/whisper.cpp"
UPSTREAM_REVISION = "5359861c739e955e79d9a303bcbc70fb988958b1"


def asset_filenames() -> list[str]:
    """Return every asset the downloader verifies against the manifest.

    That is each model's ggml weights, plus the OpenVINO encoder (xml, bin) for the
    models flagged `has_openvino_encoder`.
    """
    names: list[str] = []
    for model in AVAILABLE_MODELS:
        names.append(ggml_name(model.name))
        if model.has_openvino_encoder:
            names.append(encoder_xml_name(model.name))
            names.append(encoder_bin_name(model.name))
    return names


def build_manifest(source_dir: Path) -> tuple[dict[str, str], list[Path]]:
    """Compute `{filename: sha256}` for every expected asset file.

    Returns the manifest mapping and the resolved source paths (for upload).
    Raises FileNotFoundError listing every missing file, so a partial asset
    set fails loudly instead of publishing a manifest the downloader can't
    satisfy.
    """
    manifest: dict[str, str] = {}
    paths: list[Path] = []
    missing: list[str] = []

    for name in asset_filenames():
        src = source_dir / name

        if not src.exists():
            missing.append(name)
            continue

        manifest[name] = dl._sha256(src)
        paths.append(src)

    if missing:
        raise FileNotFoundError(
            "Missing asset file(s) under "
            f"{source_dir}:\n  "
            + "\n  ".join(missing)
            + "\n\nThe ggml weights mirror whisper.cpp's ggml-<model>-q8_0.bin; "
            "build the encoders with whisper.cpp's "
            "models/convert-whisper-to-openvino.py. "
        )
    return manifest, paths


def verify_upstream(
    manifest: dict[str, str], source_dir: Path, opener=urllib.request.urlopen
) -> None:
    """Check each ggml weight against its Hugging Face LFS hash at the pinned revision.

    One `paths-info` request returns every weight's upstream sha256 and size, and
    silently omits a path upstream lacks. The encoders are converted locally and have
    no upstream to compare with. Exits the process listing every weight that is
    missing upstream or differs from it.
    """
    names = [ggml_name(model.name) for model in AVAILABLE_MODELS]
    request = urllib.request.Request(
        f"https://huggingface.co/api/models/{UPSTREAM_REPO}/paths-info/"
        f"{UPSTREAM_REVISION}",
        data=json.dumps({"paths": names}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with opener(request, timeout=60) as response:
            upstream = {
                item["path"]: item.get("lfs", {}) for item in json.load(response)
            }
    except (OSError, ValueError) as error:
        sys.exit(f"Could not read {UPSTREAM_REPO} at {UPSTREAM_REVISION}: {error}")

    problems: list[str] = []
    for name in names:
        lfs = upstream.get(name)
        if lfs is None:
            problems.append(f"{name}: not found upstream")
            continue
        size = (source_dir / name).stat().st_size
        if (lfs.get("oid"), lfs.get("size")) != (manifest[name], size):
            problems.append(
                f"{name}: sha256 {manifest[name]}, {size} bytes; upstream has "
                f"sha256 {lfs.get('oid')}, {lfs.get('size')} bytes"
            )
    if problems:
        sys.exit(
            f"ggml weights differ from {UPSTREAM_REPO} at {UPSTREAM_REVISION}:\n  "
            + "\n  ".join(problems)
        )
    print(f"{len(names)} ggml weight(s) match {UPSTREAM_REPO} at {UPSTREAM_REVISION}.")


def write_manifest(manifest: dict[str, str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / dl.MANIFEST_FILENAME
    dest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return dest


def _print_summary(
    manifest: dict[str, str], paths: list[Path], manifest_path: Path
) -> None:
    by_name = {p.name: p for p in paths}
    print(f"\nmanifest → {manifest_path}\n")

    for name, sha in manifest.items():
        size_mb = by_name[name].stat().st_size / 1_000_000
        print(f"  {name:<40} {size_mb:>7.1f} MB  {sha}")

    print(f"\n{len(manifest)} asset file(s) hashed.")


def upload(tag: str, manifest_path: Path, asset_paths: list[Path]) -> None:
    """Upload the manifest and asset files to the draft GitHub release for `tag`.

    The release must exist as a draft. It is published once every asset is attached,
    and release immutability then fixes them. Exits the process when there is no
    draft or the upload fails.
    """
    view = subprocess.run(
        ["gh", "release", "view", tag, "--json", "isDraft", "--jq", ".isDraft"],
        capture_output=True,
        text=True,
    )
    if view.returncode != 0:
        sys.exit(
            f"Release '{tag}' not found. Push the tag and create the release as a "
            f"draft first (gh release create {tag} --draft --verify-tag), then re-run "
            f"with --upload.\ngh said: {view.stderr.strip()}"
        )
    if view.stdout.strip() != "true":
        sys.exit(
            f"Release '{tag}' is already published. Assets go onto its draft, "
            "which is published once every asset is attached."
        )

    assets = [str(manifest_path), *(str(p) for p in asset_paths)]
    print(f"\nUploading {len(assets)} asset(s) to release '{tag}'…")

    result = subprocess.run(["gh", "release", "upload", tag, *assets])

    if result.returncode != 0:
        sys.exit(f"Upload failed (gh exit {result.returncode}).")

    print("Upload complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=_DEFAULT_SOURCE,
        help=f"Directory containing the built encoder files (default: {_DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Where to write manifest.json (default: {_DEFAULT_OUT})",
    )
    parser.add_argument(
        "--tag",
        default=dl.ASSETS_TAG,
        help=f"Release tag to upload to (default: {dl.ASSETS_TAG})",
    )
    parser.add_argument(
        "--verify-upstream",
        action="store_true",
        help="Check the ggml weights against Hugging Face at the pinned revision.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload assets + manifest to the GitHub release via gh.",
    )
    args = parser.parse_args()

    manifest, asset_paths = build_manifest(args.source)
    if args.verify_upstream:
        verify_upstream(manifest, args.source)
    manifest_path = write_manifest(manifest, args.out)
    _print_summary(manifest, asset_paths, manifest_path)

    if args.upload:
        upload(args.tag, manifest_path, asset_paths)
    else:
        print("\n(generate-only; pass --upload to publish to the release)")


if __name__ == "__main__":
    main()
