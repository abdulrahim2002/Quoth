#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Prepare signed static APT and Flatpak repositories from tested packages."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import ipaddress
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

APP_ID = "dev.quoth.Quoth"
FLATPAK_REF = f"app/{APP_ID}/x86_64/master"
# Each release's Corresponding Source is a release asset, not part of this site: its
# GCC source RPM alone outgrows GitHub's file limit once stored as an OSTree object.
SOURCE_RELEASES = "https://github.com/abullard1/Quoth/releases/tag"
# GitHub refuses a push that holds any file larger than this.
GITHUB_FILE_LIMIT = 100 * 1024 * 1024
_FINGERPRINT = re.compile(r"[0-9A-Fa-f]{40}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_CHECKSUM = re.compile(r"([0-9a-f]{64})  (?:\./)?([A-Za-z0-9][A-Za-z0-9._+~-]*)\n?\Z")
_IMPORT = re.compile(r"Importing ([^\s]+) \(([0-9a-f]{64})\)\n?\Z")
_VERSION = re.compile(r"[0-9][0-9A-Za-z.+:~-]*\Z")
_SCHEMA = {"version", "release", "environment", "source_commit", "target"}
_PACKAGES = {"deb": "*.deb", "arch": "*.pkg.tar.zst", "flatpak": "*.flatpak"}
_MANIFEST_FILES = {"COMPLETE", "site-manifest.json", "site-manifest.sig"}


def _run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> bytes:
    command_env = os.environ.copy() if env is None else env.copy()
    command_env["LC_ALL"] = "C"
    return subprocess.run(
        command, cwd=cwd, env=command_env, check=True, capture_output=True, timeout=300
    ).stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_url(value: str) -> str:
    decoded = unquote(value)
    if (
        any(char.isspace() or ord(char) == 127 for char in decoded)
        or re.search(r"%(?![0-9A-Fa-f]{2})", value)
        or "\\" in value
    ):
        raise ValueError("base URL contains whitespace or control characters")
    parsed = urlsplit(value)
    try:
        _ = parsed.port
    except ValueError as error:
        raise ValueError("base URL has an invalid port") from error
    if (
        not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base URL must be an absolute repository URL")
    if parsed.scheme == "http":
        host = parsed.hostname or ""
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host == "localhost"
        if not loopback:
            raise ValueError("HTTP is allowed only for loopback rehearsal")
    elif parsed.scheme != "https":
        raise ValueError("base URL must use HTTPS")
    return value.rstrip("/")


def _load_metadata(path: Path) -> dict[str, str]:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate metadata key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(), object_pairs_hook=no_duplicates)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid metadata: {path}") from error
    if not isinstance(value, dict) or set(value) != _SCHEMA:
        raise ValueError(f"metadata must contain exactly {sorted(_SCHEMA)}")
    if not all(isinstance(item, str) for item in value.values()):
        raise ValueError("metadata values must be strings")
    return value


def _validate_inputs(
    artifacts: Path,
) -> tuple[dict[str, str], dict[str, Path], dict[str, str]]:
    metadata: dict[str, str] | None = None
    packages: dict[str, Path] = {}
    checksums: dict[str, str] = {}
    for channel, pattern in _PACKAGES.items():
        directory = artifacts / channel
        sidecars = [
            directory / "SHA256SUMS",
            directory / "source-commit.txt",
            directory / "release-metadata.json",
        ]
        if directory.is_symlink() or any(
            not path.is_file() or path.is_symlink() for path in sidecars
        ):
            raise ValueError(f"{channel} artifact sidecars must be regular files")
        found = list(directory.glob(pattern))
        if len(found) != 1 or not found[0].is_file() or found[0].is_symlink():
            raise ValueError(f"{channel} must contain exactly one {pattern} package")
        package = found[0]
        match = _CHECKSUM.fullmatch((directory / "SHA256SUMS").read_text())
        if not match or match.group(2) != package.name:
            raise ValueError(f"{channel} has an unsafe or mismatched SHA256SUMS entry")
        actual = _sha256(package)
        if match.group(1) != actual:
            raise ValueError(f"{channel} package checksum does not match")
        current = _load_metadata(directory / "release-metadata.json")
        source = (directory / "source-commit.txt").read_text().strip()
        if not _COMMIT.fullmatch(source) or source != current["source_commit"]:
            raise ValueError(f"{channel} source commit does not match metadata")
        if metadata is not None and current != metadata:
            raise ValueError("release metadata differs between package channels")
        metadata = current
        packages[channel] = package
        checksums[channel] = actual

    assert metadata is not None
    version = metadata["version"]
    if not _VERSION.fullmatch(version):
        raise ValueError("metadata version is not a safe package version")
    target = metadata["target"]
    expected = {
        "staging": ("candidate", f"quoth@{version}+g{metadata['source_commit']}"),
        "production": ("production", f"quoth@{version}"),
    }.get(target)
    if expected is None or (metadata["environment"], metadata["release"]) != expected:
        raise ValueError("metadata release identity does not match its target")
    return metadata, packages, checksums


def _copy_payload(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != _sha256(source):
            raise ValueError(f"refusing changed payload collision: {destination}")
        return
    shutil.copy2(source, destination)


def _copy_previous(previous: Path, output: Path) -> None:
    shutil.copytree(previous, output, dirs_exist_ok=True)
    (output / "COMPLETE").unlink()


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _site_files(site: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in site.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError(f"site contains an unsafe entry: {path}")
        relative = path.relative_to(site).as_posix()
        if path.is_file() and relative not in _MANIFEST_FILES:
            files[relative] = path
    return files


def _validate_previous(
    previous: Path,
    metadata: dict[str, str],
    packages: dict[str, Path],
    checksums: dict[str, str],
    signing_key: str,
    signing_env: dict[str, str],
) -> None:
    if not (previous / "COMPLETE").is_file():
        raise ValueError("previous site is not marked COMPLETE")
    try:
        identity = json.loads((previous / "preparation.json").read_text())
        previous_metadata = {key: identity[key] for key in _SCHEMA}
        previous_inputs = identity["inputs"]
        previous_key = identity["signing_fingerprint"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("previous site has no valid preparation identity") from error
    if previous_key != signing_key:
        raise ValueError("previous site uses a different signing key")
    previous_version = previous_metadata["version"]
    if not isinstance(previous_version, str) or not _VERSION.fullmatch(
        previous_version
    ):
        raise ValueError("previous site has an invalid version")
    comparison = subprocess.run(
        ["dpkg", "--compare-versions", metadata["version"], "lt", previous_version],
        check=False,
        capture_output=True,
        timeout=10,
    )
    if comparison.returncode == 0:
        raise ValueError("release version would rewind the previous site")
    if comparison.returncode != 1:
        raise ValueError("release versions cannot be compared safely")
    current_inputs = {
        channel: {"filename": packages[channel].name, "sha256": checksums[channel]}
        for channel in _PACKAGES
    }
    if metadata["version"] == previous_version and (
        previous_metadata != metadata or previous_inputs != current_inputs
    ):
        raise ValueError("same release version has different package identity")
    status = _run(
        [
            "gpg",
            "--batch",
            "--status-fd=1",
            "--verify",
            "site-manifest.sig",
            "site-manifest.json",
        ],
        previous,
        signing_env,
    ).decode()
    valid = False
    for line in status.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[:2] == ["[GNUPG:]", "VALIDSIG"]:
            valid = fields[2] == signing_key or (
                len(fields) >= 12 and fields[-1] == signing_key
            )
    if not valid:
        raise ValueError("previous site manifest was not signed by the expected key")
    try:
        manifest = json.loads((previous / "site-manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("previous site manifest is invalid") from error
    files = _site_files(previous)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != set(files)
        or any(
            not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item)
            for item in manifest.values()
        )
        or any(_sha256(path) != manifest[name] for name, path in files.items())
    ):
        raise ValueError("previous site files do not match its signed manifest")


def _fields(paragraph: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in paragraph.splitlines():
        if not line or line[0].isspace() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key] = value.strip()
    return fields


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode() if isinstance(content, str) else content)


def prepare(
    artifacts: Path,
    output: Path,
    base_url: str,
    signing_key: str,
    previous_site: Path | None = None,
    pages_domain: str | None = None,
) -> dict[str, object]:
    """Build a signed APT and Flatpak site in `output` from tested package artifacts.

    `artifacts` holds one directory per channel, each with its package and its
    SHA256SUMS, source-commit and release-metadata sidecars. Signing uses `signing_key`
    from the keyring `GNUPGHOME` names. A `previous_site` must verify against its
    signed manifest and is carried over, and the release may not rewind its version.
    Returns the identity also written to `preparation.json`; raises ValueError on any
    unsafe or inconsistent input, or on a site file too large for GitHub to accept.
    """
    base_url = _base_url(base_url)
    if pages_domain is not None and pages_domain != urlsplit(base_url).hostname:
        raise ValueError("Pages domain must match the base URL host")
    if not _FINGERPRINT.fullmatch(signing_key):
        raise ValueError("signing key must be a full 40-hex fingerprint")
    signing_key = signing_key.upper()
    gnupg_home_value = os.environ.get("GNUPGHOME")
    if not gnupg_home_value or not Path(gnupg_home_value).is_dir():
        raise ValueError("GNUPGHOME must name an existing external keyring")
    gnupg_home = Path(gnupg_home_value).resolve()
    signing_env = os.environ.copy()
    signing_env["GNUPGHOME"] = str(gnupg_home)
    artifacts = artifacts.resolve()
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be new or empty")
    if _overlaps(output, artifacts) or _overlaps(output, gnupg_home):
        raise ValueError("output must not overlap artifacts or GNUPGHOME")
    if previous_site is not None:
        previous_site = previous_site.resolve()
        protected = (output, artifacts, gnupg_home)
        if any(_overlaps(previous_site, path) for path in protected):
            raise ValueError(
                "previous site must not overlap output, artifacts, or GNUPGHOME"
            )

    metadata, packages, checksums = _validate_inputs(artifacts)
    if previous_site is not None:
        _validate_previous(
            previous_site, metadata, packages, checksums, signing_key, signing_env
        )
    output.mkdir(parents=True, exist_ok=True)
    if previous_site is not None:
        _copy_previous(previous_site, output)

    deb_target = output / "apt/pool/main/q/quoth" / packages["deb"].name
    _copy_payload(packages["deb"], deb_target)

    apt_root = output / "apt"
    packages_file = apt_root / "dists/stable/main/binary-amd64/Packages"
    package_index = _run(
        ["apt-ftparchive", "packages", "pool/main/q/quoth"], cwd=apt_root
    )
    _write(packages_file, package_index)
    paragraphs = [item for item in package_index.decode().split("\n\n") if item.strip()]
    expected_filename = f"pool/main/q/quoth/{packages['deb'].name}"
    matching = [
        fields
        for fields in map(_fields, paragraphs)
        if fields.get("Filename") == expected_filename
    ]
    fields = matching[0] if len(matching) == 1 else {}
    deb_version = fields.get("Version", "").split(":")[-1].rsplit("-", 1)[0]
    if fields.get("Package") != "quoth" or fields.get("Architecture") != "amd64":
        raise ValueError("Debian package must be quoth for amd64")
    if deb_version != metadata["version"]:
        raise ValueError("Debian upstream version does not match release metadata")
    with packages_file.with_suffix(".gz").open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as archive:
            archive.write(package_index)

    release_options = [
        "-o",
        "APT::FTPArchive::Release::Origin=Quoth",
        "-o",
        "APT::FTPArchive::Release::Label=Quoth",
        "-o",
        "APT::FTPArchive::Release::Suite=stable",
        "-o",
        "APT::FTPArchive::Release::Codename=stable",
        "-o",
        "APT::FTPArchive::Release::Architectures=amd64",
        "-o",
        "APT::FTPArchive::Release::Components=main",
    ]
    release = packages_file.parents[2] / "Release"
    for name in ("Release", "InRelease", "Release.gpg"):
        release.with_name(name).unlink(missing_ok=True)
    _write(
        release,
        _run(["apt-ftparchive", *release_options, "release", "dists/stable"], apt_root),
    )
    for name, mode in (("InRelease", "--clearsign"), ("Release.gpg", "--detach-sign")):
        _run(
            [
                "gpg",
                "--batch",
                "--yes",
                "--local-user",
                signing_key,
                mode,
                "--output",
                str(release.with_name(name)),
                str(release),
            ],
            cwd=output,
            env=signing_env,
        )
    public_key = _run(["gpg", "--batch", "--export", signing_key], output, signing_env)
    if not public_key:
        raise ValueError("signing key could not be exported")
    _write(output / "quoth.gpg", public_key)

    repo = output / "flatpak/repo"
    repo.mkdir(parents=True, exist_ok=True)
    if (repo / "config").exists():
        mode = _run(["ostree", f"--repo={repo}", "config", "get", "core.mode"], output)
        if mode.decode().strip() != "archive-z2":
            raise ValueError("previous Flatpak repository is not archive-z2")
    else:
        _run(["ostree", f"--repo={repo}", "init", "--mode=archive-z2"], output)
    imported = _run(
        [
            "flatpak",
            "build-import-bundle",
            f"--gpg-sign={signing_key}",
            f"--gpg-homedir={gnupg_home}",
            str(repo),
            str(packages["flatpak"]),
        ],
        output,
        signing_env,
    ).decode()
    imported_identity = _IMPORT.fullmatch(imported)
    if not imported_identity or imported_identity.group(1) != FLATPAK_REF:
        raise ValueError(f"Flatpak bundle did not import {FLATPAK_REF}")
    flatpak_commit = imported_identity.group(2)
    repository_commit = (
        _run(["ostree", f"--repo={repo}", "rev-parse", FLATPAK_REF], output)
        .decode()
        .strip()
    )
    if repository_commit != flatpak_commit:
        raise ValueError("Flatpak repository commit differs from the imported bundle")
    _run(
        [
            "flatpak",
            "build-update-repo",
            "--title=Quoth",
            "--default-branch=master",
            f"--gpg-sign={signing_key}",
            f"--gpg-homedir={gnupg_home}",
            str(repo),
        ],
        output,
        signing_env,
    )

    encoded_key = base64.b64encode(public_key).decode()
    _write(
        output / "quoth.sources",
        f"""Types: deb
URIs: {base_url}/apt
Suites: stable
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/quoth.gpg
""",
    )
    _write(
        output / "quoth.flatpakrepo",
        f"""[Flatpak Repo]
Title=Quoth
Url={base_url}/flatpak/repo
GPGKey={encoded_key}
DefaultBranch=master
""",
    )
    _write(
        output / "quoth.flatpakref",
        f"""[Flatpak Ref]
Name={APP_ID}
Branch=master
Title=Quoth
IsRuntime=false
Url={base_url}/flatpak/repo
GPGKey={encoded_key}
RuntimeRepo=https://dl.flathub.org/repo/flathub.flatpakrepo
""",
    )
    # The GPL's directions to the Corresponding Source, next to the object code.
    version = metadata["version"]
    _write(
        output / "index.html",
        f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Quoth packages</title>
<h1>Quoth packages</h1>
<p>Signed APT and Flatpak repositories for Quoth, free software under
GPL-3.0-or-later.</p>
<h2>Source code</h2>
<p>The source of each release served here is attached to that release at
<code>{SOURCE_RELEASES}/vVERSION</code>. <code>quoth-sources.flatpak</code> is the
Flatpak's complete Corresponding Source: every module it bundles, with its
patches. <code>quoth-deb-sources.tar.gz</code> is the unmodified source of the
wl-clipboard, ydotool and ThorVG the .deb bundles, at their pinned commits, with the
wayland-protocols files wl-clipboard's build generates code from.
<code>quoth-VERSION.tar.gz</code> is Quoth's own source. The .deb names the sources
of the components it bundles in <code>/usr/share/doc/quoth/copyright</code>.</p>
<p>Latest release: <a href="{SOURCE_RELEASES}/v{version}">Quoth {version}</a>.</p>
</html>
""",
    )
    if pages_domain is not None:
        _write(output / "CNAME", f"{pages_domain}\n")
        _write(output / ".nojekyll", b"")
    result: dict[str, object] = {
        **metadata,
        "signing_fingerprint": signing_key,
        "flatpak_commit": flatpak_commit,
        "inputs": {
            channel: {"filename": packages[channel].name, "sha256": checksums[channel]}
            for channel in _PACKAGES
        },
    }
    _write(
        output / "preparation.json", json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    files = _site_files(output)
    oversized = sorted(
        name for name, path in files.items() if path.stat().st_size > GITHUB_FILE_LIMIT
    )
    if oversized:
        raise ValueError(f"GitHub refuses files over 100 MiB: {', '.join(oversized)}")
    manifest = {name: _sha256(path) for name, path in files.items()}
    _write(
        output / "site-manifest.json",
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
    )
    _run(
        [
            "gpg",
            "--batch",
            "--yes",
            "--local-user",
            signing_key,
            "--detach-sign",
            "--output",
            str(output / "site-manifest.sig"),
            str(output / "site-manifest.json"),
        ],
        output,
        signing_env,
    )
    _write(output / "COMPLETE", "prepared\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--signing-key", required=True)
    parser.add_argument("--previous-site", type=Path)
    parser.add_argument("--pages-domain")
    args = parser.parse_args()
    try:
        result = prepare(
            args.artifacts,
            args.output,
            args.base_url,
            args.signing_key,
            args.previous_site,
            args.pages_domain,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors="replace").strip()
        parser.error(detail or str(error))
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
