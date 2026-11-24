# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `scripts/prepare_repositories.py`.

Its `_run` tool runner is faked, so no apt, gpg or flatpak command runs; the
previous-site tests still call the host's `dpkg`.
"""

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from scripts import prepare_repositories as repositories

_SOURCE = "a" * 40
_FINGERPRINT = "B" * 40

_needs_dpkg = pytest.mark.skipif(
    shutil.which("dpkg") is None, reason="compares versions with the host's dpkg"
)


def _complete(
    previous: Path,
    artifacts: Path,
    version: str = "0.1.0",
    fingerprint: str = _FINGERPRINT,
) -> None:
    source = _SOURCE if version == "0.1.0" else "0" * 40
    metadata = {
        "version": version,
        "release": f"quoth@{version}+g{source}",
        "environment": "candidate",
        "source_commit": source,
        "target": "staging",
    }
    inputs = {}
    for channel, pattern in repositories._PACKAGES.items():
        package = next((artifacts / channel).glob(pattern))
        inputs[channel] = {
            "filename": package.name,
            "sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
        }
    (previous / "COMPLETE").write_text("prepared\n")
    (previous / "preparation.json").write_text(
        json.dumps({**metadata, "inputs": inputs, "signing_fingerprint": fingerprint})
    )
    manifest = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in repositories._site_files(previous).items()
    }
    (previous / "site-manifest.json").write_text(json.dumps(manifest))
    (previous / "site-manifest.sig").write_bytes(b"test signature")


def _valid_signature(*args, **kwargs) -> bytes:
    return f"[GNUPG:] VALIDSIG {_FINGERPRINT}\n".encode()


def _artifacts(root: Path) -> tuple[Path, dict[str, bytes]]:
    payloads = {
        "deb": b"tested deb bytes",
        "arch": b"tested arch bytes",
        "flatpak": b"tested flatpak bytes",
    }
    names = {
        "deb": "quoth_0.1.0_amd64.deb",
        "arch": "quoth-0.1.0-1-x86_64.pkg.tar.zst",
        "flatpak": "quoth.flatpak",
    }
    metadata = {
        "version": "0.1.0",
        "release": f"quoth@0.1.0+g{_SOURCE}",
        "environment": "candidate",
        "source_commit": _SOURCE,
        "target": "staging",
    }
    artifacts = root / "artifacts"
    for channel, payload in payloads.items():
        directory = artifacts / channel
        directory.mkdir(parents=True)
        (directory / names[channel]).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        prefix = "./" if not channel.startswith("flatpak") else ""
        (directory / "SHA256SUMS").write_text(f"{digest}  {prefix}{names[channel]}\n")
        (directory / "source-commit.txt").write_text(f"{_SOURCE}\n")
        (directory / "release-metadata.json").write_text(json.dumps(metadata))
    return artifacts, payloads


def _environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gnupg = tmp_path / "gnupg"
    gnupg.mkdir()
    monkeypatch.setenv("GNUPGHOME", str(gnupg))


@pytest.mark.parametrize(
    ("url", "fingerprint", "message"),
    [
        ("http://packages.example.test", _FINGERPRINT, "loopback rehearsal"),
        ("https://packages.example.test\n.invalid", _FINGERPRINT, "whitespace"),
        ("https://packages.example.test/bad path", _FINGERPRINT, "whitespace"),
        ("https://packages.example.test", "short", "40-hex fingerprint"),
    ],
)
def test_rejects_malformed_identity_before_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    fingerprint: str,
    message: str,
):
    artifacts, _ = _artifacts(tmp_path)
    _environment(tmp_path, monkeypatch)
    monkeypatch.setattr(
        repositories, "_run", lambda *args, **kwargs: pytest.fail("tool was invoked")
    )

    with pytest.raises(ValueError, match=message):
        repositories.prepare(artifacts, tmp_path / "site", url, fingerprint)


def test_rejects_unsafe_checksum_filename_before_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifacts, _ = _artifacts(tmp_path)
    _environment(tmp_path, monkeypatch)
    digest = hashlib.sha256(b"tested deb bytes").hexdigest()
    (artifacts / "deb/SHA256SUMS").write_text(f"{digest}  ../package.deb\n")
    monkeypatch.setattr(
        repositories, "_run", lambda *args, **kwargs: pytest.fail("tool was invoked")
    )

    with pytest.raises(ValueError, match="unsafe"):
        repositories.prepare(
            artifacts, tmp_path / "site", "https://packages.example.test", _FINGERPRINT
        )


@pytest.mark.parametrize(
    "pages_domain",
    ["other.example.test", "packages.example.test..", "packages.example.test\n"],
)
def test_rejects_invalid_pages_domain_before_output_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pages_domain: str
):
    artifacts, _ = _artifacts(tmp_path)
    _environment(tmp_path, monkeypatch)
    output = tmp_path / "site"
    monkeypatch.setattr(
        repositories, "_run", lambda *args, **kwargs: pytest.fail("tool was invoked")
    )

    with pytest.raises(ValueError, match="Pages domain"):
        repositories.prepare(
            artifacts,
            output,
            "https://packages.example.test/releases",
            _FINGERPRINT,
            pages_domain=pages_domain,
        )

    assert not output.exists()


def test_rejects_wrong_debian_identity_before_signing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifacts, _ = _artifacts(tmp_path)
    _environment(tmp_path, monkeypatch)
    commands: list[list[str]] = []

    def run(command: list[str], cwd: Path, env=None) -> bytes:
        commands.append(command)
        return (
            b"Package: other\nVersion: 0.1.0\nArchitecture: amd64\n"
            b"Filename: pool/main/q/quoth/quoth_0.1.0_amd64.deb\n\n"
        )

    monkeypatch.setattr(repositories, "_run", run)

    with pytest.raises(ValueError, match="must be quoth"):
        repositories.prepare(
            artifacts,
            tmp_path / "site",
            "https://packages.example.test",
            _FINGERPRINT,
        )

    assert not any(command[0] == "gpg" for command in commands)


@_needs_dpkg
def test_rejects_changed_payload_from_complete_previous_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifacts, _ = _artifacts(tmp_path)
    _environment(tmp_path, monkeypatch)
    previous = tmp_path / "previous"
    old = previous / "apt/pool/main/q/quoth/quoth_0.1.0_amd64.deb"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"different build with the same version")
    _complete(previous, artifacts)
    monkeypatch.setattr(repositories, "_run", _valid_signature)

    with pytest.raises(ValueError, match="collision"):
        repositories.prepare(
            artifacts,
            tmp_path / "site",
            "https://packages.example.test/releases",
            _FINGERPRINT,
            previous,
        )


def test_rejects_overlapping_previous_site_before_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifacts, _ = _artifacts(tmp_path)
    _environment(tmp_path, monkeypatch)
    (tmp_path / "COMPLETE").write_text("prepared\n")
    monkeypatch.setattr(
        repositories, "_run", lambda *args, **kwargs: pytest.fail("tool was invoked")
    )

    with pytest.raises(ValueError, match="must not overlap"):
        repositories.prepare(
            artifacts,
            tmp_path / "site",
            "https://packages.example.test/releases",
            _FINGERPRINT,
            tmp_path,
        )


@_needs_dpkg
def test_rejects_tampered_file_in_previous_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifacts, _ = _artifacts(tmp_path)
    _environment(tmp_path, monkeypatch)
    previous = tmp_path / "previous"
    old = previous / "apt/pool/main/q/quoth/quoth_0.0.9_amd64.deb"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"original old deb")
    _complete(previous, artifacts, "0.0.9")
    old.write_bytes(b"tampered old deb")
    monkeypatch.setattr(repositories, "_run", _valid_signature)

    with pytest.raises(ValueError, match="signed manifest"):
        repositories.prepare(
            artifacts,
            tmp_path / "site",
            "https://packages.example.test/releases",
            _FINGERPRINT,
            previous,
        )


@pytest.mark.parametrize(
    ("version", "fingerprint", "message"),
    [
        pytest.param("0.2.0", _FINGERPRINT, "rewind", marks=_needs_dpkg),
        ("0.0.9", "C" * 40, "different signing key"),
    ],
)
def test_rejects_previous_site_rewind_or_key_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    fingerprint: str,
    message: str,
):
    artifacts, _ = _artifacts(tmp_path)
    _environment(tmp_path, monkeypatch)
    previous = tmp_path / "previous"
    previous.mkdir()
    _complete(previous, artifacts, version, fingerprint)
    monkeypatch.setattr(
        repositories, "_run", lambda *args, **kwargs: pytest.fail("tool was invoked")
    )

    with pytest.raises(ValueError, match=message):
        repositories.prepare(
            artifacts,
            tmp_path / "site",
            "https://packages.example.test/releases",
            _FINGERPRINT,
            previous,
        )


@_needs_dpkg
def test_rejects_changed_identity_at_same_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifacts, _ = _artifacts(tmp_path)
    _environment(tmp_path, monkeypatch)
    previous = tmp_path / "previous"
    previous.mkdir()
    _complete(previous, artifacts)
    identity = json.loads((previous / "preparation.json").read_text())
    identity["source_commit"] = "d" * 40
    (previous / "preparation.json").write_text(json.dumps(identity))
    monkeypatch.setattr(
        repositories, "_run", lambda *args, **kwargs: pytest.fail("tool was invoked")
    )

    with pytest.raises(ValueError, match="different package identity"):
        repositories.prepare(
            artifacts,
            tmp_path / "site",
            "https://packages.example.test/releases",
            _FINGERPRINT,
            previous,
        )


@_needs_dpkg
def test_prepares_repositories_without_changing_artifact_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifacts, payloads = _artifacts(tmp_path)
    _environment(tmp_path, monkeypatch)
    output = tmp_path / "site"
    previous = tmp_path / "previous"
    old_deb = previous / "apt/pool/main/q/quoth/quoth_0.0.9_amd64.deb"
    old_object = previous / "flatpak/repo/objects/aa/old"
    old_deb.parent.mkdir(parents=True)
    old_object.parent.mkdir(parents=True)
    old_deb.write_bytes(b"old deb")
    old_object.write_bytes(b"old object")
    (previous / "flatpak/repo/config").write_text("[core]\nmode=archive-z2\n")
    stale_release = previous / "apt/dists/stable/Release"
    stale_release.parent.mkdir(parents=True)
    stale_release.write_text("stale release\n")
    stale_release.with_name("InRelease").write_text("stale signature\n")
    stale_release.with_name("Release.gpg").write_text("stale signature\n")
    (previous / "CNAME").write_text("packages.example.test\n")
    (previous / ".nojekyll").write_bytes(b"")
    _complete(previous, artifacts, "0.0.0~rehearsal1")
    commit = "c" * 64
    commands: list[list[str]] = []

    def run(command: list[str], cwd: Path, env=None) -> bytes:
        commands.append(command)
        if command[:2] == ["apt-ftparchive", "packages"]:
            return (
                b"Package: quoth\nVersion: 0.0.9\nArchitecture: amd64\n"
                b"Filename: pool/main/q/quoth/quoth_0.0.9_amd64.deb\n\n"
                b"Package: quoth\nVersion: 0.1.0\nArchitecture: amd64\n"
                b"Filename: pool/main/q/quoth/quoth_0.1.0_amd64.deb\n\n"
            )
        if command[0] == "apt-ftparchive":
            release = output / "apt/dists/stable/Release"
            assert not release.exists()
            assert not release.with_name("InRelease").exists()
            assert not release.with_name("Release.gpg").exists()
            return b"Suite: stable\nCodename: stable\n"
        if command[0] == "gpg" and "--export" in command:
            return b"public key"
        if command[0] == "gpg" and "--verify" in command:
            return _valid_signature()
        if command[0] == "gpg":
            Path(command[command.index("--output") + 1]).write_bytes(b"signature")
            return b""
        if "config" in command and "core.mode" in command:
            return b"archive-z2\n"
        if command[:2] == ["flatpak", "build-import-bundle"]:
            return f"Importing {repositories.FLATPAK_REF} ({commit})\n".encode()
        if "rev-parse" in command:
            return f"{commit}\n".encode()
        return b""

    monkeypatch.setattr(repositories, "_run", run)
    result = repositories.prepare(
        artifacts,
        output,
        "https://packages.example.test/releases",
        _FINGERPRINT,
        previous,
        "packages.example.test",
    )

    assert (
        output / "apt/pool/main/q/quoth/quoth_0.1.0_amd64.deb"
    ).read_bytes() == payloads["deb"]
    assert not (output / "downloads").exists()
    assert (artifacts / "deb/quoth_0.1.0_amd64.deb").read_bytes() == payloads["deb"]
    assert (artifacts / "flatpak/quoth.flatpak").read_bytes() == payloads["flatpak"]
    assert (
        artifacts / "arch/quoth-0.1.0-1-x86_64.pkg.tar.zst"
    ).read_bytes() == payloads["arch"]
    assert (output / old_deb.relative_to(previous)).read_bytes() == b"old deb"
    assert (output / old_object.relative_to(previous)).read_bytes() == b"old object"
    assert (output / "COMPLETE").is_file()
    assert result["flatpak_commit"] == commit
    imports = [c[-1] for c in commands if c[:2] == ["flatpak", "build-import-bundle"]]
    assert imports == [str(artifacts / "flatpak/quoth.flatpak")]
    directions = (output / "index.html").read_text()
    assert f"{repositories.SOURCE_RELEASES}/v0.1.0" in directions
    assert "quoth-sources.flatpak" in directions
    assert "quoth-deb-sources.tar.gz" in directions
    assert json.loads((output / "preparation.json").read_text()) == result
    assert (
        "Signed-By: /etc/apt/keyrings/quoth.gpg"
        in (output / "quoth.sources").read_text()
    )
    assert "Branch=master" in (output / "quoth.flatpakref").read_text()
    assert (output / "CNAME").read_text() == "packages.example.test\n"
    assert (output / ".nojekyll").read_bytes() == b""
    manifest = json.loads((output / "site-manifest.json").read_text())
    assert manifest["CNAME"] == hashlib.sha256(b"packages.example.test\n").hexdigest()
    assert manifest[".nojekyll"] == hashlib.sha256(b"").hexdigest()
    assert "index.html" in manifest
    assert any(
        command[:2] == ["flatpak", "build-import-bundle"] for command in commands
    )
    assert not any("build-commit-from" in command for command in commands)
    assert str(tmp_path / "gnupg") not in (output / "preparation.json").read_text()


def _first_site_run(imported_ref: str, object_size: int = 0):
    """Fake the tools for a first preparation whose import stores one object."""
    commit = "c" * 64

    def run(command: list[str], cwd: Path, env=None) -> bytes:
        if command[:2] == ["apt-ftparchive", "packages"]:
            return (
                b"Package: quoth\nVersion: 0.1.0\nArchitecture: amd64\n"
                b"Filename: pool/main/q/quoth/quoth_0.1.0_amd64.deb\n\n"
            )
        if command[0] == "gpg" and "--export" in command:
            return b"public key"
        if command[0] == "gpg" and "--output" in command:
            Path(command[command.index("--output") + 1]).write_bytes(b"signature")
        if command[:2] == ["flatpak", "build-import-bundle"]:
            stored = Path(command[-2]) / "objects/c1/object.filez"
            stored.parent.mkdir(parents=True)
            with stored.open("wb") as stream:
                stream.truncate(object_size)
            return f"Importing {imported_ref} ({commit})\n".encode()
        if "rev-parse" in command:
            return f"{commit}\n".encode()
        return b""

    return run


def test_rejects_a_flatpak_bundle_that_is_not_the_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifacts, _ = _artifacts(tmp_path)
    _environment(tmp_path, monkeypatch)
    output = tmp_path / "site"
    # The sources runtime where the app belongs.
    sources = f"runtime/{repositories.APP_ID}.Sources/x86_64/master"
    monkeypatch.setattr(repositories, "_run", _first_site_run(sources))

    with pytest.raises(
        ValueError, match=re.escape(f"did not import {repositories.FLATPAK_REF}")
    ):
        repositories.prepare(
            artifacts, output, "https://packages.example.test", _FINGERPRINT
        )
    assert not (output / "COMPLETE").exists()


def test_refuses_a_site_file_larger_than_github_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifacts, _ = _artifacts(tmp_path)
    _environment(tmp_path, monkeypatch)
    output = tmp_path / "site"
    assert repositories.GITHUB_FILE_LIMIT == 100 * 1024 * 1024
    run = _first_site_run(repositories.FLATPAK_REF, repositories.GITHUB_FILE_LIMIT + 1)
    monkeypatch.setattr(repositories, "_run", run)

    with pytest.raises(
        ValueError, match=r"over 100 MiB: flatpak/repo/objects/c1/object\.filez\Z"
    ):
        repositories.prepare(
            artifacts, output, "https://packages.example.test", _FINGERPRINT
        )
    assert not (output / "site-manifest.json").exists()
    assert not (output / "COMPLETE").exists()


def test_accepts_a_site_file_at_github_file_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifacts, _ = _artifacts(tmp_path)
    _environment(tmp_path, monkeypatch)
    output = tmp_path / "site"
    run = _first_site_run(repositories.FLATPAK_REF, repositories.GITHUB_FILE_LIMIT)
    monkeypatch.setattr(repositories, "_run", run)

    repositories.prepare(
        artifacts, output, "https://packages.example.test", _FINGERPRINT
    )

    assert (output / "COMPLETE").is_file()
