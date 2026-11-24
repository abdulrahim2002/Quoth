# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `scripts/build_release.py`."""

import hashlib
import io
import json
import subprocess
from pathlib import Path

import pytest

from scripts import build_release

_ROOT = Path(__file__).parents[1]


def _fake_gh(monkeypatch, view_returncode=0, is_draft="true"):
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
        commands.append(command)
        if command[:3] == ["gh", "release", "view"]:
            return subprocess.CompletedProcess(
                command, view_returncode, stdout=f"{is_draft}\n", stderr="not found"
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(build_release.subprocess, "run", run)
    return commands


def test_upload_does_not_overwrite_existing_release_assets(monkeypatch, tmp_path):
    commands = _fake_gh(monkeypatch)
    manifest = tmp_path / "manifest.json"
    asset = tmp_path / "model.bin"

    build_release.upload("models-v1", manifest, [asset])

    assert commands[1] == [
        "gh",
        "release",
        "upload",
        "models-v1",
        str(manifest),
        str(asset),
    ]


@pytest.mark.parametrize(
    ("view_returncode", "is_draft", "message"),
    [(1, "", "create the release as a draft"), (0, "false", "already published")],
)
def test_upload_needs_a_draft_release(
    monkeypatch, tmp_path, view_returncode, is_draft, message
):
    commands = _fake_gh(monkeypatch, view_returncode, is_draft)

    with pytest.raises(SystemExit, match=message):
        build_release.upload("v0.1.0", tmp_path / "manifest.json", [])

    assert [command[2] for command in commands] == ["view"]


_WEIGHTS = [
    "ggml-tiny-q8_0.bin",
    "ggml-base-q8_0.bin",
    "ggml-small-q8_0.bin",
    "ggml-large-v3-turbo-q8_0.bin",
]


def _staged(source: Path) -> tuple[dict[str, str], list[dict]]:
    """Stage each weight; return its manifest and the matching upstream entries."""
    manifest: dict[str, str] = {}
    entries: list[dict] = []
    for name in _WEIGHTS:
        payload = f"{name} weights".encode()
        (source / name).write_bytes(payload)
        manifest[name] = hashlib.sha256(payload).hexdigest()
        lfs = {"oid": manifest[name], "size": len(payload)}
        entries.append({"type": "file", "path": name, "lfs": lfs})
    return manifest, entries


def _paths_info(entries: list[dict], requests: list):
    def opener(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps(entries).encode())

    return opener


def test_verify_upstream_asks_for_every_weight_at_the_pinned_revision(tmp_path):
    manifest, entries = _staged(tmp_path)
    requests: list = []

    build_release.verify_upstream(manifest, tmp_path, _paths_info(entries, requests))

    [request] = requests
    assert request.get_method() == "POST"
    assert request.full_url == (
        "https://huggingface.co/api/models/ggerganov/whisper.cpp/paths-info/"
        f"{build_release.UPSTREAM_REVISION}"
    )
    assert json.loads(request.data) == {"paths": _WEIGHTS}


def test_verify_upstream_lists_every_missing_or_differing_weight(tmp_path):
    manifest, entries = _staged(tmp_path)
    entries[0]["lfs"]["oid"] = "0" * 64
    entries[1]["lfs"]["size"] += 1
    del entries[2]

    with pytest.raises(SystemExit) as failure:
        build_release.verify_upstream(manifest, tmp_path, _paths_info(entries, []))

    message = str(failure.value)
    assert f"ggml-tiny-q8_0.bin: sha256 {manifest['ggml-tiny-q8_0.bin']}" in message
    assert f"upstream has sha256 {'0' * 64}" in message
    assert "ggml-base-q8_0.bin: sha256" in message
    assert "ggml-small-q8_0.bin: not found upstream" in message
    assert "ggml-large-v3-turbo-q8_0.bin" not in message


def test_ci_fixture_model_comes_from_the_release_revision():
    workflow = (_ROOT / ".github/workflows/package.yml").read_text()

    assert (
        f"huggingface.co/{build_release.UPSTREAM_REPO}/resolve/"
        f"{build_release.UPSTREAM_REVISION}/ggml-base-q8_0.bin"
    ) in workflow
