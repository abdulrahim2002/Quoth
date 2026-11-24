# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `scripts/render_flatpak_audit_project.py`.

The last test pins the dependency-review workflow that runs it.
"""

import json
from pathlib import Path

import pytest

from scripts import render_flatpak_audit_project as audit_project

_ROOT = Path(__file__).parents[1]


def _write_manifest(path: Path, sources: str) -> Path:
    path.write_text(f"name: python3-test\nbuildsystem: simple\nsources:\n{sources}")
    return path


def _wheel_source(
    filename: str,
    *,
    source_type: str = "file",
    sha256: str | None = "a" * 64,
) -> str:
    source = (
        f"  - type: {source_type}\n"
        f"    url: https://files.pythonhosted.org/packages/00/00/{filename}\n"
    )
    if sha256 is not None:
        source += f"    sha256: {sha256}\n"
    return source


def test_real_manifests_render_nine_exact_pins_and_deduplicate_numpy(tmp_path):
    manifests = [
        _ROOT / "packaging/flatpak/python3-build-deps.yaml",
        _ROOT / "packaging/flatpak/python3-deps.yaml",
    ]

    pins = audit_project.render_audit_project(manifests, tmp_path)

    assert [(pin.name, str(pin.version)) for pin in pins] == [
        ("certifi", "2026.10.19"),
        ("hatchling", "1.30.1"),
        ("loguru", "0.7.3"),
        ("numpy", "2.4.6"),
        ("pathspec", "1.1.1"),
        ("pluggy", "1.6.0"),
        ("sentry-sdk", "2.72.0"),
        ("trove-classifiers", "2026.6.1.19"),
        ("urllib3", "2.8.0"),
    ]
    numpy = next(pin for pin in pins if pin.name == "numpy")
    assert len(numpy.sources) == 2
    assert {source.only_arches for source in numpy.sources} == {
        ("aarch64",),
        ("x86_64",),
    }

    assert (tmp_path / "pyproject.toml").read_text() == (
        "[project]\n"
        'name = "quoth-flatpak-python-audit"\n'
        'version = "0"\n'
        'requires-python = ">=3.13,<3.14"\n'
        "dependencies = [\n"
        '    "certifi==2026.10.19",\n'
        '    "hatchling==1.30.1",\n'
        '    "loguru==0.7.3",\n'
        '    "numpy==2.4.6",\n'
        '    "pathspec==1.1.1",\n'
        '    "pluggy==1.6.0",\n'
        '    "sentry-sdk==2.72.0",\n'
        '    "trove-classifiers==2026.6.1.19",\n'
        '    "urllib3==2.8.0",\n'
        "]\n"
        "\n"
        "[tool.uv]\n"
        "environments = [\"sys_platform == 'linux'\"]\n"
    )
    inputs = json.loads((tmp_path / "flatpak-wheel-inputs.json").read_text())
    assert inputs["python"] == "3.13"
    assert inputs["platform"] == "linux"
    assert (tmp_path / ".python-version").read_text() == "3.13\n"
    assert len(inputs["wheels"]) == 10
    assert sum(item["name"] == "numpy" for item in inputs["wheels"]) == 2


def test_manifest_count_is_exactly_two(tmp_path):
    manifest = _write_manifest(
        tmp_path / "one.yaml", _wheel_source("demo-1.0-py3-none-any.whl")
    )

    with pytest.raises(audit_project.ManifestError, match="exactly two"):
        audit_project.collect_wheel_pins([manifest])


def test_manifests_must_be_distinct_paths(tmp_path):
    manifest = _write_manifest(
        tmp_path / "manifest.yaml", _wheel_source("demo-1.0-py3-none-any.whl")
    )

    with pytest.raises(audit_project.ManifestError, match="distinct"):
        audit_project.collect_wheel_pins([manifest, manifest])


def test_malformed_manifest_is_rejected(tmp_path):
    first = tmp_path / "first.yaml"
    first.write_text("sources: [\n")
    second = _write_manifest(
        tmp_path / "second.yaml", _wheel_source("demo-1.0-py3-none-any.whl")
    )

    with pytest.raises(audit_project.ManifestError, match="valid YAML"):
        audit_project.collect_wheel_pins([first, second])


def test_each_manifest_must_contain_a_wheel(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("name: empty\nsources: []\n")
    populated = _write_manifest(
        tmp_path / "populated.yaml", _wheel_source("demo-1.0-py3-none-any.whl")
    )

    with pytest.raises(audit_project.ManifestError, match=r"empty.yaml.*wheel sources"):
        audit_project.collect_wheel_pins([populated, empty])


def test_two_empty_manifests_are_rejected(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("name: first\nsources: []\n")
    second.write_text("name: second\nsources: []\n")

    with pytest.raises(audit_project.ManifestError, match=r"first.yaml.*wheel sources"):
        audit_project.collect_wheel_pins([first, second])


@pytest.mark.parametrize(
    ("sources", "message"),
    [
        (_wheel_source("demo-1.0.tar.gz"), "wheel filename"),
        (
            _wheel_source("demo-1.0-py3-none-any.whl", source_type="archive"),
            "source type",
        ),
        (_wheel_source("demo-1.0-py3-none-any.whl", sha256=None), "sha256"),
        (_wheel_source("demo-1.0-py3-none-any.whl", sha256="not-a-hash"), "sha256"),
    ],
)
def test_unexpected_wheel_source_is_rejected(tmp_path, sources, message):
    first = _write_manifest(tmp_path / "first.yaml", sources)
    second = _write_manifest(
        tmp_path / "second.yaml", _wheel_source("other-1.0-py3-none-any.whl")
    )

    with pytest.raises(audit_project.ManifestError, match=message):
        audit_project.collect_wheel_pins([first, second])


def test_conflicting_versions_are_rejected(tmp_path):
    first = _write_manifest(
        tmp_path / "first.yaml", _wheel_source("demo-1.0-py3-none-any.whl")
    )
    second = _write_manifest(
        tmp_path / "second.yaml", _wheel_source("demo-2.0-py3-none-any.whl")
    )

    with pytest.raises(audit_project.ManifestError, match="conflicting versions"):
        audit_project.collect_wheel_pins([first, second])


def test_dependency_workflow_covers_all_locked_and_bundled_graphs():
    workflow = (_ROOT / ".github/workflows/dependency-review.yml").read_text()

    assert "cron: '23 6 * * 1'" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "defaults:\n  run:\n    shell: bash" in workflow
    assert "uv audit --locked 2>&1" in workflow
    assert "uv audit --locked --project lab" in workflow
    assert "--ignore" not in workflow
    assert "uv run --locked --only-group dev python" in workflow
    assert "scripts/render_flatpak_audit_project.py" in workflow
    assert "packaging/flatpak/python3-build-deps.yaml" in workflow
    assert "packaging/flatpak/python3-deps.yaml" in workflow
    version = '"$(cat "$audit_dir/.python-version")"'
    assert f"uv lock --python {version}" in workflow
    assert (
        f"uv audit --locked --python-version {version} \\\n"
        "            --python-platform linux" in workflow
    )
    assert "timeout-minutes: 10" in workflow
