# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `scripts/check_package_run.py`.

Promotion accepts only fully green, tested package runs.
"""

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts import check_package_run
from scripts.check_package_run import check_run


@pytest.fixture
def evidence():
    run = {
        "event": "workflow_dispatch",
        "path": ".github/workflows/package.yml",
        "head_repository": {"full_name": "owner/quoth"},
        "status": "completed",
        "head_sha": "a" * 40,
        "repository": {"private": True},
    }
    names = [
        "Resolve package identity",
        "Build Ubuntu 24.04 package",
        "Build Arch package",
        "Install Debian package and transcribe on CPU",
        "Install Arch package and transcribe on CPU",
        "Install Flatpak bundle and transcribe on CPU",
    ]
    jobs = [{"name": name, "conclusion": "success"} for name in names]
    # Only a production run in a public repository attests.
    jobs.append({"name": "Attest production packages", "conclusion": "skipped"})
    jobs.append(
        {
            "name": "Build Flatpak bundle and lint manifest",
            "conclusion": "success",
            "steps": [
                {"name": "Build and export", "conclusion": "success"},
                {
                    "name": "Lint Flatpak manifest",
                    "conclusion": "success",
                },
                {"name": "Run actions/upload-artifact", "conclusion": "success"},
                {"name": "Run actions/upload-artifact", "conclusion": "success"},
            ],
        }
    )
    checks = [
        dict(run, path=path, id=number, conclusion="success")
        for number, path in enumerate(
            [".github/workflows/ci.yml", ".github/workflows/dependency-review.yml"]
        )
    ]
    return run, jobs, checks, "owner/quoth", "a" * 40, "staging"


def test_green_package_run_permits_private_preparation(evidence):
    check_run(*evidence)


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "skipped"])
@pytest.mark.parametrize("job", [0, 1, 2, 3, 4, 5, -1])
def test_an_unsuccessful_required_job_refuses_preparation(evidence, job, conclusion):
    evidence[1][job]["conclusion"] = conclusion
    with pytest.raises(ValueError, match="Required package build"):
        check_run(*evidence)


@pytest.mark.parametrize("conclusion", [None, "skipped"])
def test_missing_or_skipped_manifest_lint_refuses_private_preparation(
    evidence, conclusion
):
    if conclusion is None:
        evidence[1][-1]["steps"].pop(1)
    else:
        evidence[1][-1]["steps"][1]["conclusion"] = conclusion
    with pytest.raises(ValueError, match="manifest lint"):
        check_run(*evidence)


@pytest.mark.parametrize("missing", range(8))
def test_missing_required_job_refuses_preparation(evidence, missing):
    evidence[1].pop(missing)
    with pytest.raises(ValueError, match="Required package build"):
        check_run(*evidence)


def test_duplicate_required_flatpak_step_refuses_preparation(evidence):
    evidence[1][-1]["steps"].append(deepcopy(evidence[1][-1]["steps"][0]))
    with pytest.raises(ValueError, match="manifest lint"):
        check_run(*evidence)


def test_cancelled_step_is_not_a_manifest_lint_failure(evidence):
    evidence[1][-1]["steps"].append({"name": "Upload", "conclusion": "cancelled"})
    with pytest.raises(ValueError, match="Flatpak build or manifest lint"):
        check_run(*evidence)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event", "pull_request"),
        ("path", ".github/workflows/ci.yml"),
        ("head_repository", {"full_name": "fork/quoth"}),
        ("status", "in_progress"),
    ],
)
def test_untrusted_or_incomplete_run_refuses_preparation(evidence, field, value):
    evidence[0][field] = value
    with pytest.raises(ValueError, match="Select a completed manual package run"):
        check_run(*evidence)


def test_newer_failed_source_check_cannot_reuse_older_success(evidence):
    failed = dict(evidence[2][0], id=100, conclusion="failure")
    evidence[2].append(failed)
    with pytest.raises(ValueError, match="Latest source check"):
        check_run(*evidence)


def test_source_check_must_match_package_commit(evidence):
    evidence[2][0]["head_sha"] = "b" * 40
    with pytest.raises(ValueError, match="Latest source check"):
        check_run(*evidence)


def test_source_check_must_come_from_the_package_repository(evidence):
    evidence[2][0]["head_repository"] = {"full_name": "fork/quoth"}
    with pytest.raises(ValueError, match="Latest source check"):
        check_run(*evidence)


def test_package_run_must_match_reviewed_commit(evidence):
    with pytest.raises(ValueError, match="reviewed source"):
        check_run(*evidence[:4], "b" * 40, evidence[5])


def _public_production(evidence):
    run, jobs, checks, repository, source_commit, _target = evidence
    run["repository"] = {"private": False}
    return run, jobs, checks, repository, source_commit, "production"


def test_public_production_run_with_attested_packages_permits_preparation(evidence):
    evidence = _public_production(evidence)
    evidence[1][6]["conclusion"] = "success"
    check_run(*evidence)


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "skipped", None])
def test_public_production_run_must_have_attested_its_packages(evidence, conclusion):
    evidence = _public_production(evidence)
    evidence[1][6]["conclusion"] = conclusion
    with pytest.raises(ValueError, match="Attestation must run"):
        check_run(*evidence)


@pytest.mark.parametrize(
    ("target", "private"),
    [("staging", True), ("staging", False), ("production", True)],
)
@pytest.mark.parametrize("conclusion", ["success", "skipped", "failure", None])
def test_only_public_production_runs_attest(evidence, target, private, conclusion):
    evidence[0]["repository"] = {"private": private}
    evidence[1][6]["conclusion"] = conclusion
    evidence = (*evidence[:5], target)
    if conclusion == "skipped":
        check_run(*evidence)
    else:
        with pytest.raises(ValueError, match="Attestation must run"):
            check_run(*evidence)


def test_unexpected_or_duplicate_jobs_fail_closed(evidence):
    duplicate = deepcopy(evidence[1][0])
    evidence[1].append(duplicate)
    with pytest.raises(ValueError, match="Required package build"):
        check_run(*evidence)
    duplicate["name"] = "New required release check"
    with pytest.raises(ValueError, match="Required package build"):
        check_run(*evidence)


_FAKE_GH = """\
import json, pathlib, sys

state = json.loads(pathlib.Path(sys.argv[0]).with_name("state.json").read_text())
args = sys.argv[1:]
if args[0] == "api":
    print(json.dumps(state["api"][args[1]]))
else:
    assert args[:2] == ["run", "download"], args
    name = args[args.index("--name") + 1]
    directory = pathlib.Path(args[args.index("--dir") + 1])
    directory.mkdir(parents=True)
    channel = name.split("-")[0]
    commit = state["commits"].get(channel, state["head_sha"])
    (directory / "source-commit.txt").write_text(commit + "\\n")
    target = state["targets"].get(channel, state["target"])
    (directory / "release-metadata.json").write_text(json.dumps({"target": target}))
    with open(pathlib.Path(sys.argv[0]).with_name("downloads.log"), "a") as log:
        log.write(name + "\\n")
"""


def _promote(tmp_path, monkeypatch, evidence, commits, targets=None):
    """Run the script's command line against a fake `gh` serving `evidence`."""
    run, jobs, checks, repository, source_commit, target = evidence
    run = dict(run, run_attempt=2)
    prefix = f"repos/{repository}/actions"
    state = {
        "api": {
            f"{prefix}/runs/7": run,
            f"{prefix}/runs/7/attempts/2/jobs?per_page=100": {
                "total_count": len(jobs),
                "jobs": jobs,
            },
            f"{prefix}/runs?head_sha={run['head_sha']}&per_page=100": {
                "total_count": len(checks),
                "workflow_runs": checks,
            },
        },
        "head_sha": run["head_sha"],
        "commits": commits,
        "target": target,
        "targets": targets or {},
    }
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "state.json").write_text(json.dumps(state))
    gh = bin_dir / "gh"
    gh.write_text(f"#!{sys.executable}\n{_FAKE_GH}")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_package_run.py",
            "--repository",
            repository,
            "--run-id",
            "7",
            "--source-commit",
            source_commit,
            "--output",
            str(tmp_path / "package-run.json"),
            "--artifacts",
            str(tmp_path / "artifacts"),
        ],
    )
    check_package_run.main()
    return (bin_dir / "downloads.log").read_text().split()


def test_promotion_downloads_each_channel_of_the_checked_attempt(
    tmp_path, monkeypatch, evidence
):
    downloads = _promote(tmp_path, monkeypatch, evidence, {})

    assert downloads == ["deb-7-2", "arch-7-2", "flatpak-7-2"]
    for channel in ("deb", "arch", "flatpak"):
        assert (tmp_path / "artifacts" / channel / "source-commit.txt").exists()
    saved = json.loads((tmp_path / "package-run.json").read_text())
    assert saved["run"]["run_attempt"] == 2


def test_promotion_refuses_an_artifact_from_another_commit(
    tmp_path, monkeypatch, evidence
):
    with pytest.raises(ValueError, match="arch artifact was built from another"):
        _promote(tmp_path, monkeypatch, evidence, {"arch": "b" * 40})


@pytest.mark.parametrize("attested", [True, False])
def test_promotion_takes_the_target_from_the_deb_artifact(
    tmp_path, monkeypatch, evidence, attested
):
    # A public run whose deb names production must have attested its packages.
    evidence = _public_production(evidence)
    if attested:
        evidence[1][6]["conclusion"] = "success"
        _promote(tmp_path, monkeypatch, evidence, {})
        assert (tmp_path / "package-run.json").exists()
    else:
        with pytest.raises(ValueError, match="Attestation must run"):
            _promote(tmp_path, monkeypatch, evidence, {})
        assert not (tmp_path / "package-run.json").exists()


def test_promotion_takes_the_target_from_the_deb_and_no_other_channel(
    tmp_path, monkeypatch, evidence
):
    # The deb says production, so this public run must have attested its packages.
    evidence = _public_production(evidence)
    targets = {"arch": "staging", "flatpak": "staging"}
    with pytest.raises(ValueError, match="Attestation must run"):
        _promote(tmp_path, monkeypatch, evidence, {}, targets)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event", "push"),
        ("path", ".github/workflows/ci.yml"),
        ("head_repository", {"full_name": "fork/quoth"}),
        ("status", "in_progress"),
        ("head_sha", "b" * 40),
    ],
)
def test_promotion_refuses_another_run_before_downloading_from_it(
    tmp_path, monkeypatch, evidence, field, value
):
    evidence[0][field] = value
    with pytest.raises(ValueError, match="Select a completed manual package run"):
        _promote(tmp_path, monkeypatch, evidence, {})
    assert not (tmp_path / "bin/downloads.log").exists()
    assert not (tmp_path / "artifacts").exists()


def test_the_evidence_is_what_package_yml_runs(evidence):
    # A job or step renamed there must be renamed in the script, or no run promotes.
    path = Path(__file__).resolve().parents[1] / ".github/workflows/package.yml"
    workflow = yaml.safe_load(path.read_text())
    jobs = {job["name"]: job for job in workflow["jobs"].values()}
    assert sorted(jobs) == sorted(job["name"] for job in evidence[1])
    flatpak = evidence[1][-1]
    steps = {step.get("name") for step in jobs[flatpak["name"]]["steps"]}
    assert {step["name"] for step in flatpak["steps"][:2]} <= steps
    # The script expects attestation exactly on the runs this condition admits.
    assert jobs["Attest production packages"]["if"] == (
        "needs.identity.outputs.target == 'production'"
        " && !github.event.repository.private"
    )
