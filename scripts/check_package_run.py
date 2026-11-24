#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Require a tested, same-repository manual package run before promotion.

It downloads the run's deb, arch and flatpak artifacts into --artifacts, one directory
each, built from the run's commit, and on success writes the run's evidence to --output.
"""

import argparse
import json
import subprocess
from pathlib import Path

_ATTEST = "Attest production packages"


def check_selected(run: dict, repository: str, source_commit: str) -> None:
    """Raise ValueError unless `run` is a completed manual package run of the commit.

    Checked before anything is downloaded from the run.
    """
    if (
        run["event"] != "workflow_dispatch"
        or run["path"] != ".github/workflows/package.yml"
        or run["head_repository"]["full_name"] != repository
        or run["status"] != "completed"
        or run["head_sha"] != source_commit
    ):
        raise ValueError(
            "Select a completed manual package run for the reviewed source commit"
        )


def check_run(
    run: dict,
    jobs: list[dict],
    checks: list[dict],
    repository: str,
    source_commit: str,
    target: str,
) -> None:
    """Raise ValueError unless `run` is a passing manual package run of the commit.

    The run must be a completed `package.yml` dispatch from `repository` at
    `source_commit`, every required job and Flatpak step must succeed, and the latest
    CI and dependency-review runs for that commit must pass. `target` is the run's
    package target. Only a production run in a public repository attests, so its
    attestation job must succeed there and have been skipped everywhere else.
    """
    check_selected(run, repository, source_commit)

    required = {
        "Resolve package identity",
        "Build Ubuntu 24.04 package",
        "Build Arch package",
        "Build Flatpak bundle and lint manifest",
        "Install Debian package and transcribe on CPU",
        "Install Arch package and transcribe on CPU",
        "Install Flatpak bundle and transcribe on CPU",
        _ATTEST,
    }
    by_name = {job["name"]: job for job in jobs}
    if (
        set(by_name) != required
        or len(by_name) != len(jobs)
        or any(
            by_name.get(name, {}).get("conclusion") != "success"
            for name in required - {_ATTEST}
        )
    ):
        raise ValueError(
            "Required package build or installed-package check did not pass"
        )
    attests = target == "production" and not run["repository"]["private"]
    if by_name[_ATTEST].get("conclusion") != ("success" if attests else "skipped"):
        raise ValueError("Attestation must run exactly on public production runs")
    flatpak = by_name.get("Build Flatpak bundle and lint manifest", {})
    steps = flatpak.get("steps", [])
    required_steps = {"Build and export", "Lint Flatpak manifest"}
    if any(
        [step["conclusion"] for step in steps if step["name"] == name] != ["success"]
        for name in required_steps
    ) or any(step["conclusion"] not in {"success", "skipped"} for step in steps):
        raise ValueError("Flatpak build or manifest lint did not pass")

    for path in (".github/workflows/ci.yml", ".github/workflows/dependency-review.yml"):
        matching = [
            check
            for check in checks
            if check["path"] == path
            and check["head_sha"] == run["head_sha"]
            and check["head_repository"]["full_name"] == repository
        ]
        latest = max(matching, key=lambda check: check["id"], default={})
        if latest.get("conclusion") != "success":
            raise ValueError(f"Latest source check did not pass: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    args = parser.parse_args()

    def api(path: str) -> dict:
        return json.loads(subprocess.check_output(["gh", "api", path], text=True))

    prefix = f"repos/{args.repository}/actions"
    run = api(f"{prefix}/runs/{args.run_id}")
    check_selected(run, args.repository, args.source_commit)
    jobs = api(
        f"{prefix}/runs/{args.run_id}/attempts/{run['run_attempt']}/jobs?per_page=100"
    )
    checks = api(f"{prefix}/runs?head_sha={run['head_sha']}&per_page=100")
    if jobs["total_count"] > 100 or checks["total_count"] > 100:
        raise ValueError(
            "Run history exceeds the bounded query; inspect this run manually"
        )
    # Artifact names carry the attempt, so these are the files of the attempt checked,
    # and the deb's release metadata names the target that attestation depends on.
    for channel in ("deb", "arch", "flatpak"):
        directory = args.artifacts / channel
        subprocess.run(
            [
                "gh",
                "run",
                "download",
                str(args.run_id),
                "--repo",
                args.repository,
                "--name",
                f"{channel}-{args.run_id}-{run['run_attempt']}",
                "--dir",
                str(directory),
            ],
            check=True,
        )
        commit = (directory / "source-commit.txt").read_text().rstrip("\n")
        if commit != run["head_sha"]:
            raise ValueError(f"The {channel} artifact was built from another commit")
    metadata = json.loads((args.artifacts / "deb/release-metadata.json").read_text())
    check_run(
        run,
        jobs["jobs"],
        checks["workflow_runs"],
        args.repository,
        args.source_commit,
        metadata["target"],
    )
    args.output.write_text(
        json.dumps({"run": run, "jobs": jobs, "checks": checks}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
