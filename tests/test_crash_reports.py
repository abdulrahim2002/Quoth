# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `crash_reports`.

Reports stay off unless asked for, and carry only exceptions when on.
"""

import getpass
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quoth import __version__, crash_reports, log


@pytest.fixture(autouse=True)
def _fresh_run(monkeypatch):
    monkeypatch.setattr(crash_reports, "_reported", set())
    crash_reports._user_directory.cache_clear()


def test_context_tag_distinguishes_requested_device_from_observed_offload(monkeypatch):
    monkeypatch.setattr(crash_reports, "_device", "NPU")
    monkeypatch.setattr(crash_reports, "_openvino_active", None)
    monkeypatch.setattr(crash_reports, "_tags", lambda: {})

    def tags():
        crash_reports._reported.clear()
        return crash_reports._prepare(
            {"exception": {"values": [{"type": "RuntimeError"}]}}, {}
        )["tags"]

    assert tags() == {"device": "NPU"}
    crash_reports.note_context(False)
    assert tags() == {"device": "NPU", "openvino_active": "false"}
    crash_reports.note_context(True)
    assert tags() == {"device": "NPU", "openvino_active": "true"}
    crash_reports.note_context(None)
    assert tags() == {"device": "NPU"}


@pytest.mark.parametrize(
    ("release", "environment"),
    [
        (f"quoth@{__version__}", "production"),
        (f"quoth@{__version__}+g{'a' * 40}", "candidate"),
        (f"quoth@{__version__}+g{'b' * 40}", "development"),
    ],
)
def test_build_identity_accepts_release_formats(
    release, environment, tmp_path, monkeypatch
):
    identity = tmp_path / "build-identity.json"
    identity.write_text(json.dumps({"release": release, "environment": environment}))
    monkeypatch.setattr(crash_reports, "_IDENTITY_PATH", identity)
    crash_reports._identity.cache_clear()

    assert crash_reports._identity() == (release, environment)


def test_build_identity_is_exact_and_read_once(tmp_path, monkeypatch):
    identity = tmp_path / "build-identity.json"
    candidate = {
        "release": f"quoth@{__version__}+g{'a' * 40}",
        "environment": "candidate",
    }
    identity.write_text(json.dumps(candidate))
    monkeypatch.setattr(crash_reports, "_IDENTITY_PATH", identity)
    crash_reports._identity.cache_clear()

    assert crash_reports._identity() == tuple(candidate.values())
    identity.write_text("not json")
    assert crash_reports._identity() == tuple(candidate.values())


def test_absent_build_identity_uses_development_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(crash_reports, "_IDENTITY_PATH", tmp_path / "missing.json")
    crash_reports._identity.cache_clear()

    assert crash_reports._identity() == (
        f"quoth@{__version__}+dev",
        "development",
    )


@pytest.mark.parametrize(
    "contents",
    [
        "not json",
        json.dumps({"release": f"quoth@{__version__}", "environment": "candidate"}),
        json.dumps(
            {
                "release": f"quoth@{__version__}",
                "environment": "production",
                "extra": True,
            }
        ),
    ],
)
def test_malformed_build_identity_falls_back(contents, tmp_path, monkeypatch):
    identity = tmp_path / "build-identity.json"
    identity.write_text(contents)
    monkeypatch.setattr(crash_reports, "_IDENTITY_PATH", identity)
    crash_reports._identity.cache_clear()

    assert crash_reports._identity() == (
        f"quoth@{__version__}+dev",
        "development",
    )


def test_colocated_vendor_path_is_added_once(tmp_path, monkeypatch):
    vendor = tmp_path / "_vendor"
    vendor.mkdir()
    monkeypatch.setattr(crash_reports, "_VENDOR_DIR", vendor)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    crash_reports._add_vendor_path.cache_clear()

    crash_reports._add_vendor_path()
    crash_reports._add_vendor_path()

    assert sys.path.count(str(vendor)) == 1


def test_sdk_lifecycle_and_privacy(tmp_path):
    env = os.environ.copy()
    # The package sits under HOME, as in a source checkout, so stack frames carry the
    # home path that must not reach a report.
    env["HOME"] = str(Path(crash_reports.__file__).parents[1])
    env["XDG_STATE_HOME"] = str(tmp_path)
    env["SENTRY_SPOTLIGHT"] = "http://127.0.0.1:65535/stream"
    result = subprocess.run(
        [sys.executable, Path(__file__).parent / "fixtures" / "crash_privacy_probe.py"],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr


def test_a_startup_without_consent_never_loads_the_sdk(tmp_path):
    # The import alone costs about 160 ms of startup, so it waits for consent.
    script = """
import sys
from quoth import crash_reports
crash_reports.set_enabled(False)
crash_reports.report_native_crash()
assert "sentry_sdk" not in sys.modules
"""
    env = os.environ.copy()
    env["XDG_STATE_HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr


def test_native_crash_stays_local_without_consent(tmp_path, monkeypatch):
    previous = tmp_path / "crash.prev.log"
    previous.write_text("Fatal Python error: Segmentation fault\n")
    monkeypatch.setattr(log, "previous_crash_log", previous)
    monkeypatch.setattr(crash_reports, "_enabled", False)

    crash_reports.report_native_crash()

    assert previous.exists()


def test_a_repeating_failure_is_reported_once_per_run(monkeypatch):
    monkeypatch.setattr(crash_reports, "_tags", lambda: {})

    def event(line):
        frame = {"filename": "quoth/overlay.py", "function": "_draw", "lineno": line}
        values = [{"type": "RuntimeError", "stacktrace": {"frames": [frame]}}]
        return {"exception": {"values": values}}

    assert crash_reports._prepare(event(146), {}) is not None
    assert crash_reports._prepare(event(146), {}) is None
    assert crash_reports._prepare(event(152), {}) is not None


def test_a_broken_sdk_leaves_reporting_off_and_the_app_running(monkeypatch, caplog):
    import sentry_sdk

    def broken_init(**_options):
        raise TypeError("unknown option")

    monkeypatch.setattr(crash_reports, "_enabled", False)
    monkeypatch.setattr(sentry_sdk, "init", broken_init)
    caplog.set_level("ERROR")

    crash_reports.set_enabled(True)

    assert crash_reports._enabled is False
    assert [r.levelname for r in caplog.records] == ["ERROR"]


def test_an_unreadable_native_stack_stays_local(tmp_path, monkeypatch):
    previous = tmp_path / "crash.prev.log"
    previous.mkdir()
    monkeypatch.setattr(log, "previous_crash_log", previous)
    monkeypatch.setattr(crash_reports, "_enabled", True)

    crash_reports.report_native_crash()

    assert previous.exists()


def test_a_path_outside_home_never_names_the_user(monkeypatch):
    # A checkout or venv on another drive, in a directory named for the user or with
    # the name as a word in it.
    monkeypatch.setattr(getpass, "getuser", lambda: "alice")
    frames = [
        "/run/media/alice/src/alice/quoth/app.py",
        "/mnt/alice-projects/quoth/app.py",
        "/home/alice.work/app.py",
        "/opt/malice/x.py",
        "/opt/alice2/alice_x/x.py",
    ]

    assert crash_reports._scrub({"frames": frames, "function": "alice"}) == {
        "frames": [
            "/run/media/<user>/src/<user>/quoth/app.py",
            "/mnt/<user>-projects/quoth/app.py",
            "/home/<user>.work/app.py",
            "/opt/malice/x.py",
            "/opt/alice2/alice_x/x.py",
        ],
        "function": "alice",
    }


def test_each_report_leaves_a_local_trace_without_its_contents(monkeypatch, caplog):
    monkeypatch.setattr(crash_reports, "_tags", lambda: {})
    caplog.set_level("DEBUG")

    def event():
        return {"exception": {"values": [{"type": "RuntimeError", "value": "private"}]}}

    crash_reports._prepare(event(), {})
    crash_reports._prepare(event(), {})
    crash_reports._prepare({"message": "private"}, {})
    crash_reports._options()["transport"].on_dropped_event(None, "network")

    assert [r.getMessage() for r in caplog.records] == [
        "Error report queued",
        "Error report dropped: a repeat this run",
        "Error report dropped: not an exception",
        "Error report not sent (network)",
    ]
    assert {r.levelname for r in caplog.records} == {"DEBUG"}
