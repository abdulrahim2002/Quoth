# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Prepared-image adapter for the shared desktop lab lifecycle."""

from __future__ import annotations

import json
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from lab.lifecycle import DesktopSession, command, normalize_prepared_source, wait_for


def pytest_addoption(parser):
    parser.addoption("--prepared", help="Absolute path to the prepared-image manifest")
    parser.addoption("--evidence", help="New directory for this run's evidence")


class PreparedDesktop:
    """A desktop session with prepared-image SSH and visual login checks added."""

    def __init__(self, session: DesktopSession, readiness: dict, port: int):
        self._session = session
        self.readiness = readiness
        self.port = port
        self.guest_runtime_dir = ""
        prepared = session.prepared
        ssh_config = session.evidence / "ssh_config"
        ssh_config.write_text(
            f"Host desktop\n HostName 127.0.0.1\n Port {port}\n User {prepared['user']}\n"
            f" IdentityFile {prepared['identity']}\n IdentitiesOnly yes\n BatchMode yes\n"
            f" HostKeyAlias {prepared['host_alias']}\n"
            f" UserKnownHostsFile {prepared['known_hosts']}\n"
            " StrictHostKeyChecking yes\n ConnectTimeout 3\n"
        )
        self.ssh_args = ["ssh", "-F", str(ssh_config), "desktop"]

    def __getattr__(self, name):
        return getattr(self._session, name)

    def run_guest(self, cmd, timeout=30):
        return command(
            *self.ssh_args,
            "export XDG_RUNTIME_DIR=/run/user/$(id -u); " + cmd,
            timeout=timeout,
        )

    def copy_from(self, remote):
        destination = self.evidence / Path(remote).name
        command(
            "scp",
            "-F",
            str(self.evidence / "ssh_config"),
            "desktop:" + remote,
            str(destination),
        )
        return destination

    def expect_screen(self, name):
        self.vnc.timeout = 60
        try:
            self.vnc.expectRegion(
                str(
                    Path(__file__).parent
                    / "reference"
                    / f"{self.readiness['reference_prefix']}-{name}.png"
                ),
                *self.readiness[f"{name}_coordinates"],
                maxrms=10,
            )
        finally:
            self.vnc.timeout = 10

    def type_text(self, text):
        for char in text:
            self.vnc.keyPress("enter" if char == "\n" else char)
            time.sleep(0.03)

    def _ssh_ready(self):
        try:
            result = subprocess.run(
                [*self.ssh_args, "true"], capture_output=True, timeout=8
            )
        except subprocess.TimeoutExpired as error:
            (self.evidence / "ssh-startup.log").write_text(str(error))
            return False
        (self.evidence / "ssh-startup.log").write_bytes(result.stderr)
        return result.returncode == 0

    def _login(self):
        prepared = self.prepared
        wait_for(self._ssh_ready, "SSH did not become ready")
        self.expect_screen("login")
        self.capture("login.png")
        if self.readiness["login_activate"]:
            self.vnc.keyPress("enter")
            time.sleep(1)
        self.type_text(Path(prepared["password_file"]).read_text().strip())
        self.vnc.keyPress("enter")
        wait_for(
            lambda: (
                self.run_guest(
                    "loginctl list-sessions --no-legend | awk '$3 == \""
                    + prepared["user"]
                    + "\" {print $1}' | xargs -r -n1 loginctl show-session -p Type --value"
                )
                .splitlines()
                .count("wayland")
            ),
            "Normal Wayland login did not complete",
        )
        uid = self.run_guest("id -u").strip()
        if not uid.isdecimal():
            raise RuntimeError(f"Guest returned an invalid UID: {uid!r}")
        self.guest_runtime_dir = f"/run/user/{uid}"
        self.expect_screen("desktop")
        self.capture("desktop.png")
        (self.evidence / "guest.txt").write_text(
            self.run_guest(
                "cat /etc/os-release; id; " + self.readiness["version_command"] + "; "
                "systemctl --user is-active pipewire; stat -c '%a %U %G' /dev/uinput"
            )
        )

    def boot_and_login(self):
        self._session.boot()
        self._login()

    def cold_boot(self):
        self._session.cold_boot()
        self._login()


@pytest.fixture(scope="module")
def desktop(request):
    """Yield a booted, logged-in prepared desktop; skip without `--prepared`."""
    manifest = request.config.getoption("--prepared")
    if not manifest:
        pytest.skip("requires an explicitly selected prepared desktop")
    prepared = normalize_prepared_source(manifest)
    profiles = json.loads((Path(__file__).parent / "profiles.json").read_text())
    profile = next(
        (
            candidate
            for candidate in profiles["profiles"]
            if candidate["id"] == prepared["profile"]
        ),
        None,
    )
    if profile is None:
        pytest.fail(f"Prepared profile is not declared: {prepared['profile']}")
    readiness = profile.get("readiness")
    if readiness is None:
        pytest.fail(
            f"Prepared profile has no desktop readiness configuration: {prepared['profile']}"
        )
    destination = request.config.getoption("--evidence")
    if destination:
        evidence = Path(destination)
        evidence.mkdir(mode=0o700)
    else:
        parent = Path(prepared["evidence_root"])
        parent.mkdir(parents=True, exist_ok=True)
        evidence = Path(tempfile.mkdtemp(prefix="readiness-", dir=parent))
    print(f"\nEvidence: {evidence}", flush=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with DesktopSession(
        prepared,
        evidence,
        audio=True,
        port_forwards=((port, 22),),
    ) as session:
        (evidence / "prepared.json").write_text(
            json.dumps(prepared, indent=2, default=str) + "\n"
        )
        adapter = PreparedDesktop(session, readiness, port)
        adapter.boot_and_login()
        yield adapter
