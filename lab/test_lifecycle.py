# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `lab.lifecycle`, the disposable desktop lifecycle."""

from __future__ import annotations

import hashlib
import json
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from lab import lifecycle


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path) -> dict:
    disk = tmp_path / "source.qcow2"
    nvram = tmp_path / "source_VARS.fd"
    xml = tmp_path / "source.xml"
    disk.write_bytes(b"source disk")
    nvram.write_bytes(b"source nvram")
    xml.write_text(
        "<domain><name>source</name><os><nvram>"
        f"{nvram}</nvram></os><devices><disk device='disk'><source file='{disk}'/>"
        "</disk><interface type='user'/><graphics type='vnc'><listen type='socket'/>"
        "</graphics><audio id='1' type='none'/></devices></domain>"
    )
    return {
        "kind": "prepared",
        "disk": disk,
        "nvram": nvram,
        "xml": xml,
        "disk_sha256": _sha256(disk),
        "nvram_sha256": _sha256(nvram),
        "xml_sha256": _sha256(xml),
    }


def _clone_xml(name: str, disk: Path, nvram: Path) -> str:
    return (
        f"<domain><name>{name}</name><os><nvram>{nvram}</nvram></os><devices>"
        f"<disk device='disk'><source file='{disk}'/></disk>"
        "<interface type='user'><backend type='passt'/><portForward proto='tcp'/>"
        "</interface><graphics type='vnc'><listen type='socket'/>"
        "</graphics><audio id='1' type='none'/></devices></domain>"
    )


class FakeVNC:
    def __init__(self):
        self.captures: list[str] = []
        self.disconnected = False

    def captureScreen(self, path: str) -> None:
        self.captures.append(path)

    def disconnect(self) -> None:
        self.disconnected = True


@pytest.mark.parametrize("existing", [False, True])
def test_evidence_directory_is_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    source = _source(tmp_path)
    evidence = tmp_path / "evidence"
    if existing:
        evidence.mkdir(mode=0o755)
        evidence.chmod(0o755)

    monkeypatch.setattr(lifecycle.DesktopSession, "_setup", lambda self: None)
    monkeypatch.setattr(
        lifecycle.DesktopSession, "_retire_domain", lambda self, errors: True
    )
    monkeypatch.setattr(
        lifecycle.DesktopSession,
        "_write_cleanup",
        lambda self, errors: {"errors": errors},
    )
    monkeypatch.setattr(
        lifecycle,
        "command",
        lambda *args, **kwargs: pytest.fail(f"unexpected external command: {args}"),
    )

    with lifecycle.DesktopSession(source, evidence):
        pass

    assert evidence.stat().st_mode & 0o777 == 0o700


def test_vnc_reactor_survives_session_close() -> None:
    script = textwrap.dedent(
        """
        import threading
        from types import SimpleNamespace

        from lab import lifecycle

        def dispatched():
            event = threading.Event()
            lifecycle.api.reactor.callFromThread(event.set)
            return event.wait(1)

        vnc_disconnected = threading.Event()
        lifecycle.api.connect('127.0.0.1::0', timeout=.01)
        if not dispatched():
            raise SystemExit('reactor callback before close timed out')

        session = SimpleNamespace(
            vnc=SimpleNamespace(disconnect=vnc_disconnected.set),
            stop_audio=lambda: None,
            _retire_domain=lambda errors: False,
            _write_cleanup=lambda errors: {'errors': errors},
        )
        lifecycle.DesktopSession.close(session)
        assert vnc_disconnected.is_set()
        if not dispatched():
            raise SystemExit('reactor callback after close timed out')
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr


def test_normalizes_prepared_manifest_and_registered_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    manifest = tmp_path / "prepared.json"
    manifest.write_text(
        json.dumps(
            {
                **source,
                "disk": str(source["disk"]),
                "nvram": str(source["nvram"]),
                "xml": str(source["xml"]),
                "identity": "/private/key",
            }
        )
    )

    prepared = lifecycle.normalize_prepared_source(manifest)
    assert prepared["kind"] == "prepared"
    assert prepared["disk"] == source["disk"]
    assert prepared["identity"] == Path("/private/key")

    domain_xml = source["xml"].read_text()

    def fake_command(*args, **kwargs):
        if "domstate" in args:
            return "shut off"
        if "dumpxml" in args:
            return domain_xml
        raise AssertionError(args)

    monkeypatch.setattr(lifecycle, "command", fake_command)
    registered = lifecycle.source_from_domain("source")
    assert registered == {
        "kind": "domain",
        "domain": "source",
        "disk": source["disk"],
        "nvram": source["nvram"],
        "disk_sha256": source["disk_sha256"],
        "nvram_sha256": source["nvram_sha256"],
    }
    assert "identity" not in registered


def test_stock_boot_needs_no_ssh_or_readiness_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    source_xml = source["xml"].read_text()
    source["kind"] = "domain"
    source["domain"] = "source"
    source.pop("xml")
    source.pop("xml_sha256")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    vnc = FakeVNC()
    calls: list[tuple[str, ...]] = []
    domains: set[str] = set()

    def fake_command(*args, **kwargs):
        calls.append(args)
        if args[:2] == ("qemu-img", "create"):
            Path(args[-1]).write_bytes(b"overlay")
            return ""
        if args[0] == "virt-clone":
            return _clone_xml(
                args[args.index("--name") + 1],
                Path(args[args.index("--file") + 1]),
                Path(args[args.index("--nvram") + 1]),
            )
        if "define" in args:
            domains.add(lifecycle.ET.parse(args[-1]).findtext("name"))
            return ""
        if args[-3:] == ("list", "--all", "--name"):
            return "\n".join(domains)
        if "start" in args:
            return ""
        if "dumpxml" in args:
            if args[-1] == "source":
                return source_xml
            return _clone_xml(
                next(iter(domains)), evidence / "disk.qcow2", evidence / "VARS.fd"
            ).replace(
                "<listen type='socket'/>", "<listen type='socket' socket='/vnc.sock'/>"
            )
        if "domstate" in args:
            return "shut off"
        if "undefine" in args:
            domains.clear()
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(lifecycle, "command", fake_command)
    monkeypatch.setattr(lifecycle.api, "connect", lambda *args, **kwargs: vnc)

    with lifecycle.DesktopSession(source, evidence) as desktop:
        desktop.boot()
        desktop.capture("stock.png")

    assert vnc.captures == [str(evidence / "stock.png")]
    assert not any("qemu-monitor-command" in call for call in calls)
    assert not any(call and call[0] in {"ssh", "scp"} for call in calls)
    saved = lifecycle.ET.parse(evidence / "domain.xml")
    assert saved.find("./devices/interface/backend") is None
    assert saved.find("./devices/interface/portForward") is None


@pytest.mark.parametrize("change", ["running", "remapped"])
def test_registered_source_is_rechecked_before_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    source = _source(tmp_path)
    source_xml = source["xml"].read_text()
    evidence = tmp_path / "evidence"
    calls: list[tuple[str, ...]] = []
    normalized = False

    def fake_command(*args, **kwargs):
        nonlocal normalized
        calls.append(args)
        if args[-3:] == ("list", "--all", "--name"):
            return ""
        if "domstate" in args:
            if not normalized:
                return "shut off"
            return "running" if change == "running" else "shut off"
        if "dumpxml" in args:
            if normalized and change == "remapped":
                return source_xml.replace(
                    str(source["disk"]), str(tmp_path / "other.qcow2")
                )
            return source_xml
        raise AssertionError(args)

    monkeypatch.setattr(lifecycle, "command", fake_command)
    registered = lifecycle.source_from_domain("source")
    normalized = True

    with pytest.raises(ValueError, match=r"shut off|storage changed"):
        with lifecycle.DesktopSession(registered, evidence):
            pass

    assert not any(call[:2] == ("qemu-img", "create") for call in calls)
    assert source["disk"].read_bytes() == b"source disk"
    assert source["nvram"].read_bytes() == b"source nvram"


def test_partial_setup_is_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()

    def fake_command(*args, **kwargs):
        if args[-3:] == ("list", "--all", "--name"):
            return ""
        if args[:2] == ("qemu-img", "create"):
            Path(args[-1]).write_bytes(b"partial overlay")
            return ""
        if args[0] == "virt-clone":
            raise subprocess.CalledProcessError(1, args)
        raise AssertionError(args)

    monkeypatch.setattr(lifecycle, "command", fake_command)

    with pytest.raises(subprocess.CalledProcessError):
        with lifecycle.DesktopSession(source, evidence):
            pass

    assert not (evidence / "disk.qcow2").exists()
    assert not (evidence / "VARS.fd").exists()
    cleanup = json.loads((evidence / "cleanup.json").read_text())
    assert cleanup["source_disk_unchanged"]
    assert cleanup["source_nvram_unchanged"]


def test_failed_disk_creation_removes_only_its_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    signal_changes: list[tuple[int, object]] = []

    def fake_signal(signum, handler):
        signal_changes.append((signum, handler))
        return signal.SIG_DFL

    def fake_command(*args, **kwargs):
        if args[-3:] == ("list", "--all", "--name"):
            return ""
        if args[:2] == ("qemu-img", "create"):
            Path(args[-1]).write_bytes(b"partial overlay")
            raise subprocess.CalledProcessError(1, args)
        raise AssertionError(args)

    monkeypatch.setattr(lifecycle, "command", fake_command)
    monkeypatch.setattr(lifecycle.signal, "signal", fake_signal)

    with pytest.raises(subprocess.CalledProcessError):
        with lifecycle.DesktopSession(source, evidence):
            pass

    assert not (evidence / "disk.qcow2").exists()
    assert not (evidence / "VARS.fd").exists()
    assert (signal.SIGINT, signal.SIG_IGN) in signal_changes
    assert (signal.SIGTERM, signal.SIG_IGN) in signal_changes


def test_preexisting_clone_storage_is_never_claimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    disk = evidence / "disk.qcow2"
    nvram = evidence / "VARS.fd"
    evidence.chmod(0o755)
    disk.write_bytes(b"someone else's disk")
    nvram.write_bytes(b"someone else's nvram")
    monkeypatch.setattr(
        lifecycle,
        "command",
        lambda *args, **kwargs: (
            "" if args[-3:] == ("list", "--all", "--name") else pytest.fail(str(args))
        ),
    )

    with pytest.raises(ValueError, match="new or an empty directory"):
        with lifecycle.DesktopSession(source, evidence):
            pass

    assert disk.read_bytes() == b"someone else's disk"
    assert nvram.read_bytes() == b"someone else's nvram"
    assert evidence.stat().st_mode & 0o777 == 0o755


def test_sigterm_uses_context_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    domains: set[str] = set()
    installed: dict[int, object] = {}

    def fake_signal(signum, handler):
        previous = installed.get(signum, signal.SIG_DFL)
        installed[signum] = handler
        return previous

    def fake_command(*args, **kwargs):
        if args[-3:] == ("list", "--all", "--name"):
            return "\n".join(domains)
        if args[:2] == ("qemu-img", "create"):
            Path(args[-1]).write_bytes(b"overlay")
            return ""
        if args[0] == "virt-clone":
            return _clone_xml(
                args[args.index("--name") + 1],
                Path(args[args.index("--file") + 1]),
                Path(args[args.index("--nvram") + 1]),
            )
        if "define" in args:
            domains.add(lifecycle.ET.parse(args[-1]).findtext("name"))
            return ""
        if "dumpxml" in args:
            return _clone_xml(
                next(iter(domains)), evidence / "disk.qcow2", evidence / "VARS.fd"
            )
        if "domstate" in args:
            return "shut off"
        if "undefine" in args:
            domains.clear()
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(lifecycle.signal, "signal", fake_signal)
    monkeypatch.setattr(lifecycle, "command", fake_command)

    with pytest.raises(KeyboardInterrupt):
        with lifecycle.DesktopSession(source, evidence):
            installed[signal.SIGTERM](signal.SIGTERM, None)

    assert not domains
    assert not (evidence / "disk.qcow2").exists()
    assert not (evidence / "VARS.fd").exists()


@pytest.mark.parametrize("failure", ["destroy", "undefine"])
def test_domain_cleanup_failure_preserves_owned_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    source = _source(tmp_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    domains: set[str] = set()
    state = "running"
    calls: list[tuple[str, ...]] = []

    def fake_command(*args, **kwargs):
        nonlocal state
        calls.append(args)
        if args[-3:] == ("list", "--all", "--name"):
            return "\n".join(domains)
        if args[:2] == ("qemu-img", "create"):
            Path(args[-1]).write_bytes(b"overlay")
            return ""
        if args[0] == "virt-clone":
            return _clone_xml(
                args[args.index("--name") + 1],
                Path(args[args.index("--file") + 1]),
                Path(args[args.index("--nvram") + 1]),
            )
        if "define" in args:
            domains.add(lifecycle.ET.parse(args[-1]).findtext("name"))
            return ""
        if "dumpxml" in args:
            return _clone_xml(
                next(iter(domains)), evidence / "disk.qcow2", evidence / "VARS.fd"
            )
        if "domstate" in args:
            return state
        if "destroy" in args:
            if failure == "destroy":
                raise subprocess.CalledProcessError(1, args)
            state = "shut off"
            return ""
        if "undefine" in args:
            if failure == "undefine":
                raise subprocess.CalledProcessError(1, args)
            domains.clear()
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(lifecycle, "command", fake_command)

    with pytest.raises(lifecycle.CleanupError):
        with lifecycle.DesktopSession(source, evidence):
            pass

    assert (evidence / "disk.qcow2").exists()
    assert (evidence / "VARS.fd").exists()
    cleanup = json.loads((evidence / "cleanup.json").read_text())
    assert not cleanup["clone_removed"]
    assert cleanup["errors"]
    if failure == "undefine":
        undefine = next(call for call in calls if "undefine" in call)
        assert "--keep-nvram" in undefine
