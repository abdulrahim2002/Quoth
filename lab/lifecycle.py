# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Shared lifecycle for disposable desktop lab guests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from vncdotool import api
from vncdotool.client import VNCDoToolFactory

URI = "qemu:///session"
QEMU_NS = "http://libvirt.org/schemas/domain/qemu/1.0"
ET.register_namespace("qemu", QEMU_NS)


class CleanupError(RuntimeError):
    """A failed or unverified guest cleanup; `cleanup.json` holds the details."""


def command(*args, timeout=30, **kwargs):
    """Run `args` and return its stripped stdout; a non-zero exit raises."""
    return subprocess.run(
        args, check=True, text=True, capture_output=True, timeout=timeout, **kwargs
    ).stdout.strip()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wait_for(check, description, timeout=120):
    """Poll `check` each second and return its first truthy result.

    Raises TimeoutError carrying `description` after `timeout` seconds.
    """
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if result := check():
            return result
        time.sleep(1)
    raise TimeoutError(description)


def normalize_prepared_source(manifest: str | Path | dict) -> dict:
    """Normalize a trusted prepared-image manifest for `DesktopSession`.

    Raises ValueError when the disk, NVRAM or XML entry or its SHA-256 is missing.
    """
    source = (
        json.loads(Path(manifest).read_text())
        if isinstance(manifest, (str, Path))
        else dict(manifest)
    )
    source["kind"] = "prepared"
    for key in (
        "disk",
        "nvram",
        "xml",
        "identity",
        "known_hosts",
        "password_file",
        "speech",
        "silence",
        "evidence_root",
    ):
        if value := source.get(key):
            source[key] = Path(value)
    for key in ("disk", "nvram", "xml"):
        if key not in source or key + "_sha256" not in source:
            raise ValueError(f"Prepared manifest is missing {key} or its SHA-256")
    return source


def _registered_storage(domain: str, uri: str) -> tuple[Path, Path]:
    state = command("virsh", "-c", uri, "domstate", domain)
    if state != "shut off":
        raise ValueError(f"Source domain must be shut off: {domain} ({state})")
    root = ET.fromstring(command("virsh", "-c", uri, "dumpxml", domain))
    disk_node = root.find('./devices/disk[@device="disk"]/source')
    nvram_node = root.find("./os/nvram")
    if disk_node is None or not disk_node.get("file") or nvram_node is None:
        raise ValueError(f"Source domain has no file disk or NVRAM: {domain}")
    return Path(disk_node.get("file")), Path(nvram_node.text)


def source_from_domain(domain: str, *, uri: str = URI) -> dict:
    """Snapshot the storage identity of a powered-off registered domain."""
    disk, nvram = _registered_storage(domain, uri)
    return {
        "kind": "domain",
        "domain": domain,
        "disk": disk,
        "nvram": nvram,
        "disk_sha256": sha256(disk),
        "nvram_sha256": sha256(nvram),
    }


class DesktopSession:
    """One disposable guest clone, owned from setup through verified cleanup.

    Entering defines the clone (and its audio loopback) from verified source hashes;
    exiting tears it down through `close`.
    """

    def __init__(
        self,
        source: dict,
        evidence: str | Path,
        *,
        name: str | None = None,
        audio: bool = False,
        port_forwards: tuple[tuple[int, int], ...] = (),
        uri: str = URI,
    ):
        self.prepared = source
        self.evidence = Path(evidence)
        self.name = name or "quoth-lab-" + self.evidence.name
        self.uri = uri
        self.port_forwards = port_forwards
        self.audio_enabled = audio
        self.source_name = self.name + "_source"
        self.feed_name = self.name + "_feed"
        self.capture_name = self.name + "_capture"
        self.audio = None
        self.vnc = None
        self.disk = self.evidence / "disk.qcow2"
        self.nvram = self.evidence / "VARS.fd"
        self.speech = source.get("speech")
        self.silence = source.get("silence")
        self._disk_created = False
        self._nvram_created = False
        self._define_attempted = False
        self._previous_signals: dict[int, object] = {}

    def __enter__(self):
        if self.evidence.exists():
            if not self.evidence.is_dir() or any(self.evidence.iterdir()):
                raise ValueError("Evidence path must be new or an empty directory")
        else:
            self.evidence.mkdir(mode=0o700, parents=True)
        self.evidence.chmod(0o700)
        self._install_signal_handlers()
        try:
            self._setup()
        except BaseException:
            self._ignore_interrupts()
            try:
                self.close(suppress_errors=True)
            finally:
                self._restore_signal_handlers()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._ignore_interrupts()
        try:
            self.close(suppress_errors=exc is not None)
        finally:
            self._restore_signal_handlers()
        return False

    def _install_signal_handlers(self):
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Desktop lab interrupted by signal {signum}")

        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous_signals[signum] = signal.signal(signum, interrupted)

    def _ignore_interrupts(self):
        for signum in self._previous_signals:
            signal.signal(signum, signal.SIG_IGN)

    def _restore_signal_handlers(self):
        for signum, handler in self._previous_signals.items():
            signal.signal(signum, handler)
        self._previous_signals.clear()

    def _domain_names(self) -> list[str]:
        return command("virsh", "-c", self.uri, "list", "--all", "--name").splitlines()

    def _verify_source_hashes(self):
        for key in ("disk", "nvram", "speech", "silence"):
            if key not in self.prepared:
                continue
            if sha256(self.prepared[key]) != self.prepared[key + "_sha256"]:
                raise ValueError(f"Source {key} SHA-256 changed")
        if self.prepared["kind"] == "prepared":
            if sha256(self.prepared["xml"]) != self.prepared["xml_sha256"]:
                raise ValueError("Prepared XML SHA-256 changed")

    def _setup(self):
        if self.name in self._domain_names():
            raise ValueError(f"Domain already exists: {self.name}")
        if self.prepared["kind"] == "domain":
            disk, nvram = _registered_storage(self.prepared["domain"], self.uri)
            if disk != self.prepared["disk"] or nvram != self.prepared["nvram"]:
                raise ValueError("Source domain storage changed after normalization")
        self._verify_source_hashes()
        self._disk_created = True
        command(
            "qemu-img",
            "create",
            "-f",
            "qcow2",
            "-F",
            "qcow2",
            "-b",
            str(self.prepared["disk"]),
            str(self.disk),
        )
        self._nvram_created = True
        shutil.copyfile(self.prepared["nvram"], self.nvram)
        clone_args = ["virt-clone", "--connect", self.uri]
        if self.prepared["kind"] == "prepared":
            clone_args.extend(("--original-xml", str(self.prepared["xml"])))
        else:
            clone_args.extend(("--original", self.prepared["domain"]))
        clone_args.extend(
            (
                "--name",
                self.name,
                "--file",
                str(self.disk),
                "--nvram",
                str(self.nvram),
                "--preserve-data",
                "--print-xml",
            )
        )
        root = ET.fromstring(command(*clone_args))
        interface = root.find("./devices/interface")
        if interface is not None:
            for child in list(interface):
                if child.tag in {"backend", "portForward"}:
                    interface.remove(child)
        if self.audio_enabled:
            self._configure_audio(root)
        xml_path = self.evidence / "domain.xml"
        ET.ElementTree(root).write(xml_path)
        self._define_attempted = True
        command("virsh", "-c", self.uri, "define", str(xml_path))
        if self.audio_enabled:
            self._start_audio()

    def _configure_audio(self, root):
        audio = root.find("./devices/audio")
        if audio is None:
            audio = ET.SubElement(root.find("./devices"), "audio")
        audio.clear()
        audio.attrib.update(
            id="1", type="pipewire", runtimeDir=os.environ["XDG_RUNTIME_DIR"]
        )
        ET.SubElement(
            audio, "input", {"name": self.source_name, "streamName": self.capture_name}
        )
        ET.SubElement(audio, "output", {"name": self.name + "_absent_output"})
        config = self.evidence / "qemu-pipewire.conf"
        config.write_text(
            Path("/usr/share/pipewire/client.conf").read_text() + "\n"
            "stream.properties = { node.dont-fallback = true "
            "node.dont-reconnect = true node.dont-move = true }\n"
        )
        commandline = ET.SubElement(root, f"{{{QEMU_NS}}}commandline")
        for name, value in {
            "PIPEWIRE_CONFIG_DIR": str(self.evidence),
            "PIPEWIRE_CONFIG_NAME": config.name,
        }.items():
            ET.SubElement(
                commandline, f"{{{QEMU_NS}}}env", {"name": name, "value": value}
            )

    def host_graph(self):
        return json.loads(command("pw-dump"))

    def _start_audio(self):
        log = (self.evidence / "audio-process.log").open("w")
        try:
            self.audio = subprocess.Popen(
                [
                    "pw-loopback",
                    "-n",
                    self.name,
                    "-c",
                    "1",
                    "-m",
                    "MONO",
                    "--capture-props",
                    f"node.name={self.feed_name} media.class=Audio/Sink "
                    "node.autoconnect=false node.dont-fallback=true node.dont-reconnect=true "
                    "node.dont-move=true priority.session=0",
                    "--playback-props",
                    f"node.name={self.source_name} media.class=Audio/Source "
                    "node.autoconnect=false priority.session=0",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        finally:
            log.close()
        wait_for(
            lambda: any(
                item.get("info", {}).get("props", {}).get("node.name")
                == self.source_name
                for item in self.host_graph()
            ),
            "audio source",
        )

    def play_audio(self, fixture: str | Path):
        return command("pw-play", "--target", self.feed_name, str(fixture), timeout=20)

    def stop_audio(self):
        if self.audio is None:
            return
        self.audio.terminate()
        try:
            self.audio.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.audio.kill()
            self.audio.wait(timeout=5)
        self.audio = None
        wait_for(
            lambda: (
                not any(
                    item.get("info", {}).get("props", {}).get("node.name")
                    == self.source_name
                    for item in self.host_graph()
                )
            ),
            "Audio source remained after its process exited",
            timeout=10,
        )

    def boot(self):
        """Start the clone, apply its port forwards and return its VNC client."""
        command("virsh", "-c", self.uri, "start", self.name)
        for host_port, guest_port in self.port_forwards:
            result = command(
                "virsh",
                "-c",
                self.uri,
                "qemu-monitor-command",
                "--hmp",
                self.name,
                f"hostfwd_add tcp:127.0.0.1:{host_port}-:{guest_port}",
            )
            if result:
                raise RuntimeError(result)
        if self.port_forwards:
            (self.evidence / "network.txt").write_text(
                command(
                    "virsh",
                    "-c",
                    self.uri,
                    "qemu-monitor-command",
                    "--hmp",
                    self.name,
                    "info usernet",
                )
            )
        live = ET.fromstring(command("virsh", "-c", self.uri, "dumpxml", self.name))
        graphics = live.find('./devices/graphics[@type="vnc"]')
        listen = graphics.find("listen") if graphics is not None else None
        vnc_socket = (listen.get("socket") if listen is not None else None) or (
            graphics.get("socket") if graphics is not None else None
        )
        if not vnc_socket:
            raise RuntimeError("Running domain has no VNC socket")
        factory = type("USKeyboard", (VNCDoToolFactory,), {"force_caps": True})
        self.vnc = api.connect(vnc_socket, factory_class=factory, timeout=10)
        return self.vnc

    def cold_boot(self):
        if self.vnc is not None:
            self.vnc.disconnect()
            self.vnc = None
        command("virsh", "-c", self.uri, "destroy", self.name)
        return self.boot()

    def capture(self, filename: str):
        if self.vnc is None:
            raise RuntimeError("Desktop is not booted")
        self.vnc.captureScreen(str(self.evidence / filename))

    def _domain_owns_storage(self) -> bool:
        root = ET.fromstring(command("virsh", "-c", self.uri, "dumpxml", self.name))
        disk = root.find('./devices/disk[@device="disk"]/source')
        nvram = root.find("./os/nvram")
        return (
            disk is not None
            and disk.get("file") == str(self.disk)
            and nvram is not None
            and nvram.text == str(self.nvram)
        )

    def _retire_domain(self, errors: list[str]) -> bool:
        try:
            exists = self.name in self._domain_names()
        except Exception as error:
            errors.append(f"could not list domains: {error}")
            return False
        if not exists:
            return True
        if not self._define_attempted:
            errors.append("domain exists without an ownership attempt")
            return False
        try:
            owns_storage = self._domain_owns_storage()
        except Exception as error:
            errors.append(f"could not verify domain storage ownership: {error}")
            return False
        if not owns_storage:
            errors.append("domain storage does not match the owned clone")
            return False
        try:
            state = command("virsh", "-c", self.uri, "domstate", self.name)
        except Exception as error:
            errors.append(f"could not read domain state: {error}")
            return False
        if state != "shut off":
            try:
                command("virsh", "-c", self.uri, "destroy", self.name)
            except Exception as error:
                errors.append(f"could not destroy owned domain: {error}")
                return False
        try:
            command("virsh", "-c", self.uri, "undefine", self.name, "--keep-nvram")
        except Exception as error:
            errors.append(f"could not undefine owned domain: {error}")
            return False
        try:
            if self.name in self._domain_names():
                errors.append("owned domain remains after undefine")
                return False
        except Exception as error:
            errors.append(f"could not verify domain removal: {error}")
            return False
        return True

    def _remove_storage(self, errors: list[str]):
        for path, owned in (
            (self.disk, self._disk_created),
            (self.nvram, self._nvram_created),
        ):
            if not owned:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError as error:
                errors.append(f"could not remove {path.name}: {error}")

    def _write_cleanup(self, errors: list[str]):
        result = {}
        for key in ("disk", "nvram"):
            try:
                result[f"source_{key}_unchanged"] = (
                    sha256(self.prepared[key]) == self.prepared[key + "_sha256"]
                )
            except OSError as error:
                result[f"source_{key}_unchanged"] = False
                errors.append(f"could not hash source {key}: {error}")
        try:
            result["domain_removed"] = self.name not in self._domain_names()
        except Exception as error:
            result["domain_removed"] = False
            errors.append(f"could not verify final domain removal: {error}")
        result["clone_removed"] = not self.disk.exists() and not self.nvram.exists()
        if self.audio_enabled:
            try:
                result["audio_removed"] = not any(
                    item.get("info", {})
                    .get("props", {})
                    .get("node.name", "")
                    .startswith(self.name)
                    for item in self.host_graph()
                )
            except Exception as error:
                result["audio_removed"] = False
                errors.append(f"could not verify audio removal: {error}")
        else:
            result["audio_removed"] = True
        result["errors"] = errors
        (self.evidence / "cleanup.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        return result

    def close(self, *, suppress_errors: bool = False):
        """Retire the clone and write `cleanup.json`, raising CleanupError on failure.

        Clone storage is removed only once the domain is verified as this session's own
        and undefined. `suppress_errors` records failures without raising.
        """
        errors: list[str] = []
        if self.vnc is not None:
            try:
                self.vnc.disconnect()
            except Exception as error:
                errors.append(f"could not disconnect VNC: {error}")
            self.vnc = None
        try:
            self.stop_audio()
        except Exception as error:
            errors.append(f"could not stop audio: {error}")
        storage_safe = self._retire_domain(errors)
        if storage_safe:
            self._remove_storage(errors)
        result = self._write_cleanup(errors)
        failed = errors or not all(
            value for key, value in result.items() if key != "errors"
        )
        if failed and not suppress_errors:
            raise CleanupError("Desktop cleanup failed; see cleanup.json")
