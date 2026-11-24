# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Readiness checks for an instrumented desktop, without Quoth installed."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import convolve, correlate


def _props(item):
    return item["info"]["props"]


def _capture_inputs(desktop):
    """Return the QEMU capture stream and the nodes linked into it."""
    graph = desktop.host_graph()
    nodes = [item for item in graph if item.get("type") == "PipeWire:Interface:Node"]
    links = [item for item in graph if item.get("type") == "PipeWire:Interface:Link"]
    capture_nodes = [
        node
        for node in nodes
        if _props(node).get("node.name") == desktop.capture_name
        or _props(node).get("media.name") == desktop.capture_name
    ]
    capture_ids = {node["id"] for node in capture_nodes}
    incoming_ids = {
        link["info"]["output-node-id"]
        for link in links
        if link["info"]["input-node-id"] in capture_ids
    }
    return capture_nodes, [node for node in nodes if node.get("id") in incoming_ids]


def _record_guest(
    desktop,
    remote_path: str,
    log_name: str,
    playback: tuple[Path, ...] = (),
    *,
    expect_fixture: bool,
    expect_capture_absent: bool = False,
) -> tuple[Path, int]:
    """Capture a bounded guest recording and retain its SSH transcript."""
    command = (
        f"XDG_RUNTIME_DIR={shlex.quote(desktop.guest_runtime_dir)} "
        "timeout --signal=INT 18 pw-record --rate=16000 --channels=1 --format=s16 "
        f"{shlex.quote(remote_path)}"
    )
    log_path = desktop.evidence / log_name
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [*desktop.ssh_args, command],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if (
                    desktop.run_guest(
                        "pgrep -x pw-record >/dev/null && printf active || true"
                    ).strip()
                    == "active"
                ):
                    break
                time.sleep(0.2)
            else:
                raise AssertionError("guest pw-record did not become active")

            deadline = time.monotonic() + 5
            while True:
                capture_nodes, incoming_nodes = _capture_inputs(desktop)
                if (expect_fixture and incoming_nodes) or time.monotonic() >= deadline:
                    break
                time.sleep(0.2)
            (desktop.evidence / f"{Path(log_name).stem}-graph.json").write_text(
                json.dumps(
                    {
                        "capture_nodes": [
                            {"id": node["id"], "props": _props(node)}
                            for node in capture_nodes
                        ],
                        "incoming_node_names": [
                            _props(node).get("node.name") for node in incoming_nodes
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if expect_fixture:
                assert capture_nodes, (
                    "QEMU capture stream is absent while guest recording"
                )
                assert any(
                    _props(node).get("target.object") == desktop.source_name
                    for node in capture_nodes
                ), "QEMU capture stream is not targeted at the owned fixture source"
                assert incoming_nodes, "QEMU capture stream has no incoming audio link"
                assert {_props(node).get("node.name") for node in incoming_nodes} == {
                    desktop.source_name
                }, "QEMU capture stream has a non-fixture source"
            else:
                assert not incoming_nodes, (
                    "guest capture fell back to a host audio source"
                )
            if expect_capture_absent:
                assert not capture_nodes, (
                    "QEMU opened capture without its fixture source"
                )

            for fixture in playback:
                desktop.play_audio(fixture)
            process.wait(timeout=24)
        finally:
            if process.poll() is None:
                desktop.run_guest("pkill -INT -x pw-record || true")
                process.wait(timeout=5)

    assert process.returncode == 124, log_path.read_text(encoding="utf-8")
    return desktop.copy_from(remote_path), process.returncode


def _read_mono_s16(path: Path) -> np.ndarray:
    rate, samples = wavfile.read(path)
    assert rate == 16_000, f"{path.name}: expected 16 kHz, received {rate}"
    assert samples.ndim == 1, f"{path.name}: expected mono audio"
    assert samples.dtype == np.int16, f"{path.name}: expected signed 16-bit PCM"
    return samples


def _write_audio_metrics(desktop, values) -> None:
    (desktop.evidence / "audio.json").write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_desktop_input(desktop) -> None:
    """A normal VNC console can operate the logged-in desktop."""
    desktop.capture("desktop-before-terminal.png")
    desktop.vnc.keyPress(desktop.readiness["terminal_shortcut"])
    desktop.expect_screen("terminal")
    desktop.type_text('printf quoth-desktop-canary > "$HOME/quoth-lab-canary.txt"\n')
    desktop.capture("desktop-after-typing.png")

    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if (
            desktop.run_guest(
                "test -f ~/quoth-lab-canary.txt && cat ~/quoth-lab-canary.txt || true"
            ).strip()
            == "quoth-desktop-canary"
        ):
            break
        time.sleep(0.2)
    else:
        raise AssertionError("terminal canary was not written through the VNC console")
    desktop.capture("desktop-after-terminal.png")


def test_audio(desktop) -> None:
    """The known virtual microphone reaches the guest without host fallback."""
    speech_path, speech_status = _record_guest(
        desktop,
        "/tmp/quoth-lab-speech.wav",
        "pw-record-speech.log",
        (desktop.speech,),
        expect_fixture=True,
    )
    silence_path, silence_status = _record_guest(
        desktop,
        "/tmp/quoth-lab-silence.wav",
        "pw-record-silence.log",
        (desktop.silence,),
        expect_fixture=True,
    )
    metrics = {
        "pw_record_statuses": {"speech": speech_status, "silence": silence_status},
    }
    _write_audio_metrics(desktop, metrics)

    reference = _read_mono_s16(desktop.speech).astype(np.float64)
    captured = _read_mono_s16(speech_path).astype(np.float64)
    metrics["speech_duration_seconds"] = captured.size / 16_000
    _write_audio_metrics(desktop, metrics)
    assert captured.size >= reference.size, (
        "guest recording ended before the complete speech fixture"
    )
    correlation = correlate(captured, reference, mode="valid", method="fft")
    window_energy = convolve(captured * captured, np.ones(reference.size), mode="valid")
    normalized = np.zeros_like(correlation)
    denominator = np.sqrt(np.maximum(window_energy, 0) * np.dot(reference, reference))
    np.divide(np.abs(correlation), denominator, out=normalized, where=denominator > 0)
    peak_correlation = float(np.max(normalized))

    silence = _read_mono_s16(silence_path)
    silence_peak = int(np.max(np.abs(silence.astype(np.int32)))) if silence.size else 0
    metrics.update(
        {
            "speech_peak_correlation": peak_correlation,
            "silence_duration_seconds": silence.size / 16_000,
            "silence_peak": silence_peak,
        }
    )
    _write_audio_metrics(desktop, metrics)
    assert silence.size > 0, "guest silence recording contains no measured duration"
    assert silence_peak == 0, f"silence control contains samples up to {silence_peak}"
    assert peak_correlation > 0.90, (
        f"fixture alignment correlation was {peak_correlation:.3f}"
    )

    desktop.stop_audio()
    for control in ("stopped_capture", "absent_at_boot"):
        cold_boot = control == "absent_at_boot"
        if cold_boot:
            desktop.cold_boot()
        path, status = _record_guest(
            desktop,
            f"/tmp/quoth-lab-{control}.wav",
            f"pw-record-{control}.log",
            expect_fixture=False,
            expect_capture_absent=cold_boot,
        )
        samples = (
            _read_mono_s16(path) if path.stat().st_size else np.empty(0, dtype=np.int16)
        )
        peak = int(np.max(np.abs(samples.astype(np.int32)))) if samples.size else 0
        metrics["pw_record_statuses"][control] = status
        metrics[f"{control}_duration_seconds"] = samples.size / 16_000
        metrics[f"{control}_peak"] = peak
        _write_audio_metrics(desktop, metrics)
        assert peak == 0, f"{control}: guest captured non-fixture audio"
