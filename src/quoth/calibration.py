# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Hardware calibration: measures each installed model x device pair on this machine.

Optional: with no calibration data the resolver uses its default ranking. A run times
every `(model, device)` cell for `auto`'s per-model device choice, marks a loaded cell
warm, and records a failure as no result, so the ranking leaves that cell out. Results
persist to `calibration.json`, keyed by a hardware and runtime fingerprint so they drop
after a driver or engine change. It also owns the result wording every calibration view
shares.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from quoth import atomic_write, paths
from quoth.accel import backend_for, has_accelerator
from quoth.models import model_label, needs_openvino_encoder
from quoth.whisper import (
    SAMPLE_RATE,
    TranscriptionBackend,
    WhisperContext,
    open_context,
)

if TYPE_CHECKING:
    from quoth.models import ModelManager

# A short speech-band synthetic buffer is enough to rank backends by speed (content
# barely affects the offloaded encoder cost). 16 kHz mono float32.
_SAMPLE_SECONDS = 5.0
_TIMED_RUNS = 2  # best-of, after one warm-up run that pays the first-load compile


@dataclass(frozen=True)
class CellResult:
    """One `(model, device)` measurement.

    `compiled` means the measurement succeeded; for an NPU cell it also means the
    OpenVINO blob is cached, so its next load is fast, and an eviction resets it.
    `ms` is the best inference time, kept across an eviction, or None when the cell
    was only warmed or failed. `skipped` marks a cell a run did not attempt because
    its assets are not installed; it is never stored.
    """

    compiled: bool
    ms: float | None = None
    skipped: bool = False


@dataclass(frozen=True)
class Report:
    """A fingerprinted set of cells, keyed `"model|device"`."""

    fingerprint: str
    cells: dict[str, CellResult]


# --- keys + persistence -----------------------------------------------------


def _key(model: str, device: str) -> str:
    return f"{model}|{device}"


def default_path() -> Path:
    """Return the calibration file in the state directory."""
    return paths.state_dir() / "calibration.json"


# Every resolve reads the report, so a malformed file is skipped, and warned about,
# once per version (path, mtime, size), and a stale fingerprint is noted once per pair.
_unusable: tuple[Path, int, int] | None = None
_unreadable: tuple[Path, int, int] | Path | None = None
_stale_logged: set[tuple[str, str]] = set()


def load(path: Path) -> Report | None:
    """Read the report, or None if absent, unreadable or malformed.

    Calibration is best-effort: a file its readers cannot use is dropped, not repaired.
    """
    global _unusable, _unreadable
    try:
        st = path.stat()
        version = (path, st.st_mtime_ns, st.st_size)
    except OSError:
        version = None
    if version is not None and version == _unusable:
        return None
    try:
        text = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as e:
        # Read again each time, since a permission fix changes neither mtime nor size.
        # A file that cannot even be statted is keyed by its path.
        if (version or path) != _unreadable:
            logger.warning("Calibration file {} unreadable ({}); ignoring it", path, e)
        _unreadable = version or path
        return None
    try:
        # Decoded here, so text that is not UTF-8 counts as malformed.
        data = json.loads(text)
        cells = {}
        for key, cell in data["cells"].items():
            # Every reader splits the key into its model and device, and sorts by `ms`.
            if "|" not in key:
                raise ValueError(f"cell {key!r} names no device")
            ms = cell.get("ms")
            if ms is not None:
                ms = float(ms)
                # Readers divide by it and sort by it.
                if not (math.isfinite(ms) and ms > 0):
                    raise ValueError(f"cell {key!r} has no usable time")
            cells[key] = CellResult(cell["compiled"], ms)
        # Readers compare and hash the fingerprint.
        if not isinstance(data["fingerprint"], str):
            raise ValueError("the fingerprint is not a string")
        return Report(data["fingerprint"], cells)
    except (
        KeyError,
        TypeError,
        AttributeError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as e:
        logger.warning("Calibration file {} malformed ({}); discarding", path, e)
        _unusable = version
        return None


def save(path: Path, report: Report) -> None:
    """Persist the report atomically, so a concurrent reader never sees a torn file."""
    data = {
        "fingerprint": report.fingerprint,
        "cells": {
            k: {"compiled": c.compiled, "ms": c.ms} for k, c in report.cells.items()
        },
    }
    atomic_write.write_text(path, json.dumps(data, indent=2))


def _current_report(path: Path, fingerprint: str) -> Report | None:
    """Return the stored report if its fingerprint matches this machine, else None."""
    report = load(path)
    if report and report.fingerprint != fingerprint:
        if (report.fingerprint, fingerprint) not in _stale_logged:
            _stale_logged.add((report.fingerprint, fingerprint))
            logger.debug(
                "Calibration from another fingerprint ignored (stored {}, now {})",
                report.fingerprint,
                fingerprint,
            )
        return None
    return report


def _fresh_cells(path: Path, fingerprint: str) -> dict[str, CellResult]:
    """Return the stored cells if the fingerprint still matches, else an empty dict."""
    report = _current_report(path, fingerprint)
    return dict(report.cells) if report else {}


# --- measurement ------------------------------------------------------------


def _sample() -> np.ndarray:
    n = int(SAMPLE_RATE * _SAMPLE_SECONDS)
    t = np.linspace(0, _SAMPLE_SECONDS, n, endpoint=False)
    sig = 0.3 * np.sin(2 * np.pi * 180 * t) + 0.15 * np.sin(2 * np.pi * 400 * t)
    return sig.astype(np.float32)


def measure(
    model: str,
    device: str,
    model_mgr: ModelManager,
    *,
    factory: Callable[..., TranscriptionBackend] = WhisperContext,
) -> CellResult:
    """Build the `(model, device)` context and time it.

    The first `transcribe_buffer` pays the one-time warm-up (on an NPU cell that
    compiles and caches the OpenVINO blob); later runs are the warm speed. A load or
    inference failure comes back as `compiled=False`: data, not an exception.
    """
    if needs_openvino_encoder(device, model) and not model_mgr.has_openvino(model):
        # Without the encoder `open_context` builds a CPU fallback, which must not be
        # stored as a compiled NPU cell.
        logger.debug("Skipping {} on {}: OpenVINO encoder not installed", model, device)
        return CellResult(compiled=False, skipped=True)
    ctx = open_context(model, device, model_mgr, factory=factory)
    if ctx is None:
        return CellResult(compiled=False)
    if needs_openvino_encoder(device, model) and not ctx.openvino_active:
        # The blob is on disk but its runtime init failed, so whisper.cpp silently ran
        # the encoder on CPU: the same false NPU cell as above.
        logger.debug(
            "OpenVINO encoder init failed at runtime for {} on {}", model, device
        )
        ctx.close()
        return CellResult(compiled=False)
    try:
        audio = _sample()
        ctx.transcribe_buffer(audio, language="en")  # warm-up (compiles the NPU blob)
        best = min(_time_once(ctx, audio) for _ in range(_TIMED_RUNS))
        return CellResult(compiled=True, ms=best)
    except Exception:
        logger.exception("Calibration measure failed for {} on {}", model, device)
        return CellResult(compiled=False)
    finally:
        ctx.close()


def _time_once(ctx, audio: np.ndarray) -> float:
    t0 = time.perf_counter()
    ctx.transcribe_buffer(audio, language="en")
    return (time.perf_counter() - t0) * 1000


# --- run + record -----------------------------------------------------------

# The injected measurement seam: (model, device, model_mgr) -> CellResult.
MeasureFn = Callable[[str, str, "ModelManager"], CellResult]
# A run's callbacks as its views receive them: (done, total, model, device, result)
# per cell, then the report, or None if the run raised.
ProgressFn = Callable[[int, int, str, str, CellResult], None]
DoneFn = Callable[[Report | None], None]


def cell_order(models: list[str], devices: list[str]) -> list[tuple[str, str]]:
    """Return the `(model, device)` cells a run measures, in order.

    `run` and the progress views share it, so a view's cell list matches the
    `on_progress` events.
    """
    return [(m, d) for m in models for d in devices]


def run(
    models: list[str],
    devices: list[str],
    fingerprint: str,
    model_mgr: ModelManager,
    *,
    path: Path | None = None,
    measure_fn: MeasureFn = measure,
    on_progress: ProgressFn | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Report:
    """Measure every `models x devices` cell, persisting after each.

    Persisting per cell lets a partial run survive a cancel or crash. `on_progress(done,
    total, model, device, result)` fires per cell; a truthy `should_cancel()` stops
    early. The report covers only the requested cells, so other models' stored cells
    stay out of its verdict. A requested cell that is skipped or not reached keeps what
    an earlier run stored: not attempting it says nothing about the device.
    """
    path = path or default_path()
    measured = _fresh_cells(path, fingerprint)
    todo = cell_order(models, devices)
    logger.info(
        "Calibration started: {} cells (models={}, devices={})",
        len(todo),
        models,
        devices,
    )
    for done, (model, device) in enumerate(todo, start=1):
        if should_cancel and should_cancel():
            logger.info("Calibration cancelled after {}/{} cells", done - 1, len(todo))
            break
        result = measure_fn(model, device, model_mgr)
        if not result.skipped:
            measured[_key(model, device)] = result
            try:
                save(path, Report(fingerprint, dict(measured)))
            except OSError as e:
                # Only the stored ranking misses the cell; this run still reports it.
                logger.warning(
                    "Couldn't save calibration for {} on {} ({})", model, device, e
                )
        if on_progress:
            on_progress(done, len(todo), model, device, result)
    keys = {_key(model, device) for model, device in todo}
    report = Report(fingerprint, {k: v for k, v in measured.items() if k in keys})
    logger.info(
        "Calibration finished: {}/{} cells measured, ranking={}",
        len(report.cells),
        len(todo),
        ranking(report),
    )
    return report


def mark_warm(
    model: str, device: str, fingerprint: str, *, path: Path | None = None
) -> None:
    """Mark `(model, device)`'s blob compiled and warm, keeping any measured `ms`.

    Re-warming after an eviction keeps the benchmark timing. Stale cells are wiped on
    a fingerprint change.
    """
    path = path or default_path()
    cells = _fresh_cells(path, fingerprint)
    existing = cells.get(_key(model, device))
    cells[_key(model, device)] = CellResult(
        compiled=True, ms=existing.ms if existing else None
    )
    try:
        save(path, Report(fingerprint, cells))
    except OSError as e:
        # Only costs a repeat of the one-time prepare, never the dictation itself.
        logger.warning("Couldn't record {} on {} as prepared ({})", model, device, e)
        return
    logger.debug(
        "Marked {} on {} warm (ms={})", model, device, existing.ms if existing else None
    )


def evict_warmth(fingerprint: str, *, path: Path | None = None) -> None:
    """Mark every OpenVINO (NPU) cell cold: its next load is a slow first compile.

    Call before wiping the OpenVINO blob cache, so the `is_warm` gate shows the
    preparing notice again instead of compiling during a dictation. The measured `ms`
    stays; GPU and CPU cells have no blob and are untouched.
    """
    path = path or default_path()
    cells = _fresh_cells(path, fingerprint)
    if not cells:
        return
    cold = {k for k in cells if backend_for(k.split("|", 1)[1]).wants_openvino_encoder}
    updated = {
        k: CellResult(compiled=False, ms=c.ms) if k in cold else c
        for k, c in cells.items()
    }
    save(path, Report(fingerprint, updated))
    logger.info(
        "Evicted OpenVINO warmth: {}/{} cells reset cold", len(cold), len(cells)
    )


# --- derived views (what the resolver + warm gate consume) ------------------


def _timed_by_model(report: Report) -> dict[str, list[tuple[float, str]]]:
    """Group a report's timed cells by model as `(ms, device)` pairs.

    Untimed cells (failed or only warmed) are dropped; a model with none is absent.
    """
    by_model: dict[str, list[tuple[float, str]]] = {}
    for k, cell in report.cells.items():
        if cell.ms is None:
            continue
        model, device = k.split("|", 1)
        by_model.setdefault(model, []).append((cell.ms, device))
    return by_model


def ranking(report: Report) -> dict[str, list[str]]:
    """Return, per model, the benchmarked devices best-first (fastest `ms` first).

    Cells that failed or were only warmed (no `ms`) are omitted, so the resolver falls
    back to its default for them.
    """
    return {
        model: [device for _, device in sorted(pairs)]
        for model, pairs in _timed_by_model(report).items()
    }


def winners(report: Report) -> dict[str, tuple[str, float]]:
    """Return, per model, the fastest measured `(device, ms)`.

    It filters cells as `ranking` does: a model with no timed cell is absent.
    """
    return {
        model: (device, ms)
        for model, pairs in _timed_by_model(report).items()
        for ms, device in [min(pairs)]
    }


def realtime_multiple(ms: float) -> float:
    """Return a cell's speed as audio seconds per wall-clock second (862 ms -> 5.8)."""
    return _SAMPLE_SECONDS * 1000 / ms


# --- result copy (GTK-free; shared by the benchmark dialog and onboarding) ---


def speed_label(ms: float) -> str:
    """Return a cell's speed for the UI, e.g. "5.8× realtime" or "14× realtime".

    One decimal below 10×, an integer at or above.
    """
    x = realtime_multiple(ms)
    return f"{x:.1f}× realtime" if x < 10 else f"{x:.0f}× realtime"


def measuring_label(model: str, device: str) -> str:
    """Return the live "which cell is running now" line.

    The benchmark dialog appends its own "(n of N)" count.
    """
    return f"Measuring {model_label(model)} on {device}…"


def device_phrase(devices: list[str]) -> str:
    """Return the devices under test as a phrase, e.g. "CPU, NPU and GPU"."""
    if len(devices) <= 1:
        return devices[0] if devices else "CPU"
    return f"{', '.join(devices[:-1])} and {devices[-1]}"


def preparation_note(devices: list[str]) -> str:
    """Return the caveat, with its leading space, that a device's first run compiles.

    Empty when no device in `devices` compiles, so a machine without an NPU never reads
    about one.
    """
    if any(backend_for(device).wants_openvino_encoder for device in devices):
        return " First-time NPU preparation can take longer."
    return ""


def verdict(winners: dict[str, tuple[str, float]], *, measured: int, total: int) -> str:
    """Return the benchmark's result line across all models."""
    if not winners:
        return "The benchmark didn't finish. Quoth will use a safe default."
    devices = {device for device, _ in winners.values()}
    if not has_accelerator(list(devices)):
        # No winner runs on an accelerator, so present the CPU as ready.
        body = "Quoth is tuned for your CPU and ready to go"
    elif len(devices) == 1:
        subject = (
            model_label(next(iter(winners))) if len(winners) == 1 else "Every model"
        )
        body = f"{subject} runs fastest on your {next(iter(devices))}"
    else:
        best = max(realtime_multiple(ms) for _, ms in winners.values())
        body = (
            f"Quoth will run each model on its fastest device, up to {best:.0f}× "
            "realtime"
        )
    if measured < total:
        return f"Based on {measured} of {total} measurements. {body}."
    return f"{body}."


def optimized_line(winners: dict[str, tuple[str, float]], model: str) -> str:
    """Return onboarding's single-model result: the fastest device and its speed.

    It does not name the model, since onboarding never offers a choice of one.
    """
    won = winners.get(model)
    if won is None:
        return "Quoth couldn't measure your hardware and will use a safe default."
    device, ms = won
    if not backend_for(device).is_accelerated:
        return "Quoth is tuned for your CPU and ready to go."
    return f"Quoth runs fastest on your {device}: {speed_label(ms)}."


def current_ranking(
    fingerprint: str, *, path: Path | None = None
) -> dict[str, list[str]] | None:
    """Return the ranking for the current fingerprint, or None without fresh timings.

    None sends the resolver to its default order.
    """
    report = _current_report(path or default_path(), fingerprint)
    return (ranking(report) or None) if report else None


def worth_offering(
    devices: list[str], fingerprint: str, *, path: Path | None = None
) -> bool:
    """Whether a first-run calibration offer is worthwhile.

    It is when there is an accelerator to rank and no fresh ranking yet for this
    machine.
    """
    return has_accelerator(devices) and current_ranking(fingerprint, path=path) is None


def is_warm(
    model: str, device: str, fingerprint: str, *, path: Path | None = None
) -> bool:
    """Whether `(model, device)` is known-compiled under the current fingerprint.

    Its load is then a fast cache hit rather than a slow first compile.
    """
    report = _current_report(path or default_path(), fingerprint)
    cell = report.cells.get(_key(model, device)) if report else None
    return bool(cell and cell.compiled)
