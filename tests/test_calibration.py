# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `calibration`.

The per-cell measurement is injected, so these run headless with no engine.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from quoth import calibration as cal
from quoth.calibration import CellResult, Report, measure
from tests.support import FakeModelManager

_FP = "ov=libopenvino.so.2600;dev=NPU/GPU/CPU"


class _FakeCtx:
    """An instant context; records each decode's length and language, and close()."""

    def __init__(self, *, raises=False, openvino_active=True):
        self._raises = raises
        self.openvino_active = openvino_active
        self.closed = False
        self.decoded = []

    def transcribe_buffer(self, audio, language="auto", translate=False):
        self.decoded.append((len(audio), language))
        if self._raises:
            raise RuntimeError("inference blew up")
        return ""

    def close(self):
        self.closed = True


class TestMeasure:
    def test_success_returns_timing_and_closes(self):
        # Like the real context, a CPU one never engages the OpenVINO offload.
        ctx = _FakeCtx(openvino_active=False)
        r = measure("small", "CPU", FakeModelManager(), factory=lambda *a, **k: ctx)
        assert r.compiled is True and r.ms is not None and r.ms >= 0
        assert ctx.closed is True

    def test_builds_the_context_for_the_cell_it_measures(self):
        built = []

        def factory(model_path, *, device, **_kwargs):
            built.append((model_path, device))
            return _FakeCtx(openvino_active=False)

        measure("small", "GPU", FakeModelManager(), factory=factory)
        assert built == [("/models/ggml-small.bin", "GPU")]

    def test_every_run_decodes_the_same_five_seconds_of_english(self):
        # realtime_multiple assumes 5 s of 16 kHz audio, and a fixed language keeps
        # whisper.cpp's language detection out of the timing.
        ctx = _FakeCtx(openvino_active=False)
        measure("small", "CPU", FakeModelManager(), factory=lambda *a, **k: ctx)
        assert set(ctx.decoded) == {(80000, "en")}

    def test_ms_is_the_fastest_timed_run_in_milliseconds(self, monkeypatch):
        # Each decode advances a fake clock: the untimed first-load warm-up takes 2 s,
        # the two timed runs 0.5 s and 0.25 s.
        clock = SimpleNamespace(now=0.0)
        monkeypatch.setattr(
            cal, "time", SimpleNamespace(perf_counter=lambda: clock.now)
        )
        decode_seconds = iter([2.0, 0.5, 0.25])
        ctx = _FakeCtx(openvino_active=False)

        def decode(audio, language="auto", translate=False):
            clock.now += next(decode_seconds)
            return ""

        ctx.transcribe_buffer = decode
        r = measure("small", "CPU", FakeModelManager(), factory=lambda *a, **k: ctx)
        assert r == CellResult(compiled=True, ms=250.0)

    def test_model_not_downloaded_is_not_compiled(self):
        r = measure(
            "small",
            "CPU",
            FakeModelManager(ggml=False),
            factory=lambda *a, **k: _FakeCtx(),
        )
        assert r == CellResult(compiled=False)

    def test_inference_failure_still_closes_context(self):
        ctx = _FakeCtx(raises=True)
        r = measure("small", "CPU", FakeModelManager(), factory=lambda *a, **k: ctx)
        assert r.compiled is False
        assert ctx.closed is True  # cleaned up despite the exception

    def test_npu_without_encoder_skipped_before_building(self):
        # The correctness gate: an NPU cell without its encoder on disk is no real
        # measurement, since open_context would silently fall back to the CPU encoder.
        built = []
        r = measure(
            "small",
            "NPU",
            FakeModelManager(encoder=False),
            factory=lambda *a, **k: built.append(1) or _FakeCtx(),
        )
        assert r == CellResult(compiled=False, skipped=True)
        assert built == []  # never attempted to build a bogus NPU context

    def test_npu_with_encoder_but_runtime_init_failed_is_not_compiled(self):
        # The encoder blob is on disk but its runtime init failed, so whisper.cpp
        # silently ran on CPU: the context builds with openvino_active False, and the
        # cell must be recorded compiled=False (not timed) with the context closed.
        ctx = _FakeCtx(openvino_active=False)
        r = measure(
            "small",
            "NPU",
            FakeModelManager(encoder=True),
            factory=lambda *a, **k: ctx,
        )
        assert r == CellResult(compiled=False)
        assert ctx.closed is True

    def test_npu_with_active_offload_is_compiled(self):
        ctx = _FakeCtx(openvino_active=True)
        r = measure(
            "small", "NPU", FakeModelManager(encoder=True), factory=lambda *a, **k: ctx
        )
        assert r.compiled is True and r.ms is not None
        assert ctx.closed is True


def _path(tmp_path):
    return tmp_path / "calibration.json"


def _fake_measure(table):
    """Return a fake `measure_fn` and the list of calls it records, in order.

    Each result comes from `table`, keyed by `(model, device)`; a missing entry is a
    failed cell.
    """
    calls = []

    def measure(model, device, model_mgr):
        calls.append((model, device))
        return table.get((model, device), CellResult(compiled=False))

    return measure, calls


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        p = _path(tmp_path)
        report = Report(
            _FP,
            {
                "small|GPU": CellResult(True, 440.0),
                "small|CPU": CellResult(True, 4800.0),
            },
        )
        cal.save(p, report)
        loaded = cal.load(p)
        assert loaded == report

    def test_load_missing_is_none(self, tmp_path):
        assert cal.load(_path(tmp_path)) is None

    def test_load_corrupt_is_none(self, tmp_path, caplog):
        p = _path(tmp_path)
        p.write_text("{not json")
        assert cal.load(p) is None
        assert cal.load(p) is None  # every resolve reads it, but it warns once
        assert len(caplog.messages) == 1

    def test_load_misshapen_cells_is_none(self, tmp_path, caplog):
        p = _path(tmp_path)
        p.write_text('{"fingerprint": "x", "cells": []}')
        assert cal.load(p) is None
        assert cal.load(p) is None
        assert len(caplog.messages) == 1
        # A rewritten file is read afresh.
        cal.save(p, Report("x", {}))
        assert cal.load(p) == Report("x", {})

    def test_an_unreadable_file_is_read_again_once_it_reads(
        self, tmp_path, monkeypatch, caplog
    ):
        # Fixing a permission changes neither mtime nor size, so the version is no key.
        p = _path(tmp_path)
        cal.save(p, Report("x", {}))

        def denied(*_args, **_kwargs):
            raise PermissionError(13, "Permission denied")

        with monkeypatch.context() as m:
            m.setattr(Path, "read_bytes", denied)
            assert cal.load(p) is None
            assert cal.load(p) is None
        assert cal.load(p) == Report("x", {})
        assert len(caplog.messages) == 1

    @pytest.mark.parametrize(
        "content",
        [
            '{"fingerprint": "x", "cells": {"base|CPU": {"compiled": true, "ms": '
            + "9" * 309
            + "}}}",
            "[" * 100_000 + "]" * 100_000,
            '{"fingerprint": [], "cells": {}}',
            '{"fingerprint": {}, "cells": {}}',
        ],
        ids=[
            "ms-past-float",
            "nested-too-deep",
            "fingerprint-list",
            "fingerprint-object",
        ],
    )
    def test_content_no_reader_can_use_is_malformed(self, tmp_path, caplog, content):
        p = _path(tmp_path)
        p.write_text(content)
        assert cal.load(p) is None
        assert cal.load(p) is None
        assert len(caplog.messages) == 1
        assert "malformed" in caplog.messages[0]

    @pytest.mark.parametrize("ms", ["0", "-5.0", "NaN", "Infinity", "-Infinity"])
    def test_a_time_no_reader_can_divide_or_sort_by_is_malformed(
        self, tmp_path, caplog, ms
    ):
        p = _path(tmp_path)
        p.write_text(
            '{"fingerprint": "x", "cells": {"base|CPU": {"compiled": true, "ms": '
            + ms
            + "}}}"
        )
        assert cal.load(p) is None
        assert len(caplog.messages) == 1
        assert "malformed" in caplog.messages[0]

    def test_a_file_that_is_not_utf8_is_malformed(self, tmp_path, caplog):
        p = _path(tmp_path)
        p.write_bytes(b'\xff\xfe{"cells": {}}')
        assert cal.load(p) is None
        assert cal.load(p) is None
        assert len(caplog.messages) == 1
        assert "malformed" in caplog.messages[0]

    def test_a_file_that_cannot_be_statted_or_read_warns_once(
        self, tmp_path, monkeypatch, caplog
    ):
        # As in a state directory without search permission.
        p = _path(tmp_path)
        cal.save(p, Report("x", {}))

        def denied(*_args, **_kwargs):
            raise PermissionError(13, "Permission denied")

        # As at startup, before any other file was found unreadable.
        monkeypatch.setattr(cal, "_unreadable", None)
        monkeypatch.setattr(Path, "stat", denied)
        monkeypatch.setattr(Path, "open", denied)
        assert cal.load(p) is None
        assert cal.load(p) is None
        assert len(caplog.messages) == 1
        assert "unreadable" in caplog.messages[0]

    @pytest.mark.parametrize(
        "cell",
        [
            {"base": {"compiled": True, "ms": 5.0}},
            {"base|CPU": {"compiled": True, "ms": "x"}},
        ],
        ids=["key-without-device", "ms-not-a-number"],
    )
    def test_load_a_cell_the_readers_cannot_use_is_none(self, tmp_path, cell):
        p = _path(tmp_path)
        p.write_text(json.dumps({"fingerprint": "x", "cells": cell}))
        assert cal.load(p) is None


class TestRun:
    def test_measures_every_cell_and_persists(self, tmp_path):
        p = _path(tmp_path)
        table = {
            ("small", "NPU"): CellResult(True, 780.0),
            ("small", "GPU"): CellResult(True, 440.0),
            ("small", "CPU"): CellResult(True, 4800.0),
        }
        measure, calls = _fake_measure(table)
        report = cal.run(
            ["small"],
            ["NPU", "GPU", "CPU"],
            _FP,
            model_mgr=None,
            path=p,
            measure_fn=measure,
        )
        assert calls == [("small", "NPU"), ("small", "GPU"), ("small", "CPU")]
        assert cal.load(p) == report  # persisted

    def test_a_skipped_cell_keeps_what_an_earlier_run_stored(self, tmp_path):
        p = _path(tmp_path)
        cal.save(p, Report(_FP, {"small|NPU": CellResult(True, 780.0)}))
        measure, _ = _fake_measure(
            {
                ("small", "NPU"): CellResult(False, skipped=True),
                ("small", "CPU"): CellResult(True, 4800.0),
            }
        )
        seen = []
        report = cal.run(
            ["small"],
            ["NPU", "CPU"],
            _FP,
            model_mgr=None,
            path=p,
            measure_fn=measure,
            on_progress=lambda d, t, m, dev, r: seen.append(r.skipped),
        )
        assert seen == [True, False]
        assert report.cells["small|NPU"] == CellResult(True, 780.0)
        assert cal.load(p) == report

    def test_the_report_leaves_out_other_models_stored_cells(self, tmp_path):
        p = _path(tmp_path)
        cal.save(p, Report(_FP, {"large-v3-turbo|GPU": CellResult(True, 300.0)}))
        measure, _ = _fake_measure({("base", "CPU"): CellResult(True, 900.0)})
        report = cal.run(
            ["base"], ["CPU"], _FP, model_mgr=None, path=p, measure_fn=measure
        )
        assert set(report.cells) == {"base|CPU"}
        assert set(cal.load(p).cells) == {"base|CPU", "large-v3-turbo|GPU"}

    def test_progress_fires_per_cell(self, tmp_path):
        measure, _ = _fake_measure({("small", "GPU"): CellResult(True, 1.0)})
        seen = []
        cal.run(
            ["small"],
            ["GPU", "CPU"],
            _FP,
            model_mgr=None,
            path=_path(tmp_path),
            measure_fn=measure,
            on_progress=lambda d, t, m, dev, r: seen.append((d, t, m, dev)),
        )
        assert seen == [(1, 2, "small", "GPU"), (2, 2, "small", "CPU")]

    def test_cancel_keeps_partial(self, tmp_path):
        p = _path(tmp_path)
        measure, calls = _fake_measure({("small", "NPU"): CellResult(True, 1.0)})
        # Cancel after the first cell.
        returned = cal.run(
            ["small"],
            ["NPU", "GPU", "CPU"],
            _FP,
            model_mgr=None,
            path=p,
            measure_fn=measure,
            should_cancel=lambda: len(calls) >= 1,
        )
        assert set(returned.cells) == {"small|NPU"}  # reported, not a failed run
        report = cal.load(p)
        assert set(report.cells) == {"small|NPU"}  # only the pre-cancel cell survived

    def test_a_cell_that_cannot_be_saved_does_not_end_the_run(self, tmp_path):
        # A full disk or unwritable state dir costs the stored ranking, not the run.
        blocked = tmp_path / "state"
        blocked.write_text("not a directory")
        measure, calls = _fake_measure(
            {
                ("small", "GPU"): CellResult(True, 440.0),
                ("small", "CPU"): CellResult(True, 4800.0),
            }
        )
        report = cal.run(
            ["small"],
            ["GPU", "CPU"],
            _FP,
            model_mgr=None,
            path=blocked / "calibration.json",
            measure_fn=measure,
        )
        assert calls == [("small", "GPU"), ("small", "CPU")]
        assert report.cells == {
            "small|GPU": CellResult(True, 440.0),
            "small|CPU": CellResult(True, 4800.0),
        }

    def test_each_cell_is_measured_with_the_callers_model_manager(self, tmp_path):
        model_mgr = FakeModelManager()
        seen = []
        cal.run(
            ["small"],
            ["CPU"],
            _FP,
            model_mgr,
            path=_path(tmp_path),
            measure_fn=lambda m, d, mgr: seen.append(mgr) or CellResult(True, 1.0),
        )
        assert seen == [model_mgr]

    def test_stale_fingerprint_wiped_on_run(self, tmp_path):
        p = _path(tmp_path)
        cal.save(p, Report("OLD_FP", {"small|GPU": CellResult(True, 1.0)}))
        measure, _ = _fake_measure({("base", "GPU"): CellResult(True, 2.0)})
        cal.run(["base"], ["GPU"], _FP, model_mgr=None, path=p, measure_fn=measure)
        assert set(cal.load(p).cells) == {"base|GPU"}  # old-fingerprint cell dropped


class TestRanking:
    def test_sorts_by_ms_and_omits_failed_and_unbenchmarked(self):
        report = Report(
            _FP,
            {
                "small|NPU": CellResult(True, 780.0),
                "small|GPU": CellResult(True, 440.0),
                "small|CPU": CellResult(True, 4800.0),
                # Failed, so omitted; the timed cells after it still rank.
                "turbo|CPU": CellResult(False),
                "turbo|NPU": CellResult(True, 856.0),
                "turbo|GPU": CellResult(True, 1821.0),
                "base|GPU": CellResult(
                    True, None
                ),  # warmed but not benchmarked, so omitted
            },
        )
        r = cal.ranking(report)
        assert r["small"] == ["GPU", "NPU", "CPU"]
        assert r["turbo"] == ["NPU", "GPU"]  # turbo fastest on NPU; CPU omitted
        assert "base" not in r

    def test_current_ranking_none_on_fingerprint_mismatch(
        self, tmp_path, monkeypatch, caplog
    ):
        monkeypatch.setattr(cal, "_stale_logged", set())
        p = _path(tmp_path)
        cal.save(p, Report("OLD", {"small|GPU": CellResult(True, 1.0)}))
        assert cal.current_ranking(_FP, path=p) is None
        assert cal.current_ranking(_FP, path=p) is None
        assert caplog.messages == [
            f"Calibration from another fingerprint ignored (stored OLD, now {_FP})"
        ]
        assert cal.current_ranking("OLD", path=p) == {"small": ["GPU"]}


class TestRealtimeMultiple:
    def test_sample_seconds_over_measured_time(self):
        # 5 s of audio in 1000 ms = 5x realtime; in 500 ms = 10x; in 5000 ms = 1x.
        assert cal.realtime_multiple(1000.0) == 5.0
        assert cal.realtime_multiple(500.0) == 10.0
        assert cal.realtime_multiple(5000.0) == 1.0


class TestWinners:
    def test_fastest_device_and_time_per_model(self):
        report = Report(
            _FP,
            {
                "small|NPU": CellResult(True, 780.0),
                "small|GPU": CellResult(True, 440.0),  # fastest for small
                "small|CPU": CellResult(True, 4800.0),
                "turbo|NPU": CellResult(True, 856.0),  # fastest for turbo
                "turbo|GPU": CellResult(True, 1821.0),
                "turbo|CPU": CellResult(False),  # failed, so ignored
                "base|GPU": CellResult(True, None),  # warmed, not timed, so ignored
            },
        )
        w = cal.winners(report)
        assert w["small"] == ("GPU", 440.0)
        assert w["turbo"] == ("NPU", 856.0)
        assert "base" not in w  # no timed cell


class TestSpeedLabel:
    def test_one_decimal_under_ten_integer_at_or_above(self):
        assert cal.speed_label(862.0) == "5.8× realtime"  # 5000/862 ≈ 5.8
        assert cal.speed_label(500.0) == "10× realtime"  # exactly 10
        assert cal.speed_label(357.0) == "14× realtime"  # 5000/357 ≈ 14


class TestMeasuringLabel:
    def test_capitalises_the_model_and_names_the_device(self):
        assert cal.measuring_label("small", "GPU") == "Measuring Small on GPU…"


class TestDevicePhrase:
    def test_joins_with_commas_and_a_trailing_and(self):
        assert cal.device_phrase(["CPU", "NPU", "GPU"]) == "CPU, NPU and GPU"
        assert cal.device_phrase(["NPU", "GPU"]) == "NPU and GPU"

    def test_one_or_none_falls_back_sensibly(self):
        assert cal.device_phrase(["CPU"]) == "CPU"
        assert cal.device_phrase([]) == "CPU"


class TestVerdict:
    def test_mixed_devices_leads_with_the_best_speed(self):
        winners = {"tiny": ("NPU", 357.0), "small": ("GPU", 833.0)}
        # The best speed is 5000/357 ≈ 14x.
        assert (
            cal.verdict(winners, measured=6, total=6)
            == "Quoth will run each model on its fastest device, up to 14× realtime."
        )

    def test_all_same_device_says_so(self):
        winners = {"tiny": ("NPU", 357.0), "small": ("NPU", 862.0)}
        assert (
            cal.verdict(winners, measured=4, total=4)
            == "Every model runs fastest on your NPU."
        )

    def test_a_single_model_is_named_rather_than_called_every_model(self):
        assert (
            cal.verdict({"small": ("GPU", 440.0)}, measured=2, total=2)
            == "Small runs fastest on your GPU."
        )

    def test_cpu_only_is_framed_as_ready_not_missing(self):
        winners = {"tiny": ("CPU", 1200.0), "small": ("CPU", 4800.0)}
        assert (
            cal.verdict(winners, measured=2, total=2)
            == "Quoth is tuned for your CPU and ready to go."
        )

    def test_partial_run_prefixes_the_measurement_count(self):
        winners = {"tiny": ("NPU", 357.0)}
        assert cal.verdict(winners, measured=6, total=9).startswith(
            "Based on 6 of 9 measurements. "
        )

    def test_no_winners_ends_on_a_safe_default(self):
        assert (
            cal.verdict({}, measured=0, total=9)
            == "The benchmark didn't finish. Quoth will use a safe default."
        )


class TestOptimizedLine:
    def test_accelerator_winner_gives_device_and_speed_without_the_model(self):
        # 5000/440 ≈ 11x. Onboarding offers no model choice, so naming one here
        # would introduce a concept the user has not met.
        assert (
            cal.optimized_line({"small": ("GPU", 440.0)}, "small")
            == "Quoth runs fastest on your GPU: 11× realtime."
        )

    def test_cpu_winner_is_framed_as_ready(self):
        assert (
            cal.optimized_line({"tiny": ("CPU", 4800.0)}, "tiny")
            == "Quoth is tuned for your CPU and ready to go."
        )

    def test_unmeasurable_model_falls_back_to_safe_default(self):
        assert (
            cal.optimized_line({}, "small")
            == "Quoth couldn't measure your hardware and will use a safe default."
        )


class TestIsWarm:
    def test_true_only_for_compiled_cell_matching_fingerprint(self, tmp_path):
        p = _path(tmp_path)
        cal.save(
            p,
            Report(
                _FP,
                {"turbo|NPU": CellResult(True, 856.0), "turbo|CPU": CellResult(False)},
            ),
        )
        assert cal.is_warm("turbo", "NPU", _FP, path=p) is True
        assert cal.is_warm("turbo", "CPU", _FP, path=p) is False  # failed cell
        assert cal.is_warm("turbo", "GPU", _FP, path=p) is False  # never measured
        assert cal.is_warm("turbo", "NPU", "OTHER_FP", path=p) is False  # stale

    def test_mark_warm_marks_a_cell_warm(self, tmp_path):
        p = _path(tmp_path)
        cal.mark_warm("small", "NPU", _FP, path=p)
        assert cal.is_warm("small", "NPU", _FP, path=p) is True


class TestMarkWarm:
    def test_preserves_existing_ms(self, tmp_path):
        # An evicted but measured cell (compiled=False, ms set) keeps its timing.
        p = _path(tmp_path)
        cal.save(p, Report(_FP, {"turbo|NPU": CellResult(False, 856.0)}))
        cal.mark_warm("turbo", "NPU", _FP, path=p)
        cell = cal.load(p).cells["turbo|NPU"]
        assert cell.compiled is True and cell.ms == 856.0

    def test_never_benchmarked_stays_untimed(self, tmp_path):
        p = _path(tmp_path)
        cal.mark_warm("small", "NPU", _FP, path=p)
        assert cal.load(p).cells["small|NPU"].ms is None

    def test_unwritable_storage_is_not_an_error(self, tmp_path):
        blocked = tmp_path / "state"
        blocked.write_text("not a directory")
        cal.mark_warm("small", "NPU", _FP, path=blocked / "calibration.json")


class TestEvictWarmth:
    """Clearing the OpenVINO blob cache makes NPU cells cold and keeps every timing."""

    def test_npu_goes_cold_gpu_untouched_ranking_survives(self, tmp_path):
        p = _path(tmp_path)
        cal.save(
            p,
            Report(
                _FP,
                {
                    "turbo|NPU": CellResult(True, 856.0),
                    "turbo|GPU": CellResult(True, 1821.0),
                },
            ),
        )
        cal.evict_warmth(_FP, path=p)
        assert (
            cal.is_warm("turbo", "NPU", _FP, path=p) is False
        )  # blob cleared, so the gate re-fires
        assert cal.is_warm("turbo", "GPU", _FP, path=p) is True  # no blob, untouched
        assert cal.ranking(cal.load(p)) == {
            "turbo": ["NPU", "GPU"]
        }  # timings preserved

    def test_noop_when_empty(self, tmp_path):
        p = _path(tmp_path)
        cal.evict_warmth(_FP, path=p)  # no calibration file, no crash
        assert cal.load(p) is None

    def test_evict_then_recompile_preserves_benchmark(self, tmp_path):
        p = _path(tmp_path)
        cal.save(p, Report(_FP, {"turbo|NPU": CellResult(True, 856.0)}))
        cal.evict_warmth(_FP, path=p)  # clear cache
        assert cal.is_warm("turbo", "NPU", _FP, path=p) is False
        cal.mark_warm("turbo", "NPU", _FP, path=p)  # warm gate recompiles
        assert cal.is_warm("turbo", "NPU", _FP, path=p) is True
        assert cal.load(p).cells["turbo|NPU"].ms == 856.0  # benchmark intact


class TestWorthOffering:
    """Worth offering only with an accelerator to rank and no fresh ranking yet."""

    def test_accelerator_and_no_ranking_is_worthwhile(self, tmp_path):
        assert (
            cal.worth_offering(["NPU", "GPU", "CPU"], _FP, path=_path(tmp_path)) is True
        )

    def test_cpu_only_is_not_worthwhile(self, tmp_path):
        # One device, nothing to choose: a benchmark would rank a field of one.
        assert cal.worth_offering(["CPU"], _FP, path=_path(tmp_path)) is False

    def test_already_ranked_is_not_worthwhile(self, tmp_path):
        p = _path(tmp_path)
        cal.save(p, Report(_FP, {"small|GPU": CellResult(True, 440.0)}))
        assert cal.worth_offering(["NPU", "GPU", "CPU"], _FP, path=p) is False

    def test_stale_ranking_is_still_worthwhile(self, tmp_path):
        # A ranking under a different fingerprint doesn't count: the machine changed.
        p = _path(tmp_path)
        cal.save(p, Report("OLD_FP", {"small|GPU": CellResult(True, 440.0)}))
        assert cal.worth_offering(["NPU", "GPU", "CPU"], _FP, path=p) is True


@pytest.mark.parametrize(
    ("devices", "note"),
    [
        (["CPU"], ""),
        (["CPU", "GPU"], ""),
        (["CPU", "NPU", "GPU"], " First-time NPU preparation can take longer."),
    ],
)
def test_the_preparation_note_appears_only_where_a_device_compiles(devices, note):
    assert cal.preparation_note(devices) == note
