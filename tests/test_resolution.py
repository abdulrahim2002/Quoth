# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `resolution`, the composer over detection, rules, catalog and calibration.

The pure rules are covered in `test_accel`; these check the composition, above all that
a calibrated ranking overrides the default order.
"""

from quoth import calibration, paths
from quoth.hardware import fingerprint
from quoth.resolution import (
    current_ranking,
    evict_warmth,
    expects_slow_compile,
    mark_compiled,
    provision_for,
    resolve_for,
    worth_offering,
)
from tests.support import FakeModelManager


def _resolve(model, preference, *, encoder, **kwargs):
    """Resolve on the devices the test detects."""
    return resolve_for(model, preference, FakeModelManager(encoder=encoder), **kwargs)


def _calibrate(timings):
    """Store a benchmark run on this machine: each `model|device` cell's best ms."""
    cells = {key: calibration.CellResult(True, ms) for key, ms in timings.items()}
    report = calibration.Report(fingerprint(), cells)
    calibration.save(calibration.default_path(), report)


class TestRuntime:
    def test_npu_honoured_with_encoder(self, detected_devices):
        detected_devices.extend(["NPU", "GPU"])
        r = _resolve("small", "NPU", encoder=True)
        assert r.device == "NPU" and r.degraded is False

    def test_npu_without_encoder_degrades_to_gpu(self, detected_devices):
        detected_devices.extend(["NPU", "GPU"])
        r = _resolve("small", "NPU", encoder=False)
        assert r.device == "GPU" and r.degraded is True

    def test_auto_prefers_gpu_by_default(self, detected_devices):
        # Without calibration the default order (GPU first) applies to every model.
        detected_devices.extend(["NPU", "GPU"])
        assert _resolve("large-v3-turbo", "auto", encoder=True).device == "GPU"

    def test_gpu_missing_here_degrades_to_cpu(self):
        r = _resolve("small", "GPU", encoder=True)
        assert r.device == "CPU" and r.degraded is True


class TestProvisioning:
    def test_assume_encoder_keeps_npu_before_download(self, detected_devices):
        # The download gate must target NPU even though the encoder isn't on disk yet.
        detected_devices.extend(["NPU", "GPU"])
        r = _resolve("small", "NPU", encoder=False, assume_encoder=True)
        assert r.device == "NPU"

    def _provision(self, preference):
        return provision_for("small", preference, FakeModelManager(encoder=False))

    def test_auto_provisions_a_present_npu_over_the_default_gpu(self, detected_devices):
        detected_devices.extend(["NPU", "GPU"])
        assert self._provision("auto") == "NPU"

    def test_auto_without_an_npu_provisions_what_it_runs(self, detected_devices):
        detected_devices.append("GPU")
        assert self._provision("auto") == "GPU"

    def test_an_explicit_choice_provisions_itself(self, detected_devices):
        detected_devices.extend(["NPU", "GPU"])
        assert self._provision("NPU") == "NPU"
        assert self._provision("GPU") == "GPU"


class TestCalibratedRanking:
    def test_auto_uses_measured_best_over_default(self, detected_devices):
        # Calibration measured turbo fastest on the NPU here: auto must pick it over the
        # GPU-first default.
        detected_devices.extend(["NPU", "GPU"])
        _calibrate({"large-v3-turbo|NPU": 300, "large-v3-turbo|GPU": 400})
        r = _resolve("large-v3-turbo", "auto", encoder=True)
        assert r.device == "NPU" and r.degraded is False

    def test_auto_falls_back_to_default_for_unranked_model(self, detected_devices):
        detected_devices.extend(["NPU", "GPU"])
        _calibrate({"small|NPU": 300, "small|GPU": 400})  # no turbo entry
        assert _resolve("large-v3-turbo", "auto", encoder=True).device == "GPU"

    def test_explicit_preference_ignores_ranking(self, detected_devices):
        # An explicit device is the user's call; the ranking only informs `auto`.
        detected_devices.extend(["NPU", "GPU"])
        _calibrate({"small|NPU": 300, "small|GPU": 400})
        assert _resolve("small", "GPU", encoder=True).device == "GPU"


class TestExpectsSlowCompile:
    def test_non_npu_never_compiles(self):
        # CPU/GPU run the GGML graph: no OpenVINO encoder, so nothing to compile.
        assert expects_slow_compile("small", "GPU") is False
        assert expects_slow_compile("small", "CPU") is False

    def test_npu_cold_expects_compile(self):
        # A populated cache, so the cold record alone decides.
        cache = paths.openvino_cache_dir()
        cache.mkdir(parents=True)
        (cache / "blob").write_bytes(b"x")
        assert expects_slow_compile("large-v3-turbo", "NPU") is True

    def test_npu_already_warm_skips(self):
        cache = paths.openvino_cache_dir()
        cache.mkdir(parents=True)
        (cache / "blob").write_bytes(b"x")
        calibration.mark_warm("large-v3-turbo", "NPU", fingerprint())
        assert expects_slow_compile("large-v3-turbo", "NPU") is False

    def test_a_deleted_cache_is_cold_whatever_the_record_says(self):
        calibration.mark_warm("large-v3-turbo", "NPU", fingerprint())
        assert expects_slow_compile("large-v3-turbo", "NPU") is True
        paths.openvino_cache_dir().mkdir(parents=True)
        assert expects_slow_compile("large-v3-turbo", "NPU") is True


class TestMarkCompiled:
    def test_marks_warm_cell_for_npu(self):
        _calibrate({"large-v3-turbo|NPU": 300})
        evict_warmth()
        mark_compiled("large-v3-turbo", "NPU")
        assert calibration.is_warm("large-v3-turbo", "NPU", fingerprint())
        # Re-warming never wipes a benchmark.
        assert current_ranking() == {"large-v3-turbo": ["NPU"]}

    def test_noop_for_non_npu(self):
        mark_compiled("small", "GPU")
        mark_compiled("small", "CPU")
        assert not calibration.default_path().exists()


class TestDelegators:
    """Fingerprint-keyed `calibration` delegators, so the app never calls it."""

    def test_current_ranking_delegates_under_fingerprint(self, detected_devices):
        detected_devices.append("NPU")
        _calibrate({"small|NPU": 300, "small|CPU": 900})
        assert current_ranking() == {"small": ["NPU", "CPU"]}
        detected_devices.append("GPU")  # other devices, so another fingerprint
        assert current_ranking() is None

    def test_worth_offering_delegates_with_devices_and_fingerprint(
        self, detected_devices
    ):
        assert worth_offering() is False  # nothing to rank on the CPU alone
        detected_devices.append("NPU")
        assert worth_offering() is True
        _calibrate({"small|NPU": 300, "small|CPU": 900})
        assert worth_offering() is False

    def test_evict_warmth_delegates_under_fingerprint(self):
        calibration.mark_warm("large-v3-turbo", "NPU", fingerprint())
        evict_warmth()
        assert not calibration.is_warm("large-v3-turbo", "NPU", fingerprint())
