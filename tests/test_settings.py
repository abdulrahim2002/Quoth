# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `settings`: model and device changes and on-demand downloads.

A real `DownloadCoordinator` runs over `FakeScheduler`, a fake download and a recording
`SettingsView`, with no GTK or network; tests join its workers in `events["threads"]`.
"""

import threading
import types

import pytest

from quoth.config import Config
from quoth.downloader import DownloadCancelled, DownloadProgress
from quoth.downloads import DownloadCoordinator, download_async
from quoth.settings import ModelSwitcher, SettingsController, SettingsView
from tests.support import FakeModelManager, FakeScheduler, join_thread

# Change recorders to disconnect after each test, so none hears a later test's writes.
_recorders: list[tuple[Config, int]] = []


@pytest.fixture(autouse=True)
def _disconnect_recorders():
    yield
    while _recorders:
        config, handler = _recorders.pop()
        config.settings.disconnect(handler)


def _config(**overrides):
    config = Config()
    config.model = "small"
    config.device = "CPU"  # concrete, so resolution returns CPU without a probe
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def _ok_download(fractions=(0.5, 1.0)):
    def fn(models_dir, model, device, on_progress=None, cancel=None):
        for f in fractions:
            if on_progress:
                on_progress(DownloadProgress(f, 0, 0, 0.0))

    return fn


def _fail_download(msg="boom"):
    def fn(models_dir, model, device, on_progress=None, cancel=None):
        raise RuntimeError(msg)

    return fn


def _cancellable_download():
    """Block until cancelled, then raise `DownloadCancelled` like `_download_file`.

    It never succeeds, so a cancel that misses it leaves the worker for `_wait` to fail.
    """

    def fn(models_dir, model, device, on_progress=None, cancel=None):
        cancel.wait()
        raise DownloadCancelled()

    return fn


def _held(gate):
    """Hold the download until `gate` is set; raise `DownloadCancelled` if cancelled."""

    def fn(models_dir, model, device, on_progress=None, cancel=None):
        while not gate.is_set():
            if cancel is not None and cancel.wait(0.01):
                raise DownloadCancelled()

    return fn


def _make(*, config=None, model_mgr=None, download_fn=None, switcher=None):
    """Build a controller plus a record of view effects and notifications."""
    events = {
        "changed": [],  # what the app would apply
        "reconciled": [],  # the engine refitted to files on disk, with no change
        "download_active": [],
        "download_progress": [],
        "toast": [],
        "revert": [],
        "model_changed": [],
        "announced": [],  # desktop notifications, when no window is attached
        "threads": [],
    }
    view = SettingsView(
        set_download_active=lambda m: events["download_active"].append(m),
        set_download_progress=lambda p: events["download_progress"].append(p),
        toast=lambda m: events["toast"].append(m),
        revert=lambda k: events["revert"].append(k),
        on_model_changed=lambda: events["model_changed"].append(True),
    )
    # Built per call: a config writes the shared settings, and a fake shared between
    # tests would carry one test's changes into the next.
    config = config or _config()
    model_mgr = model_mgr or FakeModelManager(encoder=False)
    download_fn = download_fn or _ok_download()

    def download(*args, **kw):
        thread = download_async(*args, download_fn=download_fn, **kw)
        events["threads"].append(thread)
        return thread

    handler = config.connect_changed(
        lambda name: events["changed"].append((name, getattr(config, name)))
    )
    _recorders.append((config, handler))

    def reconcile():
        events["reconciled"].append(True)

    switcher = switcher or ModelSwitcher(
        config,
        model_mgr,
        DownloadCoordinator(model_mgr.models_dir, FakeScheduler(), download=download),
        reconcile=reconcile,
        notify=events["announced"].append,
    )
    controller = SettingsController(
        config,
        model_mgr,
        switcher,
        view,
        reconcile=reconcile,
    )
    controller.open()  # as the window does once its widgets exist
    events["switcher"] = switcher
    return controller, config, events


def _wait(events) -> None:
    """Join every download worker started so far."""
    join_thread(types.SimpleNamespace(threads=events["threads"]), "threads")


def _recording_download(devices_asked, gate=None):
    """A download that records the device each request fetched for, then succeeds.

    With a `gate` it lands only once the gate is set.
    """

    def fn(models_dir, model, device, on_progress=None, cancel=None):
        devices_asked.append(device)
        if gate is not None:
            gate.wait(2)

    return fn


class TestAssetsReady:
    def _ready(self, model_mgr, device):
        _, _, events = _make(model_mgr=model_mgr)
        return events["switcher"].assets_ready("small", device)

    def test_false_when_not_downloaded(self):
        assert self._ready(FakeModelManager(ggml=False), "CPU") is False

    def test_true_when_downloaded_and_no_encoder_needed(self):
        no_encoder = FakeModelManager(ggml=True, encoder=False)
        assert self._ready(no_encoder, "CPU") is True
        # Vulkan needs no encoder.
        assert self._ready(no_encoder, "GPU") is True

    def test_true_when_encoder_needed_and_present(self):
        assert self._ready(FakeModelManager(ggml=True, encoder=True), "NPU") is True

    def test_false_when_encoder_needed_but_missing(self):
        assert self._ready(FakeModelManager(ggml=True, encoder=False), "NPU") is False


class TestProvisioning:
    """Settings and the tray fetch by one rule; the delete fallback re-homes at once."""

    def test_auto_with_an_npu_fetches_the_encoder_for_a_downloaded_model(
        self, detected_devices
    ):
        detected_devices.extend(["NPU", "GPU"])
        asked = []
        c, config, events = _make(
            config=_config(device="auto"),
            model_mgr=FakeModelManager(ggml=True, encoder=False),
            download_fn=_recording_download(asked),
        )
        c.select_model("base")
        _wait(events)
        assert asked == ["NPU"]
        assert config.model == "base"

    def test_auto_without_an_npu_applies_at_once(self, detected_devices):
        detected_devices.append("GPU")
        c, config, events = _make(
            config=_config(device="auto"),
            model_mgr=FakeModelManager(ggml=True, encoder=False),
        )
        c.select_model("base")
        assert config.model == "base"
        assert events["threads"] == []

    def test_a_switch_with_no_window_fetches_the_npu_encoder_first(
        self, detected_devices
    ):
        # The tray's path: an explicit NPU choice must not drop to another device.
        detected_devices.append("NPU")
        asked, gate = [], threading.Event()
        c, config, events = _make(
            config=_config(model="small", device="NPU"),
            model_mgr=FakeModelManager(ggml=True, encoder=False),
            download_fn=_recording_download(asked, gate),
        )
        c.close()
        assert events["switcher"].select("model", "base") is True
        assert config.model == "small"  # waits for the encoder
        gate.set()
        _wait(events)
        assert asked == ["NPU"]
        assert config.model == "base"
        assert events["announced"] == ["Downloading Base for the NPU", "Base ready"]

    def test_a_switch_with_no_window_is_refused_by_notification(self):
        gate = threading.Event()
        c, config, events = _make(
            model_mgr=FakeModelManager(ggml=False), download_fn=_held(gate)
        )
        c.select_model("base")
        c.close()
        assert events["switcher"].select("model", "tiny") is False
        assert events["announced"] == ["Base is still downloading"]
        gate.set()
        _wait(events)
        assert config.model == "base"

    def test_the_delete_fallback_switches_at_once_without_a_download(
        self, detected_devices
    ):
        detected_devices.append("NPU")
        mgr, _deleted = _stateful_mgr({"tiny", "base"})
        asked = []
        c, config, events = _make(
            config=_config(model="base", device="NPU"),
            model_mgr=mgr,
            download_fn=_recording_download(asked),
        )
        c.delete_model("base")
        assert config.model == "tiny"
        _wait(events)
        assert asked == []

    def test_the_delete_fallback_holds_while_another_model_downloads(self):
        mgr, _deleted = _stateful_mgr({"tiny", "base"})
        c, config, events = _make(
            config=_config(model="base"),
            model_mgr=mgr,
            download_fn=_cancellable_download(),
        )
        c.select_model("small")
        c.delete_model("base")
        assert config.model == "tiny"
        c.cancel_download()
        _wait(events)
        assert config.model == "tiny"

    def test_a_failed_encoder_is_left_to_the_device_resolve_to_explain(
        self, detected_devices
    ):
        # Applying NPU re-resolves the device, and the app names the one used instead;
        # a second, differently worded toast here would contradict it.
        detected_devices.extend(["NPU", "GPU"])
        c, config, events = _make(
            config=_config(device="CPU"),
            model_mgr=FakeModelManager(ggml=False, encoder=False),
            download_fn=_ok_download(),
        )
        c.select_device("NPU")
        _wait(events)
        assert config.device == "NPU"
        assert events["changed"] == [("device", "NPU")]
        assert events["toast"] == []

    def test_a_failed_encoder_under_auto_says_only_ready(self, detected_devices):
        detected_devices.extend(["NPU", "GPU"])
        c, config, events = _make(
            config=_config(device="auto"),
            model_mgr=FakeModelManager(ggml=False, encoder=False),
            download_fn=_ok_download(),
        )
        c.select_model("base")
        _wait(events)
        assert config.model == "base"
        assert events["toast"] == ["Base ready"]

    def test_a_failed_encoder_after_a_device_change_to_auto_says_nothing(
        self, detected_devices
    ):
        detected_devices.extend(["NPU", "GPU"])
        c, config, events = _make(
            config=_config(model="small", device="CPU"),
            model_mgr=FakeModelManager(ggml=True, encoder=False),
            download_fn=_ok_download(),
        )
        c.select_device("auto")
        _wait(events)
        assert events["download_active"] == ["small", None]  # the encoder came first
        assert config.device == "auto"
        assert events["toast"] == []


class TestSelectWhenReady:
    def test_applies_directly_without_download(self):
        on_disk, _deleted = _stateful_mgr({"base"})  # the chosen model only
        c, config, events = _make(config=_config(model="small"), model_mgr=on_disk)
        c.select_model("base")
        assert config.model == "base"
        assert events["changed"] == [("model", "base")]
        assert events["threads"] == []  # no download spawned
        assert events["download_active"] == []  # UI untouched, no spinner

    def test_select_device_applies_directly_when_ready(self):
        c, config, events = _make(model_mgr=FakeModelManager(ggml=True))
        c.select_device("auto")
        assert config.device == "auto"
        assert events["changed"] == [("device", "auto")]
        assert events["threads"] == []
        assert events["model_changed"] == []  # device row self-updates; no rebuild


class TestSelectWithDownload:
    def test_model_download_then_apply(self):
        c, config, events = _make(
            config=_config(model="small", device="CPU"),
            model_mgr=FakeModelManager(ggml=False),
            download_fn=_ok_download(),
        )
        c.select_model("base")
        _wait(events)
        assert config.model == "base"
        assert events["changed"] == [("model", "base")]
        assert events["download_active"] == ["base", None]  # spinner raised, cleared
        # Reset before download.
        assert events["download_progress"][0] == DownloadProgress(0.0, 0, 0, 0.0)
        assert (
            events["model_changed"] == []
        )  # the None above re-renders; no separate refresh
        assert events["toast"] == ["Base ready"]  # download confirms completion
        assert events["reconciled"] == []  # the change itself refits the engine

    def test_a_download_for_the_model_already_set_refits_the_engine(self):
        # Re-fetching the configured model after its deletion changes no setting.
        c, config, events = _make(
            config=_config(model="small"),
            model_mgr=FakeModelManager(ggml=False),
        )
        c.select_model("small")
        _wait(events)
        assert config.model == "small"
        assert events["changed"] == []
        assert events["reconciled"] == [True]
        assert events["toast"] == ["Small ready"]

    def test_npu_download_with_the_encoder_has_no_toast(self, detected_devices):
        detected_devices.append("NPU")  # the NPU path is what this checks
        c, config, events = _make(
            config=_config(model="small", device="CPU"),
            model_mgr=FakeModelManager(ggml=False),
            download_fn=_ok_download(),
        )
        c.select_device("NPU")
        _wait(events)
        assert config.device == "NPU"
        assert events["toast"] == []

    def test_npu_model_download_without_encoder_says_only_ready(self, detected_devices):
        detected_devices.append("NPU")  # the NPU path is what this checks
        c, config, events = _make(
            config=_config(model="small", device="NPU"),
            model_mgr=FakeModelManager(ggml=False, encoder=False),
            download_fn=_ok_download(),
        )
        c.select_model("base")
        _wait(events)
        assert config.model == "base"  # still applied (CPU fallback)
        assert events["toast"] == ["Base ready"]

    def test_progress_reaches_the_view_after_a_reset(self):
        c, _, events = _make(
            model_mgr=FakeModelManager(ggml=False),
            download_fn=_ok_download(fractions=(0.25, 0.5, 1.0)),
        )
        c.select_model("base")
        _wait(events)
        # 0.0 reset on the main thread, then the download fractions.
        assert [p.fraction for p in events["download_progress"]] == [
            0.0,
            0.25,
            0.5,
            1.0,
        ]


class TestDownloadFailure:
    def test_failure_reverts_and_leaves_config_unchanged(self):
        c, config, events = _make(
            config=_config(model="small"),
            model_mgr=FakeModelManager(ggml=False),
            download_fn=_fail_download("network down"),
        )
        c.select_model("base")
        _wait(events)
        assert config.model == "small"  # unchanged
        assert events["changed"] == []  # app not told
        assert events["revert"] == ["model"]  # dropdown restored
        assert events["download_active"] == ["base", None]  # spinner shown then cleared
        # The error's own text stays in the log; the user is told what to do.
        assert events["toast"] == [
            "Couldn't download Base. Check your connection and try again."
        ]


def _stateful_mgr(downloaded):
    """Return a model manager fake over a mutable set, and the models it deleted.

    `delete_model` removes from the set, so the post-delete fallback sees what remains.
    """
    present = set(downloaded)
    deleted = []

    def _delete(m):
        deleted.append(m)
        present.discard(m)

    mgr = types.SimpleNamespace(
        is_downloaded=lambda m: m in present,
        has_openvino=lambda m: False,
        openvino_path=lambda m: None,
        models_dir="/models",
        delete_model=_delete,
    )
    return mgr, deleted


class TestDeleteModel:
    def test_delete_inactive_model_removes_and_refreshes(self):
        mgr, deleted = _stateful_mgr({"tiny", "base", "small"})
        c, config, events = _make(config=_config(model="small"), model_mgr=mgr)
        c.delete_model("base")
        assert deleted == ["base"]
        assert config.model == "small"  # active unchanged
        assert events["changed"] == []  # no model change, so no context churn
        assert events["toast"] == []
        assert events["model_changed"] == [True]

    def test_delete_active_switches_to_largest_remaining(self):
        mgr, deleted = _stateful_mgr({"tiny", "base", "small"})
        c, config, events = _make(config=_config(model="base"), model_mgr=mgr)
        c.delete_model("base")
        assert deleted == ["base"]
        assert config.model == "small"  # largest of the remaining {tiny, small}
        assert events["changed"] == [("model", "small")]  # frees old ctx, loads it
        assert events["toast"] == ["Deleted Base, now using Small"]
        assert events["model_changed"] == [True]

    def test_delete_last_model_keeps_config_and_invalidates_context(self):
        mgr, deleted = _stateful_mgr({"small"})
        c, config, events = _make(config=_config(model="small"), model_mgr=mgr)
        c.delete_model("small")
        assert deleted == ["small"]
        assert config.model == "small"  # stays put for one-click re-download
        assert events["changed"] == []
        assert events["reconciled"] == [True]  # drops the stale context
        assert events["toast"] == ["Deleted Small: download a model before dictating"]
        assert events["model_changed"] == [True]

    def test_a_model_that_cannot_be_removed_stays_and_says_so(self):
        mgr, _deleted = _stateful_mgr({"base", "small"})

        def read_only(_model):
            raise PermissionError("read-only file system")

        mgr.delete_model = read_only
        c, config, events = _make(config=_config(model="small"), model_mgr=mgr)
        c.delete_model("small")
        assert config.model == "small"
        assert events["changed"] == []
        assert events["toast"] == ["Couldn't delete Small"]
        assert events["model_changed"] == [True]


class TestConcurrentDownload:
    def test_effective_device_shows_the_pending_device(self, detected_devices):
        # The dropdown shows the device being downloaded for (config isn't persisted
        # until it lands); a model download leaves the device as committed.
        detected_devices.append("NPU")
        gate = threading.Event()
        c, config, events = _make(
            config=_config(device="CPU"),
            model_mgr=FakeModelManager(ggml=False),
            download_fn=_held(gate),
        )
        c.select_device("NPU")
        assert (c.effective_device(), config.device) == ("NPU", "CPU")
        gate.set()
        _wait(events)
        gate.clear()
        c.select_model("base")
        assert c.effective_device() == "NPU"  # the committed device, not the model
        gate.set()
        _wait(events)

    def test_second_selection_is_refused_while_downloading(self):
        # A download in flight must block a conflicting change with a toast and revert
        # (not by disabling controls), and must not spawn a second download.
        gate = threading.Event()

        def blocking(models_dir, model, device, on_progress=None, cancel=None):
            gate.wait(2.0)

        c, config, events = _make(
            config=_config(model="small", device="CPU"),
            model_mgr=FakeModelManager(ggml=False),
            download_fn=blocking,
        )
        c.select_model("base")  # download in flight (gate held)

        c.select_device("GPU")  # conflicting change while busy
        assert any("downloading" in t.lower() for t in events["toast"])
        assert events["revert"] == ["device"]  # dropdown snapped back
        assert config.device == "CPU"  # unchanged
        assert events["changed"] == []  # nothing committed yet
        assert len(events["threads"]) == 1  # no second download spawned

        gate.set()
        _wait(events)
        assert config.model == "base"  # the original download finished
        assert events["changed"] == [("model", "base")]


class TestDownloadCancel:
    def test_cancel_reverts_quietly_without_applying(self):
        c, config, events = _make(
            config=_config(model="small", device="CPU"),
            model_mgr=FakeModelManager(ggml=False),
            download_fn=_cancellable_download(),
        )
        c.select_model("base")  # download in flight, waiting on the cancel Event
        c.cancel_download()  # leaves the download (the last one waiting: it stops)
        _wait(events)

        assert config.model == "small"  # not applied
        assert events["changed"] == []  # nothing committed
        assert events["download_active"] == ["base", None]  # spinner raised, cleared
        assert events["revert"] == ["model"]  # dropdown snapped back
        assert events["toast"] == ["Download canceled"]  # not a "failed" toast

    def test_cancel_download_is_a_noop_when_idle(self):
        c, _, events = _make(model_mgr=FakeModelManager(ggml=True))
        c.cancel_download()
        assert events["download_active"] == []
        assert events["toast"] == []
        assert events["revert"] == []


class TestWindowLifetime:
    """A pending model change belongs to the app's `ModelSwitcher`, not one window."""

    def test_closed_window_download_still_applies_and_announces(self):
        gate = threading.Event()
        c, config, events = _make(
            config=_config(model="small", device="CPU"),
            model_mgr=FakeModelManager(ggml=False),
            download_fn=_held(gate),
        )
        c.select_model("base")
        c.close()
        gate.set()
        _wait(events)
        assert config.model == "base"
        assert events["changed"] == [("model", "base")]
        assert events["toast"] == []  # nothing drawn on the closed window
        assert events["announced"] == ["Base ready"]

    def test_reopened_window_shows_and_cancels_the_running_download(self):
        reported = threading.Event()

        def halfway(models_dir, model, device, on_progress=None, cancel=None):
            on_progress(DownloadProgress(0.5, 0, 0, 0.0))
            reported.set()
            cancel.wait()
            raise DownloadCancelled()

        first, config, events = _make(
            config=_config(model="small", device="CPU"),
            model_mgr=FakeModelManager(ggml=False),
            download_fn=halfway,
        )
        first.select_model("base")
        assert reported.wait(2)
        first.close()
        second, _, shown = _make(
            config=config,
            model_mgr=FakeModelManager(ggml=False),
            switcher=first._switcher,
        )
        assert shown["download_active"] == ["base"]  # picked up on open
        assert shown["download_progress"] == [DownloadProgress(0.5, 0, 0, 0.0)]
        second.cancel_download()
        _wait(events)
        assert config.model == "small"
        assert shown["revert"] == ["model"] and shown["toast"] == ["Download canceled"]
        assert events["changed"] == [] and events["announced"] == []
