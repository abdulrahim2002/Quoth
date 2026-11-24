# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.downloads`: the single-flight coordinator and `download_async`."""

import errno
import threading
import types
import urllib.error

import pytest

from quoth.downloader import IntegrityError
from quoth.downloads import DownloadCoordinator
from tests.support import DeferredScheduler, FakeScheduler

_RETRY = "Couldn't download Base. Check your connection and try again."


class FakeDownload:
    """A fake `download_async` whose recorded callbacks a test fires synchronously."""

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        models_dir,
        model,
        device,
        scheduler,
        *,
        on_progress,
        on_done,
        on_error,
        on_cancelled=None,
        cancel=None,
    ):
        self.calls.append(
            types.SimpleNamespace(
                models_dir=models_dir,
                model=model,
                device=device,
                on_progress=on_progress,
                on_done=on_done,
                on_error=on_error,
                on_cancelled=on_cancelled,
                cancel=cancel,
            )
        )
        return None


def _coord(download):
    return DownloadCoordinator("/models", FakeScheduler(), download=download)


class TestSingleFlight:
    def test_starts_one_download(self):
        fake = FakeDownload()
        coordinator = _coord(fake)
        assert not coordinator.is_downloading("base")
        coordinator.start_or_join("base", "CPU")
        assert coordinator.is_downloading("base")
        assert not coordinator.is_downloading("small")
        assert len(fake.calls) == 1 and fake.calls[0].model == "base"
        assert fake.calls[0].models_dir == "/models"

    def test_second_request_joins_not_restarts(self):
        fake = FakeDownload()
        c = _coord(fake)
        c.start_or_join("base", "CPU")
        c.start_or_join("base", "CPU")
        assert len(fake.calls) == 1  # single-flight: no second worker on the same file

    def test_different_models_each_download(self):
        fake = FakeDownload()
        c = _coord(fake)
        c.start_or_join("base", "CPU")
        c.start_or_join("small", "CPU")
        assert len(fake.calls) == 2


class TestFanOut:
    def test_a_failing_subscriber_does_not_silence_the_rest(self, callback_raises):
        fake = FakeDownload()
        c = _coord(fake)

        def broken(*_args):
            raise RuntimeError("view gone")

        progress, done = [], []
        c.start_or_join("base", "CPU", on_progress=broken, on_done=broken)
        c.start_or_join(
            "base",
            "CPU",
            on_progress=progress.append,
            on_done=lambda: done.append("done"),
        )
        fake.calls[0].on_progress("p1")
        fake.calls[0].on_done()
        assert progress == ["p1"] and done == ["done"]
        assert not c.is_downloading("base")
        assert callback_raises == ["Callback broken raised"] * 2

    def test_progress_reaches_all_subscribers(self):
        fake = FakeDownload()
        c = _coord(fake)
        a, b = [], []
        c.start_or_join("base", "CPU", on_progress=a.append)
        c.start_or_join("base", "CPU", on_progress=b.append)  # joins
        fake.calls[0].on_progress("p1")
        assert a == ["p1"] and b == ["p1"]

    def test_late_subscriber_is_caught_up(self):
        fake = FakeDownload()
        c = _coord(fake)
        c.start_or_join("base", "CPU")
        fake.calls[0].on_progress("p1")
        late = []
        c.start_or_join("base", "CPU", on_progress=late.append)  # joins after p1
        assert late == ["p1"]

    def test_done_and_error_reach_every_subscriber(self):
        fake = FakeDownload()
        c = _coord(fake)
        done, errs = [], []
        c.start_or_join("base", "CPU", on_done=lambda: done.append("a"))
        c.start_or_join("base", "CPU", on_done=lambda: done.append("b"))
        fake.calls[0].on_done()
        c.start_or_join("small", "CPU", on_error=errs.append)
        fake.calls[1].on_error("boom")
        assert done == ["a", "b"] and errs == ["boom"]

    def test_a_joiner_arriving_during_progress_is_not_called_twice(self):
        fake = FakeDownload()
        c = _coord(fake)
        seen = []

        # This subscriber joins the same model from inside its own progress callback,
        # appending to the list the fan-out is walking.
        def greedy(p):
            seen.append(p)
            c.start_or_join("base", "CPU", on_progress=seen.append)

        c.start_or_join("base", "CPU", on_progress=greedy)
        fake.calls[0].on_progress("p1")
        # greedy once, plus the joiner's one catch-up call. Without the copy the joiner
        # is also reached by the walk in progress and sees "p1" twice.
        assert seen == ["p1", "p1"]


class TestAfterItEnds:
    def test_a_later_request_downloads_again(self):
        # No replay of an old result: the model may have been deleted since. A re-run
        # skips files already on disk, so a present model costs no network.
        for end in ("on_done", "on_error"):
            fake = FakeDownload()
            c = _coord(fake)
            c.start_or_join("base", "CPU")
            getattr(fake.calls[0], end)(*([] if end == "on_done" else ["boom"]))
            assert not c.is_downloading("base")
            c.start_or_join("base", "CPU")
            assert c.is_downloading("base")
            assert len(fake.calls) == 2, end


class TestCancel:
    def test_leaving_keeps_the_transfer_for_others(self):
        fake = FakeDownload()
        c = _coord(fake)
        leaver = c.start_or_join(
            "base",
            "CPU",
            on_done=lambda: pytest.fail("a subscriber that left was told"),
        )
        done = []
        c.start_or_join("base", "CPU", on_done=lambda: done.append("done"))
        leaver.cancel()
        assert not fake.calls[0].cancel.is_set()
        fake.calls[0].on_done()
        assert done == ["done"]  # the one that left heard nothing

    def test_the_last_to_leave_stops_the_transfer(self):
        fake = FakeDownload()
        coordinator = _coord(fake)
        sub = coordinator.start_or_join("base", "CPU")
        sub.cancel()
        assert fake.calls[0].cancel.is_set()
        assert not coordinator.is_downloading("base")
        fake.calls[0].on_cancelled()
        assert not coordinator.is_downloading("base")
        coordinator.start_or_join("base", "CPU")  # forgotten, so this starts afresh
        assert len(fake.calls) == 2

    def test_a_request_during_unwind_restarts_after_it(self):
        # The cancelled worker still deletes its partial on the way out, so a new
        # request must not share it: it gets a fresh transfer once that one ends.
        fake = FakeDownload()
        c = _coord(fake)
        c.start_or_join("base", "CPU").cancel()
        done = []
        c.start_or_join("base", "NPU", on_done=lambda: done.append("done"))
        assert len(fake.calls) == 1  # waits; does not race the unwinding worker
        fake.calls[0].on_cancelled()
        assert len(fake.calls) == 2
        # The waiting request is the fresh transfer's first, so its device decides.
        assert (fake.calls[1].model, fake.calls[1].device) == ("base", "NPU")
        fake.calls[1].on_done()
        assert done == ["done"]

    def test_leaving_an_ended_transfer_does_not_touch_a_later_one(self):
        fake = FakeDownload()
        c = _coord(fake)
        stale = c.start_or_join("base", "CPU")
        fake.calls[0].on_done()
        c.start_or_join("base", "CPU")
        stale.cancel()
        assert not fake.calls[1].cancel.is_set()


def test_a_download_reports_progress_then_done_through_the_scheduler():
    from quoth.downloads import download_async

    cancel = threading.Event()
    received, outcome = [], []

    def download_fn(models_dir, model, device, *, on_progress, cancel):
        received.append((models_dir, model, device, cancel))
        on_progress("p1")

    thread = download_async(
        "/models",
        "base",
        "NPU",
        FakeScheduler(),
        on_progress=outcome.append,
        on_done=lambda: outcome.append("done"),
        on_error=outcome.append,
        on_cancelled=lambda: outcome.append("cancelled"),
        cancel=cancel,
        download_fn=download_fn,
    )
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert received == [("/models", "base", "NPU", cancel)]
    assert outcome == ["p1", "done"]
    assert thread.daemon  # quitting mid-download does not wait for it


def test_a_cancel_racing_the_last_byte_is_still_a_cancel():
    from quoth.downloads import download_async

    cancel = threading.Event()

    def finishes_after_cancel(*a, **k):
        cancel.set()

    outcome = []
    thread = download_async(
        "/nowhere",
        "base",
        "CPU",
        FakeScheduler(),
        on_progress=lambda p: None,
        on_done=lambda: outcome.append("done"),
        on_error=outcome.append,
        on_cancelled=lambda: outcome.append("cancelled"),
        cancel=cancel,
        download_fn=finishes_after_cancel,
    )
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert outcome == ["cancelled"]


def test_a_download_reports_through_the_main_loop():
    from quoth.downloads import download_async

    def lands(*a, on_progress, **k):
        on_progress("half")

    scheduler, seen = DeferredScheduler(), []
    thread = download_async(
        "/nowhere",
        "base",
        "CPU",
        scheduler,
        on_progress=seen.append,
        on_done=lambda: seen.append("done"),
        on_error=seen.append,
        download_fn=lands,
    )
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert seen == []
    scheduler.fire_dispatches()
    assert seen == ["half", "done"]


@pytest.mark.parametrize(
    ("error", "traceback"),
    [
        (OSError(errno.EIO, "Input/output error"), False),
        (RuntimeError("defect"), True),
    ],
)
def test_failed_download_logs_once_with_traceback_only_for_defects(
    error, traceback, caplog
):
    """A network, HTTP, disk or integrity failure logs once without a traceback.

    Any other exception keeps its traceback; `on_error` tells the user either way.
    """
    from quoth.downloads import download_async

    def raiser(*a, **k):
        raise error

    if not traceback:
        error.add_note("ggml-base.bin")  # as the downloader names the failed file
    caplog.set_level("WARNING")
    errors = []
    thread = download_async(
        "/nowhere",
        "base",
        "CPU",
        FakeScheduler(),
        on_progress=lambda p: None,
        on_done=lambda: None,
        on_error=errors.append,
        download_fn=raiser,
    )
    thread.join(timeout=5)
    assert not thread.is_alive()
    [record] = caplog.records
    assert record.levelname == "ERROR"
    assert (record.exc_info is not None) is traceback
    if not traceback:
        assert caplog.messages == [
            "Model download of base failed on ggml-base.bin: "
            "OSError: [Errno 5] Input/output error"
        ]
    assert errors == [_RETRY]


def test_a_malformed_proxy_setting_never_reaches_the_log(
    tmp_path, malformed_proxy, caplog
):
    from quoth.downloads import download_async

    errors = []
    thread = download_async(
        tmp_path,
        "base",
        "CPU",
        FakeScheduler(),
        on_progress=lambda p: None,
        on_done=lambda: None,
        on_error=errors.append,
    )
    thread.join(timeout=5)
    assert errors == [_RETRY]
    assert caplog.messages[-1] == (
        "Model download of base failed on manifest.json: InvalidURL"
    )
    assert not any("s3cret" in message for message in caplog.messages)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (urllib.error.URLError(ConnectionRefusedError(111, "refused")), _RETRY),
        (
            urllib.error.HTTPError(
                "https://x/ggml-base.bin", 404, "Not Found", {}, None
            ),
            _RETRY,
        ),
        (IntegrityError("set QUOTH_ALLOW_UNVERIFIED=1 to override"), _RETRY),
        (
            OSError(errno.ENOSPC, "No space left on device", "/m/ggml-base.bin.tmp"),
            "Couldn't download Base. Free up disk space and try again.",
        ),
        (
            OSError(errno.EDQUOT, "Disk quota exceeded", "/m/ggml-base.bin.tmp"),
            "Couldn't download Base. Free up disk space and try again.",
        ),
    ],
)
def test_the_user_is_told_what_to_do_not_the_raw_error(error, message):
    from quoth.downloads import download_async

    def raiser(*a, **k):
        raise error

    errors = []
    thread = download_async(
        "/nowhere",
        "base",
        "CPU",
        FakeScheduler(),
        on_progress=lambda p: None,
        on_done=lambda: None,
        on_error=errors.append,
        download_fn=raiser,
    )
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == [message]


def test_a_worker_that_cannot_start_fails_the_transfer():
    def unstartable(*_args, **_kwargs):
        raise RuntimeError("can't start new thread")

    c = _coord(unstartable)
    errors = []
    c.start_or_join("base", "CPU", on_error=errors.append)
    assert errors == [_RETRY]
    assert not c.is_downloading("base")
