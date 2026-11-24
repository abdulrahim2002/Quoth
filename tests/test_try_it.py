# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `try_it`, the onboarding step's guided dictation and its choreography.

State-table checks run without a display; the `display` tests drive real widgets over a
fake engine and fake downloads.
"""

import types
from unittest.mock import ANY, Mock, patch

import pytest
from gi.repository import Gtk

from quoth.try_it import (
    _BUTTON_LABEL,
    _BUTTON_STYLE,
    _HEADER,
    _REVEAL_TRANSITION,
    _STATUS_TEXT,
    _VISIBLE,
    Step,
    demo_device,
    outcome,
)
from tests.support import FakeModelManager, FakeScheduler, fake_engine, wait_until


class _Subscription:
    def __init__(self):
        self.cancelled = 0

    def cancel(self):
        self.cancelled += 1


class _Downloads:
    def __init__(self):
        self.calls = []
        self.devices = []

    def start_or_join(self, _model, device, **callbacks):
        self.calls.append(callbacks)
        self.devices.append(device)
        return _Subscription()


class _Download:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        _models_dir,
        model,
        device,
        _scheduler,
        *,
        on_progress,
        on_done,
        on_error,
        on_cancelled,
        cancel,
    ):
        self.calls.append(
            types.SimpleNamespace(
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


def _page(
    *,
    downloaded=True,
    downloads=None,
    engine=None,
    devices=("CPU",),
    on_success=None,
):
    from quoth.config import Config
    from quoth.try_it import TryItPage

    return TryItPage(
        Config(),
        FakeModelManager(ggml=downloaded),
        list(devices),
        engine=engine or fake_engine(),
        downloads=downloads or _Downloads(),
        motion_enabled=False,
        on_success=on_success,
    )


class TestDemoDevice:
    def test_prefers_gpu_when_present(self):
        assert demo_device(["CPU", "GPU", "NPU"]) == "GPU"

    def test_never_npu_even_when_only_accelerator(self):
        # NPU present but no GPU: still CPU, to avoid the one-time compile stall.
        assert demo_device(["CPU", "NPU"]) == "CPU"


class TestOutcome:
    @pytest.mark.parametrize(
        ("text", "peak", "step"),
        [
            ("hello there", 0.9, Step.SUCCESS),
            # Whisper produced words, so the take worked however faint the level was.
            ("hello", 0.0, Step.SUCCESS),
            ("", 0.01, Step.QUIET),
            ("", 0.8, Step.UNCLEAR),
        ],
    )
    def test_step(self, text, peak, step):
        assert outcome(text, peak=peak) is step

    @pytest.mark.parametrize(
        ("db", "expected"),
        [
            (-55, Step.QUIET),
            (-49, Step.QUIET),
            (-48, Step.UNCLEAR),
            (-36, Step.UNCLEAR),
        ],
    )
    def test_rms_meter_levels_classify_empty_takes(self, db, expected):
        from quoth.audio import level_from_db

        level = level_from_db(db)
        assert outcome("", peak=level) is expected
        assert outcome("hello", peak=level) is Step.SUCCESS


class TestVisibility:
    def test_every_step_has_a_rule(self):
        assert set(_VISIBLE) == set(Step)

    def test_prompt_stays_up_while_recording(self):
        # The user is reading it aloud; retiring it mid-take would be hostile.
        assert "prompt" in _VISIBLE[Step.READY]
        assert "prompt" in _VISIBLE[Step.RECORDING]

    def test_transcribing_retires_the_controls(self):
        # Nothing to do while it works, and the pill carries the state.
        assert _VISIBLE[Step.TRANSCRIBING] == frozenset({"pill", "status"})

    def test_pill_is_retired_once_the_words_are_up(self):
        # It is a live meter; after the take it would animate room noise beside the
        # result, competing with the one thing the step exists to show.
        assert "pill" not in _VISIBLE[Step.SUCCESS]
        assert "pill" in _VISIBLE[Step.RECORDING]

    def test_transcript_and_praise_only_on_success(self):
        for step in Step:
            expected = step is Step.SUCCESS
            assert ("field" in _VISIBLE[step]) is expected
            assert ("praise" in _VISIBLE[step]) is expected

    def test_failed_outcomes_offer_the_prompt_again(self):
        for step in (Step.QUIET, Step.UNCLEAR):
            assert {"prompt", "status", "button"} <= _VISIBLE[step]

    def test_waiting_on_the_model_offers_nothing_to_operate(self):
        # The model is still arriving, so there is nothing to record with yet. The
        # loader and the status line report the wait; neither is operable.
        assert _VISIBLE[Step.PREPARING] == frozenset({"loader", "status"})
        assert not {"button", "pill", "prompt"} & _VISIBLE[Step.PREPARING]

    def test_a_failed_download_can_be_retried(self):
        # Onboarding must stay passable, so the step explains and offers a retry
        # rather than dead-ending on a dropped connection.
        assert _VISIBLE[Step.MODEL_MISSING] == frozenset({"button"})
        assert _BUTTON_LABEL[Step.MODEL_MISSING] == "Retry download"

    def test_mic_failure_offers_nothing_to_operate(self):
        # No working capture path, so every control would be a dead end.
        assert _VISIBLE[Step.MIC_FAILED] == frozenset()


class TestLayout:
    def test_hidden_elements_release_their_space(self):
        # Crossfade and NONE keep holding the child's height while hidden, which
        # reserves a gap for every retired element and hollows out the column.
        assert _REVEAL_TRANSITION in {
            Gtk.RevealerTransitionType.SLIDE_UP,
            Gtk.RevealerTransitionType.SLIDE_DOWN,
        }


class TestHeader:
    def test_every_step_has_a_header(self):
        assert set(_HEADER) == set(Step)

    def test_header_retires_once_the_words_are_up(self):
        # Leaving "tap to speak" as the largest thing on screen would instruct the
        # user to do what they just did, and outrank the result while doing it.
        assert _HEADER[Step.SUCCESS] == ("", "")

    def test_waiting_keeps_the_title_but_changes_the_lead(self):
        # Same title so the wait reads as this page getting ready, not a new page;
        # a different lead so it stops instructing a tap that is not available yet.
        title, lead = _HEADER[Step.PREPARING]
        assert title == _HEADER[Step.READY][0]
        assert lead and lead != _HEADER[Step.READY][1]

    def test_header_instructs_while_there_is_something_to_do(self):
        for step in (Step.READY, Step.RECORDING, Step.QUIET, Step.UNCLEAR):
            title, lead = _HEADER[step]
            assert title and lead

    @pytest.mark.parametrize("step", [Step.MODEL_MISSING, Step.MIC_FAILED])
    def test_failures_put_the_explanation_in_the_header(self, step):
        title, lead = _HEADER[step]
        assert title and lead
        # A line another step also shows could not be explaining this failure.
        for other in Step:
            if other is not step:
                assert title != _HEADER[other][0]
                assert lead != _HEADER[other][1]


class TestButtonWeight:
    def test_only_one_primary_action_after_success(self):
        # Continue, in the chrome, is the primary action once the take has worked.
        assert _BUTTON_STYLE[Step.SUCCESS] == ""

    def test_recording_is_the_only_destructive_step(self):
        for step, style in _BUTTON_STYLE.items():
            is_recording = step is Step.RECORDING
            assert (style == "destructive-action") is is_recording

    def test_no_style_for_steps_without_a_button(self):
        for step in _BUTTON_STYLE:
            assert "button" in _VISIBLE[step], f"{step} styles a button it never shows"


class TestCopy:
    def test_every_step_showing_a_button_labels_it(self):
        for step in Step:
            if "button" in _VISIBLE[step]:
                assert _BUTTON_LABEL.get(step)

    def test_every_step_showing_a_status_fills_it(self):
        for step in Step:
            if "status" in _VISIBLE[step]:
                assert _STATUS_TEXT.get(step)

    def test_no_stale_copy_for_hidden_elements(self):
        for step in _BUTTON_LABEL:
            assert "button" in _VISIBLE[step], f"{step} labels a button it never shows"
        for step in _STATUS_TEXT:
            assert "status" in _VISIBLE[step], f"{step} sets status it never shows"


def test_the_model_is_fetched_for_the_device_downloads_provision(
    display, detected_devices
):
    # Under auto with an NPU present the encoder comes too, so Optimize can measure it.
    detected_devices.extend(["NPU", "GPU"])
    downloads = _Downloads()
    page = _page(downloaded=False, downloads=downloads)
    page.start()
    downloads.calls[0]["on_error"]("offline")
    page._button.emit("clicked")
    assert downloads.devices == ["NPU", "NPU"]


def test_download_retry_restores_the_ready_presentation(display):
    from quoth.config import Config
    from quoth.try_it import TryItPage

    downloads = _Downloads()
    engine = fake_engine()
    page = TryItPage(
        Config(),
        FakeModelManager(ggml=False),
        ["CPU"],
        engine=engine,
        downloads=downloads,
        motion_enabled=False,
    )
    page.start()
    downloads.calls[0]["on_error"]("offline")
    assert page._heading.get_label() == "Couldn't download the speech model"
    assert page._button.get_label() == "Retry download"

    # A double click: the second lands on the button as it slides out.
    page._button.emit("clicked")
    page._button.emit("clicked")
    assert page._step is Step.PREPARING
    assert (len(downloads.calls), engine.begin_take.call_count) == (2, 0)
    downloads.calls[1]["on_done"]()

    assert page._step is Step.READY
    assert (page._heading.get_label(), page._description.get_label()) == _HEADER[
        Step.READY
    ]
    assert page._button.get_label() == "Tap to speak"
    assert page._button.get_cursor().get_name() == "pointer"
    page._button.set_sensitive(False)
    assert page._button.get_cursor().get_name() == "default"
    page._button.set_sensitive(True)
    assert page._button.get_cursor().get_name() == "pointer"
    assert page._status.get_label() == ""
    assert {
        key for key, revealer in page._revealers.items() if revealer.get_reveal_child()
    } == _VISIBLE[Step.READY]
    page.stop()


def test_successful_revisit_stays_inert_until_retry(display):
    engine = fake_engine()
    downloads = _Downloads()
    page = _page(downloads=downloads, engine=engine)
    page.start()
    page._button.emit("clicked")
    page._button.emit("clicked")
    engine.on_text("hello")
    page.stop()

    assert page._step is Step.SUCCESS
    # Plain words: a screen reader would read an emoji's name aloud.
    assert page._praise.get_label() == "Nice, that's you dictating, fully offline."
    assert engine.watchers == set()
    assert page._watchdog is None
    page.start()
    assert engine.watchers == set()
    assert downloads.calls == []
    assert page._watchdog is None

    page._button.emit("clicked")
    assert page._step is Step.RECORDING
    assert engine.watchers == {page}
    assert engine.begin_take.call_count == 2
    page.stop()


def test_takes_transcribe_on_the_demo_device(display):
    engine = fake_engine()
    page = _page(engine=engine, devices=("CPU", "GPU", "NPU"))
    page.start()
    page._button.emit("clicked")
    page._button.emit("clicked")
    engine.end_take.assert_called_once_with("GPU", ANY)
    page.stop()


def test_a_tap_as_the_button_retires_leaves_the_take_alone(display):
    engine = fake_engine()
    page = _page(engine=engine)
    page.start()
    for _ in range(3):
        page._button.emit("clicked")
    assert page._step is Step.TRANSCRIBING
    engine.begin_take.assert_called_once_with()
    engine.end_take.assert_called_once_with("CPU", ANY)
    engine.on_text("hello")
    assert page._step is Step.SUCCESS
    page.stop()


def test_failed_take_retries_on_the_open_meter(display):
    engine = fake_engine()
    page = _page(engine=engine)
    page.start()
    page._button.emit("clicked")
    page._button.emit("clicked")
    engine.on_text("")

    assert page._step is Step.QUIET
    page._button.emit("clicked")
    assert page._step is Step.RECORDING
    assert engine.begin_take.call_count == 2
    page.stop()


def test_a_heard_take_with_no_words_is_unclear_not_quiet(display):
    engine = fake_engine()
    page = _page(engine=engine)
    page.start()
    page._button.emit("clicked")
    page.push_level(0.6)
    page._button.emit("clicked")
    engine.on_text("")

    assert page._step is Step.UNCLEAR
    page.stop()


def test_download_progress_replaces_the_waiting_status(display):
    from quoth.downloader import DownloadProgress

    downloads = _Downloads()
    page = _page(downloaded=False, downloads=downloads)
    page.start()
    downloads.calls[0]["on_progress"](
        DownloadProgress(0.5, 50_000_000, 100_000_000, 10_000_000.0)
    )

    assert "(50%)" in page._status.get_label()
    page.stop()


def test_success_retry_keeps_capture_setup_failure_passable(display):
    engine = fake_engine()
    page = _page(engine=engine)
    page.start()
    page._button.emit("clicked")
    page._button.emit("clicked")
    engine.on_text("hello")
    page.stop()

    engine.unavailable = True
    page._button.emit("clicked")
    assert page._step is Step.MIC_FAILED
    assert page._live is False


def test_stop_leaves_prefetch_running_and_ignores_hidden_completion(display):
    from quoth.downloads import DownloadCoordinator

    download = _Download()
    coordinator = DownloadCoordinator("/models", FakeScheduler(), download=download)
    prefetch_done = []
    coordinator.start_or_join(
        "base", "CPU", on_done=lambda: prefetch_done.append("done")
    )
    page = _page(downloaded=False, downloads=coordinator)
    page.start()
    assert len(download.calls) == 1

    page.stop()
    assert page._download is None
    assert page._live is False
    assert page._watchdog is None
    assert not download.calls[0].cancel.is_set()

    download.calls[0].on_done()
    assert prefetch_done == ["done"]
    assert page._step is Step.READY
    assert page._watchdog is None


def test_stop_cancels_the_take_and_drops_late_transcription(display, monkeypatch):
    engine = fake_engine()
    successes = []
    page = _page(
        engine=engine,
        on_success=lambda: successes.append(True),
    )
    page.start()
    page._button.emit("clicked")
    page.stop()
    assert engine.cancel.call_count == 1
    levels = []
    monkeypatch.setattr(page._pill, "push_level", levels.append)
    page.push_level(0.8)
    assert page._step is Step.READY
    assert levels == []

    page.start()
    page._button.emit("clicked")
    page._button.emit("clicked")
    assert page._step is Step.TRANSCRIBING
    page.stop()
    assert engine.cancel.call_count == 2
    engine.on_text("late")

    assert page._step is Step.READY
    assert successes == []
    assert engine.watchers == set()


def test_stop_outside_a_take_cancels_nothing(display):
    # The engine is the app's: leaving the step must not touch work it did not start.
    engine = fake_engine()
    page = _page(engine=engine)
    page.start()
    page.stop()
    engine.cancel.assert_not_called()


def test_synchronous_first_level_does_not_arm_a_watchdog(display):
    page = _page(engine=fake_engine(first_level=0.5))
    with patch("quoth.try_it.GLib.timeout_add") as timeout_add:
        page.start()
    assert page._heard is True
    assert page._watchdog is None
    timeout_add.assert_not_called()
    page.stop()


def test_a_mic_that_starts_after_the_watchdog_recovers(display):
    page = _page()
    with patch("quoth.try_it.GLib.timeout_add") as timeout_add:
        page.start()
    timeout_add.call_args.args[1]()
    assert page._step is Step.MIC_FAILED

    page.push_level(0.4)

    assert page._step is Step.READY
    assert "button" in _VISIBLE[page._step]
    page.stop()


def test_live_motion_change_reaches_views_and_native_revealers(display):
    page = _page()

    page.set_motion_enabled(True)
    assert page._motion_enabled
    assert page._loader._motion
    assert all(
        revealer.get_transition_duration() > 0 for revealer in page._revealers.values()
    )

    page.set_motion_enabled(False)
    assert not page._loader._motion
    assert all(
        revealer.get_transition_duration() == 0 for revealer in page._revealers.values()
    )


def test_the_pill_alone_shows_the_real_level(display, monkeypatch):
    page = _page()
    levels = []
    monkeypatch.setattr(page._pill, "push_level", levels.append)

    page.start()
    page.push_level(0.37)
    page.stop()

    assert levels == [0.37]
    assert page._revealers["pill"].get_child() is page._pill


def test_terminal_result_is_announced_once(display):
    engine = fake_engine()
    page = _page(engine=engine)
    page.content.announce = Mock()
    page.start()
    page._button.emit("clicked")
    page._button.emit("clicked")

    engine.on_text("hello")
    engine.on_text("late duplicate")

    page.content.announce.assert_called_once_with(
        "Dictation complete: hello", Gtk.AccessibleAnnouncementPriority.MEDIUM
    )
    page.stop()


def test_a_returning_button_takes_back_the_focus(display):
    # Transcribing retires the focused button, and GTK moves the focus up to the
    # scroller around the page, as in onboarding.
    engine = fake_engine()
    page = _page(engine=engine)
    window = Gtk.Window(child=Gtk.ScrolledWindow(child=page.content))
    window.present()
    try:
        page.start()
        page._button.grab_focus()
        page._button.emit("clicked")
        page._button.emit("clicked")
        wait_until(lambda: window.get_focus() is not page._button)

        engine.on_text("hello")

        assert window.get_focus() is page._button
    finally:
        page.stop()
        window.destroy()


@pytest.mark.parametrize("failure", [Step.MODEL_MISSING, Step.MIC_FAILED])
def test_a_failure_is_read_out(display, failure):
    # Nothing on a failure step takes the focus, so its heading is announced.
    downloads = _Downloads()
    engine = fake_engine()
    engine.begin_take.return_value = False
    missing = failure is Step.MODEL_MISSING
    page = _page(downloaded=not missing, downloads=downloads, engine=engine)
    page.content.announce = Mock()
    page.start()
    if missing:
        downloads.calls[0]["on_error"]("offline")
    else:
        page._button.emit("clicked")

    assert page._step is failure
    page.content.announce.assert_called_once_with(
        ". ".join(_HEADER[failure]), Gtk.AccessibleAnnouncementPriority.MEDIUM
    )
    page.stop()


@pytest.mark.parametrize(("idle", "cause"), [(True, "microphone"), (False, "engine")])
def test_a_microphone_that_cannot_start_is_a_mic_failure(display, caplog, idle, cause):
    engine = fake_engine()
    engine.begin_take.return_value = False
    engine.is_idle = idle
    page = _page(engine=engine)
    page.start()
    page._button.emit("clicked")
    assert page._step is Step.MIC_FAILED
    [warning] = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warning.startswith(f"Demo take did not start ({cause} ")
