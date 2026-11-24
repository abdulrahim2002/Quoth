# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.onboarding` and the steps it hosts.

These build real widgets, so they need a display.
"""

from unittest.mock import Mock

import pytest

from quoth.config import Config
from tests.support import (
    FakeBenchmark,
    FakePortalShortcuts,
    appearance,
    button,
    descendants,
    fake_engine,
    legibility,
    wait_until,
)


class _NoDownloads:
    def is_downloading(self, _model):
        return False

    def start_or_join(self, *_args, **_kwargs):
        return None


@pytest.fixture(scope="module")
def app():
    from gi.repository import Adw, Gio

    # Parallel test processes must each own a local application.
    app = Adw.Application(
        application_id="dev.quoth.TestOnboarding", flags=Gio.ApplicationFlags.NON_UNIQUE
    )
    assert app.register(None)
    assert not app.get_is_remote()
    return app


@pytest.fixture
def assistant_factory(display, app):
    from quoth.models import ModelManager
    from quoth.onboarding import OnboardingAssistant

    assistants = []

    def make(
        *,
        devices=None,
        on_complete=lambda _ready: None,
        engine=None,
        start_benchmark=None,
        portal_shortcuts=None,
        motion_enabled=False,
        downloads=None,
        downloaded=True,
    ):
        devices = devices or ["CPU"]
        model_mgr = ModelManager()
        model_mgr.is_downloaded = lambda _model: downloaded
        assistant = OnboardingAssistant(
            app,
            model_mgr,
            Config(),
            downloads or _NoDownloads(),
            devices,
            on_complete=on_complete,
            on_open_settings=lambda: None,
            motion_enabled=motion_enabled,
            engine=engine or fake_engine(),
            start_benchmark=start_benchmark or FakeBenchmark(),
            portal_shortcuts=portal_shortcuts,
        )
        assistants.append(assistant)
        return assistant

    yield make
    for assistant in assistants:
        assistant._win.destroy()


@pytest.fixture
def assistant(assistant_factory):
    return assistant_factory()


def _page_titles(assistant) -> list[str]:
    stack = assistant._nav.get_navigation_stack()
    return [stack.get_item(i).get_title() for i in range(stack.get_n_items())]


def _label(widget, text: str):
    from gi.repository import Gtk

    labels = (w for w in descendants(widget) if isinstance(w, Gtk.Label))
    return next((w for w in labels if w.get_label() == text), None)


@pytest.fixture
def ready_download(assistant):
    from unittest.mock import Mock

    from quoth.downloads import DownloadCoordinator
    from tests.support import FakeScheduler

    download = Mock(return_value=None)
    assistant._downloads = DownloadCoordinator(
        "/models", FakeScheduler(), download=download
    )
    background = assistant._downloads.start_or_join(assistant._config.model, "CPU")
    page = assistant._build_ready()
    note = _label(
        page,
        "Your speech model is still downloading in the background. "
        "Dictation will be ready when it finishes.",
    )
    return assistant, page, note, download, background


def _coordinator():
    from quoth.downloads import DownloadCoordinator
    from tests.support import FakeScheduler

    download = Mock(return_value=None)
    return DownloadCoordinator("/models", FakeScheduler(), download=download), download


def test_the_welcome_note_follows_the_prefetch(assistant_factory):
    from quoth.download_format import format_download_progress
    from quoth.downloader import DownloadProgress

    downloads, download = _coordinator()
    assistant = assistant_factory(downloads=downloads, downloaded=False)
    note = assistant._welcome_note
    assert note.get_label().startswith("Getting ready: downloading the speech model")
    progress = DownloadProgress(0.5, 50, 100, 10.0)
    download.call_args.kwargs["on_progress"](progress)
    assert note.get_label() == format_download_progress(progress)
    download.call_args.kwargs["on_done"]()
    assert note.get_label() == "Ready to go."


def test_the_welcome_note_takes_the_photographs_dim_colour(assistant_factory):
    from quoth import brand

    with appearance(brand.css(dark=True), dark=True):
        assistant, text = _show_welcome(assistant_factory, (900, 700))
        [tagline] = [w for w in text if w.get_label() == "Local. Private. Yours."]
        note, shown = (
            (c.red, c.green, c.blue, c.alpha)
            for c in (assistant._welcome_note.get_color(), tagline.get_color())
        )

    bone_dim = (0xCF / 255, 0xCC / 255, 0xC3 / 255, 1.0)
    assert note == pytest.approx(shown, abs=0.01)
    assert note == pytest.approx(bone_dim, abs=0.01)


def test_a_failed_prefetch_retires_the_welcome_note(assistant_factory):
    from quoth.downloader import DownloadProgress

    downloads, download = _coordinator()
    assistant = assistant_factory(downloads=downloads, downloaded=False)
    note = assistant._welcome_note
    download.call_args.kwargs["on_progress"](DownloadProgress(0.5, 50, 100, 10.0))
    assert note.get_visible()

    download.call_args.kwargs["on_error"]("connection reset")

    assert not note.get_visible()


def _show_welcome(assistant_factory, size):
    """Present the welcome at `size` with its intro finished; return it and its copy.

    The copy is the welcome's labels on screen, which always include the note.
    """
    from gi.repository import Gtk

    assistant = assistant_factory(downloaded=False)
    assistant._win.set_default_size(*size)
    assistant._win.present()
    note = assistant._welcome_note
    wait_until(note.get_mapped)
    assistant._finish_intro(skipped=True)
    text = [
        line
        for line in assistant._welcome_lines
        if isinstance(line, Gtk.Label) and line.get_mapped()
    ]
    assert note in text
    return assistant, text


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("size", [(420, 560), (900, 700), (1280, 800)])
def test_the_welcome_copy_reads_against_the_smoke(assistant_factory, size, dark):
    from quoth import brand

    with appearance(brand.css(dark=dark), dark=dark):
        # The blurb shows only in the hero layout, which a 1280 by 800 window takes.
        assistant, text = _show_welcome(assistant_factory, size)
        for label in text:
            assert legibility(assistant._win, label) >= 4.5, label.get_label()


def test_the_welcome_copy_reads_over_the_videos_brightest_frame(
    assistant_factory, monkeypatch
):
    from pathlib import Path

    from quoth import brand

    # Of the video's frames, this is the brightest under the copy once the copy has
    # arrived, so it is the tightest ground the copy must read against.
    frame = Path(__file__).with_name("fixtures") / "welcome-brightest-frame.jpg"
    monkeypatch.setattr("quoth.onboarding._backdrop", lambda _name: frame)
    with appearance(brand.css(dark=True), dark=True):
        assistant, text = _show_welcome(assistant_factory, (1280, 720))
        for label in text:
            assert legibility(assistant._win, label) >= 4.5, label.get_label()


def test_activation_takes_the_prefetch_over(assistant_factory):
    downloads, download = _coordinator()
    handed = []
    assistant = assistant_factory(
        downloads=downloads, downloaded=False, on_complete=handed.append
    )
    assistant._finish()
    landed = Mock()
    downloads.start_or_join(assistant._config.model, "CPU", on_done=landed)
    handed[0]()
    note = assistant._welcome_note.get_label()
    assert not download.call_args.kwargs["cancel"].is_set()
    download.call_args.kwargs["on_done"]()
    landed.assert_called_once_with()
    assert assistant._welcome_note.get_label() == note


def test_an_abandoned_onboarding_leaves_its_prefetch_running(assistant_factory):
    # The process exits with it, which keeps the partial file for the next launch.
    downloads, download = _coordinator()
    assistant = assistant_factory(downloads=downloads, downloaded=False)
    assistant._win.emit("close-request")
    assert not download.call_args.kwargs["cancel"].is_set()


def test_a_model_already_on_disk_is_not_fetched_again(assistant_factory):
    downloads, download = _coordinator()
    assistant = assistant_factory(downloads=downloads, downloaded=True)
    assert download.call_count == 0
    assert not assistant._welcome_note.get_visible()


def test_ready_says_so_when_the_model_did_not_download(assistant_factory):
    downloads, download = _coordinator()
    assistant = assistant_factory(downloads=downloads, downloaded=False)
    page = assistant._build_ready()
    page.emit("shown")
    assert assistant._ready_title.get_label() == "You're all set!"

    download.call_args.kwargs["on_error"]("connection reset")

    assert assistant._ready_title.get_label() == "Almost ready"
    assert assistant._ready_description.get_label() == (
        "The speech model didn't download. Open Settings to download it, "
        "then start dictating."
    )


@pytest.mark.parametrize("outcome", ["on_done", "on_error"])
def test_ready_download_note_tracks_existing_transfer(ready_download, outcome):
    assistant, page, note, download, _background = ready_download
    page.emit("shown")

    assert note.get_visible()
    assert note.has_css_class("dim-label")
    assert note.get_prev_sibling().get_label() == "You're all set!"
    assert download.call_count == 1

    download.call_args.kwargs[outcome](*([] if outcome == "on_done" else ["failed"]))
    assert not note.get_visible()
    page.emit("hidden")
    page.emit("shown")
    assert not note.get_visible()
    assert download.call_count == 1

    assistant._downloads.start_or_join(assistant._config.model, "CPU")
    page.emit("shown")
    assert note.get_visible()
    assert download.call_count == 2


def test_ready_does_not_restart_cancelled_download(ready_download):
    _assistant, page, note, download, background = ready_download
    background.cancel()
    page.emit("shown")
    assert not note.get_visible()
    download.call_args.kwargs["on_cancelled"]()
    page.emit("shown")
    assert not note.get_visible()
    assert download.call_count == 1


@pytest.mark.parametrize("leave", ["hidden", "close"])
def test_ready_retires_observer_without_stopping_prefetch(ready_download, leave):
    from unittest.mock import patch

    assistant, page, note, download, _background = ready_download
    page.emit("shown")
    page.emit("shown")  # A cancelled back gesture shows the same page again.
    if leave == "close":
        assistant._win.emit("close-request")
    else:
        page.emit("hidden")
    assert not note.get_visible()
    assert not download.call_args.kwargs["cancel"].is_set()

    with patch.object(note, "set_visible") as update:
        download.call_args.kwargs["on_done"]()
        update.assert_not_called()


def test_continue_after_going_back_opens_the_next_page(assistant):
    from gi.repository import Adw

    # Stand-in pages: what is under test is which page comes next, not the pages.
    assistant._builders = [
        lambda title=title: Adw.NavigationPage(title=title)
        for title in ("one", "two", "three", "four")
    ]
    assistant._advance()
    assistant._advance()
    assistant._nav.pop()
    assistant._advance()
    assert _page_titles(assistant)[1:] == ["two", "three"]


@pytest.mark.parametrize(
    ("devices", "clicks", "titles"),
    [
        (
            ["CPU"],
            ["Get started", "Skip for now", "Continue", "Continue"],
            ["Welcome", "Try it", "Shortcut", "Privacy", "Ready"],
        ),
        (
            ["CPU", "NPU"],
            ["Get started", "Skip for now", "Skip", "Continue", "Continue"],
            ["Welcome", "Try it", "Optimize", "Shortcut", "Privacy", "Ready"],
        ),
    ],
    ids=["cpu-only", "with-npu"],
)
def test_each_page_forward_button_opens_the_next_step(
    assistant_factory, monkeypatch, devices, clicks, titles
):
    from quoth import onboarding

    monkeypatch.setattr(onboarding, "keyboard_available", lambda _hotkey: True)
    assistant = assistant_factory(devices=devices)
    for label in clicks:
        button(assistant._nav.get_visible_page(), label).emit("clicked")
    assert _page_titles(assistant) == titles


@pytest.mark.parametrize("devices", [["CPU"], ["CPU", "NPU"]])
def test_every_page_is_built_once_across_repeated_back_and_forward(
    assistant_factory, monkeypatch, devices
):
    from quoth import onboarding

    monkeypatch.setattr(onboarding, "keyboard_available", lambda _hotkey: True)
    assistant = assistant_factory(devices=devices)
    built = [0] * len(assistant._builders)
    original = assistant._builders

    def counted(index, builder):
        def build():
            built[index] += 1
            return builder()

        return build

    assistant._builders = [counted(i, builder) for i, builder in enumerate(original)]
    built[0] = 1  # The welcome page is constructed during assistant initialisation.

    for _ in range(1, len(assistant._builders)):
        assistant._advance()
    pages = [assistant._nav.find_page(f"step-{i}") for i in range(len(built))]

    for _ in range(2):
        while assistant._nav.get_navigation_stack().get_n_items() > 1:
            assistant._nav.pop()
        for _ in range(1, len(assistant._builders)):
            assistant._advance()

    assert built == [1] * len(built)
    assert [assistant._nav.find_page(f"step-{i}") for i in range(len(built))] == pages


def test_reused_pages_keep_choices_and_refresh_ready_copy(
    assistant, find_widget, monkeypatch
):
    from gi.repository import Adw

    from quoth import crash_reports, onboarding

    monkeypatch.setattr(onboarding, "keyboard_available", lambda _hotkey: True)
    assistant._advance()  # Try it
    assistant._advance()  # Shortcut
    shortcut = assistant._nav.get_visible_page()
    assistant._on_hotkey_chosen("KEY_F13")
    assistant._advance()  # Privacy
    privacy = assistant._nav.get_visible_page()
    reporting = find_widget(privacy, Adw.SwitchRow, crash_reports.SWITCH_TITLE)
    reporting.set_active(True)
    assistant._advance()  # Ready
    ready = assistant._nav.get_visible_page()
    ready.emit("shown")
    assert _label(ready, "Quoth is ready. Press F13 any time to dictate.")

    assistant._nav.pop()
    assistant._nav.pop()
    assert assistant._nav.get_visible_page() is shortcut
    assert assistant._config.hotkey == "KEY_F13"
    assistant._advance()
    assert assistant._nav.get_visible_page() is privacy
    assert reporting.get_active()

    assistant._config.hotkey = "KEY_PAUSE"
    assistant._advance()
    ready.emit("shown")
    assert assistant._nav.get_visible_page() is ready
    assert _label(ready, "Quoth is ready. Press Pause any time to dictate.")


def test_sandbox_shortcut_confirmation_survives_back_navigation(assistant, monkeypatch):
    from quoth import onboarding

    monkeypatch.setattr(onboarding, "keyboard_available", lambda _hotkey: False)
    assistant._advance()
    assistant._advance()
    shortcut = assistant._nav.get_visible_page()
    assistant.notify_toggle_detected()
    assistant._advance()
    assistant._nav.pop()

    assert assistant._nav.get_visible_page() is shortcut
    assert assistant._shortcut_confirmed
    assert assistant._shortcut_next.get_sensitive()
    assert assistant._shortcut_status.get_label().startswith("Shortcut detected.")


def test_a_detected_shortcut_is_read_out(assistant, monkeypatch):
    from gi.repository import Gtk

    from quoth import onboarding

    monkeypatch.setattr(onboarding, "keyboard_available", lambda _hotkey: False)
    assistant._advance()
    assistant._advance()
    assistant._shortcut_status.announce = Mock()
    assistant.notify_toggle_detected()
    assistant.notify_toggle_detected()
    assistant._shortcut_status.announce.assert_called_once_with(
        "Shortcut detected. You're ready to dictate.",
        Gtk.AccessibleAnnouncementPriority.MEDIUM,
    )


def test_copy_and_keyboard_settings_feedback_is_read_out(assistant, monkeypatch):
    from gi.repository import Gtk

    from quoth import onboarding, shortcut_setup

    monkeypatch.setattr(onboarding, "keyboard_available", lambda _hotkey: False)
    monkeypatch.setattr(shortcut_setup, "open_keyboard_settings", lambda: False)
    assistant._advance()
    assistant._advance()
    status = assistant._shortcut_status
    status.announce = Mock()
    medium = Gtk.AccessibleAnnouncementPriority.MEDIUM

    assistant._copy_command("quoth --toggle")
    copied = "Copied. Now bind it in your keyboard settings."
    assert status.get_label() == copied
    status.announce.assert_called_once_with(copied, medium)

    status.announce.reset_mock()
    assistant._on_open_keyboard_settings(None)
    failed = shortcut_setup.KEYBOARD_SETTINGS_FAILED_MSG
    assert status.get_label() == failed
    status.announce.assert_called_once_with(failed, medium)


def test_portal_shortcut_refreshes_and_binding_change_requires_reconfirmation(
    assistant_factory, monkeypatch
):
    from quoth import onboarding

    monkeypatch.setattr(onboarding, "keyboard_available", lambda _hotkey: True)
    portal = FakePortalShortcuts()
    assistant = assistant_factory(portal_shortcuts=portal)
    assistant._advance()
    assistant._advance()
    shortcut = assistant._nav.get_visible_page()

    assert assistant._portal_shortcut_page.get_title() == "Checking shortcut support"
    assert not assistant._portal_manual_command.get_visible()
    assert not hasattr(assistant, "_keycap")
    assert portal.configure_calls == 0

    portal.change("unbound")
    assistant._portal_shortcut_action.emit("clicked")
    assert portal.configure_calls == 1

    portal.change("ready", "Super+D")
    assert assistant._nav.get_visible_page() is shortcut
    assert assistant._portal_shortcut_page.get_title() == "Super+D"
    assert (
        assistant._portal_shortcut_page.get_description()
        == "Your desktop manages this shortcut."
    )
    assert assistant._portal_shortcut_modes.get_visible()
    assert not assistant._shortcut_next.get_sensitive()
    button(shortcut, "Hold while you talk").set_active(True)
    assert assistant._config.mode == "push-to-hold"

    assistant.notify_toggle_detected()
    assert assistant._shortcut_next.get_sensitive()
    portal.change("ready", "Super+Space")
    assert not assistant._shortcut_confirmed
    assert not assistant._shortcut_next.get_sensitive()
    assert assistant._shortcut_status.get_label().startswith("Waiting:")


def test_portal_manual_fallback_and_close_unsubscribe(assistant_factory):
    portal = FakePortalShortcuts(status="unavailable")
    assistant = assistant_factory(portal_shortcuts=portal)
    assistant._advance()
    assistant._advance()

    assert assistant._portal_manual_command.get_visible()
    assert not assistant._portal_shortcut_modes.get_visible()
    assistant.notify_toggle_detected()
    assert assistant._shortcut_next.get_sensitive()
    portal.change("unbound")
    assert not assistant._shortcut_next.get_sensitive()

    assistant._on_close_request(None)
    assistant._on_close_request(None)
    assert portal.unsubscribe_calls == 1


def test_portal_ready_copy_tracks_binding_and_skipped_setup(assistant_factory):
    portal = FakePortalShortcuts(status="ready", binding="Press F8")
    assistant = assistant_factory(portal_shortcuts=portal)
    ready = assistant._build_ready()
    assert _label(
        ready, "Quoth is ready. Shortcut: Press F8. Use it any time to dictate."
    )

    portal.change("unbound")
    assert _label(
        ready,
        "Quoth is ready. Set up your dictation shortcut in Settings before you start.",
    )


def test_microphone_failure_keeps_the_trial_skippable(assistant):
    from gi.repository import Gtk

    from quoth.try_it import Step

    assistant._engine.unavailable = True
    page = assistant._build_try_it()
    assistant._try_it.start()

    skip = button(page, "Skip for now")
    assert assistant._try_it._step is Step.MIC_FAILED
    assert skip is not None and skip.get_sensitive()
    scroller = assistant._try_it._heading.get_ancestor(Gtk.ScrolledWindow)
    assert scroller is not None
    assert skip.get_ancestor(Gtk.ScrolledWindow) == scroller


def test_successful_trial_changes_skip_to_continue(assistant):
    assistant._model_mgr.is_downloaded = lambda _model: True
    page = assistant._build_try_it()
    assistant._try_it.start()

    assistant._try_it._button.emit("clicked")
    assistant._try_it._button.emit("clicked")
    assistant._engine.on_text("hello from Quoth")

    assert button(page, "Continue") is not None
    assert button(page, "Skip for now") is None
    assistant._try_it.stop()


def test_privacy_always_offers_reporting_and_persists_consent(
    assistant, find_widget, monkeypatch
):
    from gi.repository import Adw, GLib, Gtk

    from quoth import crash_reports
    from quoth.about import PRIVACY_URL

    page = assistant._build_privacy()
    row = find_widget(page, Adw.SwitchRow, crash_reports.SWITCH_TITLE)
    assert row is not None
    assert row.get_subtitle() == crash_reports.SWITCH_SUBTITLE
    assert row.get_tooltip_text() == crash_reports.SWITCH_HELP
    assert not row.get_active()
    opened = []
    monkeypatch.setattr("quoth.onboarding.open_uri", opened.append)
    notice = find_widget(page, Adw.ActionRow, crash_reports.NOTICE_TITLE)
    assert notice.get_tooltip_text() == crash_reports.NOTICE_HELP
    assert notice.get_activatable()
    notice.emit("activated")
    assert opened == [PRIVACY_URL]

    assistant._nav.push(page)
    rendered = False

    def on_frame(*_args):
        nonlocal rendered
        rendered = True
        return GLib.SOURCE_REMOVE

    row.add_tick_callback(on_frame)
    wait_until(lambda: rendered)

    scroller = row.get_ancestor(Gtk.ScrolledWindow)
    assert scroller is not None
    assert button(page, "Continue").get_ancestor(Gtk.ScrolledWindow) == scroller
    visible, bounds = row.compute_bounds(scroller)
    assert visible
    assert bounds.get_height() > 0
    assert bounds.get_y() >= 0
    assert bounds.get_y() + bounds.get_height() <= scroller.get_height()

    row.set_active(True)
    assert Config().crash_reports is True
    rebuilt = assistant._build_privacy()
    assert find_widget(rebuilt, Adw.SwitchRow, crash_reports.SWITCH_TITLE).get_active()
    row.set_active(False)
    assert Config().crash_reports is False
    Config().crash_reports = True  # another writer, such as Settings
    assert row.get_active()


def test_finish_keeps_only_ready_and_starts_activation_once(assistant):
    activated = []
    assistant._on_complete = activated.append
    for _ in range(1, len(assistant._builders)):
        assistant._advance()
    ready = assistant._nav.get_visible_page()

    assistant._finish()
    assistant._finish()

    stack = assistant._nav.get_navigation_stack()
    assert stack.get_n_items() == 1
    assert stack.get_item(0) is ready
    assert ready.get_title() == "Ready"
    assert len(activated) == 1
    assert assistant._config.onboarding_complete


@pytest.mark.parametrize(
    ("label", "settings_opened"),
    [("Start dictating", []), ("Customize in Settings", [True])],
)
def test_ready_buttons_hand_over_and_settings_wait_for_the_live_app(
    assistant, label, settings_opened
):
    activated = []
    opened = []
    assistant._on_complete = activated.append
    assistant._on_open_settings = lambda: opened.append(True)
    ready = assistant._build_ready()
    assistant._nav.push(ready)

    button(ready, label).emit("clicked")
    assert len(activated) == 1
    assert opened == []  # the app is not live yet

    activated[0]()
    assert opened == settings_opened


def test_closing_before_finish_does_not_activate(assistant, caplog):
    activated = []
    assistant._on_complete = activated.append

    assert assistant._on_close_request(None) is False
    assert activated == []
    # The app quits next, and the log says why.
    assert any(
        m.startswith("Onboarding closed at step ") and m.endswith("before finishing")
        for m in caplog.messages
    )


def test_lazy_pages_inherit_latest_motion_preference(assistant_factory):
    assistant = assistant_factory(devices=["CPU", "NPU"])
    assistant.set_motion_enabled(True)

    assert assistant._intro._motion
    assert assistant._backdrop._motion_enabled

    assistant._build_try_it()
    assistant._build_optimize()
    assistant._build_privacy()
    assistant._build_ready()

    assert assistant._try_it._motion_enabled
    assert assistant._optimize._motion
    assert assistant._privacy._motion
    assert assistant._ready_mark._motion
    assert assistant._ready_ripple._motion

    assistant.set_motion_enabled(False)
    assert not assistant._try_it._motion_enabled
    assert assistant._optimize._stack.get_transition_duration() == 0
    assert assistant._optimize.actions.get_transition_duration() == 0
    assert not assistant._privacy._motion
    assert not assistant._ready_mark._motion
    assert not assistant._ready_ripple._motion


@pytest.mark.parametrize("motion_enabled", [True, False])
def test_page_slides_follow_the_motion_preference(assistant_factory, motion_enabled):
    assistant = assistant_factory(motion_enabled=motion_enabled)
    assert assistant._nav.get_animate_transitions() is motion_enabled

    assistant.set_motion_enabled(not motion_enabled)
    assert assistant._nav.get_animate_transitions() is not motion_enabled


def test_a_key_skips_the_welcome_intro(assistant_factory):
    from gi.repository import Gtk

    assistant = assistant_factory(motion_enabled=True)
    page = assistant._welcome_start.get_ancestor(Gtk.Overlay)
    keys = next(
        c for c in page.observe_controllers() if isinstance(c, Gtk.EventControllerKey)
    )

    handled = keys.emit("key-pressed", 0, 0, 0)

    assert assistant._welcome_start.get_sensitive()
    assert not handled  # the key still reaches the focused widget


def test_disabling_motion_settles_pending_welcome_arrivals(assistant_factory):
    assistant = assistant_factory(motion_enabled=True)
    assistant._finish_intro()

    assistant.set_motion_enabled(False)

    assert assistant._welcome_arrival_timeouts == set()
    assert all(line.get_opacity() == 1.0 for line in assistant._welcome_lines)
    assert all(line.get_margin_top() == 0 for line in assistant._welcome_lines)


def test_close_cancels_pending_arrivals_and_late_callbacks_noop(assistant_factory):
    assistant = assistant_factory(motion_enabled=True)
    assistant._finish_intro()
    line = assistant._welcome_lines[-1]

    assert assistant._welcome_arrival_timeouts
    assert line.get_opacity() == 0.0
    assistant._on_close_request(None)

    assert assistant._closed
    assert assistant._welcome_arrival_timeouts == set()
    assistant._arrive(line)
    assistant._finish_intro()
    assert line.get_opacity() == 0.0


def test_arrival_exception_is_logged_and_source_is_forgotten(
    assistant_factory, monkeypatch, callback_raises, caplog
):
    from types import SimpleNamespace

    from quoth import onboarding

    caplog.set_level("ERROR")
    assistant = assistant_factory(motion_enabled=True)
    scheduled = []
    next_id = iter(range(100, 110))

    def schedule(_delay, callback, line):
        source_id = next(next_id)
        scheduled.append((source_id, callback, line))
        return source_id

    monkeypatch.setattr(onboarding.GLib, "timeout_add", schedule)
    monkeypatch.setattr(
        onboarding.GLib,
        "main_current_source",
        lambda: SimpleNamespace(get_id=lambda: scheduled[0][0]),
    )

    def fail(_line):
        raise RuntimeError("arrival failed")

    monkeypatch.setattr(assistant, "_arrive", fail)
    assistant._finish_intro()
    source_id, callback, line = scheduled[0]

    assert callback(line) is False

    assert source_id not in assistant._welcome_arrival_timeouts
    assert [str(record.exc_info[1]) for record in caplog.records] == ["arrival failed"]


def test_try_it_callbacks_stay_with_the_page_that_owns_them(assistant_factory):
    engine = fake_engine()
    assistant = assistant_factory(engine=engine)
    first_page = assistant._build_try_it()
    first = assistant._try_it
    second_page = assistant._build_try_it()
    second = assistant._try_it

    first_page.emit("shown")
    first_page.emit("hiding")
    first_page.emit("shown")  # A cancelled swipe returns to the same live page.
    assert engine.watchers == {first}

    first_page.emit("hidden")
    assert engine.watchers == set()
    second_page.emit("shown")
    assert engine.watchers == {second}
    second_page.emit("hiding")
    second_page.emit("hidden")  # A completed transition retires its own meter.
    assert engine.watchers == set()


def test_engine_levels_reach_the_live_try_it_pill(assistant, monkeypatch):
    assistant.push_level(0.5)  # before the step exists: nothing to show it
    page = assistant._build_try_it()
    levels = []
    monkeypatch.setattr(assistant._try_it._pill, "push_level", levels.append)
    page.emit("shown")
    assistant.push_level(0.37)
    assert levels == [0.37]
    page.emit("hidden")


def test_the_shortcut_page_names_the_modes_as_settings_does(assistant):
    from gi.repository import Gtk

    page = assistant._build_shortcut_native()
    toggles = [w for w in descendants(page) if isinstance(w, Gtk.ToggleButton)]
    assert [toggle.get_label() for toggle in toggles] == [
        "Tap to start and stop",
        "Hold while you talk",
    ]


class TestShortcutDemo:
    @pytest.fixture
    def press(self, assistant):
        from quoth.listener import scancode_for_key_name

        assistant._nav.push(assistant._build_shortcut_native())
        keycode = scancode_for_key_name(assistant._config.hotkey) + 8

        def press(down: bool) -> str:
            if down:
                assistant._on_shortcut_key_pressed(None, 0, keycode, 0)
            else:
                assistant._on_shortcut_key_released(None, 0, keycode, 0)
            return assistant._shortcut_lead.get_label()

        return press

    def test_tap_starts_on_one_press_and_stops_on_the_next(self, assistant, press):
        assistant._config.mode = "tap-to-toggle"
        assert press(True).startswith("Listening")
        assert press(True).startswith("Listening")  # key repeat while held
        assert press(False).startswith("Listening")
        assert press(True).startswith("That's it")

    def test_a_new_key_restarts_the_demo(self, assistant, press):
        assistant._config.mode = "push-to-hold"
        press(True)
        press(False)  # "That's it. Hold <old key> anywhere to dictate."
        assistant._on_hotkey_chosen("KEY_F14")
        assert assistant._shortcut_lead.get_label() == "Hold it now."

    def test_the_demo_reads_out_each_answer_to_the_key(self, assistant, press):
        from gi.repository import Gtk

        assistant._config.mode = "tap-to-toggle"
        assistant._shortcut_lead.announce = Mock()
        heard = press(True)
        press(False)
        done = press(True)
        medium = Gtk.AccessibleAnnouncementPriority.MEDIUM
        assert assistant._shortcut_lead.announce.call_args_list == [
            ((heard, medium),),
            ((done, medium),),
        ]

    def test_hold_listens_until_the_key_is_let_go(self, assistant, press):
        assistant._config.mode = "push-to-hold"
        assert press(True).startswith("Listening")
        assert "pressed" in assistant._keycap.get_css_classes()
        assert press(False).startswith("That's it")
        assert "pressed" not in assistant._keycap.get_css_classes()


def test_skipped_calibration_keeps_its_page_locked_until_it_lands(
    assistant_factory, monkeypatch
):
    from quoth import onboarding

    monkeypatch.setattr(onboarding, "keyboard_available", lambda _hotkey: True)
    benchmark = FakeBenchmark()
    activated = []
    assistant = assistant_factory(
        devices=["CPU", "NPU"], on_complete=activated.append, start_benchmark=benchmark
    )
    assistant._advance()
    assistant._advance()
    optimize_page = assistant._nav.get_visible_page()
    optimize = assistant._optimize

    optimize._start()
    optimize._skip()

    assert not optimize_page.get_can_pop()
    assert assistant._page(2) is optimize_page
    assert assistant._optimize is optimize

    assistant._advance()
    assistant._advance()
    assistant._finish()
    # Handed over at once: the app's engine holds activation until the run lands.
    assert len(activated) == 1

    benchmark.finish(None)
    assert optimize_page.get_can_pop()


def test_a_run_that_lands_after_close_touches_no_widgets(
    assistant_factory, monkeypatch
):
    from quoth import onboarding

    monkeypatch.setattr(onboarding, "keyboard_available", lambda _hotkey: True)
    benchmark = FakeBenchmark()
    engine = fake_engine()
    assistant = assistant_factory(
        devices=["CPU", "NPU"], engine=engine, start_benchmark=benchmark
    )
    assistant._advance()
    assistant._advance()
    optimize = assistant._optimize
    optimize._start()

    assistant._on_close_request(None)
    engine.cancel_calibration.assert_called_once_with()  # stops at the next cell
    optimize._optimize = None
    optimize._stack = None
    benchmark.finish(object())
