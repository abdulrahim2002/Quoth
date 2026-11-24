# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.engine_section`: view helpers and the device-selection lifecycle."""

from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from quoth import calibration
from quoth.engine_section import (
    CalibrationBridge,
    EngineSection,
    _benchmark_status,
    _model_row_state,
    _model_subtitle,
)
from quoth.hardware import fingerprint
from quoth.models import AVAILABLE_MODELS, MODELS_BY_NAME
from tests.support import (
    FakeModelManager,
    accessible_mismatch,
    button,
    descendants,
    wait_until,
)


@pytest.fixture
def npu_cache(display, find_widget):
    """Build the real Hardware acceleration group, shown, over a `size`-byte NPU cache.

    The bridge's `clear_cache` reports success; `row()` finds the cache row, if any.
    """
    from gi.repository import Adw

    from quoth.config import Config
    from quoth.models import ModelManager

    Adw.init()
    windows = []

    def build(size=64):
        manager = create_autospec(ModelManager, instance=True)
        manager.cache_size_bytes.return_value = size
        bridge = CalibrationBridge(
            ranking=None, run=MagicMock(), clear_cache=MagicMock(return_value=True)
        )
        toasts = []
        section = EngineSection(
            Config(),
            manager,
            calibration_bridge=bridge,
            enumerate_microphones=lambda: [],
            toast=toasts.append,
        )
        page = Adw.PreferencesPage()
        window = Adw.Window(content=page)
        windows.append(window)
        section.attach(window)
        section.build_acceleration(page)
        window.present()
        return SimpleNamespace(
            section=section,
            manager=manager,
            bridge=bridge,
            toasts=toasts,
            window=window,
            row=lambda: find_widget(page, Adw.ActionRow, "NPU cache"),
        )

    yield build
    for window in windows:
        window.destroy()


def _confirmation(state):
    """Press the cache row's Clear; return the confirmation it opens."""
    button(state.row(), "Clear").emit("clicked")
    wait_until(lambda: state.window.get_visible_dialog() is not None)
    return state.window.get_visible_dialog()


def _clear(state):
    """Clear the cache as the user does: press Clear, then confirm."""
    button(_confirmation(state), "Clear").emit("clicked")


class TestNpuCache:
    def test_success_hides_empty_row_and_reports_outcome(self, npu_cache):
        state = npu_cache()
        row = state.row()
        state.manager.cache_size_bytes.return_value = 0

        _clear(state)

        state.bridge.clear_cache.assert_called_once_with()
        assert not row.get_visible()
        assert state.toasts == ["NPU cache cleared"]

    def test_busy_keeps_current_size_and_reports_outcome(self, npu_cache):
        from gi.repository import GLib

        state = npu_cache(size=64)
        state.bridge.clear_cache.return_value = False

        _clear(state)

        row = state.row()
        assert row.get_subtitle() == f"{GLib.format_size(64)} on disk"
        assert row.get_visible()
        assert state.toasts == ["NPU cache is busy. Try again when Quoth is idle."]

    def test_failure_keeps_unknown_size_distinct_from_empty(self, npu_cache):
        state = npu_cache()
        state.bridge.clear_cache.side_effect = OSError("delete failed")
        state.manager.cache_size_bytes.side_effect = OSError("inspect failed")

        _clear(state)

        row = state.row()
        assert row.get_subtitle() == "Size unavailable"
        assert row.get_visible()
        assert state.toasts == ["Couldn't clear NPU cache"]

    def test_a_failed_clear_is_logged_without_a_traceback(self, npu_cache, caplog):
        # An unwritable cache is expected, so it is no crash report.
        caplog.set_level("ERROR")
        state = npu_cache()
        state.bridge.clear_cache.side_effect = PermissionError(13, "Permission denied")

        _clear(state)

        [record] = caplog.records
        assert record.levelname == "ERROR"
        assert record.exc_info is None
        assert state.toasts == ["Couldn't clear NPU cache"]

    def test_refresh_creates_row_when_cache_appears(self, npu_cache):
        state = npu_cache(size=0)
        assert state.row() is None
        state.manager.cache_size_bytes.return_value = 64

        state.section.refresh_cache_status()

        row = state.row()
        assert row.get_visible()
        assert button(row, "Clear").get_cursor().get_name() == "pointer"

    def test_confirmation_uses_approved_wording_and_refreshes_pointer_policy(
        self, npu_cache
    ):
        from gi.repository import Gtk

        dialog = _confirmation(npu_cache())

        assert dialog.get_heading() == "Clear NPU cache?"
        assert dialog.get_body() == (
            "Removes prepared NPU model files to free disk space. Downloaded models and "
            "benchmark results are kept. The next time a model needs NPU preparation, "
            "loading it may take longer."
        )
        wait_until(dialog.get_mapped)
        buttons = [w for w in descendants(dialog) if isinstance(w, Gtk.Button)]
        assert buttons
        assert all(b.get_cursor().get_name() == "pointer" for b in buttons)

    def test_runtime_refresh_rechecks_cache(self, npu_cache):
        state = npu_cache()
        row = state.row()
        state.manager.cache_size_bytes.return_value = 0
        state.section.refresh_runtime(None)
        assert not row.get_visible()

        state.manager.cache_size_bytes.return_value = 64
        state.section.refresh_runtime(None)
        assert row.get_visible()


class TestBenchmarkStatus:
    """The benchmark subtitle shows status only, never live progress or the device."""

    def test_not_benchmarked_yet(self):
        assert _benchmark_status(False) == "Not benchmarked yet"

    def test_benchmarked(self):
        assert _benchmark_status(True) == "Benchmarked"


class TestModelRowState:
    @pytest.mark.parametrize(
        ("model", "active", "downloading", "downloaded", "state"),
        [
            ("base", "base", None, True, "active"),
            ("base", "small", None, True, "downloaded"),
            ("base", "small", None, False, "available"),
            # A device switch can re-download the *active* model; that row must spin,
            # not show the checkmark: downloading outranks active.
            ("base", "base", "base", True, "downloading"),
            # First run: the active model is mid-download and not on disk yet; the
            # spinner wins over both the active marker and the missing file.
            ("base", "base", "base", False, "downloading"),
            # Active but just deleted: re-downloadable, never a phantom checkmark on a
            # model the gate would refuse.
            ("small", "small", None, False, "available"),
        ],
    )
    def test_state(self, model, active, downloading, downloaded, state):
        assert _model_row_state(model, active, downloading, downloaded) == state


class TestModelSubtitle:
    def test_shows_name_and_size_when_downloaded(self):
        assert _model_subtitle("small", True) == "Small, 264 MB"

    def test_shows_not_installed_when_absent(self):
        assert _model_subtitle("small", False) == "Small, not installed"


def _calibrate(timings):
    """Store a benchmark run on this machine: each `model|device` cell's best ms."""
    cells = {key: calibration.CellResult(True, ms) for key, ms in timings.items()}
    report = calibration.Report(fingerprint(), cells)
    calibration.save(calibration.default_path(), report)


@pytest.fixture
def device_settings(display, detected_devices):
    from gi.repository import Adw

    from quoth.config import Config
    from quoth.settings import ModelSwitcher, SettingsController, SettingsView
    from tests.support import FakeModelManager

    Adw.init()
    detected_devices.extend(["NPU", "GPU"])
    _calibrate({"base|GPU": 300, "base|CPU": 900})
    config = Config()
    manager = FakeModelManager(encoder=False)
    section = EngineSection(
        config,
        manager,
        calibration_bridge=None,
        enumerate_microphones=lambda: [],
        toast=lambda _message: None,
    )
    page = Adw.PreferencesPage()
    section.build_transcription(page)
    section.build_acceleration(page)
    downloads = MagicMock()
    toast = MagicMock()
    view = SettingsView(
        section.set_download_active,
        section.set_download_progress,
        toast,
        section.revert,
        section.on_model_changed,
    )

    # Settings shows every change through the section's sync.
    handler = config.connect_changed(section.sync)
    switcher = ModelSwitcher(
        config, manager, downloads, reconcile=MagicMock(), notify=MagicMock()
    )
    controller = SettingsController(
        config, manager, switcher, view, reconcile=MagicMock()
    )
    section.attach_controller(controller)
    controller.open()
    yield SimpleNamespace(
        section=section,
        config=config,
        manager=manager,
        downloads=downloads,
        toast=toast,
        controller=controller,
        switcher=switcher,
        page=page,
    )
    controller.close()
    config.settings.disconnect(handler)


@pytest.mark.parametrize(
    ("on_disk", "details"),
    [
        ({"base", "large-v3-turbo"}, {"base": "82 MB", "large-v3-turbo": "874 MB"}),
        (set(), {"base": "82 MB download", "large-v3-turbo": "874 MB download"}),
    ],
)
def test_model_rows_say_download_only_for_models_not_on_disk(display, on_disk, details):
    # The active and the other downloaded rows show the size alone.
    from gi.repository import Adw

    from quoth.config import Config

    section = EngineSection(
        Config(),
        FakeModelManager(ggml=on_disk),
        calibration_bridge=None,
        enumerate_microphones=lambda: [],
        toast=lambda _message: None,
    )
    section.build_transcription(Adw.PreferencesPage())
    for model, size in details.items():
        subtitle = section._model_subtitles[model].get_label()
        assert subtitle == f"{MODELS_BY_NAME[model].description}\n{size}"


TRANSLATES = "Outputs English"
TRANSCRIBES = "This model only transcribes. Choose a model with English translation."


def test_the_translate_help_names_the_catalogs_translating_models(device_settings):
    # One text whatever the model, so it must follow the catalog.
    tip = device_settings.section._translate_row.get_tooltip_text()
    can, cannot = tip.split("; ")
    for info in AVAILABLE_MODELS:
        assert info.label in (can if info.supports_translation else cannot)


def test_translation_row_follows_the_models_capability(device_settings):
    state = device_settings
    row = state.section._translate_row
    assert (row.get_subtitle(), row.get_sensitive()) == (TRANSLATES, True)

    state.config.model = "large-v3-turbo"

    assert (row.get_subtitle(), row.get_sensitive()) == (TRANSCRIBES, False)


def test_a_saved_translation_outlives_turbo_and_can_be_switched_off(device_settings):
    state = device_settings
    row = state.section._translate_row
    state.config.translate = True
    state.config.model = "large-v3-turbo"
    assert row.get_active() and row.get_sensitive()

    state.config.model = "base"
    assert state.config.translate and row.get_active() and row.get_sensitive()
    assert row.get_subtitle() == TRANSLATES

    state.config.model = "large-v3-turbo"
    row.set_active(False)
    assert state.config.translate is False
    assert not row.get_sensitive()


@pytest.mark.parametrize(
    ("device", "failed", "plain"),
    [
        ("auto", "NPU", "Picks the best device for each model"),
        ("NPU", "NPU", "Selected device: NPU"),
        ("GPU", "GPU", "Selected device: GPU"),
    ],
)
def test_a_fallback_note_shows_for_the_device_in_effect(
    device_settings, device, failed, plain
):
    state = device_settings
    row = state.section._device_row
    state.config.device = device
    note = f"{failed} couldn't start; using CPU"

    state.section.refresh_runtime(note)
    assert row.get_subtitle() == note
    assert state.config.device == device

    state.section.refresh_runtime(None)
    assert row.get_subtitle() == plain


def test_a_choice_not_yet_applied_has_not_fallen_back(device_settings):
    state = device_settings
    row = state.section._device_row
    state.section.refresh_runtime("NPU couldn't start; using CPU")

    row.set_selected(state.section._device_values.index("NPU"))  # downloads first

    assert state.config.device == "auto"
    assert row.get_subtitle() == "Selected device: NPU"


def test_model_rebuild_refreshes_pointer_policy(device_settings):
    section = device_settings.section
    with patch("quoth.engine_section.brand.pointer_cursors") as pointers:
        section.on_model_changed()
    pointers.assert_called_once_with(section._model_expander)


def _translation_badge(root):
    from gi.repository import Gtk

    images = (w for w in descendants(root) if isinstance(w, Gtk.Image))
    icon = "preferences-desktop-locale-symbolic"
    return next((w for w in images if w.get_icon_name() == icon), None)


@pytest.mark.parametrize("model", ["tiny", "base", "small", "large-v3-turbo"])
def test_translation_badge_matches_model_capability(device_settings, model):
    from gi.repository import Gtk

    row = device_settings.section._model_rows[model]
    badge = _translation_badge(row)
    if model == "large-v3-turbo":
        assert badge is None
    else:
        assert badge is not None
        assert badge.get_tooltip_text() == "Translates to English"
        title = badge.get_prev_sibling()
        assert isinstance(title, Gtk.Label)
        assert title.get_label() == model.capitalize()


def test_translation_badge_persists_during_download(device_settings):
    from quoth.downloader import DownloadProgress

    section = device_settings.section
    section.set_download_active("small")
    row = section._model_rows["small"]
    badge = _translation_badge(row)
    assert badge is not None

    section.set_download_progress(DownloadProgress(0.5, 500_000, 1_000_000, 100_000))
    subtitle = section._model_subtitles["small"].get_label()
    assert "50%" in subtitle
    assert "English translation" not in subtitle
    assert _translation_badge(row) is badge


@pytest.mark.parametrize(
    ("model", "name"), [("small", "Small"), ("large-v3-turbo", "Turbo")]
)
def test_screen_readers_hear_which_models_translate(device_settings, model, name):
    # The badge is an image, so the row's name must include it.
    from gi.repository import Gtk

    from quoth.downloader import DownloadProgress

    section = device_settings.section
    row = section._model_rows[model]
    labels = (w for w in descendants(row) if isinstance(w, Gtk.Label))
    title = next(w for w in labels if w.get_label() == name)
    badge = _translation_badge(row)
    names = [title] if badge is None else [title, badge]
    assert (badge is None) == (model == "large-v3-turbo")
    labelled_by = Gtk.AccessibleRelation.LABELLED_BY
    description = Gtk.AccessibleProperty.DESCRIPTION
    subtitle = section._model_subtitles[model]
    assert accessible_mismatch(row, labelled_by, *names) is None
    assert accessible_mismatch(row, description, subtitle.get_label()) is None

    section.set_download_active(model)
    section.set_download_progress(DownloadProgress(0.5, 500_000, 1_000_000, 100_000))
    row = section._model_rows[model]
    subtitle = section._model_subtitles[model]
    assert "50%" in subtitle.get_label()
    assert accessible_mismatch(row, description, subtitle.get_label()) is None


def test_screen_readers_hear_which_model_is_active(device_settings):
    # Shown in a window: a list box overwrites a rooted row's selected state, so
    # the row's name must carry the active mark.
    from gi.repository import Adw, Gtk

    state = device_settings
    window = Adw.Window(content=state.page)
    window.present()
    labelled_by = Gtk.AccessibleRelation.LABELLED_BY

    def names(model, *, active):
        row = state.section._model_rows[model]
        title = next(
            w
            for w in descendants(row)
            if isinstance(w, Gtk.Label) and w.get_label() == model.capitalize()
        )
        check = next(
            (
                w
                for w in descendants(row)
                if isinstance(w, Gtk.Image)
                and w.get_icon_name() == "object-select-symbolic"
            ),
            None,
        )
        assert (check is not None) == active
        if check is not None:
            label = Gtk.AccessibleProperty.LABEL
            assert accessible_mismatch(check, label, "Active") is None
        marks = [] if check is None else [check]
        return row, [title, _translation_badge(row), *marks]

    try:
        assert state.config.model == "base"
        for model, active in (("base", True), ("small", False)):
            row, expected = names(model, active=active)
            assert accessible_mismatch(row, labelled_by, *expected) is None

        state.config.model = "small"

        for model, active in (("base", False), ("small", True)):
            row, expected = names(model, active=active)
            assert row.get_root() is window
            assert accessible_mismatch(row, labelled_by, *expected) is None
    finally:
        window.destroy()


def test_download_progress_keeps_turbo_without_a_translation_badge(device_settings):
    from quoth.downloader import DownloadProgress

    section = device_settings.section
    section.set_download_active("large-v3-turbo")
    row = section._model_rows["large-v3-turbo"]
    subtitle = section._model_subtitles["large-v3-turbo"]
    assert subtitle.get_label().endswith("Downloading…")
    assert _translation_badge(row) is None

    section.set_download_progress(DownloadProgress(0.5, 500_000, 1_000_000, 100_000))
    assert "50%" in subtitle.get_label()


@pytest.mark.parametrize("outcome", ["complete", "error", "cancel"])
def test_npu_selection_download_lifecycle(device_settings, outcome):
    state = device_settings
    row = state.section._device_row
    npu = state.section._device_values.index("NPU")

    row.set_selected(npu)

    state.downloads.start_or_join.assert_called_once()
    assert state.downloads.start_or_join.call_args.args == ("base", "NPU")
    state.toast.assert_not_called()
    assert state.config.device == "auto"
    assert row.get_selected() == npu
    callbacks = state.downloads.start_or_join.call_args.kwargs
    if outcome == "complete":
        state.manager._encoder = True
        callbacks["on_done"]()
        assert state.config.device == "NPU"
        assert row.get_selected() == npu
        state.toast.assert_not_called()
    else:
        if outcome == "error":
            message = "Couldn't download Base. Check your connection and try again."
            callbacks["on_error"](message)
            state.toast.assert_called_once_with(message)
        else:
            state.controller.cancel_download()
            state.downloads.start_or_join.return_value.cancel.assert_called_once()
            state.toast.assert_called_once_with("Download canceled")
        assert state.config.device == "auto"
        assert row.get_selected() == 0
    assert state.switcher.pending is None


def test_repeated_device_switches_with_assets_ready(device_settings):
    state = device_settings
    state.manager._encoder = True
    row = state.section._device_row

    for device in ["NPU", "CPU", "GPU", "auto"] * 3:
        selected = state.section._device_values.index(device)
        row.set_selected(selected)
        assert state.config.device == device
        assert row.get_selected() == selected

    state.downloads.start_or_join.assert_not_called()
    state.toast.assert_not_called()


def test_device_label_refresh_preserves_selection_without_starting_download(
    device_settings,
):
    state = device_settings
    row = state.section._device_row
    _calibrate({"base|CPU": 300, "base|GPU": 900})

    state.section.on_model_changed()

    assert row.get_model().get_string(0) == "Automatic (CPU)"
    assert row.get_selected() == 0
    assert state.config.device == "auto"
    state.downloads.start_or_join.assert_not_called()
    state.toast.assert_not_called()


def test_selectors_show_a_change_from_elsewhere(device_settings):
    from quoth.config import Config
    from quoth.languages import LANGUAGES

    state = device_settings
    Config().language = "de"  # the tray, say
    Config().model = "small"
    codes = [code for _, code in LANGUAGES]
    assert state.section._lang_row.get_selected() == codes.index("de")
    assert state.section._model_expander.get_subtitle().startswith("Small")
    assert state.config.language == "de"  # showing it wrote nothing back
    state.downloads.start_or_join.assert_not_called()


def test_an_unplugged_microphone_stays_saved(display):
    from gi.repository import Adw

    from quoth.config import Config

    config = Config()
    config.mic_source = "usb-desk"
    section = EngineSection(
        config,
        MagicMock(),
        calibration_bridge=None,
        enumerate_microphones=lambda: [
            {"name": "usb-desk", "display_name": "Desk microphone"}
        ],
        toast=lambda _message: None,
    )
    section.build_audio(Adw.PreferencesPage())
    names = section._mic_row.get_model()
    assert [names.get_string(i) for i in range(names.get_n_items())] == [
        "Auto",
        "Desk microphone",
    ]
    assert section._mic_row.get_selected() == 1
    section._mic_row.set_selected(0)
    assert config.mic_source == "auto"
    section._mic_row.set_selected(1)
    assert config.mic_source == "usb-desk"
    config.mic_source = "usb-headset"  # another writer names one not plugged in
    section.sync("mic_source")
    assert section._mic_row.get_selected() == 0  # shown as Auto
    assert config.mic_source == "usb-headset"


def test_a_failed_microphone_discovery_offers_only_auto(display, caplog):
    from gi.repository import Adw

    from quoth.config import Config

    section = EngineSection(
        Config(),
        MagicMock(),
        calibration_bridge=None,
        enumerate_microphones=MagicMock(side_effect=OSError("no audio server")),
        toast=lambda _message: None,
    )
    caplog.set_level("ERROR")
    caplog.clear()  # building the settings logs where they are kept

    section.build_audio(Adw.PreferencesPage())

    names = section._mic_row.get_model()
    assert [names.get_string(i) for i in range(names.get_n_items())] == ["Auto"]
    [record] = caplog.records
    assert record.levelname == "ERROR"


def test_a_saved_device_this_machine_lacks_is_shown_and_survives_a_refusal(display):
    # A vanished NPU, say. A refused change snaps back to it without raising, and
    # the other rows still answer the user.
    from gi.repository import Adw

    from quoth.config import Config
    from quoth.settings import PendingChange, SettingsController, SettingsView
    from tests.support import FakeModelManager

    Adw.init()
    config = Config()
    config.device = "NPU"
    manager = FakeModelManager(encoder=False)
    section = EngineSection(
        config,
        manager,
        calibration_bridge=None,
        enumerate_microphones=lambda: [],
        toast=lambda _message: None,
    )
    page = Adw.PreferencesPage()
    section.build_transcription(page)
    section.build_acceleration(page)
    row = section._device_row
    assert row.get_model().get_string(row.get_selected()) == "NPU (not available)"

    view = SettingsView(
        section.set_download_active,
        section.set_download_progress,
        MagicMock(),
        section.revert,
        section.on_model_changed,
    )
    # A model download in flight refuses the device change.
    switcher = MagicMock(pending=PendingChange("model", "base", "base", "CPU"))
    switcher.select.return_value = False
    controller = SettingsController(
        config, manager, switcher, view, reconcile=MagicMock()
    )
    section.attach_controller(controller)
    row.set_selected(section._device_values.index("CPU"))
    assert config.device == "NPU"
    assert row.get_model().get_string(row.get_selected()) == "NPU (not available)"
    assert row.get_subtitle() == "Selected device: NPU"
    section._lang_row.set_selected(section._lang_codes.index("de"))
    assert config.language == "de"
