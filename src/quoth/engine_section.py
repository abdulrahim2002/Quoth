# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Settings for the transcription engine: model, language, device, benchmark, mic."""

from collections.abc import Callable
from dataclasses import dataclass

import gi
from loguru import logger

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, GLib, Gtk

from quoth import brand
from quoth.config import Config
from quoth.download_format import format_download_progress
from quoth.downloader import DownloadProgress
from quoth.hardware import available_devices
from quoth.languages import LANGUAGES
from quoth.models import AVAILABLE_MODELS, MODELS_BY_NAME, ModelInfo, ModelManager
from quoth.resolution import resolve_for
from quoth.settings import SettingsController


def fallback_note(device: str, runs_on: str) -> str:
    """Return the note shown when `device` could not start and `runs_on` took over."""
    return f"{device} couldn't start; using {runs_on}"


def _select_value(row: Adw.ComboRow, values: list[str], current: str) -> None:
    """Select the entry matching `current` in a value-mapped row, else the first one."""
    try:
        row.set_selected(values.index(current))
    except ValueError:
        row.set_selected(0)


def _benchmark_status(benchmarked: bool) -> str:
    """Return the benchmark row's subtitle: status only, not the effective device."""
    return "Benchmarked" if benchmarked else "Not benchmarked yet"


def _model_row_state(
    name: str, active: str, downloading: str | None, is_downloaded: bool
) -> str:
    """Return a row's state: "downloading", "active", "downloaded" or "available".

    A download in flight wins even over the active model, whose encoder a device switch
    may be fetching. "active" also needs the model on disk, so a deleted but still
    configured model shows as "available".
    """
    if name == downloading:
        return "downloading"
    if name == active and is_downloaded:
        return "active"
    return "downloaded" if is_downloaded else "available"


def _model_subtitle(model_name: str, is_downloaded: bool) -> str:
    """Return the model expander's subtitle: the active model's name and size.

    It reads "…, not installed" when the weights are not on disk, so the subtitle
    never claims a model the record gate would refuse.
    """
    info = MODELS_BY_NAME[model_name]
    if not is_downloaded:
        return f"{info.label}, not installed"
    return f"{info.label}, {info.ggml_size_mb} MB"


def _model_row_subtitle(info: ModelInfo, download: str) -> str:
    return f"{info.description}\n{download}"


@dataclass
class CalibrationBridge:
    """The app-side calibration affordances the settings view consumes.

    `ranking` (the measured per-model order, or None if never benchmarked) seeds the
    benchmark status at open; `run` opens the benchmark dialog. The app pushes finished
    state back through `refresh_benchmark_status`.
    """

    ranking: dict[str, list[str]] | None
    run: Callable[[], None]
    clear_cache: Callable[[], bool]  # False means the engine was busy


class EngineSection:
    """The transcription engine's settings: model, language, device and microphone.

    They live together because the effective device is model-aware: selecting a model
    re-labels the Automatic device. A coordinator, not a widget: it adds the
    Transcription, Hardware acceleration and Audio groups to the pages it is given.
    """

    def __init__(
        self,
        config: Config,
        model_mgr: ModelManager,
        *,
        calibration_bridge: CalibrationBridge | None,
        enumerate_microphones: Callable[[], list[dict]],
        toast: Callable[[str], None],
    ):
        self._config = config
        self._model_mgr = model_mgr
        self._enumerate_microphones = enumerate_microphones
        self._cal_bridge = calibration_bridge
        self._toast = toast
        self._controller: SettingsController | None = None
        self._window: Adw.PreferencesWindow | None = None
        self._suppress = False  # guards programmatic combo reverts/re-selects
        self._downloading: str | None = None  # model whose download is in flight
        self._dl_progress: DownloadProgress | None = None
        self._cal_benchmarked = bool(calibration_bridge and calibration_bridge.ranking)
        self._cal_status_row: Adw.ActionRow | None = None
        self._acceleration_group: Adw.PreferencesGroup | None = None
        self._cache_row: Adw.ActionRow | None = None
        self._model_expander: Adw.ExpanderRow | None = None
        self._model_rows: dict[str, Adw.ActionRow] = {}
        self._model_subtitles: dict[str, Gtk.Label] = {}
        self._lang_row: Adw.ComboRow | None = None
        self._translate_row: Adw.SwitchRow | None = None
        self._lang_codes: list[str] = []
        self._device_row: Adw.ComboRow | None = None
        self._fallback: str | None = None
        self._device_values: list[str] = []
        self._mic_row: Adw.ComboRow | None = None
        self._mic_values: list[str] = []

    def attach_controller(self, controller: SettingsController) -> None:
        """Wire the controller in after construction; its view callbacks point here."""
        self._controller = controller

    def attach(self, window: Adw.PreferencesWindow) -> None:
        """Set the window that parents this section's confirm dialogs.

        Call before building any group.
        """
        self._window = window

    def build_transcription(self, page: Adw.PreferencesPage) -> None:
        transcription = Adw.PreferencesGroup(title="Transcription")
        page.add(transcription)

        self._model_expander = Adw.ExpanderRow(
            title="Model",
            subtitle=self._model_expander_subtitle(),
            tooltip_text=(
                "Larger models are more accurate but slower and use more memory"
            ),
        )
        # Named by its title: an expander row has no name of its own, and GTK
        # would otherwise take its tooltip.
        self._model_expander.update_property(
            [Gtk.AccessibleProperty.LABEL], [self._model_expander.get_title()]
        )
        transcription.add(self._model_expander)
        self._model_rows = {}
        self._model_subtitles = {}
        for info in AVAILABLE_MODELS:
            row = self._build_model_row(info)
            self._model_expander.add_row(row)
            self._model_rows[info.name] = row

        lang_row = Adw.ComboRow(
            title="Spoken language",
            subtitle="The language you speak, even when translating",
            tooltip_text="Auto-detect guesses the language from each recording",
        )
        lang_row.set_model(Gtk.StringList.new([label for label, _ in LANGUAGES]))
        self._lang_codes = [code for _, code in LANGUAGES]
        _select_value(lang_row, self._lang_codes, self._config.language)
        lang_row.connect("notify::selected", self._on_language_changed)
        self._lang_row = lang_row
        transcription.add(lang_row)

        translate_row = Adw.SwitchRow(
            title="Translate to English",
            tooltip_text="Tiny, Base and Small can translate; Turbo cannot",
        )
        self._config.bind("translate", translate_row, "active")
        self._translate_row = translate_row
        self._refresh_translation_row()
        transcription.add(translate_row)

    def build_acceleration(self, page: Adw.PreferencesPage) -> None:
        """Add the Hardware acceleration group: which device runs inference.

        The benchmark and NPU cache rows are built only when the app supplies a bridge.
        """
        group = Adw.PreferencesGroup(
            title="Hardware acceleration",
            description=(
                "Choose the device that runs transcription, or benchmark to find the "
                "fastest for each model. The benchmark runs locally on your downloaded "
                "models."
            ),
        )
        page.add(group)
        self._acceleration_group = group

        # The combo offers only what this build can run, plus a saved device the
        # machine has lost, so the stored choice survives a missing driver.
        # 'Automatic' is labelled with what the model-aware resolver picks.
        available = available_devices()
        auto_dev = resolve_for(self._config.model, "auto", self._model_mgr).device
        self._device_values = ["auto", *available]
        labels = [f"Automatic ({auto_dev})", *available]
        if self._config.device not in self._device_values:
            self._device_values.append(self._config.device)
            labels.append(f"{self._config.device} (not available)")
        device_row = Adw.ComboRow(
            title="Device",
            tooltip_text=(
                "Automatic uses benchmark results, or else GPU, then NPU, then CPU"
            ),
        )
        device_row.set_model(Gtk.StringList.new(labels))
        _select_value(device_row, self._device_values, self._config.device)
        self._describe_device(device_row, self._config.device)
        device_row.connect("notify::selected", self._on_device_changed)
        group.add(device_row)
        self._device_row = device_row

        if self._cal_bridge is None:
            return
        row = Adw.ActionRow(
            title="Benchmark",
            subtitle=_benchmark_status(self._cal_benchmarked),
            tooltip_text=(
                "Results are kept until the devices or the speech engine change"
            ),
        )
        run = Gtk.Button(label="Run", valign=Gtk.Align.CENTER)
        run.add_css_class("suggested-action")
        run.connect("clicked", lambda _b: self._cal_bridge.run())
        row.add_suffix(run)
        group.add(row)
        self._cal_status_row = row

        self.refresh_cache_status()

    def _add_cache_row(self, group: Adw.PreferencesGroup) -> Adw.ActionRow:
        """Add the NPU disk-cache control when storage exists or cannot be inspected."""
        row = Adw.ActionRow(
            title="NPU cache",
            tooltip_text=(
                "Holds compiled NPU encoder files, rebuilt when the NPU next loads a model"
            ),
        )
        clear = Gtk.Button(label="Clear", valign=Gtk.Align.CENTER)
        clear.add_css_class("destructive-action")
        clear.connect("clicked", lambda _b: self._confirm_clear_cache())
        row.add_suffix(clear)
        group.add(row)
        brand.pointer_cursors(row)
        self._cache_row = row
        return row

    def refresh_cache_status(self) -> None:
        """Refresh cache size and row visibility from the filesystem's current state."""
        if self._cal_bridge is None or self._acceleration_group is None:
            return
        try:
            size_bytes: int | None = self._model_mgr.cache_size_bytes()
        except OSError as error:
            logger.warning("Could not inspect NPU cache: {}", error)
            size_bytes = None
        if size_bytes == 0:
            if self._cache_row is not None:
                self._cache_row.set_visible(False)
            return
        row = self._cache_row or self._add_cache_row(self._acceleration_group)
        row.set_subtitle(
            "Size unavailable"
            if size_bytes is None
            else f"{GLib.format_size(size_bytes)} on disk"
        )
        row.set_visible(True)

    def _confirm_clear_cache(self) -> None:
        dialog = Adw.AlertDialog(
            heading="Clear NPU cache?",
            body=(
                "Removes prepared NPU model files to free disk space. Downloaded models and "
                "benchmark results are kept. The next time a model needs NPU preparation, "
                "loading it may take longer."
            ),
        )
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("clear", "Clear")
        dialog.set_response_appearance("clear", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_default_response("cancel")
        dialog.set_close_response("cancel")
        dialog.connect("response", self._on_clear_cache_response)
        dialog.connect("map", brand.pointer_cursors)
        dialog.present(self._window)

    def _on_clear_cache_response(self, _dialog, response: str) -> None:
        if response != "clear":
            return
        try:
            cleared = self._cal_bridge.clear_cache()
        except OSError as e:
            logger.error("Could not clear NPU cache: {}", e)
            self._toast("Couldn't clear NPU cache")
        else:
            self._toast(
                "NPU cache cleared"
                if cleared
                else "NPU cache is busy. Try again when Quoth is idle."
            )
        self.refresh_cache_status()

    def build_audio(self, page: Adw.PreferencesPage) -> None:
        audio = Adw.PreferencesGroup(title="Audio")
        page.add(audio)

        mic_row = Adw.ComboRow(
            title="Microphone",
            subtitle="Input source for recording",
            tooltip_text="Auto follows the system's default input",
        )
        try:
            mics = self._enumerate_microphones()
        except Exception:
            logger.exception("Failed to enumerate microphones")
            mics = []
        mic_names = ["Auto"] + [m["display_name"] for m in mics]
        self._mic_values = ["auto"] + [m["name"] for m in mics]
        mic_row.set_model(Gtk.StringList.new(mic_names))
        _select_value(mic_row, self._mic_values, self._config.mic_source)
        mic_row.connect("notify::selected", self._on_mic_changed)
        audio.add(mic_row)
        self._mic_row = mic_row

    def _model_expander_subtitle(self) -> str:
        return _model_subtitle(
            self._config.model, self._model_mgr.is_downloaded(self._config.model)
        )

    def _build_model_row(self, info: ModelInfo) -> Adw.ActionRow:
        """Build one model row; the suffix only reflects state.

        Clicking a downloaded or available row uses that model, downloading it first if
        needed. Active and downloading rows do not activate; only their buttons click.
        """
        state = _model_row_state(
            info.name,
            self._config.model,
            self._downloading,
            self._model_mgr.is_downloaded(info.name),
        )
        if state == "downloading":
            download = (
                format_download_progress(self._dl_progress)
                if self._dl_progress is not None
                else "Downloading…"
            )
        elif state == "available":
            download = f"{info.ggml_size_mb} MB download"
        else:
            download = f"{info.ggml_size_mb} MB"
        row = Adw.ActionRow()
        # ActionRow has no title-widget slot; keep the name and badge in one prefix.
        block = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            valign=Gtk.Align.CENTER,
            hexpand=True,
        )
        line = Gtk.Box(spacing=8)
        title = Gtk.Label(label=info.label, xalign=0)
        title.add_css_class("title")
        line.append(title)
        names = [title]
        if info.supports_translation:
            badge = Gtk.Image.new_from_icon_name("preferences-desktop-locale-symbolic")
            badge.set_valign(Gtk.Align.CENTER)
            badge.set_tooltip_text("Translates to English")
            badge.update_property(
                [Gtk.AccessibleProperty.LABEL], ["Translates to English"]
            )
            line.append(badge)
            names.append(badge)
        if state == "active":
            # A list box overwrites its rows' selected state, so the name carries it.
            check = Gtk.Image.new_from_icon_name("object-select-symbolic")
            check.update_property([Gtk.AccessibleProperty.LABEL], ["Active"])
            check.add_css_class("dim-label")
            names.append(check)
        subtitle = Gtk.Label(
            label=_model_row_subtitle(info, download), xalign=0, wrap=True
        )
        subtitle.add_css_class("subtitle")
        block.append(line)
        block.append(subtitle)
        row.add_prefix(block)
        # The row's empty title would name it; point it at the title line instead,
        # so the badge's meaning and the active mark are read out too.
        row.update_relation(
            [Gtk.AccessibleRelation.LABELLED_BY],
            [Gtk.AccessibleList.new_from_list(names)],
        )
        row.reset_relation(Gtk.AccessibleRelation.DESCRIBED_BY)
        row.update_property(
            [Gtk.AccessibleProperty.DESCRIPTION], [subtitle.get_label()]
        )
        self._model_subtitles[info.name] = subtitle

        # Spinner + a Cancel button while its own download is in flight.
        if state == "downloading":
            row.add_suffix(Gtk.Spinner(spinning=True, valign=Gtk.Align.CENTER))
            row.add_suffix(self._download_cancel_button())
            return row

        # Active: a marker and a delete button, since recording is gated on
        # availability. Not activatable, so a stray click can't reload the model.
        if state == "active":
            row.add_suffix(check)
            row.add_suffix(self._model_delete_button(info))
            return row

        if state == "downloaded":
            row.add_suffix(self._model_delete_button(info))
        else:  # "available": not on disk; selecting it triggers the download
            hint = Gtk.Image.new_from_icon_name("folder-download-symbolic")
            hint.add_css_class("dim-label")
            hint.set_tooltip_text("Downloads when selected")
            row.add_suffix(hint)

        row.set_activatable(True)
        row.connect("activated", lambda _r, n=info.name: self._on_model_selected(n))
        return row

    def _on_model_selected(self, name: str):
        self._controller.select_model(name)

    def _on_model_delete(self, name: str):
        self._controller.delete_model(name)

    def _download_cancel_button(self) -> Gtk.Button:
        """Build the flat stop button that cancels the in-flight download.

        The controller aborts the download, which reverts the row.
        """
        btn = Gtk.Button(
            icon_name="process-stop-symbolic",
            valign=Gtk.Align.CENTER,
            tooltip_text="Cancel download",
        )
        btn.add_css_class("flat")
        btn.update_property([Gtk.AccessibleProperty.LABEL], ["Cancel download"])
        btn.connect("clicked", lambda _b: self._controller.cancel_download())
        return btn

    def _model_delete_button(self, info: ModelInfo) -> Gtk.Button:
        """Build the dim, flat trash button shared by the active and downloaded rows."""
        label = f"Delete {info.label}"
        btn = Gtk.Button(
            icon_name="user-trash-symbolic",
            valign=Gtk.Align.CENTER,
            tooltip_text=label,
        )
        btn.add_css_class("flat")
        btn.add_css_class("dim-label")  # destructive stays secondary
        btn.update_property([Gtk.AccessibleProperty.LABEL], [label])
        btn.connect("clicked", lambda _b, n=info.name: self._on_model_delete(n))
        return btn

    def set_download_active(self, model: str | None):
        """Record that a download started (`model`) or finished or failed (None).

        The rows are rebuilt so the affected one shows or clears its spinner and
        progress subtitle.
        """
        self._downloading = model
        self._dl_progress = None
        self.on_model_changed()

    def set_download_progress(self, progress: DownloadProgress):
        """Show live progress for the in-flight download, updating only its subtitle."""
        self._dl_progress = progress
        if self._downloading:
            row = self._model_rows.get(self._downloading)
            if row is not None:
                text = _model_row_subtitle(
                    MODELS_BY_NAME[self._downloading],
                    format_download_progress(progress),
                )
                self._model_subtitles[self._downloading].set_label(text)
                row.update_property([Gtk.AccessibleProperty.DESCRIPTION], [text])

    def revert(self, key: str):
        """Restore the touched widget to the value it should currently show.

        That is the committed config, or the in-flight target mid-download
        (`effective_device`). Used both on download failure and when a conflicting
        change is refused.
        """
        if key == "model":
            self.on_model_changed()
        elif key == "device":
            self._refresh_device_label()

    def on_model_changed(self):
        """Re-render every display derived from the active model, whoever changed it.

        It spans both groups because the "Automatic (X)" device hint is model-aware.
        """
        self._render_model_rows()
        self._refresh_translation_row()
        self._refresh_device_label()

    def _render_model_rows(self):
        """Rebuild every model row and the expander subtitle from current state.

        The catalog is small, so a full rebuild keeps the render a pure function of
        state.
        """
        self._model_expander.set_subtitle(self._model_expander_subtitle())
        for info in AVAILABLE_MODELS:
            old_row = self._model_rows[info.name]
            self._model_expander.remove(old_row)
            new_row = self._build_model_row(info)
            self._model_expander.add_row(new_row)
            self._model_rows[info.name] = new_row
        brand.pointer_cursors(self._model_expander)

    def _refresh_device_label(self):
        """Relabel "Automatic (X)" with what `auto` resolves to for the current model.

        The effective device is model-aware, so switching models can change it. The
        reselect is suppressed so it does not fire as a user edit.
        """
        if self._device_row is None:
            return
        auto_dev = resolve_for(self._config.model, "auto", self._model_mgr).device
        label = f"Automatic ({auto_dev})"
        model = self._device_row.get_model()
        self._suppress = True
        # Replacing the model during selection notification invalidates GTK's active
        # selection object. Keep it alive and change only the automatic label.
        if model.get_string(0) != label:
            model.splice(0, 1, [label])
        # The pending device while its download runs, like the model rows.
        shown = (
            self._controller.effective_device()
            if self._controller is not None
            else self._config.device
        )
        _select_value(self._device_row, self._device_values, shown)
        self._suppress = False
        self._describe_device(self._device_row, shown)

    def _refresh_translation_row(self) -> None:
        """Show whether the active model can honour the saved translation request."""
        if self._translate_row is None:
            return
        supported = MODELS_BY_NAME[self._config.model].supports_translation
        if supported:
            subtitle = "Outputs English"
        else:
            subtitle = (
                "This model only transcribes. Choose a model with English translation."
            )
        self._translate_row.set_subtitle(subtitle)
        # Keep a saved on value actionable so the user can turn it off. Once off,
        # an unsupported model cannot create the invalid combination again.
        self._translate_row.set_sensitive(supported or self._config.translate)

    def _on_language_changed(self, row, pspec):
        if self._suppress:  # programmatic re-select from sync, not the user
            return
        idx = row.get_selected()
        if idx < len(self._lang_codes):
            self._config.language = self._lang_codes[idx]

    def sync(self, name: str) -> None:
        """Show a setting that changed, whoever wrote it.

        The writer may be this window, the tray, a landed download or `gsettings`.
        Reselection is suppressed so it does not re-fire as a user edit.
        """
        if name == "language" and self._lang_row is not None:
            self._suppress = True
            _select_value(self._lang_row, self._lang_codes, self._config.language)
            self._suppress = False
        elif name == "mic_source" and self._mic_row is not None:
            self._suppress = True
            _select_value(self._mic_row, self._mic_values, self._config.mic_source)
            self._suppress = False
        elif name == "translate":
            self._refresh_translation_row()
        elif name == "model":
            self.on_model_changed()
        elif name == "device":
            self._refresh_device_label()

    def _on_device_changed(self, row, pspec):
        if self._suppress:
            return
        value = self._device_values[row.get_selected()]
        self._describe_device(row, value)
        self._controller.select_device(value)

    def refresh_runtime(self, fallback: str | None) -> None:
        """Refresh the device row and NPU cache row for the engine's runtime state.

        `fallback` is the note for a device that could not start, None when none failed.
        """
        self._fallback = fallback
        self._refresh_device_label()
        self.refresh_cache_status()

    def _describe_device(self, row: Adw.ComboRow, value: str) -> None:
        """Set the subtitle to what the chosen option does, not what Automatic would.

        A fallback belongs to the setting in effect, never to a choice not yet applied.
        """
        if self._fallback is not None and value == self._config.device:
            row.set_subtitle(self._fallback)
        elif value == "auto":
            row.set_subtitle("Picks the best device for each model")
        else:
            row.set_subtitle(f"Selected device: {value.upper()}")

    def _on_mic_changed(self, row, pspec):
        # A re-select from sync may fall back to Auto for an unplugged microphone,
        # which must not overwrite the saved one.
        if self._suppress:
            return
        idx = row.get_selected()
        if idx < len(self._mic_values):
            self._config.mic_source = self._mic_values[idx]

    def refresh_benchmark_status(self, benchmarked: bool) -> None:
        """Show the benchmark state on its row once the benchmark dialog has closed."""
        self._cal_benchmarked = benchmarked
        if self._cal_status_row is not None:
            self._cal_status_row.set_subtitle(_benchmark_status(benchmarked))
