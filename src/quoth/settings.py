# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Settings logic: model and device changes, which may need a download first.

GTK-free so the decision logic (asset gating, the download-then-apply flow,
revert-on-failure) is testable without GTK. The window (`settings_window.py`) is the
humble view: it builds widgets, forwards model and device choices here, and implements
the `SettingsView` effects this controller drives. Its other rows write their setting
directly, and the app applies each change as it lands (`Config.connect_changed`).

Every model or device change goes through the app's `ModelSwitcher`, which outlives
any window and also serves the tray: one with missing assets downloads through the app's
`DownloadCoordinator` (quoth/downloads.py), applies when the assets land, and shows its
progress in whichever Settings window is open, or none.
"""

from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

from quoth import downloader
from quoth.downloads import DownloadCoordinator, Subscription
from quoth.models import AVAILABLE_MODELS, model_label, needs_openvino_encoder
from quoth.resolution import provision_for


@dataclass
class SettingsView:
    """The view-side effects the controller drives, supplied by the GTK dialog.

    This is the humble-object boundary, in the same role as `RecordingCallbacks`.
    """

    # The model name while fetching; None when the download ends.
    set_download_active: Callable[[str | None], None]
    # Live download progress: the overall fraction plus the current file's bytes/speed.
    set_download_progress: Callable[[downloader.DownloadProgress], None]
    toast: Callable[[str], None]
    # Restore the touched widget to the value it should currently show.
    revert: Callable[[str], None]
    # The active model changed: re-render everything derived from it (the model rows and
    # subtitle, and the model-aware "Automatic (X)" device hint).
    on_model_changed: Callable[[], None]


@dataclass(frozen=True)
class PendingChange:
    """A model or device change waiting on its download."""

    key: str  # "model" or "device"
    value: str
    model: str  # whose assets are downloading
    device: str  # the device they are fetched for (decides the NPU encoder)


class ModelSwitcher:
    """The one pending model or device change, owned by the app for its lifetime.

    A change whose assets are missing downloads first and applies when they land,
    whether or not a Settings window is open; an open window shows the progress, and
    with none the outcome is a desktop notification. Only one change can be pending.
    Main-thread only.
    """

    def __init__(
        self,
        config,
        model_mgr,
        downloads: DownloadCoordinator,
        *,
        reconcile: Callable[[], None],
        notify: Callable[[str], None],
    ):
        self._config = config
        self._model_mgr = model_mgr
        self._downloads = downloads
        # The app fits the engine to the files on disk when no setting changes.
        self._reconcile = reconcile
        self._notify = notify  # a desktop notification, when no window is open
        self._pending: PendingChange | None = None
        self._subscription: Subscription | None = None
        self._progress: downloader.DownloadProgress | None = None
        self._view: SettingsView | None = None

    @property
    def pending(self) -> PendingChange | None:
        return self._pending

    def attach(self, view: SettingsView) -> None:
        """Show a newly opened Settings window any download already running."""
        self._view = view
        if self._pending is not None:
            view.set_download_active(self._pending.model)
            if self._progress is not None:
                view.set_download_progress(self._progress)

    def detach(self, view: SettingsView) -> None:
        """Detach a closed window; its download carries on and is announced instead."""
        if self._view is view:
            self._view = None

    def select(self, key: str, value: str) -> bool:
        """Apply `key` = `value`, downloading the assets it needs first.

        The one selection rule behind Settings and the tray. While another change is
        pending it is refused, since that change's landing would overwrite this one.
        Returns whether the change was accepted.
        """
        if self._pending is not None:
            logger.debug(
                "Refused {} change to '{}': '{}' download in flight",
                key,
                value,
                self._pending.model,
            )
            self._announce(f"{model_label(self._pending.model)} is still downloading")
            return False
        model = value if key == "model" else self._config.model
        preference = value if key == "device" else self._config.device
        device = provision_for(model, preference, self._model_mgr)
        if self.assets_ready(model, device):
            setattr(self._config, key, value)
        else:
            self.request(key, value, model, device)
            if self._view is None:
                # No window shows the progress, so say what the wait is for.
                self._notify(f"Downloading {model_label(model)} for the {device}")
        return True

    def assets_ready(self, model: str, device: str) -> bool:
        """Whether the on-disk assets for (model, device) are complete.

        Always the ggml weights, plus the OpenVINO encoder when the device needs it.
        """
        if not self._model_mgr.is_downloaded(model):
            return False
        return not needs_openvino_encoder(
            device, model
        ) or self._model_mgr.has_openvino(model)

    def request(self, key: str, value: str, model: str, device: str) -> None:
        """Download `model`'s assets for `device`, then apply `key` = `value`.

        Callers check `pending` first: one change at a time.
        """
        self._pending = PendingChange(key, value, model, device)
        self._progress = downloader.DownloadProgress(0.0, 0, 0, 0.0)
        if self._view is not None:
            self._view.set_download_active(model)
            self._view.set_download_progress(self._progress)
        self._subscription = self._downloads.start_or_join(
            model,
            device,
            on_progress=self._on_progress,
            on_done=self._on_done,
            on_error=self._on_error,
        )

    def cancel(self) -> None:
        """Cancel the pending change at the user's request and snap the view back.

        This leaves the download, which stops unless someone else is waiting on it. A
        no-op when nothing is pending.
        """
        change = self._pending
        if change is None:
            return
        self._subscription.cancel()
        self._end()
        if self._view is not None:
            self._view.set_download_active(None)
            self._view.toast("Download canceled")
            self._view.revert(change.key)

    def _on_progress(self, progress: downloader.DownloadProgress) -> None:
        self._progress = progress
        if self._view is not None:
            self._view.set_download_progress(progress)

    def _on_done(self) -> None:
        change = self._pending
        # Cleared before committing, so the view renders the committed value.
        self._end()
        if getattr(self._config, change.key) == change.value:
            # Files for the value already set: no change will apply them.
            self._reconcile()
        else:
            setattr(self._config, change.key, change.value)
        if self._view is not None:
            self._view.set_download_active(None)
        # Applying the change re-resolves the device, and the app explains a degrade.
        if change.key == "model":
            self._announce(f"{model_label(change.model)} ready")

    def _on_error(self, message: str) -> None:
        change = self._pending
        self._end()
        if self._view is not None:
            self._view.set_download_active(None)
            self._view.revert(change.key)
        self._announce(message)

    def _end(self) -> None:
        self._pending = None
        self._subscription = None
        self._progress = None

    def _announce(self, text: str) -> None:
        if self._view is not None:
            self._view.toast(text)
        else:
            self._notify(text)


class SettingsController:
    """The GTK-free owner of one Settings window's model and device changes.

    It also deletes models. Every change goes through the app's `ModelSwitcher`.
    """

    def __init__(
        self,
        config,
        model_mgr,
        switcher: ModelSwitcher,
        view: SettingsView,
        *,
        reconcile: Callable[[], None],
    ):
        self._config = config
        self._model_mgr = model_mgr
        self._switcher = switcher
        self._view = view
        # The app fits the engine to the files on disk when no setting changes.
        self._reconcile = reconcile

    # --- model management ---

    def select_model(self, value: str) -> None:
        self._select("model", value)

    def delete_model(self, name: str) -> None:
        """Delete the model's on-disk files, the active model included.

        Recording refuses to start while no model is available. A deleted active model
        falls back to the largest one left; with none left, the loaded context is
        dropped. A model still downloading lands afterwards and wins.
        """
        try:
            self._model_mgr.delete_model(name)
        except OSError as e:
            logger.warning("Could not fully delete model '{}': {}", name, e)
            if self._model_mgr.is_downloaded(name):
                self._view.toast(f"Couldn't delete {model_label(name)}")
                self._view.on_model_changed()
                return
        if name == self._config.model:
            fallback = self._largest_downloaded()
            if fallback:
                # The app applies the change: the fallback loads in place of the old.
                logger.debug(
                    "Deleted active model '{}': falling back to '{}'", name, fallback
                )
                # Not a selection: the setting must name a model on disk now, and a
                # delete never starts a download.
                self._config.model = fallback
                self._view.toast(
                    f"Deleted {model_label(name)}, now using {model_label(fallback)}"
                )
            else:
                # No fallback: config keeps pointing at the (absent) model for easy
                # re-download, so no change reaches the app. It drops the loaded
                # context, which the keyed cache would otherwise keep using.
                logger.warning(
                    "Deleted last model '{}': download required before dictating", name
                )
                self._reconcile()
                self._view.toast(
                    f"Deleted {model_label(name)}: download a model before dictating"
                )
        self._view.on_model_changed()

    def _largest_downloaded(self) -> str | None:
        """Return the downloaded model with the most weight (best accuracy), or None.

        It re-homes the active marker when the active model is deleted.
        """
        present = [m for m in AVAILABLE_MODELS if self._model_mgr.is_downloaded(m.name)]
        if not present:
            return None
        return max(present, key=lambda m: m.ggml_size_mb).name

    # --- model / device: gated on asset availability ---

    def select_device(self, value: str) -> None:
        self._select("device", value)

    def effective_device(self) -> str:
        """Return the device the UI should currently show.

        That is the in-flight target while a device download runs, else the committed
        value, so a conflicting revert snaps the dropdown back to the right value.
        """
        pending = self._switcher.pending
        if pending is not None and pending.key == "device":
            return pending.value
        return self._config.device

    def _select(self, key: str, value: str) -> None:
        if not self._switcher.select(key, value):
            # Refused during a download: the widget reverts to what it should show.
            self._view.revert(key)

    def cancel_download(self) -> None:
        self._switcher.cancel()

    def open(self) -> None:
        """Show the ready window any download an earlier window left running."""
        self._switcher.attach(self._view)

    def close(self) -> None:
        """Detach the closing window; its download, if any, carries on without it."""
        self._switcher.detach(self._view)
