# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Model-download coordination: the single owner of in-flight downloads.

`download_async` runs one download off the main thread. `DownloadCoordinator` adds
single-flight over it, so a second request for a model already downloading joins the
running transfer instead of racing it on the same partial file. The worker's callbacks
arrive through the injected scheduler, so the coordinator's state lives on the main
thread and needs no lock.
"""

from __future__ import annotations

import errno
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger

from quoth import downloader
from quoth.models import model_label
from quoth.scheduler import guarded


def download_async(
    models_dir,
    model: str,
    device: str,
    scheduler,
    *,
    on_progress: Callable[[downloader.DownloadProgress], None],
    on_done: Callable[[], None],
    on_error: Callable[[str], None],
    on_cancelled: Callable[[], None] | None = None,
    cancel: threading.Event | None = None,
    download_fn: Callable[..., None] = downloader.download_model,
) -> threading.Thread:
    """Run a model download off the main thread, with callbacks through `scheduler`.

    Returns the worker thread (joinable in tests). `on_done()` fires on success,
    `on_error(message)` on failure with the copy to show the user, and
    `on_cancelled()` when the caller sets `cancel` mid-download.
    """

    def _work():
        try:
            download_fn(
                models_dir,
                model,
                device,
                on_progress=lambda p: scheduler.dispatch(on_progress, p),
                cancel=cancel,
            )
            # A cancel that raced the last byte is still a cancel: the caller has
            # already moved on and must not be told the download completed.
            if cancel is not None and cancel.is_set():
                raise downloader.DownloadCancelled()
            scheduler.dispatch(on_done)
        except downloader.DownloadCancelled:
            logger.info("Model download of {} cancelled", model)
            if on_cancelled is not None:
                scheduler.dispatch(on_cancelled)
        except downloader.DOWNLOAD_ERRORS as e:
            # The downloader notes the file that failed, which a disk error never names.
            if failed := getattr(e, "__notes__", None):
                logger.error(
                    "Model download of {} failed on {}: {}",
                    model,
                    failed[-1],
                    downloader.failure_reason(e),
                )
            else:
                logger.error(
                    "Model download of {} failed: {}",
                    model,
                    downloader.failure_reason(e),
                )
            scheduler.dispatch(on_error, _failure_message(model, e))
        except Exception as e:
            logger.exception("Model download of {} failed", model)
            scheduler.dispatch(on_error, _failure_message(model, e))

    thread = threading.Thread(target=_work, name="model-download", daemon=True)
    thread.start()
    return thread


def _failure_message(model: str, error: Exception) -> str:
    """Return what the user is told about a failed download: what to do next.

    The error itself is never shown; the log names it by `downloader.failure_reason`,
    or an unexpected one by its type and traceback.
    """
    name = model_label(model)
    if isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EDQUOT):
        return f"Couldn't download {name}. Free up disk space and try again."
    return f"Couldn't download {name}. Check your connection and try again."


# Compared by identity: two requesters with equal callbacks are still two, and
# leaving must remove the one that left.
@dataclass(eq=False)
class _Subscriber:
    device: str
    on_progress: Callable[[downloader.DownloadProgress], None] | None = None
    on_done: Callable[[], None] | None = None
    on_error: Callable[[str], None] | None = None


@dataclass
class _Transfer:
    """One running download of a model and everyone waiting on it.

    `progress` keeps the latest update so a late joiner can be caught up; `thread` is
    the worker.
    """

    model: str
    cancel: threading.Event = field(default_factory=threading.Event)
    subscribers: list[_Subscriber] = field(default_factory=list)
    progress: downloader.DownloadProgress | None = None
    thread: threading.Thread | None = None


class Subscription:
    """One requester's place on a download."""

    def __init__(self, coordinator: DownloadCoordinator, subscriber: _Subscriber):
        self._coordinator = coordinator
        self._subscriber = subscriber

    def cancel(self) -> None:
        """Leave the download; the requester hears nothing more from it.

        The transfer itself stops only when nobody else is waiting on it.
        """
        self._coordinator._leave(self._subscriber)


class DownloadCoordinator:
    """Single-flight downloads per model, owned by the app for its lifetime.

    Cancelling means leaving: the transfer stops only when its last requester leaves.
    A transfer is forgotten as soon as its worker ends, so a later request re-checks
    the disk rather than replaying an old result. A request that arrives while a
    cancelled worker unwinds starts afresh after it, so the dying worker's cleanup
    never deletes a live partial file. The first requester's device decides whether
    the NPU encoder is fetched.
    """

    def __init__(self, models_dir, scheduler, *, download=download_async):
        self._models_dir = models_dir
        self._scheduler = scheduler
        self._download = download
        self._transfers: dict[str, _Transfer] = {}

    def is_downloading(self, model: str) -> bool:
        """Whether a transfer is active, not mid-cancellation; main thread only."""
        transfer = self._transfers.get(model)
        return transfer is not None and not transfer.cancel.is_set()

    def start_or_join(
        self,
        model: str,
        device: str,
        *,
        on_progress: Callable[[downloader.DownloadProgress], None] | None = None,
        on_done: Callable[[], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> Subscription:
        """Request `model`: join the transfer already running for it, or start one.

        Call on the main thread.
        """
        sub = _Subscriber(device, on_progress, on_done, on_error)
        transfer = self._transfers.get(model)
        if transfer is None:
            self._start(model, [sub])
        else:
            transfer.subscribers.append(sub)
            if sub.on_progress and transfer.progress and not transfer.cancel.is_set():
                sub.on_progress(transfer.progress)  # catch the joiner up
        return Subscription(self, sub)

    def _start(self, model: str, subscribers: list[_Subscriber]) -> None:
        # Subscribers are in place before the worker starts: its first callback may
        # arrive at once.
        transfer = _Transfer(model, subscribers=subscribers)
        self._transfers[model] = transfer
        try:
            transfer.thread = self._download(
                self._models_dir,
                model,
                subscribers[0].device,
                self._scheduler,
                on_progress=lambda p: self._on_progress(transfer, p),
                on_done=lambda: self._end(transfer, "on_done"),
                on_error=lambda msg: self._end(transfer, "on_error", msg),
                on_cancelled=lambda: self._end(transfer, None),
                cancel=transfer.cancel,
            )
        except Exception as e:
            # Ends like a failed worker would, on a later turn of the main loop.
            logger.exception("Could not start the download of {}", model)
            self._scheduler.dispatch(
                self._end, transfer, "on_error", _failure_message(model, e)
            )

    def _leave(self, sub: _Subscriber) -> None:
        # Looked up, not remembered: a subscriber may have moved to a fresh transfer.
        transfer = next(
            (t for t in self._transfers.values() if sub in t.subscribers), None
        )
        if transfer is None:
            return  # already ended, or already left
        transfer.subscribers.remove(sub)
        if not transfer.subscribers:
            transfer.cancel.set()

    def _on_progress(self, transfer: _Transfer, p) -> None:
        transfer.progress = p
        # Walk a copy, since a callback may join this model again, and guard each so
        # one failing requester does not silence the others.
        for s in list(transfer.subscribers):
            if s.on_progress:
                guarded(s.on_progress)(p)

    def _end(self, transfer: _Transfer, callback: str | None, *args) -> None:
        """Forget the transfer whose worker ended and tell whoever is still waiting.

        After a cancel (`callback` None) the waiting are only requests that arrived
        while it unwound, and they get a fresh transfer.
        """
        if self._transfers.get(transfer.model) is transfer:
            del self._transfers[transfer.model]
        waiting, transfer.subscribers = transfer.subscribers, []
        if callback is None:
            if waiting:
                self._start(transfer.model, waiting)
            return
        for s in waiting:
            fn = getattr(s, callback)
            if fn:
                guarded(fn)(*args)
