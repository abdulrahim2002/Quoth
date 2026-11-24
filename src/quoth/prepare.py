# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""The one-time OpenVINO encoder compile, as a gate on recording.

Switching a model onto the NPU compiles its encoder once, off the main thread.
Meanwhile a dictation is refused rather than stalled behind it, with the user told
why, and the finished cell is recorded warm. The gate stays up until the compile
lands, even after a switch away, and belongs to the pair being prepared, so an older
completion never lowers a newer prepare's gate.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from loguru import logger

from quoth.recording import ContextLoadOutcome, RecordingController
from quoth.resolution import mark_compiled


class PrepareGate:
    """The recording gate held while one (model, device) pair's encoder compiles.

    `notify` shows the user the preparing notice and `withdraw` takes it down.
    """

    def __init__(
        self,
        controller: RecordingController,
        *,
        notify: Callable[[], None],
        withdraw: Callable[[], None],
    ) -> None:
        self._controller = controller
        self._notify = notify
        self._withdraw = withdraw
        self._pair: tuple[str, str] | None = None
        self._started = 0.0

    @property
    def active(self) -> bool:
        return self._pair is not None

    def raise_for(self, model: str, device: str) -> None:
        """Gate recording for this pair and show the notice."""
        logger.info("Preparing {} on {}", model, device)
        self._pair = (model, device)
        self._started = time.monotonic()
        self._notify()

    def prepare(self, model: str, device: str) -> None:
        """Raise the gate and compile now, rather than in the first dictation.

        `finish` lifts the gate when the context lands.
        """
        self.raise_for(model, device)
        self._controller.ensure_context_async(model, device, self.finish)

    def refuse_start(self) -> bool:
        """Whether a recording start must be refused.

        A refusal shows the notice again, so the keypress is not a silent no-op.
        """
        if self._pair is None:
            return False
        logger.debug("Recording start ignored: preparing NPU acceleration")
        self._notify()
        return True

    def finish(
        self,
        model: str,
        device: str,
        outcome: ContextLoadOutcome,
    ) -> None:
        """Lift the gate once the prepared pair's context is ready.

        A no-op unless this pair is the one being prepared. The cell is recorded warm
        only when the NPU offload engaged, since whisper.cpp silently falls back to the
        CPU on a failed init.
        """
        if self._pair is None or (model, device) != self._pair:
            return
        self._pair = None
        self._withdraw()
        offload = "active" if outcome.openvino_active else "inactive"
        result = f"NPU offload {offload}" if outcome.loaded else "not loaded"
        elapsed = time.monotonic() - self._started
        logger.info("Prepared {} on {} in {:.1f}s: {}", model, device, elapsed, result)
        if outcome.loaded and outcome.openvino_active:
            mark_compiled(model, device)
