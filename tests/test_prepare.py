# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.prepare`, the NPU-compile gate.

It refuses recording until the compile it raised for lands, and tracks that (model,
device) pair so a stale completion cannot lie.
"""

from unittest.mock import MagicMock, create_autospec

import pytest

from quoth.calibration import is_warm
from quoth.hardware import fingerprint
from quoth.prepare import PrepareGate
from quoth.recording import ContextLoadOutcome, RecordingController


def _gate():
    controller = create_autospec(RecordingController, instance=True)
    notify, withdraw = MagicMock(), MagicMock()
    return (
        PrepareGate(controller, notify=notify, withdraw=withdraw),
        controller,
        notify,
        withdraw,
    )


def test_prepare_raises_the_gate_and_loads_async():
    gate, controller, notify, _ = _gate()
    gate.prepare("small", "NPU")
    assert gate.active
    notify.assert_called_once()
    controller.ensure_context_async.assert_called_once_with("small", "NPU", gate.finish)
    gate.finish("small", "NPU", ContextLoadOutcome.LOADED)
    assert not gate.active


def test_refused_start_resurfaces_the_notice():
    gate, _, notify, _ = _gate()
    assert gate.refuse_start() is False
    gate.raise_for("small", "NPU")
    assert gate.refuse_start() is True
    assert notify.call_count == 2


# A CPU fallback or a failed load lifts the gate all the same; a gate left up would
# refuse every dictation.
@pytest.mark.parametrize(
    ("outcome", "marked"),
    [
        (ContextLoadOutcome.OPENVINO, True),
        (ContextLoadOutcome.LOADED, False),
        (ContextLoadOutcome.FAILED, False),
    ],
    ids=["offload", "cpu fallback", "failed"],
)
def test_finish_marks_compiled_only_when_offload_engaged(outcome, marked):
    gate, _, _, withdraw = _gate()
    gate.raise_for("small", "NPU")
    gate.finish("small", "NPU", outcome)
    assert is_warm("small", "NPU", fingerprint()) is marked
    assert not gate.active
    withdraw.assert_called_once()


def test_stale_finish_is_a_noop():
    gate, _, _, withdraw = _gate()
    gate.finish("small", "NPU", ContextLoadOutcome.OPENVINO)
    assert not is_warm("small", "NPU", fingerprint())
    withdraw.assert_not_called()


# The newer pair differs from the stale ("small", "NPU") by its device or by its model.
@pytest.mark.parametrize("newer", [("small", "GPU"), ("base", "NPU")])
def test_finish_for_superseded_pair_leaves_the_newer_gate(newer):
    gate, _, _, _ = _gate()
    gate.raise_for(*newer)
    gate.finish("small", "NPU", ContextLoadOutcome.OPENVINO)
    assert not is_warm("small", "NPU", fingerprint())
    assert gate.active
    gate.finish(*newer, ContextLoadOutcome.LOADED)
    assert not gate.active


def test_a_prepare_logs_its_start_and_how_it_ended(caplog):
    caplog.set_level("INFO")
    gate, _, _, _ = _gate()
    gate.raise_for("small", "NPU")
    gate.finish("small", "NPU", ContextLoadOutcome.LOADED)
    assert caplog.messages[0] == "Preparing small on NPU"
    assert caplog.messages[1].endswith("NPU offload inactive")
