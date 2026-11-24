# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Shared test collaborators.

Fakes live here as plain classes rather than fixtures because most are built with
arguments per test (`FakeModelManager(ggml=False)`), which a fixture would have to
express as a factory-fixture: more indirection for the same thing.
"""

from tests.support.colour import contrast
from tests.support.fakes import (
    DeferredScheduler,
    FakeAudio,
    FakeBenchmark,
    FakeClipboard,
    FakeModelManager,
    FakePortalShortcuts,
    FakeScheduler,
    FakeWhisper,
    FakeYdotoold,
    FinishedDialog,
    fake_engine,
    wl_copy_offers,
)
from tests.support.gtk import (
    accessible_mismatch,
    appearance,
    button,
    descendants,
    legibility,
    settle,
    shown_texts,
    wait_until,
)
from tests.support.patches import compile_commands, whisper_source_tree
from tests.support.threads import join_thread

__all__ = [
    "DeferredScheduler",
    "FakeAudio",
    "FakeBenchmark",
    "FakeClipboard",
    "FakeModelManager",
    "FakePortalShortcuts",
    "FakeScheduler",
    "FakeWhisper",
    "FakeYdotoold",
    "FinishedDialog",
    "accessible_mismatch",
    "appearance",
    "button",
    "compile_commands",
    "contrast",
    "descendants",
    "fake_engine",
    "join_thread",
    "legibility",
    "settle",
    "shown_texts",
    "wait_until",
    "whisper_source_tree",
    "wl_copy_offers",
]
