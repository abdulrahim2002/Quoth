# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.motion`: how the reduced-motion sources combine and change live."""

from types import SimpleNamespace

import pytest


class _Value:
    def __init__(self, *values):
        self._values = values

    def unpack(self):
        return self._values


class _Settings:
    def __init__(self, *, animations: bool, native: int | None):
        self.animations = animations
        self.native = native
        self.callbacks = {}

    def find_property(self, name):
        if name == "gtk-interface-reduced-motion" and self.native is not None:
            return object()
        return None

    def get_property(self, name):
        if name == "gtk-enable-animations":
            return self.animations
        if name == "gtk-interface-reduced-motion":
            return self.native
        raise AssertionError(name)

    def connect(self, signal, callback):
        self.callbacks[signal] = callback

    def emit(self, signal):
        self.callbacks[signal](self, None)


class _Proxy:
    def __init__(self, portal: int):
        self.portal = portal
        self.callback = None

    def call_sync(self, *_args):
        return _Value(self.portal)

    def connect(self, signal, callback):
        assert signal == "g-signal"
        self.callback = callback

    def change(self, value: int):
        self.callback(
            self,
            None,
            "SettingChanged",
            _Value("org.freedesktop.appearance", "reduced-motion", value),
        )


def _watcher(monkeypatch, *, portal=0, animations=True, native=None, on_change=None):
    from quoth import motion

    settings = _Settings(animations=animations, native=native)
    proxy = _Proxy(portal)
    monkeypatch.setattr(motion.Gtk.Settings, "get_default", lambda: settings)
    monkeypatch.setattr(
        motion.Gio.DBusProxy,
        "new_for_bus_sync",
        lambda *_args: proxy,
    )
    changes = []
    watcher = motion.ReducedMotionWatcher(on_change or changes.append)
    return SimpleNamespace(
        watcher=watcher, settings=settings, proxy=proxy, changes=changes
    )


def test_a_portal_slower_than_the_read_still_reports_changes(monkeypatch):
    from quoth import motion

    class _SlowProxy(_Proxy):
        def call_sync(self, *_args):
            raise motion.GLib.Error("Timeout was reached")

    proxy = _SlowProxy(0)
    monkeypatch.setattr(
        motion.Gtk.Settings,
        "get_default",
        lambda: _Settings(animations=True, native=None),
    )
    monkeypatch.setattr(motion.Gio.DBusProxy, "new_for_bus_sync", lambda *_a: proxy)
    changes = []
    watcher = motion.ReducedMotionWatcher(changes.append)
    assert not watcher.reduced
    proxy.change(1)
    assert watcher.reduced and changes == [True]


def test_startup_waits_on_the_portal_only_through_the_bounded_call(monkeypatch):
    from quoth import motion

    calls = []
    reads = []

    class _ReadProxy(_Proxy):
        def call_sync(self, *args):
            reads.append(args)
            return super().call_sync(*args)

    monkeypatch.setattr(
        motion.Gtk.Settings,
        "get_default",
        lambda: _Settings(animations=True, native=None),
    )
    monkeypatch.setattr(
        motion.Gio.DBusProxy,
        "new_for_bus_sync",
        lambda *args: calls.append(args) or _ReadProxy(0),
    )
    motion.ReducedMotionWatcher(lambda _reduced: None)

    flags = calls[0][1]
    assert flags & motion.Gio.DBusProxyFlags.DO_NOT_LOAD_PROPERTIES
    assert flags & motion.Gio.DBusProxyFlags.DO_NOT_AUTO_START_AT_CONSTRUCTION
    timeout_ms = reads[0][3]
    assert timeout_ms == 500


@pytest.mark.parametrize(
    ("portal", "animations", "native", "expected"),
    [
        (0, True, None, False),
        (0, False, None, True),
        (1, True, None, True),
        (0, True, 0, False),
        (0, True, 1, True),
        (0, False, 0, True),
        (1, True, 0, True),
    ],
)
def test_reduction_is_any_supported_source(
    monkeypatch, portal, animations, native, expected
):
    state = _watcher(monkeypatch, portal=portal, animations=animations, native=native)

    assert state.watcher.reduced is expected
    assert state.changes == []


def test_other_portal_settings_leave_reduction_unchanged(monkeypatch):
    state = _watcher(monkeypatch)

    # color-scheme 1 is "prefer dark", the same value as a reduction request.
    state.proxy.callback(
        state.proxy,
        None,
        "SettingChanged",
        _Value("org.freedesktop.appearance", "color-scheme", 1),
    )

    assert state.watcher.reduced is False
    assert state.changes == []


def test_live_sources_only_report_effective_changes(monkeypatch):
    state = _watcher(monkeypatch, native=0)

    state.proxy.change(0)
    state.settings.animations = False
    state.settings.emit("notify::gtk-enable-animations")
    state.proxy.change(1)
    state.settings.animations = True
    state.settings.emit("notify::gtk-enable-animations")
    state.proxy.change(0)
    state.settings.native = 1
    state.settings.emit("notify::gtk-interface-reduced-motion")
    state.settings.emit("notify::gtk-interface-reduced-motion")
    state.settings.native = 0
    state.settings.emit("notify::gtk-interface-reduced-motion")

    assert state.changes == [True, False, True, False]


@pytest.mark.parametrize("source", ["gtk", "portal"])
def test_failed_propagation_is_logged_and_same_signal_retries(
    monkeypatch, caplog, callback_raises, source
):
    attempts = []
    delivered = []

    def on_change(reduced):
        attempts.append(reduced)
        if len(attempts) == 1:
            raise RuntimeError("surface failed")
        delivered.append(reduced)

    caplog.set_level("ERROR")
    state = _watcher(monkeypatch, native=0, on_change=on_change)

    def trigger():
        if source == "gtk":
            state.settings.animations = False
            state.settings.emit("notify::gtk-enable-animations")
        else:
            state.proxy.change(1)

    trigger()
    assert state.watcher.reduced is False
    trigger()

    assert attempts == [True, True]
    assert delivered == [True]
    assert state.watcher.reduced is True
    [record] = caplog.records
    assert record.exc_info[0] is RuntimeError
