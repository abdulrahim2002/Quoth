# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.portal_shortcuts`, the app-owned GlobalShortcuts portal session."""

from types import SimpleNamespace

import pytest
from gi.repository import Gio, GLib

from quoth.portal_shortcuts import SHORTCUT_ID, PortalShortcuts


@pytest.fixture(autouse=True)
def _outside_plasma_5(monkeypatch):
    monkeypatch.delenv("KDE_SESSION_VERSION", raising=False)


def _finish(result):
    if isinstance(result, BaseException):
        raise result
    return result


class FakeSignals:
    def __init__(self):
        self._handlers = {}
        self.callbacks = {}
        self._next_handler = 1

    def connect(self, name, callback):
        handler = self._next_handler
        self._next_handler += 1
        self._handlers[handler] = (name, callback)
        self.callbacks.setdefault(name, []).append(callback)
        return handler

    def disconnect(self, handler):
        self._handlers.pop(handler)

    def emit(self, name, *args):
        for registered_name, callback in list(self._handlers.values()):
            if registered_name == name:
                callback(self, *args)


class FakeNativeSession(FakeSignals):
    pass


class FakeSession(FakeSignals):
    def __init__(self):
        super().__init__()
        self.native = FakeNativeSession()
        self.bind_callbacks = []
        self.configure_callbacks = []
        self.bind_calls = []
        self.configure_calls = []
        self.closed = 0

    def get_session(self):
        return self.native

    def bind_shortcuts(self, shortcuts, parent_window, _cancellable, callback, _data):
        self.bind_calls.append((shortcuts, parent_window))
        self.bind_callbacks.append(callback)

    def bind_shortcuts_finish(self, result):
        return _finish(result)

    def resolve_bind(self, result, index=-1):
        self.bind_callbacks[index](self, result)

    def configure_shortcuts(
        self, parent_window, activation_token, _cancellable, callback, _data
    ):
        self.configure_calls.append((parent_window, activation_token))
        self.configure_callbacks.append(callback)

    def configure_shortcuts_finish(self, result):
        return _finish(result)

    def resolve_configure(self, result=True, index=-1):
        self.configure_callbacks[index](self, result)

    def close(self):
        self.closed += 1
        self.native.emit("closed")


class FakePortal:
    def __init__(self):
        self.create_callback = None
        self.create_calls = 0

    def create_global_shortcuts_session(self, _cancellable, callback, _data):
        self.create_calls += 1
        self.create_callback = callback

    def create_global_shortcuts_session_finish(self, result):
        return _finish(result)

    def resolve_create(self, result):
        self.create_callback(self, result)


class FakeBus:
    def __init__(self):
        self.call_args = None
        self.probe_callback = None

    def call(self, *args):
        self.call_args = args
        self.probe_callback = args[-1]

    def call_finish(self, result):
        return _finish(result)

    def resolve_probe(self, result):
        self.probe_callback(self, result)


class FakeBusAccess:
    def __init__(self):
        self.callback = None

    def get(self, _bus_type, _cancellable, callback):
        self.callback = callback

    def finish(self, result):
        return _finish(result)

    def resolve(self, result):
        self.callback(None, result)


class Assignment:
    def __init__(self, shortcut_id=SHORTCUT_ID, trigger="F13"):
        self.shortcut_id = shortcut_id
        self.trigger = trigger

    def get_shortcut_id(self):
        return self.shortcut_id

    def get_trigger_description(self):
        return self.trigger


class Harness:
    def __init__(self, *, xdp=True):
        self.access = FakeBusAccess()
        self.bus = FakeBus()
        self.portal = FakePortal()
        self.session = FakeSession()
        self.created_shortcuts = []
        self.events = []

        def new_shortcut(*args):
            self.created_shortcuts.append(args)
            return args

        fake_xdp = None
        if xdp:
            fake_xdp = SimpleNamespace(GlobalShortcut=SimpleNamespace(new=new_shortcut))
        self.owner = PortalShortcuts(
            lambda: self.events.append("pressed"),
            lambda: self.events.append("released"),
            lambda: self.events.append(f"lost:{self.owner.status}"),
            bus_get=self.access.get,
            bus_get_finish=self.access.finish,
            xdp=fake_xdp,
            portal_factory=lambda: self.portal,
        )

    def probe(self, version=2, *, restore=False):
        self.owner.start(restore=restore)
        self.access.resolve(self.bus)
        self.bus.resolve_probe(GLib.Variant("(v)", (GLib.Variant("u", version),)))

    def ready(self, assignments=(), *, version=2):
        assignments = list(assignments)
        self.probe(version, restore=bool(assignments))
        self.portal.resolve_create(self.session)
        if assignments:
            self.session.resolve_bind(assignments)


def test_start_restores_only_when_prior_user_intent_requests_it():
    harness = Harness()
    changes = []
    harness.owner.subscribe(lambda: changes.append(harness.owner.status))

    harness.ready([Assignment(trigger="Super+Space")])

    assert harness.owner.version == 2
    assert harness.owner.status == "ready"
    assert harness.owner.binding == "Super+Space"
    assert len(harness.session.bind_calls) == 1
    assert harness.session.configure_calls == []
    assert changes[-1] == "ready"

    fresh = Harness()
    fresh.ready()
    assert fresh.owner.status == "unbound"
    assert fresh.session.bind_calls == []


def test_only_the_owned_shortcut_dispatches_press_and_release():
    harness = Harness()
    harness.ready([Assignment()])

    harness.session.emit("activated", "someone-else", 1, None)
    harness.session.emit("activated", SHORTCUT_ID, 2, None)
    harness.session.emit("activated", SHORTCUT_ID, 3, None)
    harness.session.emit("deactivated", "someone-else", 4, None)
    harness.session.emit("deactivated", SHORTCUT_ID, 5, None)

    assert harness.events == ["pressed", "released"]


@pytest.mark.parametrize(
    ("error", "message", "status"),
    [
        ("org.freedesktop.DBus.Error.UnknownInterface", "absent", "unavailable"),
        (
            "org.freedesktop.DBus.Error.InvalidArgs",
            "No such interface “org.freedesktop.portal.GlobalShortcuts”",
            "unavailable",
        ),
        ("org.freedesktop.DBus.Error.NoReply", "timed out", "error"),
    ],
    ids=["missing", "cosmic", "broken"],
)
def test_missing_interface_is_unavailable_but_other_probe_error_is_error(
    error, message, status
):
    harness = Harness()
    harness.owner.start()
    harness.access.resolve(harness.bus)
    harness.bus.resolve_probe(Gio.DBusError.new_for_dbus_error(error, message))
    assert harness.owner.status == status


def test_plasma_5_is_unavailable_although_it_has_the_portal(monkeypatch):
    monkeypatch.setenv("KDE_SESSION_VERSION", "5")
    harness = Harness()
    harness.probe(version=1, restore=True)

    assert harness.owner.status == "unavailable"
    assert harness.portal.create_calls == 0


def test_present_portal_without_libportal_is_a_distinct_error():
    harness = Harness(xdp=False)
    harness.probe()

    assert harness.owner.version == 2
    assert harness.owner.status == "error"
    assert harness.portal.create_callback is None

    assert harness.owner.configure() is True
    harness.access.resolve(harness.bus)
    harness.bus.resolve_probe(GLib.Variant("(v)", (GLib.Variant("u", 2),)))
    assert harness.owner.status == "error"
    assert harness.owner.configure() is True


@pytest.mark.parametrize(
    ("response", "status"),
    [(Gio.IOErrorEnum.CANCELLED, "denied"), (Gio.IOErrorEnum.FAILED, "error")],
)
def test_configure_binds_once_and_explicit_retry_follows_bind_failure(response, status):
    harness = Harness()
    harness.ready()

    assert harness.owner.status == "unbound"
    assert harness.owner.configure() is True
    assert harness.owner.configure() is False
    assert harness.owner.status == "requesting"
    assert harness.created_shortcuts == [(SHORTCUT_ID, "Start or stop dictation", "")]
    assert harness.session.bind_calls[0][1] == ""

    error = GLib.Error.new_literal(Gio.io_error_quark(), "request failed", response)
    harness.session.resolve_bind(error)
    assert harness.owner.status == status
    assert len(harness.session.bind_calls) == 1
    assert harness.portal.create_calls == 1

    assert harness.owner.configure() is True
    replacement = FakeSession()
    harness.access.resolve(harness.bus)
    harness.bus.resolve_probe(GLib.Variant("(v)", (GLib.Variant("u", 2),)))
    harness.portal.resolve_create(replacement)
    assert len(replacement.bind_calls) == 1
    replacement.resolve_bind([Assignment(trigger="Ctrl+Space")])
    assert harness.owner.status == "ready"
    assert harness.owner.binding == "Ctrl+Space"


def test_version_two_reconfigures_but_version_one_guides_to_desktop_settings():
    modern = Harness()
    modern.ready([Assignment()], version=2)
    assert modern.owner.configure() is True
    assert modern.session.configure_calls == [("", "")]
    assert len(modern.session.bind_calls) == 1
    modern.session.resolve_configure()
    assert modern.owner.status == "ready"

    legacy = Harness()
    legacy.ready([Assignment()], version=1)
    assert legacy.owner.configure() is False
    assert legacy.session.configure_calls == []
    assert len(legacy.session.bind_calls) == 1
    assert legacy.owner.status == "ready"


def test_a_cancelled_reconfiguration_keeps_the_binding():
    harness = Harness()
    harness.ready([Assignment(trigger="F13")], version=2)
    assert harness.owner.configure() is True

    cancelled = GLib.Error.new_literal(
        Gio.io_error_quark(), "dismissed", Gio.IOErrorEnum.CANCELLED
    )
    harness.session.resolve_configure(cancelled)

    assert harness.owner.status == "ready"
    assert harness.owner.binding == "F13"
    assert harness.owner.configure() is True


@pytest.mark.parametrize(
    "assignments",
    # GNOME leaves a shortcut given no key out of a successful reply; KDE returns it
    # with an empty trigger.
    [[], [Assignment(trigger="")]],
    ids=["gnome", "kde"],
)
def test_a_shortcut_accepted_without_a_key_is_configured_through_the_desktop(
    assignments,
):
    harness = Harness()
    harness.ready()
    harness.owner.configure()
    harness.session.resolve_bind(assignments)
    assert harness.owner.status == "needs-key"
    assert harness.owner.binding == ""

    assert harness.owner.configure() is True
    assert harness.session.configure_calls == [("", "")]
    harness.session.resolve_configure()
    assert harness.owner.status == "needs-key"

    harness.session.emit("shortcuts-changed", [Assignment(trigger="Pause")])
    assert harness.owner.status == "ready"
    assert harness.owner.binding == "Pause"
    assert harness.events == []

    legacy = Harness()
    legacy.ready([Assignment(trigger="")], version=1)
    assert legacy.owner.status == "needs-key"
    assert legacy.owner.configure() is False


def test_a_press_proves_a_key_the_desktop_does_not_name():
    harness = Harness()
    harness.ready([Assignment(trigger="")])
    assert harness.owner.status == "needs-key"
    seen = []
    harness.owner.subscribe(lambda: seen.append(harness.owner.status))

    harness.session.emit("activated", SHORTCUT_ID, 1, None)

    assert seen == ["ready"]
    assert harness.owner.binding == ""
    assert harness.events == ["pressed"]


def test_losing_the_binding_while_pressed_delivers_lost_once(caplog):
    harness = Harness()
    harness.ready([Assignment()])
    harness.session.emit("activated", SHORTCUT_ID, 1, None)

    harness.session.emit("shortcuts-changed", [])
    harness.session.emit("shortcuts-changed", [])

    assert harness.owner.status == "unbound"
    assert harness.owner.binding == ""
    assert harness.events == ["pressed", "lost:unbound"]
    # The desktop dropping the shortcut is in the log, once.
    assert (
        caplog.messages.count("GlobalShortcuts shortcut no longer ready: unbound") == 1
    )
    assert caplog.messages[-2:] == [
        "GlobalShortcuts: status=unbound binding='' version=2",
        "GlobalShortcuts shortcut no longer ready: unbound",
    ]


def test_changed_binding_retires_a_held_press_after_publishing_ready():
    harness = Harness()
    harness.ready([Assignment()])
    harness.session.emit("activated", SHORTCUT_ID, 1, None)

    harness.session.emit("shortcuts-changed", [Assignment(trigger="Ctrl+Space")])

    assert harness.owner.status == "ready"
    assert harness.owner.binding == "Ctrl+Space"
    assert harness.events == ["pressed", "lost:ready"]


def test_external_session_close_cancels_a_held_press_and_disconnects_signals():
    harness = Harness()
    harness.ready([Assignment()])
    harness.session.emit("activated", SHORTCUT_ID, 1, None)
    release = harness.session.callbacks["deactivated"][0]

    harness.session.native.emit("closed")
    release(harness.session, SHORTCUT_ID, 2, None)

    assert harness.owner.status == "closed"
    assert harness.owner.binding == ""
    assert harness.events == ["pressed", "lost:closed"]
    assert harness.session._handlers == {}
    assert harness.session.native._handlers == {}


def test_external_session_close_reports_loss_even_while_idle():
    harness = Harness()
    harness.ready([Assignment()])

    harness.session.native.emit("closed")

    assert harness.owner.status == "closed"
    assert harness.events == ["lost:closed"]


def test_explicit_configure_restarts_after_a_session_failure():
    harness = Harness()
    harness.probe()
    harness.portal.resolve_create(
        Gio.DBusError.new_for_dbus_error(
            "org.freedesktop.DBus.Error.NoReply", "timed out"
        )
    )
    assert harness.owner.status == "error"

    replacement = FakeSession()
    assert harness.owner.configure() is True
    harness.access.resolve(harness.bus)
    harness.bus.resolve_probe(GLib.Variant("(v)", (GLib.Variant("u", 2),)))
    harness.portal.resolve_create(replacement)

    assert harness.owner.status == "requesting"
    assert len(replacement.bind_calls) == 1


def test_close_discards_late_create_and_signal_callbacks():
    harness = Harness()
    harness.probe()
    harness.owner.close()

    harness.portal.resolve_create(harness.session)

    assert harness.session.closed == 1
    assert harness.owner.status == "closed"
    assert harness.events == []


def test_close_discards_a_late_bind_and_previously_captured_signal():
    harness = Harness()
    harness.probe(restore=True)
    harness.portal.resolve_create(harness.session)
    activated = harness.session.callbacks["activated"][0]

    harness.owner.close()
    harness.session.resolve_bind([Assignment()])
    activated(harness.session, SHORTCUT_ID, 1, None)

    assert harness.owner.status == "closed"
    assert harness.owner.binding == ""
    assert harness.events == []


def test_unsubscribe_uses_subscription_identity_and_is_idempotent():
    harness = Harness()
    calls = []

    first = harness.owner.subscribe(lambda: calls.append("first"))
    second = harness.owner.subscribe(lambda: calls.append("second"))
    first()
    first()
    harness.ready()

    assert set(calls) == {"second"}
    before = len(calls)
    second()
    harness.owner.close()
    assert len(calls) == before


def test_a_failing_subscriber_does_not_stop_session_creation(callback_raises):
    harness = Harness()
    statuses = []

    def broken():
        raise RuntimeError("view gone")

    harness.owner.subscribe(broken)
    harness.owner.subscribe(lambda: statuses.append(harness.owner.status))
    harness.ready()
    undisturbed = Harness()
    undisturbed.ready()

    assert harness.owner.status == undisturbed.owner.status != "checking"
    assert statuses[-1] == undisturbed.owner.status
    assert callback_raises == ["Callback broken raised"] * 2


def test_close_retires_subscribers_before_publishing_closed():
    harness = Harness()
    harness.ready()
    calls = []
    harness.owner.subscribe(lambda: calls.append(harness.owner.status))

    harness.owner.close()

    assert harness.owner.status == "closed"
    assert calls == []
