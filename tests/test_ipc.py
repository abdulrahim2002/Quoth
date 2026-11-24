# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.ipc`, Quoth's D-Bus surface.

A toggle succeeds when a running instance has the action, even one too busy to answer in
time; any other failure means the caller falls through to a normal start. The script
interface reports the state, takes start, stop and cancel, and says when no instance
runs or why one refused.
"""

import sys
import threading
from unittest.mock import MagicMock, patch

import pytest
from gi.repository import Gio, GLib

from quoth import ipc
from tests.support import wait_until


def _bus(side_effect=None):
    bus = MagicMock()
    bus.call_sync.side_effect = side_effect
    return bus


def test_activates_the_action_on_the_running_instance():
    bus = _bus()
    with patch("quoth.ipc.Gio.bus_get_sync", return_value=bus) as get_bus:
        assert ipc.activate_action("toggle") is True
    assert get_bus.call_args.args[0] == Gio.BusType.SESSION
    args = bus.call_sync.call_args.args
    assert args[0] == ipc.APP_ID
    assert args[1] == ipc.OBJECT_PATH
    assert args[2:4] == ("org.freedesktop.Application", "ActivateAction")
    # Another signature unpacks alike, but the instance refuses it.
    assert args[4].get_type_string() == "(sava{sv})"
    assert args[4].unpack() == ("toggle", [], {})
    assert args[6] == Gio.DBusCallFlags.NONE


def test_no_running_instance_falls_through():
    bus = _bus(GLib.Error("no owner"))
    with patch("quoth.ipc.Gio.bus_get_sync", return_value=bus):
        assert ipc.activate_action("toggle") is False


def test_no_session_bus_falls_through():
    with patch("quoth.ipc.Gio.bus_get_sync", side_effect=GLib.Error("no bus")):
        assert ipc.activate_action("toggle") is False


def test_a_busy_instance_still_has_the_action():
    timed_out = GLib.Error.new_literal(
        Gio.io_error_quark(), "Timeout was reached", Gio.IOErrorEnum.TIMED_OUT
    )
    bus = _bus(timed_out)
    with patch("quoth.ipc.Gio.bus_get_sync", return_value=bus):
        assert ipc.activate_action("toggle") is True


def test_a_wedged_instance_holds_the_toggle_for_half_a_second_at_most():
    bus = _bus()
    with patch("quoth.ipc.Gio.bus_get_sync", return_value=bus):
        ipc.activate_action("toggle")
    assert bus.call_sync.call_args.args[7] == 500


def _dbus_error(code, name, message):
    return GLib.Error.new_literal(
        Gio.dbus_error_quark(), f"GDBus.Error:{name}: {message}", code
    )


@pytest.mark.parametrize(
    "error",
    [
        _dbus_error(
            Gio.DBusError.NAME_HAS_NO_OWNER,
            "org.freedesktop.DBus.Error.NameHasNoOwner",
            "Name does not exist",
        ),
        _dbus_error(
            Gio.DBusError.SERVICE_UNKNOWN,
            "org.freedesktop.DBus.Error.ServiceUnknown",
            "The name was not provided",
        ),
    ],
)
def test_a_script_command_without_an_instance_says_so(error):
    bus = _bus(error)
    with patch("quoth.ipc.Gio.bus_get_sync", return_value=bus):
        with pytest.raises(ipc.NotRunning):
            ipc.control_dictation("Start")
        with pytest.raises(ipc.NotRunning):
            ipc.dictation_state()


def test_a_script_command_without_a_session_bus_finds_no_instance():
    with patch("quoth.ipc.Gio.bus_get_sync", side_effect=GLib.Error("no bus")):
        with pytest.raises(ipc.NoSessionBus):
            ipc.dictation_state()


def test_a_script_command_never_starts_quoth_and_waits_the_default_timeout():
    bus = _bus()
    with patch("quoth.ipc.Gio.bus_get_sync", return_value=bus) as get_bus:
        ipc.control_dictation("Stop")
    assert get_bus.call_args.args[0] == Gio.BusType.SESSION
    args = bus.call_sync.call_args.args
    # Scripts name these; SUPPORT.md freezes them through 0.x.
    assert args[:4] == (
        "dev.quoth.Quoth",
        "/dev/quoth/Quoth",
        "dev.quoth.Dictation1",
        "Stop",
    )
    assert args[6] == Gio.DBusCallFlags.NO_AUTO_START
    assert args[7] == -1


@pytest.fixture
def session(private_bus):
    """A service connection owning Quoth's name on a private bus, and a client."""
    flags = (
        Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT
        | Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION
    )
    service = Gio.DBusConnection.new_for_address_sync(private_bus, flags, None, None)
    client = Gio.DBusConnection.new_for_address_sync(private_bus, flags, None, None)
    service.call_sync(
        "org.freedesktop.DBus",
        "/org/freedesktop/DBus",
        "org.freedesktop.DBus",
        "RequestName",
        GLib.Variant("(su)", (ipc.APP_ID, 4)),
        None,
        Gio.DBusCallFlags.NONE,
        -1,
        None,
    )
    yield service, client
    for connection in (service, client):
        connection.close_sync(None)


def _served(client, function):
    """Run a blocking client call while this thread's main loop serves the instance."""
    outcome = {}

    def run():
        with patch("quoth.ipc.Gio.bus_get_sync", return_value=client):
            try:
                outcome["value"] = function()
            except Exception as error:
                outcome["error"] = error

    thread = threading.Thread(target=run)
    thread.start()
    wait_until(lambda: not thread.is_alive())
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def test_scripts_reach_the_instance_over_the_bus(session):
    service_bus, client = session
    state = ["idle"]
    calls = []
    reasons = {"Start": "Finish setting up Quoth first"}

    def method(name):
        def call():
            calls.append(name)
            return reasons.get(name)

        return call

    service = ipc.DictationService(
        service_bus,
        state=lambda: state[0],
        start=method("Start"),
        stop=method("Stop"),
        cancel=method("Cancel"),
    )

    assert _served(client, ipc.dictation_state) == "idle"
    with pytest.raises(ipc.Refused, match=r"^Finish setting up Quoth first$"):
        _served(client, lambda: ipc.control_dictation("Start"))
    _served(client, lambda: ipc.control_dictation("Stop"))
    _served(client, lambda: ipc.control_dictation("Cancel"))
    assert calls == ["Start", "Stop", "Cancel"]

    changes = []
    client.signal_subscribe(
        None,
        "org.freedesktop.DBus.Properties",
        "PropertiesChanged",
        ipc.OBJECT_PATH,
        None,
        Gio.DBusSignalFlags.NONE,
        lambda *args: changes.append((args[5].get_type_string(), args[5].unpack())),
    )
    state[0] = "recording"
    # A round trip on the client connection makes the bus apply the match rule first.
    client.call_sync(
        "org.freedesktop.DBus",
        "/org/freedesktop/DBus",
        "org.freedesktop.DBus",
        "GetId",
        None,
        None,
        Gio.DBusCallFlags.NONE,
        -1,
        None,
    )
    service.publish("recording")
    wait_until(lambda: changes, timeout=5)
    # Watchers drop a PropertiesChanged of any other signature.
    assert changes == [("(sa{sv}as)", (ipc.INTERFACE, {"State": "recording"}, []))]
    assert _served(client, ipc.dictation_state) == "recording"

    service.close()
    with pytest.raises(ipc.Refused):
        _served(client, ipc.dictation_state)


def test_a_handler_that_raises_answers_at_once_and_reaches_the_crash_hook(
    session, monkeypatch
):
    service_bus, client = session
    hooked = []
    monkeypatch.setattr(sys, "excepthook", lambda kind, *_: hooked.append(kind))

    def start():
        raise AttributeError("a defect in the start path")

    service = ipc.DictationService(
        service_bus,
        state=lambda: "idle",
        start=start,
        stop=lambda: None,
        cancel=lambda: None,
    )
    with pytest.raises(ipc.Refused, match=r"^Quoth hit an internal error; its log"):
        _served(client, lambda: ipc.control_dictation("Start"))
    assert hooked == [AttributeError]
    service.close()


def test_the_served_interface_carries_no_text(session):
    # Any process in the session can watch the bus, so no transcript may cross it:
    # three methods without arguments, and the state alone.
    service_bus, client = session
    service = ipc.DictationService(
        service_bus,
        state=lambda: "idle",
        start=lambda: None,
        stop=lambda: None,
        cancel=lambda: None,
    )

    def introspect():
        reply = client.call_sync(
            ipc.APP_ID,
            ipc.OBJECT_PATH,
            "org.freedesktop.DBus.Introspectable",
            "Introspect",
            None,
            GLib.VariantType("(s)"),
            Gio.DBusCallFlags.NONE,
            -1,
            None,
        )
        return reply.unpack()[0]

    served = Gio.DBusNodeInfo.new_for_xml(_served(client, introspect))
    interface = served.lookup_interface(ipc.INTERFACE)
    assert sorted(
        (m.name, len(m.in_args), len(m.out_args)) for m in interface.methods
    ) == [("Cancel", 0, 0), ("Start", 0, 0), ("Stop", 0, 0)]
    assert [(p.name, p.signature) for p in interface.properties] == [("State", "s")]
    assert list(interface.signals) == []
    service.close()
