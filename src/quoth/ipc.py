# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Quoth's D-Bus surface: reaching the running instance without starting a second app.

`quoth --toggle` is what a compositor keybinding runs on every keypress. Forwarding
the command line through GApplication would first import GTK, GStreamer and numpy, a
third of a second per press, so this module imports only Gio and activates the action
through `org.freedesktop.Application` directly.

Scripts get `INTERFACE` beside it: a `State` property and `Start`, `Stop` and
`Cancel` methods. No transcript ever crosses the bus, since any process in the
session can watch it.
"""

from __future__ import annotations

from collections.abc import Callable

import gi

gi.require_version("Gio", "2.0")
from gi.repository import Gio, GLib  # noqa: E402

from quoth import APP_ID, paths  # noqa: E402

OBJECT_PATH = "/" + APP_ID.replace(".", "/")
INTERFACE = "dev.quoth.Dictation1"
# Long enough for a busy main loop, short enough that a wedged instance does
# not hold the keypress.
_TIMEOUT_MS = 500

_INTERFACE_INFO = Gio.DBusNodeInfo.new_for_xml(
    f"""<node><interface name="{INTERFACE}">
  <method name="Start"/>
  <method name="Stop"/>
  <method name="Cancel"/>
  <property name="State" type="s" access="read"/>
</interface></node>"""
).interfaces[0]
_NOT_RUNNING = (Gio.DBusError.NAME_HAS_NO_OWNER, Gio.DBusError.SERVICE_UNKNOWN)


class NotRunning(Exception):
    """No Quoth instance is running in this session."""


class NoSessionBus(NotRunning):
    """There is no session bus, so no instance can be reached."""


class Refused(Exception):
    """The running instance refused the call or did not answer; the message says why."""


def activate_action(name: str) -> bool:
    """Activate the running instance's `name` action and return whether it received it.

    False when there is no running instance (or no bus), so the caller can start one
    instead. An instance too busy to answer in time still counts: it runs the call once
    its main loop is free.
    """
    try:
        bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        bus.call_sync(
            APP_ID,
            OBJECT_PATH,
            "org.freedesktop.Application",
            "ActivateAction",
            GLib.Variant("(sava{sv})", (name, [], {})),
            None,
            Gio.DBusCallFlags.NONE,
            _TIMEOUT_MS,
            None,
        )
    except GLib.Error as error:
        # A timed-out call stays queued, so starting another instance would forward the
        # action a second time.
        return error.matches(Gio.io_error_quark(), Gio.IOErrorEnum.TIMED_OUT)
    return True


def dictation_state() -> str:
    """Return the running instance's state: `idle`, `recording` or `transcribing`."""
    reply = _call(
        "org.freedesktop.DBus.Properties",
        "Get",
        GLib.Variant("(ss)", (INTERFACE, "State")),
    )
    return reply.unpack()[0]


def control_dictation(method: str) -> None:
    """Call `Start`, `Stop` or `Cancel` on the running instance; it never starts one."""
    _call(INTERFACE, method, None)


def _call(interface: str, method: str, parameters: GLib.Variant | None):
    """Call the running instance and wait up to D-Bus's default 25 s for its answer.

    Raises `NotRunning` when no instance owns the name (`NoSessionBus` when there is
    no bus to own it on), and `Refused` with the reason for any other failure, such
    as the instance refusing or not answering.
    """
    try:
        bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    except GLib.Error as error:
        raise NoSessionBus from error
    try:
        return bus.call_sync(
            APP_ID,
            OBJECT_PATH,
            interface,
            method,
            parameters,
            None,
            Gio.DBusCallFlags.NO_AUTO_START,
            -1,
            None,
        )
    except GLib.Error as error:
        if any(error.matches(Gio.dbus_error_quark(), code) for code in _NOT_RUNNING):
            raise NotRunning from error
        remote = Gio.DBusError.get_remote_error(error)
        reason = error.message.removeprefix(f"GDBus.Error:{remote}: ")
        raise Refused(reason) from error


class DictationService:
    """The running instance's side of `INTERFACE`.

    `start`, `stop` and `cancel` return None, or the reason a call failed; `state`
    reads the current state. Calls arrive on the main loop.
    """

    def __init__(
        self,
        connection: Gio.DBusConnection,
        *,
        state: Callable[[], str],
        start: Callable[[], str | None],
        stop: Callable[[], str | None],
        cancel: Callable[[], str | None],
    ):
        self._connection = connection
        self._state = state
        self._methods = {"Start": start, "Stop": stop, "Cancel": cancel}
        self._id = connection.register_object(
            OBJECT_PATH, _INTERFACE_INFO, self._on_call, self._on_get, None
        )

    def publish(self, state: str) -> None:
        """Tell watchers the state changed, as `PropertiesChanged` on `State`."""
        self._connection.emit_signal(
            None,
            OBJECT_PATH,
            "org.freedesktop.DBus.Properties",
            "PropertiesChanged",
            GLib.Variant(
                "(sa{sv}as)", (INTERFACE, {"State": GLib.Variant("s", state)}, [])
            ),
        )

    def close(self) -> None:
        self._connection.unregister_object(self._id)

    def _on_call(self, _conn, _sender, _path, _iface, method, _params, invocation):
        try:
            reason = self._methods[method]()
        except Exception:
            # Answer at once, or the caller waits out the 25 s timeout; the raise still
            # reaches the crash hook, which logs the defect.
            invocation.return_dbus_error(
                f"{INTERFACE}.Error.Internal",
                f"Quoth hit an internal error; its log in {paths.state_dir()} says why",
            )
            raise
        if reason is None:
            invocation.return_value(None)
        else:
            invocation.return_dbus_error("org.freedesktop.DBus.Error.Failed", reason)

    def _on_get(self, _conn, _sender, _path, _iface, _name):
        return GLib.Variant("s", self._state())
