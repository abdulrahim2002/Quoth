# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""The app-owned GlobalShortcuts portal session for hotkeys in the Flatpak.

Through libportal, Quoth asks the desktop for one shortcut (`SHORTCUT_ID`) whose press
and release arrive as session signals. Each asynchronous request is tagged with a
generation, so a reply that arrives after a close or restart is discarded.
"""

import os
from collections.abc import Callable

import gi
from loguru import logger

gi.require_version("Gio", "2.0")
from gi.repository import Gio, GLib  # noqa: E402

try:
    gi.require_version("Xdp", "1.0")
    from gi.repository import Xdp as _Xdp
except (ImportError, ValueError):
    _Xdp = None

from quoth.scheduler import guarded  # noqa: E402

_PORTAL = "org.freedesktop.portal.Desktop"
_PORTAL_PATH = "/org/freedesktop/portal/desktop"
_SHORTCUTS_IFACE = "org.freedesktop.portal.GlobalShortcuts"
_PROPERTIES_IFACE = "org.freedesktop.DBus.Properties"
_UNKNOWN_INTERFACE = "org.freedesktop.DBus.Error.UnknownInterface"
_INVALID_ARGS = "org.freedesktop.DBus.Error.InvalidArgs"
_SERVICE_UNKNOWN = "org.freedesktop.DBus.Error.ServiceUnknown"
_PORTAL_TIMEOUT_MS = 5_000

SHORTCUT_ID = "toggle-dictation"
SHORTCUT_DESCRIPTION = "Start or stop dictation"


class PortalShortcuts:
    """One libportal GlobalShortcuts session and its shortcut signals.

    `on_pressed` and `on_released` report the shortcut's press and release. `on_lost`
    reports a shortcut the desktop removed or rebound, or a session the desktop closed
    (`close()` reports nothing); a held press gets no release after it.
    """

    def __init__(
        self,
        on_pressed: Callable[[], None],
        on_released: Callable[[], None],
        on_lost: Callable[[], None],
        *,
        bus_get: Callable = Gio.bus_get,
        bus_get_finish: Callable = Gio.bus_get_finish,
        xdp=_Xdp,
        portal_factory: Callable | None = None,
    ):
        self._on_pressed = on_pressed
        self._on_released = on_released
        self._on_lost = on_lost
        self._bus_get = bus_get
        self._bus_get_finish = bus_get_finish
        self._xdp = xdp
        self._portal_factory = portal_factory

        self._status = "checking"
        self._binding = ""
        self._version = 0
        self._started = False
        self._closed = False
        self._requesting = False
        self._bind_after_create = False
        self._bind_attempted = False
        self._has_binding = False
        self._pressed = False
        self._generation = 0
        self._cancellable = Gio.Cancellable()
        self._portal = None
        self._session = None
        self._native_session = None
        self._signal_handlers: list[tuple[object, int]] = []
        self._subscribers: dict[int, Callable[[], None]] = {}
        self._next_subscriber = 1

    @property
    def status(self) -> str:
        """The session state as a string the setup views match on.

        "checking", "requesting" (a portal dialog may be open), "unavailable" (no
        GlobalShortcuts portal, or one that cannot register a shortcut), "error",
        "unbound", "denied" (the user declined), "needs-key" (bound, but the desktop
        names no key), "ready" or "closed".
        """
        return self._status

    @property
    def binding(self) -> str:
        """The desktop's description of the bound key; empty when it names none."""
        return self._binding

    @property
    def version(self) -> int:
        """The portal's GlobalShortcuts version, 0 until probed.

        Version 2 and later can reconfigure an existing binding.
        """
        return self._version

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Call `callback` after each state change; return an idempotent unsubscribe."""
        subscriber = self._next_subscriber
        self._next_subscriber += 1
        self._subscribers[subscriber] = callback

        def unsubscribe() -> None:
            self._subscribers.pop(subscriber, None)

        return unsubscribe

    def start(self, *, restore: bool = False) -> None:
        """Probe the portal and create a session, binding the shortcut when `restore`.

        Pass `restore` only for a shortcut that was enabled before. Runs once, and never
        after `close`.
        """
        if self._started or self._closed:
            return
        self._started = True
        self._bind_after_create = restore
        self._begin_probe()

    def _begin_probe(self) -> None:
        generation = self._generation
        try:
            self._bus_get(
                Gio.BusType.SESSION,
                self._cancellable,
                lambda source, result, *_: self._bus_ready(generation, source, result),
            )
        except GLib.Error as error:
            self._fail("connect to the session bus", error)

    def configure(self) -> bool:
        """Start one explicit bind or reconfiguration request; return whether it began.

        False means no portal UI was opened. In particular, a version 1 portal cannot
        reconfigure an existing binding, so callers can direct the user to the desktop
        shortcut settings instead.
        """
        if self._closed or self._requesting:
            return False
        if self._session is None:
            if self._status not in {"closed", "error"}:
                return False
            return self._restart_for_configure()
        if not self._has_binding and self._bind_attempted:
            return self._restart_for_configure()
        if self._has_binding and self._version < 2:
            return False

        self._requesting = True
        self._set_state(status="requesting")
        generation = self._generation
        if self._has_binding:
            self._session.configure_shortcuts(
                "",
                "",
                self._cancellable,
                lambda session, result, *_: self._configured(
                    generation, session, result
                ),
                None,
            )
        else:
            self._bind(generation)
        return True

    def _restart_for_configure(self) -> bool:
        self._generation += 1
        self._requesting = True
        self._bind_after_create = True
        self._bind_attempted = False
        self._release_session(close=True)
        self._portal = None
        self._cancellable = Gio.Cancellable()
        self._set_state(status="requesting", binding="")
        self._begin_probe()
        return True

    def close(self) -> None:
        """End the session for good; subscribers are dropped first and hear no more."""
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        self._requesting = False
        self._bind_after_create = False
        self._bind_attempted = False
        self._pressed = False
        self._cancellable.cancel()
        self._release_session(close=True)
        self._portal = None
        self._has_binding = False
        self._subscribers.clear()
        self._set_state(status="closed", binding="")

    def _bus_ready(self, generation: int, _source, result) -> None:
        try:
            bus = self._bus_get_finish(result)
        except GLib.Error as error:
            if not self._stale(generation, error):
                self._fail("connect to the session bus", error)
            return
        if self._stale(generation):
            return
        bus.call(
            _PORTAL,
            _PORTAL_PATH,
            _PROPERTIES_IFACE,
            "Get",
            GLib.Variant("(ss)", (_SHORTCUTS_IFACE, "version")),
            GLib.VariantType.new("(v)"),
            Gio.DBusCallFlags.NONE,
            _PORTAL_TIMEOUT_MS,
            self._cancellable,
            lambda source, probe_result, *_: self._probe_ready(
                generation, source, probe_result
            ),
        )

    def _probe_ready(self, generation: int, bus, result) -> None:
        try:
            (version,) = bus.call_finish(result).unpack()
        except GLib.Error as error:
            if self._stale(generation, error):
                return
            if self._portal_missing(error):
                logger.debug("GlobalShortcuts portal unavailable")
                self._requesting = False
                self._bind_after_create = False
                self._set_state(status="unavailable")
            else:
                self._fail("probe the GlobalShortcuts portal", error)
            return
        if self._stale(generation):
            return
        # Plasma 5 registers a shortcut only from a session option that libportal never
        # sends, so its bind opens System Settings with nothing in it to set.
        if not version or os.environ.get("KDE_SESSION_VERSION") == "5":
            logger.debug(
                "GlobalShortcuts portal version {} cannot register a shortcut", version
            )
            self._requesting = False
            self._bind_after_create = False
            self._set_state(status="unavailable")
            return
        if self._xdp is None:
            logger.warning("GlobalShortcuts portal found, but libportal is unavailable")
            self._requesting = False
            self._bind_after_create = False
            self._set_state(version=int(version), status="error")
            return

        try:
            factory = self._portal_factory or self._xdp.Portal.initable_new
            self._portal = factory()
            self._set_state(version=int(version))
            self._portal.create_global_shortcuts_session(
                self._cancellable,
                lambda portal, create_result, *_: self._session_created(
                    generation, portal, create_result
                ),
                None,
            )
        except GLib.Error as error:
            self._fail("create a libportal client", error)

    def _session_created(self, generation: int, portal, result) -> None:
        try:
            session = portal.create_global_shortcuts_session_finish(result)
        except GLib.Error as error:
            if not self._stale(generation, error):
                self._fail("create a GlobalShortcuts session", error)
            return
        if self._stale(generation):
            session.close()
            return

        self._session = session
        self._native_session = session.get_session()
        self._bind_attempted = False
        self._connect(session, "activated", self._activated)
        self._connect(session, "deactivated", self._deactivated)
        self._connect(session, "shortcuts-changed", self._shortcuts_changed)
        self._connect(self._native_session, "closed", self._session_closed)
        if self._bind_after_create:
            self._bind_after_create = False
            self._requesting = True
            self._bind(generation)
        else:
            self._requesting = False
            self._set_state(status="unbound", binding="")

    def _bind(self, generation: int) -> None:
        self._bind_attempted = True
        shortcut = self._xdp.GlobalShortcut.new(SHORTCUT_ID, SHORTCUT_DESCRIPTION, "")
        self._session.bind_shortcuts(
            [shortcut],
            "",
            self._cancellable,
            lambda session, result, *_: self._bound(generation, session, result),
            None,
        )

    def _bound(self, generation: int, session, result) -> None:
        try:
            assignments = session.bind_shortcuts_finish(result)
        except GLib.Error as error:
            if self._stale(generation, error):
                return
            self._requesting = False
            if self._cancelled(error):
                logger.debug("GlobalShortcuts binding request declined")
                self._set_state(status="denied", binding="")
            else:
                self._fail("bind the GlobalShortcuts shortcut", error)
            return
        if self._stale(generation):
            return
        self._requesting = False
        if self._find_assignment(assignments) is None:
            # A successful reply without our shortcut: GNOME leaves out one the user
            # gave no key. The desktop still holds it, so it is configured there.
            self._has_binding = True
            self._set_state(status="needs-key", binding="")
        else:
            self._apply_assignments(assignments)

    def _configured(self, generation: int, session, result) -> None:
        try:
            session.configure_shortcuts_finish(result)
        except GLib.Error as error:
            if self._stale(generation, error):
                return
            self._requesting = False
            if self._cancelled(error):
                logger.debug("GlobalShortcuts configuration was cancelled")
                self._set_state(status=self._bound_status())
            else:
                self._fail("configure the GlobalShortcuts shortcut", error)
            return
        if not self._stale(generation):
            self._requesting = False
            self._set_state(status=self._bound_status())

    def _activated(self, session, shortcut_id, _timestamp, _options) -> None:
        if (
            not self._closed
            and session is self._session
            and shortcut_id == SHORTCUT_ID
            and self._has_binding
            and not self._pressed
        ):
            self._pressed = True
            # A delivered press proves a key, even one the desktop does not name.
            if self._status == "needs-key":
                self._set_state(status="ready")
            logger.debug("Portal shortcut down")
            self._on_pressed()

    def _deactivated(self, session, shortcut_id, _timestamp, _options) -> None:
        if (
            not self._closed
            and session is self._session
            and shortcut_id == SHORTCUT_ID
            and self._pressed
        ):
            self._pressed = False
            logger.debug("Portal shortcut up")
            self._on_released()

    def _shortcuts_changed(self, session, assignments) -> None:
        if not self._closed and session is self._session:
            self._apply_assignments(assignments)

    def _session_closed(self, native_session) -> None:
        if self._closed or native_session is not self._native_session:
            return
        self._generation += 1
        self._requesting = False
        self._bind_after_create = False
        self._bind_attempted = False
        self._cancellable.cancel()
        self._release_session(close=False)
        self._has_binding = False
        logger.warning("GlobalShortcuts session closed unexpectedly")
        self._set_state(status="closed", binding="")
        self._report_lost()

    def _apply_assignments(self, assignments) -> None:
        assignment = self._find_assignment(assignments)
        had_binding = self._has_binding
        self._has_binding = assignment is not None
        if assignment is None:
            self._set_state(status="unbound", binding="")
            if had_binding:
                self._report_lost()
        else:
            # An empty trigger names no key: KDE's for a shortcut the user gave none,
            # Hyprland's for every shortcut.
            binding = assignment.get_trigger_description() or ""
            changed = had_binding and bool(self._binding) and binding != self._binding
            self._set_state(status="ready" if binding else "needs-key", binding=binding)
            if changed:
                self._report_lost()

    def _bound_status(self) -> str:
        if not self._has_binding:
            return "unbound"
        return "ready" if self._binding else "needs-key"

    @staticmethod
    def _find_assignment(assignments):
        return next(
            (
                assignment
                for assignment in assignments or ()
                if assignment.get_shortcut_id() == SHORTCUT_ID
            ),
            None,
        )

    def _connect(self, source, signal: str, callback: Callable) -> None:
        self._signal_handlers.append((source, source.connect(signal, callback)))

    def _release_session(self, *, close: bool) -> None:
        for source, handler in self._signal_handlers:
            source.disconnect(handler)
        self._signal_handlers.clear()
        session = self._session
        self._session = None
        self._native_session = None
        if close and session is not None:
            session.close()

    def _report_lost(self) -> None:
        self._pressed = False
        self._on_lost()

    def _fail(self, action: str, error: GLib.Error) -> None:
        if self._cancelled(error) and self._closed:
            return
        self._requesting = False
        self._bind_after_create = False
        logger.warning("Failed to {}: {}", action, error.message)
        self._set_state(status="error")

    def _stale(self, generation: int, error: GLib.Error | None = None) -> bool:
        stale = self._closed or generation != self._generation
        if stale and error is not None and not self._cancelled(error):
            logger.debug("Discarded stale GlobalShortcuts callback: {}", error.message)
        return stale

    @staticmethod
    def _cancelled(error: GLib.Error) -> bool:
        return error.matches(Gio.io_error_quark(), Gio.IOErrorEnum.CANCELLED)

    @staticmethod
    def _portal_missing(error: GLib.Error) -> bool:
        remote = Gio.DBusError.get_remote_error(error)
        return remote in {_UNKNOWN_INTERFACE, _SERVICE_UNKNOWN} or (
            remote == _INVALID_ARGS
            and "No such interface" in error.message
            and _SHORTCUTS_IFACE in error.message
        )

    def _set_state(
        self,
        *,
        status: str | None = None,
        binding: str | None = None,
        version: int | None = None,
    ) -> None:
        changed = False
        was = self._status
        for name, value in (
            ("_status", status),
            ("_binding", binding),
            ("_version", version),
        ):
            if value is not None and getattr(self, name) != value:
                setattr(self, name, value)
                changed = True
        if changed:
            logger.debug(
                "GlobalShortcuts: status={} binding='{}' version={}",
                self._status,
                self._binding,
                self._version,
            )
            # A deliberate close at shutdown is not a lost shortcut.
            if was == "ready" and self._status != "ready" and not self._closed:
                logger.info(
                    "GlobalShortcuts shortcut no longer ready: {}", self._status
                )
            for callback in list(self._subscribers.values()):
                # A failing view must not stop the session from being created.
                guarded(callback)()
