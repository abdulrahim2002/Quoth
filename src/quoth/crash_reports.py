# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Opt-in crash reports, sent to Sentry's EU region (Frankfurt).

Off by default. When the user turns them on, an uncaught exception (main loop, GTK
callback or thread) or an exception logged at ERROR becomes one Sentry event, once per
run for a failure that repeats at the same place. A native crash's faulthandler stack
(see `log.install_crash_hooks`) is sent at the next launch with reports on, or as they
are turned on. Nothing else leaves the machine: no sessions, breadcrumbs, local
variables, hostname, exception values, log messages or home path, and never audio or
text. This is the only module that touches `sentry_sdk`.

The SDK is imported only once reports are on, since the import alone costs about 160 ms
of startup.
"""

from __future__ import annotations

import functools
import getpass
import json
import os
import platform
import re
import sys
from pathlib import Path

from loguru import logger

from quoth import __version__, log, paths
from quoth.environment import in_flatpak

# The EU project's public ingest key; account credentials never ship in the app.
DSN = "https://e045fd76a74ed8a3e52b0296b8e5dc73@o4512222832427008.ingest.de.sentry.io/4512222986043472"

# The consent wording, shared by onboarding and Settings so it reads the same.
SWITCH_TITLE = "Send error reports"
SWITCH_SUBTITLE = (
    "Send technical details about errors and crashes to Sentry (EU) to help fix Quoth"
)
SWITCH_HELP = "Reports never hold audio or dictated text"
NOTICE_TITLE = "Privacy notice"
NOTICE_SUBTITLE = "What reports contain and who receives them"
NOTICE_HELP = "Opens the privacy notice in the browser"

_PACKAGE_DIR = Path(__file__).resolve().parent
_VENDOR_DIR = _PACKAGE_DIR / "_vendor"
_IDENTITY_PATH = _PACKAGE_DIR / "lib" / "build-identity.json"
_enabled = False
_reported: set[tuple] = set()
_device: str | None = None
_openvino_active: bool | None = None


def set_enabled(enabled: bool) -> None:
    """Start or stop reporting after the stored opt-in preference is applied."""
    global _enabled
    if enabled == _enabled:
        return
    if not enabled:
        import sentry_sdk

        old_client = sentry_sdk.get_client()
        sentry_sdk.get_global_scope().set_client(None)
        _enabled = False
        old_client.close(timeout=0)
        logger.info("Crash reports off")
        return

    try:
        _add_vendor_path()
        import sentry_sdk

        sentry_sdk.init(**_options())
    except Exception:
        # Reporting is optional: a broken SDK must not stop the app from starting.
        logger.exception("Crash reports could not start; nothing is sent")
        return
    _enabled = True
    logger.info("Crash reports on")


def note_device(device: str) -> None:
    """Record the requested backend (CPU, GPU, NPU), tagged on reports for triage."""
    global _device
    _device = device


def note_context(openvino_active: bool | None) -> None:
    """Record whether the current context's NPU offload initialised; None clears it."""
    global _openvino_active
    _openvino_active = openvino_active


def report_native_crash() -> None:
    """Queue the previous native stack, or preserve it without consent."""
    path = log.previous_crash_log
    if not _enabled or not path.exists():
        return
    import sentry_sdk
    from sentry_sdk.attachments import Attachment

    # Taken off disk before queueing, so a stack is never sent twice.
    try:
        stack = _scrub(path.read_text(errors="replace"))
        path.unlink()
    except OSError as e:
        logger.warning("Previous native crash stack kept locally ({})", e)
        return
    sentry_sdk.capture_event(
        {"exception": {"values": [{"type": "NativeCrash"}]}, "level": "fatal"},
        hint={"attachments": [Attachment(bytes=stack.encode(), filename="crash.log")]},
    )
    logger.info("Queued the previous session's native crash report")


def _options() -> dict:
    from sentry_sdk.integrations.atexit import AtexitIntegration
    from sentry_sdk.integrations.dedupe import DedupeIntegration
    from sentry_sdk.integrations.excepthook import ExcepthookIntegration
    from sentry_sdk.integrations.loguru import LoguruIntegration
    from sentry_sdk.integrations.threading import ThreadingIntegration
    from sentry_sdk.transport import HttpTransport

    class Transport(HttpTransport):
        def on_dropped_event(self, reason: str) -> None:
            logger.debug("Error report not sent ({})", reason)

    release, environment = _identity()
    return {
        "dsn": DSN,
        "release": release,
        "environment": environment,
        "send_default_pii": False,
        "include_local_variables": False,
        "include_source_context": False,
        "server_name": "",
        "max_breadcrumbs": 0,
        "auto_session_tracking": False,
        "send_client_reports": False,
        "spotlight": False,
        "ignore_errors": [KeyboardInterrupt],
        # An allowlist prevents a future SDK default from collecting more data.
        "default_integrations": False,
        "integrations": [
            ExcepthookIntegration(),
            ThreadingIntegration(),
            DedupeIntegration(),
            AtexitIntegration(),
            LoguruIntegration(
                level=None,
                event_level=40,
                sentry_logs_level=None,
                capture_sentry_logs=False,
                # A callable keeps loguru from appending the traceback, values and all.
                event_format=lambda _record: "{message}",
            ),
        ],
        "before_send": _prepare,
        "transport": Transport,
    }


def _prepare(event: dict, hint: dict) -> dict | None:
    """Strip an event's private values, keeping its exception structure for triage.

    Return None to drop an event with no exception, or a repeat of a failure already
    reported this run.
    """
    values = event.get("exception", {}).get("values")
    if not values:
        logger.debug("Error report dropped: not an exception")
        return None
    # A failure that repeats (every draw, every audio buffer) is reported once per run.
    signature = _signature(values)
    if signature in _reported:
        logger.debug("Error report dropped: a repeat this run")
        return None
    _reported.add(signature)
    logger.debug("Error report queued")
    if not (len(values) == 1 and values[0].get("type") == "NativeCrash"):
        hint.pop("attachments", None)
    for exception in values:
        exception.pop("value", None)
        for frame in exception.get("stacktrace", {}).get("frames", []):
            for key in ("vars", "pre_context", "context_line", "post_context"):
                frame.pop(key, None)
    for key in ("message", "logentry", "extra"):
        event.pop(key, None)
    contexts = event.get("contexts", {})
    contexts.pop("trace", None)
    if not contexts:
        event.pop("contexts", None)
    event.setdefault("tags", {}).update(_tags())
    if _device:
        event["tags"]["device"] = _device
    if _openvino_active is not None:
        event["tags"]["openvino_active"] = str(_openvino_active).lower()
    return _scrub(event)


def _signature(values: list) -> tuple:
    """Return the exception types and the frames each one passed through."""
    parts = []
    for exception in values:
        parts.append(exception.get("type"))
        for frame in exception.get("stacktrace", {}).get("frames", []):
            parts.append(
                (frame.get("filename"), frame.get("function"), frame.get("lineno"))
            )
    return tuple(parts)


@functools.cache
def _add_vendor_path() -> None:
    """Expose the Debian package's colocated SDK without a custom loader."""
    if _VENDOR_DIR.is_dir() and str(_VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(_VENDOR_DIR))


@functools.cache
def _identity() -> tuple[str, str]:
    fallback = (f"quoth@{__version__}+dev", "development")
    if not _IDENTITY_PATH.exists():
        return fallback
    try:
        identity = json.loads(_IDENTITY_PATH.read_text())
        release = identity["release"]
        environment = identity["environment"]
        sha_release = re.fullmatch(
            rf"quoth@{re.escape(__version__)}\+g[0-9a-f]{{40}}", release
        )
        valid_release = (
            (environment == "production" and release == f"quoth@{__version__}")
            or (environment == "candidate" and sha_release is not None)
            or (
                environment == "development"
                and (release == fallback[0] or sha_release is not None)
            )
        )
        if set(identity) != {"release", "environment"} or not valid_release:
            raise ValueError("invalid build identity")
        return release, environment
    except (OSError, TypeError, ValueError, KeyError):
        logger.warning("Ignoring malformed build identity at {}", _IDENTITY_PATH)
        return fallback


@functools.cache
def _tags() -> dict[str, str]:
    try:
        distro = platform.freedesktop_os_release().get("PRETTY_NAME", "unknown")
    except OSError:
        distro = "unknown"
    return {
        "distro": distro,
        "desktop": os.environ.get("XDG_CURRENT_DESKTOP", "unknown"),
        "install": "flatpak" if in_flatpak() else "system",
    }


@functools.cache
def _user_directory() -> re.Pattern | None:
    """Match the user's name as a whole word, as in a checkout outside the home."""
    try:
        user = getpass.getuser()
    except (OSError, KeyError):
        return None
    return re.compile(rf"(?<!\w){re.escape(user)}(?!\w)") if user else None


def _scrub(value):
    """Replace the home directory with `~` and the user's name in a path, everywhere.

    A user named like a system directory (`lib`, `src`) also hides it in other paths.
    """
    if isinstance(value, str):
        user = _user_directory()
        value = paths.tilde(value)
        return user.sub("<user>", value) if user and "/" in value else value
    if isinstance(value, dict):
        return {key: _scrub(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    return value
