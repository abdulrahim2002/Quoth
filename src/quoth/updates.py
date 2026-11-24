# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""The update check: is a newer Quoth release out, and how should this install update?

Checks the latest GitHub release tag against the running `__version__` so the app
can surface a desktop notification. It holds the fetch, the daily throttle and
once-per-version rules, the state file, and the install channel's update hint.
Privacy-first policy:

- Opt-in, default-off (`config.notify_updates`): no network until enabled.
- Never downloads or installs: it only tells you; the package manager or store
  does the update.
- Hits only the GitHub release API, no telemetry or PII, and fails silent on any
  error so an offline machine sees nothing.

`update_notifier.py` owns the threading, the app owns the notification, and this
module owns the decision.
"""

from __future__ import annotations

import http.client
import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from quoth import APP_ID, __version__, atomic_write, paths
from quoth.downloader import failure_reason
from quoth.environment import in_appimage, in_flatpak, in_snap

_REPO = "abullard1/Quoth"
_DEFAULT_API_URL = f"https://api.github.com/repos/{_REPO}/releases/latest"
# An endpoint that answers like the latest-release API, for a fork, a mirror or a test.
API_URL = os.environ.get("QUOTH_UPDATE_API_URL", _DEFAULT_API_URL)
RELEASES_URL = f"https://github.com/{_REPO}/releases/latest"

CHECK_INTERVAL_S = 86_400  # check the network at most once per day
_TIMEOUT_S = 4.0
_USER_AGENT = f"quoth/{__version__}"


def parse_version(tag: str) -> tuple[int, ...] | None:
    """Parse a release tag: `"v0.1.1"` or `"0.1.1"` -> `(0, 1, 1)`.

    None when any component is non-numeric (a nightly or odd tag), treated as "not
    newer" (fail safe). A pre-release suffix (`-rc1`, `+build`) is dropped first.
    """
    core = tag.strip().lstrip("vV").split("-")[0].split("+")[0]
    parts = core.split(".")
    if not parts or not all(p.isascii() and p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def _fetch_latest_tag(timeout: float = _TIMEOUT_S) -> str:
    """Fetch the latest release's `tag_name`; raise on any network, HTTP or JSON error.

    GitHub's "latest" excludes drafts and pre-releases by definition.
    """
    request = urllib.request.Request(
        API_URL,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    tag = data.get("tag_name") if isinstance(data, dict) else None
    if not isinstance(tag, str):
        raise ValueError("The release response has no tag")
    return tag


@dataclass(frozen=True)
class UpdateResult:
    """The outcome of one update check.

    `version` is the newer release string (e.g. `"0.2.0"`) or None when nothing is
    newer; `reached` is whether the server gave a definitive answer (False when
    offline, on an HTTP error or an unparseable tag). `reached` gates the throttle and
    the failure status, so a failed check never counts as "checked" or claims "up to
    date".
    """

    version: str | None
    reached: bool


def check_for_update(
    current: str = __version__, *, fetch: Callable[[], str] = _fetch_latest_tag
) -> UpdateResult:
    """Reach GitHub and compare its latest release with `current`.

    `reached` is False on any miss (offline, rate-limited, an unparseable tag);
    otherwise `version` is the newer release or None. `fetch` is injected for tests.
    """
    logger.debug("Update check: fetching {}", API_URL)
    try:
        latest_tag = fetch()
    except (OSError, http.client.HTTPException, ValueError) as e:
        # Offline, HTTP or an unusable response: an expected failure. Anything else is
        # a defect and reaches the check worker's exception boundary.
        logger.warning("Update check failed: {}", failure_reason(e))
        return UpdateResult(version=None, reached=False)
    latest = parse_version(latest_tag)
    here = parse_version(current)
    if latest is None or here is None:
        logger.debug(
            "Update check: unparseable version (latest={!r}, current={!r})",
            latest_tag,
            current,
        )
        return UpdateResult(version=None, reached=False)
    newer = latest_tag.lstrip("vV") if latest > here else None
    logger.debug(
        "Update check result: latest={!r} current={} newer={}",
        latest_tag,
        current,
        newer,
    )
    return UpdateResult(version=newer, reached=True)


def due_for_check(state: dict, now: float, interval: float = CHECK_INTERVAL_S) -> bool:
    """Whether enough time has passed to hit the network again.

    A never-checked state is always due; otherwise gate on `interval` since
    `last_check`.
    """
    last = state.get("last_check")
    return last is None or (now - last) >= interval


def register_result(state: dict, result: UpdateResult, now: float) -> bool:
    """Fold a check result into `state`, in place, and return whether to notify.

    A check that did not reach the server records nothing, so the next one retries. A
    reached check stamps `last_check` and returns True only for a version not yet
    announced, so the user is told once per release. The caller persists `state`.
    """
    if not result.reached:
        return False
    state["last_check"] = now
    version = result.version
    if version and version != state.get("last_notified"):
        state["last_notified"] = version
        return True
    return False


def default_path() -> Path:
    """Return the update-check state file in the state directory."""
    return paths.state_dir() / "update_check.json"


def load_state(path: Path) -> dict:
    """Read the throttle and notify state (`{last_check, last_notified}`).

    Returns `{}` if the file is absent or unreadable.
    """
    return atomic_write.read_state(path, "Update-check state") or {}


def save_state(path: Path, state: dict) -> None:
    """Persist the throttle/notify state; failure is non-fatal (logged)."""
    try:
        atomic_write.write_text(path, json.dumps(state))
    except OSError as e:
        logger.warning("Could not persist update-check state: {}", e)


# --- Install channel: how should this build be updated? ---
#
# Two-layer detection: self-identifying runtime markers first (authoritative),
# then a build-time stamp for formats identical at runtime (a deb, an AUR package,
# and a source install are all just "a binary on PATH"). The stamp is a generated
# `quoth/_channel.py` written by the native install recipes.


def _read_stamp() -> str | None:
    """Return the build-time channel stamp, or None.

    The stamp is generated into the installed package as `quoth/_channel.py`; it is
    absent in an unstaged development checkout.
    """
    try:
        from quoth._channel import CHANNEL  # generated at install time
    except ImportError:
        return None
    return CHANNEL


def current_channel() -> str:
    """Return how this build is installed.

    That is `flatpak`, `snap`, `appimage`, `aur`, `deb`, `source` or `unknown`. Runtime
    markers win (they self-identify), then the build stamp, then `unknown`.
    `QUOTH_CHANNEL` overrides everything (forks, tests, a channel baked into the
    installed .desktop's Exec).
    """
    override = os.environ.get("QUOTH_CHANNEL")
    if override:
        return override.strip().lower()
    if in_flatpak():
        return "flatpak"
    if in_snap():
        return "snap"
    if in_appimage():
        return "appimage"
    return _read_stamp() or "unknown"


# Package identity does not prove a repository was added. Keep the download route
# available alongside package-manager guidance.
_HINTS = {
    "flatpak": (
        "Download the new .flatpak from the release page, or, for repository installs, "
        f"run: flatpak update {APP_ID}"
    ),
    "snap": "Run: snap refresh quoth",
    "appimage": "Replace the AppImage with the new release.",
    "aur": (
        "If you installed from the AUR, use your AUR helper. "
        "Otherwise, rebuild the package from the new release tag."
    ),
    "deb": (
        "If you added Quoth's APT repository, use your system updater. "
        "Otherwise, download the new .deb from the release page."
    ),
    "source": "Pull the latest and rebuild, or download the release.",
    "unknown": "Update via your package manager, or download the release.",
}


def update_hint(channel: str | None = None) -> str:
    """Return channel-appropriate "how to update" guidance.

    It serves the notification body and the in-app hub. `channel` defaults to the
    detected one; an unrecognised channel falls back to generic guidance.
    """
    if channel is None:
        channel = current_channel()
    logger.debug("Update hint: channel={}", channel)
    return _HINTS.get(channel, _HINTS["unknown"])


def status_line(
    available: str | None, *, checking: bool, checked: bool, failed: bool
) -> str:
    """Return the in-app status subtitle for a given update state.

    A known update outranks a later failed re-check, and failure shows only when
    nothing newer is known.
    """
    if checking:
        return "Checking…"
    if available:
        return f"{available} available. {update_hint()}"
    if failed:
        return "Couldn't check for updates"
    return "Up to date" if checked else "Not checked yet"
