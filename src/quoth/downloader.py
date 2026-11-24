# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Model downloading and integrity verification: the network layer for models.

Every asset (ggml weights, plus the OpenVINO encoder on NPU) comes from the
project's GitHub release, each file SHA-256-verified against the release
`manifest.json`. On the official origin verification fails closed: a file with no
published hash is refused. The base URL is overridable through an environment
variable (forks, mirrors, proxies, offline, tests); overriding it relaxes integrity
to lenient (warn and accept), as does `QUOTH_ALLOW_UNVERIFIED=1`.

GTK-free, unit-testable without a display.
"""

import hashlib
import http.client
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path

from loguru import logger

from quoth import atomic_write
from quoth.models import (
    encoder_bin_name,
    encoder_xml_name,
    ggml_name,
    needs_openvino_encoder,
    partial_name,
)

# Every asset lives on one release under one tag. Overriding the base URL also
# relaxes integrity; see _require_verification.
_DEFAULT_RELEASE_BASE_URL = "https://github.com/abullard1/Quoth/releases/download"
RELEASE_BASE_URL = os.environ.get("QUOTH_RELEASE_BASE_URL", _DEFAULT_RELEASE_BASE_URL)
ASSETS_TAG = os.environ.get("QUOTH_ASSETS_TAG", "v0.1.0")
MANIFEST_FILENAME = "manifest.json"
_CHUNK = 64 * 1024
# Socket idle timeout: it bounds the wait between reads, not the whole transfer, so
# only a stalled connection fails, never a slow one.
_STALL_TIMEOUT_S = 30
# A retried asset resumes its partial `.tmp` via a Range request; the backoff grows
# per attempt.
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_S = 2
# Emit UI progress at most this often (seconds); also the window over which the
# live speed is measured.
_PROGRESS_INTERVAL_S = 0.25


@dataclass(frozen=True)
class DownloadProgress:
    """Live download progress for the UI.

    `fraction` spans the whole job (all files, 0..1); `downloaded`, `total` and
    `bytes_per_sec` describe the file in flight, in bytes (`total` is 0 when the server
    sends no Content-Length).
    """

    fraction: float
    downloaded: int
    total: int
    bytes_per_sec: float


class IntegrityError(Exception):
    """A download that could not be verified.

    Its SHA-256 did not match, it has no published hash, or the release manifest is
    malformed.
    """


class IncompleteDownloadError(Exception):
    """A transfer that ended short of its Content-Length: a dropped connection.

    It is transient, so `_download_with_retries` retries it.
    """


# How a download fails without a defect: network, HTTP, disk or integrity.
DOWNLOAD_ERRORS = (
    OSError,
    http.client.HTTPException,
    IncompleteDownloadError,
    IntegrityError,
)


def failure_reason(error: Exception) -> str:
    """Name a fetch failure for the log without urllib's text for it.

    That text can echo a malformed proxy setting, password and all, as a host or port.
    """
    name = type(error).__name__
    if isinstance(error, urllib.error.HTTPError):
        return f"{name}: {error.code}"
    if isinstance(error, urllib.error.URLError):
        reason = error.reason
        return f"{name}: {reason}" if isinstance(reason, OSError) else name
    if isinstance(error, OSError) and error.errno is not None:
        return f"{name}: {error}"  # an errno and maybe a path, never a URL
    if isinstance(error, (IncompleteDownloadError, IntegrityError)):
        return f"{name}: {error}"  # written here, from names, sizes and hashes
    return name


class DownloadCancelled(Exception):
    """A download the user cancelled through the caller's cancel Event.

    It aborts immediately without retrying, and the partial file is cleaned up.
    """


def _require_verification() -> bool:
    """Whether downloads must verify: the policy fails closed on the official origin.

    There every downloaded file must carry a published SHA-256. Overriding the base URL
    (forks, mirrors, offline, tests) or setting `QUOTH_ALLOW_UNVERIFIED=1` makes it
    lenient; overriding only the tag stays official.

    Manifest and assets share GitHub's HTTPS trust path, so this guards against
    corruption and a stripped or partial release, not a TLS-breaking attacker.
    """
    return (
        RELEASE_BASE_URL == _DEFAULT_RELEASE_BASE_URL
        and os.environ.get("QUOTH_ALLOW_UNVERIFIED") != "1"
    )


def download_model(
    models_dir: Path | str,
    model: str,
    device: str,
    on_progress: Callable[[DownloadProgress], None] | None = None,
    cancel: threading.Event | None = None,
) -> None:
    """Download what `(model, device)` needs: ggml, plus the OpenVINO encoder for NPU.

    ggml is required, so its failure raises; the encoder is best-effort, so its
    failure logs and the NPU request falls back to the GPU, else the CPU. On the
    official origin every file must carry a published SHA-256. Setting `cancel` aborts
    the in-flight file with `DownloadCancelled`.
    """
    models_dir = Path(models_dir)
    cancel = cancel or threading.Event()
    want_encoder = needs_openvino_encoder(device, model)
    progress = _ProgressSpan(on_progress, 1 + (2 if want_encoder else 0))
    require = _require_verification()
    manifest = _Manifest(ASSETS_TAG, required=require, cancel=cancel)

    # Only whether a proxy is set: its URL can carry credentials.
    proxied = bool(urllib.request.getproxies().get(RELEASE_BASE_URL.split(":", 1)[0]))
    logger.debug(
        "Downloading model '{}' for device '{}' (encoder={}, verify_required={}, "
        "proxy={})",
        model,
        device,
        want_encoder,
        require,
        proxied,
    )
    _fetch_assets([ggml_name(model)], models_dir, manifest, progress, require, cancel)
    if not want_encoder:
        logger.success("Model '{}' ready", model)
        return
    try:
        _fetch_assets(
            [encoder_xml_name(model), encoder_bin_name(model)],
            models_dir,
            manifest,
            progress,
            require,
            cancel,
        )
        logger.success("Model '{}' ready (with OpenVINO encoder)", model)
    except DownloadCancelled:
        raise  # a cancel is not an encoder failure: it must not become a fallback
    except DOWNLOAD_ERRORS as e:
        logger.warning(
            "OpenVINO encoder unavailable for '{}' ({}); {} falls back to another device",
            model,
            failure_reason(e),
            device,
        )
    except Exception:
        logger.exception(
            "OpenVINO encoder download failed for '{}'; {} falls back to another device",
            model,
            device,
        )


def _fetch_assets(
    names: list[str],
    models_dir: Path,
    manifest: "_Manifest",
    progress: "_ProgressSpan",
    require: bool,
    cancel: threading.Event,
):
    """Fetch each named release asset that is not already present."""
    for name in names:
        dest = models_dir / name
        if dest.exists():
            logger.debug("{} already present, skipping download", name)
            progress.skip()
            continue
        url = f"{RELEASE_BASE_URL}/{ASSETS_TAG}/{name}"
        _fetch_one(url, dest, manifest.get(name), name, progress, require, cancel)


def _fetch_one(
    url: str,
    dest: Path,
    sha256: str | None,
    name: str,
    progress: "_ProgressSpan",
    require: bool,
    cancel: threading.Event,
):
    """Download one file to a temp path (resumably), verify, then move it into place.

    A network fault or HTTP error leaves the `.tmp` partial for a later attempt to
    resume via Range. Any other failure, such as a cancel, a bad checksum or a full
    disk, discards it.
    """
    logger.debug("Fetching {} from {} (verify_required={})", name, url, require)
    if not sha256:
        if require:
            raise IntegrityError(
                f"{name}: no published SHA-256 in the release manifest. Refusing "
                f"to install an unverified file from the official origin "
                f"(set QUOTH_ALLOW_UNVERIFIED=1 to override)"
            )
        logger.warning("No checksum for {}; skipping integrity check", name)
    tmp = dest.with_name(partial_name(dest.name))
    try:
        _download_with_retries(
            lambda: _download_file(url, tmp, progress.step, cancel), name, cancel
        )
        if sha256:
            _verify(tmp, sha256, name)
        tmp.rename(dest)
        # The rename survives a crash too, so a model reported ready stays installed.
        atomic_write.sync_directory(dest.parent)
    except _TRANSIENT_ERRORS as e:
        logger.debug(
            "Fetch failed for {} ({}): {}; partial kept to resume",
            name,
            url,
            failure_reason(e),
        )
        e.add_note(name)
        raise
    except Exception as e:
        logger.debug(
            "Fetch aborted for {} ({}): {}; discarding partial",
            name,
            url,
            failure_reason(e),
        )
        tmp.unlink(missing_ok=True)
        e.add_note(name)
        raise
    progress.advance()


class _ProgressSpan:
    """A job's progress, mapping each file's bytes onto one overall 0..1 fraction.

    It emits a `DownloadProgress` with the overall fraction across N files and the
    current file's bytes and speed.
    """

    def __init__(
        self, callback: Callable[[DownloadProgress], None] | None, total_files: int
    ):
        self._cb = callback
        self._total_files = total_files
        self._done_files = 0

    def step(self, downloaded: int, total: int, bytes_per_sec: float):
        if not self._cb:
            return
        file_frac = downloaded / total if total > 0 else 0.0
        overall = (self._done_files + file_frac) / self._total_files
        self._cb(DownloadProgress(overall, downloaded, total, bytes_per_sec))

    def advance(self):
        self._done_files += 1
        if self._cb:
            self._cb(DownloadProgress(self._done_files / self._total_files, 0, 0, 0.0))

    def skip(self):
        self.advance()


# --- network + integrity helpers ---


def _fetch_manifest(
    tag: str, *, required: bool, cancel: threading.Event
) -> dict[str, str]:
    """Return the release manifest, mapping filename -> sha256.

    Transient faults are retried as for an asset. Where verification is required, an
    unreachable manifest fails the download with its cause and a malformed one as an
    integrity failure; elsewhere either is empty.
    """
    url = f"{RELEASE_BASE_URL}/{tag}/{MANIFEST_FILENAME}"

    def fetch():
        with urllib.request.urlopen(url, timeout=_STALL_TIMEOUT_S) as response:
            return json.load(response)

    try:
        data = _download_with_retries(fetch, MANIFEST_FILENAME, cancel)
        if not isinstance(data, dict) or not all(
            isinstance(sha256, str) for sha256 in data.values()
        ):
            raise ValueError("not an object of file names to SHA-256 hashes")
    except (OSError, http.client.HTTPException, ValueError) as e:
        if not required:
            logger.warning("No download manifest for {} ({})", tag, failure_reason(e))
            return {}
        if isinstance(e, ValueError):
            raise IntegrityError(f"{MANIFEST_FILENAME} for {tag} is malformed") from e
        e.add_note(MANIFEST_FILENAME)
        raise
    return data


class _Manifest:
    """The release manifest, fetched once on first lookup.

    Files already present never consult it, so a fully present model touches no
    network.
    """

    def __init__(self, tag: str, *, required: bool, cancel: threading.Event):
        self._tag = tag
        self._required = required
        self._cancel = cancel
        self._data: dict[str, str] | None = None

    def get(self, name: str) -> str | None:
        if self._data is None:
            self._data = _fetch_manifest(
                self._tag, required=self._required, cancel=self._cancel
            )
        return self._data.get(name)


def _verify(path: Path, expected_sha256: str, name: str):
    """Raise IntegrityError unless the file's SHA-256 matches the manifest."""
    actual = _sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise IntegrityError(
            f"{name}: SHA-256 mismatch (expected {expected_sha256}, got {actual})"
        )
    logger.debug("Verified {} ({})", name, actual)


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _open_ranged(url: str, have: int):
    """Open `url`, requesting the bytes after `have` when a partial exists.

    Returns `(response, resuming)`: `resuming` is True only on a 206 that honours the
    range. A 200 (the server ignored Range) or a 416 (the existing bytes already cover
    the file, e.g. a stale partial) both fall through to a fresh, full download.
    """
    headers = {"Range": f"bytes={have}-"} if have > 0 else {}
    try:
        response = urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=_STALL_TIMEOUT_S
        )
    except urllib.error.HTTPError as e:
        if e.code == 416 and have > 0:
            response = urllib.request.urlopen(
                urllib.request.Request(url), timeout=_STALL_TIMEOUT_S
            )
            return response, False
        raise
    return response, have > 0 and response.status == 206


def _expected_total(response, have: int) -> int:
    """Return the full expected size in bytes, 0 if the server sends none.

    A 206 carries the whole size in Content-Range (`bytes start-end/total`); otherwise
    Content-Length is the whole file.
    """
    if response.status == 206:
        tail = response.headers.get("Content-Range", "").rsplit("/", 1)[-1]
        if tail.isascii() and tail.isdigit():
            return int(tail)
        return have + int(response.headers.get("Content-Length", 0))
    return int(response.headers.get("Content-Length", 0))


def _download_file(
    url: str,
    dest: Path,
    on_progress: Callable[[int, int, float], None] | None,
    cancel: threading.Event,
):
    """Stream one file to `dest`, resuming a partial `.tmp` via Range when supported.

    Reports `(downloaded, total, bytes_per_sec)` at most every `_PROGRESS_INTERVAL_S`.
    Raises `IncompleteDownloadError` if the transfer ends short of the expected total
    (a dropped connection), or `DownloadCancelled` if `cancel` is set mid-stream.
    """
    have = dest.stat().st_size if dest.exists() else 0
    response, resuming = _open_ranged(url, have)
    with response:
        if not resuming:
            if have:
                logger.debug(
                    "{}: server did not resume; restarting from 0 ({} bytes discarded)",
                    dest.name,
                    have,
                )
            have = 0  # no partial, or the server ignored Range: start fresh
        total = _expected_total(response, have)
        logger.debug(
            "Downloading {} (size={}, resume_from={})",
            dest.name,
            total or "unknown",
            have,
        )
        downloaded = have
        start = last_time = time.monotonic()
        last_bytes = downloaded
        with open(dest, "ab" if resuming else "wb") as f:
            while True:
                if cancel.is_set():
                    raise DownloadCancelled()
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if on_progress and now - last_time >= _PROGRESS_INTERVAL_S:
                    speed = (downloaded - last_bytes) / (now - last_time)
                    on_progress(downloaded, total, speed)
                    last_bytes, last_time = downloaded, now
            # On disk before the caller renames it into place: a power loss after the
            # rename must not leave a truncated model that whisper.cpp crashes on.
            f.flush()
            os.fsync(f.fileno())
        elapsed = time.monotonic() - start
        if on_progress:
            on_progress(downloaded, total, downloaded / elapsed if elapsed > 0 else 0.0)
        if total > 0 and downloaded != total:
            raise IncompleteDownloadError(
                f"{dest.name}: received {downloaded} of {total} bytes"
            )
        # Not yet verified: the checksum is checked after the transfer.
        logger.debug(
            "Received {} ({} bytes in {:.1f}s)", dest.name, downloaded - have, elapsed
        )


# Transient network faults worth retrying; a bad checksum is not one. An HTTPError
# is a URLError too, and `_retry_delay` lets only some statuses retry.
_TRANSIENT_ERRORS = (
    TimeoutError,
    ConnectionError,
    ssl.SSLError,  # a TLS record garbled mid-body: not a ConnectionError
    urllib.error.URLError,
    IncompleteDownloadError,
    http.client.IncompleteRead,  # a manifest or chunked body cut off mid-transfer
)
# The statuses that mean "try again later": rate limiting, and a gateway or service
# that is overloaded or failing upstream. Any other status is final.
_RETRY_STATUSES = frozenset({429, 502, 503, 504})
# The longest server-requested wait worth sitting through on a progress screen. A
# longer Retry-After fails at once: retrying sooner would only be refused again.
_MAX_RETRY_AFTER_S = 60


def _download_with_retries(
    fetch: Callable[[], object], name: str, cancel: threading.Event
) -> object:
    """Run the download `fetch` and return its result, retrying transient faults.

    `_retry_delay` sets the wait. A cancel aborts at once, even mid-backoff.
    """
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return fetch()
        except _TRANSIENT_ERRORS as e:
            delay = _retry_delay(e, attempt)
            if delay is None or attempt == _MAX_ATTEMPTS:
                raise  # the caller's boundary logs the final failure once
            logger.warning(
                "Download of {} failed (attempt {}/{}: {}); retrying in {:.0f}s",
                name,
                attempt,
                _MAX_ATTEMPTS,
                failure_reason(e),
                delay,
            )
            if cancel.wait(delay):
                raise DownloadCancelled() from None


def _retry_delay(error: Exception, attempt: int) -> float | None:
    """Return the seconds to wait before retrying after `error`, or None if final.

    A network fault backs off, growing per attempt. A temporary status waits for the
    server's Retry-After when it sends one, and backs off otherwise.
    """
    backoff = _RETRY_BACKOFF_S * attempt
    if not isinstance(error, urllib.error.HTTPError):
        return backoff
    if error.code not in _RETRY_STATUSES:
        return None
    delay = _retry_after(error.headers.get("Retry-After", ""))
    if delay is None:
        return backoff
    return delay if delay <= _MAX_RETRY_AFTER_S else None


def _retry_after(value: str) -> float | None:
    """Return the wait a Retry-After value asks for, in seconds.

    It is a number of seconds or an HTTP-date; anything else is None.
    """
    if value.strip().isdecimal():
        return int(value)
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except ValueError:
        return None
