# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `downloader`.

Downloads are faked or served by a local HTTP server; none reaches the internet.
"""

import email.utils
import errno
import hashlib
import http.client
import http.server
import json
import os
import ssl
import stat
import threading
import time
import urllib.error

import pytest

import quoth.downloader as dl
from quoth.models import AVAILABLE_MODELS

_GGML = b"ggml-weights"
_ENC = b"openvino-encoder"


def _fake_manifest() -> dict[str, str]:
    """A release manifest covering every catalog asset (single origin)."""
    ggml_hash = hashlib.sha256(_GGML).hexdigest()
    enc_hash = hashlib.sha256(_ENC).hexdigest()
    manifest = {}
    for model in AVAILABLE_MODELS:
        manifest[dl.ggml_name(model.name)] = ggml_hash
        if model.has_openvino_encoder:
            manifest[dl.encoder_xml_name(model.name)] = enc_hash
            manifest[dl.encoder_bin_name(model.name)] = enc_hash
    return manifest


class _InstantBackoff(threading.Event):
    """A cancel Event whose backoff waits return at once, recording each delay."""

    def __init__(self):
        super().__init__()
        self.waits: list[float] = []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return self.is_set()


class _Origin:
    """A real local HTTP origin: its scripted errors first, then `files` by name.

    Each GET takes the next `(status, headers)` from `errors` while any are left;
    `requests` counts the GETs it answered.
    """

    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.errors: list[tuple[int, dict[str, str]]] = []
        self.requests = 0
        origin = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                origin.requests += 1
                if origin.errors:
                    (status, headers), body = origin.errors.pop(0), b""
                else:
                    body = origin.files.get(self.path.rsplit("/", 1)[-1], b"")
                    status, headers = (200 if body else 404), {}
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        threading.Thread(
            target=self._server.serve_forever, args=(0.01,), daemon=True
        ).start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()


def _fetch(url: str, dest, cancel) -> None:
    """Fetch one asset as `download_model` does: retried, resumable and verified."""
    sha256 = hashlib.sha256(_GGML).hexdigest()
    dl._fetch_one(url, dest, sha256, dest.name, dl._ProgressSpan(None, 1), True, cancel)


@pytest.fixture
def origin(monkeypatch):
    monkeypatch.setenv("no_proxy", "127.0.0.1")  # loopback, even behind a proxy
    server = _Origin({"f": _GGML})
    yield server
    server.close()


def _install_fakes(monkeypatch):
    """Fake the network so downloads succeed with correct checksums.

    Like the real stream, a fetch honours its cancel Event and reports its bytes.
    """

    def fake_download(url, dest, on_progress=None, cancel=None):
        if cancel.is_set():
            raise dl.DownloadCancelled()
        data = _ENC if "openvino" in dest.name else _GGML
        dest.write_bytes(data)
        if on_progress:
            on_progress(len(data), len(data), 0.0)

    monkeypatch.setattr(dl, "_download_file", fake_download)
    monkeypatch.setattr(dl, "_fetch_manifest", lambda tag, **_: _fake_manifest())


class TestDownloadModel:
    def test_cpu_downloads_ggml_only(self, tmp_data_dir, monkeypatch, caplog):
        _install_fakes(monkeypatch)
        monkeypatch.setenv("https_proxy", "http://user:secret@proxy.example:3128")
        dl.download_model(tmp_data_dir, "small", "CPU")
        # Only that a proxy is in use: its URL can carry credentials.
        assert any("proxy=True" in m for m in caplog.messages)
        assert not any("proxy.example" in m for m in caplog.messages)
        assert (tmp_data_dir / "ggml-small-q8_0.bin").exists()
        assert not (tmp_data_dir / "ggml-small-encoder-openvino.xml").exists()

    @pytest.mark.parametrize(
        "model", [m.name for m in AVAILABLE_MODELS if m.has_openvino_encoder]
    )
    def test_npu_downloads_ggml_and_encoder(self, tmp_data_dir, monkeypatch, model):
        _install_fakes(monkeypatch)
        dl.download_model(tmp_data_dir, model, "NPU")
        assert (tmp_data_dir / dl.ggml_name(model)).exists()
        assert (tmp_data_dir / dl.encoder_xml_name(model)).exists()
        assert (tmp_data_dir / dl.encoder_bin_name(model)).exists()

    def test_encoder_failure_is_best_effort(self, tmp_data_dir, monkeypatch):
        _install_fakes(monkeypatch)

        def ggml_only(url, dest, on_progress=None, cancel=None):
            if "openvino" in dest.name:
                raise RuntimeError("404")
            dest.write_bytes(_GGML)

        monkeypatch.setattr(dl, "_download_file", ggml_only)
        # The ggml lands and the encoder fails, with no crash.
        dl.download_model(tmp_data_dir, "small", "NPU")
        assert (tmp_data_dir / "ggml-small-q8_0.bin").exists()
        assert not (tmp_data_dir / "ggml-small-encoder-openvino.xml").exists()

    @pytest.mark.parametrize(
        ("error", "logged"),
        [
            (OSError("404"), ("WARNING", False)),
            (RuntimeError("defect"), ("ERROR", True)),
        ],
    )
    def test_encoder_fallback_keeps_a_traceback_only_for_defects(
        self, tmp_data_dir, monkeypatch, error, logged, caplog
    ):
        _install_fakes(monkeypatch)

        def ggml_only(url, dest, on_progress=None, cancel=None):
            if "openvino" in dest.name:
                raise error
            dest.write_bytes(_GGML)

        monkeypatch.setattr(dl, "_download_file", ggml_only)
        caplog.set_level("WARNING")
        dl.download_model(tmp_data_dir, "small", "NPU")
        [record] = caplog.records
        assert (record.levelname, record.exc_info is not None) == logged

    def test_cancel_during_encoder_is_not_a_cpu_fallback(
        self, tmp_data_dir, monkeypatch
    ):
        _install_fakes(monkeypatch)

        def cancelled_at_encoder(url, dest, on_progress=None, cancel=None):
            if "openvino" in dest.name:
                raise dl.DownloadCancelled()
            dest.write_bytes(_GGML)

        monkeypatch.setattr(dl, "_download_file", cancelled_at_encoder)
        with pytest.raises(dl.DownloadCancelled):
            dl.download_model(tmp_data_dir, "small", "NPU")

    @pytest.mark.parametrize("device", ["CPU", "NPU"])
    def test_progress_runs_to_one_across_every_file(
        self, tmp_data_dir, monkeypatch, device
    ):
        _install_fakes(monkeypatch)
        emitted = []
        dl.download_model(tmp_data_dir, "small", device, on_progress=emitted.append)
        assert emitted[-1].fraction == 1.0
        assert any(p.downloaded > 0 for p in emitted)  # a file's bytes reach the caller

    def test_a_partly_present_encoder_fetches_only_the_missing_file(
        self, tmp_data_dir, monkeypatch
    ):
        _install_fakes(monkeypatch)
        for name in ("ggml-small-q8_0.bin", "ggml-small-encoder-openvino.xml"):
            (tmp_data_dir / name).write_bytes(b"present")
        dl.download_model(tmp_data_dir, "small", "NPU")
        assert (
            tmp_data_dir / "ggml-small-encoder-openvino.xml"
        ).read_bytes() == b"present"
        assert (tmp_data_dir / "ggml-small-encoder-openvino.bin").read_bytes() == _ENC

    def test_skips_present_files_without_touching_network(
        self, tmp_data_dir, monkeypatch
    ):
        (tmp_data_dir / "ggml-small-q8_0.bin").write_bytes(b"x")
        calls = []
        monkeypatch.setattr(dl, "_download_file", lambda *a, **k: calls.append(a))
        monkeypatch.setattr(
            dl, "_fetch_manifest", lambda tag, **_: pytest.fail("manifest fetched")
        )
        dl.download_model(tmp_data_dir, "small", "CPU")
        assert calls == []

    def test_ggml_hash_mismatch_raises_and_cleans_tmp(self, tmp_data_dir, monkeypatch):
        monkeypatch.setattr(
            dl,
            "_download_file",
            lambda url, dest, on_progress=None, cancel=None: dest.write_bytes(
                b"corrupt"
            ),
        )
        monkeypatch.setattr(
            dl, "_fetch_manifest", lambda tag, **_: {"ggml-small-q8_0.bin": "00" * 32}
        )
        with pytest.raises(dl.IntegrityError):
            dl.download_model(tmp_data_dir, "small", "CPU")
        assert not (tmp_data_dir / "ggml-small-q8_0.bin").exists()
        assert not (tmp_data_dir / "ggml-small-q8_0.bin.tmp").exists()

    def test_transient_failure_keeps_partial_for_resume(
        self, tmp_data_dir, monkeypatch
    ):
        # A dropped connection surfaces as a socket TimeoutError; the partial .tmp is
        # kept on disk so a later attempt resumes it instead of restarting.
        def stall(url, dest, on_progress=None, cancel=None):
            dest.write_bytes(b"partial-bytes")  # some bytes landed before the drop
            raise TimeoutError("connection stalled")

        monkeypatch.setattr(dl, "_download_file", stall)
        monkeypatch.setattr(dl, "_fetch_manifest", lambda tag, **_: _fake_manifest())
        with pytest.raises(TimeoutError):
            dl.download_model(tmp_data_dir, "small", "CPU", cancel=_InstantBackoff())
        assert not (tmp_data_dir / "ggml-small-q8_0.bin").exists()
        assert (tmp_data_dir / "ggml-small-q8_0.bin.tmp").exists()  # kept to resume

    def test_a_full_disk_discards_the_partial(self, tmp_data_dir, monkeypatch):
        # Not resumable, and a partial nothing lists would only hold the space.
        def disk_full(url, dest, on_progress=None, cancel=None):
            dest.write_bytes(b"partial-bytes")
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(dl, "_download_file", disk_full)
        monkeypatch.setattr(dl, "_fetch_manifest", lambda tag, **_: _fake_manifest())
        with pytest.raises(OSError, match="No space left") as raised:
            dl.download_model(tmp_data_dir, "small", "CPU", cancel=_InstantBackoff())
        assert not (tmp_data_dir / "ggml-small-q8_0.bin.tmp").exists()
        # The error names no file, so the boundary logs the one this note gives.
        assert raised.value.__notes__ == ["ggml-small-q8_0.bin"]

    def test_cancel_discards_partial(self, tmp_data_dir, monkeypatch):
        # An explicit cancel means the user opted out: the partial is not kept.
        def cancelled(url, dest, on_progress=None, cancel=None):
            dest.write_bytes(b"partial-bytes")
            raise dl.DownloadCancelled()

        monkeypatch.setattr(dl, "_download_file", cancelled)
        monkeypatch.setattr(dl, "_fetch_manifest", lambda tag, **_: _fake_manifest())
        with pytest.raises(dl.DownloadCancelled):
            dl.download_model(tmp_data_dir, "small", "CPU")
        assert not (tmp_data_dir / "ggml-small-q8_0.bin.tmp").exists()

    def test_manifest_fetched_once_for_full_npu_download(
        self, tmp_data_dir, monkeypatch
    ):
        _install_fakes(monkeypatch)
        fetches = []
        manifest = _fake_manifest()
        monkeypatch.setattr(
            dl, "_fetch_manifest", lambda tag, **_: fetches.append(tag) or manifest
        )
        dl.download_model(tmp_data_dir, "small", "NPU")
        assert fetches == [dl.ASSETS_TAG]


class _FakeResponse:
    """Minimal stand-in for an http.client.HTTPResponse.

    Each read hands over the next of `chunks`, then end of stream.
    """

    def __init__(self, status: int, headers: dict, *chunks: bytes):
        self.status = status
        self.headers = headers
        self._chunks = list(chunks)

    def read(self, n: int = -1) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestResume:
    """`_download_file` resumes a partial `.tmp` via HTTP Range, or restarts cleanly."""

    def test_resumes_from_partial_with_206(self, tmp_data_dir, monkeypatch, caplog):
        full = b"0123456789" * 10
        dest = tmp_data_dir / "f.tmp"
        dest.write_bytes(full[:60])  # 60 of 100 bytes already on disk
        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["range"] = req.get_header("Range")
            rest = full[60:]
            return _FakeResponse(
                206,
                {
                    "Content-Length": str(len(rest)),
                    "Content-Range": f"bytes 60-{len(full) - 1}/{len(full)}",
                },
                rest,
            )

        monkeypatch.setattr(dl.urllib.request, "urlopen", fake_urlopen)
        dl._download_file("http://x/f", dest, None, threading.Event())
        assert seen["range"] == "bytes=60-"
        assert dest.read_bytes() == full
        assert caplog.messages[-1].startswith("Received f.tmp (40 bytes in ")

    def test_restarts_when_server_ignores_range(
        self, tmp_data_dir, monkeypatch, caplog
    ):
        full = b"abcde" * 20
        dest = tmp_data_dir / "f.tmp"
        dest.write_bytes(b"OLDPARTIAL")

        def fake_urlopen(req, timeout=None):
            # 200, not 206: the server ignored the Range and sends the whole body.
            return _FakeResponse(200, {"Content-Length": str(len(full))}, full)

        monkeypatch.setattr(dl.urllib.request, "urlopen", fake_urlopen)
        dl._download_file("http://x/f", dest, None, threading.Event())
        assert dest.read_bytes() == full  # truncated + rewritten, not appended
        assert (
            "f.tmp: server did not resume; restarting from 0 (10 bytes discarded)"
            in caplog.messages
        )

    def test_restarts_on_416_range_not_satisfiable(self, tmp_data_dir, monkeypatch):
        full = b"xy" * 15
        dest = tmp_data_dir / "f.tmp"
        dest.write_bytes(b"STALE-PARTIAL-LONGER")  # a stale, oversized partial
        requests = []

        def fake_urlopen(req, timeout=None):
            rng = req.get_header("Range")
            requests.append((rng, timeout))
            if rng:
                raise urllib.error.HTTPError(
                    "http://x/f", 416, "Range Not Satisfiable", {}, None
                )
            return _FakeResponse(200, {"Content-Length": str(len(full))}, full)

        monkeypatch.setattr(dl.urllib.request, "urlopen", fake_urlopen)
        dl._download_file("http://x/f", dest, None, threading.Event())
        assert dest.read_bytes() == full
        # Ranged, then fresh, each with the stall timeout.
        assert requests == [
            ("bytes=20-", dl._STALL_TIMEOUT_S),
            (None, dl._STALL_TIMEOUT_S),
        ]

    def test_a_206_without_content_length_takes_its_size_from_content_range(
        self, tmp_data_dir, monkeypatch
    ):
        full = b"0123456789" * 10
        dest = tmp_data_dir / "f.tmp"
        dest.write_bytes(full[:60])
        response = _FakeResponse(206, {"Content-Range": "bytes 60-99/100"}, full[60:])
        monkeypatch.setattr(
            dl.urllib.request, "urlopen", lambda req, timeout=None: response
        )
        dl._download_file("http://x/f", dest, None, threading.Event())
        assert dest.read_bytes() == full

    # "*" is HTTP's unknown total; a superscript digit is no number int() can read.
    @pytest.mark.parametrize("total", ["*", "\u00b2"])
    def test_a_206_without_a_numeric_total_takes_its_size_from_content_length(
        self, tmp_data_dir, monkeypatch, total
    ):
        full = b"0123456789" * 10
        dest = tmp_data_dir / "f.tmp"
        dest.write_bytes(full[:60])
        response = _FakeResponse(
            206,
            {"Content-Range": f"bytes 60-99/{total}", "Content-Length": "40"},
            full[60:],
        )
        monkeypatch.setattr(
            dl.urllib.request, "urlopen", lambda req, timeout=None: response
        )
        dl._download_file("http://x/f", dest, None, threading.Event())
        assert dest.read_bytes() == full


class TestIntegrity:
    def test_sha256_matches_hashlib(self, tmp_data_dir):
        f = tmp_data_dir / "b.bin"
        f.write_bytes(b"data")
        assert dl._sha256(f) == hashlib.sha256(b"data").hexdigest()

    def test_verify_accepts_match(self, tmp_data_dir):
        f = tmp_data_dir / "b.bin"
        f.write_bytes(b"data")
        dl._verify(f, hashlib.sha256(b"data").hexdigest(), "b.bin")

    def test_verify_rejects_mismatch(self, tmp_data_dir):
        f = tmp_data_dir / "b.bin"
        f.write_bytes(b"data")
        with pytest.raises(dl.IntegrityError):
            dl._verify(f, "deadbeef", "b.bin")


class TestIntegrityPolicy:
    """Fail closed on the official origin, lenient when overridden."""

    def _be_official(self, monkeypatch):
        monkeypatch.setattr(dl, "RELEASE_BASE_URL", dl._DEFAULT_RELEASE_BASE_URL)
        monkeypatch.delenv("QUOTH_ALLOW_UNVERIFIED", raising=False)

    def test_official_origin_requires_verification(self, monkeypatch):
        self._be_official(monkeypatch)
        assert dl._require_verification() is True

    def test_overridden_url_is_lenient(self, monkeypatch):
        monkeypatch.setattr(dl, "RELEASE_BASE_URL", "http://localhost:8000")
        assert dl._require_verification() is False

    def test_escape_hatch_disables_strict(self, monkeypatch):
        self._be_official(monkeypatch)
        monkeypatch.setenv("QUOTH_ALLOW_UNVERIFIED", "1")
        assert dl._require_verification() is False

    def test_strict_missing_hash_is_refused_before_the_transfer(
        self, tmp_data_dir, monkeypatch
    ):
        # On the official origin a file absent from the manifest is refused unfetched.
        monkeypatch.setattr(dl, "_require_verification", lambda: True)
        fetched = []
        monkeypatch.setattr(
            dl,
            "_download_file",
            lambda url, dest, on_progress=None, cancel=None: fetched.append(url),
        )
        monkeypatch.setattr(dl, "_fetch_manifest", lambda tag, **_: {"other": "00"})
        with pytest.raises(dl.IntegrityError):
            dl.download_model(tmp_data_dir, "small", "CPU")
        assert fetched == []
        assert not (tmp_data_dir / "ggml-small-q8_0.bin").exists()

    def test_strict_unreachable_manifest_reports_its_cause(
        self, tmp_data_dir, monkeypatch
    ):
        # Offline on the official origin is a network failure, not a missing hash.
        monkeypatch.setattr(dl, "_require_verification", lambda: True)
        monkeypatch.setattr(
            dl.urllib.request,
            "urlopen",
            lambda url, timeout=None: (_ for _ in ()).throw(
                urllib.error.URLError("offline")
            ),
        )
        backoff = _InstantBackoff()
        with pytest.raises(urllib.error.URLError) as raised:
            dl.download_model(tmp_data_dir, "small", "CPU", cancel=backoff)
        assert len(backoff.waits) == dl._MAX_ATTEMPTS - 1  # retried before failing
        assert raised.value.__notes__ == ["manifest.json"]

    def test_strict_encoder_without_a_hash_is_refused(self, tmp_data_dir, monkeypatch):
        # The NPU's encoder is held to the same rule, so the request falls back.
        _install_fakes(monkeypatch)
        monkeypatch.setattr(dl, "_require_verification", lambda: True)
        monkeypatch.setattr(
            dl,
            "_fetch_manifest",
            lambda tag, **_: {"ggml-small-q8_0.bin": hashlib.sha256(_GGML).hexdigest()},
        )
        dl.download_model(tmp_data_dir, "small", "NPU")
        assert (tmp_data_dir / "ggml-small-q8_0.bin").exists()
        assert not (tmp_data_dir / "ggml-small-encoder-openvino.xml").exists()

    def test_lenient_missing_hash_accepts(self, tmp_data_dir, monkeypatch):
        # An overridden origin (fork or offline) warns and accepts without a manifest.
        monkeypatch.setattr(dl, "_require_verification", lambda: False)
        monkeypatch.setattr(
            dl,
            "_download_file",
            lambda url, dest, on_progress=None, cancel=None: dest.write_bytes(_GGML),
        )
        monkeypatch.setattr(dl, "_fetch_manifest", lambda tag, **_: {})
        dl.download_model(tmp_data_dir, "small", "CPU")
        assert (tmp_data_dir / "ggml-small-q8_0.bin").exists()


class TestSingleOrigin:
    """Every asset comes from the project release, manifest-verified."""

    def test_ggml_url_uses_release_origin(self, tmp_data_dir, monkeypatch):
        urls = []

        def fake_download(url, dest, on_progress=None, cancel=None):
            urls.append(url)
            dest.write_bytes(_GGML)

        monkeypatch.setattr(dl, "_download_file", fake_download)
        monkeypatch.setattr(dl, "_fetch_manifest", lambda tag, **_: _fake_manifest())
        dl.download_model(tmp_data_dir, "tiny", "CPU")
        assert urls == [f"{dl.RELEASE_BASE_URL}/{dl.ASSETS_TAG}/ggml-tiny-q8_0.bin"]


class TestDownloadFile:
    def test_a_fetched_file_is_on_disk_before_and_after_its_rename(
        self, tmp_data_dir, origin, monkeypatch
    ):
        # Synced bytes first, then the directory entry the rename wrote.
        dest = tmp_data_dir / "f.bin"
        synced = []
        real_fsync = os.fsync

        def fsync(fd):
            kind = "dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
            synced.append((kind, dest.exists()))
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", fsync)
        _fetch(f"{origin.url}/f", dest, _InstantBackoff())
        assert dest.read_bytes() == _GGML
        assert synced == [("file", False), ("dir", True)]

    def test_streams_and_reports_progress(self, tmp_data_dir, monkeypatch):
        response = _FakeResponse(200, {"Content-Length": "20"}, b"a" * 10, b"b" * 10)
        monkeypatch.setattr(
            dl.urllib.request, "urlopen", lambda req, timeout=None: response
        )
        monkeypatch.setattr(dl, "_PROGRESS_INTERVAL_S", 1e-9)  # report every chunk
        calls = []
        dest = tmp_data_dir / "out.bin"
        dl._download_file(
            "http://x/f",
            dest,
            lambda d, t, s: calls.append((d, t, s)),
            threading.Event(),
        )
        assert dest.read_bytes() == b"a" * 10 + b"b" * 10
        assert calls[0][:2] == (10, 20)  # reported while streaming
        assert calls[-1][:2] == (20, 20)  # and the full size at the end
        assert calls[0][2] > 0 and calls[-1][2] > 0  # with a speed

    def test_passes_stall_timeout(self, tmp_data_dir, monkeypatch):
        # The socket carries the idle timeout, so a frozen connection fails instead
        # of hanging forever; a steady-but-slow link is unaffected (it's per-read).
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["timeout"] = timeout
            return _FakeResponse(200, {"Content-Length": "0"})

        monkeypatch.setattr(dl.urllib.request, "urlopen", fake_urlopen)
        dl._download_file(
            "http://x/f", tmp_data_dir / "out.bin", None, threading.Event()
        )
        assert captured["timeout"] == dl._STALL_TIMEOUT_S

    def test_an_unknown_size_streams_to_the_end(self, tmp_data_dir, monkeypatch):
        # No Content-Length, as in a chunked body: the total reads 0.
        response = _FakeResponse(200, {}, b"a" * 10, b"b" * 10)
        monkeypatch.setattr(
            dl.urllib.request, "urlopen", lambda req, timeout=None: response
        )
        calls = []
        dest = tmp_data_dir / "out.bin"
        dl._download_file(
            "http://x/f",
            dest,
            lambda d, t, s: calls.append((d, t)),
            threading.Event(),
        )
        assert dest.read_bytes() == b"a" * 10 + b"b" * 10
        assert calls[-1] == (20, 0)

    def test_incomplete_transfer_raises(self, tmp_data_dir, monkeypatch):
        # The server closes early, short of Content-Length: a retryable error rather
        # than a silently truncated file.
        response = _FakeResponse(200, {"Content-Length": "20"}, b"a" * 10)
        monkeypatch.setattr(
            dl.urllib.request, "urlopen", lambda req, timeout=None: response
        )
        with pytest.raises(dl.IncompleteDownloadError):
            dl._download_file(
                "http://x/f", tmp_data_dir / "out.bin", None, threading.Event()
            )


class TestProgressSpan:
    def test_maps_file_bytes_onto_overall_fraction(self):
        emitted = []
        span = dl._ProgressSpan(emitted.append, 2)
        # First file half done: overall (0 + 0.5) / 2 = 0.25, bytes passed through.
        span.step(50, 100, 4_000_000.0)
        assert emitted[-1] == dl.DownloadProgress(0.25, 50, 100, 4_000_000.0)
        # First file complete: overall 1/2, byte counters reset for the next file.
        span.advance()
        assert emitted[-1] == dl.DownloadProgress(0.5, 0, 0, 0.0)
        # Second file half done: overall (1 + 0.5) / 2 = 0.75.
        span.step(50, 100, 0.0)
        assert emitted[-1].fraction == 0.75

    def test_an_unknown_size_holds_the_fraction(self):
        emitted = []
        span = dl._ProgressSpan(emitted.append, 2)
        span.step(50, 0, 0.0)  # no Content-Length
        assert emitted == [dl.DownloadProgress(0.0, 50, 0, 0.0)]

    def test_no_callback_is_safe(self):
        span = dl._ProgressSpan(None, 1)
        span.step(1, 2, 0.0)
        span.advance()  # no error either way


class TestDownloadRetries:
    @pytest.mark.parametrize(
        "fault",
        [
            TimeoutError("stalled"),
            ConnectionResetError("reset"),
            ssl.SSLError(1, "[SSL: RECORD_LAYER_FAILURE] record layer failure"),
            dl.IncompleteDownloadError("short"),
            http.client.IncompleteRead(b"{", 63),  # a manifest or chunk cut short
        ],
    )
    def test_retries_transient_then_succeeds(self, fault):
        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise fault
            return "ok"

        backoff = _InstantBackoff()
        assert dl._download_with_retries(flaky, "f.bin", backoff) == "ok"
        assert attempts["n"] == 3
        # The backoff grows per attempt and is waited out on the cancel Event.
        assert backoff.waits == [dl._RETRY_BACKOFF_S, 2 * dl._RETRY_BACKOFF_S]

    def test_gives_up_after_max_attempts(self):
        def always_stall():
            raise TimeoutError("stalled")

        backoff = _InstantBackoff()
        with pytest.raises(TimeoutError):
            dl._download_with_retries(always_stall, "f.bin", backoff)
        assert len(backoff.waits) == dl._MAX_ATTEMPTS - 1

    @pytest.mark.parametrize("status", [429, 502, 503, 504])
    def test_temporary_status_waits_its_retry_after(self, tmp_data_dir, origin, status):
        origin.errors.append((status, {"Retry-After": "7"}))
        backoff = _InstantBackoff()
        dest = tmp_data_dir / "f.bin"
        _fetch(f"{origin.url}/f", dest, backoff)
        assert dest.read_bytes() == _GGML
        assert backoff.waits == [7]

    def test_retry_after_may_be_an_http_date(self, tmp_data_dir, origin, monkeypatch):
        # A fixed clock makes the wait exact, however long the request takes.
        now = 1_800_000_000.0
        monkeypatch.setattr(time, "time", lambda: now)
        when = email.utils.formatdate(now + 30, usegmt=True)
        origin.errors.append((503, {"Retry-After": when}))
        backoff = _InstantBackoff()
        dest = tmp_data_dir / "f.bin"
        _fetch(f"{origin.url}/f", dest, backoff)
        assert dest.read_bytes() == _GGML
        assert backoff.waits == [30]

    def test_temporary_status_without_retry_after_backs_off(self, tmp_data_dir, origin):
        origin.errors.append((503, {}))
        backoff = _InstantBackoff()
        dest = tmp_data_dir / "f.bin"
        _fetch(f"{origin.url}/f", dest, backoff)
        assert dest.read_bytes() == _GGML
        assert backoff.waits == [dl._RETRY_BACKOFF_S]

    def test_retry_after_at_the_cap_is_waited_out(self, tmp_data_dir, origin):
        origin.errors.append((429, {"Retry-After": str(dl._MAX_RETRY_AFTER_S)}))
        backoff = _InstantBackoff()
        dest = tmp_data_dir / "f.bin"
        _fetch(f"{origin.url}/f", dest, backoff)
        assert dest.read_bytes() == _GGML
        assert backoff.waits == [dl._MAX_RETRY_AFTER_S]

    def test_retry_after_past_the_cap_fails_at_once(self, tmp_data_dir, origin):
        # A retry sooner than the server asked would be refused again.
        after = str(dl._MAX_RETRY_AFTER_S + 1)
        origin.errors.append((429, {"Retry-After": after}))
        backoff = _InstantBackoff()
        with pytest.raises(urllib.error.HTTPError, match="429"):
            _fetch(f"{origin.url}/f", tmp_data_dir / "f.bin", backoff)
        assert backoff.waits == []
        assert origin.requests == 1

    @pytest.mark.parametrize("status", [403, 404, 500])
    def test_other_statuses_are_final(self, tmp_data_dir, origin, status):
        origin.errors.append((status, {"Retry-After": "1"}))
        backoff = _InstantBackoff()
        with pytest.raises(urllib.error.HTTPError, match=str(status)):
            _fetch(f"{origin.url}/f", tmp_data_dir / "f.bin", backoff)
        assert backoff.waits == []
        assert origin.requests == 1

    def test_the_manifest_retries_like_the_assets(
        self, tmp_data_dir, origin, monkeypatch
    ):
        # The manifest is the first request of every download, so a rate limit
        # usually meets it before any asset.
        name = dl.ggml_name("tiny")
        manifest = json.dumps({name: hashlib.sha256(_GGML).hexdigest()}).encode()
        origin.files = {dl.MANIFEST_FILENAME: manifest, name: _GGML}
        origin.errors.append((503, {"Retry-After": "3"}))
        monkeypatch.setattr(dl, "RELEASE_BASE_URL", origin.url)
        monkeypatch.setattr(dl, "_require_verification", lambda: True)
        backoff = _InstantBackoff()
        dl.download_model(tmp_data_dir, "tiny", "CPU", cancel=backoff)
        assert (tmp_data_dir / name).read_bytes() == _GGML
        assert backoff.waits == [3]

    def test_integrity_failure_is_not_retried(self, tmp_data_dir, monkeypatch):
        # A complete-but-wrong file fails verification once; that's not a network
        # fault, so the download is not retried.
        calls = {"n": 0}

        def wrong_bytes(url, dest, on_progress=None, cancel=None):
            calls["n"] += 1
            dest.write_bytes(b"corrupt")

        monkeypatch.setattr(dl, "_download_file", wrong_bytes)
        monkeypatch.setattr(dl, "_fetch_manifest", lambda tag, **_: _fake_manifest())
        with pytest.raises(dl.IntegrityError):
            dl.download_model(tmp_data_dir, "small", "CPU")
        assert calls["n"] == 1


class TestFetchManifest:
    def test_parses_json(self, monkeypatch):
        body = json.dumps({"f": "h"}).encode()
        monkeypatch.setattr(
            dl.urllib.request,
            "urlopen",
            lambda url, timeout=None: _FakeResponse(200, {}, body),
        )
        assert dl._fetch_manifest(
            "v0.1.0", required=True, cancel=threading.Event()
        ) == {"f": "h"}

    def test_passes_stall_timeout(self, monkeypatch):
        captured = {}

        def fake_urlopen(url, timeout=None):
            captured["timeout"] = timeout
            return _FakeResponse(200, {}, b"{}")

        monkeypatch.setattr(dl.urllib.request, "urlopen", fake_urlopen)
        dl._fetch_manifest("v0.1.0", required=True, cancel=threading.Event())
        assert captured["timeout"] == dl._STALL_TIMEOUT_S

    def test_lenient_returns_empty_on_error(self, monkeypatch):
        monkeypatch.setattr(
            dl.urllib.request,
            "urlopen",
            lambda url, timeout=None: (_ for _ in ()).throw(OSError("no net")),
        )
        assert (
            dl._fetch_manifest("v0.1.0", required=False, cancel=threading.Event()) == {}
        )

    def test_a_malformed_proxy_setting_never_reaches_the_log(
        self, malformed_proxy, caplog
    ):
        cancel = threading.Event()
        assert dl._fetch_manifest("v0.1.0", required=False, cancel=cancel) == {}
        assert caplog.messages[-1] == "No download manifest for v0.1.0 (InvalidURL)"
        assert not any("s3cret" in message for message in caplog.messages)

    @pytest.mark.parametrize("body", [b"{not json", b"[1, 2]", b'{"f": 1}'])
    def test_a_malformed_manifest_is_an_integrity_failure(self, monkeypatch, body):
        monkeypatch.setattr(
            dl.urllib.request,
            "urlopen",
            lambda url, timeout=None: _FakeResponse(200, {}, body),
        )
        with pytest.raises(dl.IntegrityError, match="malformed"):
            dl._fetch_manifest("v0.1.0", required=True, cancel=threading.Event())
        assert (
            dl._fetch_manifest("v0.1.0", required=False, cancel=threading.Event()) == {}
        )


class TestDownloadCancel:
    def test_download_file_aborts_when_cancelled(self, tmp_data_dir, monkeypatch):
        # Short of its length, so an ignored cancel would end in an incomplete download.
        response = _FakeResponse(200, {"Content-Length": "100"}, b"x" * 10)
        monkeypatch.setattr(
            dl.urllib.request, "urlopen", lambda req, timeout=None: response
        )
        cancel = threading.Event()
        cancel.set()  # already cancelled, so the first loop iteration aborts
        with pytest.raises(dl.DownloadCancelled):
            dl._download_file("http://x/f", tmp_data_dir / "out.bin", None, cancel)

    def test_cancel_is_not_retried(self):
        calls = {"n": 0}

        def cancelled():
            calls["n"] += 1
            raise dl.DownloadCancelled()

        backoff = _InstantBackoff()
        with pytest.raises(dl.DownloadCancelled):
            dl._download_with_retries(cancelled, "f", backoff)
        assert calls["n"] == 1  # aborted at once, not retried
        assert backoff.waits == []

    def test_cancel_during_backoff_aborts(self):
        attempts = []

        class CancelledWhileWaiting(_InstantBackoff):
            def wait(self, timeout=None):
                self.set()  # the user cancels while the retry waits
                return super().wait(timeout)

        def stall():
            attempts.append("f")
            raise TimeoutError("stalled")

        with pytest.raises(dl.DownloadCancelled):
            dl._download_with_retries(stall, "f", CancelledWhileWaiting())
        assert len(attempts) == 1  # the wait ends the download: no second attempt

    def test_download_model_cancel_cleans_tmp(self, tmp_data_dir, monkeypatch):
        # The real stream opens its partial before its first read sees the cancel.
        response = _FakeResponse(200, {"Content-Length": "100"}, b"x" * 10)
        monkeypatch.setattr(
            dl.urllib.request, "urlopen", lambda req, timeout=None: response
        )
        monkeypatch.setattr(dl, "_fetch_manifest", lambda tag, **_: _fake_manifest())
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(dl.DownloadCancelled):
            dl.download_model(tmp_data_dir, "small", "CPU", cancel=cancel)
        assert not (tmp_data_dir / "ggml-small-q8_0.bin").exists()
        assert not (tmp_data_dir / "ggml-small-q8_0.bin.tmp").exists()
