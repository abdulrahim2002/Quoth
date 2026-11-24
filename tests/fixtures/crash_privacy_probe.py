# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Run by `test_crash_reports`: the SDK lifecycle and what a report may carry."""

import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import sentry_sdk.client
import sentry_sdk.spotlight
from loguru import logger
from sentry_sdk.attachments import Attachment
from sentry_sdk.transport import Transport

from quoth import crash_reports, log
from quoth.scheduler import guarded
from quoth.whisper import open_context


class MemoryTransport(Transport):
    def __init__(self, options):
        super().__init__(options)
        self.envelopes = []

    def capture_envelope(self, envelope):
        self.envelopes.append(envelope)


transports = []


def make_transport(options):
    transport = MemoryTransport(options)
    transports.append(transport)
    return transport


def reject_spotlight(_url):
    raise AssertionError("Spotlight must stay disabled")


def fail(message):
    sentinel_local = "private-local-" + message
    raise RuntimeError(message + sentinel_local)


def missing_library(*_args, **_kwargs):
    raise OSError("private-library-missing")


def events():
    return [
        event
        for transport in transports
        for envelope in transport.envelopes
        if (event := envelope.get_event()) is not None
    ]


sentry_sdk.client.make_transport = make_transport
sentry_sdk.spotlight.SpotlightClient = reject_spotlight
assert os.environ["SENTRY_SPOTLIGHT"] == "http://127.0.0.1:65535/stream"
crash_reports.DSN = "https://public@example.invalid/1"
crash_reports._enabled = False
crash_reports.note_device("NPU")
log.install_crash_hooks()
logentries = []
prepare = crash_reports._prepare
crash_reports._prepare = lambda event, hint: (
    logentries.append(json.dumps(event.get("logentry"))) or prepare(event, hint)
)

crash_reports.set_enabled(False)
assert transports == []
crash_reports.set_enabled(True)
crash_reports.set_enabled(True)
assert sentry_sdk.get_client().options["spotlight"] is False
assert sentry_sdk.get_client().spotlight is None

try:
    fail("private-handled-")
except RuntimeError:
    logger.exception("private-log-message", private_extra="private-extra")
# The SDK drops the message anyway, so the traceback is never rendered into it.
assert "Traceback" not in logentries[0], logentries
logger.info("private-info")
logger.trace("private-trace")
logger.error("private-plain-error")

try:
    fail("private-main-")
except RuntimeError:
    sys.excepthook(*sys.exc_info())

thread = threading.Thread(target=lambda: fail("private-thread-"))
thread.start()
thread.join()
guarded(lambda: fail("private-callback-"))()

assert len(events()) == 4, events()
sentry_sdk.capture_event(
    {"exception": {"values": [{"type": "Synthetic", "value": "private-sdk-value"}]}},
    hint={
        "attachments": [
            Attachment(bytes=b"private-sdk-attachment", filename="private.txt")
        ]
    },
)
assert len(events()) == 5
assert all(
    item.type != "attachment"
    for envelope in transports[-1].envelopes
    for item in envelope.items
)
crash_reports.set_enabled(False)
logger.exception("private-off")
assert len(events()) == 5

crash_reports.set_enabled(True)
try:
    fail("private-second-cycle-")
except RuntimeError:
    logger.exception("private-second-log")
assert len(events()) == 6, events()

previous = Path(log.previous_crash_log)
previous.write_text(f'File "{Path.home()}/private/native.py", line 1\n')
crash_reports.report_native_crash()
assert not previous.exists()
assert len(events()) == 7

native_envelope = transports[-1].envelopes[-1]
native = native_envelope.get_event()
assert native["exception"]["values"] == [{"type": "NativeCrash"}]
attachment = next(item for item in native_envelope.items if item.type == "attachment")
attachment_bytes = attachment.payload.get_bytes()
assert b"private/native.py" in attachment_bytes
assert str(Path.home()).encode() not in attachment_bytes
item_types = [
    item.type
    for transport in transports
    for envelope in transport.envelopes
    for item in envelope.items
]
assert item_types.count("event") == 7
assert item_types.count("attachment") == 1
assert set(item_types) == {"event", "attachment"}

payload = json.dumps(events())
assert str(Path.home()) not in payload
# What triage needs survives the stripping: the type, the mechanism and the frames.
handled = events()[0]["exception"]["values"][-1]
assert handled["type"] == "RuntimeError"
assert handled["mechanism"]
assert [frame["function"] for frame in handled["stacktrace"]["frames"]] == [
    "<module>",
    "fail",
]
for private in (
    "private-handled",
    "private-main",
    "private-thread",
    "private-callback",
    "private-second-cycle",
    "private-log-message",
    "private-extra",
    "private-info",
    "private-trace",
    "private-plain-error",
    "private-sdk-value",
    "private-sdk-attachment",
    "sentinel_local",
):
    assert private not in payload, private
for event in events():
    assert event["tags"]["device"] == "NPU"
    assert {"distro", "desktop", "install"} <= event["tags"].keys()
    assert not event.get("server_name")
    assert "message" not in event
    assert "logentry" not in event
    assert "extra" not in event
    assert "contexts" not in event or "trace" not in event["contexts"]
    for exception in event["exception"]["values"]:
        assert "value" not in exception
        for frame in exception.get("stacktrace", {}).get("frames", []):
            assert (
                not {"vars", "pre_context", "context_line", "post_context"}
                & frame.keys()
            )

reported = len(events())
for _ in range(3):
    try:
        fail("private-repeat-")
    except RuntimeError:
        logger.exception("private-repeat")
sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
assert len(events()) == reported + 1, events()

# A broken install (a library missing or unloadable) reaches a report, once a run.
models = SimpleNamespace(model_path=lambda _model: Path("small.bin"), cache_dir="")
for _ in range(2):
    assert open_context("small", "CPU", models, factory=missing_library) is None
assert len(events()) == reported + 2, events()
assert events()[-1]["exception"]["values"][-1]["type"] == "OSError"
assert "private-library" not in json.dumps(events())

crash_reports.set_enabled(False)
