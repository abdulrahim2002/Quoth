# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Atomic persistence, bounded readers for state and TOML files, and their log form.

Every writer syncs the file contents and then its directory entry; `create_bytes`
and `create_named` refuse to replace an existing name.

No GI, so the modules that persist state (stats, calibration, update checks,
themes, skins) stay importable and testable without a display.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import tomllib
from pathlib import Path

from loguru import logger

# Far above any real theme or skin and far below what could stall the parser or
# balloon memory; larger files are refused whole.
TOML_MAX_BYTES = 256 * 1024


def sync_directory(path: Path) -> None:
    """Persist the entries of directory `path`: a create, rename or unlink inside it."""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _ensure_directory(path: Path) -> None:
    """Create a directory tree and persist each new entry before descending."""
    if path.is_dir():
        # Also repairs a prior attempt whose mkdir landed but parent fsync failed.
        sync_directory(path.parent)
        return
    _ensure_directory(path.parent)
    path.mkdir(exist_ok=True)
    sync_directory(path.parent)


def create_bytes(path: Path, data: bytes) -> None:
    """Create `path` durably without replacing an existing directory entry."""
    _ensure_directory(path.parent)
    output = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    tmp = Path(output.name)
    try:
        with output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    sync_directory(path.parent)


def create_named(directory: Path, name: str, data: bytes, *, fallback: str) -> str:
    """Create a new `.toml` file in `directory` named after `name`; return its stem.

    The stem keeps ASCII letters, digits, `_` and `-`, turns other runs into `-`, trims
    `-` and `_` from its ends and stops at 80 characters, or is `fallback` if empty. A
    clash gets a numeric suffix rather than replacing a file.
    """
    raw = name[:-5] if name.lower().endswith(".toml") else Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", raw).strip("-_")[:80].rstrip("-_")
    stem = stem or fallback
    suffix = 1
    while True:
        suffix_text = "" if suffix == 1 else f"-{suffix}"
        saved = stem[: 80 - len(suffix_text)] + suffix_text
        destination = directory / f"{saved}.toml"
        try:
            create_bytes(destination, data)
        except FileExistsError as error:
            if error.filename2 != str(destination):
                raise
            suffix += 1
            continue
        return saved


def read_toml(path: Path, noun: str) -> tuple[bytes, dict]:
    """Read one regular TOML file of at most `TOML_MAX_BYTES`: its bytes and its table.

    The non-blocking open and type check keep a FIFO or device from hanging the read.
    Raises ValueError naming `noun` for an unusable file, OSError when it can't be read.
    """
    too_large = f"{noun.capitalize()} files must be 256 KiB or smaller"
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"Choose a regular file to use as a {noun}")
        if file_stat.st_size > TOML_MAX_BYTES:
            raise ValueError(too_large)
        with os.fdopen(fd, "rb", closefd=False) as file:
            content = file.read(TOML_MAX_BYTES + 1)
    finally:
        os.close(fd)
    if len(content) > TOML_MAX_BYTES:
        raise ValueError(too_large)
    try:
        return content, tomllib.loads(content.decode("utf-8"))
    except UnicodeDecodeError:
        raise ValueError(f"The {noun} file must use UTF-8 text") from None
    except (ValueError, RecursionError, OverflowError):
        raise ValueError(f"The {noun} file is not valid TOML") from None


def replace_bytes(path: Path, data: bytes, *, mode: int | None = None) -> None:
    """Replace `path` durably with exact bytes."""
    _ensure_directory(path.parent)
    output = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    tmp = Path(output.name)
    try:
        with output:
            output.write(data)
            output.flush()
            if mode is not None:
                os.fchmod(output.fileno(), mode)
            os.fsync(output.fileno())
        os.replace(tmp, path)
        sync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def remove(path: Path) -> None:
    """Remove `path` durably; a missing path is already removed."""
    try:
        path.unlink()
    except FileNotFoundError:
        return
    sync_directory(path.parent)


def write_text(path: Path, text: str, *, mode: int | None = None) -> None:
    """Replace `path` durably with `text` encoded as UTF-8, as `replace_bytes` does."""
    replace_bytes(path, text.encode(), mode=mode)


def read_state(path: Path, label: str) -> dict | None:
    """Read a JSON object from the state dir, best-effort.

    Return None when the file is absent (expected on first use), or with a warning
    naming `label` when it is unreadable or malformed.
    """
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        logger.warning("{} unreadable ({}); starting fresh", label, e)
        return None
    if not isinstance(data, dict):
        logger.warning("{} malformed; starting fresh", label)
        return None
    return data


# The longest value a log line repeats from a file, so a file cannot flood the log.
_LOG_MAX = 60


def diagnostic(value: object) -> str:
    """Return a short repr of `value` for a log line.

    A list or table shows only its size.
    """
    if isinstance(value, str):
        text = repr(value[:_LOG_MAX])
        return text[:_LOG_MAX] + (
            "..." if len(text) > _LOG_MAX or len(value) > _LOG_MAX else ""
        )
    if isinstance(value, (list, dict)):
        return f"{type(value).__name__} with {len(value)} items"
    text = repr(value)
    return text[:_LOG_MAX] + ("..." if len(text) > _LOG_MAX else "")
