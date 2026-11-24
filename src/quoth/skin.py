# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""App skin model, TOML loader, bundled presets and the user skin shelf.

A skin is a colour-only `.toml` file: one palette of app surface roles per light
and dark mode, plus optional credits. The active skin is a copy of a preset or
shelf file in skin.toml, and Default is that file's absence. Every file read here
is untrusted, so reads are bounded and validated whole.
"""

from __future__ import annotations

import re
import stat
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path

from loguru import logger

from quoth import atomic_write
from quoth.atomic_write import diagnostic

# Files past the cap are ignored, so a runaway directory cannot flood the picker.
_USER_SKIN_CAP = 50
_USER_PREFIX = "user:"
_OPAQUE_COLOR = re.compile(r"#[0-9A-Fa-f]{6}\Z")

_META_NAME_MAX = 60
_META_AUTHOR_MAX = 60
_META_DESCRIPTION_MAX = 200


@dataclass(frozen=True)
class SkinPalette:
    """One mode's lowercase `#rrggbb` colours; a None role keeps Adwaita's own.

    The one derived role: in dark mode a None `card` becomes a wash of `text`.
    """

    window: str | None
    view: str | None
    card: str | None
    popover: str | None
    text: str | None
    accent: str
    on_accent: str


@dataclass(frozen=True)
class SkinMeta:
    """Who made a skin and what it is; all optional, display-only."""

    name: str | None = None
    author: str | None = None
    description: str | None = None


_DEFAULT_LIGHT = SkinPalette(
    window=None,
    view=None,
    card=None,
    popover=None,
    text=None,
    accent="#0e0e11",
    on_accent="#f2f0ea",
)
_DEFAULT_DARK = SkinPalette(
    window="#08080a",
    view="#0e0e11",
    card=None,
    popover="#181819",
    text="#f2f0ea",
    accent="#f2f0ea",
    on_accent="#0e0e11",
)


@dataclass(frozen=True)
class Skin:
    """An app skin. Equality ignores `meta`: identity is the palettes."""

    light: SkinPalette = _DEFAULT_LIGHT
    dark: SkinPalette = _DEFAULT_DARK
    meta: SkinMeta = field(default_factory=SkinMeta, compare=False)


DEFAULT_SKIN = Skin()

_PALETTE_KEYS = frozenset(
    {"window", "view", "card", "popover", "text", "accent", "on_accent"}
)
_KNOWN_KEYS = frozenset({"kind", "schema", "light", "dark", "meta"})


# Only the active read warns: the messages name skin.toml, and the picker rereads
# every shelf skin on each render.
def _warn_unknown(
    section: str, table: dict, known: frozenset[str], active: bool
) -> None:
    unknown = [key for key in table if key not in known]
    if unknown and active:
        logger.warning(
            "skin.toml [{}]: {} unknown key(s) ignored: {}",
            section,
            len(unknown),
            ", ".join(diagnostic(key) for key in unknown[:3]),
        )


def _table(data: dict, name: str) -> dict:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"skin.toml [{name}] must be a table")
    return value


def _load_palette(
    data: dict, name: str, default: SkinPalette, active: bool
) -> SkinPalette:
    table = _table(data, name)
    _warn_unknown(name, table, _PALETTE_KEYS, active)
    values = {}
    for key in _PALETTE_KEYS:
        if key not in table:
            continue
        value = table[key]
        if not isinstance(value, str) or _OPAQUE_COLOR.fullmatch(value) is None:
            raise ValueError(
                f"skin.toml [{name}] {key} must be an opaque #RRGGBB color"
            )
        values[key] = value.lower()
    return replace(default, **values)


def _meta_str(table: dict, key: str, limit: int, active: bool) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        if active:
            logger.warning("skin.toml [meta]: '{}' must be a string, ignored", key)
        return None
    cleaned = "".join(ch if ch.isprintable() else " " for ch in value).strip()
    if not cleaned:
        return None
    if len(cleaned) > limit:
        if active:
            logger.warning(
                "skin.toml [meta]: '{}' longer than {} chars, truncated", key, limit
            )
        cleaned = cleaned[:limit].rstrip()
    return cleaned


def _load_meta(data: dict, active: bool) -> SkinMeta:
    table = _table(data, "meta")
    _warn_unknown("meta", table, frozenset({"name", "author", "description"}), active)
    return SkinMeta(
        name=_meta_str(table, "name", _META_NAME_MAX, active),
        author=_meta_str(table, "author", _META_AUTHOR_MAX, active),
        description=_meta_str(table, "description", _META_DESCRIPTION_MAX, active),
    )


def _skin_from_data(data: dict, *, active: bool) -> Skin:
    if data.get("kind") != "skin":
        raise ValueError(
            "App skins must set kind = 'skin'. "
            "For overlay themes, use Overlay theme → Import."
        )
    schema = data.get("schema", 1)
    if type(schema) is not int or schema != 1:
        raise ValueError("Only app skin schema = 1 is supported")
    _warn_unknown("skin", data, _KNOWN_KEYS, active)
    return Skin(
        light=_load_palette(data, "light", _DEFAULT_LIGHT, active),
        dark=_load_palette(data, "dark", _DEFAULT_DARK, active),
        meta=_load_meta(data, active),
    )


def _read_skin_file(path: Path, *, active: bool = False) -> tuple[bytes, Skin]:
    content, data = atomic_write.read_toml(path, "skin")
    return content, _skin_from_data(data, active=active)


def _read_shared(path: Path) -> tuple[bytes, Skin]:
    """Read a picked, bundled or shelved skin, raising ValueError on any failure.

    An unreadable file is a rejection like an invalid one; only a failed write
    afterwards leaves the outcome uncertain.
    """
    try:
        return _read_skin_file(path)
    except OSError as e:
        raise ValueError(f"Couldn't read the skin file ({e.strerror or e})") from e


def load_skin(path: Path) -> Skin:
    """Load the active skin, raising when it is invalid. A missing file is Default."""
    try:
        skin = _read_skin_file(path, active=True)[1]
    except FileNotFoundError:
        return DEFAULT_SKIN
    logger.debug("Skin loaded from {}", path)
    return skin


_PRESET_DIR = resources.files("quoth") / "skins"


def list_presets() -> list[str]:
    """Return the bundled preset ids, alphabetical.

    Empty if the package data is unreadable, so a broken install costs only the
    picker.
    """
    try:
        names = [
            entry.name.removesuffix(".toml")
            for entry in _PRESET_DIR.iterdir()
            if entry.name.endswith(".toml")
        ]
    except OSError as e:
        logger.warning("Bundled skins unreadable at {} ({})", _PRESET_DIR, e)
        return []
    return sorted(names)


def user_skin_dir(config_dir: Path) -> Path:
    """Return the user skin shelf inside `config_dir`."""
    return config_dir / "skins"


def _skin_path(skin_id: str, user_dir: Path) -> Path:
    """Return the file behind a preset id or a 'user:<stem>' shelf id.

    The prefix keeps a shelf nord.toml from shadowing the preset.
    """
    if skin_id.startswith(_USER_PREFIX):
        stem = skin_id.removeprefix(_USER_PREFIX)
        if not stem or "/" in stem or "\0" in stem:
            raise ValueError("Invalid user skin id")
        return user_dir / f"{stem}.toml"
    if skin_id in list_presets():
        return Path(_PRESET_DIR / f"{skin_id}.toml")
    raise ValueError(f"Unknown skin id: {skin_id!r}")


def skin_from_id(skin_id: str, user_dir: Path) -> Skin:
    """Parse a picker id into a Skin: Default, a bundled preset or a shelf file."""
    if skin_id == "default":
        return DEFAULT_SKIN
    return _read_skin_file(_skin_path(skin_id, user_dir))[1]


def list_user_skins(user_dir: Path) -> list[str]:
    """List bounded regular shelf files, alphabetical and capped, without parsing.

    A missing or unreadable directory is an empty shelf.
    """
    try:
        entries = sorted(
            (entry for entry in user_dir.iterdir() if entry.suffix == ".toml"),
            key=lambda entry: entry.stem,
        )
    except FileNotFoundError:
        return []
    except OSError as e:
        logger.warning("User skin dir {} unreadable ({})", user_dir, e)
        return []
    stems = []
    for entry in entries:
        try:
            entry_stat = entry.stat()
        except OSError as e:
            logger.debug("Skin {} unreadable ({})", diagnostic(entry.name), e)
            continue
        if stat.S_ISREG(entry_stat.st_mode) and (
            entry_stat.st_size <= atomic_write.TOML_MAX_BYTES
        ):
            stems.append(entry.stem)
    if len(stems) > _USER_SKIN_CAP:
        logger.warning(
            "User skin dir {} holds {} skins; showing the first {}",
            user_dir,
            len(stems),
            _USER_SKIN_CAP,
        )
    return [_USER_PREFIX + stem for stem in stems[:_USER_SKIN_CAP]]


def identify_skin(path: Path, user_dir: Path) -> str | None:
    """Return which known skin `path` holds: "default", a preset or a shelf id.

    Parsed palettes are compared, so comments and credits do not matter, and a
    preset wins a tie with its shelf copy. None means the file is the user's own or
    unreadable.
    """
    try:
        current = _read_skin_file(path)[1]
    except FileNotFoundError:
        return "default"
    except (ValueError, OSError):
        return None
    for skin_id in ["default", *list_presets(), *list_user_skins(user_dir)]:
        try:
            if skin_from_id(skin_id, user_dir) == current:
                return skin_id
        except (ValueError, OSError):
            continue
    return None


def import_user_skin(source: Path, user_dir: Path) -> str:
    """Validate and copy one shared skin onto the user's shelf; return its new id.

    Raises ValueError when the skin is invalid or the shelf is full, and OSError
    when the copy cannot be written; a name clash gets a numeric suffix.
    """
    content, _skin = _read_shared(source)
    if len(list_user_skins(user_dir)) >= _USER_SKIN_CAP:
        raise ValueError("Your skin shelf is full (50 skins maximum)")
    saved_stem = atomic_write.create_named(
        user_dir, source.name, content, fallback="skin"
    )
    destination = user_dir / f"{saved_stem}.toml"
    skin_id = _USER_PREFIX + saved_stem
    logger.info("Imported user skin {} to {}", diagnostic(skin_id), destination)
    return skin_id


def apply_skin_id(skin_id: str, user_dir: Path, path: Path) -> None:
    """Copy the skin behind a picker id over `path`, or remove `path` for Default."""
    if skin_id == "default":
        atomic_write.remove(path)
    else:
        content, _skin = _read_shared(_skin_path(skin_id, user_dir))
        atomic_write.replace_bytes(path, content)
    logger.info("Applied skin {} to {}", diagnostic(skin_id), path)
