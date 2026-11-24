# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Overlay theme model and TOML loader.

A Theme is an immutable tree of frozen dataclasses whose defaults reproduce
Quoth's built-in flat look. `read_theme()` overlays the user's theme.toml on those
defaults: an invalid value falls back per key, and an unreadable file raises so the
app can keep the current look, because a broken theme must never break dictation.

Bundled presets (package data) and the user's theme shelf back the Settings picker
through `list_presets`, `list_user_themes`, `apply_theme_id` and `identify_theme`,
with no second source of truth: the active file on disk is always the answer.

Pure data, no GTK imports, so everything here is headless-testable.
"""

from __future__ import annotations

import math
import re
import stat
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from loguru import logger

from quoth import atomic_write, paths
from quoth.atomic_write import diagnostic

Color = tuple[float, float, float, float]
_HEX_COLOR = re.compile(r"#(?:[0-9A-Fa-f]{2}){3,4}")

# Anchor token -> which screen edges the layer surface anchors to.
# One edge = centred on that edge; two perpendicular = that corner;
# none = centred on screen. Margins only apply to anchored edges.
_ANCHOR_EDGES: dict[str, frozenset[str]] = {
    "top-left": frozenset({"top", "left"}),
    "top-center": frozenset({"top"}),
    "top-right": frozenset({"top", "right"}),
    "center-left": frozenset({"left"}),
    "center": frozenset(),
    "center-right": frozenset({"right"}),
    "bottom-left": frozenset({"bottom", "left"}),
    "bottom-center": frozenset({"bottom"}),
    "bottom-right": frozenset({"bottom", "right"}),
}

VALID_ANCHORS = frozenset(_ANCHOR_EDGES)
DEFAULT_ANCHOR = "bottom-center"

# The Adw.Easing curves that suit short transitions.
VALID_EASINGS = frozenset(
    {
        "linear",
        "ease-in-sine",
        "ease-out-sine",
        "ease-in-out-sine",
        "ease-in-quad",
        "ease-out-quad",
        "ease-in-out-quad",
        "ease-in-cubic",
        "ease-out-cubic",
        "ease-in-out-cubic",
        "ease-in-quart",
        "ease-out-quart",
        "ease-in-out-quart",
        "ease-in-expo",
        "ease-out-expo",
        "ease-in-out-expo",
        "ease-in-circ",
        "ease-out-circ",
        "ease-in-out-circ",
        "ease-in-back",  # undershoots below 0
        "ease-out-back",  # overshoots above 1: clamp alpha at call sites
        "ease-in-out-back",
    }
)
DEFAULT_ENTRY_EASING = "ease-out-cubic"
DEFAULT_EXIT_EASING = "ease-in-cubic"


def anchor_to_edges(anchor: str) -> frozenset[str]:
    """Map an anchor token to its anchored edges (unknown -> default)."""
    return _ANCHOR_EDGES.get(anchor, _ANCHOR_EDGES[DEFAULT_ANCHOR])


def parse_color(value: str) -> Color | None:
    """Parse '#RRGGBB' or '#RRGGBBAA' into a normalised RGBA tuple, or return None."""
    if not isinstance(value, str) or _HEX_COLOR.fullmatch(value) is None:
        return None
    channels = [byte / 255 for byte in bytes.fromhex(value[1:])]
    if len(channels) == 3:
        channels.append(1.0)
    return tuple(channels)


@dataclass(frozen=True)
class Palette:
    """Semantic colour slots (style-agnostic; opacity via alpha channel)."""

    background: Color = (0.055, 0.055, 0.067, 0.85)  # Ink
    border: Color = (0.949, 0.941, 0.918, 0.10)  # Bone hairline
    accent: Color = (0.949, 0.941, 0.918, 0.90)  # waveform line
    text: Color = (0.949, 0.941, 0.918, 0.90)  # timer + error text
    dot_recording: Color = (0.949, 0.941, 0.918, 1.0)
    dot_transcribing: Color = (0.949, 0.941, 0.918, 1.0)
    dot_error: Color = (0.800, 0.161, 0.212, 1.0)  # Tomato Jam


@dataclass(frozen=True)
class Geometry:
    width: int = 240
    height: int = 48  # corner radius is derived: height / 2
    border_width: float = 1.0
    bar_gap: int = 3  # waveform density
    line_width: float = 1.5  # waveform stroke


@dataclass(frozen=True)
class Position:
    """Layer-shell anchor + per-edge margins (top, right, bottom, left)."""

    anchor: str = DEFAULT_ANCHOR
    margin: tuple[int, int, int, int] = (60, 60, 60, 60)


@dataclass(frozen=True)
class Font:
    family: str = "sans-serif"  # fontconfig-resolved; bad names fall back
    size: int = 12


@dataclass(frozen=True)
class AnimationSpec:
    duration: float  # seconds
    easing: str


@dataclass(frozen=True)
class Animation:
    enabled: bool = True  # decorative motion; system reduced-motion overrides
    duration_scale: float = 1.0
    # Entry is snappier than exit so the reveal feels instant on a hotkey press.
    entry: AnimationSpec = field(
        default_factory=lambda: AnimationSpec(0.09, DEFAULT_ENTRY_EASING)
    )
    exit: AnimationSpec = field(
        default_factory=lambda: AnimationSpec(0.15, DEFAULT_EXIT_EASING)
    )


# Metadata comes from arbitrary theme files and shows in the Settings picker, so it
# is capped to keep a shared theme from flooding or spoofing the UI.
_META_NAME_MAX = 60
_META_AUTHOR_MAX = 60
_META_DESCRIPTION_MAX = 200


@dataclass(frozen=True)
class ThemeMeta:
    """Who made a theme and what it is; all optional, display-only."""

    name: str | None = None
    author: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class Theme:
    """A complete overlay look: palette, geometry, position, font and animation."""

    palette: Palette = field(default_factory=Palette)
    geometry: Geometry = field(default_factory=Geometry)
    position: Position = field(default_factory=Position)
    font: Font = field(default_factory=Font)
    animation: Animation = field(default_factory=Animation)
    # Excluded from equality: a preset with an added [meta] credit is still that
    # preset.
    meta: ThemeMeta = field(default_factory=ThemeMeta, compare=False)


DEFAULT_THEME = Theme()


# --- TOML loading -----------------------------------------------------------
#
# Live reads fall back per key so a half-broken theme remains usable. Strict
# reads use the same parser but reject invalid known appearance fields.


def _warn_unknown(section: str, table: dict, known: frozenset[str], strict: bool):
    # Only lenient reads warn: they read theme.toml, the file the message names,
    # while the picker repeats its strict shelf and preset reads on every render.
    unknown = [key for key in table if key not in known]
    if unknown and not strict:
        samples = ", ".join(diagnostic(key) for key in unknown[:3])
        logger.warning(
            "theme.toml [{}]: {} unknown key(s) ignored: {}",
            section,
            len(unknown),
            samples,
        )


def _invalid(strict: bool, section: str, key: str, expected: str) -> None:
    if strict:
        raise ValueError(f"Invalid theme field [{section}] {key}: expected {expected}")
    logger.warning(
        "theme.toml [{}] {}: expected {}, using default", section, key, expected
    )


def _read_color(
    table: dict, section: str, key: str, default: Color, *, strict: bool = False
) -> Color:
    if key not in table:
        return default
    color = parse_color(table[key]) if isinstance(table[key], str) else None
    if color is None:
        _invalid(strict, section, key, "#RRGGBB or #RRGGBBAA")
        return default
    return color


def _read_number(
    table: dict,
    section: str,
    key: str,
    default: float,
    lo: float,
    hi: float,
    *,
    strict: bool = False,
) -> float:
    """Read a float, accepting TOML ints (but not bools), clamped to [lo, hi]."""
    if key not in table:
        return default
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid(strict, section, key, "a finite number")
        return default
    if isinstance(value, float) and not math.isfinite(value):
        _invalid(strict, section, key, "a finite number")
        return default
    clamped = min(hi, max(lo, value))
    # Only the active load warns: strict reads are preset, shelf and import checks.
    if clamped != value and not strict:
        logger.warning(
            "theme.toml [{}] {}: outside {} to {}, using {}",
            section,
            key,
            lo,
            hi,
            clamped,
        )
    return float(clamped)


def _read_int(
    table: dict,
    section: str,
    key: str,
    default: int,
    lo: int,
    hi: int,
    *,
    strict: bool = False,
) -> int:
    return int(_read_number(table, section, key, default, lo, hi, strict=strict))


def _read_bool(
    table: dict, section: str, key: str, default: bool, *, strict: bool = False
) -> bool:
    if key not in table:
        return default
    value = table[key]
    if not isinstance(value, bool):
        _invalid(strict, section, key, "true or false")
        return default
    return value


def _read_choice(
    table: dict,
    section: str,
    key: str,
    default: str,
    valid: frozenset[str],
    expected: str,
    *,
    strict: bool = False,
) -> str:
    if key not in table:
        return default
    value = table[key]
    if not isinstance(value, str) or value not in valid:
        _invalid(strict, section, key, expected)
        return default
    return value


def _read_string(
    table: dict, section: str, key: str, default: str, *, strict: bool = False
) -> str:
    if key not in table:
        return default
    value = table[key]
    if not isinstance(value, str) or not value.strip():
        _invalid(strict, section, key, "a non-empty string")
        return default
    return value


def _read_margin(
    table: dict,
    section: str,
    key: str,
    default: tuple[int, int, int, int],
    *,
    strict: bool = False,
) -> tuple[int, int, int, int]:
    """Read a margin: one int for all edges, or a [top, right, bottom, left] list."""
    if key not in table:
        return default
    value = table[key]

    def valid_int(item: object) -> bool:
        return (
            isinstance(item, int)
            and not isinstance(item, bool)
            and -(2**31) <= item <= 2**31 - 1
        )

    if valid_int(value):
        return (value,) * 4
    if (
        isinstance(value, list)
        and len(value) == 4
        and all(valid_int(item) for item in value)
    ):
        return tuple(value)
    _invalid(
        strict,
        section,
        key,
        "a signed 32-bit integer or four signed 32-bit integers",
    )
    return default


_COLOR_KEYS = frozenset(
    {
        "background",
        "border",
        "accent",
        "text",
        "dot_recording",
        "dot_transcribing",
        "dot_error",
    }
)
_GEOMETRY_KEYS = frozenset({"width", "height", "border_width", "bar_gap", "line_width"})
_POSITION_KEYS = frozenset({"anchor", "margin"})
_FONT_KEYS = frozenset({"family", "size"})
_ANIMATION_VALUE_KEYS = frozenset({"enabled", "duration_scale"})
_ANIMATION_KEYS = _ANIMATION_VALUE_KEYS | {"entry", "exit"}
_SPEC_KEYS = frozenset({"duration", "easing"})
_META_KEYS = frozenset({"name", "author", "description"})


def _load_palette(table: dict, *, strict: bool = False) -> Palette:
    d = Palette()
    _warn_unknown("colors", table, _COLOR_KEYS, strict)
    return Palette(
        background=_read_color(
            table, "colors", "background", d.background, strict=strict
        ),
        border=_read_color(table, "colors", "border", d.border, strict=strict),
        accent=_read_color(table, "colors", "accent", d.accent, strict=strict),
        text=_read_color(table, "colors", "text", d.text, strict=strict),
        dot_recording=_read_color(
            table, "colors", "dot_recording", d.dot_recording, strict=strict
        ),
        dot_transcribing=_read_color(
            table,
            "colors",
            "dot_transcribing",
            d.dot_transcribing,
            strict=strict,
        ),
        dot_error=_read_color(table, "colors", "dot_error", d.dot_error, strict=strict),
    )


def _load_geometry(table: dict, *, strict: bool = False) -> Geometry:
    d = Geometry()
    _warn_unknown("geometry", table, _GEOMETRY_KEYS, strict)
    height = _read_int(table, "geometry", "height", d.height, 24, 200, strict=strict)
    return Geometry(
        # width >= height so the derived pill radius (height/2) can't invert
        width=_read_int(
            table, "geometry", "width", d.width, height, 2000, strict=strict
        ),
        height=height,
        border_width=_read_number(
            table,
            "geometry",
            "border_width",
            d.border_width,
            0.0,
            20.0,
            strict=strict,
        ),
        bar_gap=_read_int(
            table, "geometry", "bar_gap", d.bar_gap, 0, 50, strict=strict
        ),
        line_width=_read_number(
            table,
            "geometry",
            "line_width",
            d.line_width,
            0.1,
            20.0,
            strict=strict,
        ),
    )


def _load_position(table: dict, *, strict: bool = False) -> Position:
    d = Position()
    _warn_unknown("position", table, _POSITION_KEYS, strict)
    return Position(
        anchor=_read_choice(
            table,
            "position",
            "anchor",
            d.anchor,
            VALID_ANCHORS,
            "a supported anchor",
            strict=strict,
        ),
        margin=_read_margin(table, "position", "margin", d.margin, strict=strict),
    )


def _load_font(table: dict, *, strict: bool = False) -> Font:
    d = Font()
    _warn_unknown("font", table, _FONT_KEYS, strict)
    return Font(
        family=_read_string(table, "font", "family", d.family, strict=strict),
        size=_read_int(table, "font", "size", d.size, 6, 72, strict=strict),
    )


def _load_spec(
    table: dict, section: str, default: AnimationSpec, *, strict: bool = False
) -> AnimationSpec:
    _warn_unknown(section, table, _SPEC_KEYS, strict)
    return AnimationSpec(
        duration=_read_number(
            table,
            section,
            "duration",
            default.duration,
            0.0,
            5.0,
            strict=strict,
        ),
        easing=_read_choice(
            table,
            section,
            "easing",
            default.easing,
            VALID_EASINGS,
            "a supported easing name",
            strict=strict,
        ),
    )


def _load_animation(table: dict, *, strict: bool = False) -> Animation:
    d = Animation()
    _warn_unknown("animation", table, _ANIMATION_KEYS, strict)
    entry = table.get("entry", {})
    exit_ = table.get("exit", {})
    if strict and "entry" in table and not isinstance(entry, dict):
        raise ValueError("Invalid theme field [animation] entry: expected a table")
    if strict and "exit" in table and not isinstance(exit_, dict):
        raise ValueError("Invalid theme field [animation] exit: expected a table")
    return Animation(
        enabled=_read_bool(table, "animation", "enabled", d.enabled, strict=strict),
        duration_scale=_read_number(
            table,
            "animation",
            "duration_scale",
            d.duration_scale,
            0.1,
            10.0,
            strict=strict,
        ),
        entry=_load_spec(
            entry if isinstance(entry, dict) else {},
            "animation.entry",
            d.entry,
            strict=strict,
        ),
        exit=_load_spec(
            exit_ if isinstance(exit_, dict) else {},
            "animation.exit",
            d.exit,
            strict=strict,
        ),
    )


def _meta_str(table: dict, key: str, limit: int, strict: bool) -> str | None:
    """Return one sanitised metadata string, or None when absent or unusable.

    Non-printable characters become spaces, so multi-line values flatten; the result
    is stripped and hard-truncated to `limit`.
    """
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        if not strict:
            logger.warning("theme.toml [meta]: '{}' must be a string, ignored", key)
        return None
    cleaned = "".join(ch if ch.isprintable() else " " for ch in value).strip()
    if not cleaned:
        return None
    if len(cleaned) > limit:
        if not strict:
            logger.warning(
                "theme.toml [meta]: '{}' longer than {} chars, truncated", key, limit
            )
        cleaned = cleaned[:limit].rstrip()
    return cleaned


def _load_meta(table: dict, *, strict: bool = False) -> ThemeMeta:
    _warn_unknown("meta", table, _META_KEYS, strict)
    return ThemeMeta(
        name=_meta_str(table, "name", _META_NAME_MAX, strict),
        author=_meta_str(table, "author", _META_AUTHOR_MAX, strict),
        description=_meta_str(table, "description", _META_DESCRIPTION_MAX, strict),
    )


_KNOWN_SECTIONS = frozenset(
    {"schema", "colors", "geometry", "position", "font", "animation", "meta"}
)


def _section(data: dict, name: str, *, strict: bool = False) -> dict:
    table = data.get(name, {})
    if isinstance(table, dict):
        return table
    if strict:
        raise ValueError(f"Invalid theme section [{name}]: expected a table")
    return {}


def _has_appearance_leaf(data: dict) -> bool:
    for section, known in (
        ("colors", _COLOR_KEYS),
        ("geometry", _GEOMETRY_KEYS),
        ("position", _POSITION_KEYS),
        ("font", _FONT_KEYS),
    ):
        table = data.get(section)
        if isinstance(table, dict) and any(key in known for key in table):
            return True
    animation = data.get("animation")
    if not isinstance(animation, dict):
        return False
    if any(key in _ANIMATION_VALUE_KEYS for key in animation):
        return True
    return any(
        isinstance(animation.get(spec), dict)
        and any(key in _SPEC_KEYS for key in animation[spec])
        for spec in ("entry", "exit")
    )


def _theme_from_data(data: dict, *, strict: bool = False) -> Theme:
    """Build a Theme from parsed TOML, raising ValueError on an unsupported schema.

    `strict` also raises on an app skin, a non-table section, an invalid appearance
    field or a schema-less file with no appearance field; otherwise invalid values
    fall back per key with a warning.
    """
    if strict and data.get("kind") == "skin":
        raise ValueError("This is an app skin. Use App skin → Import instead.")
    schema = data.get("schema", 1)
    if type(schema) is not int or schema != 1:
        raise ValueError("Only overlay theme schema = 1 is supported")
    _warn_unknown("theme", data, _KNOWN_SECTIONS, strict)
    colors = _section(data, "colors", strict=strict)
    geometry = _section(data, "geometry", strict=strict)
    position = _section(data, "position", strict=strict)
    font = _section(data, "font", strict=strict)
    animation = _section(data, "animation", strict=strict)
    meta = _section(data, "meta", strict=strict)
    theme = Theme(
        palette=_load_palette(colors, strict=strict),
        geometry=_load_geometry(geometry, strict=strict),
        position=_load_position(position, strict=strict),
        font=_load_font(font, strict=strict),
        animation=_load_animation(animation, strict=strict),
        meta=_load_meta(meta, strict=strict),
    )
    if strict and "schema" not in data and not _has_appearance_leaf(data):
        raise ValueError(
            "A theme without schema = 1 must define a recognized appearance field"
        )
    return theme


def _read_theme_file(path: Path, *, strict: bool = False) -> tuple[bytes, Theme]:
    content, data = atomic_write.read_toml(path, "theme")
    return content, _theme_from_data(data, strict=strict)


def _read_shared_theme(path: Path) -> tuple[bytes, Theme]:
    """Read a picked or shelved theme strictly, raising ValueError on any failure.

    An unreadable file is a rejection like an invalid one; only a failed write
    afterwards leaves the outcome uncertain.
    """
    try:
        return _read_theme_file(path, strict=True)
    except OSError as e:
        raise ValueError(f"Couldn't read the theme file ({e.strerror or e})") from e


def read_theme(path: Path) -> Theme:
    """Read one bounded regular theme file, raising on file or validation errors."""
    return _read_theme_file(path)[1]


# --- Bundled presets --------------------------------------------------------
#
# Applying a preset copies it over the user's theme.toml, so that file stays the one
# source of the current look and keeps the preset's comments as a starting point.

_PRESET_DIR = resources.files("quoth") / "themes"


def _preset_path(name: str) -> Path:
    if not name or "/" in name or "\0" in name:
        raise ValueError("Invalid theme id")
    return Path(_PRESET_DIR / f"{name}.toml")


def list_presets() -> list[str]:
    """Return the bundled preset names: 'default' first, the rest alphabetical.

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
        logger.warning("Bundled presets unreadable at {} ({})", _PRESET_DIR, e)
        return []
    return sorted(names, key=lambda name: (name != "default", name))


def _replace_active_theme(path: Path, content: bytes) -> None:
    target = path
    if path.is_symlink():
        # A loop raises RuntimeError before Python 3.13 and resolves to a link on 3.13+.
        try:
            target = path.resolve(strict=False)
        except RuntimeError:
            pass
        if target.is_symlink():
            raise ValueError("Active theme link contains a loop")
        if not target.parent.is_dir():
            raise FileNotFoundError(target.parent)

    try:
        target_stat = target.stat()
    except FileNotFoundError:
        mode = 0o600
    else:
        if not stat.S_ISREG(target_stat.st_mode):
            raise ValueError("Active theme target must be a regular file")
        mode = stat.S_IMODE(target_stat.st_mode)

    atomic_write.replace_bytes(target, content, mode=mode)


def identify_theme(path: Path, user_dir: Path | None = None) -> str | None:
    """Return which known theme `path` holds, or None when the file is the user's own.

    The answer is a preset name, a user theme id ('user:<stem>'), or 'default' when
    there is no file. Parsed Themes are compared, so comments and [meta] credits do
    not matter; DEFAULT_THEME's rounded decimals never equal a colour parsed from
    hex, so both sides come from TOML. Presets win a tie; unreadable files go
    unclaimed.
    """
    try:
        current = read_theme(path)
    except FileNotFoundError:
        return "default"
    except (ValueError, OSError):
        return None  # unreadable or mid-edit: not a theme we can claim
    for name in list_presets():
        try:
            candidate = preset_theme(name)
        except (ValueError, OSError):
            continue
        if current == candidate:
            return name
    if user_dir is not None:
        for theme_id in list_user_themes(user_dir):
            try:
                candidate = theme_from_id(theme_id, user_dir)
            except (ValueError, OSError):
                continue
            if current == candidate:
                return theme_id
    return None


def preset_theme(name: str) -> Theme:
    """Return a bundled preset parsed into a Theme, for comparing and previewing it."""
    return _read_theme_file(_preset_path(name), strict=True)[1]


# --- User theme library -----------------------------------------------------
#
# The config dir's themes/ is the user's shelf: a .toml dropped there appears in the
# picker, and applying one copies it over theme.toml like a preset. Ids are
# namespaced 'user:<stem>' so a shelf nord.toml cannot shadow the preset.

_USER_PREFIX = "user:"
# Files past the cap are ignored, so a runaway directory cannot flood the picker.
_USER_THEME_CAP = 50


def user_theme_dir(config_dir: Path) -> Path:
    """Return the user theme shelf inside `config_dir`."""
    return config_dir / "themes"


def is_user_theme(theme_id: str) -> bool:
    return theme_id.startswith(_USER_PREFIX)


# Invalid shelf files already warned about, by path and mtime: the picker relists the
# shelf on every render, so a file warns again only once it changes.
_warned_shelf: set[tuple[Path, int | None]] = set()


def _shelf_state(entry: Path) -> tuple[Path, int | None]:
    try:
        return entry, entry.stat().st_mtime_ns
    except OSError:
        return entry, None


def list_user_themes(directory: Path) -> list[str]:
    """Return the ids ('user:<stem>') of usable shelf themes, alphabetical, capped.

    Invalid files are skipped with a warning; a missing or unreadable directory is an
    empty shelf.
    """
    try:
        entries = sorted(
            (entry for entry in directory.iterdir() if entry.suffix == ".toml"),
            key=lambda entry: entry.stem,
        )
    except FileNotFoundError:
        return []
    except OSError as e:
        logger.warning("User theme dir {} unreadable ({})", directory, e)
        return []
    stems = []
    for index, entry in enumerate(entries):
        if len(stems) == _USER_THEME_CAP:
            logger.warning(
                "User theme dir {}: showing the first {}; {} more candidate(s) not inspected",
                directory,
                _USER_THEME_CAP,
                len(entries) - index,
            )
            break
        try:
            _read_theme_file(entry, strict=True)
        except (ValueError, OSError) as error:
            state = _shelf_state(entry)
            warn = state not in _warned_shelf
            _warned_shelf.add(state)
            (logger.warning if warn else logger.debug)(
                "User theme {} ignored ({}: {})",
                diagnostic(entry.name),
                type(error).__name__,
                diagnostic(paths.tilde(str(error))),
            )
            continue
        stems.append(entry.stem)
    return [_USER_PREFIX + stem for stem in stems]


def import_user_theme(source: Path, user_dir: Path) -> str:
    """Validate and copy one shared theme onto the user's shelf; return its new id.

    Raises ValueError when the theme is invalid or the shelf is full, and OSError
    when the copy cannot be written; a name clash gets a numeric suffix.
    """
    content, _theme = _read_shared_theme(source)

    if len(list_user_themes(user_dir)) >= _USER_THEME_CAP:
        raise ValueError("Your theme shelf is full (50 themes maximum)")
    saved_stem = atomic_write.create_named(
        user_dir, source.name, content, fallback="theme"
    )
    destination = user_dir / f"{saved_stem}.toml"
    theme_id = _USER_PREFIX + saved_stem
    logger.info("Imported user theme {} to {}", diagnostic(theme_id), destination)
    return theme_id


def _user_theme_path(theme_id: str, directory: Path) -> Path:
    stem = theme_id.removeprefix(_USER_PREFIX)
    if not is_user_theme(theme_id) or not stem or "/" in stem or "\0" in stem:
        raise ValueError("Invalid user theme id")
    return directory / f"{stem}.toml"


def theme_from_id(theme_id: str, user_dir: Path) -> Theme:
    """Parse a picker id strictly into a Theme: a bundled preset or a shelf file."""
    if is_user_theme(theme_id):
        return _read_theme_file(_user_theme_path(theme_id, user_dir), strict=True)[1]
    return preset_theme(theme_id)


def apply_theme_id(theme_id: str, user_dir: Path, dst: Path) -> None:
    """Write the theme behind a picker id over `dst`, preset and shelf alike."""
    if is_user_theme(theme_id):
        content, _theme = _read_shared_theme(_user_theme_path(theme_id, user_dir))
    else:
        content, _theme = _read_theme_file(_preset_path(theme_id), strict=True)

    _replace_active_theme(dst, content)
    logger.info("Applied theme {} to {}", diagnostic(theme_id), dst)
