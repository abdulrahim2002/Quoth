# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Quoth's settings, kept in GSettings."""

import os
from collections.abc import Callable
from pathlib import Path

import gi

gi.require_version("Gio", "2.0")
from gi.repository import Gio, GLib  # noqa: E402
from loguru import logger  # noqa: E402

from quoth import APP_ID  # noqa: E402

# A source checkout compiles the schema beside its XML (`make schemas`); packages
# install it system-wide.
_SOURCE_SCHEMAS = Path(__file__).resolve().parents[2] / "data"

# The Python type each schema type must arrive as. PyGObject would otherwise turn any
# object into a boolean, and bool is an int subclass.
_PY_TYPES = {"b": bool, "s": str, "u": int}


def open_settings() -> Gio.Settings:
    """Open the app's settings, raising RuntimeError when the schema is missing.

    GSettings aborts on a missing schema, so it is looked up first. A source checkout's
    compiled copy outranks every installed one, which may be older.
    """
    source = Gio.SettingsSchemaSource.get_default()
    origin = "installed"
    if (_SOURCE_SCHEMAS / "gschemas.compiled").exists():
        source = Gio.SettingsSchemaSource.new_from_directory(
            str(_SOURCE_SCHEMAS), source, False
        )
        origin = str(_SOURCE_SCHEMAS)
    schema = source.lookup(APP_ID, True) if source else None
    if schema is None:
        raise RuntimeError(
            f"Settings schema {APP_ID} is not installed; in a source checkout, "
            "run `make schemas`"
        )
    settings = Gio.Settings.new_full(schema, None, None)
    backend = settings.props.backend.__gtype__.name
    logger.debug("Settings schema: {}; backend: {}", origin, backend)
    # GIO falls back to memory when it finds no dconf, and every change is then lost at
    # exit. A memory backend asked for by name (tests, a throwaway run) is intended.
    if (
        backend == "GMemorySettingsBackend"
        and os.environ.get("GSETTINGS_BACKEND") != "memory"
    ):
        logger.warning("Settings are kept in memory only: changes will not persist")
    return settings


class Config:
    """Quoth's settings by attribute name (the schema's hyphens as underscores).

    GSettings keeps only what the user changed, checks every write against the schema
    (`data/dev.quoth.Quoth.gschema.xml`, the single source of keys, defaults and valid
    values) and emits `changed` for every writer. A wrong type or an invalid value
    raises instead of being stored. Writers only write: whoever shows or applies a
    setting follows it through `connect_changed` or `bind`.
    """

    def __init__(self, settings: Gio.Settings | None = None) -> None:
        object.__setattr__(self, "settings", settings or open_settings())

    def default(self, name: str):
        return self.settings.get_default_value(self._key(name)).unpack()

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return self.settings.get_value(self._key(name)).unpack()

    def __setattr__(self, name: str, value) -> None:
        key, variant = self._variant(name, value)
        # A write of what the key already reads is skipped: a rewrite emits `changed` on
        # the keyfile backend but not on dconf, and a row echoing a reset would pin the
        # default as the user's own value.
        if self.settings.get_value(key) != variant:
            self.settings.set_value(key, variant)

    def seed(self, name: str, value) -> None:
        """Store `value` as the user's own unless they already have one.

        Unlike an assignment it stores a value equal to the default too, so a key seeded
        once is never seeded again.
        """
        key, variant = self._variant(name, value)
        if self.settings.get_user_value(key) is None:
            self.settings.set_value(key, variant)

    def connect_changed(self, callback: Callable[[str], None]) -> int:
        """Call `callback(name)` after a setting changes, whichever process wrote it.

        Return the handler id for `settings.disconnect`.
        """
        return self.settings.connect(
            "changed", lambda _settings, key: callback(key.replace("-", "_"))
        )

    def bind(self, name: str, obj, prop: str) -> None:
        """Keep `obj.prop` and the setting equal both ways, until `obj` is finalised.

        The widget is also insensitive while the key is locked down.
        """
        self.settings.bind(self._key(name), obj, prop, Gio.SettingsBindFlags.DEFAULT)

    def _variant(self, name: str, value) -> tuple[str, GLib.Variant]:
        key = self._key(name)
        schema_key = self.settings.props.settings_schema.get_key(key)
        type_string = schema_key.get_value_type().dup_string()
        expected = _PY_TYPES[type_string]
        if type(value) is not expected:
            raise TypeError(
                f"{name} must be {expected.__name__}, not {type(value).__name__}"
            )
        variant = GLib.Variant(type_string, value)
        if not schema_key.range_check(variant):
            raise ValueError(f"{value!r} is not a valid {name}")
        return key, variant

    def _key(self, name: str) -> str:
        key = name.replace("_", "-")
        if not self.settings.props.settings_schema.has_key(key):
            raise AttributeError(f"Config has no field '{name}'")
        return key
