# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `theme`: the overlay theme model, TOML loader, presets and user shelf."""

import dataclasses
import json
import os
import stat
from importlib.resources import files
from pathlib import Path
from unittest.mock import Mock

import pytest

from quoth import atomic_write
from quoth.theme import (
    DEFAULT_ANCHOR,
    DEFAULT_ENTRY_EASING,
    DEFAULT_EXIT_EASING,
    DEFAULT_THEME,
    VALID_ANCHORS,
    VALID_EASINGS,
    Theme,
    anchor_to_edges,
    apply_theme_id,
    identify_theme,
    list_presets,
    parse_color,
    read_theme,
)


def write_theme(tmp_path, content: str):
    path = tmp_path / "theme.toml"
    path.write_text(content)
    return path


def preset_source(name: str) -> str:
    """Return a bundled preset's TOML, comments and all."""
    return (files("quoth") / "themes" / f"{name}.toml").read_text(encoding="utf-8")


class TestParseColor:
    def test_rgb(self):
        assert parse_color("#191716") == pytest.approx(
            (0x19 / 255, 0x17 / 255, 0x16 / 255, 1.0)
        )

    def test_rgba(self):
        color = parse_color("#EFEFEF1A")
        assert color[:3] == pytest.approx((0xEF / 255,) * 3)
        assert color[3] == pytest.approx(0x1A / 255)

    def test_lowercase(self):
        assert parse_color("#efefef") == parse_color("#EFEFEF")

    @pytest.mark.parametrize(
        "value",
        [
            "EFEFEF",
            "#12345",
            "#1234567",
            "#GGGGGG",
            "#-1-1-1",
            "# 1 2 3",
            "#" + "\u0661" * 6,
            123456,
        ],
        ids=[
            "missing-hash",
            "short",
            "long",
            "non-hex",
            "sign",
            "space",
            "non-ascii-digits",
            "non-string",
        ],
    )
    def test_rejects(self, value):
        assert parse_color(value) is None


class TestAnchors:
    def test_all_nine_tokens(self):
        assert anchor_to_edges("top-left") == {"top", "left"}
        assert anchor_to_edges("top-center") == {"top"}
        assert anchor_to_edges("top-right") == {"top", "right"}
        assert anchor_to_edges("center-left") == {"left"}
        assert anchor_to_edges("center") == frozenset()
        assert anchor_to_edges("center-right") == {"right"}
        assert anchor_to_edges("bottom-left") == {"bottom", "left"}
        assert anchor_to_edges("bottom-center") == {"bottom"}
        assert anchor_to_edges("bottom-right") == {"bottom", "right"}

    def test_unknown_token_falls_back_to_default(self):
        assert anchor_to_edges("middle-out") == anchor_to_edges(DEFAULT_ANCHOR)

    def test_valid_anchors_has_nine(self):
        assert len(VALID_ANCHORS) == 9


class TestEasingNames:
    def test_exposes_22_curves(self):
        assert len(VALID_EASINGS) == 22

    def test_defaults_are_valid(self):
        assert DEFAULT_ENTRY_EASING in VALID_EASINGS
        assert DEFAULT_EXIT_EASING in VALID_EASINGS

    def test_omits_toyish_families(self):
        assert not any(
            "elastic" in n or "bounce" in n or "quint" in n for n in VALID_EASINGS
        )


class TestReadTheme:
    def test_malformed_toml_raises(self, tmp_path):
        path = write_theme(tmp_path, "not [valid toml")
        with pytest.raises(ValueError, match="not valid TOML"):
            read_theme(path)

    def test_empty_file_yields_defaults(self, tmp_path):
        path = write_theme(tmp_path, "")
        assert read_theme(path) == DEFAULT_THEME

    def test_partial_override_keeps_other_defaults(self, tmp_path):
        path = write_theme(tmp_path, '[colors]\nbackground = "#000000"\n')
        theme = read_theme(path)
        assert theme.palette.background == (0.0, 0.0, 0.0, 1.0)
        assert theme.palette.accent == DEFAULT_THEME.palette.accent
        assert theme.geometry == DEFAULT_THEME.geometry

    def test_unknown_keys_ignored(self, tmp_path):
        path = write_theme(
            tmp_path,
            '[colors]\nfrobnicate = "#FF0000"\n[wormhole]\nx = 1\n',
        )
        assert read_theme(path) == DEFAULT_THEME

    def test_invalid_color_falls_back_per_key(self, tmp_path):
        path = write_theme(
            tmp_path,
            '[colors]\nbackground = "red"\naccent = "#3454D1"\n',
        )
        theme = read_theme(path)
        assert theme.palette.background == DEFAULT_THEME.palette.background
        assert theme.palette.accent == pytest.approx(
            (0x34 / 255, 0x54 / 255, 0xD1 / 255, 1.0)
        )

    def test_int_coerces_to_float(self, tmp_path):
        path = write_theme(tmp_path, "[geometry]\nborder_width = 2\n")
        theme = read_theme(path)
        assert theme.geometry.border_width == 2.0
        assert isinstance(theme.geometry.border_width, float)

    def test_bool_rejected_for_number(self, tmp_path):
        # Accepted, false would read as 0.0; true would equal the 1.0 default.
        path = write_theme(tmp_path, "[geometry]\nborder_width = false\n")
        assert read_theme(path).geometry.border_width == 1.0

    def test_geometry_clamps(self, tmp_path):
        path = write_theme(
            tmp_path,
            "[geometry]\nheight = 5\nwidth = 1\n"
            "border_width = -1\nbar_gap = -1\nline_width = 0\n",
        )
        theme = read_theme(path)
        assert theme.geometry.height == 24  # floor
        assert theme.geometry.width == 24
        assert theme.geometry.border_width == 0.0
        assert theme.geometry.bar_gap == 0
        assert theme.geometry.line_width == 0.1
        path = write_theme(
            tmp_path,
            "[geometry]\nheight = 100\nwidth = 50\n"
            "border_width = 99\nbar_gap = 99\nline_width = 99\n",
        )
        theme = read_theme(path)
        assert theme.geometry.width == 100  # width >= height
        assert theme.geometry.border_width == 20.0
        assert theme.geometry.bar_gap == 50
        assert theme.geometry.line_width == 20.0

    def test_duration_scale_clamps(self, tmp_path):
        assert (
            read_theme(
                write_theme(tmp_path, "[animation]\nduration_scale = 100\n")
            ).animation.duration_scale
            == 10.0
        )
        assert (
            read_theme(
                write_theme(tmp_path, "[animation]\nduration_scale = 0.001\n")
            ).animation.duration_scale
            == 0.1
        )

    def test_animation_durations_clamp(self, tmp_path):
        path = write_theme(
            tmp_path,
            "[animation.entry]\nduration = -1\n[animation.exit]\nduration = 99\n",
        )
        theme = read_theme(path)
        assert theme.animation.entry.duration == 0.0
        assert theme.animation.exit.duration == 5.0

    def test_animation_specs(self, tmp_path):
        path = write_theme(
            tmp_path,
            '[animation.entry]\nduration = 0.3\neasing = "ease-out-back"\n'
            '[animation.exit]\neasing = "nonsense"\n',
        )
        theme = read_theme(path)
        assert theme.animation.entry.duration == 0.3
        assert theme.animation.entry.easing == "ease-out-back"
        assert theme.animation.exit.easing == DEFAULT_THEME.animation.exit.easing

    def test_animation_enabled_bool(self, tmp_path):
        path = write_theme(tmp_path, "[animation]\nenabled = false\n")
        assert read_theme(path).animation.enabled is False

    def test_margin_int_applies_to_all_edges(self, tmp_path):
        path = write_theme(tmp_path, "[position]\nmargin = 20\n")
        assert read_theme(path).position.margin == (20, 20, 20, 20)

    def test_margin_four_list(self, tmp_path):
        path = write_theme(tmp_path, "[position]\nmargin = [1, 2, 3, 4]\n")
        assert read_theme(path).position.margin == (1, 2, 3, 4)

    def test_margin_invalid_falls_back(self, tmp_path):
        path = write_theme(tmp_path, "[position]\nmargin = [1, 2]\n")
        assert read_theme(path).position.margin == DEFAULT_THEME.position.margin

    def test_anchor_choice(self, tmp_path):
        path = write_theme(tmp_path, '[position]\nanchor = "top-right"\n')
        assert read_theme(path).position.anchor == "top-right"

    def test_invalid_anchor_falls_back(self, tmp_path):
        path = write_theme(tmp_path, '[position]\nanchor = "everywhere"\n')
        assert read_theme(path).position.anchor == DEFAULT_ANCHOR

    def test_font(self, tmp_path):
        path = write_theme(tmp_path, '[font]\nfamily = "JetBrains Mono"\nsize = 14\n')
        theme = read_theme(path)
        assert theme.font.family == "JetBrains Mono"
        assert theme.font.size == 14

    def test_font_size_clamps(self, tmp_path):
        path = write_theme(tmp_path, "[font]\nsize = 500\n")
        assert read_theme(path).font.size == 72
        path = write_theme(tmp_path, "[font]\nsize = 1\n")
        assert read_theme(path).font.size == 6

    @pytest.mark.parametrize("value", ["inf", "-inf", "nan"])
    def test_nonfinite_number_falls_back(self, tmp_path, value):
        path = write_theme(tmp_path, f"[geometry]\nborder_width = {value}\n")
        assert (
            read_theme(path).geometry.border_width
            == DEFAULT_THEME.geometry.border_width
        )

    def test_a_clamped_number_is_logged_only_for_the_active_theme(self, caplog):
        from quoth.theme import _theme_from_data

        _theme_from_data({"geometry": {"height": 500}})
        _theme_from_data({"geometry": {"height": 500}}, strict=True)
        _theme_from_data({"geometry": {"height": 48}})

        assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
            ("WARNING", "theme.toml [geometry] height: outside 24 to 200, using 200")
        ]

    def test_huge_integer_clamps_before_float_conversion(self, tmp_path):
        path = write_theme(tmp_path, "[geometry]\nwidth = " + "9" * 400 + "\n")
        assert read_theme(path).geometry.width == 2000

    @pytest.mark.parametrize("margin", [-(2**31), 2**31 - 1])
    def test_margin_accepts_signed_32_bit_boundary(self, tmp_path, margin):
        path = write_theme(tmp_path, f"[position]\nmargin = {margin}\n")
        assert read_theme(path).position.margin == (margin,) * 4

    @pytest.mark.parametrize(
        "margin",
        [str(-(2**31) - 1), str(2**31), f"[1, 2, 3, {2**31}]"],
    )
    def test_margin_outside_signed_32_bit_falls_back_whole(self, tmp_path, margin):
        path = write_theme(tmp_path, f"[position]\nmargin = {margin}\n")
        assert read_theme(path).position.margin == DEFAULT_THEME.position.margin


class TestShippedPresets:
    """The presets packaged in `quoth/themes` must always load cleanly."""

    def _presets(self):
        themes_dir = files("quoth") / "themes"
        names = sorted(e.name for e in themes_dir.iterdir() if e.name.endswith(".toml"))
        return themes_dir, names

    def test_default_and_palette_presets_ship(self):
        _, names = self._presets()
        assert "default.toml" in names and len(names) > 1

    def test_palette_presets_differ_only_in_colors(self, tmp_path):
        themes_dir, names = self._presets()
        for name in set(names) - {"default.toml"}:
            target = tmp_path / "theme.toml"
            target.write_text((themes_dir / name).read_text(encoding="utf-8"))
            theme = read_theme(target)
            assert theme.palette != DEFAULT_THEME.palette, name
            assert theme.geometry == DEFAULT_THEME.geometry, name
            assert theme.position == DEFAULT_THEME.position, name
            assert theme.font == DEFAULT_THEME.font, name
            assert theme.animation == DEFAULT_THEME.animation, name


class TestPresetPicker:
    """The picker stores nothing, so a broken apply/identify round trip makes it lie."""

    def test_default_is_listed_first(self):
        # It is the stock look and the commented reference: the picker leads with it.
        assert list_presets()[0] == "default"
        assert sorted(list_presets()[1:]) == list_presets()[1:]

    def test_every_preset_identifies_as_itself(self, tmp_path):
        target = tmp_path / "theme.toml"
        for name in list_presets():
            apply_theme_id(name, tmp_path, target)
            assert identify_theme(target) == name

    def test_no_file_reads_as_default(self, tmp_path):
        # No file is the stock look, which is what "default" means to the user.
        assert identify_theme(tmp_path / "theme.toml") == "default"

    def test_hand_edited_reads_as_custom(self, tmp_path):
        target = write_theme(tmp_path, '[colors]\nbackground = "#ABCDEF"\n')
        assert identify_theme(target) is None

    def test_malformed_reads_as_custom(self, tmp_path):
        # Unreadable is not ours to claim, and it means the confirm prompt still
        # fires before we overwrite whatever the user was mid-way through writing.
        target = write_theme(tmp_path, "not valid toml {{{")
        assert identify_theme(target) is None

    def test_comments_do_not_affect_identity(self, tmp_path):
        # Identity is the parsed Theme, not the bytes, so a user's notes survive.
        target = write_theme(tmp_path, "# my notes\n\n" + preset_source("nord"))
        assert identify_theme(target) == "nord"

    def test_apply_creates_missing_parent(self, tmp_path):
        target = tmp_path / "fresh" / "theme.toml"
        apply_theme_id("nord", tmp_path, target)
        assert identify_theme(target) == "nord"

    def test_apply_preserves_preset_comments(self, tmp_path):
        # The copied file is the user's editing on-ramp: it must keep its comments.
        target = tmp_path / "theme.toml"
        apply_theme_id("default", tmp_path, target)
        assert target.read_text(encoding="utf-8") == preset_source("default")

    def test_apply_replaces_a_custom_theme_wholesale(self, tmp_path):
        target = write_theme(tmp_path, "[geometry]\nwidth = 900\n")
        apply_theme_id("nord", tmp_path, target)
        assert read_theme(target).geometry.width == DEFAULT_THEME.geometry.width

    def test_broken_install_never_raises(self, tmp_path, monkeypatch):
        """Unreadable preset data costs the picker, not the whole Settings window."""
        import quoth.theme as theme_mod

        monkeypatch.setattr(theme_mod, "_PRESET_DIR", tmp_path / "gone")
        target = write_theme(tmp_path, '[colors]\nbackground = "#ABCDEF"\n')
        assert list_presets() == []
        assert identify_theme(target) is None
        assert identify_theme(tmp_path / "absent.toml") == "default"
        # The user's own theme still loads: broken presets affect only the picker.
        assert read_theme(target).palette.background == pytest.approx(
            (0xAB / 255, 0xCD / 255, 0xEF / 255, 1.0)
        )

    def test_malformed_bundled_preset_does_not_break_identity(
        self, tmp_path, monkeypatch
    ):
        import quoth.theme as theme_mod

        presets = tmp_path / "presets"
        presets.mkdir()
        (presets / "broken.toml").write_text("not toml [[[")
        monkeypatch.setattr(theme_mod, "_PRESET_DIR", presets)
        target = write_theme(tmp_path, '[colors]\nbackground = "#ABCDEF"\n')

        assert identify_theme(target) is None

    def test_bundled_preset_reads_and_apply_are_strict(self, tmp_path, monkeypatch):
        import quoth.theme as theme_mod

        presets = tmp_path / "presets"
        presets.mkdir()
        (presets / "bad.toml").write_text(
            'schema = 1\n[colors]\naccent = "private-invalid"\n'
        )
        monkeypatch.setattr(theme_mod, "_PRESET_DIR", presets)
        active = tmp_path / "theme.toml"
        active.write_bytes(b"keep me")

        with pytest.raises(ValueError, match=r"accent.*expected"):
            theme_mod.preset_theme("bad")
        with pytest.raises(ValueError, match=r"accent.*expected"):
            theme_mod.apply_theme_id("bad", tmp_path, active)
        assert active.read_bytes() == b"keep me"

    def test_default_preset_is_not_equal_to_default_theme(self, tmp_path):
        """`identify_theme` must compare parsed themes, never `DEFAULT_THEME` itself.

        Its colours are rounded (0.055) but parsed hex is not (0.0549...): comparing
        with it would match no file, and every theme would read as Custom.
        """
        target = tmp_path / "theme.toml"
        apply_theme_id("default", tmp_path, target)
        loaded = read_theme(target)
        assert loaded != DEFAULT_THEME
        assert identify_theme(target) == "default"
        # Equal to the eye, which is why the picker can call a missing file "default".
        assert loaded.palette.background == pytest.approx(
            DEFAULT_THEME.palette.background, abs=0.005
        )


class TestImmutability:
    def test_theme_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            DEFAULT_THEME.font = None

    def test_nested_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            DEFAULT_THEME.geometry.width = 999

    def test_default_theme_reproduces_brand_look(self):
        # Pin the brand look shared with quoth.dev: ink behind bone.
        assert DEFAULT_THEME.palette.background == (0.055, 0.055, 0.067, 0.85)
        assert DEFAULT_THEME.palette.border == (0.949, 0.941, 0.918, 0.10)
        assert DEFAULT_THEME.geometry.width == 240
        assert DEFAULT_THEME.geometry.height == 48
        assert DEFAULT_THEME.animation.entry.easing == "ease-out-cubic"
        assert DEFAULT_THEME.animation.exit.easing == "ease-in-cubic"
        assert Theme() == DEFAULT_THEME


class TestThemeMeta:
    """`[meta]` is display-only and clamped: shared files' text shows in the picker."""

    def _theme(self, meta_toml: str):
        import tomllib

        from quoth.theme import _theme_from_data

        return _theme_from_data(tomllib.loads(meta_toml))

    def test_all_fields_parse(self):
        t = self._theme(
            '[meta]\nname = "Synthwave"\nauthor = "Sam"\ndescription = "Neon."\n'
        )
        assert t.meta.name == "Synthwave"
        assert t.meta.author == "Sam"
        assert t.meta.description == "Neon."

    def test_missing_meta_is_all_none(self):
        t = self._theme("schema = 1\n")
        assert t.meta.name is None and t.meta.author is None

    def test_overlong_values_truncate(self):
        long = "x" * 500
        t = self._theme(
            f'[meta]\nname = "{long}"\nauthor = "{long}"\ndescription = "{long}"\n'
        )
        assert len(t.meta.name) == 60
        assert len(t.meta.author) == 60
        assert len(t.meta.description) == 200

    def test_control_characters_flatten_to_spaces(self):
        # A name spanning lines or carrying escapes must render as one flat line.
        t = self._theme('[meta]\nname = "a\\nb\\u0007c"\n')
        assert t.meta.name == "a b c"

    def test_non_string_values_are_ignored(self):
        t = self._theme("[meta]\nname = 42\n")
        assert t.meta.name is None

    def test_metadata_never_changes_identity(self):
        # A preset copy plus a credit still is that preset: identity is the look.
        bare = self._theme("schema = 1\n")
        credited = self._theme('[meta]\nname = "Mine"\nauthor = "Me"\n')
        assert bare == credited


class TestSchema:
    @pytest.mark.parametrize("name", ["default", "full", "partial"])
    def test_v1_reference_values(self, name):
        from quoth.theme import read_theme

        fixtures = Path(__file__).parent / "fixtures" / "theme-v1"
        expected = json.loads((fixtures / "expected.json").read_text())[name]
        theme = (
            DEFAULT_THEME
            if name == "default"
            else read_theme(fixtures / f"{name}.toml")
        )
        assert json.loads(json.dumps(dataclasses.asdict(theme))) == expected

    @pytest.mark.parametrize("header", ["", "schema = 1\n"])
    def test_legacy_and_v1_have_the_same_identity(self, tmp_path, header):
        from quoth.theme import identify_theme, preset_theme, read_theme

        path = tmp_path / "theme.toml"
        path.write_text(header + preset_source("default").replace("schema = 1\n", ""))
        assert read_theme(path) == preset_theme("default")
        assert identify_theme(path) == "default"

    @pytest.mark.parametrize(
        "version", ["2", "0", "-1", "true", "1.0", '"1"', "[]", "{}"]
    )
    def test_unsupported_schema_is_not_reported_as_malformed_toml(
        self, tmp_path, version
    ):
        from quoth.theme import import_user_theme, read_theme

        path = tmp_path / "theme.toml"
        path.write_text(f"schema = {version}\n")
        with pytest.raises(ValueError, match=r"schema.*1"):
            read_theme(path)
        with pytest.raises(ValueError, match=r"schema.*1"):
            import_user_theme(path, tmp_path / "shelf")
        assert not (tmp_path / "shelf").exists()

    def test_unsupported_schema_cannot_replace_active(self, tmp_path):
        shelf = tmp_path / "themes"
        shelf.mkdir()
        source = shelf / "future.toml"
        source.write_text("schema = 2\n")
        active = tmp_path / "theme.toml"
        active.write_bytes(b"# Keep my theme\n")
        with pytest.raises(ValueError, match=r"schema.*1"):
            apply_theme_id("user:future", shelf, active)
        assert active.read_bytes() == b"# Keep my theme\n"


class TestStrictThemeValidation:
    @pytest.mark.parametrize(
        ("content", "field"),
        [
            ('schema = 1\n[colors]\naccent = "private-color"\n', "accent"),
            ("schema = 1\n[geometry]\nwidth = true\n", "width"),
            ("schema = 1\n[geometry]\nborder_width = nan\n", "border_width"),
            ('schema = 1\n[position]\nanchor = "private-anchor"\n', "anchor"),
            ("schema = 1\n[position]\nmargin = [1, 2]\n", "margin"),
            ('schema = 1\n[font]\nfamily = ""\n', "family"),
            ("schema = 1\n[font]\nsize = false\n", "size"),
            ("schema = 1\n[animation]\nenabled = 1\n", "enabled"),
            ("schema = 1\n[animation]\nduration_scale = inf\n", "duration_scale"),
            ('schema = 1\n[animation.entry]\nduration = "private"\n', "duration"),
            ('schema = 1\n[animation.exit]\neasing = "private"\n', "easing"),
            ("schema = 1\n[animation]\nentry = 1\n", "entry"),
            ("schema = 1\n[animation]\nexit = []\n", "exit"),
        ],
    )
    def test_strict_rejects_invalid_known_appearance_fields(
        self, tmp_path, content, field
    ):
        from quoth.theme import _read_theme_file

        path = write_theme(tmp_path, content)
        with pytest.raises(ValueError, match=rf"field .*{field}.*expected") as exc:
            _read_theme_file(path, strict=True)
        assert "private" not in str(exc.value)

    @pytest.mark.parametrize(
        "section", ["colors", "geometry", "position", "font", "animation", "meta"]
    )
    def test_strict_rejects_wrong_section_types(self, tmp_path, section):
        from quoth.theme import _read_theme_file

        path = write_theme(tmp_path, f"schema = 1\n{section} = 42\n")
        with pytest.raises(
            ValueError, match=rf"section \[{section}\].*expected a table"
        ):
            _read_theme_file(path, strict=True)

    @pytest.mark.parametrize(
        "content",
        [
            "",
            '[meta]\nname = "Only metadata"\n',
            "[colors]\nunknown = 1\n",
            "[animation.entry]\nunknown = 1\n",
        ],
    )
    def test_schema_less_strict_theme_needs_an_appearance_leaf(self, tmp_path, content):
        from quoth.theme import _read_theme_file

        with pytest.raises(ValueError, match="recognized appearance field"):
            _read_theme_file(write_theme(tmp_path, content), strict=True)

    @pytest.mark.parametrize(
        "content",
        [
            "schema = 1\n",
            'schema = 1\n[meta]\nname = "Metadata only"\n',
            "schema = 1\n[unknown]\nvalue = 1\n",
            '[colors]\naccent = "#FF0000"\n',
            "[animation]\nenabled = false\n",
            '[animation.entry]\neasing = "linear"\n',
        ],
    )
    def test_strict_accepts_schema_or_recognized_legacy_leaf(self, tmp_path, content):
        from quoth.theme import _read_theme_file

        _content, theme = _read_theme_file(write_theme(tmp_path, content), strict=True)
        assert isinstance(theme, Theme)

    def test_skin_error_precedes_schema_error(self, tmp_path):
        from quoth.theme import _read_theme_file

        path = write_theme(tmp_path, "kind = 'skin'\nschema = 2\n")
        with pytest.raises(ValueError, match="App skin → Import"):
            _read_theme_file(path, strict=True)

    def test_strict_keeps_metadata_leaf_sanitation(self, tmp_path):
        from quoth.theme import _read_theme_file

        path = write_theme(
            tmp_path,
            'schema = 1\n[meta]\nname = 42\nauthor = "a\\nb"\n',
        )
        _content, theme = _read_theme_file(path, strict=True)
        assert theme.meta.name is None
        assert theme.meta.author == "a b"

    def test_strict_and_live_reads_normalize_valid_values_identically(self, tmp_path):
        from quoth.theme import _read_theme_file, read_theme

        path = write_theme(
            tmp_path,
            "[geometry]\nheight = 5\nwidth = 31.9\nline_width = 2\n"
            "[position]\nmargin = 4\n",
        )
        strict = _read_theme_file(path, strict=True)[1]
        live = read_theme(path)
        assert strict == live
        assert strict.geometry.height == 24
        assert strict.geometry.width == 31
        assert strict.geometry.line_width == 2.0
        assert strict.position.margin == (4, 4, 4, 4)

    def test_public_live_reader_keeps_per_key_fallback(self, tmp_path):
        from quoth.theme import read_theme

        path = write_theme(
            tmp_path,
            'meta = 42\n[colors]\nbackground = "private-invalid"\naccent = "#FF0000"\n'
            '[font]\nfamily = ""\n[animation]\nenabled = 0\n',
        )
        theme = read_theme(path)
        assert theme.palette.background == DEFAULT_THEME.palette.background
        assert theme.palette.accent[:3] == (1.0, 0.0, 0.0)
        assert theme.meta == DEFAULT_THEME.meta
        assert theme.font.family == "sans-serif"
        assert theme.animation.enabled is True

    def test_invalid_known_value_cannot_be_imported_or_applied(self, tmp_path):
        from quoth.theme import apply_theme_id, import_user_theme

        source = write_theme(
            tmp_path, 'schema = 1\n[colors]\naccent = "private-invalid"\n'
        )
        shelf = tmp_path / "shelf"
        with pytest.raises(ValueError, match=r"accent.*expected"):
            import_user_theme(source, shelf)
        assert not shelf.exists()

        shelf.mkdir()
        (shelf / "bad.toml").write_bytes(source.read_bytes())
        active = tmp_path / "active.toml"
        active.write_bytes(b"keep me")
        with pytest.raises(ValueError, match=r"accent.*expected"):
            apply_theme_id("user:bad", shelf, active)
        assert active.read_bytes() == b"keep me"


class TestFileGuards:
    def test_public_reader_has_strict_bounded_errors(self, tmp_path, monkeypatch):
        from quoth.theme import read_theme

        with pytest.raises(FileNotFoundError):
            read_theme(tmp_path / "missing.toml")

        dangling = tmp_path / "dangling.toml"
        dangling.symlink_to("missing-target.toml")
        with pytest.raises(FileNotFoundError):
            read_theme(dangling)

        fifo = tmp_path / "theme.fifo"
        os.mkfifo(fifo)
        with pytest.raises(ValueError, match="regular file"):
            read_theme(fifo)

        oversized = tmp_path / "large.toml"
        oversized.write_bytes(b"#" * (atomic_write.TOML_MAX_BYTES + 1))
        with pytest.raises(ValueError, match="256 KiB"):
            read_theme(oversized)

        invalid = tmp_path / "invalid.toml"
        invalid.write_bytes(b"\xffprivate")
        with pytest.raises(ValueError, match="UTF-8") as exc:
            read_theme(invalid)
        assert "private" not in str(exc.value)

        monkeypatch.setattr(
            os, "open", lambda *_args: (_ for _ in ()).throw(OSError("io"))
        )
        with pytest.raises(OSError, match="io"):
            read_theme(invalid)

    def test_reader_follows_symlink_to_regular_file(self, tmp_path):
        from quoth.theme import read_theme

        target = write_theme(tmp_path, '[colors]\naccent = "#FF0000"\n')
        link = tmp_path / "linked.toml"
        link.symlink_to(target)
        assert read_theme(link).palette.accent[:3] == (1.0, 0.0, 0.0)

    def test_unknown_and_invalid_warning_details_are_bounded(self, caplog):
        from quoth.theme import _theme_from_data

        _theme_from_data(
            {
                "colors": {
                    "background": "secret-" + "x" * 500,
                    **{f"unknown-{i}-\n" + "x" * 100: i for i in range(5)},
                }
            }
        )

        messages = caplog.messages
        unknown = next(message for message in messages if "unknown key(s)" in message)
        invalid = next(message for message in messages if "#RRGGBB" in message)
        assert "5 unknown key(s)" in unknown
        assert "unknown-2" in unknown and "unknown-3" not in unknown
        assert "\n" not in unknown.replace("\\n", "")
        assert len(unknown) < 300
        assert len(invalid) < 180


class TestUserThemes:
    """`user:` ids cannot shadow bundled presets, and a cap keeps the picker short."""

    def test_missing_directory_is_an_empty_shelf(self, tmp_path):
        from quoth.theme import list_user_themes

        assert list_user_themes(tmp_path / "nope") == []

    def test_unreadable_directory_is_an_empty_shelf_with_a_warning(
        self, tmp_path, caplog
    ):
        from quoth.theme import list_user_themes

        shelf = tmp_path / "themes"
        shelf.write_text("not a directory")
        caplog.set_level("WARNING")
        assert list_user_themes(shelf) == []
        # The bare word is also in the path, which holds the test's name.
        assert any("unreadable (" in message for message in caplog.messages)

    def test_only_the_active_theme_warns(self, tmp_path, caplog):
        from quoth.theme import identify_theme, list_user_themes, theme_from_id

        shelf = tmp_path / "themes"
        shelf.mkdir()
        meta = f'[meta]\nname = 42\nauthor = "{"x" * 61}"\n'
        for name in ("ocean", "sunset"):
            (shelf / f"{name}.toml").write_text(
                'schema = 1\n[colors]\nglow = "#fff"\n' + meta
            )
        active = tmp_path / "theme.toml"
        active.write_text('schema = 1\n[colors]\nsparkle = "#fff"\n' + meta)
        caplog.set_level("WARNING")
        identify_theme(active, shelf)
        for theme_id in list_user_themes(shelf):
            theme_from_id(theme_id, shelf)
        assert caplog.messages == [
            "theme.toml [colors]: 1 unknown key(s) ignored: 'sparkle'",
            "theme.toml [meta]: 'name' must be a string, ignored",
            "theme.toml [meta]: 'author' longer than 60 chars, truncated",
        ]

    def test_an_invalid_shelf_file_warns_once_until_it_changes(self, tmp_path, caplog):
        # The picker relists the shelf on every render.
        import os

        from quoth.theme import list_user_themes

        broken = tmp_path / "broken.toml"
        broken.write_text("schema = 99")
        caplog.set_level("WARNING")
        assert list_user_themes(tmp_path) == []
        assert list_user_themes(tmp_path) == []
        assert len(caplog.messages) == 1
        os.utime(broken, ns=(0, 0))
        list_user_themes(tmp_path)
        assert len(caplog.messages) == 2

    def test_a_cut_reason_still_shows_the_home_as_a_tilde(
        self, tmp_path, long_home, monkeypatch, caplog
    ):
        from quoth import theme

        (tmp_path / "dark.toml").write_text("schema = 1")
        error = FileNotFoundError(
            2, "No such file or directory", str(long_home / ".config/quoth/dark.toml")
        )
        monkeypatch.setattr(
            theme, "_read_theme_file", Mock(side_effect=error), raising=True
        )
        assert theme.list_user_themes(tmp_path) == []
        [warning] = [m for m in caplog.messages if "User theme" in m]
        assert "~/.config" in warning
        assert "maximilian" not in warning

    def test_lists_only_toml_files_namespaced(self, tmp_path):
        from quoth.theme import list_user_themes

        (tmp_path / "synth.toml").write_text("schema = 1")
        (tmp_path / "notes.txt").write_text("not a theme")
        (tmp_path / "sub").mkdir()
        assert list_user_themes(tmp_path) == ["user:synth"]

    def test_shelf_skips_strictly_invalid_entries(self, tmp_path):
        from quoth.theme import list_user_themes

        (tmp_path / "good.toml").write_text('[colors]\naccent = "#FF0000"\n')
        (tmp_path / "bad-value.toml").write_text(
            '[colors]\naccent = "private-invalid"\n'
        )
        (tmp_path / "meta-only.toml").write_text('[meta]\nname = "No look"\n')
        assert list_user_themes(tmp_path) == ["user:good"]

    def test_shelf_logs_bounded_safe_reason_for_invalid_entry(self, tmp_path, caplog):
        from quoth.theme import list_user_themes

        filename = "bad\n" + "x" * 100 + ".toml"
        (tmp_path / filename).write_text(
            'schema = 1\n[colors]\naccent = "private-value"\n'
        )
        assert list_user_themes(tmp_path) == []

        warning = next(
            message for message in caplog.messages if "User theme" in message
        )
        assert "ValueError" in warning and "accent" in warning
        assert "private-value" not in warning
        assert "\\n" in warning and "\n" not in warning.replace("\\n", "")
        assert len(warning) < 240
        assert (tmp_path / filename).exists()

    def test_shelf_stops_parsing_after_fifty_usable_themes(
        self, tmp_path, monkeypatch, caplog
    ):
        import quoth.theme as theme_mod

        for index in range(55):
            (tmp_path / f"t{index:03}.toml").write_text("schema = 1")
        parsed = []
        real_read = theme_mod._read_theme_file

        def recording_read(path, *, strict=False):
            parsed.append(path.name)
            return real_read(path, strict=strict)

        monkeypatch.setattr(
            theme_mod,
            "_read_theme_file",
            recording_read,
        )
        caplog.set_level("WARNING")
        theme_ids = theme_mod.list_user_themes(tmp_path)

        assert theme_ids == [f"user:t{index:03}" for index in range(50)]
        assert parsed == [f"t{index:03}.toml" for index in range(50)]
        assert any(
            "5 more candidate(s) not inspected" in warning
            for warning in caplog.messages
        )

    def test_theme_from_id_parses_a_shelf_file(self, tmp_path):
        from quoth.theme import theme_from_id

        (tmp_path / "red.toml").write_text('[colors]\naccent = "#FF0000"\n')
        t = theme_from_id("user:red", tmp_path)
        assert t.palette.accent[:3] == (1.0, 0.0, 0.0)

    def test_a_broken_shelf_file_preview_raises(self, tmp_path):
        from quoth.theme import theme_from_id

        (tmp_path / "bad.toml").write_text("not toml [[[")
        with pytest.raises(ValueError, match="valid TOML"):
            theme_from_id("user:bad", tmp_path)

    def test_manual_filename_is_legal_but_traversal_id_is_not(self, tmp_path):
        from quoth.theme import (
            apply_theme_id,
            list_user_themes,
            theme_from_id,
        )

        manual = tmp_path / "my theme.v2.toml"
        manual.write_text('[colors]\naccent = "#FF0000"\n')
        assert list_user_themes(tmp_path) == ["user:my theme.v2"]
        assert theme_from_id("user:my theme.v2", tmp_path).palette.accent[:3] == (
            1.0,
            0.0,
            0.0,
        )
        with pytest.raises(ValueError, match="Invalid user theme id"):
            theme_from_id("user:../outside", tmp_path)
        with pytest.raises(ValueError, match="Invalid user theme id"):
            apply_theme_id("user:../outside", tmp_path, tmp_path / "active.toml")

    def test_manual_control_characters_are_escaped_in_apply_log(self, tmp_path, caplog):
        from quoth.theme import apply_theme_id, list_user_themes

        manual = tmp_path / ("line\n" + "x" * 80 + ".toml")
        manual.write_text("schema = 1\n")
        theme_id = list_user_themes(tmp_path)[0]
        caplog.set_level("INFO")
        apply_theme_id(theme_id, tmp_path, tmp_path / "active.toml")

        message = next(
            message for message in caplog.messages if "Applied theme" in message
        )
        assert "\\n" in message and "\n" not in message
        assert len(message) < 180
        assert theme_id.startswith("user:line\n")

    def test_apply_theme_id_copies_the_shelf_file(self, tmp_path):
        from quoth.theme import apply_theme_id

        src_dir = tmp_path / "themes"
        src_dir.mkdir()
        (src_dir / "synth.toml").write_text('[colors]\naccent = "#FF00FF"\n')
        dst = tmp_path / "theme.toml"
        apply_theme_id("user:synth", src_dir, dst)
        assert dst.read_text() == (src_dir / "synth.toml").read_text()

    def test_applying_a_vanished_shelf_theme_is_a_rejection(self, tmp_path):
        from quoth.theme import apply_theme_id

        dst = tmp_path / "theme.toml"
        with pytest.raises(ValueError, match="Couldn't read the theme file"):
            apply_theme_id("user:gone", tmp_path / "themes", dst)
        assert not dst.exists()

    def test_apply_validates_then_atomically_copies_exact_bytes(self, tmp_path):
        from quoth.theme import apply_theme_id

        shelf = tmp_path / "themes"
        shelf.mkdir()
        source = shelf / "shared.toml"
        content = b"# keep\r\n[colors]\r\naccent = '#FF00FF'\r\n"
        source.write_bytes(content)
        active = tmp_path / "theme.toml"
        active.write_bytes(b"old")

        apply_theme_id("user:shared", shelf, active)

        assert active.read_bytes() == content

        source.write_bytes(b"not valid [[[")
        before = active.read_bytes()
        with pytest.raises(ValueError, match="valid TOML"):
            apply_theme_id("user:shared", shelf, active)
        assert active.read_bytes() == before

    def test_apply_preserves_active_symlink_and_target_permissions(self, tmp_path):
        from quoth.theme import apply_theme_id

        shelf = tmp_path / "themes"
        shelf.mkdir()
        source = shelf / "shared.toml"
        source.write_bytes(b"schema = 1\r\n")
        target = tmp_path / "dotfiles" / "theme.toml"
        target.parent.mkdir()
        target.write_bytes(b"old")
        target.chmod(0o640)
        active = tmp_path / "active.toml"
        active.symlink_to(target)

        apply_theme_id("user:shared", shelf, active)

        assert active.is_symlink()
        assert target.read_bytes() == source.read_bytes()
        assert target.stat().st_mode & 0o777 == 0o640

    def test_apply_creates_dangling_target_only_with_existing_parent(self, tmp_path):
        from quoth.theme import apply_theme_id

        shelf = tmp_path / "themes"
        shelf.mkdir()
        (shelf / "shared.toml").write_text("schema = 1\n")
        target_parent = tmp_path / "dotfiles"
        target_parent.mkdir()
        active = tmp_path / "active.toml"
        active.symlink_to(target_parent / "theme.toml")

        apply_theme_id("user:shared", shelf, active)

        assert active.is_symlink()
        assert stat.S_IMODE(active.resolve().stat().st_mode) == 0o600

        external = tmp_path / "missing" / "tree" / "theme.toml"
        blocked = tmp_path / "blocked.toml"
        blocked.symlink_to(external)
        with pytest.raises(FileNotFoundError):
            apply_theme_id("user:shared", shelf, blocked)
        assert not external.parent.exists()

    @pytest.mark.parametrize("target_kind", ["directory", "loop", "self-loop"])
    def test_apply_rejects_nonregular_or_looping_active_target(
        self, tmp_path, target_kind
    ):
        from quoth.theme import apply_theme_id

        shelf = tmp_path / "themes"
        shelf.mkdir()
        (shelf / "shared.toml").write_text("schema = 1\n")
        active = tmp_path / "active.toml"
        if target_kind == "directory":
            target = tmp_path / "target"
            target.mkdir()
            active.symlink_to(target)
            match = "regular file"
        elif target_kind == "self-loop":
            active.symlink_to(active.name)
            match = "loop"
        else:
            other = tmp_path / "other.toml"
            active.symlink_to(other)
            other.symlink_to(active)
            match = "loop"

        with pytest.raises(ValueError, match=match):
            apply_theme_id("user:shared", shelf, active)

    def test_identify_matches_a_shelf_theme(self, tmp_path):
        from quoth.theme import apply_theme_id, identify_theme

        shelf = tmp_path / "themes"
        shelf.mkdir()
        # Listed before synth, so only comparing looks can pick synth.
        (shelf / "amber.toml").write_text('[colors]\naccent = "#FFBF00"\n')
        (shelf / "synth.toml").write_text('[colors]\naccent = "#FF00FF"\n')
        dst = tmp_path / "theme.toml"
        apply_theme_id("user:synth", shelf, dst)
        assert identify_theme(dst, shelf) == "user:synth"

    def test_a_preset_copy_on_the_shelf_reads_as_the_preset(self, tmp_path):
        # Presets win a tie, so a shelf copy of nord can't impersonate a new theme.
        shelf = tmp_path / "themes"
        shelf.mkdir()
        (shelf / "mynord.toml").write_text(preset_source("nord"))
        dst = tmp_path / "theme.toml"
        dst.write_text(preset_source("nord"))
        assert identify_theme(dst, shelf) == "nord"

    def test_identify_skips_candidate_that_disappears_after_listing(
        self, tmp_path, monkeypatch
    ):
        import quoth.theme as theme_mod

        shelf = tmp_path / "themes"
        shelf.mkdir()
        candidate = shelf / "racy.toml"
        candidate.write_text("schema = 1\n")
        active = tmp_path / "theme.toml"
        active.write_text("schema = 1\n")
        monkeypatch.setattr(theme_mod, "list_user_themes", lambda _path: ["user:racy"])
        candidate.unlink()

        assert theme_mod.identify_theme(active, shelf) is None

    def test_identify_never_matches_invalid_candidate_via_defaults(
        self, tmp_path, monkeypatch
    ):
        import quoth.theme as theme_mod

        shelf = tmp_path / "themes"
        shelf.mkdir()
        (shelf / "bad.toml").write_text(
            'schema = 1\n[colors]\naccent = "private-invalid"\n'
        )
        active = tmp_path / "theme.toml"
        active.write_text("schema = 1\n")
        monkeypatch.setattr(theme_mod, "list_user_themes", lambda _path: ["user:bad"])

        assert theme_mod.identify_theme(active, shelf) is None

    def test_identify_normalizes_live_values_but_reads_candidates_strictly(
        self, tmp_path
    ):
        from quoth.theme import identify_theme

        shelf = tmp_path / "themes"
        shelf.mkdir()
        (shelf / "red.toml").write_text('schema = 1\n[colors]\naccent = "#FF0000"\n')
        active = tmp_path / "theme.toml"
        active.write_text(
            '[colors]\nbackground = "private-invalid"\naccent = "#FF0000"\n'
        )

        assert identify_theme(active, shelf) == "user:red"


class TestImportUserTheme:
    @pytest.mark.parametrize("schema", ["", "schema = 2\n"])
    def test_rejects_app_skin_without_requiring_kind_on_legacy_themes(
        self, tmp_path, schema
    ):
        from quoth.theme import import_user_theme, read_theme

        skin = tmp_path / "skin.toml"
        skin.write_text("kind = 'skin'\n" + schema)
        shelf = tmp_path / "themes"
        with pytest.raises(ValueError, match="App skin → Import"):
            import_user_theme(skin, shelf)
        if not schema:
            assert read_theme(skin) == DEFAULT_THEME
        assert not shelf.exists()

        legacy = tmp_path / "legacy.toml"
        legacy.write_text("kind = 'legacy-extension'\n[colors]\naccent = '#FF0000'\n")
        assert import_user_theme(legacy, shelf) == "user:legacy"

    def test_preserves_exact_bytes_and_returns_shelf_id(self, tmp_path):
        from quoth.theme import import_user_theme

        source = tmp_path / "shared.toml"
        content = b"# keep this comment\r\nschema = 1\r\n"
        source.write_bytes(content)
        shelf = tmp_path / "themes"

        assert import_user_theme(source, shelf) == "user:shared"
        assert (shelf / "shared.toml").read_bytes() == content

    def test_source_symlink_copies_target_bytes_under_selected_name(self, tmp_path):
        from quoth.theme import import_user_theme

        original = tmp_path / "original.toml"
        content = b"# shared theme\r\nschema = 1\r\n"
        original.write_bytes(content)
        source = tmp_path / "shared.toml"
        source.symlink_to(original)
        shelf = tmp_path / "themes"

        assert import_user_theme(source, shelf) == "user:shared"
        assert (shelf / "shared.toml").read_bytes() == content
        assert not (shelf / "shared.toml").is_symlink()
        assert source.is_symlink()

    def test_accepts_exact_size_limit_and_rejects_one_byte_more(self, tmp_path):
        from quoth.theme import import_user_theme

        prefix = b"schema = 1\n#"
        exact = tmp_path / "exact.toml"
        exact.write_bytes(prefix + b"x" * (atomic_write.TOML_MAX_BYTES - len(prefix)))
        shelf = tmp_path / "themes"
        assert import_user_theme(exact, shelf) == "user:exact"

        oversized = tmp_path / "oversized.toml"
        oversized.write_bytes(exact.read_bytes() + b"x")
        with pytest.raises(ValueError, match="256 KiB"):
            import_user_theme(oversized, shelf)
        assert not (shelf / "oversized.toml").exists()

    @pytest.mark.parametrize(
        ("content", "message"),
        [(b"\xffsecret", "UTF-8"), (b"secret = [[[", "valid TOML")],
    )
    def test_rejects_invalid_content_without_echoing_it(
        self, tmp_path, content, message
    ):
        from quoth.theme import import_user_theme

        source = tmp_path / "bad.toml"
        source.write_bytes(content)
        with pytest.raises(ValueError, match=message) as exc:
            import_user_theme(source, tmp_path / "themes")
        assert "secret" not in str(exc.value)

    @pytest.mark.parametrize("kind", ["fifo", "directory"])
    def test_rejects_non_regular_source_without_blocking(self, tmp_path, kind):
        from quoth.theme import import_user_theme

        source = tmp_path / "source"
        if kind == "fifo":
            os.mkfifo(source)
        else:
            source.mkdir()
        with pytest.raises(ValueError, match="regular file"):
            import_user_theme(source, tmp_path / "themes")

    def test_an_unreadable_source_is_a_rejection_not_a_publication_doubt(
        self, tmp_path
    ):
        from quoth.theme import import_user_theme

        shelf = tmp_path / "themes"
        with pytest.raises(ValueError, match="Couldn't read the theme file") as raised:
            import_user_theme(tmp_path / "missing.toml", shelf)
        assert isinstance(raised.value.__cause__, FileNotFoundError)
        assert not shelf.exists()

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("My weird.theme.TOML", "user:My-weird-theme"),
            ("💥.toml", "user:theme"),
            ("a" * 100 + ".toml", "user:" + "a" * 80),
        ],
    )
    def test_normalizes_and_bounds_destination_name(self, tmp_path, filename, expected):
        from quoth.theme import import_user_theme

        source = tmp_path / filename
        source.write_text("schema = 1\n")
        shelf = tmp_path / "themes"
        assert import_user_theme(source, shelf) == expected
        assert (shelf / (expected.removeprefix("user:") + ".toml")).is_file()

    def test_collisions_and_symlinks_are_kept(self, tmp_path):
        from quoth.theme import import_user_theme

        source = tmp_path / "shared.toml"
        source.write_text("schema = 1\n")
        shelf = tmp_path / "themes"
        shelf.mkdir()
        first = shelf / "shared.toml"
        first.write_text("first")
        victim = tmp_path / "victim"
        victim.write_text("victim")
        second = shelf / "shared-2.toml"
        second.symlink_to(victim)

        assert import_user_theme(source, shelf) == "user:shared-3"
        assert first.read_text() == "first"
        assert second.is_symlink()
        assert victim.read_text() == "victim"

    def test_collision_suffix_keeps_final_stem_within_limit(self, tmp_path):
        from quoth.theme import import_user_theme

        stem = "a" * 80
        source = tmp_path / f"{stem}.toml"
        source.write_text("schema = 1\n")
        shelf = tmp_path / "themes"
        shelf.mkdir()
        (shelf / f"{stem}.toml").write_text("first")

        theme_id = import_user_theme(source, shelf)
        saved_stem = theme_id.removeprefix("user:")
        assert saved_stem == "a" * 78 + "-2"
        assert len(saved_stem) == 80

    def test_creation_race_uses_the_next_name(self, tmp_path, monkeypatch):
        from quoth.theme import import_user_theme

        source = tmp_path / "race.toml"
        source.write_text("schema = 1\n")
        shelf = tmp_path / "themes"
        real_link = os.link
        raced = False

        def racing_link(source, destination):
            nonlocal raced
            if not raced:
                raced = True
                with open(destination, "xb") as competing:
                    competing.write(b"competitor")
            real_link(source, destination)

        monkeypatch.setattr(os, "link", racing_link)
        assert import_user_theme(source, shelf) == "user:race-2"
        assert (shelf / "race.toml").read_bytes() == b"competitor"
        assert (shelf / "race-2.toml").read_bytes() == source.read_bytes()

    def test_full_visible_shelf_is_rejected_before_writing(self, tmp_path):
        from quoth.theme import _USER_THEME_CAP, import_user_theme

        shelf = tmp_path / "themes"
        shelf.mkdir()
        for index in range(_USER_THEME_CAP):
            (shelf / f"saved-{index}.toml").write_text("schema = 1\n")
        source = tmp_path / "new.toml"
        source.write_text("schema = 1\n")

        with pytest.raises(ValueError, match="shelf is full"):
            import_user_theme(source, shelf)
        assert not (shelf / "new.toml").exists()

    def test_destination_io_errors_propagate(self, tmp_path, monkeypatch):
        from quoth.theme import import_user_theme

        source = tmp_path / "broken.toml"
        source.write_text("schema = 1\n")
        shelf = tmp_path / "themes"

        def fail_create(path, content):
            raise OSError("disk full")

        monkeypatch.setattr(atomic_write, "create_bytes", fail_create)
        with pytest.raises(OSError, match="disk full"):
            import_user_theme(source, shelf)
        assert not (shelf / "broken.toml").exists()

    def test_existing_file_at_shelf_path_is_not_a_name_collision(self, tmp_path):
        from quoth.theme import import_user_theme

        source = tmp_path / "shared.toml"
        source.write_text("schema = 1\n")
        shelf = tmp_path / "themes"
        shelf.write_text("existing file")

        with pytest.raises(FileExistsError):
            import_user_theme(source, shelf)
        assert shelf.read_text() == "existing file"

    def test_only_complete_theme_becomes_visible(self, tmp_path, monkeypatch):
        from quoth.theme import import_user_theme, list_user_themes

        source = tmp_path / "shared.toml"
        content = b"# original\r\n[colors]\r\naccent = '#FF0000'\r\n"
        source.write_bytes(content)
        shelf = tmp_path / "themes"
        real_link = os.link

        def observe_publication(temporary, destination):
            assert list_user_themes(shelf) == []
            assert not destination.exists()
            real_link(temporary, destination)
            assert list_user_themes(shelf) == ["user:shared"]
            assert destination.read_bytes() == content

        monkeypatch.setattr(os, "link", observe_publication)
        assert import_user_theme(source, shelf) == "user:shared"
        assert list(shelf.iterdir()) == [shelf / "shared.toml"]

    def test_abandoned_temporary_file_is_not_a_theme(self, tmp_path):
        from quoth.theme import import_user_theme, list_user_themes

        shelf = tmp_path / "themes"
        shelf.mkdir()
        (shelf / ".shared.toml.interrupted.tmp").write_bytes(b"[colors]\naccent = ")
        assert list_user_themes(shelf) == []
        source = tmp_path / "shared.toml"
        source.write_text("schema = 1\n")

        assert import_user_theme(source, shelf) == "user:shared"
        assert list_user_themes(shelf) == ["user:shared"]

    def test_preserves_existing_numeric_normalization(self, tmp_path):
        from quoth.theme import import_user_theme, theme_from_id

        source = tmp_path / "compatible.toml"
        source.write_text(
            '[colors]\naccent = "#FF0000"\n[geometry]\nwidth = 10000\nline_width = 2\n'
        )
        shelf = tmp_path / "themes"
        theme_id = import_user_theme(source, shelf)
        theme = theme_from_id(theme_id, shelf)

        assert theme.palette.accent[:3] == (1.0, 0.0, 0.0)
        assert theme.geometry.width == 2000
        assert theme.geometry.line_width == 2.0
        assert isinstance(theme.geometry.line_width, float)

    @pytest.mark.parametrize(
        "content",
        [
            "value = " + "[" * 2000 + "0" + "]" * 2000,
        ],
    )
    def test_parser_resource_errors_are_user_safe(self, tmp_path, content):
        from quoth.theme import import_user_theme

        source = tmp_path / "hostile.toml"
        source.write_text(content)
        shelf = tmp_path / "themes"

        with pytest.raises(ValueError, match="not valid TOML") as exc:
            import_user_theme(source, shelf)
        assert "maximum recursion" not in str(exc.value)
        assert not shelf.exists()
