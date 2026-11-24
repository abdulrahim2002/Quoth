# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `skin`: the app skin model and its persistence."""

import dataclasses
import os
from importlib import resources
from pathlib import Path

import pytest

from quoth import atomic_write
from quoth.skin import (
    DEFAULT_SKIN,
    Skin,
    SkinMeta,
    apply_skin_id,
    identify_skin,
    import_user_skin,
    list_presets,
    list_user_skins,
    load_skin,
    skin_from_id,
    user_skin_dir,
)


def write_skin(path: Path, content: str = "kind = 'skin'\n") -> Path:
    path.write_text(content)
    return path


class TestSkinModel:
    def test_default_palettes_are_canonical(self):
        assert dataclasses.asdict(DEFAULT_SKIN.light) == {
            "window": None,
            "view": None,
            "card": None,
            "popover": None,
            "text": None,
            "accent": "#0e0e11",
            "on_accent": "#f2f0ea",
        }
        assert dataclasses.asdict(DEFAULT_SKIN.dark) == {
            "window": "#08080a",
            "view": "#0e0e11",
            "card": None,
            "popover": "#181819",
            "text": "#f2f0ea",
            "accent": "#f2f0ea",
            "on_accent": "#0e0e11",
        }
        assert Skin() == DEFAULT_SKIN

    def test_skin_is_immutable_and_metadata_does_not_change_identity(self):
        credited = dataclasses.replace(
            DEFAULT_SKIN, meta=SkinMeta(name="Quoth", author="Sam")
        )
        assert credited == DEFAULT_SKIN
        with pytest.raises(dataclasses.FrozenInstanceError):
            credited.dark = credited.light


class TestLoadSkin:
    def test_overlay_file_points_to_the_overlay_importer(self, tmp_path):
        path = write_skin(tmp_path / "theme.toml", "schema = 1\n[colors]\n")
        with pytest.raises(ValueError, match="Overlay theme → Import"):
            load_skin(path)

    def test_missing_file_is_default(self, tmp_path):
        assert load_skin(tmp_path / "skin.toml") is DEFAULT_SKIN

    def test_kind_only_is_default(self, tmp_path):
        assert load_skin(write_skin(tmp_path / "skin.toml")) == DEFAULT_SKIN

    def test_schema_one_is_the_only_format(self, tmp_path):
        current = write_skin(tmp_path / "skin.toml", "kind = 'skin'\nschema = 1\n")
        assert load_skin(current) == DEFAULT_SKIN
        for schema in ("2", "'1'", "true"):
            future = write_skin(
                tmp_path / "skin.toml", f"kind = 'skin'\nschema = {schema}\n"
            )
            with pytest.raises(ValueError, match="schema = 1"):
                load_skin(future)

    def test_partial_palettes_overlay_defaults_and_normalize_hex(self, tmp_path):
        skin = load_skin(
            write_skin(
                tmp_path / "skin.toml",
                "kind = 'skin'\n"
                "[light]\nwindow = '#ABCDEF'\naccent = '#123456'\n"
                "[dark]\ncard = '#FEDCBA'\non_accent = '#654321'\n",
            )
        )
        assert skin.light.window == "#abcdef"
        assert skin.light.accent == "#123456"
        assert skin.light.view is None
        assert skin.dark.card == "#fedcba"
        assert skin.dark.on_accent == "#654321"
        assert skin.dark.window == DEFAULT_SKIN.dark.window

    def test_unknown_keys_are_ignored(self, tmp_path):
        skin = load_skin(
            write_skin(
                tmp_path / "skin.toml",
                "kind = 'skin'\nfuture = true\n[dark]\nfuture = '#112233'\n",
            )
        )
        assert skin == DEFAULT_SKIN

    def test_unknown_key_warnings_are_bounded_and_escape_control_characters(
        self, tmp_path, caplog
    ):
        caplog.set_level("WARNING")
        skin = load_skin(
            write_skin(
                tmp_path / "skin.toml",
                "kind = 'skin'\n"
                '"line\\n\\u001b[31m" = true\n'
                + "\n".join(f"unknown_{index} = true" for index in range(1000)),
            )
        )

        assert skin == DEFAULT_SKIN
        [message] = caplog.messages
        assert "1001 unknown key(s)" in message
        assert "\\n\\x1b" in message
        assert "\n" not in message and "\x1b" not in message
        assert len(message) < 300

    def test_picker_reads_do_not_warn_about_the_active_file(self, tmp_path, caplog):
        write_skin(
            tmp_path / "shared.toml",
            "kind = 'skin'\nfuture = true\n[meta]\nauthor = 42\n",
        )
        caplog.set_level("WARNING")
        assert skin_from_id("user:shared", tmp_path) == DEFAULT_SKIN

        assert caplog.messages == []

    def test_metadata_is_sanitized_capped_and_ignored_for_identity(self, tmp_path):
        from quoth.skin import _META_DESCRIPTION_MAX, _META_NAME_MAX

        skin = load_skin(
            write_skin(
                tmp_path / "skin.toml",
                "kind = 'skin'\n[meta]\n"
                'name = "  First\\nLight\\u0007  "\n'
                f'description = "{"x" * (_META_DESCRIPTION_MAX + 20)}"\n'
                "author = 42\n",
            )
        )
        assert skin.meta.name == "First Light"
        assert len(skin.meta.name) <= _META_NAME_MAX
        assert skin.meta.description == "x" * _META_DESCRIPTION_MAX
        assert skin.meta.author is None
        assert skin == DEFAULT_SKIN

    @pytest.mark.parametrize("key", ["name", "author"])
    def test_credits_are_capped_at_sixty_characters(self, tmp_path, key):
        skin = load_skin(
            write_skin(
                tmp_path / "skin.toml",
                f"kind = 'skin'\n[meta]\n{key} = '{'x' * 59} x'\n",
            )
        )

        # The cut lands just after the space, which is trimmed with it.
        assert getattr(skin.meta, key) == "x" * 59

    def test_the_active_file_warns_about_each_ignored_table_entry(
        self, tmp_path, caplog
    ):
        caplog.set_level("WARNING")
        load_skin(
            write_skin(
                tmp_path / "skin.toml",
                "kind = 'skin'\n"
                "[light]\nfuture = '#112233'\n"
                "[dark]\nfuture = '#112233'\n"
                "[meta]\nfuture = 1\nname = 42\nauthor = 42\n"
                f"description = '{'x' * 201}'\n",
            )
        )

        assert len(caplog.messages) == 6
        assert "[meta]: 1 unknown key(s) ignored: 'future'" in caplog.messages[2]

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            ("", "kind = 'skin'"),
            ("kind = 'theme'\n", "kind = 'skin'"),
            ("kind = 'skin'\nlight = 1\n", "light.*table"),
            ("kind = 'skin'\nmeta = []\n", "meta.*table"),
            ("kind = 'skin'\n[dark]\ntext = 'red'\n", "dark.*text"),
            ("kind = 'skin'\n[dark]\ntext = '#11223344'\n", "dark.*text"),
            ("kind = 'skin'\n[light]\naccent = 1\n", "light.*accent"),
        ],
    )
    def test_invalid_known_values_raise_clear_errors(self, tmp_path, content, message):
        with pytest.raises(ValueError, match=message):
            load_skin(write_skin(tmp_path / "skin.toml", content))

    @pytest.mark.parametrize(
        ("content", "message"),
        [(b"not [[[", "valid TOML"), (b"kind = 'skin'\n#\xff", "UTF-8")],
    )
    def test_malformed_files_raise_safe_errors(self, tmp_path, content, message):
        path = tmp_path / "skin.toml"
        path.write_bytes(content)
        with pytest.raises(ValueError, match=message):
            load_skin(path)

    def test_huge_integer_raises_bounded_validation_error(self, tmp_path):
        path = tmp_path / "skin.toml"
        path.write_text("kind = 'skin'\nfuture = " + "9" * 5000)

        with pytest.raises(ValueError, match="valid TOML") as exc:
            load_skin(path)
        assert "integer string conversion" not in str(exc.value)

    def test_oversize_and_non_regular_files_are_rejected(self, tmp_path):

        oversized = tmp_path / "oversized.toml"
        oversized.write_bytes(b"kind = 'skin'\n#" + b"x" * atomic_write.TOML_MAX_BYTES)
        with pytest.raises(ValueError, match="256 KiB"):
            load_skin(oversized)

        fifo = tmp_path / "pipe.toml"
        os.mkfifo(fifo)
        with pytest.raises(ValueError, match="regular file"):
            load_skin(fifo)

    def test_io_errors_other_than_missing_propagate(self, tmp_path, monkeypatch):
        def fail_open(_path, _flags):
            raise PermissionError("denied")

        monkeypatch.setattr(os, "open", fail_open)
        with pytest.raises(PermissionError, match="denied"):
            load_skin(tmp_path / "skin.toml")

    def test_deep_toml_has_bounded_errors_at_every_parsing_entry(self, tmp_path):
        shelf = tmp_path / "skins"
        shelf.mkdir()
        source = shelf / "deep.toml"
        source.write_text("kind = 'skin'\nvalue = " + "[" * 2000 + "0" + "]" * 2000)
        active = write_skin(tmp_path / "active.toml")

        readers = [
            lambda: load_skin(source),
            lambda: skin_from_id("user:deep", shelf),
            lambda: import_user_skin(source, tmp_path / "imported"),
            lambda: apply_skin_id("user:deep", shelf, active),
        ]
        for read in readers:
            with pytest.raises(ValueError, match="valid TOML") as exc:
                read()
            assert "maximum recursion" not in str(exc.value)
        assert active.read_text() == "kind = 'skin'\n"


class TestUserSkins:
    def test_user_skin_dir_and_missing_shelf(self, tmp_path):
        assert user_skin_dir(tmp_path) == tmp_path / "skins"
        assert list_user_skins(tmp_path / "missing") == []

    def test_an_unreadable_shelf_is_empty_with_a_warning(
        self, tmp_path, monkeypatch, caplog
    ):
        shelf = tmp_path / "skins"
        shelf.mkdir()
        real_iterdir = Path.iterdir

        def fail_iterdir(path):
            if path == shelf:
                raise PermissionError("denied")
            return real_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", fail_iterdir)
        caplog.set_level("WARNING")
        assert list_user_skins(shelf) == []
        [warning] = caplog.messages
        assert "unreadable" in warning

    def test_entries_that_cannot_be_statted_are_skipped(self, tmp_path):
        write_skin(tmp_path / "good.toml")
        (tmp_path / "dangling.toml").symlink_to(tmp_path / "gone.toml")
        (tmp_path / "loop.toml").symlink_to("loop.toml")

        assert list_user_skins(tmp_path) == ["user:good"]

    def test_list_is_sorted_bounded_and_does_not_parse(self, tmp_path):
        from quoth.skin import _USER_SKIN_CAP

        write_skin(tmp_path / "broken.toml", "broken [[[")
        write_skin(tmp_path / "a.toml")
        write_skin(tmp_path / "a-b.toml")  # sorts after "a" by stem, before it by name
        write_skin(tmp_path / "bad name.toml")
        (tmp_path / "notes.txt").write_text("kind = 'skin'")
        (tmp_path / "dir.toml").mkdir()
        (tmp_path / "huge.toml").write_bytes(b"x" * (atomic_write.TOML_MAX_BYTES + 1))
        for index in range(_USER_SKIN_CAP + 2):
            write_skin(tmp_path / f"s{index:02}.toml")

        skin_ids = list_user_skins(tmp_path)

        assert skin_ids == sorted(skin_ids)
        # dir, huge and notes would sort between broken and s00 if listed.
        assert skin_ids[:5] == [
            "user:a",
            "user:a-b",
            "user:bad name",
            "user:broken",
            "user:s00",
        ]
        assert len(skin_ids) == _USER_SKIN_CAP
        assert "user:a" in skin_ids
        assert "user:broken" in skin_ids
        assert "user:bad name" in skin_ids

    @pytest.mark.parametrize(
        "skin_id",
        ["", "dark", "user:", "user:../escape", "user:/escape", "user:bad\0name"],
    )
    def test_user_ids_reject_unknown_or_escaping_names(self, tmp_path, skin_id):
        with pytest.raises(ValueError, match="skin id"):
            skin_from_id(skin_id, tmp_path)
        with pytest.raises(ValueError, match="skin id"):
            apply_skin_id(skin_id, tmp_path, tmp_path / "active.toml")

    @pytest.mark.parametrize("stem", ["bad name.v2", ".."])
    def test_manual_filename_ids_are_accepted(self, tmp_path, stem):
        source = write_skin(tmp_path / f"{stem}.toml")
        assert f"user:{stem}" in list_user_skins(tmp_path)
        assert skin_from_id(f"user:{stem}", tmp_path) == DEFAULT_SKIN

        active = tmp_path / "active.toml"
        apply_skin_id(f"user:{stem}", tmp_path, active)
        assert active.read_bytes() == source.read_bytes()

    def test_manual_control_characters_are_escaped_in_apply_log(self, tmp_path, caplog):
        stem = "line\n" + "x" * 80
        source = write_skin(tmp_path / f"{stem}.toml")
        skin_id = f"user:{stem}"
        caplog.set_level("INFO")
        active = tmp_path / "active.toml"
        apply_skin_id(skin_id, tmp_path, active)

        message = next(
            message for message in caplog.messages if "Applied skin" in message
        )
        assert "\\n" in message and "\n" not in message
        assert len(message) < 180
        assert active.read_bytes() == source.read_bytes()

    def test_skin_from_id_loads_default_and_user_skin(self, tmp_path):
        write_skin(
            tmp_path / "rose.toml", "kind = 'skin'\n[dark]\naccent = '#AA0000'\n"
        )
        assert skin_from_id("default", tmp_path) is DEFAULT_SKIN
        assert skin_from_id("user:rose", tmp_path).dark.accent == "#aa0000"
        with pytest.raises(FileNotFoundError):
            skin_from_id("user:missing", tmp_path)

    def test_identify_default_absent_and_with_metadata(self, tmp_path):
        shelf = tmp_path / "skins"
        assert identify_skin(tmp_path / "active.toml", shelf) == "default"
        active = write_skin(
            tmp_path / "active.toml", "kind = 'skin'\n[meta]\nname = 'Credited'\n"
        )
        assert identify_skin(active, shelf) == "default"

    def test_identify_matches_both_palettes_ignoring_metadata_and_skips_broken(
        self, tmp_path
    ):
        shelf = tmp_path / "skins"
        shelf.mkdir()
        write_skin(shelf / "broken.toml", "broken [[[")
        write_skin(
            shelf / "rose.toml",
            "kind = 'skin'\n[dark]\naccent = '#AA0000'\n[meta]\nname = 'Shelf'\n",
        )
        active = write_skin(
            tmp_path / "active.toml",
            "kind = 'skin'\n[dark]\naccent = '#aa0000'\n[meta]\nname = 'Active'\n",
        )
        assert identify_skin(active, shelf) == "user:rose"

        active.write_text("kind = 'skin'\n[light]\naccent = '#0000aa'\n")
        assert identify_skin(active, shelf) is None
        active.write_text("broken [[[")
        assert identify_skin(active, shelf) is None


class TestPresets:
    def test_bundled_presets_are_listed_alphabetically(self):
        presets = list_presets()
        assert presets and presets == sorted(presets)

    def test_unreadable_package_data_lists_no_presets(self, tmp_path, monkeypatch):
        monkeypatch.setattr("quoth.skin._PRESET_DIR", tmp_path / "gone")

        assert list_presets() == []

    @pytest.mark.parametrize("skin_id", list_presets())
    def test_each_preset_credits_its_source_and_sets_every_role(
        self, tmp_path, skin_id
    ):
        preset = skin_from_id(skin_id, tmp_path)

        assert preset.meta.name and preset.meta.author and preset.meta.description
        assert None not in dataclasses.astuple(preset.light)
        assert None not in dataclasses.astuple(preset.dark)

    def test_applying_a_preset_copies_it_and_identifies_it(self, tmp_path):
        shelf = tmp_path / "skins"
        active = tmp_path / "skin.toml"

        apply_skin_id("nord", shelf, active)

        bundled = resources.files("quoth") / "skins" / "nord.toml"
        assert active.read_bytes() == bundled.read_bytes()
        assert identify_skin(active, shelf) == "nord"

    def test_a_shelf_copy_neither_shadows_nor_outranks_the_preset(self, tmp_path):
        shelf = tmp_path / "skins"
        shelf.mkdir()
        bundled = resources.files("quoth") / "skins" / "nord.toml"
        (shelf / "nord.toml").write_bytes(bundled.read_bytes())
        active = tmp_path / "skin.toml"

        apply_skin_id("user:nord", shelf, active)

        assert skin_from_id("user:nord", shelf) == skin_from_id("nord", shelf)
        assert identify_skin(active, shelf) == "nord"


class TestImportUserSkin:
    def test_accepts_the_exact_size_limit(self, tmp_path):

        prefix = b"kind = 'skin'\n#"
        source = tmp_path / "exact.toml"
        source.write_bytes(prefix + b"x" * (atomic_write.TOML_MAX_BYTES - len(prefix)))

        skin_id = import_user_skin(source, tmp_path / "skins")

        assert skin_id == "user:exact"
        assert (tmp_path / "skins" / "exact.toml").stat().st_size == len(
            source.read_bytes()
        )
        assert list_user_skins(tmp_path / "skins") == ["user:exact"]

    def test_preserves_exact_bytes_and_source_symlink_name(self, tmp_path):
        original = tmp_path / "original.toml"
        content = b"# comment\r\nkind = 'skin'\r\n"
        original.write_bytes(content)
        source = tmp_path / "Shared skin.toml"
        source.symlink_to(original)
        shelf = tmp_path / "skins"

        assert import_user_skin(source, shelf) == "user:Shared-skin"
        assert (shelf / "Shared-skin.toml").read_bytes() == content
        assert not (shelf / "Shared-skin.toml").is_symlink()

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("💥.toml", "user:skin"),
            ("a" * 100 + ".toml", "user:" + "a" * 80),
            ("mixed.TOML", "user:mixed"),
        ],
    )
    def test_normalizes_and_bounds_name(self, tmp_path, filename, expected):
        source = write_skin(tmp_path / filename)
        shelf = tmp_path / "skins"
        assert import_user_skin(source, shelf) == expected

    def test_keeps_collisions_and_destination_symlinks(self, tmp_path):
        source = write_skin(tmp_path / "shared.toml")
        shelf = tmp_path / "skins"
        shelf.mkdir()
        (shelf / "shared.toml").write_text("first")
        victim = tmp_path / "victim"
        victim.write_text("victim")
        (shelf / "shared-2.toml").symlink_to(victim)

        assert import_user_skin(source, shelf) == "user:shared-3"
        assert (shelf / "shared.toml").read_text() == "first"
        assert (shelf / "shared-2.toml").is_symlink()
        assert victim.read_text() == "victim"

    def test_a_shelf_that_cannot_be_made_fails_instead_of_renaming(
        self, tmp_path, monkeypatch
    ):
        source = write_skin(tmp_path / "shared.toml")
        shelf = tmp_path / "skins"
        shelf.symlink_to(tmp_path / "unmounted" / "skins")
        attempts = []
        real_create = atomic_write.create_bytes

        def create(path, content):
            attempts.append(path.name)
            assert len(attempts) < 5, "import kept trying new names"
            real_create(path, content)

        monkeypatch.setattr(atomic_write, "create_bytes", create)
        with pytest.raises(FileExistsError) as raised:
            import_user_skin(source, shelf)
        assert raised.value.filename == str(shelf)
        assert attempts == ["shared.toml"]

    def test_creation_race_uses_next_name(self, tmp_path, monkeypatch):
        source = write_skin(tmp_path / "race.toml")
        shelf = tmp_path / "skins"
        real_link = os.link
        raced = False

        def racing_link(temporary, destination):
            nonlocal raced
            if not raced:
                raced = True
                Path(destination).write_bytes(b"competitor")
            real_link(temporary, destination)

        monkeypatch.setattr(os, "link", racing_link)
        assert import_user_skin(source, shelf) == "user:race-2"
        assert (shelf / "race.toml").read_bytes() == b"competitor"

    def test_full_shelf_and_invalid_source_do_not_write(self, tmp_path):
        from quoth.skin import _USER_SKIN_CAP

        shelf = tmp_path / "skins"
        shelf.mkdir()
        for index in range(_USER_SKIN_CAP):
            write_skin(shelf / f"saved-{index}.toml")
        source = write_skin(tmp_path / "new.toml")
        with pytest.raises(ValueError, match="shelf is full"):
            import_user_skin(source, shelf)
        assert not (shelf / "new.toml").exists()

        bad_shelf = tmp_path / "other"
        bad = write_skin(tmp_path / "bad.toml", "kind = 'theme'\n")
        with pytest.raises(ValueError, match="kind = 'skin'"):
            import_user_skin(bad, bad_shelf)
        assert not bad_shelf.exists()


class TestApplySkin:
    def test_applies_validated_exact_bytes(self, tmp_path):
        shelf = tmp_path / "skins"
        shelf.mkdir()
        content = b"# keep\r\nkind = 'skin'\r\n[dark]\r\naccent = '#112233'\r\n"
        (shelf / "shared.toml").write_bytes(content)
        active = tmp_path / "config" / "skin.toml"

        apply_skin_id("user:shared", shelf, active)

        assert active.read_bytes() == content

    def test_invalid_shelf_file_does_not_replace_active(self, tmp_path):
        shelf = tmp_path / "skins"
        shelf.mkdir()
        (shelf / "broken.toml").write_text("not [[[")
        active = write_skin(tmp_path / "active.toml")

        with pytest.raises(ValueError, match="valid TOML"):
            apply_skin_id("user:broken", shelf, active)

        assert active.read_text() == "kind = 'skin'\n"

    def test_an_unreadable_shelf_file_is_a_rejection(self, tmp_path):
        active = write_skin(tmp_path / "active.toml")

        with pytest.raises(ValueError, match="Couldn't read the skin file"):
            apply_skin_id("user:missing", tmp_path / "skins", active)
        with pytest.raises(ValueError, match="Couldn't read the skin file"):
            import_user_skin(tmp_path / "missing.toml", tmp_path / "skins")

        assert active.read_text() == "kind = 'skin'\n"

    def test_default_durably_removes_active_and_is_idempotent(self, tmp_path):
        active = write_skin(tmp_path / "active.toml")

        apply_skin_id("default", tmp_path / "skins", active)
        apply_skin_id("default", tmp_path / "skins", active)

        assert not active.exists()

    def test_apply_replaces_destination_symlink_not_target(self, tmp_path):
        shelf = tmp_path / "skins"
        shelf.mkdir()
        source = write_skin(shelf / "shared.toml")
        victim = write_skin(tmp_path / "victim.toml", "kind = 'skin'\n# victim\n")
        active = tmp_path / "active.toml"
        active.symlink_to(victim)

        apply_skin_id("user:shared", shelf, active)

        assert not active.is_symlink()
        assert active.read_bytes() == source.read_bytes()
        assert victim.read_text() == "kind = 'skin'\n# victim\n"
