# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `skin_section`: the Settings app-skin chooser and picker."""

from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

from quoth.atomic_write import diagnostic
from quoth.skin import DEFAULT_SKIN, SkinMeta, load_skin
from quoth.skin_section import (
    _CUSTOM_SKIN,
    SkinBridge,
    SkinSection,
    _palette_ribbon,
    _ribbon_bands,
    _skin_action,
    _skin_label,
    _skin_texts,
)
from tests.support import accessible_mismatch, descendants, settle, wait_until


def _skin(name="", author="", description=""):
    return replace(
        DEFAULT_SKIN,
        meta=SkinMeta(name or None, author or None, description or None),
    )


def _filter_matches(file_filter, name):
    from gi.repository import Gio

    info = Gio.FileInfo()
    info.set_display_name(name)
    info.set_name(name)
    return file_filter.match(info)


def test_labels_and_metadata_text():
    assert _skin_label("default") == "Default"
    assert _skin_label(_CUSTOM_SKIN) == "Custom"
    assert _skin_label("user:soft-blue") == "Soft Blue"
    assert _skin_texts("user:soft-blue", _skin("Soft", "Sam")) == (
        "Soft",
        "by Sam",
    )


def test_manual_filename_label_is_printable_and_bounded():
    skin_id = "user:shared\n‮" + "x" * 100

    label = _skin_label(skin_id)

    assert len(label) <= 60
    assert all(character.isprintable() for character in label)
    assert "‮" not in label
    assert "\n" not in diagnostic(skin_id)
    assert len(diagnostic(skin_id)) <= 63


def test_only_hand_edited_skin_needs_confirmation():
    assert _skin_action("default", None) == "confirm"
    assert _skin_action("user:blue", "default") == "apply"
    assert _skin_action("default", "default") == "ignore"


@pytest.mark.usefixtures("display")
class TestSkinSectionRuntime:
    @pytest.fixture(autouse=True)
    def _close_built(self):
        self._built = []
        yield
        # Popped, since pytest keeps each test instance until exit, where finalising
        # a widget fails.
        while self._built:
            section, window = self._built.pop()
            section.close()
            window.destroy()

    def _build(
        self,
        current="default",
        user_skins=None,
        skin_of=None,
        load_error=None,
        presets=(),
        toast=None,
    ):
        from gi.repository import Adw

        Adw.init()
        window = Adw.PreferencesWindow()
        page = Adw.PreferencesPage()
        window.add(page)
        ids = list(user_skins or [])
        bridge = SkinBridge(
            presets=list(presets),
            user_skins=lambda: ids,
            current=lambda: current,
            skin_of=skin_of or (lambda _skin_id: _skin()),
            load_error=load_error or (lambda: None),
            apply=Mock(),
            import_file=Mock(),
        )
        section = SkinSection(bridge, toast=toast or Mock())
        section.build(page, window)
        self._built.append((section, window))
        return section, bridge, window

    def test_error_and_high_contrast_share_existing_group_description(
        self, monkeypatch
    ):
        from gi.repository import Adw

        high_contrast = [True]
        monkeypatch.setattr(
            Adw.StyleManager,
            "get_high_contrast",
            lambda manager: high_contrast[0],
        )
        errors = ["Couldn't load the skin. Keeping the current appearance."]
        section, _bridge, _window = self._build(load_error=lambda: errors[0])
        description = section._group.get_description()
        assert errors[0] in description
        assert "System colors are shown" in description

        high_contrast[0] = False
        errors[0] = None
        Adw.StyleManager.get_default().notify("high-contrast")
        assert section._group.get_description() == ""

    def test_rows_name_the_active_choice_and_keep_a_decorative_check(self):
        # The page is in a window, where a list box overwrites a row's selected
        # state, so the active row's name says it.
        from gi.repository import Gtk

        section, bridge, window = self._build(user_skins=["user:rose"])
        label = Gtk.AccessibleProperty.LABEL

        def named_active(skin):
            row = section._skin_rows[skin]
            if not Gtk.test_accessible_has_property(row, label):
                return False
            assert row.get_root() is window
            assert accessible_mismatch(row, label, f"{row.get_title()} Active") is None
            # The row's template labels it by its title, which outranks the label.
            assert accessible_mismatch(row, Gtk.AccessibleRelation.LABELLED_BY) is None
            return True

        assert named_active("default")
        assert not named_active("user:rose")
        bridge.current = lambda: "user:rose"
        section.refresh_skin()
        assert not named_active("default")
        assert named_active("user:rose")
        bridge.current = lambda: "default"
        section.refresh_skin()

        checks = [
            widget
            for widget in descendants(section._skin_rows["default"])
            if isinstance(widget, Gtk.Image)
            and widget.get_icon_name() == "object-select-symbolic"
        ]
        assert len(checks) == 1
        assert checks[0].get_accessible_role() == Gtk.AccessibleRole.PRESENTATION
        assert section._skin_rows["default"].get_focusable()

        window.present()
        section._skin_expander.set_expanded(True)
        settle()
        assert section._skin_rows["default"].grab_focus()
        assert window.get_focus() is section._skin_rows["default"]
        section._skin_rows["default"].activate()
        section._skin_rows["user:rose"].activate()
        section._bridge.apply.assert_called_once()
        assert section._bridge.apply.call_args.args[0] == "user:rose"

    def test_default_then_user_rows_with_plain_metadata(self):
        malicious = "<b>Not markup</b>"
        section, _bridge, _window = self._build(
            user_skins=["user:shared"],
            skin_of=lambda skin_id: (
                _skin(malicious, "<i>Sam</i>", "Description")
                if skin_id == "user:shared"
                else _skin()
            ),
        )
        assert list(section._skin_rows) == ["default", "user:shared"]
        row = section._skin_rows["user:shared"]
        assert row.get_title() == malicious
        assert row.get_subtitle() == "by <i>Sam</i>"
        assert not row.get_use_markup()
        assert row.get_tooltip_text() == (
            "Description\nPatterned bands show system colors or tints of the text color"
        )

    def test_presets_follow_default_and_the_divider_heads_the_shelf(self):
        section, _bridge, _window = self._build(
            current="nord", presets=["dracula", "nord"]
        )
        assert list(section._skin_rows) == ["default", "dracula", "nord"]
        assert section._shelf_divider is None
        assert not section._skin_rows["nord"].get_activatable()
        assert section._skin_expander.get_subtitle() == "Nord"

        section._bridge.user_skins = lambda: ["user:rose"]
        section.refresh_skin()

        rows = list(section._skin_rows.values())
        assert list(section._skin_rows) == [
            "default",
            "dracula",
            "nord",
            "user:rose",
        ]
        assert section._shelf_divider.get_next_sibling() is rows[-1]
        assert rows[2].get_next_sibling() is section._shelf_divider

    def test_palette_ribbons_use_active_scheme_and_stay_neutral_in_high_contrast(
        self, tmp_path
    ):
        from gi.repository import Gtk

        from quoth.skin import Skin

        light = replace(
            DEFAULT_SKIN.light,
            window="#111111",
            view="#222222",
            card="#333333",
            accent="#444444",
        )
        dark = replace(
            DEFAULT_SKIN.dark,
            window="#aaaaaa",
            view="#bbbbbb",
            card="#cccccc",
            accent="#dddddd",
        )
        skin = Skin(light=light, dark=dark)
        omitted = tmp_path / "omitted.toml"
        explicit = tmp_path / "explicit.toml"
        omitted.write_text("kind = 'skin'\n")
        explicit.write_text(
            f"kind = 'skin'\n[dark]\nwindow = '{DEFAULT_SKIN.dark.window}'\n"
            f"view = '{DEFAULT_SKIN.dark.view}'\n"
        )
        omitted_skin = load_skin(omitted)
        explicit_skin = load_skin(explicit)
        manager = Mock()
        manager.get_dark.return_value = False
        manager.get_high_contrast.return_value = False

        with patch.object(
            Gtk.DrawingArea, "add_tick_callback", autospec=True
        ) as add_tick:
            ribbon = _palette_ribbon(skin, manager)

        assert ribbon.get_content_width() == 40
        assert ribbon.get_content_height() == 40
        assert ribbon.get_accessible_role() == Gtk.AccessibleRole.PRESENTATION
        add_tick.assert_not_called()
        assert ribbon._quoth_palette_bands == (
            "#111111",
            "#222222",
            "#333333",
            "#444444",
        )
        assert _ribbon_bands(skin, dark=True, high_contrast=False) == (
            "#aaaaaa",
            "#bbbbbb",
            "#cccccc",
            "#dddddd",
        )
        assert omitted_skin == explicit_skin == DEFAULT_SKIN
        assert _ribbon_bands(
            omitted_skin, dark=True, high_contrast=False
        ) == _ribbon_bands(explicit_skin, dark=True, high_contrast=False)
        assert _ribbon_bands(omitted_skin, dark=True, high_contrast=False) == (
            DEFAULT_SKIN.dark.window,
            DEFAULT_SKIN.dark.view,
            DEFAULT_SKIN.dark.card,
            DEFAULT_SKIN.dark.accent,
        )
        assert _ribbon_bands(omitted_skin, dark=False, high_contrast=False) == (
            DEFAULT_SKIN.light.window,
            DEFAULT_SKIN.light.view,
            DEFAULT_SKIN.light.card,
            DEFAULT_SKIN.light.accent,
        )
        assert _ribbon_bands(skin, dark=False, high_contrast=True) == (
            None,
            None,
            None,
            None,
        )

    def test_ribbon_stays_square_in_a_tall_row(self):
        from gi.repository import Adw, Gtk

        ribbon = _palette_ribbon(DEFAULT_SKIN, Adw.StyleManager.get_default())
        row = Adw.ActionRow(title="Skin", subtitle="A taller subtitle")
        row.set_size_request(-1, 100)
        row.add_prefix(ribbon)
        rows = Gtk.ListBox()
        rows.append(row)
        window = Adw.Window(content=rows)
        try:
            window.present()
            wait_until(lambda: ribbon.get_width() > 0)
            assert row.get_height() > 60
            assert (ribbon.get_width(), ribbon.get_height()) == (40, 40)
        finally:
            window.destroy()

    def test_ribbon_preserves_the_callers_clip(self):
        import cairo
        from gi.repository import Adw, Gtk

        with patch.object(Gtk.DrawingArea, "set_draw_func") as set_draw:
            ribbon = _palette_ribbon(DEFAULT_SKIN, Adw.StyleManager.get_default())
        draw = set_draw.call_args.args[0]
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 40, 40)
        context = cairo.Context(surface)
        context.rectangle(0, 0, 15, 15)
        context.clip()
        expected = context.clip_extents()

        draw(ribbon, context, 40, 40)

        assert context.clip_extents() == expected
        # The ribbon's own clip rounds the corners, so the corner shows a leak.
        assert context.in_clip(0.5, 0.5)

    def test_ribbon_paints_set_roles_and_patterns_automatic_ones(self):
        import cairo
        from gi.repository import Gtk

        from quoth.skin import Skin

        manager = Mock()
        manager.get_dark.return_value = False
        manager.get_high_contrast.return_value = False
        light = replace(DEFAULT_SKIN.light, window="#ff0000", accent="#00ff00")
        skin = Skin(light=light)
        with patch.object(Gtk.DrawingArea, "set_draw_func") as set_draw:
            ribbon = _palette_ribbon(skin, manager)
        draw = set_draw.call_args.args[0]
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 40, 40)

        draw(ribbon, cairo.Context(surface), 40, 40)

        data, stride = surface.get_data(), surface.get_stride()

        def bgra(x, y):
            return tuple(data[y * stride + x * 4 : y * stride + x * 4 + 4])

        assert bgra(20, 7) == (0x00, 0x00, 0xFF, 0xFF)  # window, #ff0000
        assert bgra(20, 37) == (0x00, 0xFF, 0x00, 0xFF)  # accent, #00ff00
        # The automatic view band below the window band is hatched, not flat.
        assert len({bgra(x, 19) for x in range(8, 32)}) > 1

    def test_automatic_roles_use_the_pattern(self):
        from gi.repository import Gtk

        skin = replace(_skin(), dark=DEFAULT_SKIN.light)
        section, _bridge, _window = self._build(
            user_skins=["user:plain"],
            skin_of=lambda skin_id: skin if skin_id == "user:plain" else _skin(),
        )
        ribbons = [
            widget
            for widget in descendants(section._skin_rows["user:plain"])
            if isinstance(widget, Gtk.DrawingArea)
            and hasattr(widget, "_quoth_palette_bands")
        ]
        assert len(ribbons) == 1
        assert ribbons[0]._quoth_palette_bands[:3] == (None, None, None)

    def test_an_unreadable_shelf_skin_is_skipped(self, caplog):
        def skin_of(skin_id):
            if skin_id == "user:broken":
                raise ValueError("The skin file is not valid TOML")
            return _skin()

        caplog.set_level("WARNING")
        section, _bridge, _window = self._build(
            user_skins=["user:broken", "user:rose"], skin_of=skin_of
        )
        assert list(section._skin_rows) == ["default", "user:rose"]
        # Rows are rebuilt on every render; the same failure warns once.
        section._render_skin_rows()
        [warning] = caplog.messages
        assert "user:broken" in warning
        assert "not valid TOML" in warning

    def test_a_cut_preview_failure_still_shows_the_home_as_a_tilde(
        self, long_home, caplog
    ):
        def skin_of(skin_id):
            if skin_id == "user:gone":
                raise FileNotFoundError(
                    2,
                    "No such file or directory",
                    str(long_home / ".config/quoth/gone.toml"),
                )
            return _skin()

        self._build(user_skins=["user:gone"], skin_of=skin_of)
        [warning] = [m for m in caplog.messages if "Could not preview" in m]
        assert "~/.config" in warning
        assert "maximilian" not in warning

    def test_scheme_change_rebuilds_ribbons(self, monkeypatch):
        from gi.repository import Adw, Gtk

        dark = [False]
        monkeypatch.setattr(Adw.StyleManager, "get_dark", lambda _manager: dark[0])
        monkeypatch.setattr(
            Adw.StyleManager, "get_high_contrast", lambda _manager: False
        )
        section, _bridge, _window = self._build(user_skins=["user:rose"])
        old_row = section._skin_rows["user:rose"]

        def ribbon_bands(row):
            [bands] = [
                widget._quoth_palette_bands
                for widget in descendants(row)
                if isinstance(widget, Gtk.DrawingArea)
                and hasattr(widget, "_quoth_palette_bands")
            ]
            return bands

        assert ribbon_bands(old_row) == (
            DEFAULT_SKIN.light.window,
            DEFAULT_SKIN.light.view,
            DEFAULT_SKIN.light.card,
            DEFAULT_SKIN.light.accent,
        )
        dark[0] = True
        Adw.StyleManager.get_default().notify("dark")
        assert section._skin_rows["user:rose"] is not old_row
        assert ribbon_bands(section._skin_rows["user:rose"]) == (
            DEFAULT_SKIN.dark.window,
            DEFAULT_SKIN.dark.view,
            DEFAULT_SKIN.dark.card,
            DEFAULT_SKIN.dark.accent,
        )

    def test_custom_row_explains_unavailable_palette(self):
        section, _bridge, _window = self._build(current=None)
        row = section._skin_rows[_CUSTOM_SKIN]
        assert row.get_tooltip_text() == (
            "Palette preview unavailable for this custom skin"
        )

    def test_overwrite_alert_maps_shared_cursor_policy(self):
        from gi.repository import Gtk

        section, _bridge, window = self._build(current=None)
        window.present()
        section._skin_rows["default"].activate()
        wait_until(lambda: window.get_visible_dialog() is not None)
        dialog = window.get_visible_dialog()
        wait_until(dialog.get_mapped)
        buttons = [w for w in descendants(dialog) if isinstance(w, Gtk.Button)]
        assert [button.get_label() for button in buttons] == ["Cancel", "Replace"]
        cursors = [button.get_cursor().get_name() for button in buttons]
        assert cursors == ["pointer", "pointer"]

    @pytest.mark.parametrize("current", [None, "user:missing"])
    def test_unmatched_active_content_is_custom(self, current):
        section, _bridge, _window = self._build(current=current)
        assert list(section._skin_rows) == ["default", _CUSTOM_SKIN]
        assert section._skin_expander.get_subtitle() == "Custom"
        assert not section._skin_rows[_CUSTOM_SKIN].get_activatable()

    def test_the_import_button_is_named_for_what_it_imports(self):
        # A button's label names it through a relation that hides any label property.
        from gi.repository import Gtk

        section, _bridge, _window = self._build()
        button = section._import_button

        assert not Gtk.test_accessible_has_relation(
            button, Gtk.AccessibleRelation.LABELLED_BY
        )
        label = Gtk.AccessibleProperty.LABEL
        assert accessible_mismatch(button, label, "Import skin") is None

    def test_import_button_opens_only_one_chooser(self, chooser):
        opened = chooser.opened
        section, _bridge, window = self._build()
        section._import_button.emit("clicked")
        section._import_button.emit("clicked")

        assert not section._import_button.get_sensitive()
        assert len(opened) == 1
        assert opened[0].parent is window
        skin_filter = opened[0].properties["filters"].get_item(0)
        assert skin_filter.get_name() == "App skins (*.toml)"
        assert _filter_matches(skin_filter, "shared.toml")
        assert not _filter_matches(skin_filter, "shared.json")

    # An open picker already shows the new row when it is added; a closed one shows
    # it only once it expands.
    @pytest.mark.parametrize("expanded", [False, True])
    def test_successful_import_reveals_and_focuses_new_row(
        self, chooser, tmp_path, expanded
    ):
        from gi.repository import Gio

        skin_ids = []

        def import_file(_path):
            skin_ids.append("user:shared")
            return "user:shared"

        toast = Mock()
        section, bridge, window = self._build(toast=toast)
        bridge.user_skins = lambda: skin_ids
        bridge.import_file.side_effect = import_file
        window.present()
        section._skin_expander.set_expanded(expanded)
        wait_until(lambda: section._skin_rows["default"].get_mapped() == expanded)
        section._import_button.emit("clicked")

        chooser.answer(Gio.File.new_for_path(str(tmp_path / "shared.toml")))
        row = section._skin_rows["user:shared"]
        wait_until(lambda: window.get_focus() is row)

        assert section._skin_expander.get_expanded()
        bridge.import_file.assert_called_once_with(tmp_path / "shared.toml")
        toast.assert_called_once_with("Skin imported. Select it to apply.")
        assert section._import_button.get_sensitive()
        bridge.apply.assert_not_called()
        section._skin_rows["user:shared"].activate()
        bridge.apply.assert_called_once()
        assert bridge.apply.call_args.args[0] == "user:shared"

    @pytest.mark.parametrize(
        ("domain", "code"),
        [
            ("dialog", "CANCELLED"),
            ("dialog", "DISMISSED"),
            ("gio", "CANCELLED"),
        ],
    )
    def test_cancel_and_dismiss_are_silent(self, chooser, domain, code):
        from gi.repository import Gio, GLib, Gtk

        error_domain = (
            Gtk.DialogError.quark() if domain == "dialog" else Gio.io_error_quark()
        )
        error_code = (
            getattr(Gtk.DialogError, code)
            if domain == "dialog"
            else getattr(Gio.IOErrorEnum, code)
        )
        toast = Mock()
        section, bridge, _window = self._build(toast=toast)
        section._import_button.emit("clicked")

        chooser.answer(GLib.Error.new_literal(error_domain, "cancelled", error_code))

        bridge.import_file.assert_not_called()
        toast.assert_not_called()
        assert not section._skin_expander.get_expanded()
        assert section._import_button.get_sensitive()
        section._import_button.emit("clicked")
        assert len(chooser.opened) == 2

    def test_chooser_error_is_reported(self, chooser):
        from gi.repository import GLib, Gtk

        toast = Mock()
        section, bridge, _window = self._build(toast=toast)
        section._import_button.emit("clicked")

        chooser.answer(
            GLib.Error.new_literal(
                Gtk.DialogError.quark(), "failed", Gtk.DialogError.FAILED
            )
        )

        bridge.import_file.assert_not_called()
        toast.assert_called_once_with("Couldn't open the skin file")
        section._import_button.emit("clicked")
        assert len(chooser.opened) == 2

    def test_invalid_import_is_reported_without_rerender(
        self, chooser, tmp_path, caplog
    ):
        from gi.repository import Gio

        error = ValueError("That is not a Quoth skin")
        toast = Mock()
        section, bridge, _window = self._build(toast=toast)
        bridge.import_file.side_effect = error
        rows = dict(section._skin_rows)
        section._import_button.emit("clicked")

        chooser.answer(Gio.File.new_for_path(str(tmp_path / "shared.toml")))

        bridge.import_file.assert_called_once()
        toast.assert_called_once_with(str(error))
        assert section._skin_rows == rows
        assert "shared.toml" in caplog.messages[-1]
        assert caplog.messages[-1].endswith(": That is not a Quoth skin")

    def test_a_failed_write_rerenders_the_actual_shelf_without_guessing_focus(
        self, chooser, tmp_path
    ):
        from gi.repository import Gio

        toast = Mock()
        section, bridge, window = self._build(toast=toast)
        bridge.import_file.side_effect = OSError("directory sync failed")
        bridge.user_skins = lambda: ["user:shared"]  # the copy landed after all
        window.present()
        section._import_button.emit("clicked")

        chooser.answer(Gio.File.new_for_path(str(tmp_path / "shared.toml")))
        row = section._skin_rows["user:shared"]
        wait_until(row.get_mapped)
        settle()  # a focus queued on the row's map would have run

        bridge.import_file.assert_called_once()
        assert section._skin_expander.get_expanded()
        assert window.get_focus() is not row
        toast.assert_called_once_with(
            "Couldn't confirm the import. Check the skin list."
        )

    def test_nonlocal_file_is_rejected(self, chooser, caplog):
        from gi.repository import Gio

        toast = Mock()
        section, bridge, _window = self._build(toast=toast)
        section._import_button.emit("clicked")

        # A remote location's URI names the account and the host.
        chooser.answer(Gio.File.new_for_uri("sftp://bob@host.lan/skin.toml"))

        bridge.import_file.assert_not_called()
        toast.assert_called_once_with("Choose a local skin file")
        assert caplog.messages[-1] == (
            "Rejected skin import: a sftp location has no local path"
        )
        section._import_button.emit("clicked")
        assert len(chooser.opened) == 2

    def test_late_callback_after_close_has_no_side_effects(self, chooser, tmp_path):
        from gi.repository import Gio

        toast = Mock()
        section, bridge, _window = self._build(toast=toast)
        section._import_button.emit("clicked")

        section.close()
        dialog = chooser.answer(Gio.File.new_for_path(str(tmp_path / "shared.toml")))

        assert chooser.opened[0].cancellable.is_cancelled()
        assert dialog.finished
        bridge.import_file.assert_not_called()
        assert not section._import_button.get_sensitive()
        toast.assert_not_called()
        section._import_button.emit("clicked")
        assert len(chooser.opened) == 1

    def test_close_disconnects_appearance_signals(self):
        from gi.repository import Adw

        errors = [None]
        section, _bridge, _window = self._build(load_error=lambda: errors[0])
        rows = dict(section._skin_rows)

        section.close()
        errors[0] = "Couldn't load the skin. Keeping the current appearance."
        Adw.StyleManager.get_default().notify("dark")
        Adw.StyleManager.get_default().notify("high-contrast")

        assert section._group.get_description() == ""
        assert section._skin_rows == rows

    @pytest.mark.parametrize("current", [None, "user:missing"])
    def test_unmatched_active_content_is_replaced_only_on_confirmation(self, current):
        toast = Mock()
        section, bridge, window = self._build(current=current, toast=toast)
        window.present()
        section._skin_rows["default"].activate()
        wait_until(lambda: window.get_visible_dialog() is not None)
        dialog = window.get_visible_dialog()
        rows = dict(section._skin_rows)

        dialog.emit("response", "cancel")
        bridge.apply.assert_not_called()
        dialog.emit("response", "replace")

        bridge.apply.assert_called_once_with("default")
        toast.assert_not_called()
        assert section._skin_rows["default"] is not rows["default"]  # re-read
        dialog.force_close()

    def test_confirmation_does_nothing_after_close(self):
        section, bridge, window = self._build(current=None)
        window.present()
        section._skin_rows["default"].activate()
        wait_until(lambda: window.get_visible_dialog() is not None)
        dialog = window.get_visible_dialog()

        section.close()
        dialog.emit("response", "replace")

        bridge.apply.assert_not_called()
        dialog.force_close()

    @pytest.mark.parametrize(
        ("error", "message"),
        [
            (ValueError("That is not a Quoth skin"), "That is not a Quoth skin"),
            (
                OSError("directory sync failed"),
                "Couldn't confirm the change. Check the selected skin.",
            ),
        ],
    )
    def test_a_failed_apply_is_reported_and_shows_the_file(self, error, message):
        toast = Mock()
        section, bridge, _window = self._build(user_skins=["user:rose"], toast=toast)
        bridge.apply.side_effect = error
        rose = section._skin_rows["user:rose"]

        rose.activate()

        toast.assert_called_once_with(message)
        assert section._skin_rows["user:rose"] is not rose  # re-read from disk
        assert section._skin_expander.get_subtitle() == "Default"

    def test_manual_id_is_applied_unchanged(self):
        skin_id = "user:shared\n\u202e" + "x" * 100
        section, bridge, _window = self._build(user_skins=[skin_id])

        section._skin_rows[skin_id].activate()

        bridge.apply.assert_called_once_with(skin_id)


def test_settings_skin_is_independent_of_overlay_capability(
    display, find_widget, monkeypatch
):
    from gi.repository import Adw

    from quoth.config import Config
    from quoth.models import ModelManager
    from quoth.settings import ModelSwitcher
    from quoth.settings_window import SettingsWindow

    monkeypatch.setattr(
        "quoth.settings_window.presentation.overlay_available", lambda: False
    )
    bridge = SkinBridge(
        presets=[],
        user_skins=lambda: [],
        current=lambda: "default",
        skin_of=lambda _skin_id: _skin(),
        load_error=lambda: None,
        apply=Mock(),
        import_file=Mock(),
    )
    dialog = SettingsWindow(
        Config(),
        ModelManager(),
        ModelSwitcher(None, None, None, reconcile=lambda: None, notify=lambda _: None),
        skin_bridge=bridge,
        reconcile_engine=lambda: None,
        enumerate_microphones=lambda: [],
    )
    try:
        assert find_widget(dialog, Adw.ExpanderRow, "App skin") is not None
        assert dialog._overlay_section is None
        assert dialog._skin_section is not None
    finally:
        dialog._on_settings_close()
        dialog.destroy()
