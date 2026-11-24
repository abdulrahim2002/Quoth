# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `overlay_section`.

The pure view helpers run anywhere; `TestOverlaySectionRuntime` builds real widgets and
needs a display.
"""

import dataclasses
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from quoth.atomic_write import diagnostic
from quoth.config import Config
from quoth.overlay_section import (
    _CUSTOM_THEME,
    OverlaySection,
    ThemeBridge,
    _theme_action,
    _theme_label,
    _theme_texts,
    _tier_chip,
)
from quoth.theme import (
    DEFAULT_THEME,
    ThemeMeta,
    apply_theme_id,
    identify_theme,
    import_user_theme,
    list_presets,
    list_user_themes,
    theme_from_id,
)
from tests.support import accessible_mismatch, descendants, settle, wait_until


class TestThemeLabel:
    """Preset names are file stems; the picker shows them as titles."""

    def test_titlecases_and_unhyphenates(self):
        assert _theme_label("catppuccin-mocha") == "Catppuccin Mocha"

    def test_single_word(self):
        assert _theme_label("nord") == "Nord"

    def test_custom_names_the_file_it_stands_for(self):
        assert _theme_label(_CUSTOM_THEME) == "Custom (theme.toml)"

    def test_custom_sentinel_cannot_collide_with_a_preset(self):
        # The picker indexes selection by name, so a real preset called "custom"
        # would make the sentinel ambiguous.
        assert _CUSTOM_THEME not in list_presets()

    def test_every_shipped_preset_gets_a_label(self):
        assert all(_theme_label(name) for name in list_presets())

    def test_manual_filename_label_is_printable_and_bounded(self):
        theme_id = "user:shared\n\u202e" + "x" * 100

        label = _theme_label(theme_id)

        assert len(label) <= 60
        assert all(character.isprintable() for character in label)
        assert "\u202e" not in label
        assert "\n" not in diagnostic(theme_id)
        assert len(diagnostic(theme_id)) <= 63


class TestThemeAction:
    """Only a hand-edited theme.toml is worth a prompt: a preset overwrites the file."""

    @pytest.mark.parametrize(
        ("on_disk", "action"),
        [
            ("dracula", "apply"),
            # A pristine preset (or no file at all) holds nothing of theirs to lose.
            ("default", "apply"),
            # Hand edits: asks first.
            (None, "confirm"),
            # Unreachable via the UI (the active row is built non-activatable), so
            # this pins the backstop rather than the path.
            ("nord", "ignore"),
        ],
    )
    def test_applying_a_preset(self, on_disk, action):
        assert _theme_action("nord", on_disk) == action

    def test_every_preset_is_actionable_from_stock(self):
        assert all(
            _theme_action(n, "default") in {"apply", "ignore"} for n in list_presets()
        )

    def test_manual_id_is_applied_unchanged(self):
        theme_id = "user:shared\n\u202e" + "x" * 100
        section = OverlaySection(
            SimpleNamespace(),
            SimpleNamespace(current=lambda: "default"),
            toast=Mock(),
        )
        section._apply_theme = Mock()

        section._on_theme_selected(theme_id)

        section._apply_theme.assert_called_once_with(theme_id)


class TestThemeTexts:
    """Row texts: the [meta] name over the filename label, the author as a credit."""

    def test_meta_name_wins_over_the_label(self):
        themed = dataclasses.replace(
            DEFAULT_THEME, meta=ThemeMeta(name="Synthwave", author="Sam")
        )
        assert _theme_texts("user:synth-wave", themed) == ("Synthwave", "by Sam")

    def test_falls_back_to_the_filename_label(self):
        title, subtitle = _theme_texts("user:synth-wave", DEFAULT_THEME)
        assert title == "Synth Wave"
        assert subtitle is None

    def test_preset_ids_keep_their_labels(self):
        assert _theme_texts("catppuccin-mocha", DEFAULT_THEME)[0] == "Catppuccin Mocha"


class TestTierChip:
    """The maturity-chip contract: only defined tiers may badge a row."""

    def test_unknown_tier_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown feature tier"):
            _tier_chip("beta")  # release-tag words are not feature tiers


def test_preview_cycle_stops_and_resets_when_live_theme_disables_motion(monkeypatch):
    disabled = dataclasses.replace(
        DEFAULT_THEME,
        animation=dataclasses.replace(DEFAULT_THEME.animation, enabled=False),
    )
    section = OverlaySection(
        SimpleNamespace(),
        SimpleNamespace(live=lambda: disabled),
        toast=Mock(),
    )
    section._theme_preview = Mock()
    section._theme_orbs = [Mock()]
    section._preview_state = "transcribing"
    section._cycle_id = 17
    remove = Mock()
    monkeypatch.setattr("quoth.overlay_section.GLib.source_remove", remove)

    section.set_motion_enabled(True)

    section._theme_preview.set_motion_enabled.assert_called_once_with(False)
    remove.assert_called_once_with(17)
    assert section._cycle_id is None
    assert section._preview_state == "recording"
    section._theme_preview.set_state.assert_called_once_with("recording")
    section._theme_orbs[0].set_motion_enabled.assert_called_once_with(True)


@pytest.mark.usefixtures("display")
class TestOverlaySectionRuntime:
    """Pins GTK calls whose PyGObject wiring cannot be proved headlessly."""

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
        *,
        theme_ids=("default",),
        current="default",
        themes=None,
        error=None,
        motion_enabled=True,
        toast=None,
    ):
        from gi.repository import Adw

        Adw.init()
        window = Adw.PreferencesWindow()
        page = Adw.PreferencesPage()
        errors = [error]
        theme_map = themes or {theme_id: DEFAULT_THEME for theme_id in theme_ids}
        bridge = ThemeBridge(
            presets=list(theme_ids),
            user_themes=lambda: [],
            current=lambda: current,
            theme_of=lambda theme_id: theme_map[theme_id],
            live=lambda: theme_map.get(current, DEFAULT_THEME),
            load_error=lambda: errors[0],
            apply=Mock(),
            import_file=Mock(),
        )
        section = OverlaySection(
            Config(),
            bridge,
            toast=toast or Mock(),
            motion_enabled=motion_enabled,
        )
        section.build(page, window)
        window.add(page)
        self._built.append((section, window))
        return section, bridge, window, errors

    def test_accessible_label_marshals(self):
        from gi.repository import Adw, Gtk

        Adw.init()
        row = Adw.ActionRow()
        row.update_property(
            [Gtk.AccessibleProperty.LABEL], ["Orb overlay (Experimental)"]
        )

    @pytest.mark.parametrize("tier", ["preview", "experimental"])
    def test_tier_chip_constructs(self, tier):
        chip = _tier_chip(tier)
        assert chip.get_label() == tier.capitalize()
        assert chip.has_css_class("quoth-tier") and chip.has_css_class(tier)

    def test_native_meter_tracks_real_levels_and_exposes_its_name(self):
        from gi.repository import Gtk

        section, _bridge, _window, _errors = self._build()
        assert Gtk.test_accessible_has_role(section._meter, Gtk.AccessibleRole.METER)
        assert Gtk.test_accessible_has_property(
            section._meter, Gtk.AccessibleProperty.LABEL
        )
        section.push_level(0.42)
        assert section._meter.get_value() == pytest.approx(0.42)

    def test_load_error_uses_overlay_group_description(self):
        section, _bridge, _window, errors = self._build(
            error="Couldn't load the theme. Keeping the current appearance."
        )
        assert section._preview_group.get_description() == errors[0]
        errors[0] = None
        section.refresh_theme()
        assert section._preview_group.get_description() == ""

    @pytest.mark.parametrize(
        "error", [ValueError("Invalid theme"), FileNotFoundError()]
    )
    @pytest.mark.parametrize("current", ["default", "user:changing"])
    def test_stale_shelf_row_is_omitted_without_a_default_preview(self, error, current):
        section, bridge, _window, _errors = self._build(current=current)
        bridge.user_themes = lambda: ["user:changing"]
        bridge.theme_of = lambda _theme_id: DEFAULT_THEME
        section._render_theme_rows()
        assert "user:changing" in section._orb_cache
        assert section._shelf_divider is not None

        def read_theme(theme_id):
            if theme_id == "user:changing":
                raise error
            return DEFAULT_THEME

        bridge.theme_of = read_theme
        section._render_theme_rows()

        assert list(section._theme_rows) == ["default"]
        assert list(section._orb_cache) == ["default"]
        assert len(section._theme_orbs) == 1
        assert section._shelf_divider is None
        assert section._theme_expander.get_subtitle() == (
            "Default" if current == "default" else "Custom (theme.toml)"
        )
        if current == "user:changing":
            bridge.live = lambda: dataclasses.replace(
                DEFAULT_THEME, meta=ThemeMeta(name="My appearance")
            )
            section._render_theme_rows()
            assert section._theme_expander.get_subtitle() == (
                "My appearance (theme.toml)"
            )

        bridge.theme_of = lambda _theme_id: DEFAULT_THEME
        section._render_theme_rows()
        assert section._theme_rows["user:changing"].get_activatable() == (
            current != "user:changing"
        )
        assert section._shelf_divider is not None

    def test_a_cut_preview_failure_still_shows_the_home_as_a_tilde(
        self, long_home, caplog
    ):
        section, bridge, _window, _errors = self._build()
        bridge.user_themes = lambda: ["user:gone"]
        error = FileNotFoundError(
            2, "No such file or directory", str(long_home / ".config/quoth/gone.toml")
        )
        bridge.theme_of = Mock(side_effect=error)
        section._render_theme_rows()
        warning = next(m for m in caplog.messages if "Could not preview" in m)
        assert "~/.config" in warning
        assert "maximilian" not in warning

    def test_real_validation_errors_leave_active_theme_and_picker_usable(
        self, chooser, tmp_path
    ):
        from gi.repository import Gio

        toast = Mock()
        section, bridge, _window, _errors = self._build(toast=toast)
        source = tmp_path / "shared.toml"
        shelf = tmp_path / "themes"
        active = tmp_path / "theme.toml"
        before = b'[colors]\naccent = "#654321"\n'
        active.write_bytes(before)
        bridge.user_themes = lambda: list_user_themes(shelf)
        bridge.theme_of = lambda theme_id: theme_from_id(theme_id, shelf)
        bridge.import_file = lambda path: import_user_theme(path, shelf)
        bridge.apply = lambda theme_id: apply_theme_id(theme_id, shelf, active)
        source.write_text('[colors]\naccent = "invalid"\n')
        section._import_button.emit("clicked")
        chooser.answer(Gio.File.new_for_path(str(source)))

        message = toast.call_args.args[0]
        assert "accent" in message
        assert "imported" not in message
        assert not shelf.exists()
        assert active.read_bytes() == before
        assert section._import_button.get_sensitive()

        source.write_text('[colors]\naccent = "#123456"\n')
        theme_id = bridge.import_file(source)
        section.refresh_theme()
        assert theme_id in section._theme_rows

        (shelf / "shared.toml").write_text("[colors]\naccent = false\n")
        section._theme_rows[theme_id].activate()

        assert "accent" in toast.call_args.args[0]
        assert active.read_bytes() == before
        assert theme_id not in section._theme_rows
        assert section._shelf_divider is None
        assert "default" in section._theme_rows

    def test_rows_name_the_active_choice_and_keep_a_decorative_check(self):
        # The page is in a window, where a list box overwrites a row's selected
        # state, so the active row's name says it.
        from gi.repository import Gtk

        section, bridge, window, _errors = self._build(theme_ids=("default", "nord"))
        label = Gtk.AccessibleProperty.LABEL

        def named_active(theme):
            row = section._theme_rows[theme]
            if not Gtk.test_accessible_has_property(row, label):
                return False
            assert row.get_root() is window
            assert accessible_mismatch(row, label, f"{row.get_title()} Active") is None
            # The row's template labels it by its title, which outranks the label.
            assert accessible_mismatch(row, Gtk.AccessibleRelation.LABELLED_BY) is None
            return True

        assert named_active("default")
        assert not named_active("nord")
        bridge.current = lambda: "nord"
        section.refresh_theme()
        assert not named_active("default")
        assert named_active("nord")
        bridge.current = lambda: "default"
        section.refresh_theme()

        checks = [
            widget
            for widget in descendants(section._theme_rows["default"])
            if isinstance(widget, Gtk.Image)
            and widget.get_icon_name() == "object-select-symbolic"
        ]
        assert len(checks) == 1
        assert checks[0].get_accessible_role() == Gtk.AccessibleRole.PRESENTATION
        assert section._theme_rows["default"].get_focusable()

        window.present()
        section._theme_expander.set_expanded(True)
        settle()
        assert section._theme_rows["default"].grab_focus()
        assert window.get_focus() is section._theme_rows["default"]
        section._theme_rows["default"].activate()
        section._theme_rows["nord"].activate()
        section._theme_bridge.apply.assert_called_once_with("nord")

    def test_orb_cache_is_owned_by_row_identity_and_pruned(self):
        changed = dataclasses.replace(
            DEFAULT_THEME,
            palette=dataclasses.replace(
                DEFAULT_THEME.palette, accent=(0.1, 0.2, 0.3, 1.0)
            ),
        )
        themes = {"first": DEFAULT_THEME, "second": DEFAULT_THEME}
        section, _bridge, _window, _errors = self._build(
            theme_ids=("first", "second"),
            current="first",
            themes=themes,
        )
        first = section._orb_cache["first"]
        second = section._orb_cache["second"]
        assert first[1] is not second[1]

        section.refresh_theme()
        assert section._orb_cache["first"] is first
        assert section._orb_cache["second"] is second

        themes["first"] = changed
        section.refresh_theme()
        assert section._orb_cache["first"][1] is not first[1]

        section._theme_bridge.presets.remove("second")
        section.refresh_theme()
        assert set(section._orb_cache) == {"first"}

    def test_real_window_advances_only_expanded_visible_row_orbs(self, monkeypatch):
        from gi.repository import Gtk

        from quoth.orb import Orb

        advances = {}
        advance = Orb.advance

        def count_advance(orb, dt):
            advances[id(orb)] = advances.get(id(orb), 0) + 1
            return advance(orb, dt)

        def frames(count):
            # Enough frames for a ticking orb to advance, so a still one proves gating.
            clock = window.get_frame_clock()
            target = clock.get_frame_counter() + count
            wait_until(lambda: clock.get_frame_counter() >= target)

        monkeypatch.setattr(Orb, "advance", count_advance)
        theme_ids = tuple(f"theme-{index}" for index in range(20))
        section, _bridge, window, _errors = self._build(
            theme_ids=theme_ids, current=theme_ids[0]
        )
        advances.clear()  # constructing an orb warms its simulation
        window.set_default_size(430, 300)
        window.present()
        row_orb_ids = {id(cached[1]) for cached in section._orb_cache.values()}
        first = section._orb_cache[theme_ids[0]][1]
        second = section._orb_cache[theme_ids[1]][1]
        last = section._orb_cache[theme_ids[-1]][1]
        frames(10)
        assert not row_orb_ids.intersection(advances)

        section._theme_expander.set_expanded(True)
        scroller = section._theme_expander.get_parent()
        while not isinstance(scroller, Gtk.ScrolledWindow):
            scroller = scroller.get_parent()
        adjustment = scroller.get_vadjustment()
        wait_until(lambda: adjustment.get_upper() - adjustment.get_page_size() > 250)
        adjustment.set_value(250)
        wait_until(lambda: id(first) in advances and id(second) in advances)
        frames(15)
        assert id(last) not in advances
        assert first is not second
        assert abs(advances[id(first)] - advances[id(second)]) <= 1
        assert section._theme_orbs[0]._viewports
        assert all(
            isinstance(viewport, Gtk.Viewport)
            for viewport in section._theme_orbs[0]._viewports
        )

        # Only the rows' orbs: an orb ticking anywhere else would count here too.
        section._theme_expander.set_expanded(False)
        before = {orb: advances.get(orb, 0) for orb in row_orb_ids}
        frames(10)
        assert {orb: advances.get(orb, 0) for orb in row_orb_ids} == before

        section._theme_expander.set_expanded(True)
        wait_until(lambda: advances[id(first)] > before[id(first)])

    def test_the_orb_row_and_switch_are_named_by_the_visible_labels(self):
        # The row's own title is empty, and a relation to it hides any label.
        from gi.repository import Adw, Gtk

        def label(root, text):
            labels = (w for w in descendants(root) if isinstance(w, Gtk.Label))
            return next(w for w in labels if w.get_label() == text)

        section, _bridge, _window, _errors = self._build()
        switch = section._orb_switch
        row = switch.get_ancestor(Adw.ActionRow)
        title, chip = label(row, "Orb overlay"), label(row, "Experimental")
        subtitle = label(row, "Replaces the pill with your theme's orb")
        for widget in (row, switch):
            assert (
                accessible_mismatch(
                    widget, Gtk.AccessibleRelation.LABELLED_BY, title, chip
                )
                is None
            )
            assert (
                accessible_mismatch(
                    widget, Gtk.AccessibleRelation.DESCRIBED_BY, subtitle
                )
                is None
            )

    def test_the_orb_switch_writes_and_follows_the_overlay_style(self):
        from quoth.pill_view import OrbView, PillView

        section, _bridge, _window, _errors = self._build()
        config = Config()
        # Settings shows every change through the section's sync.
        handler = config.connect_changed(section.sync)
        try:
            section._orb_switch.set_active(True)
            assert config.overlay_style == "orb"
            assert isinstance(section._theme_preview, OrbView)
            Config().overlay_style = "pill"  # another writer
            assert not section._orb_switch.get_active()
            assert isinstance(section._theme_preview, PillView)
            assert config.overlay_style == "pill"
        finally:
            config.settings.disconnect(handler)

    def test_the_import_button_is_named_for_what_it_imports(self):
        # A button's label names it through a relation that hides any label property.
        from gi.repository import Gtk

        section, _bridge, _window, _errors = self._build()
        button = section._import_button

        assert not Gtk.test_accessible_has_relation(
            button, Gtk.AccessibleRelation.LABELLED_BY
        )
        label = Gtk.AccessibleProperty.LABEL
        assert accessible_mismatch(button, label, "Import theme") is None

    def test_import_button_opens_one_disabled_chooser(self, chooser):
        from gi.repository import Gio

        opened = chooser.opened
        section, _bridge, _window, _errors = self._build()

        section._import_button.emit("clicked")
        section._import_button.emit("clicked")

        assert section._import_button.get_label() == "Import…"
        assert section._import_button.get_tooltip_text() == (
            "Adds a .toml theme to this list without applying it"
        )
        assert not section._import_button.get_sensitive()
        assert len(opened) == 1
        assert isinstance(opened[0].cancellable, Gio.Cancellable)
        assert opened[0].properties["filters"].get_n_items() == 1
        theme_filter = opened[0].properties["filters"].get_item(0)
        assert theme_filter.get_name() == "Overlay themes (*.toml)"
        assert opened[0].default_filter is theme_filter
        assert theme_filter.get_attributes() == ["standard::display-name"]

    # An open picker already shows the new row when it is added; a closed one shows
    # it only once it expands.
    @pytest.mark.parametrize("expanded", [False, True])
    def test_successful_import_focuses_the_new_real_row(
        self, chooser, tmp_path, expanded
    ):
        from gi.repository import Gio

        themes = []

        def import_file(_path):
            themes.append("user:shared")
            return "user:shared"

        toast = Mock()
        section, bridge, window, _errors = self._build(toast=toast)
        bridge.user_themes = lambda: themes
        bridge.theme_of = lambda _theme_id: DEFAULT_THEME
        bridge.import_file.side_effect = import_file
        window.present()
        section._theme_expander.set_expanded(expanded)
        wait_until(lambda: section._theme_rows["default"].get_mapped() == expanded)
        section._import_button.emit("clicked")

        chooser.answer(Gio.File.new_for_path(str(tmp_path / "shared.toml")))
        row = section._theme_rows["user:shared"]
        wait_until(lambda: window.get_focus() is row)

        assert section._theme_expander.get_expanded()
        bridge.import_file.assert_called_once_with(tmp_path / "shared.toml")
        toast.assert_called_once_with("Theme imported. Select it to apply.")
        assert section._import_button.get_sensitive()
        bridge.apply.assert_not_called()

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
        section, bridge, _window, _errors = self._build(toast=toast)
        rows = dict(section._theme_rows)
        section._import_button.emit("clicked")

        dialog = chooser.answer(
            GLib.Error.new_literal(error_domain, "cancelled", error_code)
        )

        assert dialog.finished
        bridge.import_file.assert_not_called()
        toast.assert_not_called()
        assert section._theme_rows == rows
        assert not section._theme_expander.get_expanded()
        assert section._import_button.get_sensitive()
        section._import_button.emit("clicked")
        assert len(chooser.opened) == 2

    def test_chooser_failure_is_reported(self, chooser):
        from gi.repository import GLib, Gtk

        toast = Mock()
        section, bridge, _window, _errors = self._build(toast=toast)
        section._import_button.emit("clicked")

        chooser.answer(
            GLib.Error.new_literal(
                Gtk.DialogError.quark(), "chooser failed", Gtk.DialogError.FAILED
            )
        )

        bridge.import_file.assert_not_called()
        toast.assert_called_once_with("Couldn't open the theme file")

    def test_invalid_import_is_reported_without_rerender(
        self, chooser, tmp_path, caplog
    ):
        from gi.repository import Gio

        error = ValueError("That is not a Quoth theme")
        toast = Mock()
        section, bridge, _window, _errors = self._build(toast=toast)
        bridge.import_file.side_effect = error
        rows = dict(section._theme_rows)
        section._import_button.emit("clicked")

        chooser.answer(Gio.File.new_for_path(str(tmp_path / "shared.toml")))

        bridge.import_file.assert_called_once()
        toast.assert_called_once_with(str(error))
        assert section._theme_rows == rows
        assert "shared.toml" in caplog.messages[-1]
        assert caplog.messages[-1].endswith(": That is not a Quoth theme")

    def test_import_sync_failure_rerenders_actual_shelf_without_guessing_focus(
        self, chooser, tmp_path, caplog
    ):
        from gi.repository import Gio

        toast = Mock()
        section, bridge, window, _errors = self._build(toast=toast)
        bridge.import_file.side_effect = OSError("disk full")
        bridge.user_themes = lambda: ["user:shared"]  # the copy landed after all
        bridge.theme_of = lambda _theme_id: DEFAULT_THEME
        window.present()
        section._import_button.emit("clicked")

        caplog.set_level("WARNING")
        caplog.clear()  # building the section logs where settings are kept
        chooser.answer(Gio.File.new_for_path(str(tmp_path / "shared.toml")))
        row = section._theme_rows["user:shared"]
        wait_until(row.get_mapped)
        settle()  # a focus queued on the row's map would have run

        assert [(r.levelname, r.exc_info) for r in caplog.records] == [("ERROR", None)]
        assert "disk full" in caplog.messages[0]
        bridge.import_file.assert_called_once()
        assert section._theme_expander.get_expanded()
        assert window.get_focus() is not row
        toast.assert_called_once_with(
            "Couldn't confirm the import. Check the theme list."
        )

    def test_nonlocal_file_is_rejected_before_import(self, chooser, caplog):
        from gi.repository import Gio

        toast = Mock()
        section, bridge, _window, _errors = self._build(toast=toast)
        section._import_button.emit("clicked")

        # A remote location's URI names the account and the host.
        chooser.answer(Gio.File.new_for_uri("sftp://bob@host.lan/theme.toml"))

        bridge.import_file.assert_not_called()
        toast.assert_called_once_with("Choose a local theme file")
        assert caplog.messages[-1] == (
            "Rejected theme import: a sftp location has no local path"
        )

    def test_closed_window_finishes_without_side_effects(self, chooser, tmp_path):
        from gi.repository import Gio

        toast = Mock()
        section, bridge, _window, _errors = self._build(toast=toast)
        rows = dict(section._theme_rows)
        section._import_button.emit("clicked")

        section.close()
        dialog = chooser.answer(Gio.File.new_for_path(str(tmp_path / "shared.toml")))

        assert chooser.opened[0].cancellable.is_cancelled()
        assert dialog.finished
        bridge.import_file.assert_not_called()
        assert not section._import_button.get_sensitive()
        assert section._theme_rows == rows
        toast.assert_not_called()
        section._import_button.emit("clicked")
        assert len(chooser.opened) == 1

    def test_imported_theme_waits_for_confirmation_over_custom_content(
        self, chooser, tmp_path
    ):
        from gi.repository import Gio

        source = tmp_path / "shared.toml"
        source.write_text('[meta]\nname = "Shared"\n[colors]\naccent = "#123456"\n')
        shelf = tmp_path / "themes"
        active = tmp_path / "theme.toml"
        active.write_bytes(b'[colors]\naccent = "#654321"\n')
        before = active.read_bytes()
        section, bridge, window, _errors = self._build(current=None)
        bridge.current = lambda: identify_theme(active, shelf)
        bridge.user_themes = lambda: list_user_themes(shelf)
        bridge.theme_of = lambda theme_id: theme_from_id(theme_id, shelf)
        bridge.import_file = lambda path: import_user_theme(path, shelf)
        bridge.apply = lambda theme_id: apply_theme_id(theme_id, shelf, active)
        window.present()
        section._import_button.emit("clicked")

        chooser.answer(Gio.File.new_for_path(str(source)))
        section._theme_rows["user:shared"].activate()
        wait_until(lambda: window.get_visible_dialog() is not None)

        assert active.read_bytes() == before
        dialog = window.get_visible_dialog()
        section.close()  # the window closes with the question still open
        dialog.emit("response", "replace")
        assert active.read_bytes() == before
        dialog.force_close()

    def test_confirmed_replace_overwrites_custom_content(self, tmp_path):
        shelf = tmp_path / "themes"
        active = tmp_path / "theme.toml"
        active.write_bytes(b'[colors]\naccent = "#654321"\n')
        section, bridge, window, _errors = self._build(
            theme_ids=("default", "nord"), current=None
        )
        bridge.current = lambda: identify_theme(active, shelf)
        bridge.apply = lambda theme_id: apply_theme_id(theme_id, shelf, active)
        window.present()
        section._theme_rows["nord"].activate()
        wait_until(lambda: window.get_visible_dialog() is not None)

        window.get_visible_dialog().emit("response", "replace")

        assert identify_theme(active, shelf) == "nord"
        assert section._theme_expander.get_subtitle() == "Nord"

    @pytest.mark.parametrize(
        ("error", "level", "message"),
        [
            (
                ValueError("That is not a Quoth theme"),
                "WARNING",
                "That is not a Quoth theme",
            ),
            (
                OSError("directory sync failed"),
                "ERROR",
                "Couldn't confirm the change. Check the selected theme.",
            ),
        ],
    )
    def test_a_failed_apply_is_reported_and_shows_the_file(
        self, caplog, error, level, message
    ):
        toast = Mock()
        section, bridge, _window, _errors = self._build(
            theme_ids=("default", "nord"), toast=toast
        )
        bridge.apply.side_effect = error
        nord = section._theme_rows["nord"]

        caplog.set_level("WARNING")
        caplog.clear()  # building the section logs where settings are kept
        nord.activate()

        assert [(r.levelname, r.exc_info) for r in caplog.records] == [(level, None)]
        assert str(error) in caplog.messages[0]
        toast.assert_called_once_with(message)
        assert section._theme_rows["nord"] is not nord  # re-read from disk
        assert section._theme_expander.get_subtitle() == "Default"
