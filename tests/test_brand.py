# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `brand`."""

import re
from dataclasses import replace

import pytest

from quoth import brand
from quoth.skin import DEFAULT_SKIN, Skin, SkinPalette, list_presets, skin_from_id
from tests.support import appearance, contrast, descendants, legibility, settle


def test_status_column_omits_absent_description_but_keeps_empty_label(display):
    column = brand.status_column(None, "Private", None)
    assert column.get_first_child().get_label() == "Private"
    assert column.get_first_child().get_next_sibling() is None

    column = brand.status_column(None, "Your dictation key", "")
    assert column.get_last_child().get_label() == ""
    assert column.get_first_child() != column.get_last_child()


# Every role set, each to its own colour, so a role wired to the wrong one shows.
_ROLES = SkinPalette(
    window="#111111",
    view="#222222",
    card="#333333",
    popover="#444444",
    text="#555555",
    accent="#666666",
    on_accent="#777777",
)


def _defines(sheet: str) -> dict[str, str]:
    """The colours a stylesheet's `@define-color` lines set, by name."""
    return dict(re.findall(r"@define-color (\w+) ([^;]+);", sheet))


@pytest.mark.parametrize("dark", [False, True])
def test_every_set_role_reaches_its_adwaita_colours(dark):
    skin = Skin(**{"dark" if dark else "light": _ROLES})
    expected = {
        "accent_bg_color": "#666666",
        "accent_fg_color": "#777777",
        "window_bg_color": "#111111",
        "headerbar_bg_color": "#111111",
        "headerbar_backdrop_color": "#111111",
        "view_bg_color": "#222222",
        "dialog_bg_color": "#222222",
        "card_bg_color": "#333333",
        "popover_bg_color": "#444444",
        **{
            f"{role}_fg_color": "#555555"
            for role in ("window", "view", "headerbar", "card", "dialog", "popover")
        },
    }

    defines = _defines(brand.css(skin, dark=dark))

    assert {name: defines.get(name) for name in expected} == expected


def test_dark_cards_without_a_colour_are_a_wash_of_the_text():
    skin = Skin(dark=replace(_ROLES, card=None))

    defines = _defines(brand.css(skin, dark=True))

    assert defines["card_bg_color"] == f"alpha(#555555, {brand.SURFACE_WASH})"


def test_animations_draw_in_the_accent():
    sheet = brand.css(dark=True)
    assert "drawingarea.quoth-animation { color: @accent_color; }" in sheet


def test_dark_scheme_adds_the_brand_red_and_card_outlines():
    sheet = brand.css(dark=True)
    assert "@define-color destructive_bg_color #9a1f2e;" in sheet
    assert "@define-color destructive_fg_color #f2f0ea;" in sheet
    assert "@define-color error_color #dc5a4a;" in sheet
    assert (
        ".card, list.boxed-list { box-shadow: 0 0 0 1px alpha(currentColor, 0.08); }"
        in sheet
    )


def test_light_scheme_keeps_adwaita_background():
    sheet = brand.css(dark=False)
    assert ".quoth-welcome drawingarea.quoth-animation { color: #f2f0ea; }" in sheet
    assert "@define-color window_bg_color" not in sheet


@pytest.mark.parametrize(
    ("accent", "dark", "standalone"),
    [
        ("#0e0e11", False, "#0e0e11"),
        ("#f2f0ea", True, "#f2f0ea"),
        ("#ffff00", False, "#6f6800"),
        ("#0000ff", True, "#5fb9ff"),
    ],
)
def test_standalone_accent_uses_adwaita_lightness_rule(accent, dark, standalone):
    palette = DEFAULT_SKIN.dark if dark else DEFAULT_SKIN.light
    skin = Skin(**{"dark" if dark else "light": replace(palette, accent=accent)})

    sheet = brand.css(skin, dark=dark)

    assert f"accent_bg_color {accent}" in sheet
    assert f"accent_color {standalone}" in sheet


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("skin_id", list_presets())
def test_bundled_skins_keep_text_and_accents_readable(tmp_path, skin_id, dark):
    preset = skin_from_id(skin_id, tmp_path)
    palette = preset.dark if dark else preset.light
    accent_text = brand._standalone_accent(palette.accent, dark=dark)

    assert contrast(palette.on_accent, palette.accent) >= 4.5
    for surface in (palette.window, palette.view, palette.card, palette.popover):
        assert contrast(palette.text, surface) >= 4.5
        assert contrast(accent_text, surface) >= 4.5


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("skin_id", ["default", *list_presets()])
def test_a_suggested_action_reads_in_every_skin(display, tmp_path, skin_id, dark):
    from gi.repository import Adw, Gtk

    window = Adw.Window(default_width=320, default_height=120)
    button = Gtk.Button(label="View release", valign=Gtk.Align.CENTER)
    button.add_css_class("suggested-action")
    window.set_content(button)
    with appearance(brand.css(skin_from_id(skin_id, tmp_path), dark=dark), dark=dark):
        try:
            window.present()
            label = button.get_child()
            hide = "button.suggested-action > label { color: transparent; }"
            assert legibility(window, label, hide=hide) >= 4.5
        finally:
            window.destroy()


def test_new_gtk_also_receives_the_standalone_accent_variable(monkeypatch):
    monkeypatch.setattr(brand.Gtk, "get_minor_version", lambda: 16)

    sheet = brand.css(
        Skin(light=replace(DEFAULT_SKIN.light, accent="#ffff00")), dark=False
    )

    assert ":root { --accent-color: #6f6800; }" in sheet


def test_set_light_cards_are_outlined():
    sheet = brand.css(Skin(light=_ROLES), dark=False)
    assert ".card, list.boxed-list { box-shadow" in sheet


def test_partial_light_text_keeps_native_card_surfaces():
    skin = Skin(light=replace(DEFAULT_SKIN.light, text="#123456"))

    sheet = brand.css(skin, dark=False)

    assert "window_fg_color #123456" in sheet
    assert "card_bg_color" not in sheet
    assert ".card, list.boxed-list" not in sheet


def test_high_contrast_keeps_adwaita_colours_and_brand_geometry():
    sheet = brand.css(dark=True, high_contrast=True)
    assert "@define-color" not in sheet
    assert "drawingarea.quoth-animation { color: @accent_color; }" in sheet
    assert "button.quoth-cta" in sheet
    assert "alpha(currentColor, 0.25)" in sheet
    assert ".quoth-keycap, .quoth-keycap.pressed { background: none;" in sheet
    assert "button.quoth-ghost, button.quoth-ghost:hover { background: none;" in sheet
    assert "border-bottom: 1px solid @borders" in sheet
    assert "background-color: alpha(@card_shade_color, 0.5)" in sheet


def test_normal_settings_surfaces_are_scoped_to_settings():
    sheet = brand.css(dark=True)

    assert ".quoth-settings row:not(:last-child)" in sheet
    assert f"alpha(currentColor, {brand.SURFACE_LINE})" in sheet
    assert ".quoth-settings row.expander list.nested" in sheet
    assert "background-color: alpha(currentColor, 0.04)" in sheet
    assert "row:not(:last-child)" not in sheet.replace(
        ".quoth-settings row:not(:last-child)", ""
    )


def test_artwork_fallback_is_ink_in_the_light_scheme():
    sheet = brand.css(dark=False)

    assert ".quoth-welcome, .quoth-artwork { background-color: #08080a; }" in sheet


def test_reduced_motion_keeps_pressed_state_without_css_animation():
    sheet = brand.css(dark=True, motion_enabled=False)

    assert ".quoth-keycap.pressed { box-shadow: 0 2px" in sheet
    assert "transform: translateY(4px)" in sheet
    assert ".quoth-keycap, .quoth-keycap.pressed { transition: none; }" in sheet
    assert ".quoth-support:hover image.quoth-heart { animation: none; }" in sheet


def test_pointer_cursors_follow_live_interactivity_from_the_root(display):
    from gi.repository import Adw, Gtk

    root = Gtk.Button(label="root")
    inner = Gtk.Button(label="inner")
    inert = Gtk.ListBoxRow()
    active = Gtk.ListBoxRow(activatable=True)
    active.connect("activate", lambda _row: None)
    target = Gtk.Switch()
    targeted = Adw.ActionRow(activatable=True)
    targeted.add_suffix(target)
    targeted.set_activatable_widget(target)
    targeted.connect("activated", lambda _row: None)
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    box.append(inner)
    box.append(inert)
    box.append(active)
    box.append(targeted)
    root.set_child(box)

    brand.pointer_cursors(root)

    assert root.get_cursor().get_name() == "pointer"
    assert inner.get_cursor().get_name() == "pointer"
    assert inert.get_cursor().get_name() == "default"
    assert active.get_cursor().get_name() == "pointer"
    assert targeted.get_cursor().get_name() == "pointer"

    inner.set_sensitive(False)
    active.set_activatable(False)
    target.set_sensitive(False)
    assert inner.get_cursor().get_name() == "default"
    assert active.get_cursor().get_name() == "default"
    assert targeted.get_cursor().get_name() == "default"

    replacement = Gtk.Switch(sensitive=False)
    targeted.add_suffix(replacement)
    targeted.set_activatable_widget(replacement)
    target.set_sensitive(True)
    assert targeted.get_cursor().get_name() == "default"
    replacement.set_sensitive(True)
    assert targeted.get_cursor().get_name() == "pointer"

    combo = Adw.ComboRow()
    expander = Adw.ExpanderRow()
    box.append(combo)
    box.append(expander)
    brand.pointer_cursors(root)
    assert combo.get_cursor().get_name() == "default"
    assert expander.get_cursor().get_name() == "pointer"

    combo.set_model(Gtk.StringList.new(["Automatic"]))
    expander.set_show_enable_switch(True)
    expander.set_enable_expansion(False)
    assert combo.get_cursor().get_name() == "pointer"
    assert expander.get_cursor().get_name() == "default"

    inner.set_sensitive(True)
    inner.set_can_target(False)
    assert inner.get_cursor().get_name() == "default"


def test_expander_pointer_covers_real_hit_targets_and_native_activation(display):
    from gi.repository import Adw, Graphene, Gtk

    expander = Adw.ExpanderRow(title="App skin", expanded=True)
    import_button = Gtk.Button(label="Import…", valign=Gtk.Align.CENTER)
    expander.add_suffix(import_button)
    available = Adw.ActionRow(title="Available", activatable=True)
    available.connect("activated", lambda _row: None)
    current = Adw.ActionRow(title="Current")
    unhandled = Adw.ActionRow(title="Unavailable", activatable=True)
    divider = Gtk.ListBoxRow(activatable=False, selectable=False)
    divider.set_child(Gtk.Label(label="Your skins"))
    for row in (available, divider, current, unhandled):
        expander.add_row(row)

    rows = Gtk.ListBox()
    rows.append(expander)
    window = Gtk.Window(child=rows, default_width=500, default_height=300)

    def point_for(widget):
        point = Graphene.Point()
        point.init(widget.get_width() / 2, widget.get_height() / 2)
        success, translated = widget.compute_point(expander, point)
        assert success
        return translated.x, translated.y

    def effective_cursor_at(x, y):
        widget = expander.pick(x, y, Gtk.PickFlags.INSENSITIVE)
        assert widget is not None
        while widget is not None:
            if cursor := widget.get_cursor():
                return cursor.get_name()
            widget = widget.get_parent()
        return None

    try:
        window.present()
        settle()
        brand.pointer_cursors(expander)

        arrow = next(
            w for w in descendants(expander) if w.has_css_class("expander-row-arrow")
        )
        header = expander.pick(8, 8, Gtk.PickFlags.INSENSITIVE)
        while header is not None and not isinstance(header, Adw.ActionRow):
            header = header.get_parent()
        assert header is not None and header.has_css_class("header")
        assert header.get_cursor() is None
        assert effective_cursor_at(8, 8) == "pointer"
        assert effective_cursor_at(*point_for(arrow)) == "pointer"
        assert effective_cursor_at(*point_for(import_button)) == "pointer"
        assert effective_cursor_at(*point_for(available)) == "pointer"
        assert effective_cursor_at(*point_for(divider)) == "default"
        assert effective_cursor_at(*point_for(current)) == "default"
        assert effective_cursor_at(*point_for(unhandled)) == "default"

        import_button.set_sensitive(False)
        assert effective_cursor_at(*point_for(import_button)) == "default"
        import_button.set_sensitive(True)
        assert effective_cursor_at(*point_for(import_button)) == "pointer"

        expander.set_sensitive(False)
        assert effective_cursor_at(8, 8) == "default"
        expander.set_sensitive(True)
        assert effective_cursor_at(8, 8) == "pointer"

        expander.set_show_enable_switch(True)
        expander.set_enable_expansion(False)
        settle()
        assert effective_cursor_at(8, 8) == "default"
        expander.set_enable_expansion(True)
        assert effective_cursor_at(8, 8) == "pointer"

        expander.set_expanded(True)
        expander.emit("activate")
        settle()
        assert not expander.get_expanded()
        expander.emit("activate")
        settle()
        assert expander.get_expanded()
    finally:
        window.destroy()


def test_pointer_cursors_follow_inherited_sensitivity_and_are_idempotent(display):
    from unittest.mock import patch

    from gi.repository import Gtk

    inner = Gtk.Button(label="inner")
    box = Gtk.Box()
    box.append(inner)
    brand.pointer_cursors(box)
    brand.pointer_cursors(box)

    assert inner.get_cursor().get_name() == "pointer"

    real_set_cursor = Gtk.Widget.set_cursor
    updates = []

    def set_cursor(widget, cursor):
        updates.append(cursor.get_name())
        real_set_cursor(widget, cursor)

    # A second walk binds no second handler, so one state change updates once.
    with patch.object(Gtk.Button, "set_cursor", set_cursor):
        box.set_sensitive(False)
    assert updates == ["default"]
    assert inner.get_cursor().get_name() == "default"
    box.set_sensitive(True)
    assert inner.get_cursor().get_name() == "pointer"


def test_pointer_target_observer_does_not_retain_removed_row(display):
    import gc
    import weakref

    from gi.repository import Adw, Gtk

    target = Gtk.Switch()
    row = Adw.ActionRow(activatable=True)
    row.add_suffix(target)
    row.set_activatable_widget(target)
    box = Gtk.Box()
    box.append(row)
    brand.pointer_cursors(box)
    row_ref = weakref.ref(row)

    box.remove(row)
    del row
    gc.collect()

    assert row_ref() is None


def test_stylesheets_parse():
    """GTK drops a rejected CSS rule with only a warning, so every variant is parsed."""
    from gi.repository import Gtk

    errors: list[str] = []
    for dark in (True, False):
        for high_contrast in (True, False):
            for motion_enabled in (True, False):
                provider = Gtk.CssProvider()
                provider.connect(
                    "parsing-error", lambda _p, _s, err: errors.append(err.message)
                )
                provider.load_from_string(
                    brand.css(
                        dark=dark,
                        high_contrast=high_contrast,
                        motion_enabled=motion_enabled,
                    )
                )
    assert errors == []


def test_skin_switch_updates_inherited_colour_and_default_restores_it(display):
    from gi.repository import Adw, Gdk, Gtk

    Adw.init()
    provider = Gtk.CssProvider()
    display = Gdk.Display.get_default()
    Gtk.StyleContext.add_provider_for_display(
        display, provider, Gtk.STYLE_PROVIDER_PRIORITY_USER + 20
    )
    label = Gtk.Label(label="Skin")
    window = Adw.Window(content=label)

    try:
        provider.load_from_string(brand.css(dark=False))
        window.present()
        settle()
        default = label.get_color().to_string()

        custom = Skin(
            light=replace(DEFAULT_SKIN.light, text="#123456"),
            dark=DEFAULT_SKIN.dark,
        )
        provider.load_from_string(brand.css(custom, dark=False))
        settle()
        assert label.get_color().to_string() == "rgb(18,52,86)"

        provider.load_from_string(brand.css(dark=False))
        settle()
        assert label.get_color().to_string() == default
    finally:
        window.destroy()
        Gtk.StyleContext.remove_provider_for_display(display, provider)


def test_install_reuses_provider_and_tracks_both_appearance_modes(monkeypatch):
    from unittest.mock import Mock

    provider = Mock()
    manager = Mock()
    manager.get_dark.return_value = False
    manager.get_high_contrast.return_value = False
    callbacks = {}
    manager.connect.side_effect = lambda signal, callback: callbacks.setdefault(
        signal, callback
    )
    monkeypatch.setattr(brand, "_PROVIDER", None)
    monkeypatch.setattr(brand.Gtk, "CssProvider", lambda: provider)
    monkeypatch.setattr(brand.Adw.StyleManager, "get_default", lambda: manager)
    add_provider = Mock()
    monkeypatch.setattr(
        brand.Gtk.StyleContext, "add_provider_for_display", add_provider
    )
    monkeypatch.setattr(brand, "_queue_animation_draws", Mock())

    custom = Skin(
        light=replace(DEFAULT_SKIN.light, accent="#123456"),
        dark=DEFAULT_SKIN.dark,
    )
    monkeypatch.setattr(brand, "_SKIN", custom)
    brand.install()
    installed = brand._PROVIDER
    brand.install()
    brand.apply_skin(DEFAULT_SKIN)
    manager.get_dark.return_value = True
    callbacks["notify::dark"]()
    manager.get_high_contrast.return_value = True
    callbacks["notify::high-contrast"]()

    assert brand._PROVIDER is installed
    assert manager.connect.call_count == 2
    assert add_provider.call_count == 1
    assert (
        "accent_bg_color #123456" in provider.load_from_string.call_args_list[0].args[0]
    )
    assert (
        "accent_bg_color #123456" in provider.load_from_string.call_args_list[1].args[0]
    )
    assert (
        f"window_bg_color {DEFAULT_SKIN.dark.window}"
        in provider.load_from_string.call_args_list[-2].args[0]
    )
    assert "@define-color" not in provider.load_from_string.call_args_list[-1].args[0]


def test_motion_change_reloads_the_same_provider_only_when_changed(monkeypatch):
    from unittest.mock import Mock

    provider = Mock()
    provider.find_property.return_value = None
    manager = Mock()
    manager.get_dark.return_value = True
    manager.get_high_contrast.return_value = False
    monkeypatch.setattr(brand, "_PROVIDER", provider)
    monkeypatch.setattr(brand, "_MOTION_ENABLED", True)
    monkeypatch.setattr(brand.Adw.StyleManager, "get_default", lambda: manager)
    monkeypatch.setattr(brand, "_queue_animation_draws", Mock())

    brand.set_motion_enabled(False)
    brand.set_motion_enabled(False)

    assert brand._PROVIDER is provider
    assert provider.load_from_string.call_count == 1
    assert "animation: none" in provider.load_from_string.call_args.args[0]
    provider.set_property.assert_not_called()  # this GTK has no such property


def test_new_gtk_provider_receives_reduced_motion_state(monkeypatch):
    from unittest.mock import Mock

    provider = Mock()
    provider.find_property.return_value = object()
    manager = Mock()
    manager.get_dark.return_value = False
    manager.get_high_contrast.return_value = False
    monkeypatch.setattr(brand, "_PROVIDER", provider)
    monkeypatch.setattr(brand, "_MOTION_ENABLED", False)
    monkeypatch.setattr(brand.Adw.StyleManager, "get_default", lambda: manager)
    monkeypatch.setattr(brand, "_queue_animation_draws", Mock())

    brand._reload()

    provider.set_property.assert_called_once_with("prefers-reduced-motion", True)


def test_skin_reload_queues_only_tagged_drawings(monkeypatch, display):
    from unittest.mock import patch

    from gi.repository import Gtk

    tagged = Gtk.DrawingArea()
    tagged.add_css_class("quoth-animation")
    ordinary = Gtk.DrawingArea()
    box = Gtk.Box()
    box.append(tagged)
    box.append(ordinary)
    window = Gtk.Window(child=box)
    monkeypatch.setattr(Gtk.Window, "get_toplevels", lambda: [window])

    queued = []
    with patch.object(Gtk.DrawingArea, "queue_draw", lambda area: queued.append(area)):
        brand._queue_animation_draws()

    assert queued == [tagged]
    window.destroy()
