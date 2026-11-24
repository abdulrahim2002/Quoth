# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `about`."""

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from quoth import about, brand, presentation
from quoth.skin import list_presets, skin_from_id
from tests.support import appearance, descendants, legibility, wait_until
from tests.support.gtk import _next_frame


@pytest.mark.parametrize(
    ("url", "doc"),
    [
        (presentation.NATIVE_INSTALL_URL, "README.md"),
        (about.TROUBLESHOOTING_URL, "SUPPORT.md"),
    ],
)
def test_a_doc_link_names_a_heading_of_its_file(url, doc):
    text = (Path(__file__).parents[1] / doc).read_text()
    # GitHub's heading anchors: lower case, punctuation dropped, spaces to hyphens.
    anchors = {
        re.sub(r"[^\w\- ]", "", line.lstrip("#").strip().lower()).replace(" ", "-")
        for line in text.splitlines()
        if line.startswith("#")
    }
    assert url.partition("#")[2] in anchors


def test_the_privacy_link_names_a_file_of_the_repository():
    path = about.PRIVACY_URL.removeprefix(f"{about.SOURCE_URL}/blob/main/")
    assert (Path(__file__).parents[1] / path).is_file()


def test_a_link_with_no_browser_is_reported_not_raised(caplog):
    from gi.repository import GLib

    missing = GLib.Error.new_literal(GLib.quark_from_string("gio"), "no handler", 1)
    caplog.set_level("WARNING")
    with patch.object(about.Gio.AppInfo, "launch_default_for_uri", side_effect=missing):
        assert about.open_uri("https://quoth.dev") is False
    [record] = caplog.records
    assert record.levelname == "WARNING"
    assert "https://quoth.dev" in record.getMessage()


@pytest.mark.parametrize("suggested", [True, False])
def test_the_native_package_is_offered_only_where_it_restores_the_overlay(
    display, monkeypatch, suggested
):
    from gi.repository import Adw

    shown = {}
    monkeypatch.setattr(presentation, "suggests_native_overlay", lambda: suggested)
    monkeypatch.setattr(
        Adw.Dialog, "present", lambda self, parent=None: shown.setdefault("d", self)
    )
    about.show_about_dialog(None)

    titles = [
        row.get_title()
        for row in descendants(shown["d"].get_child())
        if isinstance(row, Adw.ActionRow)
    ]
    assert ("Native package" in titles) is suggested


def test_about_dialog_rows_get_the_pointer(display):
    """A dialog's tree is unreachable before present, so content gets the cursors."""
    from gi.repository import Adw, Gtk

    captured = {}
    with patch.object(
        Adw.Dialog, "present", lambda self, parent=None: captured.setdefault("d", self)
    ):
        about.show_about_dialog(None)
    toolbar = captured["d"].get_child()
    rows = [w for w in descendants(toolbar) if isinstance(w, Gtk.ListBoxRow)]
    assert rows and all(
        r.get_cursor() and r.get_cursor().get_name() == "pointer" for r in rows
    )


def _about_over(parent) -> tuple[list, object]:
    """Open About over `parent`; return its widgets and its mapped close button."""
    from gi.repository import Gtk

    parent.present()
    about.show_about_dialog(parent)
    widgets = list(descendants(parent))
    close = next(
        w for w in widgets if isinstance(w, Gtk.Button) and w.has_css_class("close")
    )
    wait_until(close.get_mapped)
    return widgets, close


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("skin_id", ["default", *list_presets()])
def test_what_the_hero_shows_reads_against_it_in_every_skin(
    display, tmp_path, skin_id, dark
):
    from gi.repository import Adw, Gtk

    parent = Adw.Window(default_width=640, default_height=720)
    with appearance(brand.css(skin_from_id(skin_id, tmp_path), dark=dark), dark=dark):
        try:
            widgets, close = _about_over(parent)
            text = [
                w
                for w in widgets
                if isinstance(w, Gtk.Label)
                and w.get_label()
                in ("Local. Private. Yours.", f"Version {about.__version__}")
            ]

            assert len(text) == 2
            for label in text:
                assert legibility(parent, label) >= 4.5, label.get_label()
            assert legibility(parent, close) >= 3
        finally:
            parent.destroy()


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("skin_id", ["default", *list_presets()])
def test_the_hero_fades_into_the_dialog_without_a_seam(
    display, tmp_path, skin_id, dark
):
    from gi.repository import Adw

    parent = Adw.Window(default_width=640, default_height=720)
    with appearance(brand.css(skin_from_id(skin_id, tmp_path), dark=dark), dark=dark):
        try:
            widgets, _ = _about_over(parent)
            hero = next(w for w in widgets if w.has_css_class("quoth-artwork"))
            shown, stride = _next_frame(parent)
            _, frame = parent.compute_bounds(parent)
            _, bounds = hero.compute_bounds(parent)
            x0 = round(bounds.get_x() - frame.get_x())
            foot = round(bounds.get_y() + bounds.get_height() - frame.get_y())

            # The hero's last row against the dialog's sheet just below it, which
            # the fade's last stop may miss by a step of rounding.
            step = max(
                abs(
                    shown[(foot - 1) * stride + x * 3 + c]
                    - shown[foot * stride + x * 3 + c]
                )
                for x in range(x0, x0 + round(bounds.get_width()))
                for c in range(3)
            )
            assert step <= 2
        finally:
            parent.destroy()


def test_the_close_button_reads_once_the_hero_has_scrolled_away(display, tmp_path):
    from gi.repository import Adw, Gtk

    parent = Adw.Window(default_width=640, default_height=420)
    with appearance(
        brand.css(skin_from_id("default", tmp_path), dark=False), dark=False
    ):
        try:
            widgets, close = _about_over(parent)
            scroller = next(w for w in widgets if isinstance(w, Gtk.ScrolledWindow))
            adjustment = scroller.get_vadjustment()
            adjustment.set_value(adjustment.get_upper())

            # Only the glyph is hidden for the ground, so it is measured on its disc.
            glyph = "headerbar windowcontrols > button > image { color: transparent; }"
            assert legibility(parent, close, hide=glyph) >= 3
        finally:
            parent.destroy()
