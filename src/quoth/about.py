# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""The Quoth About dialog, shown from the settings window and the tray menu.

Quoth's own rather than `Adw.AboutDialog`, which offers neither the mark over the
smoke nor links that carry their icons. Its licence and issue link are kept as rows.
"""

from importlib import resources
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk
from loguru import logger

from quoth import APP_ID, SUPPORT_URL, WEBSITE_URL, __version__, brand, presentation
from quoth.backdrop import Backdrop

SOURCE_URL = "https://github.com/abullard1/Quoth"
ISSUES_URL = f"{SOURCE_URL}/issues"
TROUBLESHOOTING_URL = f"{SOURCE_URL}/blob/main/SUPPORT.md#troubleshooting"
SCRIPTING_URL = f"{SOURCE_URL}/blob/main/SUPPORT.md#scripting"
PRIVACY_URL = f"{SOURCE_URL}/blob/main/PRIVACY.md"
LICENSE_URL = "https://www.gnu.org/licenses/gpl-3.0.html"

_WIDTH = 420
_HERO_HEIGHT = 320
_MARK_PX = 96


def _link_row(icon: str, title: str, url: str, subtitle: str = "") -> Adw.ActionRow:
    row = Adw.ActionRow(title=title, subtitle=subtitle, activatable=True)
    row.add_prefix(Gtk.Image.new_from_icon_name(icon))
    row.add_suffix(Gtk.Image.new_from_icon_name("adw-external-link-symbolic"))
    row.connect("activated", lambda _r: open_uri(url))
    return row


def open_uri(url: str) -> bool:
    """Open an outward link in the user's browser; return False when it cannot.

    A desktop with no handler for the link is logged, never raised.
    """
    try:
        Gio.AppInfo.launch_default_for_uri(url, None)
    except GLib.Error as e:
        logger.warning("Could not open {}: {}", url, e.message)
        return False
    return True


def _hero() -> Gtk.Overlay:
    """Build the banner: the mark, the tagline and the version over the smoke.

    The smoke fades into the dialog, in the light scheme only below the text.
    """
    still = Path(str(resources.files("quoth") / "backgrounds" / "onboarding.jpg"))
    hero = Gtk.Overlay(child=Backdrop(still=still))
    hero.add_css_class("quoth-artwork")
    hero.set_size_request(-1, _HERO_HEIGHT)
    fade = Gtk.Box(can_target=False)
    fade.add_css_class("quoth-fade")
    hero.add_overlay(fade)

    mark = Gtk.Image.new_from_icon_name(f"{APP_ID}-symbolic")
    mark.set_pixel_size(_MARK_PX)
    mark.add_css_class("quoth-mark")
    tagline = Gtk.Label(label="Local. Private. Yours.")
    tagline.add_css_class("quoth-tagline")
    tagline.set_margin_top(12)
    version = Gtk.Label(label=f"Version {__version__}")
    version.add_css_class("quoth-dim")
    version.add_css_class("caption")
    column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
    column.set_halign(Gtk.Align.CENTER)
    column.set_valign(Gtk.Align.CENTER)
    column.set_margin_top(24)  # under the header bar's close button
    column.set_margin_bottom(40)  # above the fade into a light dialog
    for child in (mark, tagline, version):
        column.append(child)
    hero.add_overlay(column)
    return hero


def show_about_dialog(parent) -> None:
    """Present the About dialog over `parent`."""
    dialog = Adw.Dialog(title="About Quoth", content_width=_WIDTH)
    header = Adw.HeaderBar()
    header.add_css_class("flat")
    header.add_css_class("quoth-over-artwork")
    header.set_title_widget(Gtk.Label())  # the mark is the title here
    toolbar = Adw.ToolbarView()
    toolbar.add_top_bar(header)
    toolbar.set_extend_content_to_top_edge(True)

    links = Adw.PreferencesGroup()
    links.add(_link_row(f"{APP_ID}-symbolic", "Website", WEBSITE_URL, "quoth.dev"))
    links.add(
        _link_row(f"{APP_ID}-github-symbolic", "Source code", SOURCE_URL, "GitHub")
    )
    links.add(_link_row(f"{APP_ID}-coffee-symbolic", "Buy Me a Coffee", SUPPORT_URL))
    # Where the sandbox costs the Flatpak its on-screen overlay, point users to the
    # native package that restores it.
    if presentation.suggests_native_overlay():
        links.add(
            _link_row(
                "package-x-generic-symbolic",
                "Native package",
                presentation.NATIVE_INSTALL_URL,
                "Adds the on-screen overlay",
            )
        )
    project = Adw.PreferencesGroup()
    project.add(_link_row(f"{APP_ID}-bug-symbolic", "Report an issue", ISSUES_URL))
    project.add(
        _link_row(
            "text-x-generic-symbolic",
            "License",
            LICENSE_URL,
            "GPL-3.0-or-later. © 2026 Samuel Ruairí Bullard",
        )
    )

    column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    column.set_margin_top(18)
    column.set_margin_start(18)
    column.set_margin_end(18)
    column.set_margin_bottom(18)
    column.append(links)
    column.append(project)

    page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    page.append(_hero())
    page.append(column)
    scroller = Gtk.ScrolledWindow(vexpand=True, propagate_natural_height=True)
    scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    scroller.set_child(page)
    toolbar.set_content(scroller)
    dialog.set_child(toolbar)
    # On the content now (a dialog parents its child only on present), and again
    # once mapped, when the header bar has built its close button.
    brand.pointer_cursors(toolbar)
    dialog.connect("map", brand.pointer_cursors)
    dialog.present(parent)
