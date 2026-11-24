# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Settings > Feedback > Overlay: the theme picker, live preview and orb switch."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import gi
from loguru import logger

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gdk, Gio, GLib, Gtk

from quoth import brand, paths
from quoth.atomic_write import diagnostic
from quoth.config import Config
from quoth.pill_view import (
    OrbView,
    OverlayState,
    PillView,
    animations_on,
    fit_height,
    make_orb,
    orb_seed,
)
from quoth.scheduler import guarded
from quoth.theme import Theme

# Stands for "theme.toml matches no known theme". It is a state the picker reports,
# never a theme id, so it cannot collide with one.
_CUSTOM_THEME = "custom"

# Row orb diameter: large enough for the orb's interior flow to show, since at icon
# size its regions blur into one tint.
_ORB_SIZE = 40
# The live preview orb, when the orb overlay style is selected: big enough to
# read as the overlay it previews.
_PREVIEW_ORB_SIZE = 96

# The feature-maturity tiers that may badge a Settings row (styling in the
# section's CSS). Stable features carry no chip.
_TIERS = ("preview", "experimental")

# Installed once and kept for the process. A display provider is never removed, and the
# Settings window is rebuilt on every open, so a per-instance provider would pile up.
_TIER_CSS: Gtk.CssProvider | None = None

# The width the preview has to fit inside this window; only an unusually wide theme
# ever reaches it, and only then is the preview drawn at anything but 1:1.
_PREVIEW_MAX_W = 420
_PREVIEW_CYCLE_MS = 4000
_FALLBACK_LABEL_MAX = 60


def _display_label(value: str) -> str:
    cleaned = "".join(ch if ch.isprintable() else " " for ch in value).strip()
    return cleaned[:_FALLBACK_LABEL_MAX].rstrip() or "Unnamed theme"


def _tier_chip(tier: str) -> Gtk.Label:
    """Return a pill-shaped maturity badge for a pre-stable feature's row."""
    if tier not in _TIERS:
        raise ValueError(f"Unknown feature tier: {tier!r}")
    chip = Gtk.Label(label=tier.capitalize(), valign=Gtk.Align.CENTER)
    chip.add_css_class("quoth-tier")
    chip.add_css_class(tier)
    return chip


def _preview_height(theme: Theme) -> int:
    """Return the preview pill's height: the theme's own, so its geometry reads true.

    Only a pill wider than `_PREVIEW_MAX_W` is scaled down to fit.
    """
    return fit_height(theme, _PREVIEW_MAX_W)


def _theme_label(name: str) -> str:
    """Return the fallback picker label for a theme id, such as 'Catppuccin Mocha'.

    A [meta] name, when the file carries one, wins over this in `_theme_texts`.
    """
    if name == _CUSTOM_THEME:
        return "Custom (theme.toml)"
    return _display_label(name.removeprefix("user:").replace("-", " ").title())


def _shelf_divider_row() -> Gtk.ListBoxRow:
    """Build the caption row separating shipped presets from the user's shelf.

    A plain row, not an ActionRow, so it reads as a heading rather than a peer row.
    """
    label = Gtk.Label(label="Your themes", xalign=0)
    label.add_css_class("caption-heading")
    label.add_css_class("dim-label")
    label.set_margin_top(14)
    label.set_margin_bottom(4)
    label.set_margin_start(12)
    row = Gtk.ListBoxRow()
    row.set_child(label)
    row.set_activatable(False)
    row.set_selectable(False)
    row.set_focusable(False)
    return row


def _theme_texts(theme_id: str, theme: Theme) -> tuple[str, str | None]:
    """Return a theme row's title and subtitle.

    The title is the theme's [meta] name over the filename-derived label, and the
    subtitle credits the author. The loader caps and sanitises these strings and the
    rows render with markup off, so a shared theme file cannot spoof or restyle the
    picker.
    """
    title = theme.meta.name or _theme_label(theme_id)
    subtitle = f"by {theme.meta.author}" if theme.meta.author else None
    return title, subtitle


def _theme_action(picked: str, on_disk: str | None) -> str:
    """Return what clicking the `picked` row does: "apply", "confirm" or "ignore".

    `on_disk` is the theme id theme.toml holds, None when the file is the user's own.
    Only that case prompts, since the copy replaces the whole file.
    """
    if picked == on_disk:
        return "ignore"  # already applied
    if on_disk is None:
        return "confirm"
    return "apply"


@dataclass
class ThemeBridge:
    """App-side theming affordances the settings view consumes.

    The app owns theme.toml and the monitor watching it, so the view never touches the
    file, and `apply` also restyles the live overlay. `presets` is a snapshot, since
    package data cannot change at runtime.
    """

    presets: list[str]  # bundled preset names, in picker order
    # The 'user:<stem>' ids of the user themes directory, re-read on every render.
    user_themes: Callable[[], list[str]]
    # The applied theme id: "default" when there is no file, None for the user's own.
    current: Callable[[], str | None]
    theme_of: Callable[[str], Theme]  # any picker id parsed, for drawing it
    live: Callable[[], Theme]  # the theme in force now, a hand-edited one included
    load_error: Callable[[], str | None]  # last active-file load failure, if any
    apply: Callable[[str], None]  # copy a preset or user theme over theme.toml
    import_file: Callable[[Path], str]  # copy one local TOML file to the user shelf


class OverlaySection:
    """The Overlay section of the Settings Feedback page.

    It holds the theme picker, the live preview that doubles as the input meter, and
    the orb overlay switch. A coordinator, not a widget: it adds two PreferencesGroups
    to the page and owns their state.
    """

    def __init__(
        self,
        config: Config,
        theme_bridge: ThemeBridge,
        *,
        toast: Callable[[str], None],
        motion_enabled: bool = True,
    ):
        self._config = config
        self._theme_bridge = theme_bridge
        self._toast = toast
        self._motion_enabled = motion_enabled  # mirrors the system preference
        self._window: Adw.PreferencesWindow | None = None
        self._theme_expander: Adw.ExpanderRow | None = None
        self._import_button: Gtk.Button | None = None
        self._import_cancellable: Gio.Cancellable | None = None
        self._theme_rows: dict[str, Adw.ActionRow] = {}
        self._theme_orbs: list[OrbView] = []  # one per row, for reduced-motion sync
        # A rebuilt row continues its own orb only while its palette is unchanged.
        self._orb_cache: dict[str, tuple[int, object]] = {}
        self._shelf_divider: Gtk.ListBoxRow | None = None
        self._orb_switch: Gtk.Switch | None = None
        self._preview_group: Adw.PreferencesGroup | None = None
        self._theme_preview: PillView | OrbView | None = None
        self._meter: Gtk.LevelBar | None = None
        self._preview_state = OverlayState.RECORDING
        self._cycle_id: int | None = None

    def build(self, page: Adw.PreferencesPage, window: Adw.PreferencesWindow) -> None:
        """Add the preview group and the theme picker group to `page`.

        Clicking a row applies that theme through the bridge, which also restyles the
        overlay. `window` parents the confirm dialogs.
        """
        self._window = window
        self._apply_css()
        # The preview gets its own group above the selector: a PreferencesGroup packs
        # non-row widgets below its rows, so in the selector's group the expander would
        # push it down on open.
        preview_group = Adw.PreferencesGroup(title="Overlay")
        page.add(preview_group)

        # The overlay is only on screen mid-dictation, so this preview is the way to see
        # a theme without recording. Drawn at the theme's own size and fed the real
        # microphone, it is also the input meter for the Dictation page's microphone.
        self._preview_group = preview_group
        self._update_group_description()
        # A slot tall enough for either renderer, so switching pill and orb (or
        # applying a theme with taller geometry) never moves the rows below.
        self._preview_slot = Adw.Bin(valign=Gtk.Align.CENTER)
        self._theme_preview = self._make_theme_preview()
        self._preview_slot.set_child(self._theme_preview)
        self._size_preview_slot()
        preview_group.add(self._preview_slot)
        self._meter = Gtk.LevelBar(min_value=0.0, max_value=1.0, value=0.0)
        self._meter.set_size_request(220, -1)
        self._meter.set_halign(Gtk.Align.CENTER)
        self._meter.set_margin_bottom(6)
        self._meter.update_property(
            [Gtk.AccessibleProperty.LABEL], ["Microphone level"]
        )
        preview_group.add(self._meter)

        selector_group = Adw.PreferencesGroup()
        page.add(selector_group)
        self._theme_expander = Adw.ExpanderRow(
            title="Overlay theme",
            tooltip_text=(
                "A theme sets the overlay's colors and can set its font, shape and motion"
            ),
        )
        # Named by its title: an expander row has no name of its own, and GTK
        # would otherwise take its tooltip.
        self._theme_expander.update_property(
            [Gtk.AccessibleProperty.LABEL], [self._theme_expander.get_title()]
        )
        # The subtitle can carry a custom theme's own [meta] name; arbitrary file
        # content renders as text here, never as Pango styling.
        self._theme_expander.set_use_markup(False)
        self._import_button = Gtk.Button(label="Import…", valign=Gtk.Align.CENTER)
        self._import_button.set_tooltip_text(
            "Adds a .toml theme to this list without applying it"
        )
        # The label child names the button through a relation that outranks the
        # label property, so a screen reader would hear only "Import…".
        self._import_button.reset_relation(Gtk.AccessibleRelation.LABELLED_BY)
        self._import_button.update_property(
            [Gtk.AccessibleProperty.LABEL], ["Import theme"]
        )
        self._import_button.connect("clicked", self._on_import_clicked)
        self._theme_expander.add_suffix(self._import_button)
        selector_group.add(self._theme_expander)
        self._render_theme_rows()
        self._build_orb_overlay_row(selector_group)
        self._sync_preview_cycle(self._preview_motion_enabled())

    def _apply_css(self) -> None:
        """Install the maturity-chip style once for the process.

        The chip is a pill-shaped tag beside a pre-stable feature's control, drawn in
        named Adwaita colours so both colour schemes work.
        """
        global _TIER_CSS
        if _TIER_CSS is not None:
            return
        _TIER_CSS = Gtk.CssProvider()
        _TIER_CSS.load_from_string(
            "label.quoth-tier { border-radius: 999px; padding: 1px 8px;"
            "  font-size: 0.75em; font-weight: bold; }"
            "label.quoth-tier.preview {"
            "  background-color: alpha(@accent_bg_color, 0.18); color: @accent_color; }"
            "label.quoth-tier.experimental {"
            "  background-color: alpha(@warning_bg_color, 0.2); color: @warning_color; }"
        )
        Gtk.StyleContext.add_provider_for_display(
            self._window.get_display() or Gdk.Display.get_default(),
            _TIER_CSS,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _build_orb_overlay_row(self, group: Adw.PreferencesGroup):
        """Add the Experimental orb switch below the theme picker and its orbs.

        A switch, not a combo row: there are exactly two styles, and a ComboRow's
        selection signal also fires for programmatic updates.
        """
        row = Adw.ActionRow(tooltip_text="A fluid orb that reacts to your voice")
        # ActionRow has no title-line slot for the tier chip, so the row's title stays
        # empty and this block is the expanding prefix.
        block = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            valign=Gtk.Align.CENTER,
            hexpand=True,
        )
        line = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        title = Gtk.Label(label="Orb overlay", xalign=0)
        title.add_css_class("title")
        chip = _tier_chip("experimental")
        line.append(title)
        line.append(chip)
        subtitle = Gtk.Label(
            label="Replaces the pill with your theme's orb",
            xalign=0,
            wrap=True,
        )
        subtitle.add_css_class("subtitle")
        block.append(line)
        block.append(subtitle)
        row.add_prefix(block)
        switch = Gtk.Switch(valign=Gtk.Align.CENTER)
        # Set before connecting, so building the row emits nothing.
        switch.set_active(self._config.overlay_style == "orb")
        switch.connect("notify::active", self._on_orb_overlay_changed)
        row.add_suffix(switch)
        row.set_activatable_widget(switch)
        # The row and its switch are named by the row's empty title labels; point
        # them at the block's labels instead.
        relations = [
            Gtk.AccessibleRelation.LABELLED_BY,
            Gtk.AccessibleRelation.DESCRIBED_BY,
        ]
        for widget in (row, switch):
            widget.update_relation(
                relations,
                [
                    Gtk.AccessibleList.new_from_list([title, chip]),
                    Gtk.AccessibleList.new_from_list([subtitle]),
                ],
            )
        group.add(row)
        self._orb_switch = switch

    def _on_orb_overlay_changed(self, switch, _pspec):
        self._config.overlay_style = "orb" if switch.get_active() else "pill"

    def sync(self, name: str) -> None:
        """Show an overlay style change, whoever wrote it, on the switch and preview."""
        if name == "overlay_style" and self._orb_switch is not None:
            self._orb_switch.set_active(self._config.overlay_style == "orb")
            self._swap_theme_preview()

    def push_level(self, amplitude: float) -> None:
        """Feed the preview and the meter a real microphone level.

        They meter the microphone chosen in the Dictation page's Audio group, so silence
        stays flat: a wave here means that microphone is being heard.
        """
        if self._theme_preview is not None:
            self._theme_preview.push_level(amplitude)
        if self._meter is not None:
            self._meter.set_value(amplitude)

    def close(self) -> None:
        """Release the window: stop the preview cycle and cancel a pending import."""
        # The cycle timer holds a reference to the window; drop it on close or it
        # outlives the window and keeps firing.
        self._sync_preview_cycle(False)
        if self._import_cancellable is not None:
            self._import_cancellable.cancel()
            self._import_cancellable = None
        self._window = None

    def _update_group_description(self) -> None:
        if self._preview_group is not None:
            self._preview_group.set_description(self._theme_bridge.load_error() or "")

    def _on_import_clicked(self, button: Gtk.Button) -> None:
        if self._import_cancellable is not None or self._window is None:
            return
        button.set_sensitive(False)
        window = self._window
        cancellable = Gio.Cancellable()
        self._import_cancellable = cancellable
        theme_filter = Gtk.FileFilter()
        theme_filter.set_name("Overlay themes (*.toml)")
        theme_filter.add_pattern("*.toml")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(theme_filter)
        dialog = Gtk.FileDialog(title="Import theme", filters=filters)
        dialog.set_default_filter(theme_filter)
        dialog.open(
            window,
            cancellable,
            lambda chooser, result: self._on_import_file_chosen(
                chooser, result, window, cancellable
            ),
        )

    def _on_import_file_chosen(
        self,
        dialog: Gtk.FileDialog,
        result,
        window: Adw.PreferencesWindow,
        cancellable: Gio.Cancellable,
    ) -> None:
        error: GLib.Error | None = None
        try:
            file = dialog.open_finish(result)
        except GLib.Error as exc:
            file = None
            error = exc

        if self._window is not window or self._import_cancellable is not cancellable:
            return
        self._import_cancellable = None
        self._import_button.set_sensitive(True)

        if error is not None:
            if any(
                error.matches(domain, code)
                for domain, code in (
                    (Gtk.DialogError.quark(), Gtk.DialogError.CANCELLED),
                    (Gtk.DialogError.quark(), Gtk.DialogError.DISMISSED),
                    (Gio.io_error_quark(), Gio.IOErrorEnum.CANCELLED),
                )
            ):
                return
            logger.error("Could not choose theme file: {}", error)
            self._toast("Couldn't open the theme file")
            return

        path = file.get_path()
        if path is None:
            # The scheme alone: a remote URI names the account and the host.
            logger.warning(
                "Rejected theme import: a {} location has no local path",
                file.get_uri_scheme(),
            )
            self._toast("Choose a local theme file")
            return
        try:
            theme_id = self._theme_bridge.import_file(Path(path))
        except ValueError as exc:
            logger.warning("Rejected theme import {}: {}", path, exc)
            self._toast(str(exc))
            return
        except OSError as exc:
            logger.error("Could not import theme file {}: {}", path, exc)
            self._render_theme_rows()
            self._theme_expander.set_expanded(True)
            self._toast("Couldn't confirm the import. Check the theme list.")
            return

        self._render_theme_rows()
        if row := self._theme_rows.get(theme_id):

            def focus_imported_row() -> bool:
                if (
                    self._window is window
                    and self._theme_rows.get(theme_id) is row
                    and row.get_mapped()
                ):
                    row.grab_focus()
                return False

            map_handler: int | None = None

            def focus_after_map(_row=None) -> None:
                nonlocal map_handler
                if map_handler is not None:
                    row.disconnect(map_handler)
                    map_handler = None
                GLib.idle_add(focus_imported_row)

            if row.get_mapped():
                focus_after_map()
            else:
                map_handler = row.connect("map", focus_after_map)
        self._theme_expander.set_expanded(True)
        self._toast("Theme imported. Select it to apply.")

    def _render_theme_rows(self) -> None:
        """Rebuild every theme row and the expander subtitle from the file on disk.

        It may run from a row's own `activated` handler, since rebuilding emits no
        `activated`. `on_disk` is read once, so every row and the subtitle agree.
        """
        on_disk = self._theme_bridge.current()
        self._update_group_description()
        self._theme_orbs.clear()
        for row in self._theme_rows.values():
            self._theme_expander.remove(row)
        self._theme_rows.clear()
        if self._shelf_divider is not None:
            self._theme_expander.remove(self._shelf_divider)
            self._shelf_divider = None
        # Bundled presets first, then the user's themes beneath a divider that exists
        # only while a shelf theme has a row. The shelf is re-listed per render, so new
        # files appear on the next theme event or reopen without a directory monitor.
        user_themes = self._theme_bridge.user_themes()
        for theme_id in self._theme_bridge.presets:
            row = self._build_theme_row(theme_id, on_disk)
            if row is None:
                continue
            self._theme_expander.add_row(row)
            self._theme_rows[theme_id] = row
        for theme_id in user_themes:
            row = self._build_theme_row(theme_id, on_disk)
            if row is None:
                continue
            if self._shelf_divider is None:
                self._shelf_divider = _shelf_divider_row()
                self._theme_expander.add_row(self._shelf_divider)
            self._theme_expander.add_row(row)
            self._theme_rows[theme_id] = row
        self._orb_cache = {
            theme_id: cached
            for theme_id, cached in self._orb_cache.items()
            if theme_id in self._theme_rows
        }
        if on_disk not in self._theme_rows:
            on_disk = None
        # A shelf file can change after identification. Only surviving rows can
        # identify the active copy; otherwise show its own metadata as Custom.
        if on_disk is None and (live_name := self._theme_bridge.live().meta.name):
            self._theme_expander.set_subtitle(f"{live_name} (theme.toml)")
        else:
            self._theme_expander.set_subtitle(_theme_label(on_disk or _CUSTOM_THEME))
        brand.pointer_cursors(self._theme_expander)

    def _build_theme_row(self, name: str, on_disk: str | None) -> Adw.ActionRow | None:
        """Build one theme row, or return None when the theme cannot be read.

        The active row gets the marker and is not activatable, so a stray click cannot
        rewrite theme.toml with what it already holds.
        """
        try:
            theme = self._theme_bridge.theme_of(name)
        except (ValueError, OSError) as exc:
            logger.warning(
                "Could not preview theme {} ({}: {})",
                diagnostic(name),
                type(exc).__name__,
                diagnostic(paths.tilde(str(exc))),
            )
            return None
        title, subtitle = _theme_texts(name, theme)
        row = Adw.ActionRow(title=title)
        # Theme names and credits are arbitrary file content; markup stays off so
        # they render as text, never as Pango styling.
        row.set_use_markup(False)
        if subtitle:
            row.set_subtitle(subtitle)
        if theme.meta.description:
            row.set_tooltip_text(theme.meta.description)
        selected = name == on_disk
        # The palette as a drifting orb, drawn from the theme's own colours, so the big
        # preview can stay on what is applied.
        seed = orb_seed(theme)
        cached = self._orb_cache.get(name)
        if cached is None or cached[0] != seed:
            cached = (seed, make_orb(theme))
            self._orb_cache[name] = cached
        orb = OrbView(theme, size=_ORB_SIZE, orb_instance=cached[1])
        orb.set_motion_enabled(self._motion_enabled)
        orb.set_valign(Gtk.Align.CENTER)
        self._theme_orbs.append(orb)
        row.add_prefix(orb)
        if selected:
            check = Gtk.Image(
                icon_name="object-select-symbolic",
                accessible_role=Gtk.AccessibleRole.PRESENTATION,
            )
            check.add_css_class("dim-label")
            row.add_suffix(check)
            # A list box overwrites its rows' selected state, so the name carries it.
            row.reset_relation(Gtk.AccessibleRelation.LABELLED_BY)
            row.update_property([Gtk.AccessibleProperty.LABEL], [f"{title} Active"])
            return row
        row.set_activatable(True)
        row.connect("activated", lambda _r, n=name: self._on_theme_selected(n))
        return row

    def _make_theme_preview(self):
        """Build the live preview of the selected overlay renderer: the pill or the orb.

        Either way it is fed the real microphone, so it stays the input meter.
        """
        theme = self._theme_bridge.live()
        if self._config.overlay_style == "orb":
            view = OrbView(theme, size=_PREVIEW_ORB_SIZE, live=True)
        else:
            view = PillView(
                theme,
                height=_preview_height(theme),
                animated=True,
                scripted=False,
                timer=True,
            )
        view.set_halign(Gtk.Align.CENTER)
        view.set_valign(Gtk.Align.CENTER)  # centred in the fixed-height slot
        view.set_margin_top(6)
        view.set_margin_bottom(6)
        view.set_motion_enabled(self._preview_motion_enabled())
        return view

    def _swap_theme_preview(self) -> None:
        """Rebuild the preview for the newly selected renderer, in place."""
        if self._theme_preview is None:
            return
        self._theme_preview = self._make_theme_preview()
        self._preview_slot.set_child(self._theme_preview)
        self._sync_preview_cycle(self._preview_motion_enabled())

    def _size_preview_slot(self) -> None:
        """Size the slot to fit either renderer of the live theme, margins included."""
        pill = _preview_height(self._theme_bridge.live())
        self._preview_slot.set_size_request(-1, max(pill, _PREVIEW_ORB_SIZE) + 12)

    def _sync_preview_to_live(self) -> None:
        """Point the preview at the applied theme.

        The row orbs preview the themes on offer; this one shows what is in force now.
        The pill is resized as well as restyled, since at 1:1 a theme's geometry is
        part of what it previews.
        """
        if self._theme_preview is None:
            return
        theme = self._theme_bridge.live()
        if isinstance(self._theme_preview, OrbView):
            self._theme_preview.set_theme(theme)
        else:
            self._theme_preview.set_theme(theme, height=_preview_height(theme))
        self._theme_preview.set_motion_enabled(self._preview_motion_enabled())
        self._size_preview_slot()
        self._sync_preview_cycle(self._preview_motion_enabled())

    def _preview_motion_enabled(self) -> bool:
        return animations_on(self._theme_bridge.live(), self._motion_enabled)

    def _sync_preview_cycle(self, run: bool) -> None:
        if run and self._cycle_id is None:
            self._cycle_id = GLib.timeout_add(
                _PREVIEW_CYCLE_MS,
                guarded(self._cycle_preview, on_error=self._forget_cycle),
            )
        elif not run:
            if self._cycle_id is not None:
                GLib.source_remove(self._cycle_id)
                self._cycle_id = None
            self._preview_state = OverlayState.RECORDING
            if self._theme_preview is not None:
                self._theme_preview.set_state(OverlayState.RECORDING)

    def _forget_cycle(self) -> None:
        self._cycle_id = None

    def _cycle_preview(self) -> bool:
        """Alternate the preview between its recording and transcribing states.

        Recording alone would never show `dot_transcribing`, which most themes colour
        differently.
        """
        self._preview_state = (
            OverlayState.TRANSCRIBING
            if self._preview_state == OverlayState.RECORDING
            else OverlayState.RECORDING
        )
        self._theme_preview.set_state(self._preview_state)
        return True

    def refresh_theme(self) -> None:
        """Re-render after theme.toml changes, from an apply or a hand edit."""
        if self._theme_expander is None:
            return
        self._render_theme_rows()  # rebuilt orbs inherit motion via _build_theme_row
        self._sync_preview_to_live()

    def set_motion_enabled(self, enabled: bool) -> None:
        """Follow the system motion preference, as the live overlay does.

        It reaches the preview and every row orb; orbs built later read the stored
        state.
        """
        self._motion_enabled = enabled
        if self._theme_preview is None:
            return
        preview_motion = self._preview_motion_enabled()
        self._theme_preview.set_motion_enabled(preview_motion)
        self._sync_preview_cycle(preview_motion)
        for orb in self._theme_orbs:
            orb.set_motion_enabled(enabled)

    def _on_theme_selected(self, name: str) -> None:
        action = _theme_action(name, self._theme_bridge.current())
        if action == "confirm":
            self._confirm_theme_overwrite(name)
        elif action == "apply":
            self._apply_theme(name)

    def _confirm_theme_overwrite(self, name: str) -> None:
        """Ask before copying `name` over a theme.toml that matches no known theme.

        Palette-only presets carry no geometry or animation, so those revert to stock.
        """
        window = self._window
        if window is None:
            return
        dialog = Adw.AlertDialog(
            heading="Replace your theme?",
            body=(
                f"Your theme.toml has changes that aren't in any bundled theme. "
                f"Applying {_theme_label(name)} replaces the whole file, including "
                f"any geometry, position and animation you set."
            ),
        )
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("replace", "Replace")
        dialog.set_response_appearance("replace", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_default_response("cancel")
        dialog.set_close_response("cancel")
        dialog.connect("response", self._on_theme_overwrite_response, name, window)
        dialog.connect("map", brand.pointer_cursors)
        dialog.present(window)

    def _on_theme_overwrite_response(
        self,
        _dialog,
        response: str,
        name: str,
        window: Adw.PreferencesWindow,
    ) -> None:
        if response == "replace" and self._window is window:
            self._apply_theme(name)

    def _apply_theme(self, name: str) -> None:
        try:
            self._theme_bridge.apply(name)
        except ValueError as exc:
            logger.warning("Could not apply theme {}: {}", diagnostic(name), exc)
            self._toast(str(exc))
        except OSError as exc:
            logger.error("Could not apply theme {}: {}", diagnostic(name), exc)
            self._toast("Couldn't confirm the change. Check the selected theme.")
        self._render_theme_rows()  # to the truth on disk, applied or not
