# Design: Quoth app

Quoth uses bone-coloured text on ink surfaces, matching its website. This file
records the default GTK tokens and component rules implemented in `brand.py`
and the overlay themes. Keep the Default surfaces consistent when changing a
token. App skins replace named GTK palette roles only, never images, fonts,
widget geometry or motion; see the [customisation guide](SUPPORT.md#app-skins).

## Tokens in use

| Token | Value | Where in the app |
|---|---|---|
| ink-950 | `#08080a` | Default window and header-bar background (`skin.py`, rendered by `brand.py`) |
| ink-900 | `#0e0e11` | Pill background (`theme.py` Palette, `default.toml`), Default alert dialogs and views (`skin.py`), app icon tile, lit to `#2a2a31` at its top edge (`scripts/generate_icons.py`) |
| surface | bone 5% wash, 8% hairline | Cards and rows, as the site's `.surface` (`brand.py`); popovers flatten it to `#181819` |
| bone | `#f2f0ea` | Pill text, border, waveform, and recording and transcribing dots; GTK accent (`brand.py`) |
| bone-dim | `#cfccc3` | Secondary lines over the photograph in onboarding's Welcome and About, in place of `dim-label` (`brand.py`) |
| red | `#9a1f2e` fill, `#dc5a4a` text | Destructive buttons and error text (`brand.py`) |
| error dot | `#cc2936` | Pill error dot (`theme.py` Palette, `default.toml`) |
| recording red | `#e5484d` | Tray icon while recording (`scripts/generate_icons.py`) |
| pill radius | fully round | Action buttons (Adwaita's `.pill`); maturity chips, 999px (`overlay_section.py`); the overlay pill, half its height (`theme.py`) |

Software centres such as Flathub draw the store banner in vellum (`#dcc49c`) for
light and sepia (`#614730`) for dark, set as the metainfo's `<branding>`. The app
never uses them.

## GTK mapping (`src/quoth/brand.py`)

- `@define-color accent_*` is redefined per colour scheme: bone on ink when
  dark, ink on bone when light. Loaded at USER priority so a desktop's own
  accent in `~/.config/gtk-4.0/gtk.css` (COSMIC writes one) does not win;
  the user's `~/.config/quoth/style.css` has its own provider at USER+1 and
  takes precedence.
- In the dark scheme the named surface colours (`window_bg_color`,
  `dialog_bg_color`, `view_bg_color`, `card_bg_color`, `popover_bg_color`) are
  the ink scale, so structure reads as lift: page black, alert dialogs one step up,
  cards a bone wash with a hairline (the site's `.surface`), popovers the same
  flattened to opaque. The site is dark-only; the light scheme keeps Adwaita's
  surfaces rather than dark text on ink.
- The Default app skin supplies the dark palette and leaves light surfaces
  native. Quoth derives standalone accent ink by constraining OKLab lightness
  for the current scheme, so accent labels and native widgets agree. High
  contrast retains native palettes. The overlay's `theme.toml` palette remains
  separate.
- Skin choices use static 40×40 logical-pixel ribbons: window, view, card and
  accent bands are 14/10/10/6 pixels high. Resolved colours render solid;
  native or derived roles use neutral patterned bands.
- `button.quoth-cta` is the site's primary button: 48px pill, bold label,
  32px side padding. One filled (`suggested-action`) per page at most; a
  second takes `quoth-ghost`, the site's hairline pill.
- `.quoth-keycap` is the Shortcut page's key: a bone-wash face on a 6px side
  that sinks 4px when pressed, down at once and back up slowly.
- The About hero is artwork: its text and the header bar over it stay bone in
  both schemes. Its smoke fades into the dialog, across the hero when dark and
  only below the text when light, so that text keeps its dark ground. The close
  button carries an ink disc, so it reads once the hero has scrolled away.
- The Welcome page's smoke darkens towards its foot, as the site's hero does,
  under a veil over the rest, so its copy reads over the video's brightest
  frames at any window size.

## Type

GTK uses the system UI font (Cantarell or the desktop's choice), not Geist:
the app should look native on the desktop it runs on. Hierarchy comes from
Adwaita's `title-1`, `heading`, `dim-label` and `caption` classes, except over
the photograph, where `dim-label`'s opacity would let the smoke through:
secondary lines there take bone-dim.

## Interaction and help

`brand.pointer_cursors` owns the hand cursor policy for a widget tree; reapply it
after rebuilding a shelf. Disabled actions and inert information rows keep the
default cursor, even inside a hand-cursor parent. Expander headers share their
row's cursor so title and arrow stay one action. The custom-drawn overlay owns
its click-to-cancel cursor; idle overlays are click-through.

In a Settings group every row has a tooltip or none has; the rows of a list,
such as the models, themes or skins, count as the list's row. A tooltip on every
row is this app's choice, where GNOME asks for few; GNOME's rule that a
container's controls all have one or none keeps it consistent. A tooltip adds
one fact about its own control that the title and subtitle do not show: what the
feature is for, when it applies, exactly what changes, or a limit. It never
repeats them, points at another control, states the opposite of the title or
reassures about other behaviour, and it carries no status or warning. What a
user needs in order to decide goes in the title, the subtitle, the group
description or the privacy notice, since a tooltip shows on hover alone and a
screen reader may not read it. Controls keep their accessible names; an expander
row is named by its title, or its tooltip would name it. A tooltip is one short
present-tense statement or phrase of at most 75 characters, without a full stop.
Put it on the owning row so it shows over the row's switch, or a button without
its own; a nested row needs its own, or it shows its expander's. An icon-only
control's tooltip is a short name for its action. A control may say why it is
unavailable, while it is. App-provided help uses sentence case, a project choice
since GNOME capitalises label-like tooltips; native header help and
author-supplied skin and theme descriptions stay as written, in plain text.
Settings and onboarding share the error-report rows' help.
Interface text, including the metainfo and settings schema, is in American
English, as GNOME source strings are; documents stay British. A toast is a
label: one sentence takes no full stop, and two sentences each take one. A
notification's body is prose, so its sentences end with full stops; a status
phrase such as "Recording…" does not, and a body that ends in a command leaves
the command bare. Text shown both as a toast and as a notification is worded as
a toast, and text also shown as a group description keeps its stop.

## Rules

- Default uses bone on ink for reading text, with no second accent colour.
- Keyboard focus uses GTK's own focus ring.
- Depth is tonal (Adwaita cards and hairlines), no floating shadows. Settings
  owns the custom row shading; high contrast restores native separators and
  nested shading.
- One authored motion moment per page (the welcome mark; lines arrive after
  it), never an identical entrance on every page.
