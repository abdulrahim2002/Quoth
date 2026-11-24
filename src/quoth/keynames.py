# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Human-readable names and hotkey classification for evdev key tokens.

Shared by Settings and onboarding, so a hotkey reads the same in both. The layout code
point comes from Gdk in `hotkey_capture`, which keeps this module free of GTK.
"""

import unicodedata

# Display names by evdev token, KEY_ prefix stripped. Bare single letters and digits
# are not listed: they pass through unchanged as the `.get` default below.
_DISPLAY_NAMES: dict[str, str] = {
    "SCROLLLOCK": "Scroll Lock",
    "CAPSLOCK": "Caps Lock",
    "NUMLOCK": "Num Lock",
    "ESC": "Escape",
    "COMPOSE": "Menu",
    "SYSRQ": "Print Screen",
    "LEFTCTRL": "Ctrl (Left)",
    "RIGHTCTRL": "Ctrl (Right)",
    "LEFTALT": "Alt (Left)",
    "RIGHTALT": "Alt (Right)",
    "LEFTSHIFT": "Shift (Left)",
    "RIGHTSHIFT": "Shift (Right)",
    "LEFTMETA": "Super (Left)",
    "RIGHTMETA": "Super (Right)",
    "KPPLUS": "Numpad +",
    "KPMINUS": "Numpad -",
    "KPASTERISK": "Numpad *",
    "KPSLASH": "Numpad /",
    "KPENTER": "Numpad Enter",
    "ENTER": "Enter",
    "BACKSPACE": "Backspace",
    "SPACE": "Space",
    "TAB": "Tab",
    "DELETE": "Delete",
    "INSERT": "Insert",
    "HOME": "Home",
    "END": "End",
    "PAGEUP": "Page Up",
    "PAGEDOWN": "Page Down",
    "PAUSE": "Pause",
    "MINUS": "-",
    "EQUAL": "=",
    "GRAVE": "`",
    "COMMA": ",",
    "DOT": ".",
    "SLASH": "/",
    "SEMICOLON": ";",
    "APOSTROPHE": "'",
    "BACKSLASH": "\\",
    "LEFTBRACE": "[",
    "RIGHTBRACE": "]",
}


def format_key(key: str) -> str:
    """Return the display name of an evdev token (`KEY_LEFTCTRL` -> "Ctrl (Left)").

    An unknown token falls through as its name without the `KEY_` prefix.
    """
    display = key.removeprefix("KEY_")
    return _DISPLAY_NAMES.get(display, display)


# Keys pressed during ordinary input: a global hotkey on one would fire on every
# keystroke (typing keys) or every shortcut (bare modifiers). The listener reads keys
# without taking them, so a key that acts in the focused app still does so on every
# press. Everything not listed (F13 to F24, Pause, media keys) is a safe hotkey, so an
# unknown token defaults to safe.
_MODIFIER_KEYS = frozenset(
    {"LEFTCTRL", "RIGHTCTRL", "LEFTALT", "RIGHTALT",
     "LEFTSHIFT", "RIGHTSHIFT", "LEFTMETA", "RIGHTMETA"}
)  # fmt: skip
_TYPING_KEYS = frozenset(
    {"SPACE", "ENTER", "KPENTER", "TAB", "BACKSPACE",
     "MINUS", "EQUAL", "GRAVE", "COMMA", "DOT", "SLASH", "SEMICOLON",
     "APOSTROPHE", "BACKSLASH", "LEFTBRACE", "RIGHTBRACE", "102ND",
     "KPPLUS", "KPMINUS", "KPASTERISK", "KPSLASH", "KPDOT",
     "KP0", "KP1", "KP2", "KP3", "KP4", "KP5", "KP6", "KP7", "KP8", "KP9"}
)  # fmt: skip
_FOCUSED_APP_KEYS = frozenset(
    {"DELETE", "HOME", "END", "PAGEUP", "PAGEDOWN", "UP", "DOWN", "LEFT", "RIGHT",
     "SYSRQ", "ESC", "CAPSLOCK", "NUMLOCK", "COMPOSE",
     *(f"F{n}" for n in range(1, 13))}
)  # fmt: skip


def hotkey_risk(key: str) -> str | None:
    """Classify a hotkey token such as `KEY_A`; None means a safe dedicated key.

    "modifier" is a bare Ctrl, Shift, Alt or Super; "typing" a key hit during normal
    input (letters, digits, punctuation, Space); "focused-app" a key the app being typed
    in also acts on (F1 to F12, Escape, Delete, arrows, Caps Lock, Num Lock, Menu);
    "paste" is Insert, part of the Shift+Insert paste. `hotkey_warning_text` gives the
    layout-aware caution.
    """
    token = key.removeprefix("KEY_")
    if token in _MODIFIER_KEYS:
        return "modifier"
    if token in _TYPING_KEYS or (len(token) == 1 and token.isalnum()):
        return "typing"
    if token == "INSERT":
        return "paste"
    if token in _FOCUSED_APP_KEYS:
        return "focused-app"
    return None


def hotkey_warning_text(label: str, risk: str) -> str:
    """Return the warn-and-allow caution for a risky hotkey.

    It names the key by the user's own keycap `label` (layout-aware, supplied by the
    caller) and explains the `risk` from `hotkey_risk`.
    """
    if risk == "modifier":
        return (
            f"{label} is a modifier key. Dictation would start whenever you use it in a "
            "shortcut, but you can still choose it."
        )
    if risk == "paste":
        return (
            f"{label} is part of Shift+Insert, the paste Quoth sends, so a paste could "
            "start dictation again, but you can still choose it."
        )
    if risk == "focused-app":
        return (
            f"{label} also reaches the app you're typing in, so each press would act "
            "there too, but you can still choose it."
        )
    return (
        f"{label} is a normal typing key. Dictation would start every time you type it, "
        "but you can still choose it."
    )


def printable_label(codepoint: int, token: str) -> str:
    """Return the user-facing label for a key.

    That is the glyph the layout prints (upper-cased for a single letter, to match the
    keycap), else the friendly evdev name ("F13", "Space", "Escape"). `codepoint` comes
    from the active layout, 0 when the key prints nothing (an F-key); `token` is the
    evdev fallback. Dead or combining marks and whitespace fall back to the name, so a
    hotkey never reads as an invisible or floating glyph.
    """
    if codepoint:
        ch = chr(codepoint)
        if ch.isprintable() and not ch.isspace() and not unicodedata.combining(ch):
            upper = ch.upper()
            return upper if len(upper) == 1 else ch
    return format_key(token)
