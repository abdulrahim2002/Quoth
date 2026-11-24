# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Where Quoth keeps its files, under the XDG base directories."""

import os
import re
from pathlib import Path


def _base(variable: str, default: str) -> Path:
    # The spec ignores an unset, empty or relative value; a relative one would put files
    # in whatever directory Quoth was started from.
    value = os.environ.get(variable, "")
    return Path(value) if os.path.isabs(value) else Path.home() / default


def config_dir() -> Path:
    """Return the config directory: theme.toml, style.css and the skin files."""
    return _base("XDG_CONFIG_HOME", ".config") / "quoth"


def autostart_dir() -> Path:
    """Return the directory of the desktop's login-autostart entries."""
    return _base("XDG_CONFIG_HOME", ".config") / "autostart"


def state_dir() -> Path:
    """Return where logs, crash files and runtime records such as calibration live."""
    return _base("XDG_STATE_HOME", ".local/state") / "quoth"


def models_dir() -> Path:
    """Return the directory of downloaded models and their encoders."""
    return _base("XDG_DATA_HOME", ".local/share") / "quoth" / "models"


def openvino_cache_dir() -> Path:
    """Return the cache of OpenVINO's compiled NPU encoders.

    The next compile rebuilds anything deleted from it.
    """
    return _base("XDG_CACHE_HOME", ".cache") / "quoth" / "openvino"


def _home_pattern() -> re.Pattern | None:
    # The home as given and as resolved (for a symlinked home), each only as a whole
    # path prefix: `/home/alice2` and `/var/home/alice` are other directories.
    home = Path.home()
    forms = sorted({str(home), str(home.resolve())} - {"/"}, key=len, reverse=True)
    if not forms:
        return None
    alternatives = "|".join(re.escape(form) for form in forms)
    return re.compile(rf"(?<![\w.-])(?:{alternatives})(?![\w.-])")


_HOME = _home_pattern()


def tilde(text: str) -> str:
    """Show the home directory in `text`, which names the user, as `~`."""
    return _HOME.sub("~", text) if _HOME else text
