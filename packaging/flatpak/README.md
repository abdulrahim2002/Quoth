# Quoth: Flatpak

The **CPU + Vulkan + Intel NPU** Flatpak runs on the GNOME 50 runtime from
Flathub, so it installs on distributions whose system libadwaita is too old for
the `.deb`. Desktop integration still depends on the compositor: the overlay is
suppressed on GNOME, and on COSMIC, Sway, Hyprland and niri, which deny it to
sandboxed apps. The Intel NPU path is encoder-only and needs a compatible host
kernel, firmware and hardware; the bundle carries the OpenVINO runtime, Level
Zero, the Intel NPU user-mode driver and its compiler, TBB and hwloc. The build
reuses the shared `Makefile` engine; see the manifest header.

## Shortcut

Quoth uses the Wayland GlobalShortcuts portal when the desktop provides it. The
Flatpak bundles libportal and its Python GIR bindings; native packages keep evdev
and do not require libportal. The user sets the shortcut up once and Quoth
registers it again on each launch; if it is denied or removed, they retry in
Settings. The desktop owns the binding; Quoth offers tap and hold modes.

The bundle carries PyGObject 3.56.3 with a patch for a double free in
full-transfer pointer arrays; with the runtime's own PyGObject, Quoth crashes
when libportal returns assigned shortcuts.

Without the portal, the user binds the toggle command in the desktop's keyboard
settings; see [Shortcuts and commands](../../SUPPORT.md#shortcuts-and-commands).

## Build & validate locally

Needs `flatpak`, `org.flatpak.Builder` from Flathub, `python3-yaml`, `git` and
`xvfb-run`. The harness builds a `git archive` of HEAD, so commit a change
before validating it.

```sh
# Build in a clean sandbox, install, smoke-test and export the .flatpak:
packaging/flatpak/.flatpakvalidate/run.sh
# Lint the manifest:
packaging/flatpak/.flatpakvalidate/lint.sh
```

### Regenerating the Python deps

`python3-deps.yaml` holds the runtime wheels (numpy, loguru, sentry-sdk with its
urllib3 and certifi) and `python3-build-deps.yaml` the build-only hatchling, which
`cleanup: ['*']` removes. pycairo comes from the runtime and PyGObject is its own
module, so the quoth wheel installs `--no-deps`; evdev is left out, since Flatpak
hotkeys use the portal. After a dependency change in `pyproject.toml`, regenerate
both with the **upstream** `flatpak-pip-generator` (from
`flatpak/flatpak-builder-tools`, not the PyPI fork of the same name), then restore
each file's header notes:

```sh
flatpak-pip-generator --runtime=org.gnome.Sdk//50 --prefer-wheels=numpy --yaml \
  -o packaging/flatpak/python3-deps loguru numpy sentry-sdk
flatpak-pip-generator --runtime=org.gnome.Sdk//50 --build-only --yaml \
  -o packaging/flatpak/python3-build-deps hatchling
```

### Pins

[`packaging/pins.sh`](../pins.sh) is the single source of truth for the pins it
names (whisper.cpp, gtk4-layer-shell, ThorVG, spirv-headers, wl-clipboard, ydotool,
OpenVINO, Level Zero, the Intel NPU driver and sentry-sdk). `scripts/sync_pins.sh`
propagates them into this manifest and its modules, and `scripts/sync_pins.sh --check`
guards drift and app-version parity. Do not hand-edit those URLs or hashes here; edit
`pins.sh` and run `sync_pins.sh`. Other Flatpak-only sources, such as libportal and
wtype, keep their pins directly in the manifest.

## Permissions

**`--device=all`** grants Vulkan and Intel NPU access inside the sandbox. It is
broad and can expose other device nodes that Unix access rules permit. The bundled
typing helpers and the setup check run on the host and use the host's
`/dev/uinput`, so the sandbox needs no input-device access. Where ydotool cannot
write the host device, Quoth offers an Enable typing dialog at launch, or after a
failed paste where the launch check could not tell. With the user's approval it
installs an active-seat `uaccess` rule through host `pkexec`;
[Troubleshooting](../../SUPPORT.md#troubleshooting) covers the Polkit agent this
needs. The grant is per seat, not per account; the dialog warns about shared
computers and
[Permissions and known limitations](../../SUPPORT.md#permissions-and-known-limitations)
gives the opt-out.

> [!WARNING]
> **`--talk-name=org.freedesktop.Flatpak`** allows arbitrary command execution on
> the host as the user. Quoth uses it for its bundled clipboard and typing
> helpers, the host permission check and the typing setup, but nothing limits it
> to those commands, so the sandbox is not a security boundary against host
> access. It is needed because COSMIC filters the clipboard and virtual-keyboard
> protocols a sandboxed paste would use, and RemoteDesktop does not replace the
> clipboard transaction or its restoration.

**`--share=network`** fetches the selected speech model on first use only, verified
against a SHA-256 manifest. The only other network uses are off by default: the
update check and error reports (sent to Sentry's EU region, never audio or text).

Everything else is standard: Wayland only (paste needs it), the PipeWire socket
read-only plus PulseAudio for capture, one more `--talk-name`
(`org.kde.StatusNotifierWatcher`, for the tray), and `--persist=.local/state`,
which keeps the log and calibration on Flatpak before 1.13, where no
`XDG_STATE_HOME` is set.

## Distribution

The build self-hosts: `flatpak build-bundle` produces a `quoth.flatpak`
(exported by the validation harness) for direct distribution. The direct remote
can provide signed updates; a standalone bundle can be installed manually.

Flathub publication is deferred. Because `--talk-name=org.freedesktop.Flatpak`
allows host command execution, a submission must replace that text-delivery path
or pass Flathub's permission review.
