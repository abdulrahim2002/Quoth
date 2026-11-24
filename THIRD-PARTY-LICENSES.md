# Third-party licenses

Quoth is licensed under [GPL-3.0-or-later](LICENSE). This file indexes software
and model assets copied into one or more Quoth packages. Libraries supplied by
the operating system or Flatpak runtime are outside this inventory.

Package recipes and upstream distributions remain the source of the full
copyright and license notices. The `.deb` records bundled native components in
`packaging/debian/copyright`; the Flatpak stages component notices under
`/app/share/licenses` and preserves wheel `.dist-info` metadata; AUR package
sources and license files are declared in `packaging/aur/PKGBUILD`. Bundled
versions are pinned in `packaging/pins.sh`, the Flatpak manifest
`packaging/flatpak/dev.quoth.Quoth.yaml` and its `python3-deps.yaml`.

The Flatpak's complete corresponding source is `quoth-sources.flatpak`, attached
to each release beside `quoth.flatpak`. `quoth-deb-sources.tar.gz`, beside the
`.deb`, holds the unmodified source of the wl-clipboard, ydotool and ThorVG it
bundles at their pinned commits, and the files of the build distribution's
wayland-protocols that wl-clipboard's build generates code from, with that
package's version. The package site at `packages.quoth.dev` says where to find
both.

## Native and desktop components

| Component | Purpose | License | Bundled in |
|---|---|---|---|
| [whisper.cpp and ggml](https://github.com/ggml-org/whisper.cpp) | Speech inference | MIT; compiles in YaRN (MIT) and Vulkan-Hpp (Apache-2.0 OR MIT, used under MIT) | `.deb`, Flatpak, AUR |
| [gtk4-layer-shell](https://github.com/wmww/gtk4-layer-shell) | Wayland overlay | MIT; compiles in code generated from Wayland protocols under MIT and HPND-sell-variant terms | `.deb`, Flatpak; system package on AUR |
| [ThorVG](https://github.com/thorvg/thorvg) | Onboarding Lottie rendering | MIT; compiles in RapidJSON (MIT) and code under the FreeType License and MPL-2.0 | `.deb`, Flatpak, AUR |
| [OpenVINO](https://github.com/openvinotoolkit/openvino) | Intel NPU encoder runtime | Apache-2.0 plus upstream third-party notices; every package takes it from the PyPI wheel | `.deb`, Flatpak, AUR |
| [oneTBB](https://github.com/uxlfoundation/oneTBB) | OpenVINO threading runtime | Apache-2.0 | `.deb`, Flatpak, AUR as part of OpenVINO |
| [hwloc](https://github.com/open-mpi/hwloc) | OpenVINO hardware-locality runtime | BSD-3-Clause | `.deb`, Flatpak, AUR as part of OpenVINO |
| [Intel NPU user-mode driver](https://github.com/intel/linux-npu-driver) | Intel NPU device runtime | MIT plus upstream third-party notices | `.deb`, Flatpak; system package on AUR |
| [Intel NPU compiler](https://github.com/openvinotoolkit/npu_compiler) | Compiles the encoder for the Intel NPU | Apache-2.0; compiles in LLVM and MLIR (Apache-2.0 WITH LLVM-exception, with the legacy NCSA licence and BSD-2-Clause, BSD-3-Clause, Spencer and Unicode notices) | `.deb`, Flatpak; system package on AUR |
| [Level Zero](https://github.com/oneapi-src/level-zero) | NPU driver loader | MIT | `.deb`, Flatpak; system package on AUR |
| [wl-clipboard](https://github.com/bugaevc/wl-clipboard) | Temporary clipboard delivery | GPL-3.0-or-later; compiles in code generated from Wayland protocols under MIT and HPND-sell-variant terms | `.deb`, Flatpak; system package on AUR |
| [wtype](https://github.com/atx/wtype) | Paste chord on wlroots compositors | MIT; compiles in code generated from a Wayland protocol under MIT terms | Flatpak; system package on native channels |
| [ydotool](https://github.com/ReimuNotMoe/ydotool) | Paste chord through `/dev/uinput` | AGPL-3.0-or-later | `.deb`, Flatpak; system package on AUR |
| [libportal](https://github.com/flatpak/libportal) | GlobalShortcuts and Background portal access | LGPL-3.0-only | Flatpak |
| [PyGObject](https://gitlab.gnome.org/GNOME/pygobject) | Python GI bindings, patched against a crash with desktop shortcuts | LGPL-2.1-or-later | Flatpak |
| [SPIR-V Headers](https://github.com/KhronosGroup/SPIRV-Headers) | Vulkan build input | MIT | Flatpak build only; removed from the runtime image |

## Bundled theme and skin palettes

| Palette | Quoth asset | Upstream license | Notice source | Bundled in |
|---|---|---|---|---|
| [Catppuccin](https://github.com/catppuccin/catppuccin) | `themes/catppuccin-mocha.toml`, `skins/catppuccin.toml` | MIT | [`d09787dd`](https://github.com/catppuccin/catppuccin/blob/d09787dd98ca6fba08af5ef2ae94a7e09f17daca/LICENSE), 2026-07-25 | `.deb`, Flatpak, AUR |
| [Dracula](https://github.com/dracula/dracula-theme), with its Alucard light variant | `themes/dracula.toml`, `skins/dracula.toml` | MIT | [`e22ebcc0`](https://github.com/dracula/dracula-theme/blob/e22ebcc048b8aff586f9bc94db30ff5347070b42/LICENSE), 2026-10-01 | `.deb`, Flatpak, AUR |
| [Nord](https://github.com/nordtheme/nord) | `themes/nord.toml`, `skins/nord.toml` | MIT | [`1cef7160`](https://github.com/nordtheme/nord/blob/1cef71605416a222e57225b544540ce0fcec18d4/license), 2023-05-03 | `.deb`, Flatpak, AUR |

The upstream notices are installed unchanged under `share/licenses/quoth`, and
each package's install test checks their SHA-256 against the digest of the
upstream file linked above. Quoth's TOML ports are adaptations licensed under
GPL-3.0-or-later as part of Quoth; the upstream palette copyrights and MIT
permissions remain attributed above.

## Quoth's own artwork

Apart from the five Lucide icons below, the icons in `data/icons/`, the brand
marks in `data/brand/`, the onboarding backdrop (`src/quoth/backgrounds/`), the
onboarding animations (`src/quoth/animations/`) and the recording cues
(`src/quoth/sounds/`, generated by `scripts/generate_sounds.py`) are original
Quoth work under GPL-3.0-or-later. The Lottie JSON files are the animations' editable source.

## Bundled icons

Five symbolic icons (`bug`, `coffee`, `gauge`, `github`, `heart`) come from
[Lucide](https://github.com/lucide-icons/lucide), licensed ISC. They ship as
`dev.quoth.Quoth-*-symbolic.svg` in every package, with the notice installed as
`share/licenses/quoth/LICENSE.lucide-icons`.

## Bundled Python packages

Native packages normally use distribution Python packages. The `.deb` privately
bundles Sentry SDK for opt-in error reports. The Flatpak bundles these runtime
wheels.

| Package | License | Bundled in |
|---|---|---|
| [Loguru](https://github.com/Delgan/loguru) | MIT | Flatpak |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause plus bundled third-party licenses | Flatpak |
| [certifi](https://github.com/certifi/python-certifi) | MPL-2.0 | Flatpak |
| [urllib3](https://github.com/urllib3/urllib3) | MIT | Flatpak |
| [Sentry SDK for Python](https://github.com/getsentry/sentry-python) | MIT | `.deb`, Flatpak; system package on AUR |

NumPy's x86_64 wheel bundles OpenBLAS (BSD-3-Clause) and two GCC runtime
libraries: libgfortran (GPL-3.0-or-later WITH GCC-exception-3.1) and libquadmath
(LGPL-2.1-or-later). Their notices are in NumPy's `.dist-info`, and the LGPL-2.1
text also ships as PyGObject's `COPYING`. The libraries come from CentOS 7's
`gcc-libraries-8.3.1-2.1.1.el7` and `gcc-4.8.5-44.el7` packages, whose source
packages are in `quoth-sources.flatpak`.

Loguru 0.7.3's wheel omits its license file. Quoth installs the
[upstream notice](https://github.com/Delgan/loguru/blob/0.7.3/LICENSE) as
`share/licenses/quoth/LICENSE.loguru` in every package; native packages still
use distribution Loguru. The notice's SHA-256 is
`b35d026cc7aca9d5859a02eb87ddf7a386a24c986838651bd1f283f94e003327`.

Hatchling and its dependencies appear in the Flatpak build manifest but are
removed from the final runtime image.

## Model assets

Quoth bundles one model file in every package:

- `src/quoth/weights/ggml-silero-v6.2.0.bin` is whisper.cpp's GGML conversion
  of [Silero VAD](https://github.com/snakers4/silero-vad), licensed MIT
  (notice [`caddb3b7`](https://github.com/snakers4/silero-vad/blob/caddb3b7ce1dee88a14d5621a0e9a8fdeb2c2c48/LICENSE),
  2020-11-23, installed as `share/licenses/quoth/LICENSE.silero-vad`). Its
  SHA-256 is
  `2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987`.

Speech models are downloaded on demand rather than bundled. Tiny, Base, Small
and Turbo (Whisper large-v3-turbo) derive from [OpenAI Whisper](https://github.com/openai/whisper),
licensed MIT. The `ggml-<model>-q8_0.bin` files use whisper.cpp's GGML format
with 8-bit weights. Each `ggml-<model>-encoder-openvino.xml` and `.bin` pair is
generated with whisper.cpp's `models/convert-whisper-to-openvino.py` from the
corresponding Whisper model.

Those downloaded files are published with the Quoth `v0.1.0` release, whose
`manifest.json` records the SHA-256 of every GGML and OpenVINO encoder file.
Quoth checks each download against it before installing it. These hashes show
that a file is intact; the upstream project and license above give its origin
and terms of reuse.
