# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# shellcheck shell=bash
# shellcheck disable=SC2034  # every value here is read by whoever sources this file
# Single source of truth for Quoth's pinned packaging dependencies.
#
# Repo-built consumers source this directly (zero copies): the .deb build harness,
# the scripts/vendor_*.sh scripts, and the Flatpak build. The standalone
# AUR PKGBUILDs/.SRCINFO and the Flatpak manifest carry their own copies (they are
# published/built without repo access); scripts/sync_pins.sh propagates these values
# into them and `scripts/sync_pins.sh --check` guards against drift.
#
# To bump an engine: edit the value here, then run `scripts/sync_pins.sh`.

# Sentry SDK: bundled privately in the Debian package, installed from this exact
# wheel in Flatpak, and used as the minimum supported version elsewhere.
QUOTH_SENTRY_SDK_VERSION="2.72.0"
QUOTH_SENTRY_SDK_WHEEL="sentry_sdk-2.72.0-py3-none-any.whl"
QUOTH_SENTRY_SDK_URL="https://files.pythonhosted.org/packages/d0/0b/0dd9686842d9dc49c3ee84cced31c54bceb0fb52a826be7d3d4fe30d882a/sentry_sdk-2.72.0-py3-none-any.whl"
QUOTH_SENTRY_SDK_SHA256="3e13ace4ffd0b3cc78288236edfc4e4bd90eb12a396cc5845fa10a58ff289f50"

# whisper.cpp: includes Vulkan Flash Attention memory-correctness fixes.
QUOTH_WHISPER_TAG="v1.9.5"
QUOTH_WHISPER_SHA256="ff1a9053feb509ff9d7729703355541ae9690073a6b1c40eb692c962e0dc1720"

# OpenVINO: the Intel-NPU encoder runtime (manylinux wheel; trimmed to ~40 MiB by
# scripts/vendor_openvino.sh). The core soname tracks the release; the keep-set and
# preload are soname-agnostic, so recipes do not hardcode the soname.
QUOTH_OPENVINO_VERSION="2026.4.1"
QUOTH_OPENVINO_WHEEL="openvino-2026.4.1-22982-cp310-cp310-manylinux_2_28_x86_64.whl"
QUOTH_OPENVINO_URL="https://files.pythonhosted.org/packages/c4/57/b2d53cfb56be03449a95df18a76fe443ae94b16b3ae15128ca65e08dcd91/openvino-2026.4.1-22982-cp310-cp310-manylinux_2_28_x86_64.whl"
QUOTH_OPENVINO_SHA256="47dcccbab40a23aed1dc4af7c43a9c813cc61f684d66498e1627d407d5f2770b"

# Intel NPU runtime. The .deb and the Flatpak take OpenVINO from the wheel above and
# bundle the driver release's user-mode driver and compiler with the Level Zero loader
# its validated matrix lists, 1.32.0.
QUOTH_NPU_DRIVER_URL="https://github.com/intel/linux-npu-driver/releases/download/v1.38.0/linux-npu-driver-v1.38.0.20260910-34487311128-ubuntu2404.tar.gz"
QUOTH_NPU_DRIVER_SHA256="1efcd4b60c22abee751d8f2705962cbcc2a569de45c7e0e670cf08afbfcdc1d2"
QUOTH_NPU_UMD_DEB="intel-level-zero-npu_1.38.0.20260910-34487311128~ubuntu24.04_amd64.deb"
QUOTH_NPU_UMD_SHA256="e53f2563de9072e9fc36227d4e28aad5bdb7d75d03d43f87436f8f7ab0b98955"
QUOTH_NPU_COMPILER_DEB="intel-driver-compiler-npu_1.38.0.20260910-34487311128~ubuntu24.04_amd64.deb"
QUOTH_NPU_COMPILER_SHA256="472cdce7808e5dd5f9a239f9f3f918f0673fa8ccf93a5e1b45290f4daff41a88"
QUOTH_NPU_LICENSE_URL="https://raw.githubusercontent.com/intel/linux-npu-driver/v1.38.0/LICENSE.md"
QUOTH_NPU_LICENSE_SHA256="451963b6682694730dbe4889fff2ef1c20def68992e2594880c15a28e6c87be5"
QUOTH_NPU_NOTICES_URL="https://raw.githubusercontent.com/intel/linux-npu-driver/v1.38.0/third-party-programs.txt"
QUOTH_NPU_NOTICES_SHA256="7ae92abe83bee3b7bac9197a4b9a151cecfce28ed8eb0a8cbd4ace5ee4ccb75a"
# The compiler's licence, at the npu_compiler revision the driver release builds it from.
QUOTH_NPU_COMPILER_LICENSE_URL="https://raw.githubusercontent.com/openvinotoolkit/npu_compiler/0b38f7d42113ff329ac2bdd33583d123de4ccf2f/LICENSE"
QUOTH_NPU_COMPILER_LICENSE_SHA256="9ae406b07d50d65dcdae0de9286c0a56f52f5159d7f4df05a345409be8cd262c"
QUOTH_LEVEL_ZERO_URL="https://github.com/oneapi-src/level-zero/releases/download/v1.32.0/libze1_1.32.0%2Bu24.04_amd64.deb"
QUOTH_LEVEL_ZERO_SHA256="3c846af24f84a89150f6a4c6adcb4ea4ebef74dc119fe44f4e269bfaa72c7ba6"
QUOTH_LEVEL_ZERO_LICENSE_URL="https://raw.githubusercontent.com/oneapi-src/level-zero/v1.32.0/LICENSE"
QUOTH_LEVEL_ZERO_LICENSE_SHA256="f2e82047a47ca742088331ae2bc85b97949ed1e97dfa057cdafd03e8709fc0a2"

# gtk4-layer-shell: built from source for channels whose distro lacks the package
# (.deb, Flatpak); the AUR uses the Arch system package. The commit pin makes the
# Flatpak source and every tag clone reproducible.
QUOTH_LAYER_SHELL_TAG="v1.3.0"
QUOTH_LAYER_SHELL_COMMIT="1c963c51514581c41b9bdae08cdf69171265cdda"

# ThorVG: the Lottie renderer behind the onboarding animations. Built from source on
# every channel; no distro ships a usable libthorvg (Arch has it in the AUR only,
# Debian and Ubuntu not at all). Built with the lottie loader alone, which links only
# libstdc++/libm/libc, so it adds no system dependency anywhere.
QUOTH_THORVG_TAG="v1.1.3"
QUOTH_THORVG_SHA256="ba9957aeeadb4151d4486f11c638ae118c00210fa4906daa7b30143aa341e690"
QUOTH_THORVG_COMMIT="589bc4120f2097fb1b135e552c2ef39f6581612b"

# wl-clipboard: the .deb and the Flatpak bundle 2.3.0, the first release with
# `wl-copy --sensitive` and ext-data-control-v1 (the only data-control protocol on
# KWin 6.5 and later). Every Debian and Ubuntu target ships 2.2.1; Arch ships 2.3.0.
QUOTH_WL_CLIPBOARD_TAG="v2.3.0"
QUOTH_WL_CLIPBOARD_COMMIT="67a7b937895bceec1ae5ccebb10216f63f70ca1b"

# ydotool: the uinput paste chord. The .deb and the Flatpak bundle 1.0.4, which Ubuntu
# 26.04 and Arch ship; Debian 13 packages it only in backports.
QUOTH_YDOTOOL_TAG="v1.0.4"
QUOTH_YDOTOOL_COMMIT="57ba7d0af525e82da2de0e275d169477f293b197"

# SPIRV-Headers: whisper.cpp's Vulkan build hard-requires find_package(SPIRV-Headers),
# whose CMake config the GNOME SDK does not ship, so the Flatpak supplies it as a
# header-only module. Flatpak-only (the .deb and AUR get it from the distro's
# spirv-headers package); pinned here for visibility and the drift guard. Must track
# whisper.cpp's minimum.
QUOTH_SPIRV_HEADERS_TAG="vulkan-sdk-1.4.350.0"
QUOTH_SPIRV_HEADERS_COMMIT="ad9184e76a66b1001c29db9b0a3e87f646c64de0"
