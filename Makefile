# Quoth: shared build/stage engine.
#
# Source installs and every native package (AUR / .deb / Flatpak) call these same
# targets; a recipe differs only in PREFIX/DESTDIR and how dependencies arrive.
#
#   make build                                  # libwhisper.so (+ggml) + C shim → lib/
#   make install-data DESTDIR=… PREFIX=/usr     # desktop/icons/metainfo/settings schema + bundle libs
#   make schemas                                # compile the settings schema in data/ (source runs)
#   make vendor-openvino                        # trim the OpenVINO runtime into lib/openvino/ (NPU)
#   make vendor-npu-driver                      # Intel's NPU driver stack → lib/npu/ (.deb)
#   make vendor-layer-shell                     # build gtk4-layer-shell → lib/ (.deb, Flatpak)
#   make vendor-thorvg                          # build the Lottie renderer → lib/
#   make vendor-wl-clipboard                    # build wl-copy/wl-paste → lib/bin/ (.deb)
#   make vendor-ydotool                         # build ydotool/ydotoold → lib/bin/ (.deb)
#   make lab-check PREPARED=… EVIDENCE=…        # desktop-lab readiness checks
#   make clean
#
# Left to each channel's recipe:
#   * the Python wheel: channel-specific (PKGBUILD, pybuild, pip).
#   * icon/desktop cache refresh and settings schema compilation: the packager's job
#     (pacman hooks / dpkg triggers / the Flatpak manifest); running them into a DESTDIR
#     corrupts the staging root, and Arch and Debian do them automatically.

PREFIX  ?= /usr/local
DESTDIR ?=

# whisper.cpp source tree + its Quoth build dir (Vulkan, relocatable: no dev-home RUNPATH).
WHISPER_DIR       ?= $(HOME)/whisper.cpp
WHISPER_BUILD_DIR ?= $(WHISPER_DIR)/build-quoth
WITH_OPENVINO     ?= OFF
OpenVINO_DIR      ?=
OPENVINO_WHEEL_DIR ?=
NPU_DRIVER_SRC    ?=
LAYER_SHELL_SRC   ?=
THORVG_SRC        ?=
WL_CLIPBOARD_SRC  ?=
YDOTOOL_SRC       ?=
JOBS              ?= $(shell nproc)

# Where the installed Python package lives, so native libs land in quoth/lib/;
# whisper.py:_find_lib looks there first. Overridable (a packager can pin the target
# purelib); lazy via DESTLIB (`=`) below so `clean`/`build` never shell out for it.
PYTHON ?= python3
PYSITE ?= $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
QUOTH_RELEASE ?=
QUOTH_ENVIRONMENT ?=
QUOTH_CHANNEL ?= source

REPO    := $(CURDIR)
LIBDIR  := $(REPO)/lib
DATADIR := $(REPO)/data
APPID   := dev.quoth.Quoth
DESTPKG = $(DESTDIR)$(PYSITE)/quoth
DESTLIB = $(DESTPKG)/lib

# Build the Vulkan libwhisper.so (+ggml backends) and the C shim,
# then co-locate them in lib/ so the $ORIGIN RUNPATH resolves.
build:
	WHISPER_DIR="$(WHISPER_DIR)" BUILD_DIR="$(WHISPER_BUILD_DIR)" \
	  WITH_OPENVINO="$(WITH_OPENVINO)" $(if $(OpenVINO_DIR),OpenVINO_DIR="$(OpenVINO_DIR)",) \
	  JOBS="$(JOBS)" "$(REPO)/scripts/build_whisper.sh"
	WHISPER_DIR="$(WHISPER_DIR)" "$(REPO)/scripts/stage_libs.sh" "$(WHISPER_BUILD_DIR)" "$(LIBDIR)"
	WHISPER_DIR="$(WHISPER_DIR)" WHISPER_BUILD_DIR="$(WHISPER_BUILD_DIR)" \
	  "$(REPO)/lib/build_helper.sh"

# Extract the trimmed OpenVINO runtime into lib/openvino/ (Intel NPU encoder).
# OPENVINO_WHEEL_DIR (an extracted wheel) lets a packaging recipe feed its pinned
# source so this never touches the network; unset, it downloads the pinned wheel.
vendor-openvino:
	OPENVINO_WHEEL_DIR="$(OPENVINO_WHEEL_DIR)" "$(REPO)/scripts/vendor_openvino.sh" "$(LIBDIR)/openvino"

# Optional, channel-specific: the .deb bundles Intel's NPU driver stack, since Debian
# and Ubuntu package no NPU driver. Arch packages it and the Flatpak bundles its own.
# NPU_DRIVER_SRC (the pinned downloads) lets a recipe build without the network.
vendor-npu-driver:
	NPU_DRIVER_SRC="$(NPU_DRIVER_SRC)" "$(REPO)/scripts/vendor_npu_driver.sh" "$(LIBDIR)/npu"

# Optional, channel-specific (like vendor-openvino): build a bundled gtk4-layer-shell
# (.so + GI typelib) into lib/ where the platform lacks the package: Ubuntu 24.04 /
# Pop!_OS and the GNOME runtime ship none, so the .deb and the Flatpak bundle it. AUR
# uses the Arch system package and never calls this. LAYER_SHELL_SRC (a pre-fetched
# source tree) lets a recipe feed its pinned checkout so this never touches the
# network.
vendor-layer-shell:
	LAYER_SHELL_SRC="$(LAYER_SHELL_SRC)" "$(REPO)/scripts/vendor_layer_shell.sh" "$(LIBDIR)"

# Build the bundled ThorVG (the Lottie renderer behind the onboarding animations)
# into lib/. Unlike vendor-layer-shell this runs on every channel: no distro ships a
# usable libthorvg. THORVG_SRC (a pre-fetched source tree) lets a recipe feed its
# pinned checkout so this never touches the network.
vendor-thorvg:
	THORVG_SRC="$(THORVG_SRC)" "$(REPO)/scripts/vendor_thorvg.sh" "$(LIBDIR)"

# Optional, channel-specific: the .deb bundles wl-clipboard, since Debian and Ubuntu
# ship a release without `wl-copy --sensitive` and ext-data-control. Arch's has both
# and the Flatpak builds its own. WL_CLIPBOARD_SRC (a pre-fetched source tree) lets a
# recipe feed its pinned checkout so this never touches the network.
vendor-wl-clipboard:
	WL_CLIPBOARD_SRC="$(WL_CLIPBOARD_SRC)" "$(REPO)/scripts/vendor_wl_clipboard.sh" "$(LIBDIR)"

# Optional, channel-specific: the .deb bundles ydotool, since Debian 13 packages it only
# in backports. Arch ships the pinned release and the Flatpak builds its own. YDOTOOL_SRC
# (a pre-fetched git checkout) lets a recipe feed its pinned clone so this never touches
# the network.
vendor-ydotool:
	YDOTOOL_SRC="$(YDOTOOL_SRC)" "$(REPO)/scripts/vendor_ydotool.sh" "$(LIBDIR)"

# A source checkout's settings schema, which config.open_settings() loads from data/.
schemas:
	glib-compile-schemas --strict "$(DATADIR)"

# Desktop entry, icons, metainfo, settings schema, udev rules, licence texts, the
# channel stamp and build identity, and the bundled native libs.
install-data:
	install -Dm644 "$(DATADIR)/$(APPID).desktop" \
	  "$(DESTDIR)$(PREFIX)/share/applications/$(APPID).desktop"
	install -Dm644 "$(DATADIR)/$(APPID).metainfo.xml" \
	  "$(DESTDIR)$(PREFIX)/share/metainfo/$(APPID).metainfo.xml"
	install -Dm644 "$(DATADIR)/$(APPID).gschema.xml" \
	  "$(DESTDIR)$(PREFIX)/share/glib-2.0/schemas/$(APPID).gschema.xml"
	install -Dm644 "$(DATADIR)/licenses/LICENSE.lucide-icons" \
	  "$(DESTDIR)$(PREFIX)/share/licenses/quoth/LICENSE.lucide-icons"
	install -Dm644 "$(DATADIR)/licenses/LICENSE.catppuccin" \
	  "$(DESTDIR)$(PREFIX)/share/licenses/quoth/LICENSE.catppuccin"
	install -Dm644 "$(DATADIR)/licenses/LICENSE.dracula" \
	  "$(DESTDIR)$(PREFIX)/share/licenses/quoth/LICENSE.dracula"
	install -Dm644 "$(DATADIR)/licenses/LICENSE.nord" \
	  "$(DESTDIR)$(PREFIX)/share/licenses/quoth/LICENSE.nord"
	install -Dm644 "$(DATADIR)/licenses/LICENSE.silero-vad" \
	  "$(DESTDIR)$(PREFIX)/share/licenses/quoth/LICENSE.silero-vad"
	install -Dm644 "$(DATADIR)/licenses/LICENSE.loguru" \
	  "$(DESTDIR)$(PREFIX)/share/licenses/quoth/LICENSE.loguru"
	cd "$(DATADIR)/icons" && find hicolor -type f -exec \
	  install -Dm644 "{}" "$(DESTDIR)$(PREFIX)/share/icons/{}" \;
# Host access to uinput for ydotool and to the NPU, from the native channels: a sandbox
# cannot install host rules.
	@if [ -n "$(QUOTH_CHANNEL)" ]; then \
	  install -Dm644 -t "$(DESTDIR)$(PREFIX)/lib/udev/rules.d" \
	    "$(DATADIR)/60-quoth-uinput.rules" "$(DATADIR)/60-quoth-accel.rules"; \
	fi
	install -d "$(DESTLIB)"
	@set -e; channel="$(QUOTH_CHANNEL)"; \
	  if [ -n "$$channel" ]; then \
	    case "$$channel" in deb|aur|source) ;; \
	      *) echo "install-data: invalid QUOTH_CHANNEL: $$channel" >&2; exit 2 ;; \
	    esac; \
	    tmp="$$(mktemp)"; trap 'rm -f "$$tmp"' EXIT; \
	    printf 'CHANNEL = "%s"\n' "$$channel" >"$$tmp"; \
	    install -m644 "$$tmp" "$(DESTPKG)/_channel.py"; \
	  else \
	    rm -f "$(DESTPKG)/_channel.py"; \
	  fi
	@set -e; if [ -n "$(QUOTH_RELEASE)$(QUOTH_ENVIRONMENT)" ]; then \
	  [ -n "$(QUOTH_RELEASE)" ] && [ -n "$(QUOTH_ENVIRONMENT)" ] || { \
	    echo "install-data: QUOTH_RELEASE and QUOTH_ENVIRONMENT must be set together" >&2; \
	    exit 2; \
	  }; \
	  tmp="$$(mktemp)"; trap 'rm -f "$$tmp"' EXIT; \
	  "$(PYTHON)" -c 'import json, sys; release, environment = sys.argv[1:]; environment in {"development", "candidate", "production"} or sys.exit("invalid Quoth build environment"); print(json.dumps({"environment": environment, "release": release}, separators=(",", ":"), sort_keys=True))' \
	    "$(QUOTH_RELEASE)" "$(QUOTH_ENVIRONMENT)" >"$$tmp"; \
	  install -m644 "$$tmp" "$(DESTLIB)/build-identity.json"; \
	else \
	  rm -f "$(DESTLIB)/build-identity.json"; \
	fi
	cp -aP "$(LIBDIR)"/libwhisper.so* "$(LIBDIR)"/libggml*.so* "$(DESTLIB)/"
	install -m755 "$(LIBDIR)/libwhisper_helper.so" "$(DESTLIB)/"
	install -m644 "$(LIBDIR)/LICENSE.whisper-cpp" "$(DESTLIB)/"
	@if [ -d "$(LIBDIR)/openvino" ]; then \
	  echo "install-data: bundling vendored OpenVINO"; \
	  install -d "$(DESTLIB)/openvino"; \
	  cp -aP "$(LIBDIR)/openvino/." "$(DESTLIB)/openvino/"; \
	fi
	@if [ -d "$(LIBDIR)/npu" ]; then \
	  echo "install-data: bundling the Intel NPU driver stack"; \
	  install -d "$(DESTLIB)/npu"; \
	  cp -aP "$(LIBDIR)/npu/." "$(DESTLIB)/npu/"; \
	fi
	@if [ -e "$(LIBDIR)/libthorvg-1.so" ]; then \
	  echo "install-data: bundling thorvg"; \
	  cp -aP "$(LIBDIR)"/libthorvg-1.so* "$(DESTLIB)/"; \
	  install -m644 "$(LIBDIR)/LICENSE.thorvg" "$(DESTLIB)/"; \
	fi
	@if [ -e "$(LIBDIR)/libgtk4-layer-shell.so" ]; then \
	  echo "install-data: bundling gtk4-layer-shell (.so + typelib)"; \
	  cp -aP "$(LIBDIR)"/libgtk4-layer-shell.so* "$(DESTLIB)/"; \
	  install -m644 "$(LIBDIR)/Gtk4LayerShell-1.0.typelib" "$(DESTLIB)/"; \
	  install -m644 "$(LIBDIR)/LICENSE.gtk4-layer-shell" "$(DESTLIB)/"; \
	fi
	@if [ -e "$(LIBDIR)/bin/wl-copy" ]; then \
	  echo "install-data: bundling wl-clipboard"; \
	  install -Dm755 -t "$(DESTLIB)/bin" "$(LIBDIR)/bin/wl-copy" "$(LIBDIR)/bin/wl-paste"; \
	  install -m644 "$(LIBDIR)/LICENSE.wl-clipboard" "$(DESTLIB)/"; \
	fi
	@if [ -e "$(LIBDIR)/bin/ydotool" ]; then \
	  echo "install-data: bundling ydotool"; \
	  install -Dm755 -t "$(DESTLIB)/bin" "$(LIBDIR)/bin/ydotool" "$(LIBDIR)/bin/ydotoold"; \
	  install -m644 "$(LIBDIR)/LICENSE.ydotool" "$(DESTLIB)/"; \
	fi

# Remove what the build and vendor targets staged in lib/, and the whisper build dir.
clean:
	rm -f "$(LIBDIR)"/libwhisper.so* "$(LIBDIR)"/libggml*.so* "$(LIBDIR)/LICENSE.whisper-cpp"
	rm -f "$(LIBDIR)/libwhisper_helper.so"
	rm -f "$(LIBDIR)"/libgtk4-layer-shell.so* "$(LIBDIR)/Gtk4LayerShell-1.0.typelib" \
	  "$(LIBDIR)/LICENSE.gtk4-layer-shell"
	rm -f "$(LIBDIR)"/libthorvg-1.so* "$(LIBDIR)/LICENSE.thorvg"
	rm -rf "$(LIBDIR)/bin" "$(LIBDIR)/LICENSE.wl-clipboard" "$(LIBDIR)/LICENSE.ydotool"
	rm -rf "$(LIBDIR)/openvino" "$(LIBDIR)/npu" "$(WHISPER_BUILD_DIR)"

export QUOTH_LAB_PREPARED := $(value PREPARED)
export QUOTH_LAB_EVIDENCE := $(value EVIDENCE)

.PHONY: build vendor-openvino vendor-npu-driver vendor-layer-shell vendor-thorvg vendor-wl-clipboard vendor-ydotool install-data schemas clean \
	lab-check

lab-check:
	@test -n "$$QUOTH_LAB_PREPARED" -a -n "$$QUOTH_LAB_EVIDENCE" || { echo "make lab-check requires PREPARED=<manifest> EVIDENCE=<new-directory>" >&2; exit 64; }
	@uv run --locked --project "$(REPO)/lab" pytest "$(REPO)/lab/test_desktop.py" \
		--prepared "$$QUOTH_LAB_PREPARED" --evidence "$$QUOTH_LAB_EVIDENCE" \
		--junitxml "$$QUOTH_LAB_EVIDENCE/junit.xml"
