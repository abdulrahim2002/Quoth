# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `scripts/package_sbom.py`.

Each package's bill of materials lists exactly what its build bundles from the pins.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

import pytest
import yaml

from scripts import package_sbom

_ROOT = Path(__file__).resolve().parents[1]
_COMMIT = "a" * 40
_UUID = r"urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
# CycloneDX 1.6's component types and the two reference types used here.
_TYPES = {
    "application",
    "framework",
    "library",
    "container",
    "platform",
    "operating-system",
    "device",
    "device-driver",
    "firmware",
    "file",
    "machine-learning-model",
    "data",
    "cryptographic-asset",
}


def _write(tmp_path, channel, commit=_COMMIT, name="sbom.json"):
    output = tmp_path / name
    subprocess.run(
        [
            sys.executable,
            _ROOT / "scripts/package_sbom.py",
            channel,
            "--commit",
            commit,
            "--output",
            output,
        ],
        check=True,
    )
    return output


def _urls(document):
    return {
        reference["url"]
        for component in document["components"]
        for reference in component["externalReferences"]
    }


@pytest.mark.parametrize("channel", ["deb", "flatpak"])
def test_the_bill_has_cyclonedx_required_fields(tmp_path, channel):
    document = json.loads(_write(tmp_path, channel).read_text())

    # actions/attest takes JSON with these three fields as CycloneDX.
    assert document["bomFormat"] == "CycloneDX"
    assert document["specVersion"] == "1.6"
    assert re.fullmatch(_UUID, document["serialNumber"])
    assert document["version"] == 1
    assert document["metadata"]["component"] == {
        "type": "application",
        "name": "quoth",
        "version": "0.1.0",
    }
    names = [component["name"] for component in document["components"]]
    assert names == sorted(set(names))
    for component in document["components"]:
        assert component["type"] in _TYPES
        assert component["name"] and component["version"]
        assert not component["version"].startswith("v")
        for digest in component.get("hashes", []):
            assert digest["alg"] == "SHA-256"
            assert re.fullmatch(r"[0-9a-f]{64}", digest["content"])
        (reference,) = component["externalReferences"]
        assert reference["type"] in {"vcs", "distribution"}
        assert reference["url"].startswith("https://")
        for digest in reference.get("hashes", []):
            assert re.fullmatch(r"[0-9a-f]{64}", digest["content"])
        # A source is either a hashed download or a git commit.
        if reference["type"] == "vcs":
            assert re.fullmatch(
                r"pkg:github/[a-z0-9._-]+/[a-z0-9._-]+@[0-9a-f]{40}", component["purl"]
            )
        else:
            assert len(reference["hashes"]) == 1


@pytest.mark.parametrize("channel", ["deb", "flatpak"])
def test_the_bill_is_byte_stable_and_named_by_commit_and_channel(tmp_path, channel):
    first = _write(tmp_path, channel, name="first.json").read_bytes()
    second = _write(tmp_path, channel, name="second.json").read_bytes()
    other = _write(tmp_path, channel, "b" * 40, name="other.json").read_bytes()

    assert first == second
    assert first.endswith(b"}\n")
    serial = json.loads(first)["serialNumber"]
    assert json.loads(other)["serialNumber"] != serial
    assert (
        package_sbom.sbom("flatpak" if channel == "deb" else "deb", _COMMIT)[
            "serialNumber"
        ]
        != serial
    )


def test_the_bill_refuses_an_abbreviated_commit(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        _write(tmp_path, "deb", "a" * 12)


def test_pins_are_read_as_the_shell_reads_them():
    shell = subprocess.run(
        [
            "bash",
            "-c",
            '. "$1"; for v in $(compgen -v QUOTH_); do printf "%s=%s\\n" "$v" "${!v}"; done',
            "bash",
            _ROOT / "packaging/pins.sh",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    expected = dict(line.split("=", 1) for line in shell.splitlines())
    assert package_sbom.read_pins(_ROOT / "packaging/pins.sh") == expected


class _Reads(dict):
    """A mapping that records which keys were read."""

    def __init__(self, *args):
        super().__init__(*args)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)


def test_every_pin_is_listed_or_named_as_bundling_nothing():
    # A new pin fails here until the bill lists it or says why it bundles nothing.
    pins = _Reads(package_sbom.read_pins(_ROOT / "packaging/pins.sh"))
    package_sbom.pinned_components(pins)

    assert not pins.read & package_sbom.UNLISTED_PINS
    assert pins.read | package_sbom.UNLISTED_PINS == set(pins)


def test_the_deb_lists_the_components_its_harness_fetches(tmp_path):
    document = json.loads(_write(tmp_path, "deb").read_text())
    harness = (_ROOT / "packaging/debian/.dockervalidate/run.sh").read_text()
    pins = package_sbom.read_pins(_ROOT / "packaging/pins.sh")

    # The harness clones each from the URL the bill names, at the pinned tag.
    for name, url in package_sbom.GIT_SOURCES.items():
        assert re.search(
            rf'git clone --depth 1 --branch "\$\w+" {re.escape(url)} ', harness
        )
        (component,) = (c for c in document["components"] if c["name"] == name)
        assert component["externalReferences"][0]["url"] == url
    assert (
        "https://github.com/ggml-org/whisper.cpp/archive/refs/tags/${WHISPER_TAG}.tar.gz"
        in harness
    )
    assert {c["name"] for c in document["components"]} == {
        "whisper.cpp",
        "openvino",
        "intel-level-zero-npu",
        "intel-driver-compiler-npu",
        "level-zero",
        "gtk4-layer-shell",
        "thorvg",
        "wl-clipboard",
        "ydotool",
        "sentry-sdk",
    }
    # Its downloads are the pinned ones, each with its pinned hash.
    hashed = {
        r["url"]: r["hashes"][0]["content"]
        for c in document["components"]
        for r in c["externalReferences"]
        if r["type"] == "distribution"
    }
    for url, digest in hashed.items():
        assert digest in pins.values()
        assert url in pins.values() or url.endswith(
            f"/{pins['QUOTH_WHISPER_TAG']}.tar.gz"
        )


# Manifest modules that put nothing of a third party into the installed app.
_FLATPAK_BUILD_ONLY = {
    # The build backend, removed again by its `cleanup: ['*']`.
    "python3-hatchling",
    # Header-only; the manifest's cleanup removes /include and /share/cmake.
    "spirv-headers",
    # Builds nothing: it puts the NumPy wheel's GCC runtime source in the source bundle.
    "numpy-gcc-runtime-source",
}


def _manifest_sources():
    """Every x86_64 download of the Flatpak manifest, by module name."""
    modules = package_sbom.manifest_modules(package_sbom._MANIFEST)
    return modules, {
        name: [
            source
            for source in module.get("sources", [])
            if source["type"] in {"git", "archive", "file"}
            and "x86_64" in source.get("only-arches", ["x86_64"])
        ]
        for name, module in modules.items()
    }


def test_the_flatpak_lists_every_module_source_it_installs(tmp_path):
    document = json.loads(_write(tmp_path, "flatpak").read_text())
    modules, sources = _manifest_sources()
    manifest = yaml.safe_load(package_sbom._MANIFEST.read_text())

    assert modules["python3-hatchling"]["cleanup"] == ["*"]
    assert {"/include", "/share/cmake"} <= set(manifest["cleanup"])
    assert modules["numpy-gcc-runtime-source"]["build-commands"] == []
    expected = set()
    for name, downloads in sources.items():
        if name in _FLATPAK_BUILD_ONLY:
            continue
        for source in downloads:
            licence = re.search(
                r"LICENSE|third-party-programs", source.get("dest-filename", "")
            )
            # The quoth module's source without a destination is Quoth itself.
            app = name == "quoth" and "dest" not in source
            if not (licence or app):
                expected.add(source["url"])
    assert _urls(document) == expected


def test_the_flatpak_adds_only_its_own_modules_to_the_deb_list(tmp_path):
    deb = json.loads(_write(tmp_path, "deb", name="deb.json").read_text())
    flatpak = json.loads(_write(tmp_path, "flatpak", name="flatpak.json").read_text())
    _modules, sources = _manifest_sources()

    added = [c for c in flatpak["components"] if c not in deb["components"]]
    assert len(added) == len(package_sbom.FLATPAK_ONLY)
    assert len(flatpak["components"]) == len(deb["components"]) + len(added)
    for module in package_sbom.FLATPAK_ONLY:
        (source,) = sources[module]
        (component,) = (
            c for c in added if c["externalReferences"][0]["url"] == source["url"]
        )
        # The version is the one the manifest's tag or file name pins.
        pinned = source.get("tag") or unquote(source["url"].rsplit("/", 1)[1])
        assert component["version"] in pinned
        assert source.get("commit", "") in component.get("purl", "")


def test_purls_take_the_canonical_case_and_names(tmp_path):
    # The purl github type lowercases owner and repository; pypi also folds -_. runs.
    document = json.loads(_write(tmp_path, "flatpak").read_text())
    purls = {c["name"]: c["purl"] for c in document["components"] if "purl" in c}
    assert purls["ydotool"].startswith("pkg:github/reimunotmoe/ydotool@")
    for purl in purls.values():
        kind, name = re.fullmatch(r"pkg:(\w+)/(.+)@.+", purl).groups()
        assert name == name.lower()
        if kind == "pypi":
            assert not re.search(r"[_.]|--", name)
    assert (
        package_sbom._pypi("Typing__Extensions", "1") == "pkg:pypi/typing-extensions@1"
    )


def test_whisper_cpp_names_every_patch_the_recipes_apply(tmp_path):
    # Adding or removing a patch must change what the bill says was applied.
    applied = re.findall(
        r'"\$ROOT/packaging/patches/([\w.-]+\.patch)"',
        (_ROOT / "scripts/apply_whisper_patches.sh").read_text(),
    )
    shipped = sorted(p.name for p in (_ROOT / "packaging/patches").glob("*.patch"))
    assert sorted(applied) == shipped
    for channel in ("deb", "flatpak"):
        document = json.loads(
            _write(tmp_path, channel, name=f"{channel}.json").read_text()
        )
        (whisper,) = (c for c in document["components"] if c["name"] == "whisper.cpp")
        named = re.findall(r"[\w.-]+\.patch", whisper["pedigree"]["notes"])
        assert named == shipped
        others = [c for c in document["components"] if c["name"] != "whisper.cpp"]
        assert not any("pedigree" in c for c in others)
    recipe = package_sbom._MANIFEST.read_text()
    harness = (_ROOT / "packaging/debian/.dockervalidate/run.sh").read_text()
    assert "scripts/apply_whisper_patches.sh --apply whisper.cpp" in recipe
    assert '"$SRC/scripts/apply_whisper_patches.sh" --apply whisper.cpp' in harness


def test_every_component_matches_the_manifest_source_it_names(tmp_path):
    # The Flatpak bill holds the deb's components unchanged, and the manifest is kept
    # equal to pins.sh, so this checks each value of both bills against its recipe.
    document = json.loads(_write(tmp_path, "flatpak").read_text())
    _modules, sources = _manifest_sources()
    by_url = {s["url"]: s for downloads in sources.values() for s in downloads}
    for component in document["components"]:
        (reference,) = component["externalReferences"]
        source = by_url[reference["url"]]
        pinned = source.get("tag") or unquote(source["url"].rsplit("/", 1)[1])
        assert re.match(r"[\d.]+", component["version"])[0] in pinned
        if "commit" in source:
            assert component["purl"].endswith("@" + source["commit"])
            assert component["purl"] == component["purl"].lower()
        else:
            assert reference["hashes"] == [
                {"alg": "SHA-256", "content": source["sha256"]}
            ]
        if component.get("purl", "").startswith("pkg:pypi/"):
            assert component["purl"].endswith("@" + component["version"])
