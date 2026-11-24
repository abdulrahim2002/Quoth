# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `packaging/patches/whisper-cpu-os-state.patch`.

whisper.cpp's patched x86 CPU backend scorer is compiled against faked CPUID and XGETBV
results, which needs x86_64 and a C++ compiler.
"""

import hashlib
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "tests/fixtures/whisper-cpu-feats-v1.9.5.cpp"
_FIXTURE_SHA256 = "b0f6aaaeb0b12a4f3e15ceec14d0c2031633f91e864a6c15954a001d6390f6d4"
_PATCH = _ROOT / "packaging/patches/whisper-cpu-os-state.patch"

_XGETBV = """\
#ifdef _MSC_VER
    static uint64_t xgetbv(unsigned int xcr) {
        return _xgetbv(xcr);
    }
#else
    static uint64_t xgetbv(unsigned int xcr) {
        unsigned int eax;
        unsigned int edx;
        __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(xcr));
        return (static_cast<uint64_t>(edx) << 32) | eax;
    }
#endif
"""

_CPUID = """\
#ifdef _MSC_VER
    static void cpuid(int cpu_info[4], int eax) {
        __cpuid(cpu_info, eax);
    }
    static void cpuidex(int cpu_info[4], int eax, int ecx) {
        __cpuidex(cpu_info, eax, ecx);
    }
#else
    static void cpuid(int cpu_info[4], int eax) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(0));
    }
    static void cpuidex(int cpu_info[4], int eax, int ecx) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(ecx));
    }
#endif
"""

_HARNESS = r"""
#include <cstdint>
#include <cstdlib>
#include <iostream>

static bool has_xsave;
static bool has_osxsave;
static uint64_t xcr0;
static int xgetbv_calls;

static uint64_t test_xgetbv(unsigned int) {
    ++xgetbv_calls;
    return xcr0;
}

static void test_cpuid(int cpu_info[4], int eax, int ecx) {
    cpu_info[0] = cpu_info[1] = cpu_info[2] = cpu_info[3] = 0;
    if (eax == 0) {
        cpu_info[0] = 7;
        cpu_info[1] = 0x756e6547;
        cpu_info[2] = 0x6c65746e;
        cpu_info[3] = 0x49656e69;
    } else if (eax == 1) {
        cpu_info[2] = (1U << 12) | (1U << 20) | (1U << 28) | (1U << 29);
        cpu_info[2] |= has_xsave ? 1U << 26 : 0;
        cpu_info[2] |= has_osxsave ? 1U << 27 : 0;
    } else if (eax == 7 && ecx == 0) {
        cpu_info[1] = (1U << 5) | (1U << 8) | (1U << 16) | (1U << 17) |
                      (1U << 28) | (1U << 30) | (1U << 31);
        cpu_info[2] = (1U << 1) | (1U << 11);
        cpu_info[3] = 1U << 25;
    } else if (eax == 7 && ecx == 1) {
        cpu_info[0] = (1U << 4) | (1U << 5);
    } else if (static_cast<unsigned int>(eax) == 0x80000000U) {
        cpu_info[0] = 0x80000000U;
    }
}

#include "cpu-feats.cpp"

int main(int argc, char ** argv) {
    if (argc != 4) {
        return 2;
    }
    has_xsave = argv[1][0] == '1';
    has_osxsave = argv[2][0] == '1';
    xcr0 = std::strtoull(argv[3], nullptr, 0);
    std::cout << ggml_backend_cpu_x86_score() << ' ' << xgetbv_calls << '\n';
}
"""


def _patched_scorer(tmp_path):
    assert hashlib.sha256(_FIXTURE.read_bytes()).hexdigest() == _FIXTURE_SHA256
    source = tmp_path / "whisper.cpp/ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp"
    source.parent.mkdir(parents=True)
    shutil.copyfile(_FIXTURE, source)
    result = subprocess.run(
        ["patch", "--quiet", "-d", tmp_path / "whisper.cpp", "-p1"],
        input=_PATCH.read_text(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    scorer = source.read_text()
    assert scorer.count(_XGETBV) == 1
    assert scorer.count(_CPUID) == 1
    scorer = scorer.replace(
        _XGETBV,
        "    static uint64_t xgetbv(unsigned int xcr) { return test_xgetbv(xcr); }\n",
    ).replace(
        _CPUID,
        """\
    static void cpuid(int cpu_info[4], int eax) {
        test_cpuid(cpu_info, eax, 0);
    }
    static void cpuidex(int cpu_info[4], int eax, int ecx) {
        test_cpuid(cpu_info, eax, ecx);
    }
""",
    )
    source.write_text(scorer)
    (source.parent / "ggml-backend-impl.h").write_text(
        "#define GGML_BACKEND_DL_SCORE_IMPL(score_fn)\n"
    )
    return source


def _compile(tmp_path, source, name, features):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    harness = tmp_path / f"{name}.cpp"
    harness.write_text(_HARNESS)
    binary = tmp_path / name
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            *(f"-D{feature}" for feature in features),
            f"-I{source.parent}",
            harness,
            "-o",
            binary,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return binary


def _score(binary, *, xsave, osxsave, xcr0):
    result = subprocess.run(
        [binary, str(int(xsave)), str(int(osxsave)), hex(xcr0)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return tuple(map(int, result.stdout.split()))


@pytest.mark.skipif(
    platform.machine().lower() not in {"amd64", "x86_64"},
    reason="the patched scorer only builds on x86_64",
)
def test_patched_cpu_scorer_requires_enabled_os_state(tmp_path):
    source = _patched_scorer(tmp_path)
    variants = {
        "baseline": ([], 1),
        "generic": (["GGML_SSE42", "GGML_BMI2"], 13),
        "avx": (
            ["GGML_FMA", "GGML_F16C", "GGML_AVX", "GGML_AVX2", "GGML_AVX_VNNI"],
            116,
        ),
        "avx512": (
            [
                "GGML_AVX512",
                "GGML_AVX512_VBMI",
                "GGML_AVX512_BF16",
                "GGML_AVX512_VNNI",
            ],
            1921,
        ),
        "amx": (["GGML_AMX_INT8"], 2049),
    }
    binaries = {
        name: _compile(tmp_path, source, name, features)
        for name, (features, _) in variants.items()
    }

    assert _score(binaries["baseline"], xsave=False, osxsave=False, xcr0=0) == (1, 0)
    assert _score(binaries["generic"], xsave=False, osxsave=False, xcr0=0) == (13, 0)

    for xsave, osxsave in ((False, False), (False, True), (True, False)):
        assert _score(binaries["avx"], xsave=xsave, osxsave=osxsave, xcr0=0x6) == (
            0,
            0,
        )
    assert _score(binaries["avx"], xsave=True, osxsave=True, xcr0=0x2)[0] == 0
    assert _score(binaries["avx"], xsave=True, osxsave=True, xcr0=0x6)[0] == 116

    assert _score(binaries["avx512"], xsave=True, osxsave=True, xcr0=0x6)[0] == 0
    assert _score(binaries["avx512"], xsave=True, osxsave=True, xcr0=0xE6)[0] == 1921

    assert _score(binaries["amx"], xsave=True, osxsave=True, xcr0=0xE6)[0] == 0
    assert _score(binaries["amx"], xsave=True, osxsave=True, xcr0=0x600E6)[0] == 2049


@pytest.mark.skipif(
    platform.machine().lower() not in {"amd64", "x86_64"},
    reason="the patched scorer only builds on x86_64",
)
@pytest.mark.parametrize(
    ("feature", "score", "missing", "enabled"),
    [
        ("GGML_FMA", 2, 0x2, 0x6),
        ("GGML_F16C", 3, 0x2, 0x6),
        ("GGML_AVX", 17, 0x2, 0x6),
        ("GGML_AVX2", 33, 0x2, 0x6),
        ("GGML_AVX_VNNI", 65, 0x2, 0x6),
        ("GGML_AVX512", 129, 0x6, 0xE6),
        ("GGML_AVX512_VBMI", 257, 0x6, 0xE6),
        ("GGML_AVX512_BF16", 513, 0x6, 0xE6),
        ("GGML_AVX512_VNNI", 1025, 0x6, 0xE6),
    ],
)
def test_each_feature_gate_requires_its_os_state(
    tmp_path, feature, score, missing, enabled
):
    # One feature per build (like AMX_INT8 above): a module's first gate hides the rest,
    # and GGML's sandybridge relies on the AVX gate, skylakex on AVX512's for ZMM state.
    binary = _compile(tmp_path, _patched_scorer(tmp_path), "scorer", [feature])
    assert _score(binary, xsave=True, osxsave=True, xcr0=missing)[0] == 0
    assert _score(binary, xsave=True, osxsave=True, xcr0=enabled)[0] == score
