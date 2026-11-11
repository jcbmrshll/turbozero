"""What the C++ of the CPU player is compiled for: a standard x86-64 level, not this CPU's own model,
as engines ship builds for."""

import os
from pathlib import Path

LEVELS = {
    "v4": ["-march=x86-64-v4", "-mavx512vnni", "-mtune=generic"],
    # x86-64-v3 and AVX-VNNI (Intel's client chips from Alder Lake, which have no AVX-512)
    "v3-vnni": ["-march=x86-64-v3", "-mavxvnni", "-mtune=generic"],
    "v3": ["-march=x86-64-v3", "-mtune=generic"],
    "v2": ["-march=x86-64-v2", "-mtune=generic"],
}
# the CPU flags each level needs, the fastest first
V3 = {"avx2", "bmi2", "fma", "movbe"}
NEEDS = {
    "v4": {"avx512f", "avx512bw", "avx512vl", "avx512dq", "avx512cd", "avx512_vnni"},
    "v3-vnni": V3 | {"avx_vnni"},
    "v3": V3,
    "v2": set(),
}


def cpu_flags() -> set[str]:
    """The CPU's feature flags, as Linux lists them."""
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return set()
    for line in text.splitlines():
        if line.startswith("flags"):
            return set(line.split(":", 1)[1].split())
    return set()


def levels() -> list[str]:
    """The levels this CPU runs, the fastest first."""
    flags = cpu_flags()
    return [level for level, needs in NEEDS.items() if needs <= flags]


def target_flags() -> list[str]:
    """The fastest level the CPU runs: x86-64-v4 (AVX-512) with VNNI, x86-64-v3 (AVX2) with
    AVX-VNNI, x86-64-v3, or v2, which has no native network (`cpu_net.h`). $TURBOZERO_X86_LEVEL
    (one of `LEVELS`) picks one, e.g. to play as an AVX2 build would."""
    level = os.environ.get("TURBOZERO_X86_LEVEL")
    return LEVELS[level or levels()[0]]
