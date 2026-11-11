"""What the C++ of the CPU player is compiled for: a standard x86-64 level, not this CPU's own model,
as engines ship builds for."""

import os
from pathlib import Path

LEVELS = {
    "v4": ["-march=x86-64-v4", "-mavx512vnni", "-mtune=generic"],
    "v3": ["-march=x86-64-v3", "-mtune=generic"],
    "v2": ["-march=x86-64-v2", "-mtune=generic"],
}
AVX512 = {"avx512f", "avx512bw", "avx512vl", "avx512dq", "avx512cd", "avx512_vnni"}


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


def target_flags() -> list[str]:
    """x86-64-v4 (AVX-512) with VNNI where the CPU has them, else x86-64-v3 (AVX2), or v2, which
    has no native network (`cpu_net.h`). $TURBOZERO_X86_LEVEL ("v4", "v3", "v2") picks one, e.g.
    to play as an AVX2 build would."""
    level = os.environ.get("TURBOZERO_X86_LEVEL")
    if level:
        return LEVELS[level]
    flags = cpu_flags()
    if AVX512 <= flags:
        return LEVELS["v4"]
    if {"avx2", "bmi2", "fma", "movbe"} <= flags:
        return LEVELS["v3"]
    return LEVELS["v2"]
