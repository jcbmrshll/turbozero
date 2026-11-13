"""What the C++ of the CPU player is compiled for: a standard level of its architecture (x86-64 or
ARM), not this CPU's own model, as engines ship builds for."""

import os
import platform
import subprocess
from pathlib import Path

LEVELS = {
    "v4": ["-march=x86-64-v4", "-mavx512vnni", "-mtune=generic"],
    # x86-64-v3 and AVX-VNNI (Intel's client chips from Alder Lake, which have no AVX-512)
    "v3-vnni": ["-march=x86-64-v3", "-mavxvnni", "-mtune=generic"],
    "v3": ["-march=x86-64-v3", "-mtune=generic"],
    "v2": ["-march=x86-64-v2", "-mtune=generic"],
    # ARMv8.2 with the dot product and the 8-bit matrix multiply: Apple's M2 on, AWS Graviton 3 on,
    # Google's Axion
    "arm-i8mm": ["-march=armv8.2-a+dotprod+i8mm"],
    # ARMv8.2 with the dot product: Apple's M1 on, AWS Graviton 2 on
    "arm-dotprod": ["-march=armv8.2-a+dotprod"],
    "arm": ["-march=armv8-a"],
}
# the CPU flags each level needs, as Linux names them, the fastest first
V3 = {"avx2", "bmi2", "fma", "movbe"}
NEEDS = {
    "x86": {
        "v4": {
            "avx512f",
            "avx512bw",
            "avx512vl",
            "avx512dq",
            "avx512cd",
            "avx512_vnni",
        },
        "v3-vnni": V3 | {"avx_vnni"},
        "v3": V3,
        "v2": set(),
    },
    "arm": {"arm-i8mm": {"asimddp", "i8mm"}, "arm-dotprod": {"asimddp"}, "arm": set()},
}


def cpus() -> list[int]:
    """The CPUs this process may run on: its affinity (Linux's taskset), or all of them where the OS
    has none to read (macOS)."""
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def architecture() -> str:
    return "arm" if platform.machine().lower() in ("aarch64", "arm64") else "x86"


def cpu_flags() -> set[str]:
    """The CPU's feature flags, as Linux lists them; on macOS, those of them its sysctl reports."""
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        text = ""
    for line in text.splitlines():
        if line.startswith(("flags", "Features")):
            return set(line.split(":", 1)[1].split())
    if platform.system() == "Darwin":
        names = {"asimddp": "FEAT_DotProd", "i8mm": "FEAT_I8MM"}
        found = set()
        for flag, name in names.items():
            out = subprocess.run(
                ["sysctl", "-n", f"hw.optional.arm.{name}"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            if out == "1":
                found.add(flag)
        return found
    return set()


def levels() -> list[str]:
    """The levels this CPU runs, the fastest first."""
    flags = cpu_flags()
    return [level for level, needs in NEEDS[architecture()].items() if needs <= flags]


def target_flags() -> list[str]:
    """The fastest level the CPU runs: on x86, x86-64-v4 (AVX-512) with VNNI, x86-64-v3 (AVX2) with
    AVX-VNNI, x86-64-v3, or v2; on ARM, ARMv8.2 with the dot product and i8mm, with the dot product,
    or ARMv8. The last of each has
    no native network (`cpu_net.h`). $TURBOZERO_CPU_LEVEL (one of `LEVELS`) picks one, e.g. to play
    as an AVX2 build would."""
    level = os.environ.get("TURBOZERO_CPU_LEVEL")
    return LEVELS[level or levels()[0]]
