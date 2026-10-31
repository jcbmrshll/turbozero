#!/bin/sh
# Builds Egaroucid for Console (https://github.com/Nyanyan/Egaroucid), a very strong
# open-source Othello engine, for vs_engine.py: release 7.8.4, with the evaluation weights
# it ships, into ~/.cache/turbozero/othello/engines-2026-10/egaroucid (or $EGAROUCID_DIR).
#
#     examples/othello/setup_egaroucid.sh
#
# Needs git and a C++20 compiler (g++ or clang++). Each Egaroucid process takes about
# 1.3 GB of memory, so vs_engine.py runs as many games at a time as memory allows.
set -eu

EGAROUCID_COMMIT=3b1c6ad380209162c2ad07aefe5c1fc6fd8072cb
EGAROUCID_DIR=${EGAROUCID_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/turbozero/othello/engines-2026-10/egaroucid}
CXX=${CXX:-$(command -v clang++ >/dev/null 2>&1 && echo clang++ || echo g++)}

for tool in git "$CXX"; do
    command -v "$tool" >/dev/null 2>&1 || { echo "setup_egaroucid.sh: needs $tool" >&2; exit 1; }
done

mkdir -p "$(dirname "$EGAROUCID_DIR")"
# only the pinned commit: the repository's history holds many large weight files
if [ ! -d "$EGAROUCID_DIR/.git" ]; then
    git init --quiet "$EGAROUCID_DIR"
    git -C "$EGAROUCID_DIR" remote add origin https://github.com/Nyanyan/Egaroucid.git
fi
cd "$EGAROUCID_DIR"
git fetch --quiet --depth 1 origin "$EGAROUCID_COMMIT"
git checkout --quiet FETCH_HEAD

case "$CXX" in
    *clang*) EXTRA= ;;
    *) EXTRA=-mfpmath=both ;;
esac
# the build instructions from https://www.egaroucid.nyanyan.dev/en/console/, with its option
# for a CPU with AVX-512 and the one for AMD's where those apply
if grep -q avx512vl /proc/cpuinfo; then EXTRA="$EXTRA -DHAS_AVX512"; fi
if grep -q AuthenticAMD /proc/cpuinfo; then EXTRA="$EXTRA -DHAS_AMD_PROCESSOR"; fi
"$CXX" -O2 src/Egaroucid_for_Console.cpp -o bin/Egaroucid_for_Console.out \
    -mtune=native -march=native $EXTRA -pthread -std=c++20 >build.log 2>&1 \
    || { cat build.log >&2; echo "setup_egaroucid.sh: build failed (log above)" >&2; exit 1; }

echo "Egaroucid is in $EGAROUCID_DIR (binary bin/Egaroucid_for_Console.out, weights bin/resources)"
