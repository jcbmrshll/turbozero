#!/bin/sh
# Builds Melody (https://github.com/HCyrano/Melody), Bruno Causse's Othello engine: alpha-beta
# with ProbCut, and patterns with a factorization-machines layer. GPLv3. Its Dev branch at
# 2026-11-09, what its author develops and plays on GGS, a step past its last release
# (Dev-2026-10-21-10h15) with the same weights: those come from that release's Melody.zip.
# Into ~/.cache/turbozero/othello/engines-2026-11/melody (or $MELODY_DIR).
#
#     examples/othello/setup_melody.sh
#
# Its Makefile builds for x86-64-v3 (AVX2, BMI2), which it needs; it has no AVX-512 code.
# Needs git, make, g++ 13 or later (C++23), curl and unzip.
set -eu

MELODY_COMMIT=02701642cb63d3e0a2934919aa55503c99f05b22
ZIP_URL=https://github.com/HCyrano/Melody/releases/download/Dev-2026-10-21-10h15/Melody.zip
ZIP_SHA256=abc88cd54a50b5e661fc3cd6e58ff3f8db4f9911cd0640430bd723ec82aad7f4
MELODY_DIR=${MELODY_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/turbozero/othello/engines-2026-11/melody}

for tool in git make curl unzip g++ sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || { echo "setup_melody.sh: needs $tool" >&2; exit 1; }
done

mkdir -p "$(dirname "$MELODY_DIR")"
if [ ! -d "$MELODY_DIR/.git" ]; then
    git clone --quiet https://github.com/HCyrano/Melody.git "$MELODY_DIR"
fi
cd "$MELODY_DIR"
git fetch --quiet origin "$MELODY_COMMIT" 2>/dev/null || true
git checkout --quiet "$MELODY_COMMIT"

make -j4 CXX=g++ >build.log 2>&1 \
    || { cat build.log >&2; echo "setup_melody.sh: build failed (log above)" >&2; exit 1; }

# it reads its weights from its working directory; the zip also holds a macOS binary, logs and
# scripts, which aren't needed
if [ ! -f weight_Vb_v12.bin ]; then
    tmp=$(mktemp -d)
    curl -sSL -o "$tmp/Melody.zip" "$ZIP_URL"
    echo "$ZIP_SHA256  $tmp/Melody.zip" | sha256sum -c --quiet
    unzip -q -j -o "$tmp/Melody.zip" 'Melody/weight_*' -d "$MELODY_DIR"
    rm -r "$tmp"
fi

echo "Melody is in $MELODY_DIR (binary build/Melody, weights weight_*)"
