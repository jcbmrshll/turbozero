#!/bin/sh
# Builds Edax (https://github.com/abulmo/edax-reversi), a strong open-source Othello
# engine, for vs_edax.py: the source at a pinned commit and its evaluation weights,
# into ~/.cache/turbozero/othello/edax (or $EDAX_DIR).
#
#     examples/othello/setup_edax.sh            # for this CPU (ARCH=native)
#     ARCH=x86-64-v3 examples/othello/setup_edax.sh
#
# Needs git, make, a C compiler (gcc or clang), curl and 7z (p7zip-full on Ubuntu).
set -eu

EDAX_COMMIT=14f048c05ddfa385b6bf954a9c2905bbe677e9d3
EVAL_URL=https://github.com/abulmo/edax-reversi/releases/download/v4.4/eval.7z
EVAL_SHA256=31a9fede7ae9a62bd3a78b2ba457d276d2421b15d3d85b6f8b9a9ba7beddf395
EDAX_DIR=${EDAX_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/turbozero/othello/edax}
ARCH=${ARCH:-native}
CC=${CC:-$(command -v clang >/dev/null 2>&1 && echo clang || echo gcc)}

for tool in git make curl 7z "$CC"; do
    command -v "$tool" >/dev/null 2>&1 || { echo "setup_edax.sh: needs $tool" >&2; exit 1; }
done

if [ ! -d "$EDAX_DIR/.git" ]; then
    git clone --quiet https://github.com/abulmo/edax-reversi.git "$EDAX_DIR"
fi
cd "$EDAX_DIR"
git fetch --quiet origin "$EDAX_COMMIT" 2>/dev/null || true
git checkout --quiet "$EDAX_COMMIT"

mkdir -p bin
(cd src && make build ARCH="$ARCH" CC="$CC" OS=linux >../build.log 2>&1) \
    || { cat build.log >&2; echo "setup_edax.sh: build failed (log above)" >&2; exit 1; }
# the binary is named after the architecture; give it a fixed name
for f in bin/lEdax-*; do [ -f "$f" ] && cp "$f" bin/edax; done

if [ ! -f data/eval.dat ]; then
    curl -sSL -o eval.7z "$EVAL_URL"
    echo "$EVAL_SHA256  eval.7z" | sha256sum -c --quiet
    7z x -y eval.7z >/dev/null
    rm eval.7z
fi

echo "Edax is in $EDAX_DIR (binary bin/edax, weights data/eval.dat)"
