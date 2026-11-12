#!/bin/sh
# Builds Edax, a strong open-source Othello engine, for vs_engine.py: Toshihiko Okuhara's
# edax-reversi-AVX (https://github.com/okuhara/edax-reversi-AVX), the fastest build of it we
# measured, without profile-guided optimization, which slowed it and upstream's 4.7, and its
# evaluation weights, into ~/.cache/turbozero/othello/engines-2026-10/edax (or $EDAX_DIR).
#
#     examples/othello/setup_edax.sh            # with AVX-512 (ARCH=x86-64-v4)
#     ARCH=x86-64-v3 examples/othello/setup_edax.sh
#
# Needs git, make, gcc, curl and 7z (p7zip-full on Ubuntu).
set -eu

EDAX_COMMIT=8a2b3ea5f5d4d9028a1310304670724f0cfb322d
EVAL_URL=https://github.com/abulmo/edax-reversi/releases/download/v4.4/eval.7z
EVAL_SHA256=31a9fede7ae9a62bd3a78b2ba457d276d2421b15d3d85b6f8b9a9ba7beddf395
EDAX_DIR=${EDAX_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/turbozero/othello/engines-2026-10/edax}
# ARM (Apple's M chips, Graviton, Axion): Edax's armv8 build, NEON
case "$(uname -m)" in
    aarch64 | arm64) ARCH=${ARCH:-armv8} ;;
    *) ARCH=${ARCH:-x86-64-v4} ;;
esac
# its makefile names the binary for the OS: lEdax-<arch> on Linux, mEdax-<arch> on macOS
case "$(uname -s)" in
    Darwin) OS=osx BINARY=mEdax ;;
    *) OS=linux BINARY=lEdax ;;
esac

for tool in git make curl 7z gcc; do
    command -v "$tool" >/dev/null 2>&1 || { echo "setup_edax.sh: needs $tool" >&2; exit 1; }
done

mkdir -p "$(dirname "$EDAX_DIR")"
if [ ! -d "$EDAX_DIR/.git" ]; then
    git clone --quiet https://github.com/okuhara/edax-reversi-AVX.git "$EDAX_DIR"
fi
cd "$EDAX_DIR"
git fetch --quiet origin "$EDAX_COMMIT" 2>/dev/null || true
git checkout --quiet "$EDAX_COMMIT"

mkdir -p bin
(cd src && make build ARCH="$ARCH" COMP=gcc OS="$OS" >../build.log 2>&1) \
    || { cat build.log >&2; echo "setup_edax.sh: build failed (log above)" >&2; exit 1; }
# the binary is named after the architecture; give it a fixed name
cp "bin/$BINARY-$ARCH" bin/edax

if [ ! -f data/eval.dat ]; then
    curl -sSL -o eval.7z "$EVAL_URL"
    echo "$EVAL_SHA256  eval.7z" | { sha256sum -c --quiet 2>/dev/null || shasum -a 256 -c --quiet; }
    7z x -y eval.7z >/dev/null
    rm eval.7z
fi

# an empty opening book; missing, Edax stalls for minutes making one when it's playing
[ -f data/book.dat ] || printf 'quit\n' | bin/edax -gtp -q >/dev/null

echo "Edax is in $EDAX_DIR (binary bin/edax, weights data/eval.dat)"
