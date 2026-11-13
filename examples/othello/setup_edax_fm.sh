#!/bin/sh
# Builds Edax-FM (https://github.com/Yoka0014/edax-fm-reversi): Edax 4.6 with a factorization-
# machines layer added to its evaluation, GPLv3, what its author plays on GGS as "Kalmia". Its
# master branch at 2026-07-24, with the weights of its release 4.6 (from the Windows zip, the
# only place they're published), into ~/.cache/turbozero/othello/engines-2026-11/edax-fm (or
# $EDAX_FM_DIR). It's driven as Edax is (`engines.EdaxFM`).
#
#     examples/othello/setup_edax_fm.sh            # with AVX-512 (ARCH=x86-64-v4)
#     ARCH=x86-64-v3 examples/othello/setup_edax_fm.sh
#
# Needs git, make, gcc, curl and unzip.
set -eu

EDAX_FM_COMMIT=ae1f4b5d7e2b64c6c63a9a1f7eb4760f63aad240
ZIP_URL=https://github.com/Yoka0014/edax-fm-reversi/releases/download/v4.6/edax-fm-4.6-MS-windows-x86.zip
ZIP_SHA256=988c5963de2d2360a0bbfc38a80be5f8450a9fe96fb4a36c2b92d97965fdd289
EDAX_FM_DIR=${EDAX_FM_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/turbozero/othello/engines-2026-11/edax-fm}
ARCH=${ARCH:-x86-64-v4}

for tool in git make curl unzip gcc sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || { echo "setup_edax_fm.sh: needs $tool" >&2; exit 1; }
done

mkdir -p "$(dirname "$EDAX_FM_DIR")"
if [ ! -d "$EDAX_FM_DIR/.git" ]; then
    git clone --quiet https://github.com/Yoka0014/edax-fm-reversi.git "$EDAX_FM_DIR"
fi
cd "$EDAX_FM_DIR"
git fetch --quiet origin "$EDAX_FM_COMMIT" 2>/dev/null || true
git checkout --quiet "$EDAX_FM_COMMIT"

mkdir -p bin
(cd src && make build ARCH="$ARCH" CC=gcc OS=linux >../build.log 2>&1) \
    || { cat build.log >&2; echo "setup_edax_fm.sh: build failed (log above)" >&2; exit 1; }
# the binary is named after the architecture; give it Edax's fixed name
cp "bin/lEdax-$ARCH" bin/edax

# only its weights from the zip, which also holds Windows binaries
if [ ! -f data/eval.dat ]; then
    tmp=$(mktemp -d)
    curl -sSL -o "$tmp/edax-fm.zip" "$ZIP_URL"
    echo "$ZIP_SHA256  $tmp/edax-fm.zip" | sha256sum -c --quiet
    unzip -q -o "$tmp/edax-fm.zip" data/eval.dat -d "$EDAX_FM_DIR"
    rm -r "$tmp"
fi

# an empty opening book; missing, Edax stalls for minutes making one when it's playing
[ -f data/book.dat ] || printf 'quit\n' | bin/edax -gtp -q >/dev/null

echo "Edax-FM is in $EDAX_FM_DIR (binary bin/edax, weights data/eval.dat)"
