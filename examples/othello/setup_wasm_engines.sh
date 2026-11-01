#!/bin/sh
# Builds Edax and Egaroucid to WebAssembly for the league's game page, from the commits
# setup_edax.sh and setup_egaroucid.sh pin, into a directory the dashboard serves. Emscripten goes
# in ~/.cache/turbozero/wasm-toolchain. Run it through the league's `build-browser-engines`.
#
#     examples/othello/setup_wasm_engines.sh OUT
#
# Needs git, curl, 7z and about 2 GB for the toolchain.
set -eu

EMSDK_VERSION=4.0.23
EDAX_COMMIT=8a2b3ea5f5d4d9028a1310304670724f0cfb322d
EGAROUCID_COMMIT=3b1c6ad380209162c2ad07aefe5c1fc6fd8072cb
# bits of Egaroucid's hash table: 2^23 entries, about 270 MB, where native's takes about 700 MB
EGAROUCID_HASH_LEVEL=23
EVAL_URL=https://github.com/abulmo/edax-reversi/releases/download/v4.4/eval.7z
EVAL_SHA256=31a9fede7ae9a62bd3a78b2ba457d276d2421b15d3d85b6f8b9a9ba7beddf395
CACHE=${XDG_CACHE_HOME:-$HOME/.cache}/turbozero
OUT=$1
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=$CACHE/wasm-toolchain/build

for tool in git curl 7z; do
    command -v "$tool" >/dev/null 2>&1 || { echo "setup_wasm_engines.sh: needs $tool" >&2; exit 1; }
done

EMSDK=$CACHE/wasm-toolchain/emsdk
if [ ! -d "$EMSDK" ]; then
    git clone --quiet https://github.com/emscripten-core/emsdk.git "$EMSDK"
fi
"$EMSDK/emsdk" install "$EMSDK_VERSION" >/dev/null
"$EMSDK/emsdk" activate "$EMSDK_VERSION" >/dev/null
export EM_CONFIG="$EMSDK/.emscripten"
export PATH="$EMSDK/upstream/emscripten:$PATH"

mkdir -p "$OUT" "$BUILD"
rm -f "$OUT/complete"
if [ ! -d "$BUILD/edax/.git" ]; then
    git clone --quiet https://github.com/okuhara/edax-reversi-AVX.git "$BUILD/edax"
fi
git -C "$BUILD/edax" checkout --quiet "$EDAX_COMMIT"

# wasm32 has 64-bit integer instructions, and clz, ctz and popcnt, which suit Edax's 64-bit bitscan
# path; the rest of its generic paths don't use SSE or Neon
emcc "$HERE/wasm/edax_wasm.c" -o "$OUT/edax.mjs" -I"$BUILD/edax/src" \
    -std=gnu11 -O3 -ffast-math -flto -msimd128 -DNDEBUG -D_GNU_SOURCE=1 -DUNICODE \
    -DHAS_CPU_64=1 -DPOPCOUNT \
    -DMOVE_GENERATOR=MOVE_GENERATOR_BITSCAN -DCOUNT_LAST_FLIP=COUNT_LAST_FLIP_BITSCAN \
    -w -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=worker,node \
    -sINITIAL_MEMORY=256MB -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=2GB -sSTACK_SIZE=16MB \
    -sFORCE_FILESYSTEM=1 -sEXPORTED_RUNTIME_METHODS=FS \
    -sEXPORTED_FUNCTIONS=_engine_init,_engine_search

if [ ! -f "$OUT/edax-eval.dat" ]; then
    mkdir -p "$BUILD/eval"
    curl -sSL -o "$BUILD/eval/eval.7z" "$EVAL_URL"
    echo "$EVAL_SHA256  $BUILD/eval/eval.7z" | sha256sum -c --quiet
    7z x -y "-o$BUILD/eval" "$BUILD/eval/eval.7z" >/dev/null
    cp "$BUILD/eval/data/eval.dat" "$OUT/edax-eval.dat.partial"
    mv "$OUT/edax-eval.dat.partial" "$OUT/edax-eval.dat"
fi

if [ ! -d "$BUILD/egaroucid/.git" ]; then
    git init --quiet "$BUILD/egaroucid"
    git -C "$BUILD/egaroucid" remote add origin https://github.com/Nyanyan/Egaroucid.git
fi
git -C "$BUILD/egaroucid" fetch --quiet --depth 1 origin "$EGAROUCID_COMMIT"
git -C "$BUILD/egaroucid" checkout --quiet --force FETCH_HEAD
git -C "$BUILD/egaroucid" apply "$HERE/wasm/egaroucid.patch"
em++ "$HERE/wasm/egaroucid_wasm.cpp" -o "$OUT/egaroucid.mjs" -I"$BUILD/egaroucid/src" \
    -std=c++20 -O3 -flto -msimd128 -DNDEBUG -DHAS_NO_AVX2 \
    -DEGAROUCID_HASH_LEVEL="$EGAROUCID_HASH_LEVEL" \
    -w -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=worker,node \
    -sINITIAL_MEMORY=768MB -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=2GB -sSTACK_SIZE=16MB \
    -sFORCE_FILESYSTEM=1 -sEXPORTED_RUNTIME_METHODS=FS \
    -sEXPORTED_FUNCTIONS=_engine_init,_engine_search
for file in eval.egev2 eval_move_ordering_end.egev hash/hash25.eghs; do
    cp "$BUILD/egaroucid/bin/resources/$file" "$OUT/egaroucid-$(basename "$file")"
done

# what says the build is whole
touch "$OUT/complete"
echo "WebAssembly engines are in $OUT"
