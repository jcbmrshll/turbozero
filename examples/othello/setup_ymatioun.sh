#!/bin/sh
# Builds ymatioun (https://github.com/ymatioun/reversi), Youri Matiounine's Othello engine, MIT:
# alpha-beta with multi-ProbCut and pattern weights, under that name on GGS. Its main branch at
# 2025-11-29, into ~/.cache/turbozero/othello/engines-2026-11/ymatioun (or $YMATIOUN_DIR).
#
#     examples/othello/setup_ymatioun.sh                  # for 1, 2, 4 and 8 threads
#     THREADS="1 6" ARCH=x86-64-v3 examples/othello/setup_ymatioun.sh
#
# Its threads and hash table are constants its author edits for each setting, so this builds
# a binary for each number of threads, bin/ymatioun-t<N>, with a hash table of 2^24 entries
# (128 MB) a thread, as the author's own comments set it for 1 and 16 threads locally; it keeps
# two (the second for the other game of a synchro pair). Its weights' paths are its author's
# home directory; the build reads them from data/ instead, under its working directory. And
# it rounds its large arrays' sizes up to whole pages, without which it crashes at start. The
# author's project builds with -march=native; this builds for a standard ISA level.
# Needs git and g++.
set -eu

YMATIOUN_COMMIT=39d4c35344e77baf29fa0850242e0f5099826173
YMATIOUN_DIR=${YMATIOUN_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/turbozero/othello/engines-2026-11/ymatioun}
ARCH=${ARCH:-x86-64-v4}
THREADS=${THREADS:-1 2 4 8}

for tool in git g++; do
    command -v "$tool" >/dev/null 2>&1 || { echo "setup_ymatioun.sh: needs $tool" >&2; exit 1; }
done

mkdir -p "$(dirname "$YMATIOUN_DIR")"
if [ ! -d "$YMATIOUN_DIR/.git" ]; then
    git clone --quiet https://github.com/ymatioun/reversi.git "$YMATIOUN_DIR"
fi
cd "$YMATIOUN_DIR"
git fetch --quiet origin "$YMATIOUN_COMMIT" 2>/dev/null || true
git checkout --quiet --force "$YMATIOUN_COMMIT"

# the patches: weights from data/, and threads and hash size from the command line
sed -i -e 's#//home//youri//reversi//train//#data//#g' \
    -e 's#//home//youri//reversi//temp//sigma_v3//#data//sigma_v3//#g' src/ginit.cpp
sed -i 's|^#define helper_count 15\(.*\)$|#ifndef helper_count\n#define helper_count 15\1\n#endif|' src/main.cpp
sed -i 's|^#define HASH_SIZE (16 \* 16 \* 8 \* UINT64(1024 \* 1024))\(.*\)$|#ifndef HASH_SIZE\n#define HASH_SIZE (16 * 16 * 8 * UINT64(1024 * 1024))\1\n#endif|' src/hash.h
# and its large arrays' sizes rounded up to whole 2 MB pages, as aligned_alloc needs (this glibc
# returns no memory for another size, and it crashed at start)
sed -i 's|aligned_alloc(2 \* 1024 \* 1024, sz)|aligned_alloc(2 * 1024 * 1024, (sz + (2 << 20) - 1) / (2 << 20) * (2 << 20))|' \
    src/ginit.cpp src/search.cpp
grep -q '^#ifndef helper_count' src/main.cpp && grep -q '^#ifndef HASH_SIZE' src/hash.h \
    && [ "$(cat src/ginit.cpp src/search.cpp | grep -c '(2 << 20) \* (2 << 20)')" = 4 ] \
    || { echo "setup_ymatioun.sh: the source changed; its patches don't apply" >&2; exit 1; }

SOURCES="src/ODK/GGSMessage.cpp src/ODK/GGSObjects.cpp src/ODK/GGSmain.cpp src/ODK/GetMove.cpp
    src/ODK/ODKStream.cpp src/ODK/OsMessage.cpp src/ODK/OsObjects.cpp src/ODK/ggsstream.cpp
    src/ODK/sockbuf.cpp src/bit_mob.cpp src/eval.cpp src/flip_bmi2.cpp src/ginit.cpp
    src/globals.cpp src/hash.cpp src/main.cpp src/search.cpp src/sigmas.cpp"
mkdir -p bin
: >build.log
for n in $THREADS; do
    # the options of the author's Code::Blocks project (src/project/rev.cbp), but -march
    # shellcheck disable=SC2086
    g++ -O3 -flto -m64 -march="$ARCH" -DNDEBUG -fpermissive -w -pthread \
        -Dhelper_count=$((n - 1)) "-DHASH_SIZE=($n * 16 * UINT64(1024 * 1024))" \
        $SOURCES -o "bin/ymatioun-t$n" >>build.log 2>&1 \
        || { cat build.log >&2; echo "setup_ymatioun.sh: build failed (log above)" >&2; exit 1; }
done

echo "ymatioun is in $YMATIOUN_DIR (binaries bin/ymatioun-t<threads>, weights data/eval_v18)"
