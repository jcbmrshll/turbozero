#!/bin/sh
# Builds MoHex 2.0 (Benzene, https://github.com/alcalyn/benzene-vanilla-cmake), a strong
# open-source Hex engine, for vs_engine.py: the source at a pinned commit, with Boost and
# Berkeley DB built beside it, into ~/.cache/turbozero/hex/mohex (or $MOHEX_DIR). Needs no root.
#
#     examples/hex/setup_mohex.sh
#     JOBS=8 examples/hex/setup_mohex.sh
#
# Needs git, make, a C++ compiler, curl, tar with xz and bzip2, and uv (for cmake).
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
BENZENE_COMMIT=95614769bafc9850a3cc54974660bb1795db6086
BOOST_URL=https://archives.boost.io/release/1.74.0/source/boost_1_74_0.tar.bz2
BOOST_SHA256=83bfc1507731a0906e387fc28b7ef5417d591429e51e788417fe9ff025e116b1
DB_URL=http://deb.debian.org/debian/pool/main/d/db5.3/db5.3_5.3.28+dfsg2.orig.tar.xz
DB_SHA256=ad41b507415dec8316e828b2230242af2251d2c86eefa3c7aa9ef47c5239ef33
MOHEX_DIR=${MOHEX_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/turbozero/hex/mohex}
JOBS=${JOBS:-4}

for tool in git make g++ curl tar uv; do
    command -v "$tool" >/dev/null 2>&1 || { echo "setup_mohex.sh: needs $tool" >&2; exit 1; }
done

mkdir -p "$MOHEX_DIR/deps/src"
DEPS=$MOHEX_DIR/deps

if [ ! -f "$DEPS/lib/libboost_thread.a" ]; then
    cd "$DEPS/src"
    curl -sSL -o boost.tar.bz2 "$BOOST_URL"
    echo "$BOOST_SHA256  boost.tar.bz2" | sha256sum -c --quiet
    tar xjf boost.tar.bz2
    cd boost_1_74_0
    ./bootstrap.sh --prefix="$DEPS" \
        --with-libraries=thread,system,filesystem,program_options,date_time,chrono,atomic \
        >boost-bootstrap.log 2>&1
    ./b2 -j"$JOBS" link=static threading=multi variant=release install >boost-build.log 2>&1 \
        || { tail -30 boost-build.log >&2; echo "setup_mohex.sh: Boost failed" >&2; exit 1; }
fi

if [ ! -f "$DEPS/lib/libdb.a" ]; then
    cd "$DEPS/src"
    curl -sSL -o db.tar.xz "$DB_URL"
    [ -z "$DB_SHA256" ] || echo "$DB_SHA256  db.tar.xz" | sha256sum -c --quiet
    tar xJf db.tar.xz
    cd db-5.3.28*/build_unix
    ../dist/configure --prefix="$DEPS" --disable-shared --enable-static >configure.log 2>&1
    make -j"$JOBS" library_build >make.log 2>&1 \
        || { tail -30 make.log >&2; echo "setup_mohex.sh: Berkeley DB failed" >&2; exit 1; }
    make install_include install_lib >/dev/null 2>&1
fi

if [ ! -d "$MOHEX_DIR/benzene/.git" ]; then
    git clone --quiet https://github.com/alcalyn/benzene-vanilla-cmake.git "$MOHEX_DIR/benzene"
fi
cd "$MOHEX_DIR/benzene"
git fetch --quiet origin "$BENZENE_COMMIT" 2>/dev/null || true
git checkout --quiet --force "$BENZENE_COMMIT"
# a function that can run off its end without returning, which a compiler newer than the code
# may turn into a crash
git apply "$HERE/benzene.patch"

mkdir -p build
cd build
# Benzene's CMakeLists.txt sets CMAKE_CXX_FLAGS itself, so the dependencies' paths go in the
# environment
export CPATH="$DEPS/include" LIBRARY_PATH="$DEPS/lib"
uv tool run --from 'cmake<4' cmake .. -DBOOST_ROOT="$DEPS" -DBoost_NO_SYSTEM_PATHS=ON \
    -DBoost_USE_STATIC_LIBS=ON -DCMAKE_EXE_LINKER_FLAGS=-pthread >cmake.log 2>&1 \
    || { cat cmake.log >&2; echo "setup_mohex.sh: cmake failed (log above)" >&2; exit 1; }
make -j"$JOBS" mohex >make.log 2>&1 \
    || { grep -n "error" make.log | head -30 >&2; echo "setup_mohex.sh: MoHex failed (make.log)" >&2; exit 1; }

echo "MoHex is in $MOHEX_DIR/benzene/build/src/mohex/mohex"
