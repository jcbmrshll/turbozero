#!/bin/sh
# Builds Pascal Pons's Connect Four solver (https://github.com/PascalPons/connect4), for
# vs_solver.py and label_positions.py: the source at a pinned commit and its opening book,
# into ~/.cache/turbozero/connect_four/solver (or $SOLVER_DIR). Then labels the set of positions
# train.py scores networks on, if there isn't one yet, with label_positions.py.
#
#     examples/connect_four/setup_solver.sh
#
# Needs git, make, a C++ compiler, curl and uv.
set -eu

EXAMPLE_DIR=$(cd "$(dirname "$0")" && pwd)

SOLVER_COMMIT=d6ba50d8aaf2308c769d9bf2abd42d90f34baf41
BOOK_URL=https://github.com/PascalPons/connect4/releases/download/book/7x6.book
BOOK_SHA256=f346cd449626fb81da93be0958e017ee854e5f74d85b6d38062357f5403aec53
SOLVER_DIR=${SOLVER_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/turbozero/connect_four/solver}
CXX=${CXX:-g++}

for tool in git make curl uv "$CXX"; do
    command -v "$tool" >/dev/null 2>&1 || { echo "setup_solver.sh: needs $tool" >&2; exit 1; }
done

if [ ! -d "$SOLVER_DIR/.git" ]; then
    git clone --quiet https://github.com/PascalPons/connect4.git "$SOLVER_DIR"
fi
cd "$SOLVER_DIR"
git fetch --quiet origin "$SOLVER_COMMIT" 2>/dev/null || true
git checkout --quiet "$SOLVER_COMMIT"

make c4solver CXX="$CXX" >build.log 2>&1 \
    || { cat build.log >&2; echo "setup_solver.sh: build failed (log above)" >&2; exit 1; }

if [ ! -f 7x6.book ]; then
    curl -sSL -o 7x6.book.tmp "$BOOK_URL"
    echo "$BOOK_SHA256  7x6.book.tmp" | sha256sum -c --quiet
    mv 7x6.book.tmp 7x6.book
fi

echo "The solver is in $SOLVER_DIR (binary c4solver, opening book 7x6.book)"

SET=$(dirname "$SOLVER_DIR")/solver-set.npz
if [ ! -f "$SET" ]; then
    # needs no accelerator
    SOLVER_DIR=$SOLVER_DIR JAX_PLATFORMS=cpu uv run --project "$EXAMPLE_DIR/../.." \
        python "$EXAMPLE_DIR/label_positions.py" --out "$SET"
fi
