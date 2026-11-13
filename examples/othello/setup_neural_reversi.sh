#!/bin/sh
# Builds Neural Reversi (https://github.com/natsutteatsuiyone/neural-reversi), an Othello
# engine searching like Edax with a neural network evaluation, GPLv3: its command-line program,
# from its main branch at 2026-11-13, a step past release 6.3.0 with the same weights, which it
# compiles in (from https://github.com/natsutteatsuiyone/neural-reversi-weights, v5.0.0).
# Into ~/.cache/turbozero/othello/engines-2026-11/neural-reversi (or $NEURAL_REVERSI_DIR).
#
#     examples/othello/setup_neural_reversi.sh            # with AVX-512 (ARCH=x86-64-v4)
#     ARCH=x86-64-v3 examples/othello/setup_neural_reversi.sh
#
# Needs git, curl and rustup (it builds with the nightly its rust-toolchain.toml names). The
# build takes about 700 MB of disk under target/.
set -eu

NR_COMMIT=431b7e519e62480824aa67bf75625eec86e5ddf0
WEIGHTS_URL=https://github.com/natsutteatsuiyone/neural-reversi-weights/releases/download/v5.0.0
EVAL=eval-e6bbc4f6.zst
EVAL_SHA256=bfc7e581500bc20d6fbe77d0155f988e6c010d407470bb1cfca4ceb3ffe95c41
EVAL_SM=eval_sm-e6bbc4f6.zst
EVAL_SM_SHA256=af43393d3c0b92132a3f422e7c40f75089fa8aa96725c108dc3e50fcc9dd5041
NEURAL_REVERSI_DIR=${NEURAL_REVERSI_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/turbozero/othello/engines-2026-11/neural-reversi}
ARCH=${ARCH:-x86-64-v4}

for tool in git curl cargo rustup sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || { echo "setup_neural_reversi.sh: needs $tool" >&2; exit 1; }
done

mkdir -p "$(dirname "$NEURAL_REVERSI_DIR")"
if [ ! -d "$NEURAL_REVERSI_DIR/.git" ]; then
    git clone --quiet https://github.com/natsutteatsuiyone/neural-reversi.git "$NEURAL_REVERSI_DIR"
fi
cd "$NEURAL_REVERSI_DIR"
git fetch --quiet origin "$NR_COMMIT" 2>/dev/null || true
git checkout --quiet "$NR_COMMIT"

# the weights, which the build compiles in from the repository's root
for pair in "$EVAL $EVAL_SHA256" "$EVAL_SM $EVAL_SM_SHA256"; do
    set -- $pair
    if [ ! -f "$1" ]; then
        curl -sSL -o "$1.part" "$WEIGHTS_URL/$1"
        echo "$2  $1.part" | sha256sum -c --quiet
        mv "$1.part" "$1"
    fi
done

RUSTFLAGS="-C target-cpu=$ARCH" cargo build --quiet --locked --release -p cli >build.log 2>&1 \
    || { cat build.log >&2; echo "setup_neural_reversi.sh: build failed (log above)" >&2; exit 1; }

echo "Neural Reversi is in $NEURAL_REVERSI_DIR (binary target/release/cli, weights built in)"
