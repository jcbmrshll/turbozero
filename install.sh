#!/bin/sh
# Sets up the environment with uv, with the GPU build of jax when there is an
# NVIDIA GPU. Re-run this instead of a bare `uv sync`, which removes the GPU
# build. Extra arguments go to `uv sync`.
set -e
cd "$(dirname "$0")"
if nvidia-smi >/dev/null 2>&1; then
    uv sync --extra cuda "$@"
else
    uv sync "$@"
fi
