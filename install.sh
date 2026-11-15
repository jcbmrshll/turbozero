#!/bin/sh
# Set up the project environment with uv, using the GPU build of jax when an
# NVIDIA GPU is present. Extra arguments are passed through to `uv sync`.
#
# Re-run this instead of a bare `uv sync`, which would remove the GPU libraries.
set -e
cd "$(dirname "$0")"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU found: installing with the cuda extra"
    uv sync --extra cuda "$@"
else
    echo "No NVIDIA GPU found: installing the CPU build"
    uv sync "$@"
fi
