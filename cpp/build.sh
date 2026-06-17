#!/usr/bin/env bash
# libtorch(=pip .venv torch) 경로를 자동 주입하여 cpp/ 빌드.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"

TORCH_PREFIX="$("$PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
echo "[build] torch cmake prefix: $TORCH_PREFIX"

cmake -S "$HERE" -B "$HERE/build" \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-RelWithDebInfo}"
cmake --build "$HERE/build" -j"$(nproc)"
echo "[build] done → $HERE/build/gtxc-parse"
