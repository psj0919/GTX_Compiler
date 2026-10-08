#!/usr/bin/env bash
# Build the mmBERT encoder runner against vision.cpp's selected GGML backend.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
V="${VISP_ROOT:-$ROOT/vision.cpp}"
if [ -n "${VISP_GGML_MAX_NAME:-}" ]; then
  GGML_NAME_SIZE="$VISP_GGML_MAX_NAME"
elif [ -n "${VISP_ROOT:-}" ]; then
  GGML_NAME_SIZE=64
else
  GGML_NAME_SIZE=128
fi
OUT="${1:-output/laya-multilingual}"
mkdir -p "$OUT"

# Use the unified executable instead of linking another GGML implementation.
if [ -n "${GTX_UNIFIED_BIN:-}" ]; then
  test -x "$GTX_UNIFIED_BIN/run_laya_encoder"
  ln -sfn "$(realpath "$GTX_UNIFIED_BIN/run_laya_encoder")" "$OUT/run_laya_encoder"
  echo "linked: $OUT/run_laya_encoder"
  exit 0
fi

if [ ! -f "$V/build/lib/libvisioncpp.so" ]; then
  cmake -S "$V" -B "$V/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVISP_FMT_LIB=ON
  cmake --build "$V/build"
fi

g++ -std=c++20 -O2 -DVISP_FMT_LIB -DGGML_MAX_NAME="$GGML_NAME_SIZE" \
  -I"$V/include" -I"$V/src" -I"$V/depend/llama/ggml/include" \
  -I"$V/build/_deps/fmt-src/include" -Itools \
  tools/run_laya_encoder.cpp tools/laya_encoder.cpp \
  -L"$V/build/lib" -lvisioncpp -lggml -lggml-base -lggml-cpu \
  -Wl,-rpath,"$V/build/lib" \
  -o "$OUT/run_laya_encoder"

echo "built: $OUT/run_laya_encoder"
