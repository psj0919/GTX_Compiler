#!/usr/bin/env bash
# Build both vision.cpp runners into the converted model directory.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/output/laya-multilingual}"
bash "$ROOT/tools/build_laya_encoder.sh" "$OUT"
bash "$ROOT/tools/build_laya_head.sh" "$OUT"
