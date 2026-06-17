#!/usr/bin/env bash
# 한 모델에 대해 Python IR(ground truth) vs C++ IR 1:1 parity 검증.
#   cpp/run_parity.sh [model] [input-shape]
# 예: cpp/run_parity.sh resnet18 1,3,224,224
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
MODEL="${1:-resnet18}"
# 기본 input shape: rnn→1,8,16 / yolo→1,3,640,640 / 그 외→1,3,224,224
case "$MODEL" in
  lstm*|gru*) DEF=1,8,16 ;;
  *yolo*) DEF=1,3,640,640 ;;
  *) DEF=1,3,224,224 ;;
esac
SHAPE="${2:-$DEF}"
A="$HERE/assets"

[ -x "$HERE/build/gtxc-parse" ] || bash "$HERE/build.sh"

echo "[parity] $MODEL  shape=$SHAPE"
OMP_NUM_THREADS=1 "$PY" "$HERE/tools/export_traced.py" --model "$MODEL" --out "$A/$MODEL" --input-shape "$SHAPE" >/dev/null
OMP_NUM_THREADS=1 "$PY" "$HERE/tools/dump_ir.py"       --model "$MODEL" --out "$A/$MODEL.ir.py.json" --input-shape "$SHAPE" >/dev/null
"$HERE/build/gtxc-parse" "$A/$MODEL.pt" --graph-name "$MODEL" --input-shape "$SHAPE" --out "$A/$MODEL.ir.cpp.json"
"$PY" "$HERE/tools/compare_ir.py" "$A/$MODEL.ir.py.json" "$A/$MODEL.ir.cpp.json"
