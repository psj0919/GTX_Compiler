#!/usr/bin/env bash
# C++ 생성물(cpp/out/<model>)을 vision.cpp 직접 빌드로 컴파일·실행하고 torch(.pt) 참조와
# cosine 비교. usage: verify.sh <model> <pt> [size]
#   예: verify.sh resnet18 cpp/assets/resnet18.pt 224
#       verify.sh yolo11n  cpp/assets/yolo11n.pt  640
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
M="${1:?usage: verify.sh <model> <pt> [size]}"
PT="${2:?need .pt}"
SZ="${3:-$([[ $M == *yolo* ]] && echo 640 || echo 224)}"
GEN="cpp/out/$M"
PY="${PY:-$ROOT/.venv/bin/python}"

echo "== [1/4] compile: $PT → $GEN =="
OMP_NUM_THREADS=1 cpp/build/gtxc-parse "$PT" --graph-name "$M" \
  --input-shape "1,3,$SZ,$SZ" --compile "$GEN" 2>&1 | tail -2

echo "== [2/4] torch ref (same .pt) =="
OMP_NUM_THREADS=1 "$PY" cpp/tools/cpp_ref.py "$PT" "$GEN/v" "$SZ" 2>&1 | tail -1

echo "== [3/4] build against vision.cpp =="
bash tools/build_yolo_cpp.sh "$GEN" "$M" 2>&1 | grep -E "error:|built:" | head

echo "== [4/4] run + compare =="
"$GEN/run_yolo_cpp" "$GEN/$M.gguf" "$GEN/v_input.bin" "$GEN/v_out.bin" "$SZ" none 2>&1 | tail -2
"$PY" tools/yolo_cpp_cmp.py "$GEN/v_out.bin" "$GEN/v_ref.bin"
