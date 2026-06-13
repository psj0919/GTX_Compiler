#!/usr/bin/env bash
# 여러 모델에 탭 디버거 일괄: regen(taps) → build → C++ tap dump → eager ref → 노드별 cmp.
# 사용: bash tools/tap_all_models.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# model_expr | out_dir | input_bin | size
MODELS=(
  "ultralytics.YOLO('yolov8n')|output/yolov8n|tools/v8n_input.bin|640"
  "ultralytics.YOLO('yolo11n')|output/yolo11n|tools/input_cwhn.bin|640"
  "ultralytics.YOLO('yolov10n')|output/yolov10n|tools/v10n_input.bin|640"
  "resnet18|output/resnet18|tools/r18_input.bin|224"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r EXPR DIR INP SZ <<< "$entry"
  echo "================================================================"
  echo "### $DIR  (size=$SZ)"
  echo "================================================================"
  GTX_DEBUG_TAPS=all OMP_NUM_THREADS=1 uv run g2c --model "$EXPR" --output "$DIR" \
    2>&1 | grep -iE "완료|error|Traceback|NMS-free" | tail -2
  bash tools/build_yolo_cpp.sh "$DIR" 2>&1 | tail -1
  CPP="$DIR/taps_cpp"; REF="$DIR/taps_ref"
  rm -rf "$CPP" "$REF"; mkdir -p "$CPP" "$REF"
  ./"$DIR"/run_yolo_cpp "$DIR"/*.gguf "$INP" "$DIR/tap_out.bin" "$SZ" none "$CPP" \
    2>&1 | grep -iE "dumped" || true
  OMP_NUM_THREADS=1 uv run python tools/yolo_tap_ref.py "$DIR" "$INP" "$REF" \
    2>&1 | grep -ivE "it/s|WARN|NOTE|Downloading" | tail -1
  uv run python tools/yolo_tap_cmp.py "$CPP" "$REF" 0.99 2>&1 | grep -E "common taps|FIRST DIVERGENCE|ALL "
done
echo "================================================================"
echo "DONE"
