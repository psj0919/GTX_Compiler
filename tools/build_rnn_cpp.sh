#!/usr/bin/env bash
# 생성 RNN(LSTM/GRU) arch .cpp + run_rnn_cpp 하네스를 vision.cpp(sam3) 빌드로 컴파일.
# usage: build_rnn_cpp.sh <gen_dir> [arch_name]
#   arch_name 미지정 시 gen_dir 의 단일 *.cpp(하네스 제외) basename 으로 자동 검출.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
GEN="${1:?usage: build_rnn_cpp.sh <gen_dir> [arch_name]}"
ARCH="${2:-}"
if [ -z "$ARCH" ]; then
  ARCH="$(basename "$(ls "$GEN"/*.cpp | grep -vE 'run_yolo_cpp|run_rnn_cpp' | head -1)" .cpp)"
fi
echo "arch=$ARCH gen=$GEN"

V=vision.cpp
INC="$GEN/inc"
mkdir -p "$INC/visp/arch"
cp "$GEN/$ARCH.h" "$INC/visp/arch/$ARCH.h"
FMT_INC="$V/build/_deps/fmt-src/include"

# sam3: ggml 헤더는 depend/llama/ggml/include (main 의 depend/ggml 에서 이동).
g++ -std=c++20 -O2 -DVISP_FMT_LIB \
  -DARCH="$ARCH" -DVISP_ARCH_HEADER="\"visp/arch/$ARCH.h\"" \
  -I"$INC" -I"$V/include" -I"$V/src" \
  -I"$V/depend/llama/ggml/include" -I"$FMT_INC" \
  tools/run_rnn_cpp.cpp "$GEN/$ARCH.cpp" \
  -L"$V/build/lib" -lvisioncpp -lggml -lggml-base -lggml-cpu \
  -Wl,-rpath,"$V/build/lib" \
  -o "$GEN/run_rnn_cpp"
echo "built: $GEN/run_rnn_cpp (arch=$ARCH)"
