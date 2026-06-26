#!/usr/bin/env bash
# build_cpp_dist.sh — C++/libtorch 포트(gtxc-parse) 배포본 빌더.
#
# Cython dist(Python .so) 와 대비되는 "네이티브 단일 바이너리" 배포본을 만든다.
# gtxc-parse 바이너리 + 검증 도구 + README 를 deploy/cpp_dist 로 스테이징.
# 컴파일러 본체(파서→GGUF→codegen)는 단일 ELF 바이너리 1개 — 소스 .py/.so 비노출.
#
# 사용: bash deploy/build_cpp_dist.sh
# 전제: cpp/build/gtxc-parse 가 빌드돼 있어야 함(bash cpp/build.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${STAGE:-$ROOT/deploy/cpp_dist}"
BIN="$ROOT/cpp/build/gtxc-parse"

[ -x "$BIN" ] || { echo "[cpp_dist] gtxc-parse 없음 — bash cpp/build.sh 먼저"; exit 1; }

rm -rf "$STAGE"
mkdir -p "$STAGE/bin" "$STAGE/tools"
cp "$BIN" "$STAGE/bin/gtxc-parse"
strip --strip-all "$STAGE/bin/gtxc-parse" 2>/dev/null || true
# 검증/도우미 도구(평문 — 보호할 IP 없음): trace export, torch ref, vision.cpp 빌드 verify
for f in export_traced.py _modelkit.py cpp_ref.py verify.sh; do
  [ -f "$ROOT/cpp/tools/$f" ] && cp "$ROOT/cpp/tools/$f" "$STAGE/tools/$f"
done
cp "$ROOT/tools/build_yolo_cpp.sh" "$ROOT/tools/run_yolo_cpp.cpp" "$STAGE/tools/" 2>/dev/null || true

cat > "$STAGE/README.md" <<'EOF'
# GTX Compiler — C++/libtorch 네이티브 배포본

컴파일러 본체가 **단일 ELF 바이너리 `bin/gtxc-parse`** 입니다(파서→GGUF→codegen).
소스(.py/.so) 비노출. PyTorch 모델(traced .pt) → vision.cpp arch C++(.cpp/.h) + GGUF.

## 런타임 의존성
- **libtorch** (C++ 런타임) — `pip install torch` 의 `torch/lib/*.so` 로 충족.
  바이너리는 RPATH 로 torch/lib 를 참조하므로, 같은 환경(또는 LD_LIBRARY_PATH 지정)에서 실행.
- 생성된 .cpp 를 **빌드/실행**하려면 vision.cpp(ggml) 가 필요(Cython 배포본과 동일).

## 사용
```bash
# PyTorch → traced .pt (Python eager, 1회)
python tools/export_traced.py --model resnet18 --out resnet18

# 컴파일: .pt → <dir>/<name>.{cpp,h,gguf}
bin/gtxc-parse resnet18.pt --graph-name resnet18 --compile out/resnet18
bin/gtxc-parse resnet18.pt --graph-name resnet18 --compile out/resnet18 --quantize q8_0

# 수치 verify (vision.cpp 직접 빌드, torch 대비 cosine)
bash tools/verify.sh resnet18 resnet18.pt 224
```
지원: resnet18/34/50, yolo v8~v12, LSTM/GRU(전 변형), q8_0 양자화.
EOF

echo "[cpp_dist] 완료 → $STAGE"
du -sh "$STAGE"
ls -la "$STAGE/bin/gtxc-parse"
