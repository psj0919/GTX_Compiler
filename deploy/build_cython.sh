#!/usr/bin/env bash
# build_cython.sh — Cython 소스 보호 배포본 빌더 (DISTRIBUTION.md §4).
#
# 우리 핵심 6개 패키지(shared parse qproc quantization utils nn)의 .py 를 Cython 으로
# 네이티브 .so 로 in-place 컴파일하고, .c·원본 .py·__pycache__ 를 제거해 *소스 비노출*
# 스테이징 배포본을 만든다. torch/ultralytics 등 외부 의존성은 번들하지 않는다(pip).
#
# 사용:
#   bash deploy/build_cython.sh                 # → deploy/dist_cython 에 배포본
#   STAGE=/tmp/dist PY=python bash deploy/build_cython.sh
#
# 전제: PY 가 가리키는 인터프리터에 cython>=3.0 / setuptools / gcc 설치.
#   .so 는 Python ABI(cpython-3XX)에 묶임 → 배포 서버와 같은 Python minor 로 빌드할 것.
#   (이 저장소 타겟: Python 3.12 — .python-version / .venv)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${STAGE:-$ROOT/deploy/dist_cython}"
PY="${PY:-$ROOT/.venv/bin/python}"
PKGS="shared parse qproc quantization utils nn"

[ -x "$PY" ] || PY="python"
echo "[build_cython] ROOT=$ROOT"
echo "[build_cython] STAGE=$STAGE"
echo "[build_cython] PY=$PY ($("$PY" --version 2>&1))"

# 0) 스테이징 초기화 + 6개 패키지 복사 (test/tools/resnet.py 등 예제·개발물은 미포함)
rm -rf "$STAGE"
mkdir -p "$STAGE"
for p in $PKGS; do
  cp -r "$ROOT/$p" "$STAGE/$p"
done
# 소스 트리에서 따라온 stale __pycache__ 제거 (bytecode 유출 방지)
find "$STAGE" -type d -name '__pycache__' -prune -exec rm -rf {} +

# 0b) ggml codegen(g2c) 런타임에 불필요한 항목 제외(배포본 슬림화):
#   - HW(GTX/DPU) 백엔드 C 헤더/소스: load_kernels 의 C-빌드 경로가 전부 주석 → 런타임 미사용
#   - DPU/xmodel 전용 모듈: g2c(ggml) 경로 미사용(dpu_pattern_match 만 ggml_fusion 이 사용 → 유지).
#     device_allocator 는 target_quant_info(DPU 양자화)에서 lazy import 라 g2c 엔 안 걸림.
rm -rf "$STAGE/nn/include" "$STAGE/nn/src"
rm -f "$STAGE/shared/inspector/dpu_pattern_handle.py" \
      "$STAGE/shared/inspector/dpu_pattern_transform.py" \
      "$STAGE/shared/inspector/device_allocator.py"
# 배포본에 문서 잔재 미포함
rm -f "$STAGE/shared/compile/TODO.md"
find "$STAGE" -maxdepth 2 -name 'README.md' -delete

# 1) cythonize → build_ext --inplace (제외목록은 setup_cython.py 가 처리)
( cd "$STAGE" && PKGS="$PKGS" "$PY" "$ROOT/deploy/setup_cython.py" )

# 2) .so 심볼 제거
find "$STAGE" -name '*.so' -exec strip --strip-all {} +

# 3) 중간 산출 .c 전부 삭제
find "$STAGE" -name '*.c' -delete

# 4) 컴파일된 모듈의 원본 .py 제거 (.so 가 있는 파일만 — 제외목록 .py 는 .so 가 없어 보존)
removed=0
while IFS= read -r so; do
  mod="${so%.so}"; mod="${mod%%.cpython*}"
  if [ -f "${mod}.py" ]; then rm -f "${mod}.py"; removed=$((removed+1)); fi
done < <(find "$STAGE" -name '*.so')

# 5) setuptools 중간물 + 모든 __pycache__ 제거 (컴파일 모듈의 .pyc 유출 방지)
#    .cython_skipped.txt(정적 컴파일 불가로 평문 유지된 파일 목록)는 루트로 옮겨 보고.
#    이번 빌드에서 skip 이 없으면 이전 잔재 보고를 지운다.
SKIPPED="$STAGE/.cython_skipped.txt"
rm -f "$ROOT/deploy/.cython_skipped.txt"
[ -f "$SKIPPED" ] && cp "$SKIPPED" "$ROOT/deploy/.cython_skipped.txt"
rm -rf "$STAGE"/build "$STAGE"/*.egg-info "$SKIPPED"
find "$STAGE" -type d -name '__pycache__' -prune -exec rm -rf {} +

# 6) 평문 진입물: g2c.py 런처 / requirements.txt / README.md
cat > "$STAGE/g2c.py" <<'EOF'
"""g2c 진입점 런처 (console_scripts `g2c = shared.compile.pipeline:main` 대체).

    python g2c.py --model resnet18 --output output/resnet18
    python g2c.py --model "ultralytics.YOLO('yolo11n')" --output output/yolo11n

인자 파싱·main() 호출만 하는 얇은 래퍼 — 보호할 IP 없음(평문 유지).
"""
import sys

from shared.compile.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
EOF

cat > "$STAGE/requirements.txt" <<'EOF'
# g2c 배포본 런타임 의존성 (우리 코드는 *.so 로 동봉, 아래는 외부 오픈소스 패키지)
#
# torch/torchvision 은 CUDA 12.4 휠 — 전용 인덱스에서 설치:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CPU 전용이면 https://download.pytorch.org/whl/cpu 사용.
torch>=2.5.1
torchvision>=0.20.1
ultralytics            # --model yolo* 분기에서만 필요(선택)
networkx
tqdm
numpy
gguf                   # GGUF writer (pipeline.generate_gguf)
EOF

cat > "$STAGE/README.md" <<'EOF'
# GTX Compiler — 배포본 (Cython 소스 보호, Linux)

핵심 컴파일러 로직은 `*.so` 네이티브 바이너리로 동봉되어 있습니다(소스 .py 없음).
PyTorch 등 외부 의존성만 별도 설치하면 됩니다.

## 설치
```bash
python -m venv .venv && source .venv/bin/activate     # Python 은 빌드와 동일 minor (3.12)
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#   (CPU 전용이면 .../whl/cpu)
```

## 사용
```bash
# CLI
python g2c.py --model resnet18 --output output/resnet18
python g2c.py --model "ultralytics.YOLO('yolo11n')" --output output/yolo11n
# 라이브러리
python -c "from shared.compile.pipeline import compile_model"
```
생성물: `output/<Model>.{cpp,h,gguf,py}`

## 주의
- `.so` 는 빌드 시점 Python minor(3.12)에서만 로드됩니다.
- glibc 호환: 빌드한 배포판과 같거나 더 새 버전의 Linux 에서 동작합니다.
- `nn/modules/rnn_builder/*.py` 는 jit.script 용으로 의도적으로 평문 유지됩니다.
EOF

echo
echo "[build_cython] 완료 → $STAGE"
echo "[build_cython] 제거된 원본 .py: $removed"
nso=$(find "$STAGE" -name '*.so' | wc -l)
npy=$(find "$STAGE" -name '*.py' | wc -l)
nc=$(find "$STAGE" -name '*.c' | wc -l)
echo "[build_cython] .so=$nso  남은 .py(런처+__init__+rnn_builder+컴파일불가)=$npy  .c=$nc"
if [ -f "$ROOT/deploy/.cython_skipped.txt" ]; then
  echo "[build_cython] 정적 컴파일 불가로 평문 유지된 파일:"
  sed 's/^/    /' "$ROOT/deploy/.cython_skipped.txt"
fi
