#!/usr/bin/env bash
# g2c onefile 빌드 실행기 — 사용자 터미널에서 직접 실행용.
#   (Claude Code 하네스는 10분 제한/백그라운드 정리로 장시간 빌드를 완주 못 시킴.
#    이 스크립트를 일반 터미널에서 돌리면 끝까지 진행된다. ~20-40분 소요.)
#
# 사용:
#   bash deploy/run_build.sh            # 포그라운드(진행 로그 실시간)
#   nohup bash deploy/run_build.sh &    # 백그라운드(터미널 닫아도 유지) → tail -f ~/onefile_build.log
set -euo pipefail

ROOT=/mnt/e/14_NIGHTLY/GTX_Compiler
VENV="$HOME/gtxbuild"
LOG="$HOME/onefile_build.log"

if [ ! -x "$VENV/bin/python" ]; then
  echo "빌드 venv 없음: $VENV — 먼저 환경을 만드세요:" >&2
  echo "  uv venv $VENV --python 3.12" >&2
  echo "  VIRTUAL_ENV=$VENV uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu" >&2
  echo "  VIRTUAL_ENV=$VENV uv pip install nuitka patchelf zstandard ordered-set networkx tqdm numpy gguf" >&2
  exit 1
fi

echo "빌드 시작 → 로그: $LOG"
PATH="$VENV/bin:$PATH" PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 \
  "$VENV/bin/python" -u "$ROOT/deploy/build_onefile.py" \
  --out "$HOME/onefile_out" --final "$ROOT/deploy/g2c" 2>&1 | tee "$LOG"

echo
echo "끝났습니다. 바이너리: $ROOT/deploy/g2c"
ls -lh "$ROOT/deploy/g2c" 2>/dev/null || echo "(g2c 생성 실패 — 로그 확인)"
