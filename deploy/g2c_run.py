#!/usr/bin/env python3
"""g2c_run.py — 단일 .so(g2c_entry) 런처.

build_so.py 가 만든 deploy/g2c_entry.cpython-3XX-*.so 를 import 해 g2c CLI 를 실행한다.
우리 6개 패키지(shared/parse/qproc/quantization/utils/nn)는 그 .so 안에 임베드돼 있고,
torch/ultralytics 등은 런타임 pip 의존성으로 import 된다.

사용:
    OMP_NUM_THREADS=1 uv run python deploy/g2c_run.py --model resnet18 --output out
    OMP_NUM_THREADS=1 uv run python deploy/g2c_run.py --model "ultralytics.YOLO('yolo11n')" --output out
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# .so 위치를 import 경로에 추가 (g2c_entry.* 단일 .so).
sys.path.insert(0, _HERE)

import g2c_entry  # noqa: E402  (단일 .so — shared/parse/... 임베드됨)

if __name__ == "__main__":
    sys.exit(g2c_entry.main())
