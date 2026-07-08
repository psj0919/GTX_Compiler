#!/usr/bin/env python3
# Copyright (C) Supergate - All Rights Reserved
#
# (테스트/예제 스크립트) 과거 루트 진입점.
# 정식 CLI 는 `g2c` (= shared.compile.pipeline:main, pyproject [project.scripts]).
#
# 기존 인터페이스 유지 (테스트/CI 용):
#   python test/compile_to_c.py                       # 기본 resnet18
#   python test/compile_to_c.py --model yolov8n --output output/yolov8n
#   python test/compile_to_c.py --model "ultralytics.YOLO('yolo11n')" --output output/

import os
import sys

# 프로젝트 루트(= test/ 의 부모)를 Python 경로에 추가.
from g2c.shared.compile.pipeline import main  # noqa: E402

if __name__ == "__main__":
    # 과거 기본값: --model 미지정 시 resnet18.
    if not any(a == "--model" or a.startswith("--model=") for a in sys.argv[1:]):
        sys.argv += ["--model", "resnet18"]
    main()
