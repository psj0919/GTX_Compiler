#!/usr/bin/env python3
"""build_so.py — g2c 핵심 6개 패키지를 **단일 .so** 로 컴파일 (Nuitka --module).

기존 build_cython.sh 는 파일별 .so(200개) 를 만든다. 이 스크립트는 진입점
deploy/g2c_entry.py 를 nuitka `--module` 로 컴파일하고 우리 6개 패키지를
`--include-package` 로 그 **하나의 .so 안에 임베드**한다. torch/ultralytics 등 외부
의존성은 `--nofollow-import-to` 로 번들에서 제외(런타임 pip 의존성으로 유지) → onefile
처럼 torch 전체(수 GB)를 끌어오지 않는다.

빌드는 **단일 스레드(`--jobs=1`)** — 무거운 trace/컴파일 시 thread 폭주로 머신 hang 방지.

산출: <out>/g2c_entry.cpython-3XX-*.so  →  deploy/g2c.so 로 복사.
실행: PYTHONPATH 에 g2c.so 위치 두고  `python -c "import g2c_entry as g; g.main()"`
      또는 deploy/g2c_run.py 런처 사용.

사용:
    OMP_NUM_THREADS=1 uv run python deploy/build_so.py --out /tmp/g2c_so
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRY = ROOT / "deploy" / "g2c_entry.py"

# 단일 .so 에 임베드할 우리 패키지(소스 보호 대상).
OUR_PKGS = ["shared", "parse", "qproc", "quantization", "utils", "nn"]

# 번들 제외(런타임 pip 의존성으로 유지) — onefile 의 torch 정적번들 문제 회피.
NOFOLLOW = [
    "torch", "torchvision", "ultralytics", "ultralytics_thop", "numpy", "gguf",
    "scipy", "cv2", "matplotlib", "pandas", "seaborn", "PIL", "yaml", "tqdm",
    "psutil", "requests", "sympy", "networkx",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/g2c_so", help="빌드 산출 디렉토리")
    ap.add_argument("--final-dir", default=str(ROOT / "deploy"),
                    help="최종 .so 복사 디렉토리 (import명 g2c_entry 보존 위해 basename 유지)")
    args = ap.parse_args()
    if not ENTRY.exists():
        raise SystemExit(f"진입점 없음: {ENTRY}")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "nuitka", "--module", str(ENTRY),
        f"--output-dir={out}",
        "--assume-yes-for-downloads", "--remove-output", "--lto=no",
        "--jobs=1",                                  # ★ 단일 스레드 빌드
        "--module-parameter=torch-disable-jit=no",   # torch.jit.trace/freeze 유지
    ]
    for p in OUR_PKGS:
        cmd.append(f"--include-package={p}")
    for p in NOFOLLOW:
        cmd.append(f"--nofollow-import-to={p}")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"

    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)

    # 산출 .so 탐색 (g2c_entry.cpython-3XX-*.so)
    sos = sorted(out.glob("g2c_entry*.so"))
    if not sos:
        raise SystemExit(f"빌드 산출 .so 없음 in {out}")
    produced = sos[0]
    # import 명(g2c_entry) 보존 위해 cpython 태그 basename 그대로 복사.
    final = Path(args.final_dir).resolve() / produced.name
    shutil.copy2(produced, final)
    size_mb = final.stat().st_size / (1024 * 1024)
    print(f"\n완료: {final}  ({size_mb:.1f} MB)")
    print(f"실행:  OMP_NUM_THREADS=1 uv run python deploy/g2c_run.py "
          f"--model resnet18 --output out_smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
