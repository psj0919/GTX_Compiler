#!/usr/bin/env python3
"""build_onefile.py — g2c CLI 를 *단일 실행 바이너리* 로 빌드 (Nuitka standalone onefile).

파이썬 인터프리터 + 우리 소스(.so 컴파일) + torch/ultralytics 등 모든 의존성을 하나의
실행파일에 담는다. 받는 쪽은 Python/pip 설치 불필요, 버전 충돌 없음:

    ./g2c --model resnet18 --output output/resnet18
    ./g2c --model "ultralytics.YOLO('yolo11n')" --output output/yolo11n

⚠ Nuitka 크로스컴파일 불가 — 배포 타겟과 같은 OS(여기선 Linux)에서 빌드.
⚠ 빌드 venv 에 설치된 torch 가 그대로 들어간다 → CPU 휠로 설치하면 CPU 바이너리.
⚠ standalone(Linux)은 patchelf 필요: `pip install patchelf` 후 PATH 에 venv/bin.

진입점은 영구 파일 deploy/g2c_entry.py 를 사용한다(임시 생성/삭제 race 회피).
`shared` 등 우리 패키지는 PYTHONPATH=<ROOT> 로 nuitka 에 노출한다.

사용:
    PATH=/tmp/onefilebuild/bin:$PATH \\
    /tmp/onefilebuild/bin/python deploy/build_onefile.py \\
      --out /tmp/onefile_out --final deploy/g2c
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/mnt/e/14_NIGHTLY/GTX_Compiler")
ENTRY = ROOT / "deploy" / "g2c_entry.py"

OUR_PKGS = ["shared", "parse", "qproc", "quantization", "utils", "nn"]
# 우리 pipeline 이 importlib 로 동적 로드 → standalone 이 자동 추적 못 함, 명시 포함.
# (ultralytics 는 --model yolo* 일 때만 필요 → 제외. torchvision/.pt 모델은 영향 없음.)
DYNAMIC_PKGS = ["torch", "torchvision", "gguf"]
# 런타임에 설정/가중치 메타(yaml 등) 데이터 파일을 읽는 패키지.
DATA_PKGS = ["torch"]
# ultralytics 전용 무거운 의존성 → 번들 제외(런타임 미사용). torchvision 은 PIL/numpy 만.
NOFOLLOW = ["ultralytics", "ultralytics_thop", "matplotlib", "pandas", "scipy", "cv2", "seaborn"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/onefile_out", help="빌드 산출 디렉토리")
    ap.add_argument("--final", default=str(ROOT / "deploy" / "g2c"), help="최종 바이너리 복사 위치")
    ap.add_argument("--name", default="g2c", help="실행파일 이름")
    args = ap.parse_args()

    if not ENTRY.exists():
        raise SystemExit(f"진입점 없음: {ENTRY}")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "nuitka",
        str(ENTRY),
        "--standalone", "--onefile",
        f"--output-dir={out}",
        f"--output-filename={args.name}",
        "--assume-yes-for-downloads",
        "--remove-output",
        "--lto=no",
        # 우리 파이프라인은 torch.jit.trace/freeze/ScriptModule 에 의존 → JIT 유지.
        "--module-parameter=torch-disable-jit=no",
        f"--jobs={os.cpu_count() or 4}",
    ]
    for p in OUR_PKGS + DYNAMIC_PKGS:
        cmd.append(f"--include-package={p}")
    for p in DATA_PKGS:
        cmd.append(f"--include-package-data={p}")
    for p in NOFOLLOW:
        cmd.append(f"--nofollow-import-to={p}")

    # shared 등 우리 패키지를 nuitka 모듈 탐색 경로에 노출.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)

    produced = out / args.name
    if not produced.exists():
        cands = [p for p in out.glob(f"{args.name}*") if p.is_file()]
        produced = cands[0] if cands else produced
    final = Path(args.final).resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, final)
    final.chmod(0o755)
    size_mb = final.stat().st_size / (1024 * 1024)
    print(f"\n완료: {final}  ({size_mb:.0f} MB)")
    print(f"실행: {final} --model resnet18 --output out_smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
