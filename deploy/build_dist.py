#!/usr/bin/env python3
"""build_dist.py — Nuitka 모듈 컴파일로 *소스 비노출* 배포본 생성.

이 스크립트는 `deploy/` 안에 있고, 컴파일 대상 소스 패키지는 **상위 디렉토리(프로젝트
루트)** 의 shared/parse/qproc/quantization/utils/nn 이다. 이들을 Nuitka `--module` 로
네이티브 바이너리(.pyd/.so)로 컴파일하고, torch/ultralytics 등 외부 의존성은 일반 pip
패키지로 둔다. 결과 배포본에는 우리 코드의 .py 원본이 전혀 없어 타부서가 소스를
열람할 수 없다(머신코드 — 디컴파일로 .py 복원 불가).

  python deploy/build_dist.py                 # → <루트>/dist 에 배포본 생성
  python deploy/build_dist.py --clean         # 기존 산출물 지우고 재생성
  python deploy/build_dist.py --build /tmp/b  # C 컴파일 임시 디렉토리 지정(빠른 디스크 권장)

타겟: **Linux** (.so 산출). Nuitka 는 크로스 컴파일 불가 → 배포 대상과 같은 OS·
  같은 Python minor 버전·호환 glibc 환경에서 빌드한다.
  빌드 전제: Python(타겟과 동일 minor 버전), gcc/clang, `pip install nuitka`.

검증된 동작(deploy/DISTRIBUTION.md 참조):
  - whole-package `--module` 는 패키지당 단일 바이너리를 만들고 모든 서브모듈을
    dotted import 가능하게 한다.
  - Nuitka 가 `__file__` 을 원본 레이아웃(<dist>/shared/compile/pipeline.py)으로
    가상화하므로 pipeline 의 `project_root = dirname^3(__file__)` 경로 계산이 생존.
  - 빈 `dist/nn/` 디렉토리를 둬야 `nn/load_kernels.py` 의 import 시점
    `os.listdir(_cur_dir)` 가 FileNotFoundError 없이 통과한다(.so 를 가리지 않음).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# 이 스크립트는 deploy/ 안 → 프로젝트 루트는 부모 디렉토리.
ROOT = Path(__file__).resolve().parent.parent

# Nuitka 로 컴파일할 우리 소스 패키지 (torch/ultralytics 등은 제외 → pip 의존성).
PKGS = ["shared", "parse", "qproc", "quantization", "utils", "nn"]

# import 시점에 self 디렉토리를 os.listdir 하는 패키지 → 배포본에 빈 디렉토리 필요.
NEED_EMPTY_DIR = ["nn"]

LAUNCHER = '''\
"""g2c 진입점 런처 (console_scripts `g2c = shared.compile.pipeline:main` 대체).

    python g2c.py --model resnet18 --output output/resnet18
    python g2c.py --model "ultralytics.YOLO('yolo11n')" --output output/yolo11n
"""
import sys

from shared.compile.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
'''

REQUIREMENTS = '''\
# g2c 배포본 런타임 의존성 (우리 코드는 *.pyd/*.so 로 동봉, 아래는 외부 패키지)
#
# torch/torchvision 은 CUDA 12.4 휠 — 전용 인덱스에서 설치:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CPU 전용이면 https://download.pytorch.org/whl/cpu 사용.
torch>=2.5.1
torchvision>=0.20.1
ultralytics
networkx
tqdm
numpy
gguf            # GGUF writer (pipeline.generate_gguf) — vendored 경로 대신 pip 패키지로 충당
# --- ggml 런타임(선택): 생성물 python output/<Model>.py 검증 실행용 ---
# ggml-python    # 네이티브 libggml 필요. Windows 는 사전빌드 휠/직접 빌드 확인 후 활성화.
'''

DIST_README = '''\
# GTX Compiler — 배포본 (소스 비노출, Linux)

우리 컴파일러 코드는 `*.so` 네이티브 바이너리로 동봉되어 있습니다(소스 .py 없음).
PyTorch 등 외부 의존성만 별도 설치하면 됩니다.

## 설치

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# torch 는 CUDA 휠 인덱스 필요:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#   (CPU 전용이면 .../whl/cpu)
```

## 사용

```bash
python g2c.py --model resnet18 --output output/resnet18
python g2c.py --model "ultralytics.YOLO('yolo11n')" --output output/yolo11n
```

생성물: `output/<Model>.{cpp,h,gguf,py}`

## 주의
- 이 배포본의 Python 버전은 빌드 시점과 같은 minor 버전(3.12)이어야 `.so` 가 로드됩니다.
- glibc 호환: 빌드한 배포판과 같거나 더 새 버전의 Linux 에서 동작합니다.
- `nn/` 빈 디렉토리는 삭제하지 마세요(런타임 import 에 필요).
- `python output/<Model>.py`(ggml 런타임 검증)는 `ggml-python`(네이티브 libggml)
  설치 시에만 동작합니다 — requirements.txt 참조.
'''


def run(cmd: list, **kw) -> None:
    print(">>", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, **kw)


def find_binary(build_dir: Path, pkg: str) -> Path:
    """Nuitka 가 만든 패키지 바이너리(.pyd/.so) 를 찾는다."""
    cands = [
        p for p in build_dir.glob(f"{pkg}.*")
        if p.suffix in (".pyd", ".so") and p.is_file()
    ]
    if not cands:
        raise FileNotFoundError(f"{pkg}: Nuitka 산출 바이너리(.pyd/.so) 없음 in {build_dir}")
    return cands[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Nuitka 모듈 컴파일 배포본 빌더")
    ap.add_argument("--out", default=str(ROOT / "dist"), help="배포본 출력 디렉토리")
    ap.add_argument("--build", default=str(ROOT / "build_nuitka"), help="C 컴파일 임시 디렉토리")
    ap.add_argument("--clean", action="store_true", help="기존 out/build 삭제 후 재생성")
    ap.add_argument("--pkgs", nargs="*", default=PKGS, help="컴파일할 패키지(기본: 전체)")
    args = ap.parse_args()

    dist = Path(args.out).resolve()
    build = Path(args.build).resolve()

    if args.clean:
        for d in (dist, build):
            if d.exists():
                print(f"rm {d}")
                shutil.rmtree(d)

    build.mkdir(parents=True, exist_ok=True)
    dist.mkdir(parents=True, exist_ok=True)

    # 1) 패키지별 whole-module 컴파일 (cwd=프로젝트 루트 → 패키지가 import 경로에 있음)
    for pkg in args.pkgs:
        if not (ROOT / pkg).is_dir():
            raise SystemExit(f"패키지 디렉토리 없음: {ROOT / pkg}")
        run([
            sys.executable, "-m", "nuitka",
            "--module", pkg,
            f"--include-package={pkg}",
            f"--output-dir={build}",
            "--no-pyi-file",
            "--remove-output",
            "--assume-yes-for-downloads",
        ], cwd=ROOT)
        binary = find_binary(build, pkg)
        target = dist / binary.name
        shutil.copy2(binary, target)
        print(f"  → {target.name}")

    # 2) import 시 self-listdir 하는 패키지용 빈 디렉토리(.so 를 가리지 않음 — 검증됨)
    for pkg in NEED_EMPTY_DIR:
        if pkg in args.pkgs:
            (dist / pkg).mkdir(exist_ok=True)

    # 3) 런처 / requirements / README
    (dist / "g2c.py").write_text(LAUNCHER, encoding="utf-8")
    (dist / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (dist / "README.md").write_text(DIST_README, encoding="utf-8")

    print("\n배포본 준비 완료:", dist)
    bins = sorted(p.name for p in dist.glob("*.so")) + sorted(p.name for p in dist.glob("*.pyd"))
    print("동봉 바이너리:", ", ".join(bins))
    print("스모크 테스트:")
    print(f"  cd {dist} && python -c \"import shared.compile.pipeline, nn; print('import OK')\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
