"""setup_cython.py — 현재 디렉토리(스테이징 복사본)의 우리 패키지 .py 를 Cython .so 로 컴파일.

보호가 목적이므로 cdef 타입 최적화는 하지 않고 language_level=3 으로 "있는 그대로"
컴파일한다. build_cython.sh 가 스테이징 디렉토리 안에서 호출한다.

  PKGS 환경변수로 대상 패키지 한정 가능(기본: 6개 전체). 예) PKGS="utils" 로 부분 검증.
"""
import glob
import multiprocessing
import os

from setuptools import setup
from Cython.Build import cythonize

PKGS = (os.environ.get("PKGS") or
        "shared parse qproc quantization utils nn").split()


def _excluded(path: str) -> bool:
    """DISTRIBUTION.md §3 — 평문 .py 로 남겨야 하는 모듈은 컴파일에서 제외.

      - __init__.py        : 패키지 import 집계만(IP 없음), 패키지 인식 안정성 위해 평문
      - rnn_builder/*      : stacked_lstm 이 torch.jit.script 대상 → Python 소스 필요
    """
    name = os.path.basename(path)
    norm = path.replace(os.sep, "/")
    return name == "__init__.py" or "/rnn_builder/" in norm


files = []
for p in PKGS:
    if os.path.isdir(p):
        files += [f for f in glob.glob(os.path.join(p, "**", "*.py"), recursive=True)
                  if not _excluded(f)]

if not files:
    raise SystemExit("컴파일할 .py 없음 — 스테이징 복사가 됐는지 확인")

nthreads = multiprocessing.cpu_count()
print(f"[cython] {len(files)} files, nthreads={nthreads}, pkgs={PKGS}")

# 파일별로 cythonize 하여 한 파일이 실패해도 나머지를 끝까지 컴파일한다.
# 일부 소스는 정적 컴파일 불가(동적 이름·소스의 잠재 버그 — 예: 정의 안 된 이름 호출).
# 그런 파일은 .so 를 못 만들므로 평문 .py 로 남고, build_cython.sh 가 그대로 보존한다.
# annotation_typing=False 가 핵심 — Cython 3 은 PEP 484 어노테이션(attr_name: str 등)을
# C 타입으로 강제하는데, 우리 코드는 어노테이션과 다른 duck-typed 객체를 넘기는 곳이 많다
# ("있는 그대로" 컴파일이 목표 — 타입 강제 시 batch_norm/_convolution 등 파싱이 깨짐).
DIRECTIVES = {
    "language_level": "3",
    "emit_code_comments": False,
    "annotation_typing": False,
}
ext_modules = []
skipped = []
for f in sorted(files):
    try:
        ext_modules += cythonize(f, compiler_directives=DIRECTIVES, quiet=True)
    except Exception as e:  # Cython.Compiler.Errors.CompileError 등
        skipped.append((f, str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__))

if skipped:
    print(f"[cython] WARNING — {len(skipped)} files could not be cythonized (kept as plaintext .py):")
    for f, why in skipped:
        print(f"    SKIP {f}  ({why})")
    with open(".cython_skipped.txt", "w", encoding="utf-8") as fh:
        for f, why in skipped:
            fh.write(f"{f}\t{why}\n")

if not ext_modules:
    raise SystemExit("cythonize 성공 파일 0 — 환경/소스 확인")

print(f"[cython] compiling {len(ext_modules)} extensions (skipped {len(skipped)})")
setup(
    name="gtx_compiled",
    ext_modules=ext_modules,
    script_args=["build_ext", "--inplace", "-j", str(nthreads)],
)
