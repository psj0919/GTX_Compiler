"""GGML codegen operator 커버리지 검증 (97개 대상).

ggml 정규 목록(106) 중 제외 9개를 뺀 97개에 대해:
  1) 빌더 가용성 — 현재 ggml 라이브러리에 ggml_<op> 빌더가 존재하는가
  2) render 와이어링 — codegen render 가 그 ggml 빌더를 emit 하는가

    uv run python tools/ggml_codegen_coverage.py

설계: shared/compile/ggml_ops.py (단일 소스).
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ggml  # noqa: E402

from shared.compile.ggml_ops import (  # noqa: E402
    ALL_OPS, EXCLUDED, TARGET_OPS, builder_name, emitted_builders,
)


def main():
    emitted = emitted_builders()

    avail, wired, missing_builder, not_wired = [], [], [], []
    for op in TARGET_OPS:
        b = builder_name(op, ggml)
        if b is None:
            missing_builder.append(op)
            continue
        avail.append(op)
        # 와이어링: 빌더(또는 base) 가 render/backend 코드에 emit 되는가
        if b in emitted or f"ggml_{op.lower()}" in emitted:
            wired.append(op)
        else:
            not_wired.append(op)

    print("=" * 64)
    print("GGML codegen operator 커버리지")
    print("=" * 64)
    print(f"  전체 GGML OP        : {len(ALL_OPS)}")
    print(f"  제외(view/학습/GLU) : {len(EXCLUDED)}  {sorted(EXCLUDED)}")
    print(f"  대상(TARGET)        : {len(TARGET_OPS)}")
    print("-" * 64)
    print(f"  빌더 가용(이 ggml)  : {len(avail)}/{len(TARGET_OPS)}")
    if missing_builder:
        print(f"    └ 빌더 없음       : {missing_builder}")
    print(f"  render 와이어링     : {len(wired)}/{len(TARGET_OPS)}")
    print("-" * 64)
    print("  와이어링됨(render emit):")
    print("   ", ", ".join(sorted(wired)))
    print("\n  미와이어링(빌더는 있음 — 대응 PyTorch op render 추가 시 사용):")
    print("   ", ", ".join(sorted(not_wired)))
    print("=" * 64)

    # 가용성 100%(=빌더 존재)면 성공. 와이어링은 PyTorch op 등장 여부에 따라 점증.
    ok = len(missing_builder) == 0
    print(f"RESULT: builder availability {'PASS' if ok else 'PARTIAL'} "
          f"({len(avail)}/{len(TARGET_OPS)}), wired {len(wired)}/{len(TARGET_OPS)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
