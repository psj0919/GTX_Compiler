#!/usr/bin/env python
"""PyTorch 모델을 trace 하여 TorchScript(.pt)로 직렬화 — C++ 파서 입력.

C++ 의 torch::jit::load 가 소비할 traced 그래프를 만든다. 트레이싱 자체는
언어 무관(PyTorch eager). C++ 쪽은 이 .pt 를 받아 동일한 JIT 최적화 패스를
적용한 뒤 Graph IR 로 변환한다 → Python TorchParser 와 1:1 대조.

사용:
  python cpp/tools/export_traced.py --model resnet18 --out cpp/assets/resnet18
  python cpp/tools/export_traced.py --model torchvision.models.resnet18 --input-shape 1,3,224,224
"""
import argparse
import os
import sys

# trace hang 방지 (CLAUDE.md 주의사항)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _modelkit import build, default_input_shape  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="출력 경로 prefix (.pt / .meta.json)")
    ap.add_argument("--input-shape", default=None)
    args = ap.parse_args()

    shape = (tuple(int(s) for s in args.input_shape.split(","))
             if args.input_shape else default_input_shape(args.model))
    model = build(args.model)  # randomize 포함(freeze dedup 방지)
    example = torch.randn(*shape)

    class FlattenOut(torch.nn.Module):
        """출력을 텐서 튜플로 평탄화 — yolo 처럼 혼합 컨테이너 출력을 trace 가능하게.

        Python TorchParser 의 FlattenInOutModelForTrace 와 동일 효과(출력 텐서 집합 보존,
        내부 그래프 동일).
        """
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(x)
            flat = []

            def rec(o):
                if isinstance(o, torch.Tensor):
                    flat.append(o)
                elif isinstance(o, (list, tuple)):
                    for e in o:
                        rec(e)
                elif isinstance(o, dict):
                    for e in o.values():
                        rec(e)

            rec(out)
            return tuple(flat)

    with torch.no_grad():
        try:
            traced = torch.jit.trace(model.eval(), example, strict=False)
        except RuntimeError:
            # 혼합 컨테이너 출력 → 평탄화 래퍼로 재시도
            traced = torch.jit.trace(FlattenOut(model).eval(), example, strict=False)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    pt_path = args.out + ".pt"
    traced.save(pt_path)

    # C++ 가 input shape 등 메타를 알 수 있게 사이드카 JSON
    import json
    meta = {"model": args.model, "input_shape": list(shape), "pt": os.path.basename(pt_path)}
    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[export] traced → {pt_path}  shape={shape}")
    print(f"[export] meta   → {args.out}.meta.json")


if __name__ == "__main__":
    main()
