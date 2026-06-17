#!/usr/bin/env python
"""C++ gtxc 가 쓴 GGUF 를 vision.cpp gguf-py 로 읽어 모델 state_dict 와 대조.

GGUF 텐서명 = state_dict 키, 값 = 실제 가중치(fp16). conv weight 는 OIHW(원본 레이아웃)로
저장하므로 state_dict 와 직접 비교된다. randomize seed 동일 → 값 일치.

사용: python cpp/tools/verify_gguf.py resnet18 cpp/assets/resnet18.gguf
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _modelkit import build  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GGUF_PY = os.path.join(ROOT, "vision.cpp", "depend", "llama", "gguf-py")
sys.path.insert(0, GGUF_PY)


def main():
    if len(sys.argv) != 3:
        print("usage: verify_gguf.py <model> <gguf_path>")
        sys.exit(2)
    model_spec, gguf_path = sys.argv[1], sys.argv[2]

    import gguf  # noqa: E402

    reader = gguf.GGUFReader(gguf_path)
    arch = None
    for k, field in reader.fields.items():
        if k == "general.architecture":
            arch = bytes(field.parts[field.data[0]]).decode()
    # yolo 등은 export 의 FlattenOut 래퍼(self.m=model)가 'm.' 접두를 붙인다 → state_dict
    # 매칭 위해 제거(값은 동일, self-consistent 한 cosmetic 차이).
    def strip_wrap(n):
        return n[2:] if n.startswith("m.") else n

    gtensors = {strip_wrap(t.name): np.array(t.data) for t in reader.tensors}

    model = build(model_spec)
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    print(f"GGUF: {gguf_path}  arch={arch}  tensors={len(gtensors)}")
    miss = [k for k in gtensors if k not in sd]
    bad = []
    worst = 0.0
    for name, garr in gtensors.items():
        if name not in sd:
            continue
        ref = sd[name].astype(np.float32)
        got = garr.astype(np.float32).reshape(ref.shape)
        err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-9))
        worst = max(worst, err)
        if err > 2e-3:  # fp16 상대오차 허용
            bad.append((name, err, list(ref.shape)))

    # state_dict 중 GGUF 에 없는 것(num_batches_tracked 제외)
    sd_missing = [k for k in sd if k not in gtensors and not k.endswith("num_batches_tracked")]

    print(f"매칭: {len(gtensors) - len(miss)}/{len(gtensors)}  최대 상대오차(fp16): {worst:.2e}")
    if miss:
        print(f"  ⚠ GGUF 에만 있는 텐서: {miss[:5]}")
    if sd_missing:
        print(f"  ⚠ state_dict 에만 있는(미수록) 텐서 {len(sd_missing)}개: {sd_missing[:5]}")
    if bad:
        print(f"  ❌ 값 불일치 {len(bad)}: {bad[:5]}")
        sys.exit(1)
    if miss:
        sys.exit(1)
    print(f"\n✅ GGUF {len(gtensors)} 텐서 전부 state_dict 와 일치 (fp16, 이름·shape·값)")


if __name__ == "__main__":
    main()
