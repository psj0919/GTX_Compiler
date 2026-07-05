"""_adaptive_avg_pool2d(eager ggml 백엔드)가 torch F.adaptive_avg_pool2d 와 일치하는지.

divisible / non-divisible / global(1,1) 케이스를 torch 와 대조.
  uv run --no-sync python tools/test_adaptive_pool.py
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from nn.modules.ggml_backend import _adaptive_avg_pool2d


def main():
    torch.manual_seed(0)
    cases = [
        (7, 7, 3, 3),    # non-divisible (가변커널)
        (8, 8, 4, 4),    # divisible 균일
        (7, 7, 1, 1),    # global
        (13, 11, 5, 4),  # 비대칭 non-divisible
        (6, 6, 6, 6),    # identity
    ]
    for H, W, oh, ow in cases:
        x = torch.randn(2, 3, H, W)
        ref = F.adaptive_avg_pool2d(x, (oh, ow)).numpy()
        got = _adaptive_avg_pool2d(x.numpy().astype(np.float32), oh, ow)
        assert got.shape == ref.shape, (H, W, oh, ow, got.shape, ref.shape)
        md = float(np.abs(got - ref).max())
        assert md < 1e-5, (H, W, oh, ow, md)
    print("OK — _adaptive_avg_pool2d 가 torch 와 일치 (divisible/non-divisible/global).")


if __name__ == "__main__":
    main()
