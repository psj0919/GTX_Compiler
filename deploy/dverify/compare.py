"""compare.py — 두 npy(jit vs dispatch ggml 출력) 값 비교 (cosine 아님).
사용: uv run python deploy/dverify/compare.py <jit.npy> <dispatch.npy>
"""
import sys, numpy as np
a = np.load(sys.argv[1]).flatten()
b = np.load(sys.argv[2]).flatten()
np.set_printoptions(precision=5, suppress=True)
print("shapes:", a.shape, b.shape)
print("완전 동일:", bool(np.array_equal(a, b)), "| max|diff| =", float(np.abs(a - b).max()))
print("jit [:6]:", a[:6])
print("dsp [:6]:", b[:6])
