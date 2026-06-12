"""run_one.py — 생성된 export(.py)를 ggml 실행, 출력 npy 저장 (별도 subprocess).
사용: uv run python deploy/dverify/run_one.py <export.py> <out.npy> [input.npy]
  input.npy 지정 시 그 텐서를 입력으로 주입(export 의 random _x 대체), 미지정 시 seed-0 random.
  (in-process compile+run 은 prim_ops self.node=None 실패 → 반드시 별도 subprocess.)
"""
import sys, numpy as np
py, outp = sys.argv[1], sys.argv[2]
np.random.seed(0)
if len(sys.argv) > 3:                       # 고정 입력(실이미지 등) 주입
    fixed = np.load(sys.argv[3]).astype("float32")
    _orig = np.random.randn
    def _randn(*shape):
        if int(np.prod(shape)) == fixed.size:
            return fixed.reshape(shape).astype("float32")
        return _orig(*shape)
    np.random.randn = _randn               # export 의 `_np.random.randn(shape)` 가로채기
ns = {"__name__": "__main__", "__file__": py}
exec(open(py).read(), ns)
np.save(outp, np.asarray(ns["_main"]).astype("float32"))
print("RUN_OK", outp)
