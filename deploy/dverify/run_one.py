"""run_one.py — 생성된 export(.py)를 seed-0 입력으로 ggml 실행, 출력 npy 저장 (별도 subprocess).
사용: uv run python deploy/dverify/run_one.py <export.py> <out.npy>
  (in-process compile+run 은 prim_ops self.node=None 실패 → 반드시 별도 subprocess.)
"""
import sys, numpy as np
py, outp = sys.argv[1], sys.argv[2]
np.random.seed(0)
ns = {"__name__": "__main__", "__file__": py}
exec(open(py).read(), ns)
np.save(outp, np.asarray(ns["_main"]).astype("float32"))
print("RUN_OK", outp)
