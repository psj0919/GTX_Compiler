"""verify_torch.py — ggml 출력(npy) 을 PyTorch eager(numpy) 레퍼런스와 비교.
사용: uv run python deploy/dverify/verify_torch.py <model_spec> <ggml.npy> [shape]
  입력은 export _main 과 동일하게 np.random.seed(0) 후 randn(shape).
"""
import sys, numpy as np, torch
from shared.compile.pipeline import build_model

spec, npy = sys.argv[1], sys.argv[2]
torch.manual_seed(0)  # resnet 등 랜덤 init 모델을 compile(compile_one) 과 동일 가중치로
model, name, shape = build_model(spec, pth=None, input_shape=None)
if len(sys.argv) > 3:
    shape = tuple(int(s) for s in sys.argv[3].replace(" ", "").split(","))

np.random.seed(0)
x = np.random.randn(*shape).astype("float32")     # export _main 과 동일 입력
with torch.no_grad():
    out = model(torch.from_numpy(x))
ref = out[0] if isinstance(out, (list, tuple)) else out
ref = np.asarray(ref.detach().cpu()).astype("float32").flatten()

g = np.load(npy).astype("float32").flatten()
n = min(ref.size, g.size)
a, b = ref[:n], g[:n]
cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
print(f"{name:14s} torch{ref.shape} ggml{g.shape} | cos={cos:.6f} max|diff|={float(np.abs(a-b).max()):.5f} "
      f"meanabs={float(np.abs(a-b).mean()):.5f}")
