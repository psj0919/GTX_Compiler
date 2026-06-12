"""verify_torch.py — ggml 출력(npy) 을 PyTorch eager(numpy) 레퍼런스와 max diff 비교.
사용: uv run python deploy/dverify/verify_torch.py <model_spec> <ggml.npy> [input.npy|shape]
  3번째 인자가 .npy 면 그 텐서를 입력으로(실이미지), 아니면 shape("1,3,224,224") 로 seed-0 random.
  cosine 은 쓰지 않고 max|diff| / meanabs 만 본다.
"""
import sys, numpy as np, torch
from shared.compile.pipeline import build_model

spec, npy = sys.argv[1], sys.argv[2]
torch.manual_seed(0)  # resnet 등 랜덤 init 모델을 compile(compile_one) 과 동일 가중치로
model, name, shape = build_model(spec, pth=None, input_shape=None)

arg3 = sys.argv[3] if len(sys.argv) > 3 else None
if arg3 and arg3.endswith(".npy"):
    x = np.load(arg3).astype("float32")               # 실이미지 등 고정 입력(run_one 과 동일)
else:
    if arg3:
        shape = tuple(int(s) for s in arg3.replace(" ", "").split(","))
    np.random.seed(0)
    x = np.random.randn(*shape).astype("float32")     # export _main 과 동일 입력

with torch.no_grad():
    out = model(torch.from_numpy(x))
ref = out[0] if isinstance(out, (list, tuple)) else out
ref = np.asarray(ref.detach().cpu()).astype("float32").flatten()

g = np.load(npy).astype("float32").flatten()
n = min(ref.size, g.size)
a, b = ref[:n], g[:n]
print(f"{name:14s} torch{ref.shape} ggml{g.shape} | "
      f"max|diff|={float(np.abs(a - b).max()):.5f} meanabs={float(np.abs(a - b).mean()):.5f}")
