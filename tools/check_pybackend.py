"""생성 .py 의 python ggml 백엔드 vs torch ref 비교 (graph/gguf 정합 확인).
usage: check_pybackend.py <gen_dir> <yolo_name>
"""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import importlib.util
import numpy as np
import torch
torch.set_num_threads(1)

gen_dir = sys.argv[1]
name = sys.argv[2]
arch = os.path.basename([f for f in os.listdir(gen_dir)
                         if f.endswith(".py")][0])[:-3]
py = os.path.join(gen_dir, arch + ".py")
gguf = os.path.join(gen_dir, arch + ".gguf")

torch.manual_seed(0)
x = torch.randn(1, 3, 640, 640, dtype=torch.float32)

# torch ref
import ultralytics
tm = ultralytics.YOLO(name).model.eval()
with torch.no_grad():
    ty = tm(x)
while isinstance(ty, (list, tuple)):
    ty = ty[0]
ty = ty.detach().cpu().float().numpy()

# python ggml backend
spec = importlib.util.spec_from_file_location("genmod", py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import nn as _nn
_nn.set_backend("ggml")
out = _nn.run_gguf(mod.__dict__[arch](), gguf, py, x.numpy())
out = out[0] if isinstance(out, (list, tuple)) else out
out = np.asarray(out).astype(np.float64)
ref = ty.astype(np.float64)
print(f"py-backend out={out.shape} torch={ref.shape}")
a, b = out.ravel(), ref.ravel()
n = min(a.size, b.size)
a, b = a[:n], b[:n]
cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
print(f"py-backend vs torch: cos={cos:.6f} max|diff|={np.abs(a-b).max():.4f}")
