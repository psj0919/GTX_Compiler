"""C++ 생성물(cpp/out/<model>) 검증용 입력/torch 참조 생성.

usage: cpp_ref.py <traced.pt> <out_prefix> [size]
  traced.pt: cpp/assets/<model>.pt (gguf 가중치의 출처와 동일 trace).
  → <prefix>_input.bin (CWHN f32), <prefix>_ref.bin (torch 출력 ravel f32),
    <prefix>_meta.txt

gguf 가중치는 이 .pt 에서 굽기 때문에 torch 참조도 반드시 같은 .pt(torch.jit.load)
에서 만들어야 한다(가중치 random 화 → 새 모델 재생성은 불일치). run_yolo_cpp 하네스가
input 을 CWHN(3,SZ,SZ) 으로 읽고, forward 출력(contiguous_2d_to_cwhn flat)을 torch 와
동일 선형순서로 비교.
"""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
torch.set_num_threads(1)

pt = sys.argv[1] if len(sys.argv) > 1 else "cpp/assets/resnet18.pt"
prefix = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(pt)[0]
SZ = int(sys.argv[3]) if len(sys.argv) > 3 else (640 if "yolo" in pt else 224)

torch.manual_seed(0)
x = torch.randn(1, 3, SZ, SZ, dtype=torch.float32)

model = torch.jit.load(pt).eval()
with torch.no_grad():
    y = model(x)
while isinstance(y, (list, tuple)):
    y = y[0]
y = y.detach().cpu().float().numpy()
print(f"[{pt}] torch output shape: {y.shape}")
if y.ndim == 3 and y.shape[0] == 1:
    ref = np.ascontiguousarray(y[0], dtype=np.float32)
else:
    ref = np.ascontiguousarray(y.reshape(-1), dtype=np.float32)

inp = np.ascontiguousarray(
    x[0].permute(1, 2, 0).contiguous().numpy(), dtype=np.float32).ravel()

inp.tofile(prefix + "_input.bin")
ref.ravel().tofile(prefix + "_ref.bin")
with open(prefix + "_meta.txt", "w") as f:
    f.write(f"{pt}\n{tuple(y.shape)}\n{ref.shape}\n{SZ}\n")
print(f"[{pt}] SZ={SZ} ref shape={ref.shape} min={ref.min():.4f} max={ref.max():.4f} "
      f"saved {prefix}_input.bin({inp.size}) {prefix}_ref.bin({ref.size})")
