"""yolo 계열 C++ 검증용 입력/참조 생성.

usage: yolo_cpp_ref.py <yolo_name> <out_prefix>
  예: yolo_cpp_ref.py yolov8n tools/v8n
  → <prefix>_input.bin (CWHN f32), <prefix>_ref.bin (torch 출력 ravel f32),
    <prefix>_meta.txt (torch 출력 shape)
출력 순서: 생성 .cpp 의 contiguous_2d_to_cwhn 결과 ne=[1,A,C,1] flat(anchor-fastest)이
torch (C,A) row-major 와 동일 선형순서 → ref 는 (C,A) 로 ravel.
"""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
torch.set_num_threads(1)
import ultralytics

name = sys.argv[1] if len(sys.argv) > 1 else "yolo11n"
prefix = sys.argv[2] if len(sys.argv) > 2 else "tools/y"
SZ = 640

torch.manual_seed(0)
x = torch.randn(1, 3, SZ, SZ, dtype=torch.float32)

model = ultralytics.YOLO(name).model.eval()
with torch.no_grad():
    y = model(x)
while isinstance(y, (list, tuple)):
    y = y[0]
y = y.detach().cpu().float().numpy()
print(f"[{name}] torch output shape: {y.shape}")
# (1, C, A) 가정 → (C, A) ravel. 다른 랭크면 그대로 ravel.
if y.ndim == 3 and y.shape[0] == 1:
    ref = np.ascontiguousarray(y[0], dtype=np.float32)
else:
    ref = np.ascontiguousarray(y.reshape(-1), dtype=np.float32)

inp = np.ascontiguousarray(
    x[0].permute(1, 2, 0).contiguous().numpy(), dtype=np.float32).ravel()

inp.tofile(prefix + "_input.bin")
ref.ravel().tofile(prefix + "_ref.bin")
with open(prefix + "_meta.txt", "w") as f:
    f.write(f"{name}\n{tuple(y.shape)}\n{ref.shape}\n")
print(f"[{name}] ref shape={ref.shape} min={ref.min():.4f} max={ref.max():.4f} "
      f"saved {prefix}_input.bin({inp.size}) {prefix}_ref.bin({ref.size})")
