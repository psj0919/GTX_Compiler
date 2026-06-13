"""실제 이미지로 v10 검증 참조 생성. 동일 전처리 텐서를 torch/C++ 양쪽에 사용.
usage: yolo_v10_img_ref.py <image> <prefix>
  → <prefix>_input.bin (CWHN), <prefix>_ref.bin (torch (300,6))
전처리: RGB resize 640x640, /255 (yolo 정규화). letterbox 아님(비교용, 양쪽 동일하면 무방).
"""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
torch.set_num_threads(1)
from PIL import Image
import ultralytics

img_path = sys.argv[1]
prefix = sys.argv[2]
SZ = 640

im = Image.open(img_path).convert("RGB").resize((SZ, SZ), Image.BILINEAR)
arr = np.asarray(im, dtype=np.float32) / 255.0          # (H,W,C)
x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()  # (1,3,H,W)

model = ultralytics.YOLO("yolov10n").model.eval()
with torch.no_grad():
    y = model(x)
y = y[0] if isinstance(y, (list, tuple)) else y
y = y.detach().cpu().float().numpy()
print(f"torch v10 output: {y.shape}")    # (1,300,6)
ref = np.ascontiguousarray(y.reshape(-1, 6), dtype=np.float32)

# CWHN 입력 = (H,W,C) ravel (c fastest)
inp = np.ascontiguousarray(arr, dtype=np.float32).ravel()
inp.tofile(prefix + "_input.bin")
ref.ravel().tofile(prefix + "_ref.bin")
nd = (ref[:, 4] > 0.25).sum()
print(f"saved {prefix}_input.bin({inp.size}) {prefix}_ref.bin({ref.size}) "
      f"| conf>0.25: {nd} dets, top conf={ref[:,4].max():.3f}")
