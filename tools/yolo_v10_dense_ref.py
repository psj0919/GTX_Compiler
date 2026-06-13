"""yolov10n dense(1,84,8400) 참조 — postprocess 우회(NMS-free top-k 는 CPU 후처리 영역).
usage: yolo_v10_dense_ref.py <prefix>  (입력은 <prefix>_input.bin 재사용, seed 동일)
"""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
torch.set_num_threads(1)
import ultralytics
from ultralytics.nn.modules import head as H

prefix = sys.argv[1] if len(sys.argv) > 1 else "tools/v10n"
SZ = 640
torch.manual_seed(0)
x = torch.randn(1, 3, SZ, SZ, dtype=torch.float32)

# postprocess 우회: y = self.postprocess(dense.permute(0,2,1)) → dense 그대로 반환되게
H.v10Detect.postprocess = staticmethod(lambda preds, *a, **k: preds.permute(0, 2, 1))

model = ultralytics.YOLO("yolov10n").model.eval()
with torch.no_grad():
    y = model(x)
y = y[0] if isinstance(y, (list, tuple)) else y
y = y.detach().cpu().float().numpy()
print(f"v10 dense shape: {y.shape}")   # 기대 (1,84,8400)
ref = np.ascontiguousarray(y.reshape(y.shape[-2], y.shape[-1]), dtype=np.float32)
ref.ravel().tofile(prefix + "_dense_ref.bin")
print(f"saved {prefix}_dense_ref.bin ({ref.size}) min={ref.min():.3f} max={ref.max():.3f}")
