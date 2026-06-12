"""make_input.py — ultralytics 번들 실이미지를 (1,3,H,W) fp32 텐서(npy)로 전처리.
사용: uv run python deploy/dverify/make_input.py <out.npy> <H,W> [bus|zidane]
  RGB, resize(H,W), /255, NCHW. torch·ggml 동일 입력으로 쓰기 위한 결정적 전처리.
"""
import sys, numpy as np
from PIL import Image
from ultralytics.utils import ASSETS

out = sys.argv[1]
H, W = (int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (640, 640)
name = (sys.argv[3] if len(sys.argv) > 3 else "bus") + ".jpg"
img = Image.open(ASSETS / name).convert("RGB").resize((W, H), Image.BILINEAR)
arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC, 0~1
arr = arr.transpose(2, 0, 1)[None]                        # 1,C,H,W
np.save(out, arr.astype(np.float32))
print(f"made {out} shape={arr.shape} from {name}")
