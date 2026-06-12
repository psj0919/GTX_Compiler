"""compile_one.py — seed-0 모델을 jit/dispatch 로 g2c 컴파일 (검증용, 별도 subprocess).
사용: GTX_DISPATCH_TRACER=<0|1> uv run python deploy/dverify/compile_one.py <0|1> <outdir> [model]
  model: resnet18(기본) | yolo* (yolov8n/yolov9t/yolov10n/yolo11n/yolo12n 등 ultralytics 이름)
"""
import os, sys
os.environ["GTX_DISPATCH_TRACER"] = sys.argv[1]
toggle, outdir = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "resnet18"
from shared.compile.pipeline import compile_model
if model.startswith("yolo"):
    from ultralytics import YOLO
    m = YOLO(model).model.cpu().eval()
    name, shape = "DetectionModel", (1, 3, 640, 640)
else:
    import torch
    from torchvision.models import resnet18
    torch.manual_seed(0)
    m = resnet18().eval()
    name, shape = "ResNet", (1, 3, 224, 224)
print("COMPILE_OK" if compile_model(m, name, shape, outdir) else "COMPILE_FAIL")
