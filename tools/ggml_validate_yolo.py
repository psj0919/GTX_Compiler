"""Full end-to-end YOLOv8n on the ggml backend vs ultralytics PyTorch.

Runs the generated export (export1/DetectionModel.py) through the ggml backend,
binding weights by name, and compares the decoded detection output
(output_module_292, ~(1,84,8400)) against the ultralytics model.

    uv run --no-sync python tools/export_yolo.py        # once, to generate the export
    uv run --no-sync python tools/ggml_validate_yolo.py [dtype]
"""

import importlib.util
import os
import sys

import numpy as np
import torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

import nn
from ggml_weight_binder import bind_from_state_dict


def main():
    dtype = sys.argv[1] if len(sys.argv) > 1 else "fp32"
    weights = os.path.join(HERE, "gadget", "yolov8n.pt")
    tv = YOLO(weights).model.cpu().eval()

    torch.manual_seed(0)
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = tv(x)
    y = out[0] if isinstance(out, (tuple, list)) else out
    ref = y.numpy()
    print(f"torch output : {ref.shape}")

    export_path = os.path.join(HERE, "export1", "DetectionModel.py")
    spec = importlib.util.spec_from_file_location("export1_det", export_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    nn.set_backend("ggml")
    nn.set_dtype(dtype)
    model = mod.DetectionModel()
    nbound, missing = bind_from_state_dict(model, tv.state_dict(), export_path)
    print(f"bound        : {nbound} modules  missing: {len(missing)}")
    if missing:
        for m in missing[:8]:
            print(f"   missing {m}")

    result = model(x)
    pred = np.asarray(result[0] if isinstance(result, (tuple, list)) else result)
    pred = pred.reshape(ref.shape)
    print(f"ggml output  : {pred.shape}")

    max_err = np.abs(pred - ref).max()
    rel = max_err / (np.abs(ref).max() + 1e-9)
    print(f"max_err      : {max_err:.4e}")
    print(f"rel_err      : {rel:.4e}")
    tol = {"fp32": 1e-2, "fp16": 8e-2}.get(dtype, 0.2)
    ok = not missing and rel < tol
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
