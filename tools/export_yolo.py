"""Parse a YOLO model to a  graph and write its ggml-runnable export.

    uv run --no-sync python tools/export_yolo.py [weights.pt]
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch
from ultralytics import YOLO

from parse import TorchParser
from parse.rich_in_out_helper import StandardInputData
from utils.module_util import get_module_name
from utils import TorchSymbol
from qproc.export import get_script_writer


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "gadget", "yolov8n.pt")
    print(f"[export_yolo] weights: {weights}", flush=True)
    model = YOLO(weights).model.cpu().eval()
    print(f"[export_yolo] model: {model._get_name()}", flush=True)

    inputs = torch.randn(1, 3, 640, 640)
    os.makedirs(os.path.join(HERE, "export1"), exist_ok=True)

    input_data = StandardInputData((inputs,), {})
    print("[export_yolo] parsing...", flush=True)
    graph = TorchParser()(get_module_name(model), model, input_data)
    print(f"[export_yolo] graph nodes: {len(list(getattr(graph, 'nodes', [])))}", flush=True)

    export_file = os.path.join(HERE, "export1", get_module_name(model) + TorchSymbol.SCRIPT_SUFFIX)
    get_script_writer(enable_quant=True).write(graph, file_path=export_file)
    print(f"[export_yolo] wrote: {export_file}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
