import os
import sys
import torch
import torch.nn as nn
from torchvision.models import resnet18
# Ensure project root is on sys.path so local imports (parse, qproc, etc.) work
from ultralytics import YOLO
from parse.rich_in_out_helper import StandardInputData
from gtx_utils.module_util import get_module_name
from gtx_utils import TorchSymbol
import traceback


from parse import TorchParser

def main():
    print("[yolov9t.py] Start", flush=True)
    # 1. PyTorch 모델 준비
    model = resnet18().cpu()
    # model = YOLO("yolov9t").model.cpu()
    model.eval()
    print(f"[yolov9t.py] Model: {model._get_name()}", flush=True)

    # 2. 입력 샘플 준비
    inputs = torch.randn(1, 3, 640, 640).cpu()
    print(f"[yolov9t.py] Input shape: {tuple(inputs.shape)}", flush=True)

    # Export 폴더 준비
    
    os.makedirs("export1", exist_ok=True)
    print("[yolov9t.py] Export dir: export1", flush=True)

    try:
        # TorchParser로 그래프 파싱
        input_data = StandardInputData((inputs,), {})
        print("[yolov9t.py] Parsing graph...", flush=True)
        graph = TorchParser()(get_module_name(model), model, input_data)
        try:
            num_nodes = len(list(getattr(graph, 'nodes', [])))
        except Exception:
            num_nodes = len(list(graph.all_nodes())) if hasattr(graph, 'all_nodes') else -1
        print(f"[resnet.py] Graph nodes: {num_nodes}", flush=True)

        # exporter 임포트(지연 로딩)
        print("[yolov9t.py] Import exporter ...", flush=True)
        
        from qproc.export import get_script_writer
        print("[yolov9t.py] get_script_writer imported", flush=True)
        
        # exporter로 단일 클래스 Python 파일 생성
        export_file = os.path.join("export1", get_module_name(model) + TorchSymbol.SCRIPT_SUFFIX)
        print(f"[yolov9t.py] Writing script: {export_file}", flush=True)
        exporter = get_script_writer(enable_quant=True)
        exporter.write(graph, file_path=export_file)

        print("Exporter wrote:", export_file if os.path.exists(export_file) else "<not found>", flush=True)
    except Exception:
        print("[yolov9t.py] Failed:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
