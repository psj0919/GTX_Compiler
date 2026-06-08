#!/usr/bin/env python3
# Copyright (C) Supergate - All Rights Reserved
#  Compiler - PyTorch Model → C 소스코드 변환 파이프라인
#
# 사용법:
#   python compile_to_c.py --model export1/ResNet.py --output output/
#
# 전체 흐름:
#   1. PyTorch 모델 로드
#   2. TorchParser →  Graph 파싱
#   3. prepare_quantizable_module → 모듈+그래프 동기화
#   4. ModuleHooker.update_blobs_once → shape/param 데이터 채우기
#   5. CCodeGenerator → C 소스코드 생성 (.c/.h)
#   6. (선택) RISC-V 크로스 컴파일 → .elf

import os
import sys
import argparse
import importlib.util
import torch
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def generate_ggml_code(graph, output_dir: str, model_name: str = "model"):
    """Graph → vision.cpp(ggml) arch 스타일 C++ 소스코드 생성.

    설계: docs/ggml_codegen.md
    """
    from shared.compile.ggml_codegen import VispCodeGenerator

    codegen = VispCodeGenerator(
        graph,
        output_dir=output_dir,
        model_name=model_name,
    )
    files = codegen.generate()
    return files


def generate_gguf(model, output_dir: str, model_name: str, arch_id: str):
    """모델 state_dict 를 fp16 GGUF 로 직렬화한다.

    텐서명은 PyTorch state_dict 키(`conv1.weight`, `layer1.0.conv1.weight`, `bn1.running_mean`...)
    를 그대로 쓴다 → 생성된 .cpp 의 `m["conv1"]`(=`m.weights("conv1.weight")`)·
    `batch_norm_2d(m["bn1"])` 와 정합. metadata `general.architecture = arch_id`.

    GGUF writer 는 vision.cpp 에 vendor 된 gguf-py 를 사용한다.
    """
    gguf_py = os.path.join(
        project_root, "vision.cpp", "depend", "llama", "gguf-py"
    )
    if gguf_py not in sys.path:
        sys.path.insert(0, gguf_py)
    try:
        import gguf
    except Exception as e:
        print(f"  ⚠ gguf 로드 실패 ({e}) — GGUF 생략")
        return None

    path = os.path.join(output_dir, f"{model_name}.gguf")
    writer = gguf.GGUFWriter(path, arch=arch_id)
    writer.add_architecture()  # general.architecture = arch_id

    n = 0
    for key, tensor in model.state_dict().items():
        # BN 통계 카운터는 추론에 불필요
        if key.endswith("num_batches_tracked"):
            continue
        arr = tensor.detach().cpu().numpy().astype(np.float16)
        writer.add_tensor(key, arr)
        n += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  → gguf: {path} ({n} tensors, fp16, general.architecture='{arch_id}')")
    return path


def append_ggml_runner(export_path: str, class_name: str, gguf_name: str,
                       input_shape=(1, 3, 224, 224)):
    """export 파일 끝에 ggml 실행 진입점을 추가한다.

    `python output/<Model>.py` 로 직접 실행하면 ggml(libggml.so) 커널로 forward 가
    돈다: set_backend('ggml') → nn.Module 들이 GgmlModule 로 생성 →
    run_gguf 가 옆의 .gguf 가중치를 이름 기반 바인딩 후 forward 실행.
    """
    marker = "# ----- ggml 실행 진입점 (libggml.so) -----"
    src = open(export_path).read()
    if marker in src:
        return

    # `python output/<Model>.py` 직접 실행 시 sys.path[0]=output/ 이라 프로젝트 nn 을
    # 못 찾는다. `import nn` 앞에 프로젝트 루트(=output 의 부모)를 sys.path 에 주입.
    boot = (
        "import os as _bos, sys as _bsys\n"
        "_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname("
        "_bos.path.abspath(__file__))))\n"
    )
    if "_bsys.path.insert" not in src:
        src = src.replace("import nn\n", boot + "import nn\n", 1)

    shape = ", ".join(str(int(s)) for s in input_shape)
    block = f'''

{marker}
if __name__ == "__main__":
    import os as _os
    import numpy as _np
    import nn as _nn

    _nn.set_backend("ggml")
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _gguf = _os.path.join(_here, "{gguf_name}")
    _x = _np.random.randn({shape}).astype("float32")
    _out = _nn.run_gguf({class_name}(), _gguf, __file__, _x)
    _main = _out[0] if isinstance(_out, (list, tuple)) else _out
    _main = _np.asarray(_main)
    print("[ggml] output:", _main.shape)
'''
    if not src.endswith("\n"):
        src += "\n"
    with open(export_path, "w") as f:
        f.write(src + block)


def load_torch_model(model_spec: str):
    """
    다양한 형태의 PyTorch 모델을 로드합니다.

    지원 형태:
      - "resnet18" → torchvision.models.resnet18(pretrained=False)
      - "torchvision.models.resnet18" → 동적 임포트
      - "path/to/model.pt" → torch.load()
    """
    import importlib

    # 1) torchvision 내장 모델 이름 (resnet18, vgg16, etc.)
    try:
        import torchvision.models as tv_models

        if hasattr(tv_models, model_spec):
            model_fn = getattr(tv_models, model_spec)
            model = model_fn(weights=None)
            model.eval()
            print(f"  → torchvision 모델 로드: {model_spec}")
            return model
    except Exception:
        pass

    # 2) 전체 Python 경로 (예: torchvision.models.resnet18)
    if (
        "." in model_spec
        and not model_spec.endswith(".py")
        and not model_spec.endswith(".pt")
    ):
        try:
            parts = model_spec.rsplit(".", 1)
            module = importlib.import_module(parts[0])
            model_fn = getattr(module, parts[1])
            model = model_fn(weights=None) if callable(model_fn) else model_fn
            if isinstance(model, torch.nn.Module):
                model.eval()
                print(f"  → 동적 임포트 모델: {model_spec}")
                return model
        except Exception:
            pass

    # 3) .pt/.pth 파일
    if model_spec.endswith(".pt") or model_spec.endswith(".pth"):
        model = torch.load(model_spec, map_location="cpu")
        if isinstance(model, torch.nn.Module):
            model.eval()
            print(f"  → 저장된 모델 로드: {model_spec}")
            return model

    return None


def _infer_input_shape_from_graph(graph):
    """그래프의 INPUT 노드 출력 shape 를 추론한다. 없으면 None."""
    from shared.base.key_names import OP

    try:
        for node in getattr(graph, "nodes", []):
            if getattr(node.op, "type", None) == OP.INPUT and node.out_tensors:
                shape = list(node.out_tensors[0].shape)
                if shape:
                    return shape
    except Exception:
        pass
    return None


def generate_export_python(graph, model, output_dir: str, model_name: str = "model"):
    """export1/ResNet.py 스타일의 Python 모듈 파일을 생성한다 (resnet.py 와 동일 경로).

    이미 TorchParser 그래프이면 그대로 get_script_writer 로 내보내고,
    MockGraph 이면 model 을 TorchParser 로 재파싱한 뒤 내보낸다.
    op-type 는 대문자로 기록되고, 로드 시 get_torch_op_type 가 정규화한다.
    """
    import traceback

    from parse import TorchParser
    from parse.rich_in_out_helper import StandardInputData
    from qproc.export import get_script_writer
    from utils.module_util import get_module_name

    os.makedirs(output_dir, exist_ok=True)
    export_file = os.path.join(output_dir, f"{model_name}.py")

    try:
        # MockGraph(_nodes 보유)이면 model 을 재파싱하여 TorchParser 그래프를 얻는다.
        if hasattr(graph, "_nodes"):
            input_shape = _infer_input_shape_from_graph(graph) or [1, 3, 224, 224]
            inputs = torch.randn(*input_shape)
            input_data = StandardInputData((inputs,), {})
            graph = TorchParser()(get_module_name(model), model, input_data)

        exporter = get_script_writer(enable_quant=True)
        exporter.write(graph, file_path=export_file)
        print(f"  → Python 모듈 파일 (ScriptWriter): {export_file}")
    except Exception:
        print("[compile_to_c] export 생성 실패:")
        traceback.print_exc()
        raise

    return export_file

def load_named_model(spec: str):
    """모델 이름 → (model, input_shape).

    - 'yolo*'(yolov8n/yolov9t/yolov10n/yolo11n/yolo12n …): ultralytics YOLO().model, 640x640
    - 그 외: torchvision 분류 모델(resnet18 등), 224x224
    """
    if "yolo" in spec.lower():
        from ultralytics import YOLO

        model = YOLO(spec).model.cpu().eval()
        return model, (1, 3, 640, 640)
    import torchvision.models as tvm

    model = getattr(tvm, spec)().cpu().eval()
    return model, (1, 3, 224, 224)


def main():
    # 모델 → TorchParser 그래프 → ScriptWriter export → visp/ggml arch C++ + GGUF.
    import argparse
    import traceback

    from parse import TorchParser
    from parse.rich_in_out_helper import StandardInputData
    from utils.module_util import get_module_name
    from utils import TorchSymbol

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="resnet18",
        help="resnet18 등 torchvision 모델, 또는 yolov8n/yolov9t/yolov10n/yolo11n/yolo12n",
    )
    ap.add_argument("--output", default="output")
    args, _ = ap.parse_known_args()
    output_dir = args.output

    print("[compile_to_c] Start", flush=True)
    # 1. PyTorch 모델 준비
    model, input_shape = load_named_model(args.model)
    name = get_module_name(model)
    print(f"[compile_to_c] Model: {args.model} ({model._get_name()})", flush=True)

    # 2. 입력 샘플 준비
    inputs = torch.randn(*input_shape).cpu()
    print(f"[compile_to_c] Input shape: {tuple(inputs.shape)}", flush=True)

    os.makedirs(output_dir, exist_ok=True)

    graph = None
    try:
        # TorchParser 로 그래프 파싱
        input_data = StandardInputData((inputs,), {})
        print("[compile_to_c] Parsing graph (TorchParser)...", flush=True)
        graph = TorchParser()(name, model, input_data)
        num_nodes = len(list(getattr(graph, "nodes", [])))
        print(f"[compile_to_c] Graph nodes: {num_nodes}", flush=True)

        # ScriptWriter 로 export 생성 (op-type 대문자). export1/ 와 output/ 양쪽에 둔다.
        from qproc.export import get_script_writer

        export_name = name + TorchSymbol.SCRIPT_SUFFIX
        exporter = get_script_writer(enable_quant=True)
        exporter.write(graph, file_path=os.path.join(output_dir, export_name))
        print(
            f"[compile_to_c] Export written: {output_dir}/{export_name}",
            flush=True,
        )
    except Exception:
        print("[compile_to_c] Parse/export 실패:")
        traceback.print_exc()

    # 3. visp/ggml arch C++ 생성 (TorchParser 그래프 사용)
    if graph is not None:
        print("[compile_to_c] Generating visp/ggml arch C++...", flush=True)
        files = generate_ggml_code(graph, output_dir, name)
        for ftype, fpath in files.items():
            print(f"  → {ftype}: {fpath}")

        # 4. GGUF 가중치 직렬화 (state_dict → fp16, 텐서명=state_dict 키)
        generate_gguf(model, output_dir, name, name.lower())

        # 5. export 에 ggml 실행 진입점 추가 (python output/<Model>.py → libggml 커널)
        append_ggml_runner(
            os.path.join(output_dir, name + TorchSymbol.SCRIPT_SUFFIX),
            name,
            f"{name}.gguf",
            input_shape,
        )
        print(
            f"  → 실행: python {output_dir}/{name}{TorchSymbol.SCRIPT_SUFFIX} "
            "(ggml/libggml.so 커널)"
        )

    print("=" * 60)
    print("완료!", flush=True)


if __name__ == "__main__":
    main()
