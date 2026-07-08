#
# Copyright 2025 Supergate.cc, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""모델 → ONNX 내보내기 (Netron 시각화/배포용). g2c 와 동일한 --model 스펙.

양자화 없이 `qproc.onnx.export_onnx` (= torch.onnx.export) 로 표준 ONNX 를 만든다.
Netron 에서 열면 op 타입·엣지·weight(shape/dtype) 가 모두 표시된다.

  uv run python tools/export_onnx.py --model resnet18 --output output/resnet18
  uv run python tools/export_onnx.py --model "ultralytics.YOLO('yolo11n')" --output output/y11
  uv run python tools/export_onnx.py --model torchvision.models.resnet50 --pth W.pth --output output/r50
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main(argv=None):
    import argparse

    import torch

    from shared.compile.pipeline import build_model
    from qproc.onnx import export_onnx

    ap = argparse.ArgumentParser(
        description="PyTorch 모델 → ONNX (양자화 없이 torch.onnx.export, Netron 시각화용)")
    ap.add_argument("--model", required=True,
                    help="모델 스펙 (resnet18 / torchvision.models.resnet50 / "
                         "\"ultralytics.YOLO('yolov8n')\")")
    ap.add_argument("--output", required=True, help="출력 디렉터리")
    ap.add_argument("--pth", default=None, help="state_dict 가중치(.pth) (선택)")
    ap.add_argument("--input-shape", default=None,
                    help="입력 shape (예: 1,3,224,224). 미지정 시 모델별 기본값")
    ap.add_argument("--name", default=None, help="ONNX 파일 이름 (기본: 모델 클래스명)")
    ap.add_argument("--opset", type=int, default=None, help="ONNX opset (기본: 최신)")
    ap.add_argument("--dynamic-batch", action="store_true", help="배치 축을 동적으로")
    args = ap.parse_args(argv)

    torch.set_num_threads(1)
    shape = None
    if args.input_shape:
        shape = tuple(int(x) for x in args.input_shape.replace(" ", "").split(","))

    model, name, shape = build_model(args.model, pth=args.pth, input_shape=shape)
    name = args.name or name
    x = torch.randn(*shape).cpu()
    print(f"[onnx] model={args.model} → {model._get_name()} (name={name}), input={tuple(shape)}",
          flush=True)

    path = export_onnx(model, x, args.output, model_name=name,
                       input_names=["input"], opset_version=args.opset,
                       dynamic_batch=args.dynamic_batch)
    if path:
        print(f"[onnx] → {path}", flush=True)


if __name__ == "__main__":
    main()
