"""graph_visualizer.py — Graph IR 시각화 CLI (graphviz DOT).

PyTorch 모델을 `TorchParser` 로 파싱한 **Graph IR** 을 graphviz DOT 으로 그린다.
DOT 이미터 코어는 `shared/compile/graph_viz.py`(패키지 내부)에 있고, 이 파일은
CLI 래퍼다. `g2c --visualize <path>` 로도 동일 기능을 쓸 수 있다.

사용 (g2c 와 동일한 --model 스펙):
  uv run python tools/graph_visualizer.py --model resnet18 --output output/resnet18_graph
  uv run python tools/graph_visualizer.py --model "ultralytics.YOLO('yolo11n')" --output output/y11 --format svg
  uv run python tools/graph_visualizer.py --model torchvision.models.resnet50 --pth W.pth --output output/r50
  # 트레이서 비교: --tracer dispatch (기본 jit). GTX_DISPATCH_TRACER 토글을 일시 설정.

프로그램 내 사용:
  from shared.compile.graph_viz import visualize_graph
  visualize_graph(graph, "output/foo", fmt="svg")   # 이미 만든 Graph IR 객체
"""
from __future__ import annotations

import os
import sys

# tools/ 관례: 프로젝트 루트를 sys.path 에 주입(shared/parse import 해결).
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# DOT 이미터 코어(패키지 내부, stdlib). 하위호환: 기존 `from tools.graph_visualizer import
# visualize_graph/to_dot` 도 계속 동작하도록 re-export.
from shared.compile.graph_viz import to_dot, visualize_graph  # noqa: E402,F401


def _apply_extra_opts(graph):
    """compile 경로 밖(양자화/DPU 백엔드)의 OptimizeCommander 패스를 opt-in 적용.

    QuantOptimizer 순서를 따름: DecoupleShared → FuseEmbedLnActv → SetNegativeSlope.
    resnet/yolo 처럼 해당 패턴(공유 conv 가중치/embedding-LN-actv/leaky_relu)이 없으면
    무해한 no-op. 양자화 미설정 시 SetNegativeSlope 는 값 변경 없이 경고만.
    """
    try:
        from shared.optimization.commander import OptimizeCommander
        cmd = OptimizeCommander(graph=graph)
        cmd.DecoupleSharedParamsInConv()   # 공유 conv weight/bias → 레이어별 사본 분리
        cmd.FuseEmbedLnActv()              # Embedding→LayerNorm→Sigmoid/Tanh 융합(노드 제거)
        cmd.SetNegativeSlope()             # LeakyReLU alpha → DPU 값(0.1015625), quant 시만
        print("[graph_viz] extra-opt: DecoupleShared/FuseEmbedLnActv/SetNegativeSlope 적용",
              flush=True)
    except Exception as e:
        print(f"[graph_viz] extra-opt 생략 ({type(e).__name__}: {e})", flush=True)
    return graph


def build_graph(model_spec: str, pth: str = None, input_shape=None,
                opt_level: int = 0, extra_opt: bool = False,
                fuse_activation: bool = False):
    """g2c 와 동일한 --model 스펙 → Graph IR (TorchParser).

    opt_level: `--Opt` (0 없음 / 1 Memory 최적화 / 2 OP 최적화 / 3 둘다). g2c 와 **동일한
    공용 함수**(`apply_graph_opts`)를 써서 DOT = 실제 컴파일 그래프로 정합시킨다. 시각화라
    `ensure_bn_affine=False` → **level 0 = raw**(BN·running stats 그대로). const-fold(입력 무관
    상수 baking → viz_skip_ids)는 어떤 최적화라도 켜진 opt_level≥1 에서만 적용.
    extra_opt=True: compile 경로 밖(양자화/DPU) OptimizeCommander 패스도 적용(_apply_extra_opts).
    fuse_activation=True: Conv/Dense/Add→Activation(ReLU 등)을 fused op 으로 표시(추론 최적화 뷰).
    """
    import torch
    from shared.compile.pipeline import build_model, apply_graph_opts
    from parse import TorchParser
    from parse.rich_in_out_helper import StandardInputData

    model, name, shape = build_model(model_spec, pth=pth, input_shape=input_shape)
    inputs = torch.randn(*shape).cpu()
    graph = TorchParser()(name, model, StandardInputData((inputs,), {}))
    # g2c 와 동일한 --Opt 레벨 최적화(시각화라 level 0 = raw).
    graph = apply_graph_opts(graph, level=opt_level)
    if extra_opt:
        graph = _apply_extra_opts(graph)
    if opt_level in (2, 3) or fuse_activation:   # OP 최적화 레벨에 act-fuse 포함
        try:
            from shared.optimization.commander import OptimizeCommander
            n = OptimizeCommander(graph=graph).FuseConvActivation()
            print(f"[graph_viz] Conv+Activation fuse — {n} activation 노드 흡수", flush=True)
        except Exception as e:
            print(f"[graph_viz] activation fuse 생략 ({type(e).__name__}: {e})", flush=True)
    if opt_level >= 1:
        try:
            from shared.compile.const_fold import fold_constants
            baked, skip = fold_constants(graph, (shape[-2], shape[-1]))
            graph.viz_skip_ids = skip
            if skip:
                print(f"[graph_viz] const-fold — baked {len(baked)} tensor, "
                      f"{len(skip)} node 제거(codegen 제외)", flush=True)
        except Exception as e:
            print(f"[graph_viz] const-fold 생략 ({type(e).__name__}: {e})", flush=True)
    return graph


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Graph IR 시각화기 (PyTorch 모델 → TorchParser Graph IR → graphviz DOT)")
    ap.add_argument("--model", required=True,
                    help="모델 스펙 (resnet18 / torchvision.models.resnet50 / \"ultralytics.YOLO('yolov8n')\")")
    ap.add_argument("--output", required=True, help="출력 경로 접두 (확장자 자동: .dot/.<format>)")
    ap.add_argument("--pth", default=None, help="state_dict 가중치(.pth) (선택)")
    ap.add_argument("--input-shape", default=None,
                    help="입력 shape (예: 1,3,224,224). 미지정 시 모델별 기본값")
    ap.add_argument("--format", default="png", choices=["png", "svg", "pdf"],
                    help="이미지 포맷 (기본 png)")
    ap.add_argument("--rankdir", default="TB", choices=["TB", "LR"],
                    help="레이아웃 방향 (TB=위→아래, LR=좌→우)")
    ap.add_argument("--no-shape", action="store_true", help="출력 shape 라벨 생략")
    ap.add_argument("--Opt", type=int, default=3, choices=[0, 1, 2, 3],
                    help="그래프 최적화 수준(g2c 와 동일 함수). 0: 없음(raw) / 1: Memory 최적화"
                         "(DevGraphOptimizer: topology-only) / 2: OP 최적화(OptimizeCommander:"
                         " Conv-BN fold) / 3: 1+2. ≥1 이면 const-fold 도 표시.")
    ap.add_argument("--extra-opt", action="store_true",
                    help="compile 경로 밖(양자화/DPU) OptimizeCommander 패스도 적용 "
                         "(DecoupleShared/FuseEmbedLnActv/SetNegativeSlope). 기본 off")
    ap.add_argument("--fuse-activation", action="store_true",
                    help="Conv/Dense/Add→Activation(ReLU 등)을 fused op 으로 표시(추론 최적화 뷰). "
                         "activation 노드 제거 + producer 에 activation 표식. 기본 off(정확한 그래프)")
    ap.add_argument("--tracer", default=None, choices=["jit", "dispatch"],
                    help="트레이서 선택 (기본 jit). dispatch=GTX_DISPATCH_TRACER=1 일시 설정")
    args = ap.parse_args(argv)

    if args.tracer == "dispatch":
        os.environ["GTX_DISPATCH_TRACER"] = "1"
    elif args.tracer == "jit":
        os.environ.pop("GTX_DISPATCH_TRACER", None)

    shape = None
    if args.input_shape:
        shape = tuple(int(x) for x in args.input_shape.replace(" ", "").split(","))

    graph = build_graph(args.model, pth=args.pth, input_shape=shape,
                        opt_level=args.Opt, extra_opt=args.extra_opt,
                        fuse_activation=args.fuse_activation)
    visualize_graph(graph, args.output, fmt=args.format,
                    rankdir=args.rankdir, show_shape=not args.no_shape)


if __name__ == "__main__":
    main()
