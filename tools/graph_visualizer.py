"""graph_visualizer.py — Graph IR 시각화기 (graphviz DOT).

PyTorch 모델을 `TorchParser` 로 파싱한 **Graph IR** 을 graphviz DOT 으로 그린다.
노드 라벨 = op타입 / 노드명 / 출력 shape, 엣지 = dataflow, op 타입별 색상.

의존성 0: 항상 `<out>.dot`(순수 텍스트)을 쓰고, `dot` CLI(graphviz)나 python
`graphviz` 패키지가 있으면 이미지(`<out>.<format>`)까지 best-effort 렌더한다.

사용 (g2c 와 동일한 --model 스펙):
  uv run python tools/graph_visualizer.py --model resnet18 --output output/resnet18_graph
  uv run python tools/graph_visualizer.py --model "ultralytics.YOLO('yolo11n')" --output output/y11 --format svg
  uv run python tools/graph_visualizer.py --model torchvision.models.resnet50 --pth W.pth --output output/r50
  # 트레이서 비교: --tracer dispatch (기본 jit). GTX_DISPATCH_TRACER 토글을 일시 설정.

프로그램 내 사용:
  from tools.graph_visualizer import visualize_graph
  visualize_graph(graph, "output/foo", fmt="svg")   # 이미 만든 Graph IR 객체
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

# tools/ 관례: 프로젝트 루트를 sys.path 에 주입(shared/parse import 해결).
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# op 타입(소문자) → (채움색, 카테고리). 부분일치(substring)로 매칭.
_PALETTE = [
    (("conv", "dense", "linear", "matmul"), "#bbdefb"),   # 연산(파랑)
    (("batch_norm", "layer_norm", "group_norm", "instance_norm"), "#c8e6c9"),  # 정규화(초록)
    (("relu", "silu", "sigmoid", "gelu", "tanh", "leaky", "softmax", "clamp"), "#fff9c4"),  # 활성(노랑)
    (("add", "sub", "mul", "div", "remainder", "floor_divide", "max"), "#ffe0b2"),  # 산술(주황)
    (("concat", "cat", "stack", "split", "strided_slice", "select", "slice",
      "reshape", "view", "transpose", "permute", "flatten", "unsqueeze",
      "squeeze", "expand", "repeat", "gather", "index"), "#e1bee7"),  # 형상(보라)
    (("pool",), "#b2dfdb"),                                # 풀링(청록)
    (("const", "constant", "full", "arange", "param"), "#cfd8dc"),  # 상수(회색)
    (("input", "return"), "#ffcdd2"),                     # 입출력(빨강)
]
_DEFAULT_COLOR = "#ffffff"


def _color_of(op_type: str) -> str:
    t = (op_type or "").lower()
    for keys, color in _PALETTE:
        if any(k in t for k in keys):
            return color
    return _DEFAULT_COLOR


def _esc(s) -> str:
    """DOT 라벨용 이스케이프."""
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _op_type(node) -> str:
    op = getattr(node, "op", None)
    return getattr(op, "type", None) or getattr(node, "op_type", None) or "?"


def _out_shapes(node):
    shs = []
    for t in getattr(node, "out_tensors", []) or []:
        sh = getattr(t, "shape", None)
        if sh is not None:
            shs.append(list(sh))
    return shs


def to_dot(graph, *, rankdir: str = "TB", show_shape: bool = True) -> str:
    """Graph IR → DOT 문자열."""
    nodes = list(getattr(graph, "nodes", []))
    name2id = {n.name: f"n{i}" for i, n in enumerate(nodes)}

    lines = [
        f'digraph "{_esc(getattr(graph, "name", "graph"))}" {{',
        f"  rankdir={rankdir};",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
        '  edge [color="#90a4ae"];',
    ]

    for n in nodes:
        nid = name2id[n.name]
        ot = _op_type(n)
        # 라벨: op타입 / (짧은) 노드명 / 출력 shape
        short = n.name.split("::")[-1]
        label = f"{ot}\\n{_esc(short)}"
        if show_shape:
            shs = _out_shapes(n)
            if shs:
                label += "\\n" + _esc(", ".join(str(s) for s in shs))
        lines.append(f'  {nid} [label="{label}", fillcolor="{_color_of(ot)}"];')

    seen = set()
    for n in nodes:
        src = name2id[n.name]
        for child in getattr(n, "out_nodes", []) or []:
            dst = name2id.get(child)
            if dst is None or (src, dst) in seen:
                continue
            seen.add((src, dst))
            lines.append(f"  {src} -> {dst};")

    lines.append("}")
    return "\n".join(lines)


def _render(dot_path: str, out_path: str, fmt: str) -> str | None:
    """dot CLI 또는 python graphviz 로 이미지 렌더. 성공 시 경로, 실패 시 None."""
    dot_bin = shutil.which("dot")
    if dot_bin:
        try:
            subprocess.run([dot_bin, f"-T{fmt}", dot_path, "-o", out_path],
                           check=True, capture_output=True)
            return out_path
        except (subprocess.CalledProcessError, OSError):
            pass
    try:
        import graphviz  # noqa: F401
        src = graphviz.Source(open(dot_path).read())
        src.render(outfile=out_path, format=fmt, cleanup=True)
        return out_path
    except Exception:
        return None


def visualize_graph(graph, output: str, *, fmt: str = "png",
                    rankdir: str = "TB", show_shape: bool = True) -> dict:
    """Graph IR 을 DOT(+이미지)로 출력. 반환: {'dot': 경로, 'image': 경로|None}."""
    output = output[:-4] if output.lower().endswith((".dot", ".png", ".svg", ".pdf")) else output
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    dot_path = output + ".dot"
    with open(dot_path, "w") as f:
        f.write(to_dot(graph, rankdir=rankdir, show_shape=show_shape))
    n = len(list(getattr(graph, "nodes", [])))
    print(f"[graph_viz] DOT 작성: {dot_path} ({n} nodes)")
    img = _render(dot_path, output + "." + fmt, fmt)
    if img:
        print(f"[graph_viz] 이미지 렌더: {img}")
    else:
        print(f"[graph_viz] (graphviz 'dot' 미가용 — .dot 만 생성. 렌더: dot -T{fmt} {dot_path} -o {output}.{fmt})")
    return {"dot": dot_path, "image": img}


def build_graph(model_spec: str, pth: str = None, input_shape=None):
    """g2c 와 동일한 --model 스펙 → Graph IR (TorchParser)."""
    import torch
    from shared.compile.pipeline import build_model
    from parse import TorchParser
    from parse.rich_in_out_helper import StandardInputData

    model, name, shape = build_model(model_spec, pth=pth, input_shape=input_shape)
    inputs = torch.randn(*shape).cpu()
    graph = TorchParser()(name, model, StandardInputData((inputs,), {}))
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

    graph = build_graph(args.model, pth=args.pth, input_shape=shape)
    visualize_graph(graph, args.output, fmt=args.format,
                    rankdir=args.rankdir, show_shape=not args.no_shape)


if __name__ == "__main__":
    main()
