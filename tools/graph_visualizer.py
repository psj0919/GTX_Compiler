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


# record-label 안에서 구조/파싱을 깨는 문자. Netron dot.js 는 필드를 '|', attr 를
# '\l', key:value 를 ':' 로 나누므로 이들이 값에 들어가면 안 된다. graphviz record 의
# 구조문자 '{}<>' 도 배제.
_BAD_VAL_CHARS = set('|{}:<>"\\')


def _fmt_val(v):
    """op attr 값 → record 필드 문자열. 표시 불가(Tensor/None/중첩)면 None.

    리스트/튜플은 `(a, b)` 로 → Netron dot.js 가 JSON 배열로 승격한다.
    """
    if v is None:
        return None
    if isinstance(v, bool):        # bool 이 int 보다 먼저 (issubclass)
        return str(v)
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return v if not (set(v) & _BAD_VAL_CHARS) else None
    if isinstance(v, (list, tuple)):
        parts = []
        for x in v:
            if isinstance(x, bool):
                parts.append(str(x))
            elif isinstance(x, (int, float)):
                parts.append(str(x))
            else:
                return None        # Tensor 등 중첩 → attr 전체 스킵
        return "(" + ", ".join(parts) + ")"
    return None                    # Tensor / Operation / 기타


def _param_pairs(node):
    """op-IR 파라미터(weight/bias 등)를 `dtype[shape]` 필드로 수집.

    예: weight → 'float32[64, 7, 7, 3]', bias → 'float32[64]'. Netron 에서 연산자
    parameter 의 dtype·사이즈를 hyperparameter 와 함께 한 번에 확인(요청 p3)."""
    pairs = []
    op = getattr(node, "op", None)
    params = getattr(op, "_params", None) or {}
    for name, tensor in params.items():
        key = str(getattr(name, "value", name))
        tensors = tensor if isinstance(tensor, (list, tuple)) else [tensor]
        for j, t in enumerate(tensors):
            if t is None:
                continue
            data = getattr(t, "data", None)
            sh = getattr(t, "shape", None)
            if sh is None:
                sh = getattr(data, "shape", None)
            if sh is None:
                continue
            dt = getattr(data, "dtype", None)
            k = key if len(tensors) == 1 else f"{key}_{j}"
            val = f"{dt if dt is not None else '?'}[{', '.join(str(int(d)) for d in sh)}]"
            if not (set(k) & _BAD_VAL_CHARS) and not (set(val) & _BAD_VAL_CHARS):
                pairs.append((k, val))
    return pairs


def _attr_pairs(node, *, show_shape: bool = True):
    """op-IR(op._attrs) + param(weight/bias dtype·shape) + out_shape 를 1:1 로 수집."""
    pairs = []
    short = node.name.split("::")[-1]
    if not (set(short) & _BAD_VAL_CHARS):
        pairs.append(("name", short))
    op = getattr(node, "op", None)
    for name, attr in (getattr(op, "_attrs", None) or {}).items():
        key = str(getattr(name, "value", name))
        s = _fmt_val(getattr(attr, "value", None))
        if s is not None and not (set(key) & _BAD_VAL_CHARS):
            pairs.append((key, s))
    pairs.extend(_param_pairs(node))
    if show_shape:
        shs = _out_shapes(node)
        for i, sh in enumerate(shs):
            k = "out_shape" if i == 0 else f"out_shape_{i}"
            pairs.append((k, "(" + ", ".join(str(d) for d in sh) + ")"))
    return pairs


def _record_label(nid: str, optype: str, pairs) -> str:
    """Netron dot.js 가 type+attribute 로 파싱하는 record 라벨.

    형식: `{<nid>|op_code=<optype>\\l|<k>: <v>\\l...}` (dot.js 92-122행 규칙).
    lines[0]==nid, lines[1] 이 'op_code=' 로 시작해야 optype 이 노드 type 으로 승격.
    """
    fields = [nid, f"op_code={optype}\\l"]
    if pairs:
        fields.append("".join(f"{k}: {v}\\l" for k, v in pairs))
    return "{" + "|".join(fields) + "}"


def to_dot(graph, *, rankdir: str = "TB", show_shape: bool = True) -> str:
    """Graph IR → DOT 문자열 (Netron 호환 record 라벨).

    graph.viz_skip_ids(=const-fold 로 baking 되어 codegen 에서 제거되는 노드 id 집합)가
    있으면 그 노드/엣지는 컴파일 그래프와 동일하게 렌더에서 제외한다.
    """
    skip_ids = getattr(graph, "viz_skip_ids", None) or set()
    nodes = [n for n in (getattr(graph, "nodes", []) or []) if id(n) not in skip_ids]
    name2id = {n.name: f"n{i}" for i, n in enumerate(nodes)}

    lines = [
        f'digraph "{_esc(getattr(graph, "name", "graph"))}" {{',
        f"  rankdir={rankdir};",
        # record: Netron 은 label 에서 op/attr 파싱, graphviz 는 필드 렌더. octagon 금지.
        '  node [shape=record, style=filled, fontname="Helvetica", fontsize=10];',
        '  edge [color="#90a4ae"];',
    ]

    for n in nodes:
        nid = name2id[n.name]
        ot = _op_type(n)
        label = _record_label(nid, ot, _attr_pairs(n, show_shape=show_shape))
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
    skip = getattr(graph, "viz_skip_ids", None) or set()
    n = sum(1 for x in (getattr(graph, "nodes", []) or []) if id(x) not in skip)
    print(f"[graph_viz] DOT 작성: {dot_path} ({n} nodes)")
    img = _render(dot_path, output + "." + fmt, fmt)
    if img:
        print(f"[graph_viz] 이미지 렌더: {img}")
    else:
        print(f"[graph_viz] (graphviz 'dot' 미가용 — .dot 만 생성. 렌더: dot -T{fmt} {dot_path} -o {output}.{fmt})")
    return {"dot": dot_path, "image": img}


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
                fold: bool = True, extra_opt: bool = False):
    """g2c 와 동일한 --model 스펙 → Graph IR (TorchParser).

    fold=True(기본): g2c 컴파일과 동일한 그래프 패스를 적용해 **실제로 컴파일되는
    그래프**를 그린다 — (1) Conv-BN fold(BN→conv 흡수), (2) const-fold(입력 무관
    상수 subgraph 를 GGUF 로 baking → codegen 제거; 제거 노드는 viz_skip_ids 로 표시).
    TorchParser 자체의 파싱-타임 최적화(jit constant-fold/DCE)는 항상 포함.
    fold=False 면 raw 파싱 그래프.
    extra_opt=True: compile 경로 밖(양자화/DPU) OptimizeCommander 패스도 적용(_apply_extra_opts).
    """
    import torch
    from shared.compile.pipeline import build_model, fold_conv_bn_graph
    from parse import TorchParser
    from parse.rich_in_out_helper import StandardInputData

    model, name, shape = build_model(model_spec, pth=pth, input_shape=input_shape)
    inputs = torch.randn(*shape).cpu()
    graph = TorchParser()(name, model, StandardInputData((inputs,), {}))
    if fold:
        graph = fold_conv_bn_graph(graph)
        if extra_opt:
            graph = _apply_extra_opts(graph)
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
    ap.add_argument("--no-fold", action="store_true",
                    help="Conv-BN fold 생략(raw 파싱 그래프). 기본은 g2c 와 동일하게 fold")
    ap.add_argument("--extra-opt", action="store_true",
                    help="compile 경로 밖(양자화/DPU) OptimizeCommander 패스도 적용 "
                         "(DecoupleShared/FuseEmbedLnActv/SetNegativeSlope). 기본 off")
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
                        fold=not args.no_fold, extra_opt=args.extra_opt)
    visualize_graph(graph, args.output, fmt=args.format,
                    rankdir=args.rankdir, show_shape=not args.no_shape)


if __name__ == "__main__":
    main()
