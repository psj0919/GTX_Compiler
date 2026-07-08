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
import re
import shutil
import subprocess
import sys

# tools/ 관례: 프로젝트 루트를 sys.path 에 주입(shared/parse import 해결).
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# op 타입(소문자) → (채움색, 글자색). Netron grapher.css 의 node-item-type 카테고리 색을
# 그대로 써서 ONNX(Netron) 렌더와 동일한 색으로 보이게 한다(부분일치). 어두운 배경엔 흰 글자.
_C_LAYER = ("#335588", "#ffffff")          # layer   rgb(51,85,136)
_C_NORM = ("#335544", "#ffffff")           # normalization rgb(51,85,68)
_C_ACT = ("#702921", "#ffffff")            # activation rgb(112,41,33)
_C_POOL = ("#335533", "#ffffff")           # pool    rgb(51,85,51)
_C_SHAPE = ("#6c4f47", "#ffffff")          # shape   rgb(108,79,71)
_C_CONST = ("#eeeeee", "#000000")          # constant #eee (검정 글자)
_C_DATA = ("#555555", "#ffffff")           # data    rgb(85,85,85)
_PALETTE = [
    (("conv", "dense", "linear", "matmul", "gemm"), _C_LAYER),
    (("batch_norm", "layer_norm", "group_norm", "instance_norm"), _C_NORM),
    (("relu", "silu", "sigmoid", "gelu", "tanh", "leaky", "softmax", "clamp", "clip", "hardswish"), _C_ACT),
    (("pool",), _C_POOL),
    (("concat", "cat", "stack", "split", "strided_slice", "select", "slice",
      "reshape", "view", "transpose", "permute", "flatten", "unsqueeze",
      "squeeze", "expand", "repeat", "gather", "index"), _C_SHAPE),
    (("const", "constant", "full", "arange", "param"), _C_CONST),
    (("input", "return"), _C_DATA),
]
# Netron 기본 node-item-type (#333, 흰 글자) — 카테고리 없는 op(add/sub/mul 등)
_DEFAULT_COLOR = ("#333333", "#ffffff")

# GTX 배포 dtype. GTX Architecture 는 전면 fp16 으로 구동 — weight/bias(GGUF 는
# `.astype(np.float16)`, pipeline.py:293) 와 activation 모두 fp16. torch IR 의 f32 가
# 아니라 이 값이 실제 배포 dtype. (양자화(--quantize)는 g2c 옵션이라 이 뷰어엔 없음.)
_DEPLOY_DTYPE = "float16"


def _color_of(op_type: str):
    """op 타입 → (fillcolor, fontcolor). Netron 카테고리 색과 정합."""
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


def _op_upper(node) -> str:
    """op 타입을 생성 .py(`nn.Module('CONV2D')`)와 동일한 대문자 이름으로.

    예: conv2d→CONV2D, elemwise_add→ELEMWISE_ADD, aten::topk→ATEN_TOPK."""
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(_op_type(node))).strip("_").upper()
    return s or "OP"


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


# 파라미터 라벨을 ONNX/Netron 표기로: 일반 conv/linear 은 W/B, BatchNorm 은
# scale/B/input_mean/input_var (Netron 이 BN 노드에 쓰는 이름).
_PARAM_LABEL = {"weight": "W", "bias": "B",
                "running_mean": "input_mean", "running_var": "input_var"}
_PARAM_LABEL_BN = {"weight": "scale", "bias": "B",
                   "running_mean": "input_mean", "running_var": "input_var"}


# conv weight 는 그래프 내부 레이아웃([OC,KH,KW,IC])으로 저장 → ONNX/torch 표기(OIHW)로
# 재정렬해 표시(generate_gguf 의 GGUF transpose 와 동일 규칙). depthwise 는 [1,KH,KW,C]→[C,1,KH,KW].
_CONV_REG = {"conv2d", "conv1d", "conv3d"}
_CONV_DW = {"depthwise_conv2d", "depthwise_conv1d", "depthwise_conv3d"}


def _param_pairs(node):
    """op-IR 파라미터를 ONNX/Netron 표기(`W ⟨64×3×7×7⟩`)로 수집.

    Netron 노드 박스가 초기자를 `W ⟨shape⟩` 로 보여주는 것과 동일한 키·형식.
    running_mean/var 가 있으면 BatchNorm 으로 보고 scale/B/input_mean/input_var 로."""
    op = getattr(node, "op", None)
    params = getattr(op, "_params", None) or {}
    keys = {str(getattr(n, "value", n)) for n in params}
    label_map = _PARAM_LABEL_BN if ("running_mean" in keys or "running_var" in keys) else _PARAM_LABEL
    ot = str(_op_type(node)).lower()
    pairs = []
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
            dims = [int(d) for d in sh]
            if key == "weight" and len(dims) == 4:      # ONNX OIHW 표기로 재정렬
                if ot in _CONV_REG:
                    dims = [dims[0], dims[3], dims[1], dims[2]]   # OHWI → OIHW
                elif ot in _CONV_DW:
                    dims = [dims[3], dims[0], dims[1], dims[2]]   # [1,KH,KW,C] → [C,1,KH,KW]
            base = label_map.get(key, key)
            k = base if len(tensors) == 1 else f"{base}_{j}"
            val = "⟨" + "×".join(str(d) for d in dims) + "⟩"    # Netron 표기
            if not (set(k) & _BAD_VAL_CHARS) and not (set(val) & _BAD_VAL_CHARS):
                pairs.append((k, val))
    return pairs


def _shape_str(sh):
    """텐서 shape → 'd0, d1, ...'. 심볼릭/None 차원도 문자로 관용 처리."""
    out = []
    for d in sh:
        try:
            out.append(str(int(d)))
        except (TypeError, ValueError):
            out.append(str(d))
    return ", ".join(out)


def _tensor_desc(t, show_shape: bool = True):
    """in/out 텐서 → '[name ]dtype[shape]' 필드값. 표시할 게 없으면 None.

    GTX 는 전면 fp16 배포라 weight/bias·activation 모두 배포 dtype(f16)으로 표시.
    """
    if t is None:
        return None
    dt = _DEPLOY_DTYPE
    nm = str(getattr(t, "name", "") or "")
    body = str(dt) if dt is not None else "?"
    if show_shape:
        sh = getattr(t, "shape", None)
        if sh is not None:
            body += "[" + _shape_str(sh) + "]"
    if body == "?" and not nm:            # dtype·shape·name 전부 없음 → 스킵
        return None
    val = f"{nm} {body}" if nm and not (set(nm) & _BAD_VAL_CHARS) else body
    return val if not (set(val) & _BAD_VAL_CHARS) else None


def _io_pairs(node, *, show_shape: bool = True):
    """in/out 텐서의 이름·dtype(·shape)를 in{i}/out{i} 필드로 수집(요청 p3).

    parameter(weight/bias)는 `_param_pairs`, activation in/out 은 여기서."""
    pairs = []
    for i, t in enumerate(getattr(node, "in_tensors", None) or []):
        v = _tensor_desc(t, show_shape=show_shape)
        if v is not None:
            pairs.append((f"in{i}", v))
    for i, t in enumerate(getattr(node, "out_tensors", None) or []):
        v = _tensor_desc(t, show_shape=show_shape)
        if v is not None:
            pairs.append((f"out{i}", v))
    return pairs


# op attr 키를 ONNX 이름으로. ONNX 노드가 안 쓰는 컴파일러 전용 attr 은 표시 생략.
_ATTR_RENAME = {"kernel": "kernel_shape", "stride": "strides",
                "pad": "pads", "dilation": "dilations"}
_ATTR_DROP = {"pad_mode", "bias_term", "in_dim", "out_dim", "global"}


def _attr_pairs(node, *, show_shape: bool = True):
    """ONNX 노드 박스처럼: op attr(ONNX 이름) + 파라미터(W/B ⟨shape⟩)만.

    노드명/in-out 텐서/out_shape 등 컴파일러 전용 필드는 ONNX 룩에 맞춰 생략."""
    pairs = []
    op = getattr(node, "op", None)
    for name, attr in (getattr(op, "_attrs", None) or {}).items():
        key = str(getattr(name, "value", name))
        if key in _ATTR_DROP:
            continue
        s = _fmt_val(getattr(attr, "value", None))
        if s is not None and not (set(key) & _BAD_VAL_CHARS):
            pairs.append((_ATTR_RENAME.get(key, key), s))
    # Conv+Activation fuse 로 흡수된 활성화 표식(op → type 명).
    act = getattr(node, "fused_activation", None)
    if act is not None:
        pairs.append(("activation", str(getattr(act, "type", act))))
    pairs.extend(_param_pairs(node))
    return pairs


def _h(s) -> str:
    """graphviz HTML-like label 이스케이프(&<>). ⟨⟩× 는 유니코드라 그대로 렌더."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _html_label(header: str, pairs, fill: str, font: str) -> str:
    """Netron 스타일 HTML-테이블 라벨: 색 헤더(op) + 흰 배경 컨텐츠(검정 글자).

    graphviz `shape=plaintext` 에서 렌더. record 와 달리 op_code 줄이 없고, 헤더만
    카테고리 색으로 칠해 Netron/ONNX 노드 룩과 동일하게 보인다."""
    head = (f'<tr><td bgcolor="{fill}"><font color="{font}"><b>{_h(header)}</b>'
            f'</font></td></tr>')
    rows = [head]
    if pairs:
        body = "".join(f"{_h(k)}: {_h(v)}<br align=\"left\"/>" for k, v in pairs)
        rows.append(f'<tr><td bgcolor="white" align="left"><font color="black">'
                    f'{body}</font></td></tr>')
    return ('<table border="0" cellborder="1" cellspacing="0" cellpadding="4">'
            + "".join(rows) + '</table>')


def to_dot(graph, *, rankdir: str = "TB", show_shape: bool = True) -> str:
    """Graph IR → DOT 문자열 (Netron 호환 record 라벨).

    graph.viz_skip_ids(=const-fold 로 baking 되어 codegen 에서 제거되는 노드 id 집합)가
    있으면 그 노드/엣지는 컴파일 그래프와 동일하게 렌더에서 제외한다.
    """
    skip_ids = getattr(graph, "viz_skip_ids", None) or set()
    nodes = [n for n in (getattr(graph, "nodes", []) or []) if id(n) not in skip_ids]
    # 노드 id = 생성 .py 와 동일한 대문자 op 이름 + 인덱스(유일). record 첫 칸에 op 가
    # 드러나 가독성 ↑, dot.js 의 `id == lines[0]` 조건도 유지(op_code 타입 승격 정상).
    name2id = {n.name: f"{_op_upper(n)}_{i}" for i, n in enumerate(nodes)}

    lines = [
        f'digraph "{_esc(getattr(graph, "name", "graph"))}" {{',
        f"  rankdir={rankdir};",
        # HTML-테이블 라벨(색 헤더+흰 컨텐츠) → shape=plaintext. Netron/ONNX 노드 룩.
        '  node [shape=plaintext, fontname="Helvetica", fontsize=10];',
        '  edge [color="#90a4ae"];',
    ]

    for n in nodes:
        nid = name2id[n.name]
        ot = _op_type(n)
        fill, font = _color_of(ot)
        # 헤더 = 노드 id(CONV2D_1). op_code 별도 줄 없음, 컨텐츠는 흰 배경 검정 글자.
        label = _html_label(nid, _attr_pairs(n, show_shape=show_shape), fill, font)
        lines.append(f'  {nid} [label=<{label}>];')

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
