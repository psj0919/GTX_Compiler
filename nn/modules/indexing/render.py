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
"""인덱싱/선택 op 의 ggml/vision.cpp render (head_render.py + vision_ops_render.py 에서 분리).

argmax/argsort/max/topk/gather/index/strided_slice 를 담는다. 인덱스를 생성하거나
인덱스로 원소를 선택하는 op. 전부 **실제 ggml 그래프 op** 으로 emit 한다.
런타임 의미는 ``nn/modules/ggml_backend.py`` host numpy 구현이 기준이다.
"""

from shared.compile.render_api import register_render as _rr, out_shape, ggml_axis
from shared.base import OP as _OP


def _ne_args(shape):
    """torch shape → ggml ne 인자 문자열(역순). 최대 4D."""
    ne = list(reversed([int(d) for d in shape]))
    return ne, ", ".join(str(d) for d in ne)


# ----------------------------------------------------------- 인덱스 생성(정렬/argmax)
@_rr(_OP.ARGMAX)
def render_argmax(node, ctx):
    return ctx.out(node, f"ggml_argmax(m, {ctx.inp(node)})", hint="amax")


@_rr(_OP.ARGSORT)
def render_argsort(node, ctx):
    order = "GGML_SORT_ORDER_DESC" if ctx.attr(node, "descending", False) else "GGML_SORT_ORDER_ASC"
    return ctx.out(node, f"ggml_argsort(m, {ctx.inp(node)}, {order}) /* TODO(ggml): dim=마지막축 가정 */", hint="asort")


@_rr(_OP.MAX)
def render_max(node, ctx):
    # torch max(dim) → (values, indices). ggml_argmax = 인덱스(마지막 축 reduce).
    var = ctx.out(node, f"ggml_argmax(m, {ctx.inp(node)})", hint="max")
    ctx.bind_outputs(node, var)
    return var


@_rr("aten::topk", "topk")
def render_topk(node, ctx):
    sh = out_shape(node)
    k = sh[-1] if sh else 0
    var = ctx.out(node, f"ggml_top_k(m, {ctx.inp(node)}, {k})", hint="topk")
    ctx.bind_outputs(node, var)
    return var


# ----------------------------------------------------------- 마스크 선택(where/masked_fill)
# ggml 에 where 커널이 없다 → 0/1 마스크(비교 op render 의 출력)로 산술 합성한다.
# ±inf 채움값은 곱셈에서 0*inf=NaN 이 되므로 f32 로 안전한 ±1e30 으로 대체한다
# (softmax 마스킹 용도에서 exp() 는 어차피 0 으로 포화).
_BIG = 1e30


def _finite(v, default=0.0):
    v = float(v) if isinstance(v, (int, float)) else default
    if v == float("inf"):
        return _BIG
    if v == float("-inf"):
        return -_BIG
    return v


@_rr(_OP.WHERE)
def render_where(node, ctx):
    # where(c, a, b) = a*c + b*(1-c)
    c, a, b = ctx.inp(node, 0), ctx.inp(node, 1), ctx.inp(node, 2)
    inv = f"ggml_scale_bias(m, {c}, -1.0f, 1.0f)"
    return ctx.out(node, f"ggml_add(m, ggml_mul(m, {a}, {c}), ggml_mul(m, {b}, {inv}))",
                   hint="where")


@_rr(_OP.MASKED_FILL)
def render_masked_fill(node, ctx):
    # masked_fill(x, mask, v) = x*(1-mask) + v*mask
    x, mask = ctx.inp(node, 0), ctx.inp(node, 1)
    v = _finite(ctx.attr(node, "value", 0.0))
    inv = f"ggml_scale_bias(m, {mask}, -1.0f, 1.0f)"
    return ctx.out(node, f"ggml_add(m, ggml_mul(m, {x}, {inv}), ggml_scale(m, {mask}, {v}f))",
                   hint="mfill")


@_rr(_OP.GRID_SAMPLE)
def render_grid_sample(node, ctx):
    # ggml 커널 부재 → codegen 이 .cpp 상단에 주입하는 grid_sample_2d(custom op) 호출.
    mode = str(ctx.attr(node, "mode", "bilinear")).strip("'\"")
    pad = str(ctx.attr(node, "padding_mode", "zeros")).strip("'\"")
    flags = (1 if mode == "nearest" else 0) | (2 if ctx.attr(node, "align_corners", False) else 0)
    note = "" if pad == "zeros" else f"  /* TODO(ggml): padding_mode='{pad}' → zeros 로 근사 */"
    return ctx.out(node,
                   f"grid_sample_2d(m, {ctx.inp(node, 0)}, {ctx.inp(node, 1)}, {flags}){note}",
                   hint="gs")


# ----------------------------------------------------------- 선택(gather/slice)
@_rr("aten::gather", "gather", _OP.INDEX)
def render_gather(node, ctx):
    # ggml_get_rows(a, idx): idx(I32) 행 선택. torch gather/index 의 근사.
    a = ctx.inp(node, 0)
    idx = ctx.inp(node, 1) if len([t for t in node.in_tensors if t is not None]) > 1 else a
    return ctx.out(node, f"ggml_get_rows(m, {a}, {idx})", hint="gat")


def _focus_quad(node, ctx, osh):
    """STRIDED_SLICE 가 YOLO Focus quadrant(마지막 2축 H,W 를 step=2 부분샘플)면 (sw, sh) 반환.

    `ggml_view` 는 **ne0(=W) 축에 step 을 못 준다.** 그래서 `x[..., ::2, ::2]` 를 view 로
    내면 step 이 무시돼 **좌상단 연속 블록**이 잘려 나온다 — 크래시 없이 조용히 틀린다
    (yolox·bytetrack·ocsort·strongsort cos 0.27~0.46). 이 경우만 vision.cpp 의
    `space_to_depth_quad` 로 분해한다.
    """
    starts, dims, steps = (ctx.attr(node, k, None) for k in ("start", "dim", "step"))
    if not (isinstance(starts, (list, tuple)) and isinstance(dims, (list, tuple))
            and isinstance(steps, (list, tuple)) and osh):
        return None
    nd = len(osh)                                      # torch out ndim (NCHW → 4)
    stepped = {int(dims[i]): int(steps[i]) for i in range(len(steps))
               if steps[i] is not None and int(steps[i]) != 1}
    if set(stepped) != {nd - 2, nd - 1} or any(v != 2 for v in stepped.values()):
        return None                                    # 마지막 2축(H,W) step=2 아니면 미대상
    smap = {int(dims[i]): int(starts[i]) for i in range(len(starts))}
    sh, sw = smap.get(nd - 2, 0), smap.get(nd - 1, 0)  # H, W 시작
    return (sw, sh) if sh in (0, 1) and sw in (0, 1) else None


@_rr(_OP.STRIDED_SLICE)
def render_strided_slice(node, ctx):
    # ggml_view_Nd: 정적 출력 ne + 입력 nb(슬라이스는 축 stride 보존) + byte offset.
    # offset = Σ_d begin[d] * a->nb[ggml_axis(d)]  (torch dim d → ggml 축 ndim-1-d).
    sh = out_shape(node)
    a = ctx.inp(node)
    if not sh or len(sh) > 4:
        return ctx.out(node, f"ggml_cont(m, {a}) /* slice {sh} */", hint="sl")
    q = _focus_quad(node, ctx, sh)                     # YOLO Focus step=2 2D 부분샘플
    if q is not None:
        return ctx.out(node, f"space_to_depth_quad(m, {a}, {q[0]}, {q[1]}) /* focus quad */", hint="sl")
    ndim = len(sh)
    ne, ne_args = _ne_args(sh)
    nb = ", ".join(f"{a}->nb[{i}]" for i in range(1, len(ne)))
    nb = (nb + ", ") if nb else ""
    begin = ctx.attr(node, "begin", None)
    dims = ctx.attr(node, "slice_dims", None)
    terms = []
    if isinstance(begin, (list, tuple)):
        dlist = dims if isinstance(dims, (list, tuple)) else list(range(len(begin)))
        for d, b in zip(dlist, begin):
            try:
                bi = int(b)
            except (TypeError, ValueError):
                continue
            if bi > 0:
                g = ggml_axis(int(d), ndim)
                terms.append(f"{bi}*{a}->nb[{g}]")
    off = " + ".join(terms) if terms else "0"
    return ctx.out(node, f"ggml_view_{len(ne)}d(m, {a}, {ne_args}, {nb}{off})", hint="sl")
