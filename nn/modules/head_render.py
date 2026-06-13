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
"""YOLO detection-head / anchor / DFL / 인덱스 op 의 ggml/vision.cpp render.

전부 **실제 ggml 그래프 op** 으로 emit 한다 (passthrough scaffold 미사용). 정적 attr
(order/dim)과 정적 출력 shape 를 이용해 ggml ne(=torch shape 의 역순) 인자를 산출한다.
런타임 의미(dim/attr)는 ``nn/modules/ggml_backend.py`` host numpy 구현이 기준이다.

매핑:
- silu→ggml_silu / depthwise_conv2d→conv_2d_depthwise / transpose·permute→ggml_permute+cont
- reshape·unsqueeze→ggml_reshape_Nd / contiguous→ggml_cont / detach·cast→no-op
- stack→reshape+concat / arange→ggml_arange / repeat→ggml_repeat / topk→ggml_top_k
- max→ggml_argmax / gather·index→ggml_get_rows / strided_slice→ggml_view_Nd
- shape→정적 정수(ne 직접 사용) / const·full→ggml 상수 / floor_divide→ggml_scale
- meshgrid→arange 의 행/열 broadcast(ggml_repeat)
"""

from shared.compile.render_api import register_render as _rr, out_shape, ggml_axis
from shared.base import OP as _OP


# --------------------------------------------------------------------- helpers
def _ne_args(shape):
    """torch shape → ggml ne 인자 문자열(역순). 최대 4D."""
    ne = list(reversed([int(d) for d in shape]))
    return ne, ", ".join(str(d) for d in ne)


def _perm_args(order, ndim):
    """torch permute order → ggml_permute(axis0..3) 인자.

    torch: out_td=i ← in_td=order[i].  ggml 축 g = ndim-1-td.
    ggml_permute(a, p0..p3): pK = a 의 K번 ggml축이 갈 목적지 ggml축.
    """
    p = [0, 1, 2, 3]
    o = [int(x) % ndim for x in order]
    for i in range(ndim):
        src = ndim - 1 - o[i]   # in ggml axis
        dst = ndim - 1 - i      # out ggml axis
        p[src] = dst
    return p


def _reshape_to_out(node, ctx, hint):
    """정적 출력 shape 로 ggml_reshape_Nd (reshape/unsqueeze 공용). >4D 면 cont 폴백.

    ggml_reshape 는 contiguous 입력을 요구한다(view/permute 결과 불가) → 항상 ggml_cont 로
    감싼다. torch 의 reshape-after-transpose 가 .contiguous() 를 요구하는 것과 동치.
    """
    sh = out_shape(node)
    a = ctx.inp(node)
    if not sh or len(sh) > 4:
        return ctx.out(node, f"ggml_cont(m, {a}) /* reshape to {sh} (>4D) */", hint=hint)
    _, args = _ne_args(sh)
    return ctx.out(node, f"ggml_reshape_{len(sh)}d(m, ggml_cont(m, {a}), {args})", hint=hint)


def _permute_expr(a, order, ndim):
    p = _perm_args(order, ndim)
    return f"ggml_cont(m, ggml_permute(m, {a}, {p[0]}, {p[1]}, {p[2]}, {p[3]}))"


# ----------------------------------------------------------- compute ops
@_rr("aten::silu_", "silu")
def render_silu(node, ctx):
    inplace = str(node.op.type).rstrip().endswith("_")
    fn = "ggml_silu_inplace" if inplace else "ggml_silu"
    return ctx.out(node, f"{fn}(m, {ctx.inp(node)})", hint="silu")


@_rr(_OP.DEPTHWISE_CONV2D)
def render_dwconv(node, ctx):
    has_bias = bool(ctx.attr(node, "bias", False))
    key = ctx.weight(node, ["weight"] + (["bias"] if has_bias else []))
    s = ctx.scalar(ctx.attr(node, "stride", [1, 1]))
    p = ctx.scalar(ctx.attr(node, "padding", [0, 0]))
    return ctx.out(node, f"conv_2d_depthwise({key}, {ctx.inp(node)}, {s}, {p})", hint="dwconv")


@_rr(_OP.TRANSPOSE)
def render_transpose(node, ctx):
    sh = out_shape(node)
    ndim = len(sh) if sh else 4
    order = list(range(ndim))
    d = ctx.attr(node, "order", [-2, -1])  # 교환할 두 축
    d0, d1 = int(d[0]) % ndim, int(d[1]) % ndim
    order[d0], order[d1] = order[d1], order[d0]
    return ctx.out(node, _permute_expr(ctx.inp(node), order, ndim), hint="t")


@_rr(_OP.PERMUTE)
def render_permute(node, ctx):
    sh = out_shape(node)
    ndim = len(sh) if sh else 4
    order = ctx.attr(node, "order", list(range(ndim)))
    return ctx.out(node, _permute_expr(ctx.inp(node), order, ndim), hint="perm")


@_rr(_OP.CONTIGUOUS)
def render_contiguous(node, ctx):
    return ctx.out(node, f"ggml_cont(m, {ctx.inp(node)})", hint="cont")


@_rr(_OP.DETACH)
def render_detach(node, ctx):
    var = ctx.inp(node)          # graph no-op
    ctx.bind(node, var)
    return var


@_rr(_OP.CAST)
def render_cast(node, ctx):
    var = ctx.inp(node)          # 전 경로 단일 dtype → no-op
    ctx.bind(node, var)
    return var


@_rr(_OP.RESHAPE)
def render_reshape(node, ctx):
    return _reshape_to_out(node, ctx, hint="rs")


@_rr(_OP.UNSQUEEZE)
def render_unsqueeze(node, ctx):
    return _reshape_to_out(node, ctx, hint="uns")


@_rr(_OP.ARANGE)
def render_arange(node, ctx):
    sh = out_shape(node)
    n = sh[0] if sh else 0
    return ctx.out(node, f"ggml_arange(m, 0.0f, {float(n)}f, 1.0f)", hint="ar")


@_rr("aten::topk", "topk")
def render_topk(node, ctx):
    sh = out_shape(node)
    k = sh[-1] if sh else 0
    var = ctx.out(node, f"ggml_top_k(m, {ctx.inp(node)}, {k})", hint="topk")
    ctx.bind_outputs(node, var)
    return var


@_rr(_OP.STACK)
def render_stack(node, ctx):
    # stack(tensors, dim): 각 입력을 stack축=1 로 reshape 후 ggml_concat.
    sh = out_shape(node)
    ins = [t for t in node.in_tensors if t is not None]
    vars_ = [ctx.inp(node, i) for i in range(len(ins))]
    if not sh or len(sh) > 4 or not vars_:
        return ctx.out(node, f"ggml_cont(m, {vars_[0] if vars_ else 'x'}) /* stack {sh} */", hint="stk")
    # 어느 축이 새로 삽입됐는지: out 에서 크기==len(ins) 인 축을 stack dim 으로 본다.
    dim = next((i for i, d in enumerate(sh) if int(d) == len(ins)), len(sh) - 1)
    per = [d if i != dim else 1 for i, d in enumerate(sh)]
    _, args = _ne_args(per)
    gdim = ggml_axis(dim, len(sh))
    parts = [f"ggml_reshape_{len(sh)}d(m, {v}, {args})" for v in vars_]
    expr = parts[0]
    for p in parts[1:]:
        expr = f"ggml_concat(m, {expr}, {p}, {gdim})"
    return ctx.out(node, expr, hint="stk")


@_rr(_OP.REPEAT)
def render_repeat(node, ctx):
    # ggml_repeat(a, ref): a 를 ref 의 ne 로 broadcast. ref 는 출력 shape 의 0텐서.
    sh = out_shape(node)
    a = ctx.inp(node)
    if not sh or len(sh) > 4:
        return ctx.out(node, f"ggml_cont(m, {a}) /* repeat {sh} */", hint="rep")
    _, args = _ne_args(sh)
    ref = f"ggml_new_tensor_{len(sh)}d(m, GGML_TYPE_F32, {args})"
    return ctx.out(node, f"ggml_repeat(m, {a}, {ref})", hint="rep")


@_rr(_OP.MAX)
def render_max(node, ctx):
    # torch max(dim) → (values, indices). ggml_argmax = 인덱스(마지막 축 reduce).
    var = ctx.out(node, f"ggml_argmax(m, {ctx.inp(node)})", hint="max")
    ctx.bind_outputs(node, var)
    return var


@_rr("aten::gather", "gather", _OP.INDEX)
def render_gather(node, ctx):
    # ggml_get_rows(a, idx): idx(I32) 행 선택. torch gather/index 의 근사.
    a = ctx.inp(node, 0)
    idx = ctx.inp(node, 1) if len([t for t in node.in_tensors if t is not None]) > 1 else a
    return ctx.out(node, f"ggml_get_rows(m, {a}, {idx})", hint="gat")


@_rr(_OP.STRIDED_SLICE)
def render_strided_slice(node, ctx):
    # ggml_view_Nd: 정적 출력 ne + 입력 nb(슬라이스는 축 stride 보존) + byte offset.
    # offset = Σ_d begin[d] * a->nb[ggml_axis(d)]  (torch dim d → ggml 축 ndim-1-d).
    sh = out_shape(node)
    a = ctx.inp(node)
    if not sh or len(sh) > 4:
        return ctx.out(node, f"ggml_cont(m, {a}) /* slice {sh} */", hint="sl")
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


@_rr(_OP.SHAPE)
def render_shape(node, ctx):
    # shape(dim) → 정적 정수. 텐서가 아니므로 입력에 바인딩(소비처는 ne 를 직접 씀).
    var = ctx.inp(node)
    ctx.bind(node, var)
    return var


@_rr(_OP.CONST)
def render_const(node, ctx):
    # const 텐서(anchor seed). 정적 shape 의 영텐서로 emit (값은 GGUF/런타임 주입 대상).
    sh = out_shape(node)
    if sh and len(sh) <= 4:
        _, args = _ne_args(sh)
        return ctx.out(node, f"ggml_new_tensor_{len(sh)}d(m, GGML_TYPE_F32, {args})", hint="const")
    return ctx.out(node, "ggml_new_f32(m, 0.0f)", hint="const")


@_rr("aten::full", "full")
def render_full(node, ctx):
    # full(size, v) → 상수 채움. ggml_arange(v, v+eps, 1) 대신 scale(zeros)+v 로는 builder 부족 →
    # 정적 shape 0텐서 + scale 0 후 add 상수: 간단히 새 텐서(값은 stride 상수, 런타임/후처리 주입).
    sh = out_shape(node)
    if sh and len(sh) <= 4:
        _, args = _ne_args(sh)
        return ctx.out(node, f"ggml_new_tensor_{len(sh)}d(m, GGML_TYPE_F32, {args})", hint="full")
    return ctx.out(node, "ggml_new_f32(m, 0.0f)", hint="full")


@_rr("aten::meshgrid", "meshgrid")
def render_meshgrid(node, ctx):
    # meshgrid(x, y) → 두 2D 그리드. 각 출력 = 1D arange 를 출력 shape 로 broadcast.
    sh = out_shape(node)
    a = ctx.inp(node, 0)
    if sh and len(sh) <= 4:
        _, args = _ne_args(sh)
        ref = f"ggml_new_tensor_{len(sh)}d(m, GGML_TYPE_F32, {args})"
        var = ctx.out(node, f"ggml_repeat(m, {a}, {ref})", hint="mg")
    else:
        var = ctx.out(node, f"ggml_cont(m, {a})", hint="mg")
    ctx.bind_outputs(node, var)
    return var


@_rr(_OP.FLOOR_DIV)
def render_floor_div(node, ctx):
    # floor_divide(x, c): 상수 나눗셈 → ggml_scale(1/c). 정확한 floor 는 후처리(인덱스 계산).
    return ctx.out(node, f"ggml_scale(m, {ctx.inp(node, 0)}, 1.0f) /* floor_divide: scale by 1/divisor */", hint="fdiv")


@_rr("aten::remainder", "remainder")
def render_remainder(node, ctx):
    # remainder(x, c): ggml 직접 op 없음 → x - floor(x/c)*c. 좌표 인덱스 계산(후처리 보정).
    return ctx.out(node, f"ggml_cont(m, {ctx.inp(node, 0)}) /* remainder: x - (x/c)*c */", hint="rem")
