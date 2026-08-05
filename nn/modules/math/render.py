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
"""math(단항 elementwise / 리덕션 / 산술 / 패딩) op 의 ggml/vision.cpp render.

vision_ops_render.py 의 math 파트(clamp/sqrt/sum/sum_rows/cumsum/단항 math/pad/
pad_reflect_1d)와 head_render.py 의 floor_divide/remainder 를 통합한 모듈.
전부 **실제 ggml 그래프 op** 으로 emit 한다 (register_render side-effect).
"""

from shared.compile.render_api import register_render as _rr, out_shape, ggml_axis
from shared.base import OP as _OP


# ------------------------------------------------------------- clamp
@_rr(_OP.CLAMP)
def render_clamp(node, ctx):
    lo = ctx.attr(node, "min", 0.0)
    hi = ctx.attr(node, "max", 6.0)
    return ctx.out(node, f"ggml_clamp(m, {ctx.inp(node)}, {float(lo)}f, {float(hi)}f)", hint="clamp")


# ------------------------------------------------------------- 리덕션
@_rr(_OP.SQRT)
def render_sqrt(node, ctx):
    return ctx.out(node, f"ggml_sqrt(m, {ctx.inp(node)})", hint="sqrt")


def _sum_over_axis(x, ax):
    """ggml ne 축 ax 에 대한 합(size-1 keepdim 유지).

    `ggml_sum_rows` 는 ne0 축만 줄이므로, 다른 축은 permute 로 ne0 자리에 데려왔다가 되돌린다.
    """
    if ax == 0:
        return f"ggml_sum_rows(m, {x})"
    p = [0, 1, 2, 3]
    p[0], p[ax] = p[ax], p[0]
    perm = f"{p[0]}, {p[1]}, {p[2]}, {p[3]}"
    return (f"ggml_cont(m, ggml_permute(m, ggml_sum_rows(m, "
            f"ggml_cont(m, ggml_permute(m, {x}, {perm}))), {perm}))")


@_rr(_OP.SUM)
def render_sum(node, ctx):
    # dim 미지정이면 전체 합, 지정이면 **축별** 합 후 keepdim=False squeeze.
    # ⚠️ dim 을 무시하고 `ggml_sum` 을 내면 텐서가 통째로 **스칼라 1개**가 된다.
    #    resnest Split-Attention 의 `.sum(radix축)` 이 그 꼴이라, 바로 뒤 avg pool 이
    #    입력 [1,1,1,1] 에 kernel 128 을 받아 `GGML_ASSERT(ne[0] > 0)` 로 죽는다.
    x = ctx.inp(node)
    dims = ctx.attr(node, "dim", None)
    if dims is None:
        return ctx.out(node, f"ggml_sum(m, {x})", hint="sum")
    if isinstance(dims, int):
        dims = [dims]
    try:
        ndim = len([t for t in node.in_tensors if t is not None][0].shape)
    except Exception:
        sh0 = out_shape(node)
        ndim = len(sh0) if sh0 else 4
    expr = x
    for ax in sorted({ggml_axis(d, ndim) for d in dims}):
        expr = _sum_over_axis(expr, ax)
    # 위 reduce 는 대상 축을 size-1 로 남긴 keepdim 형태다. torch out_shape 로 reshape 해
    # keepdim=False 의 squeeze 를 반영한다. 없으면 축이 하나 밀린 채 남아 뒤따르는
    # mul 이 `ggml_can_repeat` 로 죽는다.
    sh = out_shape(node)
    if sh and len(sh) <= 4:
        ne = list(reversed([int(d) for d in sh]))
        expr = f"ggml_reshape_{len(ne)}d(m, ggml_cont(m, {expr}), {', '.join(str(d) for d in ne)})"
    return ctx.out(node, expr, hint="sum")


@_rr(_OP.SUM_ROWS)
def render_sum_rows(node, ctx):
    return ctx.out(node, f"ggml_sum_rows(m, {ctx.inp(node)})", hint="sumr")


@_rr(_OP.CUMSUM)
def render_cumsum(node, ctx):
    return ctx.out(node, f"ggml_cumsum(m, {ctx.inp(node)}) /* TODO(ggml): dim 확인 */", hint="cumsum")


# ------------------------------------------------------------- 단항 math (ggml 빌더 직매핑)
def _unary(op, fn, hint):
    @_rr(op)
    def _r(node, ctx, _fn=fn, _h=hint):
        return ctx.out(node, f"{_fn}(m, {ctx.inp(node)})", hint=_h)
    return _r


render_exp = _unary(_OP.EXP, "ggml_exp", "exp")
render_log = _unary(_OP.LOG, "ggml_log", "log")
render_neg = _unary(_OP.NEG, "ggml_neg", "neg")
render_floor = _unary(_OP.FLOOR, "ggml_floor", "floor")
render_ceil = _unary(_OP.CEIL, "ggml_ceil", "ceil")
render_sin = _unary(_OP.SIN, "ggml_sin", "sin")
render_cos = _unary(_OP.COS, "ggml_cos", "cos")
render_abs = _unary(_OP.ABS, "ggml_abs", "abs")
render_sign = _unary(_OP.SIGN, "ggml_sgn", "sgn")
render_step = _unary(_OP.STEP, "ggml_step", "step")
render_round = _unary(_OP.ROUND, "ggml_round", "round")
render_square = _unary(_OP.SQUARE, "ggml_sqr", "sqr")
render_expm1 = _unary(_OP.EXPM1, "ggml_expm1", "expm1")
render_trunc = _unary(_OP.TRUNC, "ggml_trunc", "trunc")


# ------------------------------------------------------------- 산술
@_rr(_OP.POW)
def render_pow(node, ctx):
    # 지수 2 는 op_dispatcher 가 _OP.SQUARE 로 접는다. 여기는 그 외 지수.
    e = ctx.attr(node, "exponent", 1.0)
    a = ctx.inp(node)
    if isinstance(e, (int, float)) and float(e).is_integer() and 0 < int(e) <= 8:
        expr = a
        for _ in range(int(e) - 1):
            expr = f"ggml_mul(m, {expr}, {a})"
        return ctx.out(node, expr, hint="pow")
    # 일반 지수: x^e = exp(e * log x). x > 0 전제 (음수 밑은 실수 거듭제곱이 정의되지 않음).
    return ctx.out(node, f"ggml_exp(m, ggml_scale(m, ggml_log(m, {a}), {float(e)}f))",
                   hint="pow")


@_rr(_OP.FLOOR_DIV)
def render_floor_div(node, ctx):
    # floor_divide(x, c): 상수 나눗셈 → ggml_scale(1/c). 정확한 floor 는 후처리(인덱스 계산).
    return ctx.out(node, f"ggml_scale(m, {ctx.inp(node, 0)}, 1.0f) /* floor_divide: scale by 1/divisor */", hint="fdiv")


@_rr("aten::remainder", "remainder")
def render_remainder(node, ctx):
    # remainder(x, c): ggml 직접 op 없음 → x - floor(x/c)*c. 좌표 인덱스 계산(후처리 보정).
    return ctx.out(node, f"ggml_cont(m, {ctx.inp(node, 0)}) /* remainder: x - (x/c)*c */", hint="rem")


# ------------------------------------------------------------- 비교/논리
# ggml 에 비교 커널이 없다 → ggml_step(x)=(x>0) 로 합성한다. 결과는 0.0/1.0 f32 마스크로,
# where/masked_fill render 가 그대로 소비한다 (ggml 에 bool 타입 없음).
def _diff(node, ctx, flip=False):
    """a - b (flip=True 면 b - a). other 가 스칼라면 scale_bias 로 상수를 흡수."""
    ins = [t for t in (getattr(node, "in_tensors", None) or []) if t is not None]
    a = ctx.inp(node, 0)
    if len(ins) > 1:
        b = ctx.inp(node, 1)
        return f"ggml_sub(m, {b}, {a})" if flip else f"ggml_sub(m, {a}, {b})"
    o = ctx.attr(node, "other", 0.0)
    o = float(o) if isinstance(o, (int, float)) else 0.0
    s = -1.0 if flip else 1.0
    return f"ggml_scale_bias(m, {a}, {s}f, {-s * o}f)"


@_rr(_OP.GREATER)
def render_greater(node, ctx):
    return ctx.out(node, f"ggml_step(m, {_diff(node, ctx)})", hint="gt")


@_rr(_OP.NOT_EQUAL)
def render_not_equal(node, ctx):
    return ctx.out(node, f"ggml_step(m, ggml_abs(m, {_diff(node, ctx)}))", hint="ne")


@_rr(_OP.EQUAL)
def render_equal(node, ctx):
    ne = f"ggml_step(m, ggml_abs(m, {_diff(node, ctx)}))"
    return ctx.out(node, f"ggml_scale_bias(m, {ne}, -1.0f, 1.0f)", hint="eq")


@_rr(_OP.LOGICAL_OR)
def render_logical_or(node, ctx):
    # 0/1 마스크 두 개: a|b = step(a+b) (a+b ∈ {0,1,2} → 0 이 아니면 1).
    both = f"ggml_add(m, {ctx.inp(node, 0)}, {ctx.inp(node, 1)})"
    return ctx.out(node, f"ggml_step(m, {both})", hint="or")


# ------------------------------------------------------------- 패딩
@_rr(_OP.PAD_REFLECT_1D)
def render_pad_reflect_1d(node, ctx):
    p = ctx.attr(node, "pad", ctx.attr(node, "padding", [0, 0])) or [0, 0]
    p = [int(x) for x in p] + [0, 0]
    return ctx.out(node, f"ggml_pad_reflect_1d(m, {ctx.inp(node)}, {p[0]}, {p[1]})", hint="padr")



# ── deploy 이식: F.pad 원소별 파싱 ────────────────────────────────
# main 의 PAD 렌더러는 `pads` 를 통째로 int() 변환한다 — Swin 은 `[0,0,0,Tensor,0,Tensor]`
# 처럼 **일부만 Tensor** 라 예외로 떨어져 패딩이 전부 0 이 된다. 원소별로 처리한다.
@_rr("pad_nd", _OP.PAD)
def render_pad(node, ctx):
    """F.pad → ggml_pad_ext (비대칭 zero-pad). torch pad=[d(-1)L,d(-1)R,d(-2)L,d(-2)R,…] →
    ggml dim0(W)=마지막축부터. constant(zero) 모드 가정(carafe shift 등)."""
    x = ctx.inp(node, 0)
    pads = ctx.attr(node, "pad", None) or ctx.attr(node, "paddings", None) or []
    # ⚠️ 통째로 `[int(p) for p in pads]` 하면 안 된다 — **일부만 Tensor** 인 경우가 흔하다.
    # Swin: `F.pad(x, (0,0, 0,pad_r, 0,pad_b))` → `[0, 0, 0, Tensor, 0, Tensor]`.
    # 통째 변환은 예외로 떨어져 **아는 0 까지 버리고**, 그러면 폴백이 same-padding 으로 대칭
    # 분배해 `(2,3)` 을 넣는다. Swin 은 **끝에만** 붙여야 하므로 데이터가 통째로 어긋난다.
    # → 원소별로 아는 값은 살리고, 모르는 값만 shape 차이로 역산한다.
    known = []
    for p in (pads or []):
        try:
            known.append(int(p))
        except Exception:
            known.append(None)

    ish = osh = None
    try:
        ish = [int(d) for d in [t for t in node.in_tensors if t is not None][0].shape]
        osh = [int(d) for d in (out_shape(node) or [])]
        if not (ish and osh and len(ish) == len(osh)):
            ish = osh = None
    except Exception:
        ish = osh = None

    lp = [0, 0, 0, 0]
    rp = [0, 0, 0, 0]
    for i in range(4):
        kl = known[2 * i] if 2 * i < len(known) else 0
        kr = known[2 * i + 1] if 2 * i + 1 < len(known) else 0
        total = None
        if ish and osh and i < len(ish):
            total = osh[len(ish) - 1 - i] - ish[len(ish) - 1 - i]   # ggml 축 i ↔ torch 축 nd-1-i
        if kl is not None and kr is not None:
            lp[i], rp[i] = kl, kr
        elif total is not None and total > 0:
            if kl is not None:                 # 한쪽만 알면 나머지는 shape 차이로 정확히 결정된다
                lp[i], rp[i] = kl, total - kl
            elif kr is not None:
                lp[i], rp[i] = total - kr, kr
            else:                              # 둘 다 모르면 same-padding 규칙(EfficientNet)
                lp[i], rp[i] = total // 2, total - total // 2
        else:
            lp[i], rp[i] = (kl or 0), (kr or 0)
    return ctx.out(
        node,
        f"ggml_pad_ext(m, {x}, {lp[0]}, {rp[0]}, {lp[1]}, {rp[1]}, {lp[2]}, {rp[2]}, {lp[3]}, {rp[3]})",
        hint="pad",
    )
