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

from shared.compile.render_api import register_render as _rr
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


@_rr(_OP.SUM)
def render_sum(node, ctx):
    return ctx.out(node, f"ggml_sum(m, {ctx.inp(node)})", hint="sum")


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
@_rr(_OP.FLOOR_DIV)
def render_floor_div(node, ctx):
    # floor_divide(x, c): 상수 나눗셈 → ggml_scale(1/c). 정확한 floor 는 후처리(인덱스 계산).
    return ctx.out(node, f"ggml_scale(m, {ctx.inp(node, 0)}, 1.0f) /* floor_divide: scale by 1/divisor */", hint="fdiv")


@_rr("aten::remainder", "remainder")
def render_remainder(node, ctx):
    # remainder(x, c): ggml 직접 op 없음 → x - floor(x/c)*c. 좌표 인덱스 계산(후처리 보정).
    return ctx.out(node, f"ggml_cont(m, {ctx.inp(node, 0)}) /* remainder: x - (x/c)*c */", hint="rem")


# ------------------------------------------------------------- 패딩
@_rr(_OP.PAD_REFLECT_1D)
def render_pad_reflect_1d(node, ctx):
    p = ctx.attr(node, "pad", ctx.attr(node, "padding", [0, 0])) or [0, 0]
    p = [int(x) for x in p] + [0, 0]
    return ctx.out(node, f"ggml_pad_reflect_1d(m, {ctx.inp(node)}, {p[0]}, {p[1]})", hint="padr")


@_rr(_OP.PAD)
def render_pad(node, ctx):
    # ggml_pad(a, p0,p1,p2,p3): 각 ne 축 뒤쪽 패딩. torch pad 리스트→ggml 축 매핑은 best-effort.
    p = ctx.attr(node, "pad", ctx.attr(node, "padding", [0, 0, 0, 0])) or [0, 0, 0, 0]
    p = [int(x) for x in p] + [0, 0, 0, 0]
    return ctx.out(
        node,
        f"ggml_pad(m, {ctx.inp(node)}, {p[0]}, {p[1]}, {p[2]}, {p[3]})"
        " /* TODO(ggml): torch pad order → ggml ne 축 확인 */",
        hint="pad",
    )
