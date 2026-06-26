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
"""비전 모델 활성화/정규화/math op 의 ggml/vision.cpp render.

backbone 외(transformer/세그멘테이션 등) 비전 모델에 등장하는 op 들을 실제 ggml 그래프
op 으로 매핑한다. 학습 전용 op(opt_step/cross_entropy/*_back)은 제외 — 추론 그래프에
나오지 않으므로 추가하지 않는다.

- 활성화: gelu→ggml_gelu / tanh→ggml_tanh / leaky_relu→ggml_leaky_relu /
  relu6→ggml_clamp(0,6) / clamp→ggml_clamp / log_softmax→ggml_log(soft_max)
- 정규화: layer_norm→layer_norm(m[k]) / group_norm→ggml_group_norm+affine /
  instance_norm→ggml_norm
- math: sqrt→ggml_sqrt / sum→ggml_sum / mean→ggml_mean
"""

from shared.compile.render_api import register_render as _rr
from shared.base import OP as _OP


# ------------------------------------------------------------- 활성화
@_rr(_OP.GELU)
def render_gelu(node, ctx):
    # approximate: 'none'(erf, 정확) / 'tanh'(기본) / 'quick'(sigmoid 근사)
    approx = str(ctx.attr(node, "approximate", "tanh")).lower()
    fn = {"none": "ggml_gelu_erf", "quick": "ggml_gelu_quick"}.get(approx, "ggml_gelu")
    return ctx.out(node, f"{fn}(m, {ctx.inp(node)})", hint="gelu")


@_rr(_OP.TANH)
def render_tanh(node, ctx):
    return ctx.out(node, f"ggml_tanh(m, {ctx.inp(node)})", hint="tanh")


@_rr(_OP.LEAKY_RELU)
def render_leaky_relu(node, ctx):
    slope = ctx.attr(node, "negative_slope", None)
    if slope is None:
        slope = ctx.attr(node, "alpha", 0.01)
    return ctx.out(node, f"ggml_leaky_relu(m, {ctx.inp(node)}, {float(slope)}f, false)", hint="lrelu")


@_rr(_OP.CLAMP)
def render_clamp(node, ctx):
    lo = ctx.attr(node, "min", 0.0)
    hi = ctx.attr(node, "max", 6.0)
    return ctx.out(node, f"ggml_clamp(m, {ctx.inp(node)}, {float(lo)}f, {float(hi)}f)", hint="clamp")


@_rr(_OP.RELU6)
def render_relu6(node, ctx):
    return ctx.out(node, f"ggml_clamp(m, {ctx.inp(node)}, 0.0f, 6.0f)", hint="relu6")


@_rr(_OP.LOG_SOFTMAX)
def render_log_softmax(node, ctx):
    return ctx.out(node, f"ggml_log(m, ggml_soft_max(m, {ctx.inp(node)}))", hint="lsm")


# ------------------------------------------------------------- 정규화
@_rr(_OP.LAYER_NORM)
def render_layer_norm(node, ctx):
    # visp wrapper: layer_norm(m["k"], x) = ggml_norm + weight/bias affine (GGUF).
    key = ctx.weight(node, ["weight", "bias"])
    return ctx.out(node, f"layer_norm({key}, {ctx.inp(node)})", hint="ln")


@_rr(_OP.GROUP_NORM)
def render_group_norm(node, ctx):
    groups = int(ctx.attr(node, "num_groups", ctx.attr(node, "groups", 32)) or 32)
    eps = float(ctx.attr(node, "eps", 1e-5))
    key = ctx.weight(node, ["weight", "bias"])
    a = ctx.inp(node)
    # ggml_group_norm 후 per-channel affine(weight/bias)은 GGUF 텐서로 후처리(TODO).
    return ctx.out(
        node,
        f"ggml_group_norm(m, {a}, {groups}, {eps}f)"
        f" /* TODO(ggml): affine weight/bias from {key} */",
        hint="gn",
    )


@_rr(_OP.INSTANCE_NORM)
def render_instance_norm(node, ctx):
    eps = float(ctx.attr(node, "eps", 1e-5))
    return ctx.out(node, f"ggml_norm(m, {ctx.inp(node)}, {eps}f)", hint="in")


# ------------------------------------------------------------- math 리덕션
@_rr(_OP.SQRT)
def render_sqrt(node, ctx):
    return ctx.out(node, f"ggml_sqrt(m, {ctx.inp(node)})", hint="sqrt")


@_rr(_OP.SUM)
def render_sum(node, ctx):
    return ctx.out(node, f"ggml_sum(m, {ctx.inp(node)})", hint="sum")


def _mean_in_shape(node, i=0):
    ins = [t for t in (getattr(node, "in_tensors", None) or []) if t is not None]
    if i < len(ins):
        try:
            return [int(x) for x in ins[i].shape]
        except Exception:
            return None
    return None


@_rr(_OP.MEAN)
def render_mean(node, ctx):
    # ggml_mean 은 ne0(행)만 평균 → torch dim 의미와 다르면 틀린다.
    # mean([2,3]) (= [B,C,H,W] 공간평균 = global avg pool) 은 ne0,ne1 둘 다 reduce 필요 →
    # ggml AVG pool 로 [1,1,C,B] 후 keepdim 에 맞춰 reshape. (ShuffleNet classifier 전 GAP)
    dims = ctx.attr(node, "dim", None)
    insh = _mean_in_shape(node)
    a = ctx.inp(node)
    if dims is not None and insh and len(insh) == 4:
        dd = sorted(int(x) % 4 for x in (dims if isinstance(dims, (list, tuple)) else [dims]))
        if dd == [2, 3]:
            B, C, H, W = insh
            g = ctx.new_var("gap")
            ctx.line(f"    tensor {g} = ggml_pool_2d(m, {a}, GGML_OP_POOL_AVG, "
                     f"{W}, {H}, {W}, {H}, 0, 0);")          # → ne=[1,1,C,B]
            if ctx.attr(node, "keepdim", False):
                return ctx.out(node, f"ggml_cont(m, {g})", hint="gap")
            return ctx.out(node, f"ggml_reshape_2d(m, ggml_cont(m, {g}), {C}, {B})", hint="gap")
    return ctx.out(node, f"ggml_mean(m, {a})", hint="mean")


# ------------------------------------------------------------- 추가 unary
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
render_elu = _unary(_OP.ELU, "ggml_elu", "elu")
render_softplus = _unary(_OP.SOFTPLUS, "ggml_softplus", "softplus")
render_hsigmoid = _unary(_OP.HSIGMOID, "ggml_hardsigmoid", "hsig")
render_hswish = _unary(_OP.HSWISH, "ggml_hardswish", "hsw")

# B그룹 단일 unary math (ggml 빌더 직매핑)
render_sin = _unary(_OP.SIN, "ggml_sin", "sin")
render_cos = _unary(_OP.COS, "ggml_cos", "cos")
render_abs = _unary(_OP.ABS, "ggml_abs", "abs")
render_sign = _unary(_OP.SIGN, "ggml_sgn", "sgn")
render_step = _unary(_OP.STEP, "ggml_step", "step")
render_round = _unary(_OP.ROUND, "ggml_round", "round")
render_square = _unary(_OP.SQUARE, "ggml_sqr", "sqr")
render_expm1 = _unary(_OP.EXPM1, "ggml_expm1", "expm1")
render_trunc = _unary(_OP.TRUNC, "ggml_trunc", "trunc")
render_xielu = _unary(_OP.XIELU, "ggml_xielu", "xielu")


# ------------------------------------------------------------- B그룹 구조적 op
@_rr(_OP.SUM_ROWS)
def render_sum_rows(node, ctx):
    return ctx.out(node, f"ggml_sum_rows(m, {ctx.inp(node)})", hint="sumr")


@_rr(_OP.CUMSUM)
def render_cumsum(node, ctx):
    return ctx.out(node, f"ggml_cumsum(m, {ctx.inp(node)}) /* TODO(ggml): dim 확인 */", hint="cumsum")


@_rr(_OP.ARGSORT)
def render_argsort(node, ctx):
    order = "GGML_SORT_ORDER_DESC" if ctx.attr(node, "descending", False) else "GGML_SORT_ORDER_ASC"
    return ctx.out(node, f"ggml_argsort(m, {ctx.inp(node)}, {order}) /* TODO(ggml): dim=마지막축 가정 */", hint="asort")


def _pool1d(op, kind):
    @_rr(op)
    def _r(node, ctx, _k=kind):
        k = ctx.scalar(ctx.attr(node, "kernel_size", [2]))
        s = ctx.scalar(ctx.attr(node, "stride", [k]))
        p = ctx.scalar(ctx.attr(node, "padding", [0]))
        return ctx.out(node, f"ggml_pool_1d(m, {ctx.inp(node)}, {_k}, {k}, {s}, {p})", hint="pool1d")
    return _r


render_maxpool1d = _pool1d(_OP.MAX_POOL1D, "GGML_OP_POOL_MAX")
render_avgpool1d = _pool1d(_OP.AVG_POOL1D, "GGML_OP_POOL_AVG")


@_rr(_OP.CONV3D)
def render_conv3d(node, ctx):
    has_bias = bool(ctx.attr(node, "bias", False))
    key = ctx.weight(node, ["weight"] + (["bias"] if has_bias else []))
    s = ctx.scalar(ctx.attr(node, "stride", [1, 1, 1]))
    p = ctx.scalar(ctx.attr(node, "padding", [0, 0, 0]))
    d = ctx.scalar(ctx.attr(node, "dilation", [1, 1, 1]))
    return ctx.out(
        node,
        f"ggml_conv_3d(m, {key}, {ctx.inp(node)}, {s}, {s}, {s}, {p}, {p}, {p}, {d}, {d}, {d})"
        " /* TODO(ggml): conv_3d 인자 시그니처 확인 */",
        hint="conv3d",
    )


def _conv_transpose(op, builder, ndim):
    @_rr(op)
    def _r(node, ctx, _b=builder):
        has_bias = bool(ctx.attr(node, "bias", False))
        key = ctx.weight(node, ["weight"] + (["bias"] if has_bias else []))
        s = ctx.scalar(ctx.attr(node, "stride", [1]))
        return ctx.out(
            node,
            f"{_b}(m, {key}, {ctx.inp(node)}, {s})"
            " /* TODO(ggml): conv_transpose 인자(stride/pad) 확인 */",
            hint="convT",
        )
    return _r


render_convT1d = _conv_transpose(_OP.CONVTRANSPOSE1D, "ggml_conv_transpose_1d", 1)
render_convT2d = _conv_transpose(_OP.CONVTRANSPOSE2D, "ggml_conv_transpose_2d_p0", 2)


@_rr(_OP.PAD_REFLECT_1D)
def render_pad_reflect_1d(node, ctx):
    p = ctx.attr(node, "pad", ctx.attr(node, "padding", [0, 0])) or [0, 0]
    p = [int(x) for x in p] + [0, 0]
    return ctx.out(node, f"ggml_pad_reflect_1d(m, {ctx.inp(node)}, {p[0]}, {p[1]})", hint="padr")


@_rr(_OP.ARGMAX)
def render_argmax(node, ctx):
    return ctx.out(node, f"ggml_argmax(m, {ctx.inp(node)})", hint="amax")


@_rr(_OP.PRELU)
def render_prelu(node, ctx):
    # PReLU(per-channel slope)을 ggml_leaky_relu(scalar)로 근사. 정확본은 mul/max 합성(TODO).
    return ctx.out(
        node,
        f"ggml_leaky_relu(m, {ctx.inp(node)}, 0.25f, false)"
        " /* TODO(ggml): PReLU per-channel slope → scalar 근사 */",
        hint="prelu",
    )


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
