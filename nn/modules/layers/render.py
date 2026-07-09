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
"""layer op 의 ggml/vision.cpp render (register_render side-effect).

기존 {conv,linear,matmul,add,sub,multiply,concat}.py 인라인 render + mean render를
통합한 모듈.
"""

from shared.compile.render_api import register_render, weight_key, out_shape, ggml_axis
from shared.base import OP


@register_render(OP.CONV2D)
def render_conv(node, ctx):
    has_bias = bool(ctx.attr(node, "bias", False))
    key = ctx.weight(node, ["weight"] + (["bias"] if has_bias else []))
    s = ctx.scalar(ctx.attr(node, "stride", [1, 1]))
    p = ctx.scalar(ctx.attr(node, "padding", [0, 0]))
    # 양자 conv: GGUF 에 커널이 2D [IC*KH*KW, OC] 양자 형태로 저장됨 → conv_2d 대신
    # conv_2d_q(im2col+mul_mat) emit (헬퍼는 codegen 이 .cpp 상단에 주입). KH/KW 는 plan 에서.
    entry = (getattr(ctx, "quant_plan", None) or {}).get(weight_key(node))
    if entry is not None and entry.get("kind") == "conv":
        return ctx.out(
            node,
            f"conv_2d_q({key}, {ctx.inp(node)}, {s}, {p}, {entry['kh']}, {entry['kw']})",
        )
    return ctx.out(node, f"conv_2d({key}, {ctx.inp(node)}, {s}, {p})")


@register_render(OP.DENSE)
def render_linear(node, ctx):
    # visp::linear = ggml_mul_mat(weight, x): x 의 ne[0] 가 in_features 여야 한다.
    # PyTorch Linear.weight[out,in] 를 그대로 GGUF 에 쓰면 ggml ne=[in,out] 이 되어
    # mul_mat 결과가 [out, N] 으로 정합한다 (앞단 flatten 이 [in, N] 을 만든다).
    has_bias = bool(ctx.attr(node, "bias", True))
    key = ctx.weight(node, ["weight"] + (["bias"] if has_bias else []))
    return ctx.out(node, f"linear({key}, {ctx.inp(node)})", hint="fc")


@register_render(OP.MATMUL)
def render_matmul(node, ctx):
    # torch matmul(A,B): A=(...,M,K), B=(...,K,N) → C=(...,M,N).
    # ggml_mul_mat(a,b) 는 a.ne0==b.ne0 인 수축축 K 를 요구하고 결과 ne=(a.ne1,b.ne1).
    # in0(A) 는 natural reverse 라 ne0=K 이미 OK. in1(B) 는 ne0=N 이므로 앞 두 ggml축을
    # swap(=마지막 두 torch축 transpose)해 ne0=K 로 만든 뒤, 피연산자를 (Bt, A) 순서로
    # 주면 결과 ne=(N,M)=torch C 의 natural reverse 가 된다 (검증: ggml_backend._op_matmul).
    a = ctx.inp(node, 0)
    b = ctx.inp(node, 1)
    bt = f"ggml_cont(m, ggml_permute(m, {b}, 1, 0, 2, 3))"
    return ctx.out(node, f"ggml_mul_mat(m, {bt}, {a})", hint="mm")


@register_render(OP.ADD)
def render_add(node, ctx):
    return ctx.out(node, f"ggml_add(m, {ctx.inp(node, 0)}, {ctx.inp(node, 1)})")


@register_render(OP.SUB)
def render_sub(node, ctx):
    return ctx.out(node, f"ggml_sub(m, {ctx.inp(node, 0)}, {ctx.inp(node, 1)})")


@register_render(OP.MULTIPLY)
def render_mul(node, ctx):
    return ctx.out(node, f"ggml_mul(m, {ctx.inp(node, 0)}, {ctx.inp(node, 1)})")


@register_render(OP.DIV)
def render_div(node, ctx):
    return ctx.out(node, f"ggml_div(m, {ctx.inp(node, 0)}, {ctx.inp(node, 1)})")


@register_render(OP.CONCAT)
def render_concat(node, ctx):
    # ggml_concat 은 2항 → n 입력은 좌결합으로 폴딩. torch dim → ggml ne 축 변환.
    ins = [t for t in node.in_tensors if t is not None]
    vars_ = [ctx.inp(node, i) for i in range(len(ins))]
    sh = out_shape(node)
    ndim = len(sh) if sh else 4
    gdim = ggml_axis(ctx.attr(node, "dim", 1), ndim)
    if not vars_:
        return ctx.out(node, "x /* TODO(ggml): concat with no inputs */")
    expr = vars_[0]
    for v in vars_[1:]:
        expr = f"ggml_concat(m, {expr}, {v}, {gdim})"
    return ctx.out(node, expr, hint="cat")


def _mean_in_shape(node, i=0):
    ins = [t for t in (getattr(node, "in_tensors", None) or []) if t is not None]
    if i < len(ins):
        try:
            return [int(x) for x in ins[i].shape]
        except Exception:
            return None
    return None


@register_render(OP.MEAN)
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


# --- vision_ops_render.py / head_render.py 에서 흡수한 conv 계열 render ---
@register_render(OP.DEPTHWISE_CONV2D)
def render_dwconv(node, ctx):
    has_bias = bool(ctx.attr(node, "bias", False))
    key = ctx.weight(node, ["weight"] + (["bias"] if has_bias else []))
    s = ctx.scalar(ctx.attr(node, "stride", [1, 1]))
    p = ctx.scalar(ctx.attr(node, "padding", [0, 0]))
    return ctx.out(node, f"conv_2d_depthwise({key}, {ctx.inp(node)}, {s}, {p})", hint="dwconv")


@register_render(OP.CONV3D)
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


def _conv_transpose(op, builder):
    @register_render(op)
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


render_convT1d = _conv_transpose(OP.CONVTRANSPOSE1D, "ggml_conv_transpose_1d")
render_convT2d = _conv_transpose(OP.CONVTRANSPOSE2D, "ggml_conv_transpose_2d_p0")
