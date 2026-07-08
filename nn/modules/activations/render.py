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
"""활성화 함수 op 의 ggml/vision.cpp render (register_render side-effect).

기존 {relu,sigmoid,softmax}.py 인라인 render + vision_ops_render.py 의 활성화
render(gelu/tanh/leaky_relu/relu6/log_softmax/prelu)를 통합한 모듈.
"""

from shared.compile.render_api import register_render, out_shape, ggml_axis
from shared.base import OP


@register_render(OP.RELU)
def render_relu(node, ctx):
    fn = "ggml_relu_inplace" if ctx.attr(node, "inplace", False) else "ggml_relu"
    return ctx.out(node, f"{fn}(m, {ctx.inp(node)})")


@register_render(OP.SIGMOID)
def render_sigmoid(node, ctx):
    return ctx.out(node, f"ggml_sigmoid(m, {ctx.inp(node)})")


@register_render(OP.SOFTMAX)
def render_softmax(node, ctx):
    # ggml_soft_max 는 ne0(행) 축에 대해서만 정규화. torch dim 을 ggml 축으로 변환해
    # 그 축이 ne0 가 아니면 permute 로 ne0 로 옮긴 뒤 softmax, 다시 되돌린다(DFL 등).
    a = ctx.inp(node)
    sh = out_shape(node)
    ndim = len(sh) if sh else 4
    axis = int(ctx.attr(node, "dim", -1))
    g = ggml_axis(axis, ndim)           # 정규화할 ggml 축
    if g == 0:
        return ctx.out(node, f"ggml_soft_max(m, {a})", hint="sm")
    # ne0 <-> ne_g 스왑 permute (ggml_permute(src_axis→dst_axis)).
    p = [0, 1, 2, 3]
    p[0], p[g] = g, 0
    perm = f"{p[0]}, {p[1]}, {p[2]}, {p[3]}"
    expr = (f"ggml_cont(m, ggml_permute(m, "
            f"ggml_soft_max(m, ggml_cont(m, ggml_permute(m, {a}, {perm}))), {perm}))")
    return ctx.out(node, expr, hint="sm")


@register_render(OP.GELU)
def render_gelu(node, ctx):
    # approximate: 'none'(erf, 정확) / 'tanh'(기본) / 'quick'(sigmoid 근사)
    approx = str(ctx.attr(node, "approximate", "tanh")).lower()
    fn = {"none": "ggml_gelu_erf", "quick": "ggml_gelu_quick"}.get(approx, "ggml_gelu")
    return ctx.out(node, f"{fn}(m, {ctx.inp(node)})", hint="gelu")


@register_render(OP.TANH)
def render_tanh(node, ctx):
    return ctx.out(node, f"ggml_tanh(m, {ctx.inp(node)})", hint="tanh")


@register_render(OP.LEAKY_RELU)
def render_leaky_relu(node, ctx):
    slope = ctx.attr(node, "negative_slope", None)
    if slope is None:
        slope = ctx.attr(node, "alpha", 0.01)
    return ctx.out(node, f"ggml_leaky_relu(m, {ctx.inp(node)}, {float(slope)}f, false)", hint="lrelu")


@register_render(OP.RELU6)
def render_relu6(node, ctx):
    return ctx.out(node, f"ggml_clamp(m, {ctx.inp(node)}, 0.0f, 6.0f)", hint="relu6")


@register_render(OP.LOG_SOFTMAX)
def render_log_softmax(node, ctx):
    return ctx.out(node, f"ggml_log(m, ggml_soft_max(m, {ctx.inp(node)}))", hint="lsm")


@register_render(OP.PRELU)
def render_prelu(node, ctx):
    # PReLU(per-channel slope)을 ggml_leaky_relu(scalar)로 근사. 정확본은 mul/max 합성(TODO).
    return ctx.out(
        node,
        f"ggml_leaky_relu(m, {ctx.inp(node)}, 0.25f, false)"
        " /* TODO(ggml): PReLU per-channel slope → scalar 근사 */",
        hint="prelu",
    )
