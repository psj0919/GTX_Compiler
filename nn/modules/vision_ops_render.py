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
    return ctx.out(node, f"ggml_gelu(m, {ctx.inp(node)})", hint="gelu")


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


@_rr(_OP.MEAN)
def render_mean(node, ctx):
    # ggml_mean: ne0(행) 평균. torch dim 의미와 다르면 후처리 필요.
    return ctx.out(node, f"ggml_mean(m, {ctx.inp(node)})", hint="mean")
