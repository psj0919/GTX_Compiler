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
"""정규화 op 의 ggml/vision.cpp render (register_render side-effect).

기존 batch_norm.py 인라인 render + vision_ops_render.py 의 정규화 render
(layer_norm/group_norm/instance_norm)를 통합한 모듈.
"""

from shared.compile.render_api import register_render
from shared.base import OP


@register_render(OP.BATCH_NORM)
def render_batch_norm(node, ctx):
    key = ctx.weight(node, ["weight", "bias", "running_mean", "running_var"])
    return ctx.out(node, f"batch_norm_2d({key}, {ctx.inp(node)})")


@register_render(OP.LAYER_NORM)
def render_layer_norm(node, ctx):
    # visp wrapper: layer_norm(m["k"], x) = ggml_norm + weight/bias affine (GGUF).
    key = ctx.weight(node, ["weight", "bias"])
    return ctx.out(node, f"layer_norm({key}, {ctx.inp(node)})", hint="ln")


@register_render(OP.GROUP_NORM)
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


@register_render(OP.INSTANCE_NORM)
def render_instance_norm(node, ctx):
    eps = float(ctx.attr(node, "eps", 1e-5))
    return ctx.out(node, f"ggml_norm(m, {ctx.inp(node)}, {eps}f)", hint="in")
