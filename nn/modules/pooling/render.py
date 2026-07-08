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
"""pooling op 의 ggml/vision.cpp render (register_render side-effect).

기존 {maxpool,avgpool,adaptive_avg_pool,interpolate}.py 인라인 render 를 통합한 모듈.
"""

from shared.compile.render_api import register_render
from shared.base import OP


@register_render(OP.MAX_POOL)
def render_maxpool(node, ctx):
    k = ctx.scalar(ctx.attr(node, "kernel_size", [2, 2]))
    s = ctx.scalar(ctx.attr(node, "stride", [k, k]))
    p = ctx.scalar(ctx.attr(node, "padding", [0, 0]))
    return ctx.out(
        node,
        f"ggml_pool_2d(m, {ctx.inp(node)}, GGML_OP_POOL_MAX, {k}, {k}, {s}, {s}, {p}, {p})"
        " /* TODO(ggml): verify pool params/layout */",
    )


@register_render(OP.AVG_POOL)
def render_avgpool(node, ctx):
    k = ctx.scalar(ctx.attr(node, "kernel_size", [2, 2]))
    s = ctx.scalar(ctx.attr(node, "stride", [k, k]))
    p = ctx.scalar(ctx.attr(node, "padding", [0, 0]))
    return ctx.out(
        node,
        f"ggml_pool_2d(m, {ctx.inp(node)}, GGML_OP_POOL_AVG, {k}, {k}, {s}, {s}, {p}, {p})",
    )


def _shape_of(t):
    try:
        return [int(x) for x in t.shape]
    except Exception:
        return None


@register_render(OP.ADAPTIVEAVGPOOL2D)
def render_adaptive_avg_pool(node, ctx):
    # adaptive_avg_pool2d((Ho,Wo)): 입력 (Hi,Wi) 가 출력의 배수면(Hi%Ho==Wi%Wo==0) 고정 커널
    # avg pool 과 동치 → kernel=stride=(Wi/Wo, Hi/Ho) 로 정확히 tiled pooling. (1,1) global 도
    # 이 식의 특수해. 비-배수(진짜 가변커널 adaptive)는 ggml 미지원 → global 폴백 + TODO.
    a = ctx.inp(node)
    ish = _shape_of(node.in_tensors[0]) if node.in_tensors else None
    osh = _shape_of(node.out_tensors[0]) if node.out_tensors else None
    if ish and osh and len(ish) >= 2 and len(osh) >= 2:
        Hi, Wi = ish[-2], ish[-1]
        Ho, Wo = osh[-2], osh[-1]
        if Ho > 0 and Wo > 0 and Hi % Ho == 0 and Wi % Wo == 0:
            k0, k1 = Wi // Wo, Hi // Ho          # ggml: k0 on ne0(W), k1 on ne1(H)
            expr = (f"ggml_pool_2d(m, {a}, GGML_OP_POOL_AVG, "
                    f"{k0}, {k1}, {k0}, {k1}, 0, 0)")
            return ctx.out(node, expr, hint="pool")
    # 비-배수 또는 shape 미상 → 전 spatial 평균(global)로 폴백.
    expr = (f"ggml_pool_2d(m, {a}, GGML_OP_POOL_AVG, "
            f"{a}->ne[0], {a}->ne[1], {a}->ne[0], {a}->ne[1], 0, 0)"
            f" /* TODO(ggml): non-divisible adaptive avg pool — 가변커널 미지원, global 폴백 */")
    return ctx.out(node, expr, hint="pool")


@register_render(OP.RESIZE, OP.INTERPOLATE)
def render_interpolate(node, ctx):
    # YOLO neck 의 upsample 은 nearest, 정수배(보통 2x). ggml_upscale(nearest) 로 매핑.
    s = ctx.scalar(ctx.attr(node, "scale", [2, 2]), 0, 2)
    mode = str(ctx.attr(node, "mode", "nearest")).lower()
    note = "" if "near" in mode else f" /* TODO(ggml): mode={mode} (upscale=nearest) */"
    return ctx.out(
        node,
        f"ggml_upscale(m, {ctx.inp(node)}, {int(s)}, GGML_SCALE_MODE_NEAREST)"
        f" /* TODO(ggml): verify ggml_upscale signature for this build */{note}",
        hint="up",
    )
