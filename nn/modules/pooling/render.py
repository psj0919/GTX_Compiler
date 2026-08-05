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

from shared.compile.render_api import register_render, out_shape
from shared.base import OP


def _pair(v, d0, d1):
    """torch (h, w) → 정수쌍. **비대칭** kernel/stride/padding 지원.

    `ctx.scalar()` 는 리스트의 **첫 원소만** 쓴다 — `AvgPool2d((4,8))` 이 (4,4) 가 돼
    출력 spatial 이 달라지고, 한참 뒤 flatten 이 `ggml_nelements` 로 죽는다(reid 실측).
    파서는 kernel/stride/padding 을 (ne0,ne1)=(W,H) 순서로 저장하므로 ggml_pool_2d 의
    (k0,k1) 에 **그대로** 넘긴다(스왑 금지).
    """
    if isinstance(v, (list, tuple)):
        if len(v) >= 2:
            return int(v[0]), int(v[1])
        if len(v) == 1:
            return int(v[0]), int(v[0])
    if isinstance(v, int) and not isinstance(v, bool):
        return v, v
    return d0, d1


def _is_static_kernel(v):
    if isinstance(v, int) and not isinstance(v, bool):
        return True
    if isinstance(v, (list, tuple)) and v and all(
            isinstance(d, int) and not isinstance(d, bool) for d in v):
        return True
    return False


@register_render(OP.MAX_POOL)
def render_maxpool(node, ctx):
    k0, k1 = _pair(ctx.attr(node, "kernel_size", [2, 2]), 2, 2)
    s0, s1 = _pair(ctx.attr(node, "stride", [k0, k1]), k0, k1)
    p0, p1 = _pair(ctx.attr(node, "padding", [0, 0]), 0, 0)
    return ctx.out(
        node,
        f"ggml_pool_2d(m, {ctx.inp(node)}, GGML_OP_POOL_MAX, {k0}, {k1}, {s0}, {s1}, {p0}, {p1})",
    )


@register_render(OP.AVG_POOL)
def render_avgpool(node, ctx):
    ksz = ctx.attr(node, "kernel_size", [2, 2])
    # 동적 kernel(global avg pool: `F.avg_pool2d(x, x.size()[2:])`)은 정적 int 가 아닌 Tensor.
    # → 입력의 정적 spatial(H,W)로 kernel 을 잡아 전역 평균(output 1x1)으로 emit.
    if not _is_static_kernel(ksz):
        ish = None
        try:
            ish = [int(d) for d in [t for t in node.in_tensors if t is not None][0].shape]
        except Exception:
            ish = None
        if ish and len(ish) == 4:
            kh, kw = ish[2], ish[3]  # [N,C,H,W]
            return ctx.out(node, f"ggml_pool_2d(m, {ctx.inp(node)}, GGML_OP_POOL_AVG, "
                                 f"{kw}, {kh}, {kw}, {kh}, 0, 0) /* global avg */", hint="pool")
        return ctx.out(node, f"ggml_cont(m, {ctx.inp(node)}) /* TODO avg_pool dynamic kernel */",
                       hint="pool")
    k0, k1 = _pair(ksz, 2, 2)
    s0, s1 = _pair(ctx.attr(node, "stride", [k0, k1]), k0, k1)
    p0, p1 = _pair(ctx.attr(node, "padding", [0, 0]), 0, 0)
    return ctx.out(
        node,
        f"ggml_pool_2d(m, {ctx.inp(node)}, GGML_OP_POOL_AVG, {k0}, {k1}, {s0}, {s1}, {p0}, {p1})",
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
    # interpolate: size= 또는 scale_factor= 둘 다 지원. 정적 출력 shape 가 있으면 그 타깃 차원으로
    # ggml_interpolate(ne0..ne3) 를 emit (scale_factor 하드코딩 → `GGML_ASSERT(scale_factor > 1)`
    # 로 죽던 문제 해결. mmdet FPN 은 size= 로 부르는 곳이 많다).
    # 입력은 WHCN-contiguous → ggml ne = out_shape(torch NCHW) 역순 = [W,H,C,N].
    mode = str(ctx.attr(node, "mode", "nearest")).lower()
    # ⚠️ `"near" in mode` 로 판정하면 안 된다 — **"bilinear" 가 "near" 를 포함**한다
    #    ("bili-near"). 그 탓에 bilinear 업샘플이 전부 nearest 로 나갔다(DyHead 24곳).
    is_nearest = mode.startswith("nearest") or "'nearest" in mode
    gmode = "GGML_SCALE_MODE_NEAREST" if is_nearest else "GGML_SCALE_MODE_BILINEAR"
    # torch `align_corners=True` 는 ggml 의 별도 **플래그**로 표현한다(모드가 아니다).
    # 안 붙이면 격자 정렬이 반 픽셀 어긋나 bilinear 결과가 미세하게 틀린다.
    # 실측(DyHead): nearest 0.9601 → bilinear 0.9794 → +align_corners 0.9868
    if not is_nearest and bool(ctx.attr(node, "align_corners", False)):
        gmode = f"(ggml_scale_mode)({gmode} | GGML_SCALE_FLAG_ALIGN_CORNERS)"
    inp = ctx.inp(node)
    sh = out_shape(node)
    if sh and 1 <= len(sh) <= 4:
        ne = list(reversed([int(d) for d in sh]))
        while len(ne) < 4:
            ne.append(1)
        ne0, ne1, ne2, ne3 = ne[:4]
        # ⚠️ 바깥 주석 안에 들어가므로 `/* */` 를 넣으면 **중첩 주석**이 돼 C++ 이 깨진다.
        note = "" if is_nearest else f", mode={mode.strip(chr(39))}"
        return ctx.out(
            node,
            f"ggml_interpolate(m, {inp}, {ne0}, {ne1}, {ne2}, {ne3}, {gmode})"
            f" /* interpolate -> target ne{note} */",
            hint="up",
        )
    # 정적 출력 shape 미상 → 정수배 scale fallback (ggml_upscale 는 scale>=2 필요)
    s = max(int(ctx.scalar(ctx.attr(node, "scale", [2, 2]), 0, 2)), 2)
    return ctx.out(
        node,
        f"ggml_upscale(m, {inp}, {s}, {gmode}) /* TODO(ggml): out_shape 미상, 정수배 가정 */",
        hint="up",
    )


# --- vision_ops_render.py 에서 흡수한 1D pooling render ---
def _pool1d(op, kind):
    @register_render(op)
    def _r(node, ctx, _k=kind):
        k = ctx.scalar(ctx.attr(node, "kernel_size", [2]))
        s = ctx.scalar(ctx.attr(node, "stride", [k]))
        p = ctx.scalar(ctx.attr(node, "padding", [0]))
        return ctx.out(node, f"ggml_pool_1d(m, {ctx.inp(node)}, {_k}, {k}, {s}, {p})", hint="pool1d")
    return _r


render_maxpool1d = _pool1d(OP.MAX_POOL1D, "GGML_OP_POOL_MAX")
render_avgpool1d = _pool1d(OP.AVG_POOL1D, "GGML_OP_POOL_AVG")
