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
    # ⚠️ 원소가 **Tensor 일 수 있다**(NAS-FPN 은 stride 를 런타임에 계산해 부른다).
    #    `int(Tensor)` 는 TypeError 라 통째로 감싸고 기본값으로 떨어뜨린다 — 그 뒤
    #    `render_maxpool` 이 출력 shape 로 검산해 역산한다.
    try:
        if isinstance(v, (list, tuple)):
            if len(v) >= 2:
                return int(v[0]), int(v[1])
            if len(v) == 1:
                return int(v[0]), int(v[0])
        if isinstance(v, int) and not isinstance(v, bool):
            return v, v
    except (TypeError, ValueError):
        pass
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

    # kernel/stride 가 **동적**이라 `_pair` 가 기본값(2)으로 떨어지는 경우가 있다.
    # NAS-FPN 의 merge cell 이 그렇다 — 레벨을 맞추려고 `F.max_pool2d(x, s, s)` 를
    # `s = in/target` 로 계산해 부르므로 trace 에 상수로 안 남는다. 기본 2 로 나가면
    # 크기가 안 맞고(80→40 인데 20 이어야 함), 그 add 가 `40 % 20 == 0` 이라 **조용히 통과**한 뒤
    # 다음 add 에서 `20 % 40 ≠ 0` 으로 터진다 — 크래시 지점이 원인 지점이 아니다.
    # → 판단 기준은 "attr 이 있느냐" 가 아니라 **그 값이 실제 출력 shape 을 만드느냐** 다.
    try:
        ish = [int(d) for d in [t for t in node.in_tensors if t is not None][0].shape]
        osh = [int(d) for d in (out_shape(node) or [])]
        if len(ish) >= 2 and len(osh) == len(ish) and osh[-1] and osh[-2]:
            exp_h = (ish[-2] + 2 * p1 - k1) // s1 + 1
            exp_w = (ish[-1] + 2 * p0 - k0) // s0 + 1
            if (exp_h, exp_w) != (osh[-2], osh[-1]):
                rh, rw = ish[-2] // osh[-2], ish[-1] // osh[-1]
                if rh > 0 and rw > 0 and ish[-2] % osh[-2] == 0 and ish[-1] % osh[-1] == 0:
                    k0, k1 = rw, rh          # (ne0,ne1)=(W,H) 순서
                    s0, s1 = rw, rh
                    p0, p1 = 0, 0
    except Exception:
        pass
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


@register_render(OP.ADAPTIVEMAXPOOL2D)
def render_adaptive_max_pool(node, ctx):
    # BFP(Libra) 다운샘플 등. in/out 이 정수배면 kernel=stride=in//out 인 max pool 로 정확.
    # (NCHW: H,W=마지막 2. ggml ne0=W, ne1=H.) 비정수배는 크기만 맞추고 값은 근사.
    a = ctx.inp(node)

    def _hw(shp):
        try:
            s = [int(d) for d in shp]
            return (s[-2], s[-1]) if len(s) >= 2 else None
        except Exception:
            return None

    ins = [t for t in node.in_tensors if t is not None]
    ih_iw = _hw(ins[0].shape) if ins else None
    outs = [t for t in (node.out_tensors or []) if t is not None]
    oh_ow = _hw(outs[0].shape) if outs else None
    if oh_ow is None:
        os_cfg = ctx.attr(node, "output_size", None)
        try:
            oh_ow = (int(os_cfg[0]), int(os_cfg[1]))
        except Exception:
            oh_ow = None
    if ih_iw and oh_ow and oh_ow[0] > 0 and oh_ow[1] > 0:
        def _ksp(inp, outp):
            if inp % outp == 0:
                k = inp // outp
                return k, k, 0
            s = max(1, round(inp / outp))
            pad2 = outp * s - inp
            if pad2 >= 0 and pad2 % 2 == 0:
                return s, s, pad2 // 2
            k = inp - (outp - 1) * s          # 홀수 pad/음수 → 큰 kernel 로 크기를 정확히 맞춤
            return (k if k >= 1 else 1), s, 0
        kh, sh, ph = _ksp(ih_iw[0], oh_ow[0])
        kw, sw, pw = _ksp(ih_iw[1], oh_ow[1])
        exact = (ih_iw[0] % oh_ow[0] == 0 and ih_iw[1] % oh_ow[1] == 0)
        expr = (f"ggml_pool_2d(m, {a}, GGML_OP_POOL_MAX, "
                f"{kw}, {kh}, {sw}, {sh}, {pw}, {ph})")
        if not exact:
            expr += " /* TODO(ggml): non-integer adaptive max pool — 크기맞춤 pad, 값 근사 */"
        return ctx.out(node, expr, hint="pool")
    return ctx.out(node, f"ggml_pool_2d(m, {a}, GGML_OP_POOL_MAX, "
                         f"{a}->ne[0], {a}->ne[1], {a}->ne[0], {a}->ne[1], 0, 0)", hint="pool")


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
