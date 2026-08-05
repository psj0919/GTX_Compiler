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
    # grouped conv (groups>1, depthwise 아님 — RegNet/ResNeXt/ResNeSt 의 group_width).
    # 안 보면 `conv_2d` 가 전 채널을 한 번에 돌려 `ggml_im2col` 의
    # `GGML_ASSERT(a->ne[2] == b->ne[2])`(커널 IC != src C) 로 죽는다.
    # ⚠️ `conv_2d_grouped` 는 vision.cpp 헬퍼다 — submodule 포인터에 그 심볼이 있어야 링크된다.
    d = ctx.scalar(ctx.attr(node, "dilation", [1, 1]))
    g = ctx.attr(node, "groups", 1)
    g = int(g) if isinstance(g, (int, float)) else 1
    if g > 1:
        return ctx.out(node, f"conv_2d_grouped({key}, {ctx.inp(node)}, {s}, {p}, {d}, {g})")
    # dilation>1 (atrous, 예: FCN/DeepLab) 은 5번째 인자로 전달. =1 이면 기존 4-인자 형태 유지.
    if int(d) != 1:
        return ctx.out(node, f"conv_2d({key}, {ctx.inp(node)}, {s}, {p}, {d})")
    return ctx.out(node, f"conv_2d({key}, {ctx.inp(node)}, {s}, {p})")


@register_render(OP.DEFORM_CONV2D)
def render_deform_conv(node, ctx):
    # in_tensors 는 torchvision 스키마 순서 [input, weight, offset, mask, bias].
    has_bias = bool(ctx.attr(node, "bias", False))
    key = ctx.weight(node, ["weight"] + (["bias"] if has_bias else []))
    s = ctx.scalar(ctx.attr(node, "stride", [1, 1]))
    p = ctx.scalar(ctx.attr(node, "padding", [0, 0]))
    mask = ctx.inp(node, 3) if ctx.attr(node, "use_mask", False) else "nullptr"
    d = ctx.scalar(ctx.attr(node, "dilation", [1, 1]))
    g = int(ctx.attr(node, "groups", 1) or 1)
    og = int(ctx.attr(node, "offset_groups", 1) or 1)
    # ggml_conv_2d_deform 에는 dilation/groups/offset_groups 인자가 없다 — 1 이 아니면 부정확.
    note = "" if (d == 1 and g == 1 and og == 1) else (
        f"  /* TODO(ggml): dilation={d} groups={g} offset_groups={og} 미지원 */")
    return ctx.out(
        node,
        f"conv_2d_deform({key}, {ctx.inp(node, 0)}, {ctx.inp(node, 2)}, {mask}, {s}, {p}){note}",
        hint="dcn",
    )


def _param_named(node, suffix):
    # node 의 param in_tensor 중 이름이 suffix 로 끝나는 것의 (프리픽스 제거된) gguf 키.
    for t in node.in_tensors:
        if t is None or getattr(t, "_node", 1) is not None:
            continue
        nm = str(getattr(t, "name", "") or "").split("::")[-1]
        if nm.endswith(suffix):
            return nm
    return None


@register_render(OP.DENSE)
def render_linear(node, ctx):
    # visp::linear = ggml_mul_mat(weight, x): x 의 ne[0] 가 in_features 여야 한다.
    # PyTorch Linear.weight[out,in] 를 그대로 GGUF 에 쓰면 ggml ne=[in,out] 이 되어
    # mul_mat 결과가 [out, N] 으로 정합한다 (앞단 flatten 이 [in, N] 을 만든다).
    has_bias = bool(ctx.attr(node, "bias", True))
    raw_key = weight_key(node)
    wname = _param_named(node, "weight")
    # 표준 nn.Linear 는 weight 이름이 "<key>.weight" → 기존 linear() 경로. 비표준 이름
    # (예: MHA in_proj_weight / out_proj.weight)이면 정확한 gguf 텐서명으로 직접 emit.
    # ⚠️ 이게 없으면 노드명(= 모듈 경로)만으로 키를 만들어 `...attn.attn.weight` 를 찾다가
    #    런타임에 `tensor not found` 로 죽는다 — 실제 텐서는 `...attn.attn.out_proj.weight` 다.
    if wname is not None and wname != f"{raw_key}.weight":
        x = ctx.inp(node)
        ctx._weights.append((wname, []))
        expr = f'ggml_mul_mat(m, m.find("{wname}"), {x})'
        bname = _param_named(node, "bias") if has_bias else None
        if bname is not None:
            ctx._weights.append((bname, []))
            expr = f'ggml_add(m, {expr}, m.find("{bname}"))'
        return ctx.out(node, expr, hint="fc")
    key = ctx.weight(node, ["weight"] + (["bias"] if has_bias else []))
    return ctx.out(node, f"linear({key}, {ctx.inp(node)})", hint="fc")


@register_render(OP.LINEAR_DYNAMIC)
def render_linear_dynamic(node, ctx):
    # F.linear(x, W, b) = x @ W.T + b 에서 W 가 동적 그래프 텐서(예: nn.MultiheadAttention 이
    # in_proj_weight 를 split 한 q/k/v chunk). in_tensors = [input, weight, (bias)].
    # ggml_mul_mat(W, x): W ne=[in,out](torch[out,in]), x ne0=in → 결과 ne0=out. (DENSE 와 동일 규약)
    ins = [t for t in node.in_tensors if t is not None]
    x = ctx.inp(node, 0)
    w = ctx.inp(node, 1)
    expr = f"ggml_mul_mat(m, {w}, {x})"
    if bool(ctx.attr(node, "has_bias", False)) and len(ins) >= 3:
        expr = f"ggml_add(m, {expr}, {ctx.inp(node, 2)})"
    return ctx.out(node, expr, hint="fcd")


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


def _same_drop_reshape(node, small_expr, big_sh, small_sh):
    """>4D 두 피연산자를 **같은 축을 버려서** 4D 로 낮춘 small 표현식.

    `_reshape_to_out` 은 >4D 를 낮출 때 크기-1 축을 **전부** 버린다. 두 피연산자가 각자
    독립으로 그러면 버리는 축 수가 달라져 rank 가 어긋나고, 실제 축이 ne 슬롯을 옮겨 앉는다.

    Swin attention mask:
        attn = attn.view(B//nW, nW, nH, N, N)          torch [1,361,3,49,49]
          크기-1 이 1개 → [361,3,49,49]  → ne [49,49,3,361]
        mask = mask.unsqueeze(1).unsqueeze(0)          torch [1,361,1,49,49]
          크기-1 이 2개 → [361,49,49]    → ne [49,49,361,1]  ✗ head 축 소실
        같은 축(index 0)만 버리면 → [361,1,49,49] → ne [49,49,1,361]  ✓

    큰 쪽이 4D 로 내려오며 버린 축 집합을 그대로 작은 쪽에 적용한다.
    **rank 가 같고 둘 다 >4D 일 때만** 개입한다(conv 계열은 애초에 해당 없음).
    """
    if len(big_sh) != len(small_sh) or len(big_sh) <= 4:
        return None
    drop = [i for i, d in enumerate(big_sh) if int(d) == 1][:len(big_sh) - 4]
    if len(big_sh) - len(drop) != 4:
        return None
    kept = [int(d) for i, d in enumerate(small_sh) if i not in set(drop)]
    if len(kept) != 4:
        return None
    ne = list(reversed(kept))
    return (f"ggml_reshape_4d(m, ggml_cont(m, {small_expr}), "
            f"{ne[0]}, {ne[1]}, {ne[2]}, {ne[3]})")


@register_render(OP.ADD)
def render_add(node, ctx):
    # ⚠️ `align_bcast`(뒤 4개 유지) 를 여기에 쓰지 마라 —
    #    nas_fpn 은 못 고치면서 `ssd`(cos 0.0)·`yolo`(cos 0.56) 를 회귀시킨다(deploy 실측).
    #    아래 `_same_drop_reshape` 는 **큰 쪽의 축-버림 집합을 그대로 따라가는** 방식이라 다르다.
    a, b = _bcast_order(node, ctx)   # add 도 가환 — 큰 텐서를 first 로
    ins = [t for t in node.in_tensors if t is not None]
    if len(ins) >= 2:
        try:
            s0 = [int(d) for d in ins[0].shape]
            s1 = [int(d) for d in ins[1].shape]
            big_sh, small_sh = (s0, s1) if (_nelem(ins[0]) or 0) >= (_nelem(ins[1]) or 0) else (s1, s0)
            fixed = _same_drop_reshape(node, b, big_sh, small_sh)
            if fixed:
                b = fixed
        except Exception:
            pass
    return ctx.out(node, f"ggml_add(m, {a}, {b})")


@register_render(OP.SUB)
def render_sub(node, ctx):
    return ctx.out(node, f"ggml_sub(m, {ctx.inp(node, 0)}, {ctx.inp(node, 1)})")


@register_render(OP.RSUB)
def render_rsub(node, ctx):
    # rsub(x, other, alpha) = other - alpha*x
    alpha = ctx.attr(node, "alpha", 1.0)
    alpha = float(alpha) if isinstance(alpha, (int, float)) else 1.0
    ins = [t for t in node.in_tensors if t is not None]
    if len(ins) > 1:
        neg = f"ggml_scale(m, {ctx.inp(node, 0)}, {-alpha}f)"
        return ctx.out(node, f"ggml_add(m, {ctx.inp(node, 1)}, {neg})", hint="rsub")
    # other 가 스칼라(`1 - x` 형태) → scale+bias 한 번으로 접는다.
    other = ctx.attr(node, "other", 0.0)
    other = float(other) if isinstance(other, (int, float)) else 0.0
    return ctx.out(node, f"ggml_scale_bias(m, {ctx.inp(node, 0)}, {-alpha}f, {other}f)",
                   hint="rsub")


def _nelem(t):
    try:
        n = 1
        for d in t.shape:
            n *= int(d)
        return n
    except Exception:
        return None


def _bcast_order(node, ctx):
    """가환 elementwise(mul/add)의 인자 순서를 broadcast 가능하게 정한다.

    `ggml_mul/add(a, b)` 는 **b 가 a 로 repeat** 돼야 한다(단방향). 인자 순서를 그대로 쓰면
    작은 쪽이 first 로 갈 때 `GGML_ASSERT(ggml_can_repeat(b, a))` 로 죽는다.
    예) SE 블록 `se[1,1,C,N] * x[H,W,C,N]` → 큰 x 를 first 로 바꿔야 한다.
    """
    a, b = ctx.inp(node, 0), ctx.inp(node, 1)
    ins = [t for t in node.in_tensors if t is not None]
    if len(ins) >= 2:
        na, nb = _nelem(ins[0]), _nelem(ins[1])
        if na is not None and nb is not None and nb > na:
            return b, a
    return a, b


def _shape_of(t):
    try:
        return [int(d) for d in t.shape]
    except Exception:
        return None


def align_bcast(node, small_expr, big_t, small_t):
    """>4D 피연산자 둘을 **같은 축 기준으로** 4D 로 낮춘 small 표현식을 돌려준다.

    `_reshape_to_out` 은 >4D 를 4D 로 낮출 때 크기-1 축을 **전부** 버린다. 두 피연산자가 각자
    독립으로 그러면 축이 어긋난다 — ggml ne 는 torch 역순이라 torch 뒤쪽 크기-1 이 `ne0`/`ne1`
    이 되는데, 그걸 버리면 실제 축이 앞으로 당겨지기 때문이다.

    resnest Split-Attention:
        x     = x.view(B, radix, C, H, W)      torch [1,2,64,128,128] → ne [128,128,64,2]
        atten = atten.view(B, radix, C, 1, 1)  torch [1,2,64,1,1]
          전부 squeeze → [2,64]      → ne [64,2,1,1]  ✗ 축이 밀림
          뒤 4개 유지  → [2,64,1,1]  → ne [1,1,64,2]  ✓
    앞 레이어는 `128 % 64 == 0` 이라 `ggml_can_repeat` 을 통과하며 **조용히 틀린 값**을 내고,
    뒤 레이어(`64 % 128 ≠ 0`)에서 터진다 — 크래시 지점이 원인 지점이 아니다.

    두 shape 의 rank 가 같고 둘 다 >4D 이며, 큰 쪽의 초과분이 전부 크기-1(= 큰 쪽은 뒤 4개를
    그대로 유지)일 때만 개입한다. 그 외에는 None 을 돌려 기존 동작을 유지한다.
    """
    bs, ss = _shape_of(big_t), _shape_of(small_t)
    if not bs or not ss or len(bs) != len(ss) or len(bs) <= 4:
        return None
    if any(d != 1 for d in bs[:-4]):      # 큰 쪽이 뒤 4개를 그대로 유지하는 경우만
        return None
    ne = list(reversed(ss[-4:]))
    n_all = 1
    for d in ss:
        n_all *= d
    n_ne = 1
    for d in ne:
        n_ne *= d
    if n_all != n_ne:
        return None                        # 원소 수가 달라지면 개입하지 않는다
    return f"ggml_reshape_4d(m, ggml_cont(m, {small_expr}), {', '.join(str(d) for d in ne)})"


@register_render(OP.MULTIPLY)
def render_mul(node, ctx):
    a, b = _bcast_order(node, ctx)
    ins = [t for t in node.in_tensors if t is not None]
    if len(ins) >= 2:
        big_t, small_t = ((ins[0], ins[1]) if (_nelem(ins[0]) or 0) >= (_nelem(ins[1]) or 0)
                          else (ins[1], ins[0]))
        fixed = align_bcast(node, b, big_t, small_t)
        if fixed:
            b = fixed
    return ctx.out(node, f"ggml_mul(m, {a}, {b})")


@register_render("aten::div", OP.DIV, "elemwise_div")
def render_div(node, ctx):
    # `aten::div` 별칭 필수 — dispatcher 가 OP.DIV 로 못 접는 경로가 있어(MultiheadAttention 의
    # `q / sqrt(d)`) 그냥 두면 unhandled 로 떨어져 스케일이 통째로 빠진다.
    ins = [t for t in node.in_tensors if t is not None]
    a = ctx.inp(node, 0)
    if len(ins) >= 2:
        return ctx.out(node, f"ggml_div(m, {a}, {ctx.inp(node, 1)})", hint="div")
    # 스칼라 ÷(상수) → 단일 텐서 입력. 정확한 상수는 후처리/스케일 대상(best-effort).
    return ctx.out(node, f"ggml_cont(m, {a}) /* scalar div (best-effort) */", hint="div")


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

# ── mmcv (Modulated)DeformConv2d ─────────────────────────────────────────────
# main 의 `OP.DEFORM_CONV2D` 는 **torchvision::deform_conv2d** 스키마 전용이다.
# mmdet 은 mmcv 의 custom autograd Function(`DeformConv2dFunction`)을 쓰므로 그 키로
# 따로 받아야 한다(파서에선 PythonOp → unknown 으로 잡히지만 여기서 렌더된다).
def _deform_hw(t):
    try:
        s = [int(d) for d in t.shape]
        return s[-2], s[-1]
    except Exception:
        return None, None


@register_render("DeformConv2dFunction", "ModulatedDeformConv2dFunction")
def render_deform_conv(node, ctx):
    """mmcv (Modulated)DeformConv2d → vision.cpp conv_2d_deform (ggml CPU deform 커널, 검증된 스칼라).

    입력 계약: DeformConv = (x, offset, weight); ModulatedDeform = (x, offset, mask, weight).
    stride/pad 는 op attr 에 없어(custom autograd) shape 비율로 추론:
      stride = round(입력H / offset H),  pad = (kernel-1)//2 (dilation 1 가정, SAME-ish).
    NPU 에 deform 커널 없으면 스케줄러가 CPU(스칼라)로 폴백한다.
    """
    ins = [t for t in node.in_tensors if t is not None]
    modulated = str(node.op.type) == "ModulatedDeformConv2dFunction" or len(ins) >= 4
    x = ctx.inp(node, 0)
    offset = ctx.inp(node, 1)
    if modulated:
        mask = ctx.inp(node, 2)
        w_idx = 3
    else:
        mask = "nullptr"
        w_idx = 2
    weight = ctx.inp(node, w_idx)  # param 텐서 → m.find("...weight")
    # mmcv 의 (Modulated)DeformConv2d 는 **가중치가 in_tensors 에 안 실려 온다**
    # (custom autograd Function). 그러면 `inp()` 가 그래프 입력 `x`(= 이미지)로 폴백해
    # 커널에 이미지가 가중치로 들어간다 — `kernel->ne[2] == C` assert 로 터진다.
    # 노드명의 모듈 경로로 gguf 텐서를 직접 찾는다.
    if weight == "x" or len(ins) <= w_idx:
        weight = ctx.raw_weight(node, "weight")

    xh, _ = _deform_hw(ins[0]) if len(ins) > 0 else (None, None)
    oh, ow = _deform_hw(ins[1]) if len(ins) > 1 else (None, None)
    kh, _ = _deform_hw(ins[w_idx]) if len(ins) > w_idx else (None, None)
    pad = ((kh - 1) // 2) if kh else 1

    # stride 는 **출력 shape** 기준으로 잡는다(offset 기준이 아니라).
    # DyHead 의 `spatial_conv_high` 는 입력 32x32 · offset 64x64 · 출력 32x32 라
    # offset 으로 재면 stride 가 틀린다.
    osh = out_shape(node)
    out_h = int(osh[-2]) if osh and len(osh) >= 2 else None
    out_w = int(osh[-1]) if osh and len(osh) >= 1 else None
    stride = max(1, round(xh / out_h)) if (xh and out_h) else 1

    # **offset/mask 가 출력보다 크면 잘라 쓴다.** mmcv 는 출력 범위 안의 인덱스만 읽으므로
    # 큰 offset 을 넘겨도 조용히 좌상단만 쓰인다 — DyHead 가 레벨 간 offset 을 공유하는 방식이다.
    # 우리 `conv_2d_deform` 은 `OW == (W+2p-k)/s+1` 을 assert 하므로 여기서 맞춰줘야 한다.
    if out_h and out_w and oh and ow and (oh > out_h or ow > out_w):
        # ⚠️ **2D 크롭이 아니라 버퍼 전체의 평탄 재해석**이다.
        # mmcv `modulated_deform_conv.cpp` 의 인덱싱:
        #     data_offset_h_ptr = ((2*(i*kw+j))   * height_col + h_col) * width_col + w_col
        # 즉 offset 을 `[2*kh*kw, height_col, width_col]`(= **출력 크기**)로 **조밀하게** 읽는다.
        # 행 stride 뿐 아니라 **채널 stride 도** `out_h*out_w` 로 재해석된다 —
        # 원본 채널 stride(64*64)를 쓰면 채널마다 엉뚱한 위치를 읽는다.
        # → 앞에서부터 `C*out_h*out_w` 개를 완전 연속으로 재해석한다.
        def _crop(t, ch_expr):
            return (f"ggml_view_4d(m, ggml_cont(m, {t}), {out_w}, {out_h}, {ch_expr}, 1, "
                    f"(size_t){out_w}*{t}->nb[0], (size_t){out_w * out_h}*{t}->nb[0], "
                    f"(size_t){out_w * out_h}*({ch_expr})*{t}->nb[0], 0)")
        offset = _crop(offset, f"{offset}->ne[2]")
        if mask != "nullptr":
            mask = _crop(mask, f"{mask}->ne[2]")

    # ── deformable_groups > 1 → **채널을 쪼개 부분 conv 를 합산**한다 ───────────────
    # `conv_2d_deform` 은 dg=1 전제(`offset->ne[2] == 2*T` assert). mmcv 의 deform_group 은
    # **출력 채널을 나누는 게 아니라 offset 을 나눈다** — 입력 채널 c 는 그룹 `c//(C/dg)` 의
    # offset 을 쓰고, 출력은 전 채널 합이다. 따라서
    #     out = Σ_g deform(src[:, g], kernel[:, g], offset_g, mask_g)
    # 로 정확히 분해된다. vision.cpp 를 안 고치고 codegen 에서 쪼갠다.
    # 실제 사례: `nas_fcos` 의 `conv_offset.weight = (54,256,3,3)` = 2 × 27 → dg=2.
    # 커널 크기는 weight 가 ins 에 없을 수 있어 3x3 을 가정한다(pad 추론과 같은 가정).
    try:
        tt = (kh * kh) if kh else 9
        cin = int(ins[0].shape[1])                      # src torch [N, C, H, W]
        off_ch = int(ins[1].shape[1])                   # offset torch [N, dg*2T, OH, OW]
        dg = off_ch // (2 * tt)
    except Exception:
        dg = 1
    if dg > 1 and cin % dg == 0:
        cg = cin // dg
        parts = []
        for g in range(dg):
            def _sl(t, ch, ch_off):                     # ne2 축(채널)만 자른다
                return (f"ggml_view_4d(m, {t}, {t}->ne[0], {t}->ne[1], {ch}, {t}->ne[3], "
                        f"{t}->nb[1], {t}->nb[2], {t}->nb[3], (size_t){ch_off}*{t}->nb[2])")
            xg = _sl(x, cg, g * cg)
            wg = _sl(weight, cg, g * cg)                # kernel ne=[kw,kh,C,Cout] → ne2=C
            og = _sl(offset, 2 * tt, g * 2 * tt)
            mg = _sl(mask, tt, g * tt) if mask != "nullptr" else "nullptr"
            parts.append(f"conv_2d_deform(m, ggml_cont(m, {xg}), ggml_cont(m, {wg}), "
                         f"ggml_cont(m, {og}), {'ggml_cont(m, ' + mg + ')' if mg != 'nullptr' else 'nullptr'}, "
                         f"{stride}, {pad})")
        expr = parts[0]
        for p in parts[1:]:
            expr = f"ggml_add(m, {expr}, {p})"
        return ctx.out(node, expr, hint="deform")

    return ctx.out(
        node,
        f"conv_2d_deform(m, {x}, {weight}, {offset}, {mask}, {stride}, {pad})",
        hint="deform",
    )
