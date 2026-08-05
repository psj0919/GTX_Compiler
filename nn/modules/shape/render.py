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
"""형상/이동/생성 op 의 ggml/vision.cpp render (head_render.py 에서 분리).

transpose/permute/contiguous/reshape/unsqueeze/stack/shape/detach/cast + 텐서 생성
(arange/const/full/meshgrid) + repeat 를 담는다. 전부 **실제 ggml 그래프 op** 으로
emit 한다. 정적 출력 shape 를 이용해 ggml ne(=torch shape 의 역순) 인자를 산출한다.
"""

from shared.compile.render_api import register_render as _rr, out_shape, ggml_axis
from shared.base import OP as _OP


# --------------------------------------------------------------------- helpers
def _ne_args(shape):
    """torch shape → ggml ne 인자 문자열(역순). 최대 4D."""
    ne = list(reversed([int(d) for d in shape]))
    return ne, ", ".join(str(d) for d in ne)


def _perm_args(order, ndim):
    """torch permute order → ggml_permute(axis0..3) 인자.

    torch: out_td=i ← in_td=order[i].  ggml 축 g = ndim-1-td.
    ggml_permute(a, p0..p3): pK = a 의 K번 ggml축이 갈 목적지 ggml축.
    """
    p = [0, 1, 2, 3]
    o = [int(x) % ndim for x in order]
    for i in range(ndim):
        src = ndim - 1 - o[i]   # in ggml axis
        dst = ndim - 1 - i      # out ggml axis
        p[src] = dst
    return p


def _merge_perm_axes(order, shape):
    """target 에서도 인접·동순인 source 인접 축쌍을 병합해 permute 를 낮춘다.

    연속 메모리에서 그런 축쌍은 원소 순서상 한 덩어리로 움직이므로 합쳐도 결과가 같다.
    `(order, shape)` 를 더 못 줄일 때까지 반복해 돌려준다(4D 이하가 되면 permute 가능).
    """
    order = [int(x) for x in order]
    shape = [int(d) for d in shape]
    changed = True
    while len(shape) > 4 and changed:
        changed = False
        for i in range(len(shape) - 1):
            if order.index(i + 1) == order.index(i) + 1:      # target 에서도 i, i+1 이 이 순서로 인접
                shape = shape[:i] + [shape[i] * shape[i + 1]] + shape[i + 2:]
                order = [x - 1 if x > i + 1 else x for x in order if x != i + 1]
                changed = True
                break
    return order, shape


def _permute_steps(order, shape):
    """>4D permute 를 **인접 전치의 연속**으로 분해한다.

    임의의 순열은 인접 전치의 곱이고, 인접 전치 하나는 나머지 축을 앞/뒤로 뭉치면
    **항상 4D permute 로 정확히 표현**된다:

        [prod(앞), d_i, d_{i+1}, prod(뒤)]  --torch swap(1,2)-->  [prod(앞), d_{i+1}, d_i, prod(뒤)]

    병합은 연속 메모리에서 원소 순서를 안 바꾸므로 수치 불변이다. 단계마다 `cont` 복사가
    한 번씩 들어가므로 느리지만, `_merge_perm_axes` 로 안 줄어드는 순열(Swin qkv
    `[2,0,3,1,4]`)의 유일한 정확 표현이다. 그 대안은 passthrough(= 틀린 값)뿐이었다.

    돌려주는 것: `[(A, x, y, B), ...]` — 순서대로 적용하면 목표 permute 가 된다.
    """
    order = [int(x) for x in order]
    shape = [int(d) for d in shape]
    cur = list(range(len(shape)))          # 각 위치에 현재 놓인 source 축
    cshape = list(shape)
    steps = []
    for k in range(len(order)):
        j = cur.index(order[k])
        while j > k:                        # 왼쪽으로 한 칸씩 밀어 올린다
            i = j - 1
            A = 1
            for d in cshape[:i]:
                A *= d
            B = 1
            for d in cshape[i + 2:]:
                B *= d
            steps.append((A, cshape[i], cshape[i + 1], B))
            cshape[i], cshape[i + 1] = cshape[i + 1], cshape[i]
            cur[i], cur[i + 1] = cur[i + 1], cur[i]
            j -= 1
    return steps


def _render_perm_general(node, ctx, order, in_shape, hint):
    """>4D permute 를 ggml(≤4D)로 **정확히** 표현한다. 3단 캐스케이드.

    ① **크기-1 축 squeeze** — 원소를 안 옮기므로 비-1 축의 permute 와 정확히 동치.
       앞단 reshape(`_reshape_to_out`)도 같은 규칙으로 squeeze 하므로 실제 텐서와 정합한다.
    ② **인접 축 병합**(`_merge_perm_axes`) — source 에서 인접하고 target 에서도 같은 순서로
       인접한 축쌍은 합쳐도 flat 원소 순서가 안 바뀐다. Swin window partition 이 이 꼴이다:
       `[B,Hp,7,Wp,7,C] order=[0,1,3,2,4,5]` → `[B*Hp,7,Wp,7*C] order=[0,2,1,3]`.
    ③ **인접 전치의 연속**(`_permute_steps`) — 항상 가능. 단계마다 cont 1회.

    ⚠️ ③ 뒤의 **정규화가 핵심**이다. 마지막 단계의 4D 그룹핑은 그 전치에 맞춘 임시 형태라
    downstream 의 `_reshape_to_out` 규약(= 바깥 축 전부를 하나로 병합)과 어긋난다. 정규화가
    없으면 뒤따르는 select/reshape 가 **원소 수는 맞는데 축 배치가 달라** `ggml_nelements`
    assert 로 죽는다(실측: pvt · detectors · libra_rcnn · reid).
    """
    a = ctx.inp(node)
    ndim = len(in_shape)
    o = [int(x) % ndim for x in order]

    # ① 크기-1 축 squeeze
    keep = [ax for ax in range(ndim) if int(in_shape[ax]) != 1]
    pos = {ax: i for i, ax in enumerate(keep)}
    sq_order = [pos[o[i]] for i in range(ndim) if int(in_shape[o[i]]) != 1]
    if 0 < len(sq_order) <= 4:
        p = _perm_args(sq_order, len(sq_order))
        return ctx.out(node, f"ggml_cont(m, ggml_permute(m, {a}, {p[0]}, {p[1]}, {p[2]}, {p[3]}))",
                       hint=hint)

    # ② 인접 축 병합
    m_order, m_shape = _merge_perm_axes(o, [int(d) for d in in_shape])
    if 0 < len(m_shape) <= 4:
        ne = list(reversed(m_shape))
        while len(ne) < 4:
            ne.append(1)
        p = _perm_args(m_order, len(m_shape))
        src = f"ggml_reshape_4d(m, ggml_cont(m, {a}), {ne[0]}, {ne[1]}, {ne[2]}, {ne[3]})"
        return ctx.out(node, f"ggml_cont(m, ggml_permute(m, {src}, {p[0]}, {p[1]}, {p[2]}, {p[3]}))",
                       hint=hint)

    # ③ 인접 전치 분해 + 규약 정규화
    steps = _permute_steps(o, [int(d) for d in in_shape])
    if steps:
        expr = a
        for (A, x, y, B) in steps:
            # torch [A, x, y, B] → ggml ne 역순 [B, y, x, A]; torch swap(1,2) = perm(0,2,1,3)
            expr = (f"ggml_cont(m, ggml_permute(m, ggml_reshape_4d(m, ggml_cont(m, {expr}), "
                    f"{B}, {y}, {x}, {A}), 0, 2, 1, 3))")
        osh = [int(in_shape[i]) for i in o]        # permute 결과 torch shape
        head = 1
        for d in osh[:len(osh) - 3]:
            head *= d
        ne = list(reversed([head] + osh[len(osh) - 3:]))
        expr = f"ggml_reshape_4d(m, ggml_cont(m, {expr}), {ne[0]}, {ne[1]}, {ne[2]}, {ne[3]})"
        return ctx.out(node, expr, hint=hint)

    return ctx.out(node, f"ggml_cont(m, {a}) /* TODO(ggml): {ndim}D permute order={list(order)} */",
                   hint=hint)


def _reshape_to_out(node, ctx, hint):
    """정적 출력 shape 로 ggml_reshape_Nd (reshape/unsqueeze 공용).

    >4D(예: transformer MHA 의 [B,N,3,H,D]) 면 **크기-1 축을 squeeze** 해 ≤4D 로 낮춘다.
    크기-1 차원은 flat 원소 순서에 영향이 없으므로 squeeze 는 수치 불변(정확). squeeze 후에도
    >4D 면 표현 불가 → cont 폴백.

    ⚠️ **어느 크기-1 축을 버리느냐가 broadcast 를 좌우한다.** ggml ne 는 torch shape 의 역순이라
    torch **뒤쪽** 크기-1 은 `ne0`/`ne1`(가장 빠른 축)이 되고, `ggml_can_repeat` 은 ne 를 축별로
    비교하므로 이걸 버리면 실제 축이 통째로 앞당겨져 broadcast 상대가 어긋난다.
    반대로 torch **앞쪽**(바깥) 크기-1 은 ne 의 높은 슬롯이고 ggml 이 빈 상위 슬롯을 1 로 채우므로
    버려도 의미가 보존된다. → **앞쪽부터 버린다.**
    실제 사례(resnest Split-Attention): `atten.view(B, radix, -1, 1, 1)` = torch [1,2,64,1,1].
      전부 squeeze → [2,64]      → ne [64,2,1,1]  ✗ (x=[128,128,64,2] 와 축이 안 맞음)
      앞쪽만 squeeze → [2,64,1,1] → ne [1,1,64,2]  ✓
    앞 레이어는 `128 % 64 == 0` 이라 assert 를 통과하며 **조용히 틀린 값**을 내고, 뒤 레이어에서
    `64 % 128 ≠ 0` 으로 터진다 — 크래시 지점이 원인 지점이 아니다.
    """
    sh = out_shape(node)
    a = ctx.inp(node)
    if not sh:
        return ctx.out(node, f"ggml_cont(m, {a}) /* reshape: shape 미상 */", hint=hint)
    sh2 = sh
    if len(sh2) > 4:
        sh2 = [d for d in sh if int(d) != 1]  # 크기-1 squeeze
    # ⚠️ 이 순서(squeeze 먼저, 병합은 그래도 >4D 일 때만)를 바꾸지 마라.
    #    "병합 우선" 도 "rank 보존 우선" 도 시도했으나 **둘 다 `pvt` 를 cos 1.0 → 0.87 로
    #    회귀시켰다**(2026-08-03 실측, 회귀세트 `pvt·detr·resnest·ssd·yolo·regnet`).
    #    Swin 의 `mask.unsqueeze(1).unsqueeze(0)`=[1,361,1,49,49] 이 squeeze 로 3D 가 돼
    #    head broadcast 축을 잃는 문제는 남아 있다 — **이 함수가 아니라 소비하는 이항 연산에서**
    #    좁게 고쳐야 한다(`sub.py` 의 양방향 broadcast 처리와 같은 방식).
    if len(sh2) > 4:
        # 아직 >4D → **바깥 축들을 하나로 병합**한다. 연속 메모리에서 인접 축 병합은 flat
        # 원소 순서를 바꾸지 않으므로 수치 불변이고, cont 폴백(형태를 통째로 포기)보다 낫다.
        # 예) Swin qkv [3,361,3,49,32] → [1083,3,49,32]. 뒤따르는 `select(dim=0)` 이
        # 이 병합을 전제로 view 를 낸다(`select_render.render_select`).
        head = [int(d) for d in sh2[:len(sh2) - 3]]
        n = 1
        for d in head:
            n *= d
        sh2 = [n] + [int(d) for d in sh2[len(sh2) - 3:]]
    if len(sh2) > 4 or len(sh2) == 0:
        return ctx.out(node, f"ggml_cont(m, {a}) /* reshape to {sh} (>4D) */", hint=hint)
    _, args = _ne_args(sh2)
    # ggml_reshape 는 contiguous 입력 필요 → slice/view/permute 출력이 들어오면 assert.
    # cont 로 감싸 안전하게(이미 contiguous 면 복사 비용만, 수치 불변). attention 의 reshape(slice) 등.
    return ctx.out(node, f"ggml_reshape_{len(sh2)}d(m, ggml_cont(m, {a}), {args})", hint=hint)

def _permute_expr(a, order, ndim):
    # ≤4D 전용. >4D 는 render_transpose/render_permute 가 _render_perm_general 로 처리.
    p = _perm_args(order, ndim)
    return f"ggml_cont(m, ggml_permute(m, {a}, {p[0]}, {p[1]}, {p[2]}, {p[3]}))"


def _in_shape(node, i=0):
    ins = [t for t in (getattr(node, "in_tensors", None) or []) if t is not None]
    if i < len(ins):
        try:
            return [int(x) for x in ins[i].shape]
        except Exception:
            return None
    return None


# ----------------------------------------------------------- 이동/전치
@_rr(_OP.TRANSPOSE)
def render_transpose(node, ctx):
    sh = out_shape(node)
    ndim = len(sh) if sh else 4
    d = ctx.attr(node, "order", [-2, -1])  # 교환할 두 축
    order = list(range(ndim))
    d0, d1 = int(d[0]) % ndim, int(d[1]) % ndim
    order[d0], order[d1] = order[d1], order[d0]
    # >4D(전이 5D, 예: ShuffleNet channel-shuffle) → 블록 축약 ≤4D permute(일반 변환).
    insh = _in_shape(node)
    if ndim > 4 and insh and len(insh) == ndim:
        return _render_perm_general(node, ctx, order, insh, "shf")
    return ctx.out(node, _permute_expr(ctx.inp(node), order, ndim), hint="t")


@_rr(_OP.PERMUTE)
def render_permute(node, ctx):
    sh = out_shape(node)
    ndim = len(sh) if sh else 4
    order = ctx.attr(node, "order", list(range(ndim)))
    insh = _in_shape(node)
    if ndim > 4 and insh and len(insh) == ndim:   # >4D → 블록 축약 ≤4D permute
        return _render_perm_general(node, ctx, list(order), insh, "perm")
    return ctx.out(node, _permute_expr(ctx.inp(node), order, ndim), hint="perm")


@_rr(_OP.CONTIGUOUS)
def render_contiguous(node, ctx):
    return ctx.out(node, f"ggml_cont(m, {ctx.inp(node)})", hint="cont")


@_rr(_OP.DETACH)
def render_detach(node, ctx):
    var = ctx.inp(node)          # graph no-op
    ctx.bind(node, var)
    return var


@_rr(_OP.CAST)
def render_cast(node, ctx):
    var = ctx.inp(node)          # 전 경로 단일 dtype → no-op
    ctx.bind(node, var)
    return var


# ----------------------------------------------------------- reshape / stack
@_rr(_OP.RESHAPE)
def render_reshape(node, ctx):
    return _reshape_to_out(node, ctx, hint="rs")


@_rr(_OP.UNSQUEEZE)
def render_unsqueeze(node, ctx):
    return _reshape_to_out(node, ctx, hint="uns")


# squeeze/unflatten 도 정적 out_shape 기반 reshape 다 — `_reshape_to_out` 을 그대로 쓴다.
# 없으면 unhandled 로 떨어져 축이 안 바뀐 채 흘러가고, 뒤따르는 matmul 이 `can_mul_mat` 으로 죽는다.
@_rr(_OP.SQUEEZE, "squeeze")
def render_squeeze(node, ctx):
    return _reshape_to_out(node, ctx, hint="sq")


@_rr("aten::unflatten", "unflatten")
def render_unflatten(node, ctx):
    return _reshape_to_out(node, ctx, hint="unf")


@_rr(_OP.EXPAND, "expand", _OP.EXPAND_AS, "expand_as")
def render_expand(node, ctx):
    sh = out_shape(node)
    a = ctx.inp(node)
    if not sh or len(sh) > 4:
        return ctx.out(node, f"ggml_cont(m, {a}) /* expand to {sh} (>4D best-effort) */", hint="exp")
    _, args = _ne_args(sh)
    ref = f"ggml_new_tensor_{len(sh)}d(m, GGML_TYPE_F32, {args})"
    return ctx.out(node, f"ggml_repeat(m, {a}, {ref})", hint="exp")   # 타깃 shape 로 브로드캐스트


@_rr(_OP.STACK)
def render_stack(node, ctx):
    # stack(tensors, dim): 각 입력을 stack축=1 로 reshape 후 ggml_concat.
    sh = out_shape(node)
    ins = [t for t in node.in_tensors if t is not None]
    vars_ = [ctx.inp(node, i) for i in range(len(ins))]
    if not sh or len(sh) > 4 or not vars_:
        return ctx.out(node, f"ggml_cont(m, {vars_[0] if vars_ else 'x'}) /* stack {sh} */", hint="stk")
    # 어느 축이 새로 삽입됐는지: out 에서 크기==len(ins) 인 축을 stack dim 으로 본다.
    dim = next((i for i, d in enumerate(sh) if int(d) == len(ins)), len(sh) - 1)
    per = [d if i != dim else 1 for i, d in enumerate(sh)]
    _, args = _ne_args(per)
    gdim = ggml_axis(dim, len(sh))
    parts = [f"ggml_reshape_{len(sh)}d(m, {v}, {args})" for v in vars_]
    expr = parts[0]
    for p in parts[1:]:
        expr = f"ggml_concat(m, {expr}, {p}, {gdim})"
    return ctx.out(node, expr, hint="stk")


@_rr(_OP.REPEAT)
def render_repeat(node, ctx):
    # ggml_repeat(a, ref): a 를 ref 의 ne 로 broadcast. ref 는 출력 shape 의 0텐서.
    sh = out_shape(node)
    a = ctx.inp(node)
    if not sh or len(sh) > 4:
        return ctx.out(node, f"ggml_cont(m, {a}) /* repeat {sh} */", hint="rep")
    _, args = _ne_args(sh)
    ref = f"ggml_new_tensor_{len(sh)}d(m, GGML_TYPE_F32, {args})"
    return ctx.out(node, f"ggml_repeat(m, {a}, {ref})", hint="rep")


@_rr(_OP.SHAPE)
def render_shape(node, ctx):
    # shape(dim) → 정적 정수. 텐서가 아니므로 입력에 바인딩(소비처는 ne 를 직접 씀).
    var = ctx.inp(node)
    ctx.bind(node, var)
    return var


# ----------------------------------------------------------- 텐서 생성
def _num(v, default):
    return float(v) if isinstance(v, (int, float, bool)) else float(default)


def _fill_expr(sh, v):
    """정적 shape 의 상수 텐서 → ggml_fill(new_tensor_Nd, v). >4D 는 1D 스칼라로 폴백."""
    if sh and len(sh) <= 4:
        _, args = _ne_args(sh)
        return f"ggml_fill(m, ggml_new_tensor_{len(sh)}d(m, GGML_TYPE_F32, {args}), {v}f)"
    return f"ggml_fill(m, ggml_new_tensor_1d(m, GGML_TYPE_F32, 1), {v}f)"


@_rr(_OP.ARANGE)
def render_arange(node, ctx):
    sh = out_shape(node)
    n = sh[0] if sh else 0
    start = _num(ctx.attr(node, "start"), 0.0)
    step = _num(ctx.attr(node, "step"), 1.0) or 1.0
    stop = _num(ctx.attr(node, "end"), start + n * step)
    return ctx.out(node, f"ggml_arange(m, {start}f, {stop}f, {step}f)", hint="ar")


@_rr(_OP.LINSPACE)
def render_linspace(node, ctx):
    sh = out_shape(node)
    n = int(_num(ctx.attr(node, "steps"), sh[0] if sh else 2))
    start = _num(ctx.attr(node, "start"), 0.0)
    end = _num(ctx.attr(node, "end"), 1.0)
    step = (end - start) / (n - 1) if n > 1 else 1.0
    # ggml_arange 는 [start, stop) 반개구간 → 마지막 원소가 포함되도록 stop 을 반보 넘긴다.
    return ctx.out(node, f"ggml_arange(m, {start}f, {end + step / 2.0}f, {step}f)", hint="lin")


@_rr(_OP.CONST)
def render_const(node, ctx):
    # const 텐서(anchor seed). 정적 shape 의 영텐서로 emit (값은 GGUF/런타임 주입 대상).
    sh = out_shape(node)
    if sh and len(sh) <= 4:
        _, args = _ne_args(sh)
        return ctx.out(node, f"ggml_new_tensor_{len(sh)}d(m, GGML_TYPE_F32, {args})", hint="const")
    return ctx.out(node, "ggml_new_tensor_1d(m, GGML_TYPE_F32, 1)", hint="const")


@_rr("aten::full", "full", _OP.FULL)
def render_full(node, ctx):
    # full/zeros/new_ones/fill_ 공용: 정적 shape 텐서를 ggml_fill 로 상수 채움.
    v = _num(ctx.attr(node, "fill_value"), 0.0)
    return ctx.out(node, _fill_expr(out_shape(node), v), hint="full")


@_rr(_OP.ZEROS)
def render_zeros(node, ctx):
    return ctx.out(node, _fill_expr(out_shape(node), 0.0), hint="zeros")


@_rr(_OP.ROLL)
def render_roll(node, ctx):
    sh = out_shape(node)
    ndim = len(sh) if sh else 4
    shifts = ctx.attr(node, "shifts", [])
    dims = ctx.attr(node, "dims", [])
    if not isinstance(shifts, (list, tuple)):
        shifts = [shifts]
    if not isinstance(dims, (list, tuple)):
        dims = [dims]
    if not dims:
        # dims 없는 torch.roll 은 평탄화 후 회전 → ggml_roll(축별)로 표현 불가.
        return ctx.out(node, f"ggml_cont(m, {ctx.inp(node)})"
                             " /* TODO(ggml): roll without dims (flattened) */", hint="roll")
    s = [0, 0, 0, 0]
    for amount, d in zip(shifts, dims):
        s[ggml_axis(d, ndim)] = int(amount)
    return ctx.out(node, f"ggml_roll(m, {ctx.inp(node)}, {s[0]}, {s[1]}, {s[2]}, {s[3]})",
                   hint="roll")


@_rr("aten::meshgrid", "meshgrid")
def render_meshgrid(node, ctx):
    # meshgrid(x, y) → 두 2D 그리드. 각 출력 = 1D arange 를 출력 shape 로 broadcast.
    sh = out_shape(node)
    a = ctx.inp(node, 0)
    if sh and len(sh) <= 4:
        _, args = _ne_args(sh)
        ref = f"ggml_new_tensor_{len(sh)}d(m, GGML_TYPE_F32, {args})"
        var = ctx.out(node, f"ggml_repeat(m, {a}, {ref})", hint="mg")
    else:
        var = ctx.out(node, f"ggml_cont(m, {a})", hint="mg")
    ctx.bind_outputs(node, var)
    return var


@_rr("dropout", "aten::dropout", "aten::dropout_")
def render_dropout(node, ctx):
    """추론 그래프에서 dropout 은 항등(no-op) — 입력 var 를 그대로 바인딩한다.

    렌더러가 없으면 `unhandled op` 로 남아 passthrough 되는데, 그 자체는 값이 같지만
    **새 변수를 만들어 바인딩**하므로 뒤따르는 shape 추론이 어긋날 수 있다.
    Swin 계열에서 49건이 이 상태였다.
    """
    var = ctx.inp(node)
    ctx.bind(node, var)
    return var
