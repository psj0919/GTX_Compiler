"""select/unbind/split 계열 render — deploy 브랜치에서 이식.

5D↑ 텐서의 select 는 **바깥 축 병합 + ggml_view_4d 오프셋 view** 로 표현한다
(ggml ne 는 4D 가 최대. 병합은 연속 메모리에서 원소 순서 불변이라 수치 불변).
main 에는 이 렌더러가 없어 Swin 계열에서 select 36건이 passthrough 로 흘렀다.
"""

from shared.compile.render_api import register_render as _rr, out_shape, ggml_axis
from shared.base import OP as _OP


def _in_shape(node, i=0):
    """입력 i 의 정적 torch shape(list[int]) 또는 None."""
    ins = [t for t in node.in_tensors if t is not None]
    if i >= len(ins):
        return None
    try:
        return [int(d) for d in ins[i].shape]
    except Exception:
        return None


@_rr(_OP.SELECT)
def render_select(node, ctx):
    a = ctx.inp(node)
    osh = out_shape(node)
    ish = _in_shape(node)
    dim = ctx.attr(node, "dim", 0)
    index = ctx.attr(node, "index", 0)

    # 5D 입력의 dim=0 select — Swin/DETR 계열의 qkv 분리가 여기 해당한다.
    # `_reshape_to_out` 이 5D 를 **바깥 축 병합**으로 4D 화하므로(예: [3,361,3,49,32] →
    # [1083,3,49,32]), 원본 dim0 의 idx k 는 병합축의 [k*ish1, (k+1)*ish1) 연속 구간이다
    # → 그 구간을 그대로 view 한다. 이게 없으면 select 가 passthrough 로 렌더돼
    # q/k/v 가 안 갈라지고 attention score 가 통째로 틀린다(swin·glip·grounding_dino).
    if ish and len(ish) == 5 and int(dim) == 0 and osh and len(osh) == 4:
        try:
            k = int(index) % int(ish[0])
            rows = int(ish[1])                       # 원본 dim0 한 칸이 차지하는 병합축 길이
            ne = [int(ish[4]), int(ish[3]), int(ish[2]), rows]   # ggml ne = torch 역순
            return ctx.out(
                node,
                f"ggml_view_4d(m, {a}, {ne[0]}, {ne[1]}, {ne[2]}, {ne[3]}, "
                f"{a}->nb[1], {a}->nb[2], {a}->nb[3], (size_t){k * rows}*{a}->nb[3])",
                hint="sel",
            )
        except Exception:
            pass

    # 정적 정보 부족 / 표현 불가(>4D, 0D) → best-effort cont.
    if not osh or not ish or len(ish) > 4 or len(osh) > 4:
        return ctx.out(
            node,
            f"ggml_cont(m, {a}) /* TODO(ggml): select dim={dim} idx={index} on {ish} */",
            hint="sel",
        )

    ndim_in = len(ish)
    d = int(dim) % ndim_in
    g = ggml_axis(d, ndim_in)              # 선택 축(ggml ne 인덱스)
    idx = int(index)
    if idx < 0:
        idx += int(ish[d])

    out_ne = list(reversed([int(x) for x in osh]))   # ne0..ne(out-1)
    n_out = len(out_ne)
    ne_args = ", ".join(str(n) for n in out_ne)

    # 출력 ggml 축 j → 입력 ggml 축(선택 축 g 건너뜀). ggml_view_Nd 는 nb1..nb(n_out-1) 만 받음.
    nb_terms = []
    for j in range(1, n_out):
        in_ax = j if j < g else j + 1
        nb_terms.append(f"{a}->nb[{in_ax}]")
    nb_str = (", ".join(nb_terms) + ", ") if nb_terms else ""

    offset = f"(size_t){idx} * {a}->nb[{g}]"
    return ctx.out(node, f"ggml_view_{n_out}d(m, {a}, {ne_args}, {nb_str}{offset})", hint="sel")


@_rr(_OP.UNBIND)
def render_unbind(node, ctx):
    """unbind(x, dim) → dim 축을 따라 N개 select. 각 출력 텐서를 개별 view 로 바인딩(multi-output).

    timm transformer 의 `q,k,v = qkv.unbind(0)` 패턴. SELECT 와 동일한 view(축 제거 + byte offset)를
    출력 개수만큼 emit 하고, 각 out_tensor 를 그 view 변수에 바인딩한다.
    """
    a = ctx.inp(node)
    ish = _in_shape(node)
    dim = ctx.attr(node, "dim", 0)
    outs = [t for t in (node.out_tensors or []) if t is not None]
    # 입력이 >4D(예: timm qkv [3,B,heads,N,dim])면, 앞단 reshape/permute 와 동일하게 크기-1 축을
    # squeeze 해 ≤4D 로 맞춘다(실제 ggml 텐서 a 는 이미 squeeze 됨). 선택 dim 도 재매핑.
    if ish and len(ish) > 4:
        d0 = int(dim) % len(ish)
        keep = [ax for ax in range(len(ish)) if int(ish[ax]) != 1]
        if d0 in keep and len(keep) <= 4:
            dim = keep.index(d0)
            ish = [int(ish[ax]) for ax in keep]
    if not ish or len(ish) > 4 or not outs:
        for t in outs:
            ctx._var[id(t)] = a   # best-effort
        ctx.bind(node, a)
        return a
    ndim_in = len(ish)
    d = int(dim) % ndim_in
    g = ggml_axis(d, ndim_in)                     # 선택 축(ggml ne)
    out_ne = list(reversed([int(x) for x in (ish[:d] + ish[d + 1:])]))  # 축 제거 후 ggml ne
    n_out = len(out_ne)
    ne_args = ", ".join(str(n) for n in out_ne)
    nb_terms = []
    for j in range(1, n_out):
        in_ax = j if j < g else j + 1
        nb_terms.append(f"{a}->nb[{in_ax}]")
    nb_str = (", ".join(nb_terms) + ", ") if nb_terms else ""
    last = a
    for i, t in enumerate(outs):
        offset = f"(size_t){i} * {a}->nb[{g}]"
        var = ctx.new_var("ub")
        ctx.line(f"    tensor {var} = ggml_cont(m, "
                 f"ggml_view_{n_out}d(m, {a}, {ne_args}, {nb_str}{offset}));")
        ctx._var[id(t)] = var
        nm = getattr(t, "name", None)
        if nm is not None:
            ctx._var_by_name[nm] = var   # id() 재사용 대비 안정적 이름 키
        last = var
    ctx.bind(node, last)
    return last


@_rr(_OP.SPLIT)
def render_split(node, ctx):
    """split_with_sizes(x, sizes, dim) → dim 축을 sizes[i] 로 자른 N개 view(축 유지, 누적 offset).

    unbind 와 달리 선택 축을 제거하지 않고 크기만 자른다. nn.MultiheadAttention 의
    in_proj_weight[3E,E] → q/k/v[E,E] 분리 등. 각 out_tensor 를 view 변수에 바인딩(multi-output).
    """
    a = ctx.inp(node)
    ish = _in_shape(node)
    dim = ctx.attr(node, "dim", 0)
    sizes = ctx.attr(node, "split_sizes", None)
    outs = [t for t in (node.out_tensors or []) if t is not None]
    # >4D 는 크기-1 축 squeeze(상류 텐서 a 는 이미 squeeze 됨), 선택 dim 재매핑.
    if ish and len(ish) > 4:
        d0 = int(dim) % len(ish)
        keep = [ax for ax in range(len(ish)) if int(ish[ax]) != 1]
        if d0 in keep and len(keep) <= 4:
            dim = keep.index(d0)
            ish = [int(ish[ax]) for ax in keep]
    if not ish or len(ish) > 4 or not outs:
        for t in outs:
            ctx._var[id(t)] = a
        ctx.bind(node, a)
        return a
    ndim = len(ish)
    d = int(dim) % ndim
    g = ggml_axis(d, ndim)                          # 자를 축(ggml ne)
    if not sizes or len(sizes) != len(outs):
        # 크기 정보 없으면 out_tensor shape 에서 축 g 크기 추론.
        sizes = []
        for t in outs:
            try:
                sizes.append(int(t.shape[d]))
            except Exception:
                sizes = None; break
    if not sizes:
        for t in outs:
            ctx._var[id(t)] = a
        ctx.bind(node, a)
        return a
    base_ne = list(reversed([int(x) for x in ish]))  # 입력 ggml ne
    nb_terms = [f"{a}->nb[{j}]" for j in range(1, ndim)]
    nb_str = (", ".join(nb_terms) + ", ") if nb_terms else ""
    last, cum = a, 0
    for i, t in enumerate(outs):
        ne = list(base_ne); ne[g] = int(sizes[i])
        ne_args = ", ".join(str(n) for n in ne)
        offset = f"(size_t){cum} * {a}->nb[{g}]"
        var = ctx.new_var("sp")
        ctx.line(f"    tensor {var} = ggml_cont(m, "
                 f"ggml_view_{ndim}d(m, {a}, {ne_args}, {nb_str}{offset}));")
        ctx._var[id(t)] = var
        nm = getattr(t, "name", None)
        if nm is not None:
            ctx._var_by_name[nm] = var
        cum += int(sizes[i])
        last = var
    ctx.bind(node, last)
    return last


@_rr(_OP.STRIDED_SLICE_INPLACE_COPY)
def render_sslice_inplace_copy(node, ctx):
    """dest[..., strided-slice] = src  (예: pose kpts decode dest[:,0::3,:]=x, [:,1::3,:]=y).

    `ggml_set(dest, src, nb1, nb2, nb3, offset)` 로 **strided scatter** 를 표현한다 —
    step!=1 축은 view stride = step*dest->nb[axis], start 는 byte offset 으로. src(b) 가 그 영역을 채운다.
    ggml_set 은 변형된 dest 를 반환하므로, **dest 텐서의 downstream 바인딩을 결과로 갱신**해
    in-place 변형이 하류(concat 등)에 전파되게 한다(함수형 IR 의 in-place 전파 공백 메움).
    같은 dest 를 쓰는 연속 scatter(x→y)는 이 재바인딩으로 자연히 체이닝된다.

    표현 불가(정보 부족/>4D/step 이 ggml ne0 축에 걸림)면 best-effort: 출력=src 바인딩(슬라이스 값 정확,
    단 하류는 변형 전 dest 를 봄).
    """
    ins = [t for t in node.in_tensors if t is not None]
    src = ctx.inp(node, len(ins) - 1)            # 마지막 입력 = source
    dest_t = ins[0] if ins else None
    dest = ctx.inp(node, 0)
    dsh = _in_shape(node, 0)

    dims = ctx.attr(node, "dim", None)
    starts = ctx.attr(node, "start", None)
    steps = ctx.attr(node, "step", None)

    def _aslist(v):
        return list(v) if isinstance(v, (list, tuple)) else ([v] if v is not None else [])

    dims, starts, steps = _aslist(dims), _aslist(starts), _aslist(steps)

    ok = dest_t is not None and dsh and len(dsh) <= 4 and dims and len(starts) == len(dims) and len(steps) == len(dims)
    if ok:
        nd = len(dsh)
        step_by_g, off_terms = {}, []
        for d, s, st in zip(dims, starts, steps):
            g = ggml_axis(int(d), nd)            # ggml ne 축
            st = int(st)
            if st != 1:
                if g == 0:
                    ok = False                   # ne0(연속) 축의 step 은 ggml_set 으로 표현 불가
                    break
                step_by_g[g] = st
            if int(s) != 0:
                off_terms.append(f"(size_t){int(s)} * {dest}->nb[{g}]")
    if ok:
        def nb(k):
            st = step_by_g.get(k, 1)
            return f"{st} * {dest}->nb[{k}]" if st != 1 else f"{dest}->nb[{k}]"
        offset = " + ".join(off_terms) if off_terms else "0"
        var = ctx.new_var("iset")
        ctx.line(f"    tensor {var} = ggml_set(m, {dest}, {src}, {nb(1)}, {nb(2)}, {nb(3)}, {offset});")
        ctx._var[id(dest_t)] = var               # dest 의 하류 사용을 변형 결과로 전파(체이닝)
        ctx.bind(node, src)                      # 이 op 출력(슬라이스 영역)의 값 = src
        return src

    # best-effort 폴백: 슬라이스 값만 정확(하류 전파 없음).
    ctx.bind(node, src)
    return src


# ── deploy 이식: im2col(F.unfold) ─────────────────────────────
@_rr(_OP.IM2COL, "im2col", "aten::im2col")
def render_im2col(node, ctx):
    """`nn.Unfold` → `ggml_im2col`.

    ggml 시그니처: `ggml_im2col(ctx, a, b, s0,s1, p0,p1, d0,d1, is_2D, dst_type)`.
    `a` 는 **커널 shape 만** 쓰이므로(가중치 값은 안 봄) `[kw,kh,C,1]` 짜리 더미를 만들어 넘긴다.
    ⚠️ 출력 축 순서가 torch `unfold` 와 다를 수 있다 — 넣은 뒤 반드시 수치 대조할 것.
    """
    x = ctx.inp(node, 0)

    def _pair(v, d):
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                return int(v[0]), int(v[1])
            except Exception:
                return d, d
        try:
            n = int(v)
            return n, n
        except Exception:
            return d, d

    kh, kw = _pair(ctx.attr(node, "kernel_size", None), 2)
    sh, sw = _pair(ctx.attr(node, "stride", None), kh)
    ph, pw = _pair(ctx.attr(node, "padding", None), 0)
    dh, dw = _pair(ctx.attr(node, "dilation", None), 1)
    c = 1
    try:
        ish = [int(d) for d in [t for t in node.in_tensors if t is not None][0].shape]
        c = int(ish[1]) if len(ish) >= 2 else 1
    except Exception:
        c = 1
    n = kw * kh * c
    dummy = (f"ggml_reshape_4d(m, ggml_scale(m, ggml_arange(m, 0.0f, {float(n)}f, 1.0f), 0.0f), "
             f"{kw}, {kh}, {c}, 1)")
    expr = f"ggml_im2col(m, {dummy}, {x}, {sw}, {sh}, {pw}, {ph}, {dw}, {dh}, true, GGML_TYPE_F32)"

    # ⚠️ **축 순서가 torch 와 다르다.**
    #   ggml_im2col ne = [C*kh*kw, OW, OH, N]   (= torch [N, OH, OW, C*kh*kw])
    #   torch unfold  = [N, C*kh*kw, L]          (L = OH*OW, ow 가 빠른 축)
    # OW·OH 는 ggml 에서 인접·연속이므로 **reshape 로 L 로 합친 뒤 앞 두 축을 교환**하면 된다.
    # (블록 내부 순서는 양쪽 다 kw 가 가장 빠르고 그다음 kh, C — 동일하므로 건드릴 게 없다.)
    osh = [int(d) for d in (out_shape(node) or [])]
    if len(osh) == 3:
        bn, ckk, L = osh
        expr = f"ggml_reshape_4d(m, ggml_cont(m, {expr}), {ckk}, {L}, {bn}, 1)"
        expr = f"ggml_cont(m, ggml_permute(m, {expr}, 1, 0, 2, 3))"   # → ne [L, C*kh*kw, N, 1]
    return ctx.out(node, expr, hint="im2col")
