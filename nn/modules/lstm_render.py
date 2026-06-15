"""LSTM 의 ggml(vision.cpp) C++ render — 시퀀스 정적 unroll.

PyTorch nn.LSTM → ggml 그래프. 시퀀스 루프는 codegen(Python) 에서 정적 unroll 하여
time-step 마다 게이트 연산을 ggml primitive 로 emit 한다(C++ 루프 아님).

레이아웃(RNN arch 는 codegen 이 cwhn 변환을 생략):
  입력 x   ne = [feat, seq, batch]   (torch [batch,seq,feat] 의 역순)
  weight   ne = [in, 4H](weight_ih_l0) / [H, 4H](weight_hh_l0)  (GGUF 2D, transpose 없음)
  bias     ne = [4H]
게이트 chunk 순서(PyTorch): i, f, g, o. 게이트 사전합:
  pre = mul_mat(W_ih, x_t) + b_ih + b_hh + mul_mat(W_hh, h_{t-1})
  i=σ, f=σ, g=tanh, o=σ;  c = f*c_{t-1} + i*g;  h = o*tanh(c)

현재 범위: 단방향 1층, h0=c0=0(t=0 특수 처리로 내장 — 별도 zero 텐서 불필요).
멀티레이어/양방향/GRU 는 후속 phase.
"""
from shared.compile.render_api import register_render as _rr
from shared.compile.render_api import out_shape


def _wkey(t):
    """in_tensor → GGUF 텐서 키(= m.weights 인자). 'LSTM::weight_ih_l0' → 'weight_ih_l0'."""
    nm = getattr(t, "name", "") or ""
    return nm.split("::")[-1]


@_rr("aten::lstm")
def render(node, ctx):
    ins = list(getattr(node, "in_tensors", []) or [])
    # ins = [x, h0, c0, w_ih_l0, w_hh_l0, b_ih_l0, b_hh_l0]  (단방향 1층, bias)
    x = ctx.inp(node, 0)
    x_shape = [int(d) for d in (ins[0].shape or [])]      # torch [batch, seq, feat]
    batch, seq, feat = x_shape[0], x_shape[1], x_shape[2]
    w_ih = ins[3]
    H = int(w_ih.shape[0]) // 4                            # weight_ih_l0: [4H, in]

    wih = f'm.weights("{_wkey(ins[3])}")'
    whh = f'm.weights("{_wkey(ins[4])}")'
    bih = f'm.weights("{_wkey(ins[5])}")'
    bhh = f'm.weights("{_wkey(ins[6])}")'

    nv = ctx.new_var
    L = ctx.line

    h_var = None
    c_var = None
    h_steps = []
    for t in range(seq):
        # x_t = x[:, t, :]  → ne [feat, batch]
        xt = nv("xt")
        L(f"    tensor {xt} = ggml_cont(m, ggml_view_2d(m, {x}, {feat}, {batch}, "
          f"{x}->nb[2], (size_t){t}*{x}->nb[1]));")
        # 게이트 사전합 pre = W_ih·x_t + b_ih + b_hh (+ W_hh·h_{t-1})
        g = nv("g")
        L(f"    tensor {g} = ggml_add(m, ggml_mul_mat(m, {wih}, {xt}), "
          f"ggml_add(m, {bih}, {bhh}));")
        if t > 0:
            L(f"    {g} = ggml_add(m, {g}, ggml_mul_mat(m, {whh}, {h_var}));")

        def _chunk(k, name):
            v = nv(name)
            L(f"    tensor {v} = ggml_cont(m, ggml_view_2d(m, {g}, {H}, {batch}, "
              f"{g}->nb[1], (size_t){k}*{H}*sizeof(float)));")
            return v

        i_g = _chunk(0, "ig")
        g_g = _chunk(2, "gg")
        o_g = _chunk(3, "og")
        si = nv("si"); L(f"    tensor {si} = ggml_sigmoid(m, {i_g});")
        tg = nv("tg"); L(f"    tensor {tg} = ggml_tanh(m, {g_g});")
        so = nv("so"); L(f"    tensor {so} = ggml_sigmoid(m, {o_g});")

        c_new = nv("c")
        if t == 0:
            # f * c0 = 0 → c = i*g
            L(f"    tensor {c_new} = ggml_mul(m, {si}, {tg});")
        else:
            f_g = _chunk(1, "fg")
            sf = nv("sf"); L(f"    tensor {sf} = ggml_sigmoid(m, {f_g});")
            L(f"    tensor {c_new} = ggml_add(m, ggml_mul(m, {sf}, {c_var}), "
              f"ggml_mul(m, {si}, {tg}));")
        h_new = nv("h")
        L(f"    tensor {h_new} = ggml_mul(m, {so}, ggml_tanh(m, {c_new}));")
        c_var, h_var = c_new, h_new

        hr = nv("hr")
        L(f"    tensor {hr} = ggml_reshape_3d(m, {h_new}, {H}, 1, {batch});")
        h_steps.append(hr)

    # output 시퀀스: concat h_t on seq 축 → ne [H, seq, batch]
    out = h_steps[0]
    for hr in h_steps[1:]:
        nx = nv("seqcat")
        L(f"    tensor {nx} = ggml_concat(m, {out}, {hr}, 1);")
        out = nx
    ctx.bind_outputs(node, out)
    return out
