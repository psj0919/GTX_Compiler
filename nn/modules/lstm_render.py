"""LSTM 의 ggml(vision.cpp) C++ render — 시퀀스 정적 unroll.

PyTorch nn.LSTM → ggml 그래프. 시퀀스 루프는 codegen(Python) 에서 정적 unroll 하여
time-step 마다 게이트 연산을 ggml primitive 로 emit 한다(C++ 루프 아님).

레이아웃(RNN arch 는 codegen 이 cwhn 변환을 생략):
  입력 x   ne = [feat, seq, batch]              (torch [batch,seq,feat] 역순)
  weight   GGUF ne = [in, 4H](ih) / [H, 4H](hh) (2D, transpose 없음; in_tensor.shape 는 torch [4H,in])
  bias     ne = [4H]
게이트 chunk 순서(PyTorch): i, f, g, o.
  pre = mul_mat(W_ih, x_t) + b_ih + b_hh (+ mul_mat(W_hh, h_{t-1}))
  i=σ, f=σ, g=tanh, o=σ;  c = f*c_{t-1} + i*g;  h = o*tanh(c)

지원: 단/양방향, 멀티레이어. weight 는 GGUF 텐서명 규칙(weight_ih_l{n}[_reverse]) 으로 접근.
h0=c0=0 은 각 (layer,dir) 의 첫 step 특수 처리로 내장(별도 zero 텐서 불필요).
미지원(best-effort): h0/c0≠0 입력, h_n/c_n 2·3번째 출력, packed sequence.
"""
from shared.compile.render_api import register_render as _rr


def _wkey(t):
    nm = getattr(t, "name", "") or ""
    return nm.split("::")[-1]


@_rr("aten::lstm")
def render(node, ctx):
    ins = list(getattr(node, "in_tensors", []) or [])
    x = ctx.inp(node, 0)
    xs = [int(d) for d in (ins[0].shape or [])]   # torch [batch, seq, feat]
    batch, seq, feat0 = xs[0], xs[1], xs[2]

    names = [_wkey(t) for t in ins]
    bidir = any(n.endswith("_reverse") for n in names)
    ndir = 2 if bidir else 1
    layer_ids, H = set(), None
    for t in ins:
        n = _wkey(t)
        if n.startswith("weight_ih_l"):
            layer_ids.add(int(n[len("weight_ih_l"):].split("_")[0]))
        if n == "weight_hh_l0":
            H = int(t.shape[1])               # torch [4H, H] → H
    num_layers = (max(layer_ids) + 1) if layer_ids else 1

    nv, L = ctx.new_var, ctx.line
    layer_in = x
    in_size = feat0
    for layer in range(num_layers):
        dir_outs = []
        for d in range(ndir):
            suf = "_reverse" if d == 1 else ""
            wih = f'm.weights("weight_ih_l{layer}{suf}")'
            whh = f'm.weights("weight_hh_l{layer}{suf}")'
            bih = f'm.weights("bias_ih_l{layer}{suf}")'
            bhh = f'm.weights("bias_hh_l{layer}{suf}")'
            order = range(seq) if d == 0 else range(seq - 1, -1, -1)
            h_var = c_var = None
            h_at = [None] * seq
            first = True
            for t in order:
                xt = nv("xt")
                L(f"    tensor {xt} = ggml_cont(m, ggml_view_2d(m, {layer_in}, "
                  f"{in_size}, {batch}, {layer_in}->nb[2], (size_t){t}*{layer_in}->nb[1]));")
                g = nv("g")
                L(f"    tensor {g} = ggml_add(m, ggml_mul_mat(m, {wih}, {xt}), "
                  f"ggml_add(m, {bih}, {bhh}));")
                if not first:
                    L(f"    {g} = ggml_add(m, {g}, ggml_mul_mat(m, {whh}, {h_var}));")

                def _chunk(k, name):
                    v = nv(name)
                    L(f"    tensor {v} = ggml_cont(m, ggml_view_2d(m, {g}, {H}, {batch}, "
                      f"{g}->nb[1], (size_t){k}*{H}*sizeof(float)));")
                    return v

                si = nv("si"); L(f"    tensor {si} = ggml_sigmoid(m, {_chunk(0, 'ig')});")
                tg = nv("tg"); L(f"    tensor {tg} = ggml_tanh(m, {_chunk(2, 'gg')});")
                so = nv("so"); L(f"    tensor {so} = ggml_sigmoid(m, {_chunk(3, 'og')});")
                c_new = nv("c")
                if first:
                    L(f"    tensor {c_new} = ggml_mul(m, {si}, {tg});")
                else:
                    sf = nv("sf")
                    L(f"    tensor {sf} = ggml_sigmoid(m, {_chunk(1, 'fg')});")
                    L(f"    tensor {c_new} = ggml_add(m, ggml_mul(m, {sf}, {c_var}), "
                      f"ggml_mul(m, {si}, {tg}));")
                h_new = nv("h")
                L(f"    tensor {h_new} = ggml_mul(m, {so}, ggml_tanh(m, {c_new}));")
                c_var, h_var, first = c_new, h_new, False
                hr = nv("hr")
                L(f"    tensor {hr} = ggml_reshape_3d(m, {h_new}, {H}, 1, {batch});")
                h_at[t] = hr
            out = h_at[0]
            for t in range(1, seq):
                nx = nv("seqcat")
                L(f"    tensor {nx} = ggml_concat(m, {out}, {h_at[t]}, 1);")
                out = nx
            dir_outs.append(out)
        if ndir == 2:
            lo = nv("dircat")
            L(f"    tensor {lo} = ggml_concat(m, {dir_outs[0]}, {dir_outs[1]}, 0);")
            layer_in = lo
        else:
            layer_in = dir_outs[0]
        in_size = H * ndir
    ctx.bind_outputs(node, layer_in)
    return layer_in
