"""GRU 의 ggml(vision.cpp) C++ render — 시퀀스 정적 unroll.

LSTM(lstm_render) 과 동일한 unroll/레이어/방향 구조. 게이트 수식만 GRU:
  gx = mul_mat(W_ih, x_t) + b_ih           # [3H, batch], chunk r,z,n
  gh = mul_mat(W_hh, h_{t-1}) + b_hh        # [3H, batch], chunk r,z,n
  r = σ(gx_r + gh_r);  z = σ(gx_z + gh_z);  n = tanh(gx_n + r * gh_n)
  h = (1-z)*n + z*h_{t-1} = n + z*(h_{t-1} - n)   # ggml sub/mul/add 만 사용
chunk 순서(PyTorch GRU): r, z, n. h0=0 은 첫 step h_prev=scale(·,0) 으로 일반 경로 처리
(GRU 는 b_hh 가 첫 step 에도 살아 있어 LSTM 식 '항 생략' 불가).
지원: 단/양방향, 멀티레이어. weight 는 GGUF 텐서명 규칙으로 접근.
"""
from shared.compile.render_api import register_render as _rr


def _wkey(t):
    nm = getattr(t, "name", "") or ""
    return nm.split("::")[-1]


@_rr("aten::gru")
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
            H = int(t.shape[1])               # torch [3H, H] → H
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
            h_var = None
            h_at = [None] * seq
            first = True
            for t in order:
                xt = nv("xt")
                L(f"    tensor {xt} = ggml_cont(m, ggml_view_2d(m, {layer_in}, "
                  f"{in_size}, {batch}, {layer_in}->nb[2], (size_t){t}*{layer_in}->nb[1]));")
                gx = nv("gx")
                L(f"    tensor {gx} = ggml_add(m, ggml_mul_mat(m, {wih}, {xt}), {bih});")
                if first:
                    h_var = nv("h0")
                    # [H,batch] zero (h0=0): gx 의 r-chunk 를 0배.
                    L(f"    tensor {h_var} = ggml_scale(m, ggml_cont(m, ggml_view_2d(m, {gx}, "
                      f"{H}, {batch}, {gx}->nb[1], (size_t)0)), 0.0f);")
                gh = nv("gh")
                L(f"    tensor {gh} = ggml_add(m, ggml_mul_mat(m, {whh}, {h_var}), {bhh});")

                def _ch(src, k, name):
                    v = nv(name)
                    L(f"    tensor {v} = ggml_cont(m, ggml_view_2d(m, {src}, {H}, {batch}, "
                      f"{src}->nb[1], (size_t){k}*{H}*sizeof(float)));")
                    return v

                gxr, gxz, gxn = _ch(gx, 0, "gxr"), _ch(gx, 1, "gxz"), _ch(gx, 2, "gxn")
                ghr, ghz, ghn = _ch(gh, 0, "ghr"), _ch(gh, 1, "ghz"), _ch(gh, 2, "ghn")
                r = nv("r"); L(f"    tensor {r} = ggml_sigmoid(m, ggml_add(m, {gxr}, {ghr}));")
                z = nv("z"); L(f"    tensor {z} = ggml_sigmoid(m, ggml_add(m, {gxz}, {ghz}));")
                n = nv("n")
                L(f"    tensor {n} = ggml_tanh(m, ggml_add(m, {gxn}, ggml_mul(m, {r}, {ghn})));")
                # h = n + z*(h_prev - n)
                h_new = nv("h")
                L(f"    tensor {h_new} = ggml_add(m, {n}, ggml_mul(m, {z}, "
                  f"ggml_sub(m, {h_var}, {n})));")
                h_var, first = h_new, False
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
