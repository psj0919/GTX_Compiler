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
"""ggml(libggml.so) 실행 백엔드 — 생성된 export 를 ggml 커널로 돌린다.

`nn.set_backend('ggml')` 후 `nn.Module('CONV2D', ...)` 는 :class:`GgmlModule` 을
반환하며, 각 op 호출 시 ggml-python(= libggml.so ctypes 바인딩)으로 **eager per-op**
계산을 한다. ResNet(분류) + YOLOv8(detection, head 포함) full e2e 지원.

규약: 활성화는 **numpy NCHW(배치 유지)** 로 흐른다(= torch 레이아웃). heavy op
(conv/pool/relu/silu/sigmoid)은 ggml 커널, 나머지(batch_norm/concat/binary/anchor
grid/DFL/box decode = arange/meshgrid/stack/full/strided_slice/transpose/softmax/
unsqueeze/reshape/resize)는 host numpy 로 torch dim 의미 그대로 처리.
가중치는 :func:`bind_gguf` 가 export 주석경로(`Conv2d[conv1]`→`conv1`)로 GGUF
텐서명(`conv1.weight`)을 매칭해 주입한다.
"""

import os
import re

import numpy as np

_BACKEND = "pytorch"


def set_backend(name):
    global _BACKEND
    if name not in ("pytorch", "ggml"):
        raise ValueError(f"unknown backend {name!r} (expected 'pytorch'|'ggml')")
    _BACKEND = name


def get_backend():
    return _BACKEND


# --------------------------------------------------------------------------
# ggml one-shot graph runner (NCHW in/out)
# --------------------------------------------------------------------------
_GGML = None


def _ggml():
    global _GGML
    if _GGML is None:
        import ggml
        import ggml.utils as gutils

        _GGML = (ggml, gutils)
    return _GGML


def _run_nchw(build, mem=512 * 1024 * 1024, n_threads=4):
    """build(ctx, ggml, U) -> ggml tensor. 결과를 numpy NCHW([N,C,H,W])로 반환.

    ggml ne=[W,H,C,N] → numpy reshape [N,C,H,W] (메모리 순서 동일, OC=1 squeeze 우회).
    """
    ggml, U = _ggml()
    p = ggml.ggml_init_params(mem_size=mem, mem_buffer=None, no_alloc=False)
    ctx = ggml.ggml_init(p)
    try:
        out = build(ctx, ggml, U)
        gf = ggml.ggml_new_graph(ctx)
        ggml.ggml_build_forward_expand(gf, out)
        ggml.ggml_graph_compute_with_ctx(ctx, gf, n_threads)
        ne = [int(out.contents.ne[i]) for i in range(4)]  # [W,H,C,N]
        flat = np.asarray(U.to_numpy(out)).reshape(-1).copy()
        return flat.reshape(ne[3], ne[2], ne[1], ne[0])    # [N,C,H,W]
    finally:
        ggml.ggml_free(ctx)


def _f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def _as4d(x):
    """ggml feature-map op 입력을 [N,C,H,W] 로 정규화."""
    x = _f32(x)
    if x.ndim == 3:
        x = x[None]
    return x


# --------------------------------------------------------------------------
# op kernels
# --------------------------------------------------------------------------
def _scalar(v, i=0, d=1):
    if isinstance(v, (list, tuple)):
        return int(v[i]) if len(v) > i else d
    return int(v) if v is not None else d


def _op_conv2d(mod, x):
    x = _as4d(x)                         # [N,C,H,W]
    w = _f32(mod.weights["weight"])      # [OC,IC,KH,KW]
    if w.ndim < 4:
        w = w.reshape((w.shape[0] if w.ndim else 1, -1, 1, 1))
    s = _scalar(mod.attrs.get("stride", [1, 1]))
    p = _scalar(mod.attrs.get("padding", [0, 0]))
    d = _scalar(mod.attrs.get("dilation", [1, 1]))

    def build(ctx, ggml, U):
        xt = U.from_numpy(np.ascontiguousarray(x), ctx)   # ne [W,H,C,N]
        wt = U.from_numpy(np.ascontiguousarray(w), ctx)   # ne [KW,KH,IC,OC]
        return ggml.ggml_conv_2d(ctx, wt, xt, s, s, p, p, d, d)

    out = _run_nchw(build)               # [N,OC,OH,OW]
    b = mod.weights.get("bias")
    if b is not None:
        out = out + _f32(b).reshape(1, -1, 1, 1)
    return out


def _op_matmul(a, b):
    """torch.matmul 을 ggml_mul_mat 으로. 임의 배치 차원은 3D 로 flatten 후 검증된
    경로(B 마지막 두 축 swap → from_numpy → 결과 swap, rel ~2e-8) 사용."""
    a = _f32(a)
    b = _f32(b)
    if a.ndim < 2 or b.ndim < 2:
        return np.matmul(a, b)
    abatch, am, ak = a.shape[:-2], a.shape[-2], a.shape[-1]
    bn = b.shape[-1]
    A = a.reshape(-1, am, ak)
    B = b.reshape(-1, b.shape[-2], bn)
    nb = max(A.shape[0], B.shape[0])
    if A.shape[0] == 1 and nb > 1:
        A = np.broadcast_to(A, (nb, am, ak))
    if B.shape[0] == 1 and nb > 1:
        B = np.broadcast_to(B, (nb, B.shape[1], bn))
    A = np.ascontiguousarray(A)
    Bt = np.ascontiguousarray(np.swapaxes(B, -1, -2))

    def build(ctx, ggml, U):
        return ggml.ggml_mul_mat(ctx, U.from_numpy(A, ctx), U.from_numpy(Bt, ctx))

    out = np.swapaxes(_run_raw(build), -1, -2)        # [nb, am, bn]
    return out.reshape(abatch + (am, bn)) if abatch else out


def _run_raw(build, mem=256 * 1024 * 1024):
    """build 결과를 numpy(역 ne shape 그대로)로 반환 (NCHW 강제 안 함)."""
    ggml, U = _ggml()
    p = ggml.ggml_init_params(mem_size=mem, mem_buffer=None, no_alloc=False)
    ctx = ggml.ggml_init(p)
    try:
        out = build(ctx, ggml, U)
        gf = ggml.ggml_new_graph(ctx)
        ggml.ggml_build_forward_expand(gf, out)
        ggml.ggml_graph_compute_with_ctx(ctx, gf, 4)
        return np.asarray(U.to_numpy(out)).copy()
    finally:
        ggml.ggml_free(ctx)


def _op_dwconv2d(mod, x):
    # depthwise conv: ggml_conv_2d_dw 가 이 빌드에서 abort → host numpy(KH·KW 루프 벡터화).
    x = _as4d(x)
    w = _f32(mod.weights["weight"])      # [C,1,KH,KW]
    C = x.shape[1]
    KH, KW = int(w.shape[2]), int(w.shape[3])
    s = _scalar(mod.attrs.get("stride", [1, 1]))
    p = _scalar(mod.attrs.get("padding", [0, 0]))
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    H, W = xp.shape[2], xp.shape[3]
    OH = (H - KH) // s + 1
    OW = (W - KW) // s + 1
    out = np.zeros((x.shape[0], C, OH, OW), np.float32)
    for i in range(KH):
        for j in range(KW):
            out += w[:, 0, i, j].reshape(1, C, 1, 1) * \
                xp[:, :, i:i + s * OH:s, j:j + s * OW:s]
    b = mod.weights.get("bias")
    if b is not None:
        out = out + _f32(b).reshape(1, -1, 1, 1)
    return out


def _op_batch_norm(mod, x):
    x = _as4d(x)
    w = mod.weights
    eps = float(mod.attrs.get("eps", 1e-5))
    g = _f32(w["weight"]).reshape(1, -1, 1, 1)
    b = _f32(w["bias"]).reshape(1, -1, 1, 1)
    mean = _f32(w["running_mean"]).reshape(1, -1, 1, 1)
    var = _f32(w["running_var"]).reshape(1, -1, 1, 1)
    return (x - mean) / np.sqrt(var + eps) * g + b


def _ggml_unary(x, fn_name):
    # elementwise: 레이아웃 무관 → flatten 후 적용하고 원래 shape 로 복원(차원 추가 금지)
    x = _f32(x)
    shape = x.shape
    flat = np.ascontiguousarray(x.reshape(-1))

    def build(ctx, ggml, U):
        return getattr(ggml, fn_name)(ctx, U.from_numpy(flat, ctx))

    return _run_nchw_1d(build).reshape(shape)


def _op_pool(mod, x, kind):
    x = _as4d(x)
    _, C, H, W = x.shape
    if mod.type == "ADAPTIVE_AVG_POOL2D":
        k0, k1, s0, s1, p0, p1 = W, H, W, H, 0, 0
    else:
        k = _scalar(mod.attrs.get("kernel_size", [2, 2]))
        s = _scalar(mod.attrs.get("stride", [k, k]))
        p = _scalar(mod.attrs.get("padding", [0, 0]))
        k0 = k1 = k
        s0 = s1 = s
        p0 = p1 = p

    def build(ctx, ggml, U):
        xt = U.from_numpy(np.ascontiguousarray(x), ctx)
        return ggml.ggml_pool_2d(ctx, xt, kind, k0, k1, s0, s1, float(p0), float(p1))

    return _run_nchw(build)


def _op_dense(mod, x):
    xin = _f32(np.asarray(x).reshape(-1))   # [in]
    w = _f32(mod.weights["weight"])         # [out,in]

    def build(ctx, ggml, U):
        wt = U.from_numpy(np.ascontiguousarray(w), ctx)     # ne [in,out]
        xt = U.from_numpy(np.ascontiguousarray(xin), ctx)   # ne [in]
        return ggml.ggml_mul_mat(ctx, wt, xt)

    out = np.asarray(_run_nchw_1d(build)).reshape(-1)
    b = mod.weights.get("bias")
    if b is not None:
        out = out + _f32(b).reshape(-1)
    return out.reshape(1, -1)


def _run_nchw_1d(build):
    ggml, U = _ggml()
    p = ggml.ggml_init_params(mem_size=256 * 1024 * 1024, mem_buffer=None, no_alloc=False)
    ctx = ggml.ggml_init(p)
    try:
        out = build(ctx, ggml, U)
        gf = ggml.ggml_new_graph(ctx)
        ggml.ggml_build_forward_expand(gf, out)
        ggml.ggml_graph_compute_with_ctx(ctx, gf, 4)
        return np.asarray(U.to_numpy(out)).reshape(-1).copy()
    finally:
        ggml.ggml_free(ctx)


# --- host numpy ops (torch dim 의미 그대로, NCHW) ---
_INT64_MAX = 9223372036854775807


def _op_binary(a, b, fn, alpha=1, ggfn=None):
    """elementwise binary. same-shape 면 ggml 커널, broadcast/scalar 면 numpy."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if alpha != 1:
        b = b * float(alpha)
    if ggfn is not None and a.shape == b.shape and a.size > 1:
        fa = np.ascontiguousarray(a.reshape(-1))
        fb = np.ascontiguousarray(b.reshape(-1))

        def build(ctx, ggml, U):
            return getattr(ggml, ggfn)(ctx, U.from_numpy(fa, ctx), U.from_numpy(fb, ctx))

        return _run_nchw_1d(build).reshape(a.shape)
    return fn(a, b)


def _op_concat(tensors, dim):
    arrs = [np.asarray(t, dtype=np.float32) for t in tensors]
    return np.concatenate(arrs, axis=dim)


def _op_shape(x, dim):
    return int(np.asarray(x).shape[dim])


def _op_arange(end):
    return np.arange(int(end), dtype=np.float32)


def _op_meshgrid(tensors, indexing):
    grids = np.meshgrid(*[np.asarray(t, dtype=np.float32) for t in tensors],
                        indexing=indexing)
    return tuple(grids)


def _op_stack(tensors, dim):
    return np.stack([np.asarray(t, dtype=np.float32) for t in tensors], axis=dim)


def _iscalar(s):
    """python int / 0-d / 1-elem array 모두 int 로."""
    a = np.asarray(s)
    return int(a.reshape(-1)[0]) if a.ndim else int(a)


def _op_full(size, fill):
    return np.full([_iscalar(s) for s in size], float(np.asarray(fill).reshape(-1)[0])
                   if np.asarray(fill).size else float(fill), dtype=np.float32)


def _op_reshape(x, shape):
    return np.asarray(x, dtype=np.float32).reshape([_iscalar(s) for s in shape])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _rnn_layout(x, h0, c0, batch_first):
    """입력/초기상태를 [T, N, in] / [L*D, N, H] 로 정규화. (T,N,in) 반환."""
    x = _f32(x)
    if batch_first:
        x = np.transpose(x, (1, 0, 2))  # [N,T,in] → [T,N,in]
    return x


def _op_lstm(d):
    """numpy LSTM (PyTorch aten::lstm 정확 매칭) — eager 검증 ground truth.
    params 순서(PyTorch): layer/direction 마다 [w_ih, w_hh, (b_ih, b_hh)].
    gate chunk 순서: i, f, g, o. 반환: (output, h_n, c_n)."""
    x = _f32(_unwrap(d["input"]))
    h0 = _f32(_unwrap(d["hx"][0]))
    c0 = _f32(_unwrap(d["hx"][1]))
    params = [_f32(_unwrap(p)) for p in d["params"]]
    num_layers = int(d["num_layers"])
    has_bias = bool(d["has_biases"])
    bidir = bool(d["bidirectional"])
    batch_first = bool(d["batch_first"])
    ndir = 2 if bidir else 1
    per = 4 if has_bias else 2

    x = _rnn_layout(x, h0, c0, batch_first)  # [T, N, in]
    T, N, _ = x.shape
    layer_in = x
    h_n, c_n = [], []
    pi = 0
    for layer in range(num_layers):
        dir_outs = []
        for dirn in range(ndir):
            w_ih = params[pi]; w_hh = params[pi + 1]
            b_ih = params[pi + 2] if has_bias else 0.0
            b_hh = params[pi + 3] if has_bias else 0.0
            pi += per
            idx = layer * ndir + dirn
            h = h0[idx]; c = c0[idx]
            H = w_hh.shape[1]
            seq = range(T) if dirn == 0 else range(T - 1, -1, -1)
            outs = [None] * T
            for t in seq:
                g = layer_in[t] @ w_ih.T + b_ih + h @ w_hh.T + b_hh  # [N, 4H]
                i = _sigmoid(g[:, 0:H]); f = _sigmoid(g[:, H:2 * H])
                gg = np.tanh(g[:, 2 * H:3 * H]); o = _sigmoid(g[:, 3 * H:4 * H])
                c = f * c + i * gg
                h = o * np.tanh(c)
                outs[t] = h
            h_n.append(h); c_n.append(c)
            dir_outs.append(np.stack(outs, axis=0))  # [T, N, H]
        layer_in = np.concatenate(dir_outs, axis=-1) if bidir else dir_outs[0]
    output = layer_in  # [T, N, H*ndir]
    if batch_first:
        output = np.transpose(output, (1, 0, 2))  # [N, T, H*ndir]
    return output, np.stack(h_n, axis=0), np.stack(c_n, axis=0)


def _op_gru(d):
    """numpy GRU (PyTorch aten::gru 정확 매칭). gate chunk 순서: r, z, n.
    n = tanh(W_in x + b_in + r * (W_hn h + b_hn)). 반환: (output, h_n)."""
    x = _f32(_unwrap(d["input"]))
    h0 = _f32(_unwrap(d["hx"]) if not isinstance(d["hx"], (list, tuple)) else _unwrap(d["hx"][0]))
    params = [_f32(_unwrap(p)) for p in d["params"]]
    num_layers = int(d["num_layers"])
    has_bias = bool(d["has_biases"])
    bidir = bool(d["bidirectional"])
    batch_first = bool(d["batch_first"])
    ndir = 2 if bidir else 1
    per = 4 if has_bias else 2

    x = _rnn_layout(x, h0, None, batch_first)
    T, N, _ = x.shape
    layer_in = x
    h_n = []
    pi = 0
    for layer in range(num_layers):
        dir_outs = []
        for dirn in range(ndir):
            w_ih = params[pi]; w_hh = params[pi + 1]
            b_ih = params[pi + 2] if has_bias else 0.0
            b_hh = params[pi + 3] if has_bias else 0.0
            pi += per
            idx = layer * ndir + dirn
            h = h0[idx]
            H = w_hh.shape[1]
            seq = range(T) if dirn == 0 else range(T - 1, -1, -1)
            outs = [None] * T
            for t in seq:
                gx = layer_in[t] @ w_ih.T + b_ih   # [N, 3H]
                gh = h @ w_hh.T + b_hh
                r = _sigmoid(gx[:, 0:H] + gh[:, 0:H])
                z = _sigmoid(gx[:, H:2 * H] + gh[:, H:2 * H])
                n = np.tanh(gx[:, 2 * H:3 * H] + r * gh[:, 2 * H:3 * H])
                h = (1.0 - z) * n + z * h
                outs[t] = h
            h_n.append(h)
            dir_outs.append(np.stack(outs, axis=0))
        layer_in = np.concatenate(dir_outs, axis=-1) if bidir else dir_outs[0]
    output = layer_in
    if batch_first:
        output = np.transpose(output, (1, 0, 2))
    return output, np.stack(h_n, axis=0)


def _op_transpose(x, d0, d1):
    return np.ascontiguousarray(np.swapaxes(np.asarray(x, dtype=np.float32), d0, d1))


def _op_softmax(x, dim):
    x = np.asarray(x, dtype=np.float32)
    e = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e / np.sum(e, axis=dim, keepdims=True)


def _op_unsqueeze(x, dim):
    return np.expand_dims(np.asarray(x, dtype=np.float32), dim)


def _op_strided_slice(x, dim, start, end, step):
    x = np.asarray(x, dtype=np.float32)
    sl = [slice(None)] * x.ndim
    for j, d in enumerate(dim):
        d = int(d)
        e = _iscalar(end[j])
        if e >= _INT64_MAX or e > x.shape[d]:
            e = x.shape[d]
        sl[d] = slice(_iscalar(start[j]), e, _iscalar(step[j]))
    return x[tuple(sl)]


def _op_resize(x, scale_factor, mode):
    x = _as4d(x)
    sh = int(scale_factor[0]) if isinstance(scale_factor, (list, tuple)) else int(scale_factor)
    sw = int(scale_factor[1]) if isinstance(scale_factor, (list, tuple)) else int(scale_factor)
    # nearest upsample on H,W (NCHW)
    return np.repeat(np.repeat(x, sh, axis=2), sw, axis=3)


def _op_max(x, dim, keepdim):
    x = np.asarray(x, np.float32)
    vals = np.max(x, axis=dim, keepdims=keepdim)
    idx = np.argmax(x, axis=dim)
    if keepdim:
        idx = np.expand_dims(idx, dim)
    return vals, idx.astype(np.float32)


def _op_mean(x, dim, keepdim):
    """torch.mean(x, dim, keepdim). dim=None 이면 전체 평균, int 또는 int 리스트.

    ShuffleNet 등의 global avg pool(x.mean([2,3])) 의 eager 수치 기준 — codegen 의
    adaptive avg pool 렌더와 동일 결과여야 한다(가변 reduce 축 지원).
    """
    x = np.asarray(x, np.float32)
    if dim is None:
        return np.asarray(np.mean(x), np.float32)
    ax = tuple(int(d) for d in dim) if isinstance(dim, (list, tuple)) else int(dim)
    return np.mean(x, axis=ax, keepdims=bool(keepdim)).astype(np.float32)


def _op_topk(x, k, dim, largest):
    x = np.asarray(x, np.float32)
    order = np.argsort(x, axis=dim)
    if largest:
        order = np.flip(order, axis=dim)
    idx = np.take(order, np.arange(int(k)), axis=dim)
    vals = np.take_along_axis(x, idx, axis=dim)
    return vals, idx.astype(np.float32)


def _op_gather(x, dim, index):
    return np.take_along_axis(np.asarray(x, np.float32),
                              np.asarray(index).astype(np.int64), axis=dim)


def _op_index(x, index):
    x = np.asarray(x, np.float32)
    return x[tuple(np.asarray(i).astype(np.int64) for i in index)]


def _op_repeat(x, repeats):
    return np.tile(np.asarray(x, np.float32), [int(r) for r in repeats])


def _op_permute(x, dims):
    return np.ascontiguousarray(np.transpose(np.asarray(x, np.float32),
                                             [int(d) for d in dims]))


def _ggml_pool_max():
    return _ggml()[0].GGML_OP_POOL_MAX


def _ggml_pool_avg():
    return _ggml()[0].GGML_OP_POOL_AVG


# --------------------------------------------------------------------------
# GgmlModule — nn.Module('<OP>', **attrs) 의 ggml 백엔드 구현
# --------------------------------------------------------------------------
def _norm_optype(type):
    t = str(type).upper()
    if t.startswith("ATEN::"):
        t = t[len("ATEN::"):]
    return t.rstrip("_")


def _unwrap(v):
    if isinstance(v, dict):
        return v.get("self", v.get("input", next(iter(v.values()), None)))
    # nn.Parameter(requires_grad=True) 는 numpy() 직접 호출 불가 → detach.
    if hasattr(v, "detach") and hasattr(v, "numpy"):
        return v.detach()
    return v


class GgmlModule:
    def __init__(self, type, *args, **kwargs):
        self.type = _norm_optype(type)
        self.attrs = kwargs
        self.weights = {}

    def __repr__(self):
        return f"GgmlModule('{self.type}')"

    def _in(self, args, kwargs):
        if args:
            return _unwrap(args[0])
        if "input" in kwargs:
            return _unwrap(kwargs["input"])
        for v in kwargs.values():
            return _unwrap(v)
        return None

    def _ab(self, args, kwargs):
        a = kwargs.get("input", args[0] if args else None)
        b = kwargs.get("other", args[1] if len(args) > 1 else None)
        return _unwrap(a), _unwrap(b)

    def __call__(self, *args, **kwargs):
        t = self.type
        a = self.attrs
        if t == "INPUT":
            return self._in(args, kwargs)
        if t == "CONV2D":
            return _op_conv2d(self, self._in(args, kwargs))
        if t == "DEPTHWISE_CONV2D":
            return _op_dwconv2d(self, self._in(args, kwargs))
        if t == "MATMUL":
            x, y = self._ab(args, kwargs)
            return _op_matmul(x, y)
        if t == "BATCH_NORM":
            return _op_batch_norm(self, self._in(args, kwargs))
        if t == "RELU":
            return _ggml_unary(self._in(args, kwargs), "ggml_relu")
        if t == "SILU":
            return _ggml_unary(self._in(args, kwargs), "ggml_silu")
        if t == "SIGMOID":
            return _ggml_unary(self._in(args, kwargs), "ggml_sigmoid")
        if t in ("ELEMWISE_ADD", "ADD"):
            x, y = self._ab(args, kwargs)
            return _op_binary(x, y, np.add, kwargs.get("alpha", 1), "ggml_add")
        if t in ("ELEMENTWISE_SUB", "ELEMWISE_SUB", "SUB"):
            x, y = self._ab(args, kwargs)
            return _op_binary(x, y, np.subtract, kwargs.get("alpha", 1), "ggml_sub")
        if t in ("ELEMWISE_MUL", "MULTIPLY", "MUL"):
            x, y = self._ab(args, kwargs)
            return _op_binary(x, y, np.multiply, 1, "ggml_mul")
        if t in ("ELEMWISE_DIV", "DIV"):
            x, y = self._ab(args, kwargs)
            return _op_binary(x, y, np.divide, 1, "ggml_div")
        if t == "CONCAT":
            tensors = kwargs.get("tensors", args[0] if args else None)
            return _op_concat([_unwrap(z) for z in tensors], int(kwargs.get("dim", 1)))
        if t == "MAXPOOL":
            return _op_pool(self, self._in(args, kwargs), _ggml_pool_max())
        if t in ("AVG_POOL", "AVGPOOL", "ADAPTIVE_AVG_POOL2D"):
            return _op_pool(self, self._in(args, kwargs), _ggml_pool_avg())
        if t == "FLATTEN":
            x = _f32(self._in(args, kwargs))
            nd = x.ndim
            sd = int(kwargs.get("start_dim", 1))
            ed = int(kwargs.get("end_dim", -1))
            if sd < 0:
                sd += nd
            if ed < 0:
                ed += nd
            return x.reshape(x.shape[:sd] + (-1,) + x.shape[ed + 1:])
        if t == "DENSE":
            return _op_dense(self, self._in(args, kwargs))
        if t == "RESHAPE":
            return _op_reshape(self._in(args, kwargs),
                               kwargs.get("shape", kwargs.get("size")))
        if t == "TRANSPOSE":
            return _op_transpose(self._in(args, kwargs),
                                 int(kwargs["dim0"]), int(kwargs["dim1"]))
        if t == "SOFTMAX":
            return _op_softmax(self._in(args, kwargs), int(a.get("dim", -1)))
        if t == "UNSQUEEZE":
            return _op_unsqueeze(self._in(args, kwargs), int(kwargs["dim"]))
        if t == "SHAPE":
            return _op_shape(self._in(args, kwargs), int(kwargs["dim"]))
        if t == "ARANGE":
            return _op_arange(kwargs["end"])
        if t == "MESHGRID":
            d = args[0] if args else kwargs
            return _op_meshgrid([_unwrap(z) for z in d["tensors"]],
                                d.get("indexing", "ij"))
        if t == "STACK":
            return _op_stack([_unwrap(z) for z in kwargs["tensors"]],
                             int(kwargs.get("dim", 0)))
        if t == "FULL":
            d = args[0] if args else kwargs
            return _op_full(d["size"], d["fill_value"])
        if t == "ZEROS":
            return _op_full(kwargs.get("size", a.get("size")), 0.0)
        if t == "LSTM":
            return _op_lstm(args[0] if args else kwargs)
        if t == "GRU":
            return _op_gru(args[0] if args else kwargs)
        if t == "STRIDED_SLICE":
            return _op_strided_slice(self._in(args, kwargs), kwargs["dim"],
                                     kwargs["start"], kwargs["end"], kwargs["step"])
        if t == "RESIZE":
            return _op_resize(self._in(args, kwargs),
                              kwargs.get("scale_factor", [2, 2]),
                              kwargs.get("mode", "nearest"))
        if t == "CONST":
            return np.asarray(kwargs.get("data", a.get("data")), dtype=np.float32)
        if t in ("DETACH", "CONTIGUOUS"):
            return self._in(args, kwargs)
        if t == "CAST":
            return _f32(self._in(args, kwargs))
        if t == "PERMUTE":
            return _op_permute(self._in(args, kwargs), kwargs["dims"])
        if t == "FLOOR_DIVIDE":
            x, y = self._ab(args, kwargs)
            return np.floor_divide(_f32(x), _f32(y))
        if t == "REMAINDER":
            d = args[0] if args else kwargs
            return np.remainder(_f32(_unwrap(d["self"])), _f32(_unwrap(d["other"])))
        if t == "REPEAT":
            return _op_repeat(self._in(args, kwargs), kwargs["repeats"])
        if t == "MAX":
            return _op_max(self._in(args, kwargs), int(kwargs.get("dim", -1)),
                           bool(kwargs.get("keepdim", False)))
        if t == "MEAN":
            return _op_mean(self._in(args, kwargs),
                            kwargs.get("dim", a.get("dim")),
                            bool(kwargs.get("keepdim", a.get("keepdim", False))))
        if t == "TOPK":
            d = args[0] if args else kwargs
            return _op_topk(_unwrap(d["self"]), d["k"], int(d.get("dim", -1)),
                            bool(d.get("largest", True)))
        if t == "GATHER":
            d = args[0] if args else kwargs
            return _op_gather(_unwrap(d["self"]), int(d["dim"]), _unwrap(d["index"]))
        if t == "INDEX":
            return _op_index(self._in(args, kwargs),
                             [_unwrap(z) for z in kwargs["index"]])
        raise NotImplementedError(f"ggml backend: op '{t}' not implemented")


# --------------------------------------------------------------------------
# GGUF weight binding
# --------------------------------------------------------------------------
_INIT_RE = re.compile(
    r"self\.(module_\d+)\s*=\s*nn\.Module\('([^']+)'.*?\)\s*#[^:]+::(.+?)\(\)\s*$"
)
_BRACKET_RE = re.compile(r"\w+\[([\w.]+)\]")


def _node_path_to_key(node_path):
    return ".".join(_BRACKET_RE.findall(node_path))


def _load_gguf_tensors(gguf_path):
    import sys

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gguf_py = os.path.join(here, "vision.cpp", "depend", "llama", "gguf-py")
    if gguf_py not in sys.path:
        sys.path.insert(0, gguf_py)
    import gguf

    reader = gguf.GGUFReader(gguf_path)
    out = {}
    for t in reader.tensors:
        out[t.name] = np.array(t.data, dtype=np.float32).reshape(tuple(reversed(t.shape)))
    return out


def bind_gguf(model, gguf_path, export_path):
    tensors = _load_gguf_tensors(gguf_path)
    n_bound = 0
    with open(export_path) as f:
        for line in f:
            m = _INIT_RE.search(line.strip())
            if not m:
                continue
            mod_attr, _optype, node_path = m.group(1), m.group(2), m.group(3)
            prefix = _node_path_to_key(node_path)
            if not prefix:
                continue
            mod = getattr(model, mod_attr, None)
            if mod is None or not isinstance(mod, GgmlModule):
                continue
            w = {}
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                key = f"{prefix}.{suffix}"
                if key in tensors:
                    w[suffix] = tensors[key]
            if w:
                mod.weights = w
                n_bound += 1
    # 모델 직속 Parameter(LSTM/GRU 의 weight_ih_l0/weight_hh_l0/bias_*_l0 ...)는 GgmlModule 이
    # 아니라 모델 속성으로 export 된다(export 시 nn.Parameter 로 선언, forward 에 params 리스트로
    # 전달). GGUF 텐서명이 PyTorch param 이름과 동일하므로 이름 매칭으로 채운다.
    import torch
    for name, p in model.named_parameters():
        if name in tensors:
            arr = np.ascontiguousarray(tensors[name])
            if tuple(p.shape) == arr.shape:
                p.data = torch.from_numpy(arr.copy())
                n_bound += 1
    return n_bound


def run_gguf(model, gguf_path, export_path, x):
    """모델을 ggml 백엔드로 실행: 가중치 바인딩 후 forward(x)."""
    if hasattr(model, "from_script"):
        model.from_script(True)
    bind_gguf(model, gguf_path, export_path)
    # RNN(LSTM/GRU) 입력은 3D [N,T,feat] 로 의도된 것 — feature-map 4D 정규화(_as4d) 우회.
    is_rnn = False
    try:
        with open(export_path) as f:
            src = f.read()
        is_rnn = ("nn.Module('LSTM'" in src) or ("nn.Module('GRU'" in src)
    except OSError:
        pass
    return model(_f32(x) if is_rnn else _as4d(x))
