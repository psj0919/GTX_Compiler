"""dispatch_tracer.py — torch.jit 없이 aten 그래프를 캡처해 TorchGraph IR 로 빌드 (실험/병행).

`TorchDispatchMode`(`__torch_dispatch__`)로 eager forward 1회를 돌리며 실행되는 aten op 을
가로채 op 이름 + **순서 보존 인자**(텐서 dataflow / 상수) + 출력 shape/dtype + 모듈경로 +
파라미터 state_dict 이름을 기록한다. 그 캡처를 기존 `TorchGraph/TorchNode/TorchValue`
IR 로 변환해 `TorchParser._convert_graph` 가 그대로 소비하게 한다.

jit.get_trace_graph 대체 후보. jit 경로(trace_helper/jit_utils)는 건드리지 않는다.

설계:
- dispatch 는 jit 보다 한 단계 더 분해됨(linear→t+addmm, adaptive_avg_pool→mean,
  flatten→view). op_dispatcher 가 addmm/t_addmm/mean/view/t/max_pool2d_with_indices
  핸들러를 이미 보유 → 어휘 호환. 수치 동등성으로 검증(IR 노드명 동일성 아님).
- 인자 순서를 보존해 핸들러 positional 시그니처와 정합.
- 파라미터/버퍼는 id→state_dict 이름으로 정확히 명명(GGUF 매칭 공짜).
"""
from __future__ import annotations

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .torch_graph import TorchGraph, TorchBlock, TorchNode, TorchValue

# 그래프 의미 없는 보조 op(버퍼 할당/복사/노옵) — 스킵.
_SKIP = {
    "aten.empty", "aten.empty_like", "aten.empty_strided",
    "aten.detach", "aten.detach_", "aten.lift_fresh", "aten.alias",
    "aten._to_copy", "aten.clone", "aten._unsafe_view",
    "aten._local_scalar_dense",   # tensor.item() — 스칼라 추출, 그래프 op 아님
}

# dispatch overloadpacket → op_dispatcher 핸들러 kind 정규화.
_NAME_MAP = {
    "aten.convolution": "_convolution",
    "aten.native_batch_norm": "batch_norm",
    "aten._native_batch_norm_legit": "batch_norm",
    "aten._native_batch_norm_legit_no_training": "batch_norm",
    "aten.add_": "add",
    "aten.add": "add",
    "aten.mul_": "mul",
    "aten.mul": "mul",
    "aten.relu_": "relu_",
    "aten.relu": "relu",
    "aten.max_pool2d_with_indices": "max_pool2d_with_indices",
    "aten.mean": "mean",
    "aten.view": "view",
    "aten.reshape": "reshape",
    "aten.t": "t",
    "aten.addmm": "addmm",
    "aten.flatten": "flatten",
    "aten.linear": "linear",
    "aten.adaptive_avg_pool2d": "adaptive_avg_pool2d",
    # --- YOLO neck/head ---
    "aten.silu_": "silu_",
    "aten.silu": "silu",
    "aten.cat": "cat",
    "aten.split": "split",
    "aten.split_with_sizes": "split",
    "aten.stack": "stack",
    "aten.arange": "arange",
    "aten.full": "full",
    "aten.expand": "expand",
    "aten.select": "select",
    "aten.unsqueeze": "unsqueeze",
    "aten.transpose": "transpose",
    "aten.upsample_nearest2d": "upsample_nearest2d",
    "aten.sub": "sub",
    "aten.div": "div",
    "aten._softmax": "softmax",
    "aten.sigmoid": "sigmoid",
}

# 다중 출력 중 [0] 만 데이터플로우로 쓰는 op (나머지는 indices/stats).
_TAKE_FIRST_OUT = {"batch_norm", "max_pool2d_with_indices"}

# 핸들러가 jit aten 시그니처용(추가 인자 보유)이라, dispatch aten 에 없는 trailing 인자 패딩.
_ARG_PAD = {
    "_convolution": (False, False, True),   # benchmark, deterministic, cudnn_enabled
    "batch_norm": (True,),                  # cudnn_enabled
}


def _kind_of(func):
    return str(func).rsplit(".", 1)[0]   # "aten.convolution.default" → "aten.convolution"


def _is_t(x):
    return isinstance(x, torch.Tensor)


# ---------------------------------------------------------------------------
# 캡처
# ---------------------------------------------------------------------------
class _Op:
    __slots__ = ("idx", "raw_name", "kind", "args", "out_ids", "out_shapes",
                 "out_dtype", "module", "schema")

    def __init__(self, idx, raw_name, kind):
        self.idx = idx
        self.raw_name = raw_name
        self.kind = kind
        self.args = []          # 순서 보존: ("t", ref) | ("c", value) | ("tlist", [ref...])
        self.out_ids = []
        self.out_shapes = []
        self.out_dtype = None
        self.module = None
        self.schema = None      # torch._C.FunctionSchema (핸들러의 schema 검사용)


class DispatchTracer(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops: list[_Op] = []
        self._producer = {}        # id(tensor) -> (op_idx, out_pos)
        self._param_name = {}      # id(tensor) -> state_dict 이름
        self._input_idx = {}       # id(tensor) -> 입력 순번
        self._mod_stack = []
        self._hooks = []
        self._const_id = {}        # id(tensor) -> const_idx (producer/param/input 아닌 leaf)
        self.const_vals = {}       # const_idx -> np.ndarray (캐시된 anchor·상수 텐서 값)

    def attach(self, model):
        for n, p in model.named_parameters():
            self._param_name[id(p)] = n
        for n, b in model.named_buffers():
            self._param_name[id(b)] = n
        named = {m: n for n, m in model.named_modules()}

        def pre(mod, inp):
            self._mod_stack.append(named.get(mod) or mod.__class__.__name__)

        def post(mod, inp, out):
            if self._mod_stack:
                self._mod_stack.pop()

        for m in model.modules():
            self._hooks.append(m.register_forward_pre_hook(pre))
            self._hooks.append(m.register_forward_hook(post))

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def mark_inputs(self, *tensors):
        for i, t in enumerate(tensors):
            if _is_t(t):
                self._input_idx[id(t)] = i

    @staticmethod
    def _ordered_args(func, args, kwargs):
        """스키마 순서로 인자 나열 — 생략된 기본값도 채운다."""
        sch = getattr(func, "_schema", None)
        if sch is None:
            return list(args) + list(kwargs.values())
        out = []
        sargs = [a for a in sch.arguments if a.name != "out"]
        for i, sa in enumerate(sargs):
            if i < len(args):
                out.append(args[i])
            elif sa.name in kwargs:
                out.append(kwargs[sa.name])
            elif sa.has_default_value():
                out.append(sa.default_value)
        return out

    def _ref(self, t):
        tid = id(t)
        if tid in self._producer:
            return ("op", *self._producer[tid])
        if tid in self._param_name:
            return ("param", self._param_name[tid])
        if tid in self._input_idx:
            return ("input", self._input_idx[tid])
        # producer/param/input 아닌 leaf = 상수(캐시된 anchor 등) → CONST 로 값 캡처.
        if tid not in self._const_id:
            cid = len(self.const_vals)
            self._const_id[tid] = cid
            self.const_vals[cid] = t.detach().cpu().numpy()
        return ("const", self._const_id[tid])

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        raw = _kind_of(func)
        if raw in _SKIP:
            return out

        op = _Op(len(self.ops), raw, _NAME_MAP.get(raw, raw.split(".")[-1]))
        op.module = self._mod_stack[-1] if self._mod_stack else None
        op.schema = getattr(func, "_schema", None)

        # 스키마 순서로 인자 수집 — 생략된 기본값까지 채운다(jit 그래프와 정합).
        ordered = self._ordered_args(func, args, kwargs)
        for a in ordered:
            if _is_t(a):
                op.args.append(("t", self._ref(a)))
            elif isinstance(a, (list, tuple)) and a and all(_is_t(x) for x in a):
                op.args.append(("tlist", [self._ref(x) for x in a]))
            else:
                op.args.append(("c", str(a) if isinstance(a, torch.dtype) else a))
        # 핸들러가 jit aten 시그니처용이라 추가 인자가 필요한 op 패딩.
        op.args.extend(("c", v) for v in _ARG_PAD.get(op.kind, ()))

        outs = out if isinstance(out, (list, tuple)) else (out,)
        for pos, o in enumerate(outs):
            if _is_t(o):
                self._producer[id(o)] = (op.idx, pos)   # 다중 출력: pos 별 추적
                op.out_ids.append(id(o))
                op.out_shapes.append(tuple(o.shape))
                if op.out_dtype is None:
                    op.out_dtype = str(o.dtype)
        self.ops.append(op)
        return out


def trace(model, *example_inputs):
    """model 을 eager 1회 실행하며 aten op 을 캡처해 [_Op] 반환."""
    model = model.eval()
    tr = DispatchTracer()
    tr.attach(model)
    try:
        with torch.no_grad():
            with tr:
                tr.mark_inputs(*example_inputs)
                model(*example_inputs)
    finally:
        tr.detach()
    return tr


# ---------------------------------------------------------------------------
# TorchValue 서브클래스 (jit 없이 텐서 값 생성)
# ---------------------------------------------------------------------------
_DTYPE_MAP = {
    "torch.float32": "torch.float", "torch.float": "torch.float",
    "torch.float64": "torch.double", "torch.float16": "torch.half",
    "torch.int64": "torch.long", "torch.int32": "torch.int",
    "torch.bool": "torch.bool",
}


class DispatchValue(TorchValue):
    """dispatch 캡처 메타로부터 구성한 텐서 TorchValue (jit Value 우회)."""

    def __init__(self, name, shape, dtype="torch.float32", *, data=None):
        from .torch_graph import ValueDeviceInfo
        self._name = name
        self._scope_name = ""
        self._node = None
        self._shape = list(shape) if shape is not None else None
        self._is_none = False
        self._is_plain_value = False
        self._type = "Tensor"
        self._data = data
        self._device_info = ValueDeviceInfo()
        self._requires_grad = None
        self._layout = None
        self._dtype = _DTYPE_MAP.get(dtype, dtype)


# ---------------------------------------------------------------------------
# 재합성 — dispatch 분해형을 jit 고수준 op 으로 (exporter/codegen 정합)
# ---------------------------------------------------------------------------
def recompose(ops):
    """dispatch 분해형 → jit 고수준 op 재합성 (mean→adaptive_avg_pool, t+addmm→linear)."""
    by_idx = {op.idx: op for op in ops}

    def _ref_shape(ref):
        if ref and ref[0] == "op":
            src = by_idx.get(ref[1])
            if src and src.out_shapes and ref[2] < len(src.out_shapes):
                return src.out_shapes[ref[2]]
        return None

    remove = set()
    for op in ops:
        if op.kind == "mean":
            dims = next((a[1] for a in op.args
                         if a[0] == "c" and isinstance(a[1], (list, tuple))), None)
            if dims and set(int(d) for d in dims) in ({-1, -2}, {2, 3}):
                inp = next(a for a in op.args if a[0] == "t")
                op.kind = "adaptive_avg_pool2d"
                op.args = [inp, ("c", [1, 1])]
        elif op.kind == "upsample_nearest2d":
            # dispatch 는 output_size 만(scales=None) → ggml RESIZE 가 scale_factor 필요.
            # 입력/출력 H 비율로 scale 채움. args=[x, output_size, scale_h, scale_w].
            x_ref = next((a[1] for a in op.args if a[0] == "t"), None)
            if x_ref and x_ref[0] == "op" and op.out_shapes:
                src = by_idx.get(x_ref[1])
                if src and src.out_shapes:
                    ins, outs = src.out_shapes[x_ref[2]], op.out_shapes[0]
                    if len(ins) >= 2 and ins[-2]:
                        sh = outs[-2] // ins[-2] if outs[-2] % ins[-2] == 0 else outs[-2] / ins[-2]
                        sw = outs[-1] // ins[-1] if outs[-1] % ins[-1] == 0 else outs[-1] / ins[-1]
                        # size=None 이어야 scale_factor 설정(625행). scale 은 float 스칼라여야
                        # 세터가 2*[factor] 로 처리(SCALE attr=float). scale_w=None → scale=scale_h.
                        op.args = [("t", x_ref), ("c", None), ("c", float(sh)), ("c", None)]
        elif op.kind == "expand":
            # dispatch 는 meshgrid 를 view+expand 로 분해(jit 은 MESHGRID 단일노드). ggml 에
            # EXPAND 없음 → REPEAT 로 (size-1 축 broadcast = tile). repeats=out//in (torch
            # expand 우측정렬). REPEAT 은 eager(_op_repeat)·codegen(render_repeat) 양쪽 구현됨.
            x_ref = next((a[1] for a in op.args if a[0] == "t"), None)
            in_sh = _ref_shape(x_ref)
            if x_ref and in_sh and op.out_shapes:
                out_sh = list(op.out_shapes[0])
                ins = [1] * (len(out_sh) - len(in_sh)) + list(in_sh)   # 우측 정렬
                reps = [(o // i if i else 1) for i, o in zip(ins, out_sh)]
                op.kind = "repeat"
                op.args = [("t", x_ref), ("c", reps)]
                op.schema = None   # expand schema 잔류 방지(repeat 핸들러 정합)
        elif op.kind == "addmm":
            # addmm(bias, x, Wᵀ) where Wᵀ=t(W param) → linear(x, W, bias).
            targs = [a for a in op.args if a[0] == "t"]
            if len(targs) >= 3:
                bias_ref, x_ref, w_ref = targs[0][1], targs[1][1], targs[2][1]
                tprod = by_idx.get(w_ref[1]) if w_ref[0] == "op" else None
                if tprod is not None and tprod.kind == "t":
                    w_param = next((a[1] for a in tprod.args if a[0] == "t"), None)
                    if w_param is not None:
                        op.kind = "linear"
                        op.args = [("t", x_ref), ("t", w_param), ("t", bias_ref)]
                        remove.add(tprod.idx)   # 이제 미사용 t 노드 제거
    if remove:
        ops[:] = [op for op in ops if op.idx not in remove]
    return ops


# ---------------------------------------------------------------------------
# 캡처 → TorchGraph IR
# ---------------------------------------------------------------------------
def build_torch_graph(tracer, model, graph_name, *example_inputs):
    """DispatchTracer(또는 [_Op]) → TorchGraph (TorchParser._convert_graph 가 소비)."""
    if hasattr(tracer, "ops"):
        ops, const_vals = tracer.ops, tracer.const_vals
    else:
        ops, const_vals = tracer, {}
    recompose(ops)
    state = dict(model.state_dict())
    g = TorchGraph(graph_name)
    top = TorchBlock(g, None)
    g.set_top_block(top)

    # op_idx → 출력 TorchValue (pos 0)
    out_val = {}
    # 그래프 입력: Param 노드(무입력, 출력=input_i)로 생성해 IR 텐서로 등록되게 한다.
    in_vals = {}
    for i, t in enumerate(example_inputs):
        v = DispatchValue(f"input_{i}", tuple(t.shape), str(t.dtype))
        pnode = TorchNode()
        pnode.kind = "Param"
        pnode.name = f"input_{i}"
        pnode.owning_graph = g
        top.append_node(pnode)
        pnode.add_output(v)
        in_vals[i] = v

    # 파라미터 value (state_dict 이름)
    param_vals = {}

    def param_value(name):
        if name not in param_vals:
            t = state[name]
            v = DispatchValue(name, tuple(t.shape), str(t.dtype),
                              data=t.detach().cpu().numpy())
            param_vals[name] = v
            g.add_param_value(v)
        return param_vals[name]

    # 상수(캐시된 anchor 등) → Constant 노드(데이터 실린 출력). op_dispatcher.Constant 소비.
    const_nodes = {}

    def const_value(cid):
        if cid not in const_nodes:
            arr = const_vals[cid]
            dt = "torch.long" if arr.dtype.kind in "iu" else "torch.float32"
            cnode = TorchNode()
            cnode.kind = "Constant"
            cnode.owning_graph = g
            top.append_node(cnode)
            cv = DispatchValue(f"const_{cid}", tuple(arr.shape), dt, data=arr)
            cnode.add_output(cv)
            const_nodes[cid] = cv
        return const_nodes[cid]

    def ref_value(ref):
        kind = ref[0]
        if kind == "op":
            return out_val[(ref[1], ref[2])]      # (op_idx, out_pos)
        if kind == "param":
            return param_value(ref[1])
        if kind == "const":
            return const_value(ref[1])
        return in_vals.get(ref[1])

    def _emit_split(op):
        # ggml 에 SPLIT 없음 → 각 출력 청크를 strided_slice 로 (pattern match).
        x = ref_value(next(a[1] for a in op.args if a[0] == "t"))
        consts = [a[1] for a in op.args if a[0] == "c"]
        dim = consts[-1] if len(consts) >= 2 and isinstance(consts[-1], int) else 1
        nd = len(op.out_shapes[0]) if op.out_shapes else 4
        d = dim if dim >= 0 else dim + nd
        start = 0
        for pos, sh in enumerate(op.out_shapes):
            sz = sh[d]
            s = TorchNode()
            s.kind = "strided_slice"
            s.owning_graph = g
            s.add_input(x)
            s.add_input(TorchValue([d], name=f"split_{op.idx}_{pos}_dim"))
            s.add_input(TorchValue([start], name=f"split_{op.idx}_{pos}_st"))
            s.add_input(TorchValue([start + sz], name=f"split_{op.idx}_{pos}_en"))
            s.add_input(TorchValue([1], name=f"split_{op.idx}_{pos}_sp"))
            top.append_node(s)
            ov = DispatchValue(f"split_{op.idx}_{pos}", sh, op.out_dtype or "torch.float32")
            s.add_output(ov)
            out_val[(op.idx, pos)] = ov
            start += sz

    def _emit_select(op):
        # ggml 에 SELECT 없음 → strided_slice(dim, index:index+1, step 1) 로 (pattern match).
        # squeeze 생략: ggml SQUEEZE 부재 + 이 패턴(YOLO make_anchors stride 추출)의 select
        # 출력은 fill_value 상수로 폴드돼 미소비 → 차원 보존이 무해.
        x = ref_value(next(a[1] for a in op.args if a[0] == "t"))
        consts = [a[1] for a in op.args if a[0] == "c"]
        dim = int(consts[0]) if len(consts) >= 1 and isinstance(consts[0], int) else 0
        index = int(consts[1]) if len(consts) >= 2 and isinstance(consts[1], int) else 0
        out_nd = len(op.out_shapes[0]) if op.out_shapes else 0   # select 출력은 입력보다 1축 적음
        if dim < 0:
            dim += out_nd + 1
        s = TorchNode()
        s.kind = "strided_slice"
        s.owning_graph = g
        s.add_input(x)
        s.add_input(TorchValue([dim], name=f"select_{op.idx}_dim"))
        s.add_input(TorchValue([index], name=f"select_{op.idx}_st"))
        s.add_input(TorchValue([index + 1], name=f"select_{op.idx}_en"))
        s.add_input(TorchValue([1], name=f"select_{op.idx}_sp"))
        top.append_node(s)
        sh = op.out_shapes[0] if op.out_shapes else None
        ov = DispatchValue(f"select_{op.idx}", sh, op.out_dtype or "torch.float32")
        s.add_output(ov)
        out_val[(op.idx, 0)] = ov

    for op in ops:
        if op.kind == "split":
            _emit_split(op)
            continue
        if op.kind == "select":
            _emit_select(op)
            continue
        node = TorchNode()           # node=None → 빈 노드
        node.kind = op.kind
        node.schema = op.schema      # 핸들러 schema 검사용(SchemaHelper)
        # GGUF 가중치 바인딩용: bind_gguf 의 _BRACKET_RE(\w+\[(...)\]) 가 module 경로를
        # 뽑도록 M[<module>] 형식으로. (jit 의 Conv2d[conv1] 대응)
        node._scope_name = f"M[{op.module}]" if op.module else ""
        node.owning_graph = g

        for a in op.args:
            tag = a[0]
            if tag == "t":
                node.add_input(ref_value(a[1]))
            elif tag == "tlist":
                node.add_input([ref_value(r) for r in a[1]])
            else:                    # const → plain TorchValue
                val = a[1]
                node.add_input(TorchValue(val, name=f"{op.kind}_{op.idx}_c")
                               if isinstance(val, (float, int, bool, tuple, list))
                               else _NoneValue())

        top.append_node(node)        # owning_block 설정
        # 출력 value (다중 출력 대응). batch_norm/maxpool 은 [0]만(나머지는 stats/indices).
        shapes = op.out_shapes or [None]
        if op.kind in _TAKE_FIRST_OUT:
            shapes = shapes[:1]
        for pos, sh in enumerate(shapes):
            nm = f"{op.kind}_{op.idx}" + (f"_{pos}" if pos else "")
            ov = DispatchValue(nm, sh, op.out_dtype or "torch.float32")
            # node.name = scope_name + name 조합 → GGUF 바인딩용 M[<module>] 부여.
            if op.module:
                ov._scope_name = f"M[{op.module}]"
            node.add_output(ov)      # value.node=node, blob 등록
            out_val[(op.idx, pos)] = ov

    # Return 노드(무출력, 입력=그래프 출력 = 마지막 op 의 pos0).
    if ops:
        ret = TorchNode()
        ret.kind = "Return"
        ret.name = "return_0"
        ret.owning_graph = g
        top.append_node(ret)
        ret.add_input(out_val[(ops[-1].idx, 0)])

    g.connect_nodes()
    return g


class _NoneValue(TorchValue):
    """None 인자용 (bias=None 등)."""
    def __init__(self):
        from .torch_graph import ValueDeviceInfo
        self._name = "None"
        self._scope_name = ""
        self._node = None
        self._shape = None
        self._is_none = True
        self._is_plain_value = False
        self._type = "None"
        self._data = None
        self._device_info = ValueDeviceInfo()
        self._requires_grad = None
        self._layout = None
        self._dtype = None
