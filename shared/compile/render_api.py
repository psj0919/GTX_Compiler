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
"""vision.cpp(ggml) codegen 의 op 렌더링 프레임워크.

각 op 의 C++ 렌더링은 `nn/modules/<op>.py` 의 module-level ``render(node, ctx)``
함수에 두고 ``@register_render(OP.X)`` 로 등록한다. `VispCodeGenerator` 는
`RENDERERS` 레지스트리로 디스패치한다(구조적 op INPUT/FLATTEN/RETURN 은 generator 내장).

render(node, ctx) 규약:
- `ctx.inp(node, i)`   : i 번째 입력의 C++ 변수명
- `ctx.attr(node, k, d)`: op attr 값 (AttrName/IrAttr/MockGraph 모두 처리)
- `ctx.scalar(v)`      : [s, s] -> s
- `ctx.weight(node, [suffix...])` : weight key 기록 + `m["key"]` 문자열 반환
- `ctx.out(node, expr) : `tensor <var> = <expr>;` emit + 출력 바인딩 + var 반환
설계: docs/ggml_codegen.md
"""

import re

from shared.base import OP

# op 값(소문자) -> render(node, ctx)
RENDERERS = {}

# 출력 변수명 prefix (가독성)
_HINT = {
    "conv2d": "conv",
    "batch_norm": "bn",
    "relu": "relu",
    "elemwise_add": "add",
    "maxpool": "pool",
    "avgpool": "pool",
    "adaptive_avg_pool2d": "pool",
    "flatten": "flat",
    "dense": "fc",
    "matmul": "mm",
}

# `Sequential[layer1]/BasicBlock[0]/Conv2d[conv1]` -> ['layer1','0','conv1']
_BRACKET = re.compile(r"\[([^\]]+)\]")
_NON_IDENT = re.compile(r"[^0-9A-Za-z_]")

# 요청 이름 -> 실제 attr 키 후보 (실제 그래프는 AttrName enum 키, MockGraph 는 string)
_ATTR_ALIASES = {
    "stride": ("stride",),
    "padding": ("pad", "padding"),
    "kernel_size": ("kernel", "kernel_size"),
    "bias": ("bias_term", "bias"),
    "dilation": ("dilation",),
    "groups": ("group", "groups"),
    "inplace": ("inplace",),
    "output_size": ("output_size", "out_size"),
    # detection-head / meta op attrs (TorchParser 그래프 키)
    "dim": ("dim", "axis"),          # concat/softmax/max 의 축
    "order": ("order", "perm", "dims"),  # transpose/permute 의 축 순서
    "scale": ("scale", "scale_factor"),  # resize 배율
    "keep_dims": ("keep_dims", "keepdim"),
    "negative_slope": ("negative_slope", "alpha", "slope"),  # leaky_relu
    "min": ("min", "min_val", "clamp_min"),  # clamp
    "max": ("max", "max_val", "clamp_max"),
    "num_groups": ("num_groups", "groups", "group"),  # group_norm
    "begin": ("begin", "start"),     # strided_slice 시작 인덱스(per-dim)
    "slice_dims": ("dims",),         # strided_slice 대상 축 리스트
}


def register_render(*op_types):
    """op 의 render(node, ctx) 함수를 RENDERERS 에 등록하는 데코레이터."""

    def deco(fn):
        for ot in op_types:
            RENDERERS[ot] = fn
            if isinstance(ot, str):
                RENDERERS[ot.lower()] = fn
        return fn

    return deco


def op_value(node):
    return str(node.op.type).lower()


def weight_key(node):
    """노드 이름에서 PyTorch state_dict 경로(=GGUF 텐서 prefix)를 산출.

    실제 그래프의 노드명(`ResNet/Sequential[layer1]/.../Conv2d[conv1]/ret.5`)에서
    대괄호 토큰을 모아 `layer1.0.conv1` 를 만든다 (ggml_weight_binder 와 동일 규칙).
    """
    name = getattr(node, "name", "") or ""
    parts = _BRACKET.findall(name)
    if parts:
        return ".".join(parts)
    return _NON_IDENT.sub("_", name) or "unnamed"


def attr(node, key, default=None):
    """op attr 값을 읽는다. AttrName enum 키 + IrAttr 래퍼(실제 그래프),
    string 키 + raw 값(MockGraph) 모두 처리."""
    attrs = getattr(node.op, "attrs", None) or {}
    aliases = _ATTR_ALIASES.get(key, (key,))
    try:
        items = list(attrs.items())
    except AttributeError:
        items = []
    for k, v in items:
        kname = getattr(k, "value", k)  # AttrName.STRIDE -> 'stride'
        if kname in aliases or k in aliases:
            val = getattr(v, "value", v)  # IrAttr -> 실제 값
            if val is not None:
                return val
    # AttrName 을 선언하지 않은 plain Operation(clamp/arange/full/roll…)은 파라미터가
    # attrs 가 아니라 op.configs(=set_config 로 심은 인스턴스 속성)로만 실린다.
    for name in aliases:
        val = getattr(node.op, name, None)
        if val is not None:
            return val
    return default


def scalarize(v, idx=0, default=1):
    """[s, s] 같은 리스트면 idx 요소, 스칼라면 그대로."""
    if isinstance(v, (list, tuple)):
        return v[idx] if len(v) > idx else default
    return v if v is not None else default


def out_shape(node, i=0):
    """노드 i 번째 출력의 정적 torch shape (list[int]) 또는 None.

    detection-head 의 reshape/unsqueeze/arange 등은 출력 shape 가 정적으로 확정되어
    있어 (입력 해상도 고정), 동적 입력 텐서를 추적하지 않고도 ggml shape 를 산출할 수 있다.
    """
    outs = getattr(node, "out_tensors", None) or []
    if i >= len(outs) or outs[i] is None:
        return None
    try:
        return [int(d) for d in outs[i].shape]
    except Exception:
        return None


def ggml_axis(torch_dim, ndim):
    """torch 축 인덱스를 ggml ne 축으로 변환 (ggml ne 는 torch shape 의 역순)."""
    d = int(torch_dim)
    if d < 0:
        d += ndim
    return ndim - 1 - d


class RenderContext:
    """render(node, ctx) 가 사용하는 상태/헬퍼. VispCodeGenerator 가 walk 하며 채운다."""

    def __init__(self):
        self._var = {}        # id(tensor) -> C++ 변수명
        self._counter = 0
        self._weights = []    # (weight_key, [suffix...])
        self.lines = []       # forward 본문 라인

    # --- 변수/바인딩 ---
    def new_var(self, hint="t"):
        self._counter += 1
        return f"{hint}{self._counter}"

    def inp(self, node, i=0):
        ins = [t for t in node.in_tensors if t is not None]
        if i < len(ins):
            return self._var.get(id(ins[i]), "x")
        return "x"

    def bind(self, node, var):
        if node.out_tensors:
            self._var[id(node.out_tensors[0])] = var

    def bind_outputs(self, node, var):
        """모든 출력 텐서를 같은 변수에 바인딩 (multi-output op: meshgrid/max/topk).

        scaffold 단계에서 다중 출력을 단일 변수로 묶는다 — 2번째 출력(예: topk 인덱스)을
        구분하지 않으므로 정확한 head 디코드에는 추가 작업이 필요하다(TODO).
        """
        for t in node.out_tensors or []:
            if t is not None:
                self._var[id(t)] = var

    def out(self, node, expr, hint=None):
        hint = hint or _HINT.get(op_value(node), "t")
        var = self.new_var(hint)
        self.lines.append(f"    tensor {var} = {expr};")
        self.bind(node, var)
        return var

    def line(self, text):
        self.lines.append(text)

    # --- attr / weight ---
    def attr(self, node, key, default=None):
        return attr(node, key, default)

    def scalar(self, v, idx=0, default=1):
        return scalarize(v, idx, default)

    def weight(self, node, suffixes):
        key = weight_key(node)
        self._weights.append((key, list(suffixes)))
        return f'm["{key}"]'

    def raw_weight(self, node, suffix):
        """state_dict 키를 **텐서로** 직접 참조한다(`m["layer"]` 레이어 핸들이 아니라).

        custom autograd op(mmcv deform 등)은 가중치가 `in_tensors` 에 안 실려 오는 경우가
        있어 `inp()` 가 그래프 입력 `x`(= 이미지)로 폴백한다 — 그러면 이미지가 커널에
        가중치로 들어가 `kernel->ne[2] == C` assert 로 죽는다.
        그런 op 은 노드명에서 얻은 모듈 경로로 gguf 텐서를 직접 찾는다.
        """
        key = f"{weight_key(node)}.{suffix}"
        if (key, []) not in self._weights:
            self._weights.append((key, []))
        return f'm.find("{key}")'
