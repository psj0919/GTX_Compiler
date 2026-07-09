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
"""인덱싱/선택 op 의 ggml/vision.cpp render (head_render.py + vision_ops_render.py 에서 분리).

argmax/argsort/max/topk/gather/index/strided_slice 를 담는다. 인덱스를 생성하거나
인덱스로 원소를 선택하는 op. 전부 **실제 ggml 그래프 op** 으로 emit 한다.
런타임 의미는 ``nn/modules/ggml_backend.py`` host numpy 구현이 기준이다.
"""

from shared.compile.render_api import register_render as _rr, out_shape, ggml_axis
from shared.base import OP as _OP


def _ne_args(shape):
    """torch shape → ggml ne 인자 문자열(역순). 최대 4D."""
    ne = list(reversed([int(d) for d in shape]))
    return ne, ", ".join(str(d) for d in ne)


# ----------------------------------------------------------- 인덱스 생성(정렬/argmax)
@_rr(_OP.ARGMAX)
def render_argmax(node, ctx):
    return ctx.out(node, f"ggml_argmax(m, {ctx.inp(node)})", hint="amax")


@_rr(_OP.ARGSORT)
def render_argsort(node, ctx):
    order = "GGML_SORT_ORDER_DESC" if ctx.attr(node, "descending", False) else "GGML_SORT_ORDER_ASC"
    return ctx.out(node, f"ggml_argsort(m, {ctx.inp(node)}, {order}) /* TODO(ggml): dim=마지막축 가정 */", hint="asort")


@_rr(_OP.MAX)
def render_max(node, ctx):
    # torch max(dim) → (values, indices). ggml_argmax = 인덱스(마지막 축 reduce).
    var = ctx.out(node, f"ggml_argmax(m, {ctx.inp(node)})", hint="max")
    ctx.bind_outputs(node, var)
    return var


@_rr("aten::topk", "topk")
def render_topk(node, ctx):
    sh = out_shape(node)
    k = sh[-1] if sh else 0
    var = ctx.out(node, f"ggml_top_k(m, {ctx.inp(node)}, {k})", hint="topk")
    ctx.bind_outputs(node, var)
    return var


# ----------------------------------------------------------- 선택(gather/slice)
@_rr("aten::gather", "gather", _OP.INDEX)
def render_gather(node, ctx):
    # ggml_get_rows(a, idx): idx(I32) 행 선택. torch gather/index 의 근사.
    a = ctx.inp(node, 0)
    idx = ctx.inp(node, 1) if len([t for t in node.in_tensors if t is not None]) > 1 else a
    return ctx.out(node, f"ggml_get_rows(m, {a}, {idx})", hint="gat")


@_rr(_OP.STRIDED_SLICE)
def render_strided_slice(node, ctx):
    # ggml_view_Nd: 정적 출력 ne + 입력 nb(슬라이스는 축 stride 보존) + byte offset.
    # offset = Σ_d begin[d] * a->nb[ggml_axis(d)]  (torch dim d → ggml 축 ndim-1-d).
    sh = out_shape(node)
    a = ctx.inp(node)
    if not sh or len(sh) > 4:
        return ctx.out(node, f"ggml_cont(m, {a}) /* slice {sh} */", hint="sl")
    ndim = len(sh)
    ne, ne_args = _ne_args(sh)
    nb = ", ".join(f"{a}->nb[{i}]" for i in range(1, len(ne)))
    nb = (nb + ", ") if nb else ""
    begin = ctx.attr(node, "begin", None)
    dims = ctx.attr(node, "slice_dims", None)
    terms = []
    if isinstance(begin, (list, tuple)):
        dlist = dims if isinstance(dims, (list, tuple)) else list(range(len(begin)))
        for d, b in zip(dlist, begin):
            try:
                bi = int(b)
            except (TypeError, ValueError):
                continue
            if bi > 0:
                g = ggml_axis(int(d), ndim)
                terms.append(f"{bi}*{a}->nb[{g}]")
    off = " + ".join(terms) if terms else "0"
    return ctx.out(node, f"ggml_view_{len(ne)}d(m, {a}, {ne_args}, {nb}{off})", hint="sl")
