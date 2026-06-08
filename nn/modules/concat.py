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

import torch
from shared.quantization import maybe_get_quantizer
from shared.quantization import quantize_tensors
import utils as py_utils

__all__ = ["concat"]


class Concat(torch.nn.Module):
    r"""Concat operation"""

    def __init__(self, *args, **kwargs):
        super(Concat, self).__init__()
        # self.dim = kwargs.get('dim', 0)
        self.quant_mode, self.quantizer = maybe_get_quantizer()
        self.node = None

    def forward(self, tensors, dim):
        qinputs = quantize_tensors(tensors, self.node, tensor_type="input")
        output = torch.cat(qinputs, dim)
        output = quantize_tensors([output], self.node)[0]

        return output


@py_utils.register_quant_op
def concat(*args, **kwargs):
    return Concat(*args, **kwargs)


# --- ggml/vision.cpp codegen (render) ---
from shared.compile.render_api import register_render as _rr, out_shape, ggml_axis
from shared.base import OP as _OP


@_rr(_OP.CONCAT)
def render(node, ctx):
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
