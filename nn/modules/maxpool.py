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
from torch.autograd import Variable

from shared.quantization import maybe_get_quantizer
from shared.utils import Option
from shared.quantization import quantize_tensors
import utils as py_utils

__all__ = ["maxPool2d"]


class MaxPool2d(torch.nn.modules.MaxPool2d):
    r"""MaxPool2d operation, support float and double"""

    def __init__(self, *args, **kwards):
        super(MaxPool2d, self).__init__(*args, **kwards)
        self.node = None
        self.quant_mode, self.quantizer = maybe_get_quantizer()

    def forward(self, input):
        qinput = quantize_tensors([input], self.node, tensor_type="input")[0]
        output = super().forward(qinput)
        output = quantize_tensors([output], self.node)[0]
        return output


@py_utils.register_quant_op
def maxPool2d(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode == None:
        return torch.nn.MaxPool2d(*args, **kwargs)
    return MaxPool2d(*args, **kwargs)


# --- ggml/vision.cpp codegen (render) ---
from shared.compile.render_api import register_render as _register_render
from shared.base import OP as _OP


@_register_render(_OP.MAX_POOL)
def render(node, ctx):
    k = ctx.scalar(ctx.attr(node, "kernel_size", [2, 2]))
    s = ctx.scalar(ctx.attr(node, "stride", [k, k]))
    p = ctx.scalar(ctx.attr(node, "padding", [0, 0]))
    return ctx.out(
        node,
        f"ggml_pool_2d(m, {ctx.inp(node)}, GGML_OP_POOL_MAX, {k}, {k}, {s}, {s}, {p}, {p})"
        " /* TODO(ggml): verify pool params/layout */",
    )
