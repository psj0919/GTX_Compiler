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
import numpy as np
from shared.quantization import maybe_get_quantizer
from shared.quantization import quantize_tensors
from shared.utils import Option
from shared.base import GLOBAL_MAP, KEYS
from .sigmoid_table import *
from .fix_ops import (
    SigmoidTableLookup,
    SigmoidSimulation,
    SigmoidTableLookupAIE2,
)
import utils as py_utils

__all__ = ["sigmoid"]

SIGMOID_TABLE = SigmoidTable()


class Sigmoid(torch.nn.modules.Sigmoid):
    r"""Sigmoid operation"""

    def __init__(self):
        super(Sigmoid, self).__init__()
        self.quant_mode, self.quantizer = maybe_get_quantizer()
        self.node = None

    def forward(self, input):
        if self.quant_mode == 0 or (not self.node.in_quant_part):
            return super().forward(input)

        qinput = quantize_tensors([input], self.node, tensor_type="input")[0]

        if (
            Option.quant_off.value
            or self.quantizer is None
            or self.quantizer.exporting
            or Option.cv_app.value
            or Option.only_int_quant is False
        ):
            # Method 0: quant input and output (for CV)
            output = super().forward(qinput)
            output = quantize_tensors([output], self.node)[0]

        else:
            output = torch.empty_like(qinput)
            input_name = self.node.in_nodes[0]
            input_node = self.quantizer.configer.get_node(input_name)
            if not self.quantizer.configer.node_output_quantizable(input_node):
                input_name = input_node.in_nodes[0]
            elif self.quantizer.configer.will_merge_with_table(
                input_node, (not Option.cv_app.value)
            ):
                output = super().forward(qinput)
                bnfp = self.quantizer.get_quant_config(input_name, False)
                bnfp[1] = 15
                self.quantizer.set_quant_config(self.node.name, bnfp)
                return output

            bw = self.quantizer.get_quant_config(self.node.name, False)[0]
            fragpos = self.quantizer.get_quant_config(input_name, False)[1]
            # Method 1: Simulation AIE with 16 bw (for RNNT)
            if Option.op_tanh_sigmoid_mode.value == "simulation":
                SigmoidSimulation(qinput, output, fragpos)
                output = quantize_tensors([output], self.node)[0]
            # Method 2: Table Look up for AIE2 with 16 bw (based on LUT)
            elif (
                Option.op_tanh_sigmoid_mode.value == "aie2_lut_16bw"
                or Option.ip_asr.value
            ):
                SigmoidTableLookupAIE2(qinput, output, fragpos)
                output = quantize_tensors([output], self.node)[0]
            # Method 3: Table Look up for FPGA with 16 bw
            else:
                quant_device = qinput.device
                Ttable = SIGMOID_TABLE.table.to(qinput.dtype).to(quant_device)
                output = output.to(quant_device)
                SigmoidTableLookup(input, Ttable, output, fragpos)
                bnfp = self.quantizer.get_quant_config(input_name, False)
                bnfp[1] = 15
                self.quantizer.set_quant_config(self.node.name, bnfp)

        return output


@py_utils.register_quant_op
def sigmoid(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode is None:
        return torch.nn.Sigmoid(*args, **kwargs)
    return Sigmoid(*args, **kwargs)


# --- ggml/vision.cpp codegen (render) ---
from shared.compile.render_api import register_render as _rr
from shared.base import OP as _OP


@_rr(_OP.SIGMOID)
def render(node, ctx):
    return ctx.out(node, f"ggml_sigmoid(m, {ctx.inp(node)})")
