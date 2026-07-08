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
from shared.utils import Option
from ..tanh_table import *
from ..fix_ops import TanhTableLookup, TanhSimulation, TanhTableLookupAIE2

__all__ = ["Tanh"]

TANH_TABLE = TanhTable()


class Tanh(torch.nn.modules.Tanh):
    r"""Tanh operation"""

    def __init__(self):
        super(Tanh, self).__init__()
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

            fragpos = self.quantizer.get_quant_config(input_name, False)[1]
            # Method 1: Simulation AIE with 16 bw (for RNNT)
            if Option.op_tanh_sigmoid_mode.value == "simulation":
                TanhSimulation(input, output, fragpos)
                output = quantize_tensors([output], self.node)[0]
            # Method 2: Table Look up for AIE2 with 16 bw (based on LUT)
            elif (
                Option.op_tanh_sigmoid_mode.value == "aie2_lut_16bw"
                or Option.ip_asr.value
            ):
                TanhTableLookupAIE2(qinput, output, fragpos)
                output = quantize_tensors([output], self.node)[0]
            # Method 3: Table Look up for FPGA with 16 bw
            else:
                quant_device = qinput.device
                Ttable = TANH_TABLE.table.to(qinput.dtype).to(quant_device)
                output = output.to(quant_device)
                TanhTableLookup(input, Ttable, output, fragpos)
                bnfp = self.quantizer.get_quant_config(input_name, False)
                bnfp[1] = 15
                self.quantizer.set_quant_config(self.node.name, bnfp)

        return output
