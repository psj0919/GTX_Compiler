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
from shared.utils import Option, ScreenLogger, QWarning
from shared.quantization import kernel_need_quant
from shared.quantization import quantize_tensors
import numpy as np
import utils as py_utils
from .fix_ops import AIESqrt

__all__ = ["sqrt"]


class Sqrt(torch.nn.Module):
    r"""Sqrt operation"""

    def __init__(self, *args, **kwargs):
        super(Sqrt, self).__init__(*args, **kwargs)
        self.quant_mode, self.quantizer = maybe_get_quantizer()
        self.node = None

    def forward(self, input):
        # ensure input>=0
        if torch.nonzero(input < 0, as_tuple=False).numel() > 0:
            ScreenLogger().warning2user(
                QWarning.TENSOR_NEGATIVE,
                f"Elements in input tensor of node {self.node.name} are negative, which is not permittable for 'sqrt' operation. The negative numbers have been replaced by zero.",
            )
            zero = torch.zeros_like(input)
            input = torch.where(input < 0, zero, input)

        if not kernel_need_quant(self.quantizer, self.node):  # not quantizable
            output = torch.sqrt(input)
            output = quantize_tensors([output], self.node)[0]
            return output

        # quantize input tensor
        qinput = quantize_tensors([input], self.node, tensor_type="input")[0]

        if Option.op_sqrt_mode.value == "ipu_8bw":
            # sqrt=x*(1/sqrt(x)): cuda/cpu
            output = torch.empty_like(qinput)
            AIESqrt(qinput, output)  # float32

            # quantize output
            output = quantize_tensors([output], self.node, method=4)[0]
        else:
            output = torch.sqrt(qinput)
            output = quantize_tensors([output], self.node)[0]
        return output


@py_utils.register_quant_op
def sqrt(*args, **kwargs):
    return Sqrt(*args, **kwargs)
