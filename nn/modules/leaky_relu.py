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
from gtx_shared.quantization import maybe_get_quantizer
from gtx_shared.quantization import quantize_tensors
import gtx_utils as py_utils
from gtx_shared.utils import GtxOption

__all__ = ["leakyReLU"]


class LeakyReLU(torch.nn.LeakyReLU):
    r"""LeakyReLU operation"""

    def __init__(self, *args, **kwargs):
        # only support the specified slope and inplace operation
        super().__init__(*args, **kwargs)
        if GtxOption.gtx_leaky_relu_approximate.value:
            self.negative_slope = 0.1015625
        self.quant_mode, self.quantizer = maybe_get_quantizer()
        self.node = None

    def forward(self, input):
        qinput = quantize_tensors([input], self.node, tensor_type="input")[0]
        output = super().forward(qinput)
        output = quantize_tensors([output], self.node)[0]
        return output


@py_utils.register_quant_op
def leakyReLU(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode is None or GtxOption.gtx_quant_off.value:
        return torch.nn.LeakyReLU(*args, **kwargs)
    return LeakyReLU(*args, **kwargs)
