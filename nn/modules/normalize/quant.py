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
"""정규화 op 의 quant 팩토리 (@register_quant_op).

`nn.<fn명>` 조회(MODULE_PREFIX="nn")를 위해 quant fn 명을 그대로 보존한다.
"""

import torch
from shared.quantization import maybe_get_quantizer
import utils as py_utils
from .batch_norm import BatchNorm
from .instance_norm import InstanceNorm
from .group_norm import GroupNorm
from .layernorm import LayerNorm

__all__ = [
    "batchNorm",
    "instanceNorm",
    "groupNorm",
    "layerNorm",
]


@py_utils.register_quant_op
def batchNorm(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode == None:

        def _check_input_dim(self, input):
            pass

        import types

        nn = torch.nn.modules.batchnorm._BatchNorm(*args, **kwargs)

        nn._check_input_dim = types.MethodType(_check_input_dim, nn)
        return nn
    return BatchNorm(*args, **kwargs)


@py_utils.register_quant_op
def instanceNorm(*args, **kwargs):
    return InstanceNorm(*args, **kwargs)


@py_utils.register_quant_op
def groupNorm(*args, **kwargs):
    return GroupNorm(*args, **kwargs)


@py_utils.register_quant_op
def layerNorm(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode is None:
        return torch.nn.LayerNorm(*args, **kwargs)
    return LayerNorm(*args, **kwargs)
