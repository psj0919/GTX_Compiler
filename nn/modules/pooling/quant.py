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
"""pooling op 의 quant 팩토리 (@register_quant_op).

`nn.<fn명>` 조회(MODULE_PREFIX="nn")를 위해 quant fn 명을 그대로 보존한다.
"""

import torch
import utils as py_utils
from shared.quantization import maybe_get_quantizer
from shared.utils import Option
from .maxpool import MaxPool2d
from .maxpool1d import MaxPool1d
from .avgpool import AvgPool2d
from .adaptive_avg_pool import AdaptiveAvgPool2d
from .interpolate import Interpolate

__all__ = [
    "maxPool2d",
    "maxPool1d",
    "avgPool2d",
    "adaptiveAvgPool2d",
    "interpolate",
]


@py_utils.register_quant_op
def maxPool2d(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode == None:
        return torch.nn.MaxPool2d(*args, **kwargs)
    return MaxPool2d(*args, **kwargs)


@py_utils.register_quant_op
def maxPool1d(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode == None:
        return torch.nn.MaxPool1d(*args, **kwargs)
    return MaxPool1d(*args, **kwargs)


@py_utils.register_quant_op
def avgPool2d(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode is None or Option.quant_off.value:
        return torch.nn.AvgPool2d(*args, **kwargs)
    return AvgPool2d(*args, **kwargs)


@py_utils.register_quant_op
def adaptiveAvgPool2d(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode is None or Option.quant_off.value:
        return torch.nn.AdaptiveAvgPool2d(*args, **kwargs)
    return AdaptiveAvgPool2d(*args, **kwargs)


@py_utils.register_quant_op
def interpolate(*args, **kwargs):
    return Interpolate(*args, **kwargs)
