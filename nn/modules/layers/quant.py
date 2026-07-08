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
"""layer op 의 quant 팩토리 (@register_quant_op).

`nn.<fn명>` 조회(MODULE_PREFIX="nn")를 위해 quant fn 명을 그대로 보존한다.
"""

import torch
from shared.quantization import maybe_get_quantizer
import utils as py_utils
from utils.op_register import register_quant_op
from .conv import Conv2d
from .conv1d import Conv1d
from .conv_transpose import ConvTranspose2d
from .linear import Linear
from .matmul import Matmul
from .embedding import Embedding
from .add import Add
from .sub import Sub
from .multiply import Mul
from .mean import Mean
from .concat import Concat

__all__ = [
    "conv2d",
    "conv1d",
    "convTranspose2d",
    "linear",
    "matmul",
    "embedding",
    "add",
    "sub",
    "mul",
    "mean",
    "concat",
]


@register_quant_op
def conv2d(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode == None:
        return torch.nn.Conv2d(*args, **kwargs)
    return Conv2d(*args, **kwargs)


@py_utils.register_quant_op
def conv1d(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode == None:
        return torch.nn.Conv1d(*args, **kwargs)
    return Conv1d(*args, **kwargs)


@py_utils.register_quant_op
def convTranspose2d(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode == None:
        return torch.nn.ConvTranspose2d(*args, **kwargs)
    return ConvTranspose2d(*args, **kwargs)


@py_utils.register_quant_op
def linear(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode == None:
        return torch.nn.Linear(*args, **kwargs)
    return Linear(*args, **kwargs)


@py_utils.register_quant_op
def matmul(*args, **kwargs):
    return Matmul(*args, **kwargs)


@py_utils.register_quant_op
def embedding(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode == None:
        return torch.nn.Embedding(*args, **kwargs)
    return Embedding(*args, **kwargs)


@py_utils.register_quant_op
def add(*args, **kwargs):
    return Add(*args, **kwargs)


@py_utils.register_quant_op
def sub(*args, **kwargs):
    return Sub(*args, **kwargs)


@py_utils.register_quant_op
def mul(*args, **kwargs):
    return Mul(*args, **kwargs)


@py_utils.register_quant_op
def mean(*args, **kwargs):
    return Mean(*args, **kwargs)


@py_utils.register_quant_op
def concat(*args, **kwargs):
    return Concat(*args, **kwargs)
