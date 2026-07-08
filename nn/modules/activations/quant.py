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
"""활성화 함수 op 의 quant 팩토리 (@register_quant_op).

`nn.<fn명>` 조회(MODULE_PREFIX="nn")를 위해 quant fn 명을 그대로 보존한다.
"""

import torch
import utils as py_utils
from shared.quantization import maybe_get_quantizer
from shared.utils import Option
from .relu import ReLU
from .gelu import GELU
from .sigmoid import Sigmoid
from .tanh import Tanh
from .leaky_relu import LeakyReLU
from .prelu import PReLU
from .hardsigmoid import Hardsigmoid
from .hardswish import Hardswish
from .mish import Mish
from .softmax import Softmax
from .log_softmax import LogSoftmax

__all__ = [
    "relu",
    "gelu",
    "sigmoid",
    "tanh",
    "leakyReLU",
    "prelu",
    "hardsigmoid",
    "hardswish",
    "mish",
    "softmax",
    "logSoftmax",
]


@py_utils.register_quant_op
def relu(*args, **kwargs):
    # quant_mode,_ = maybe_get_quantizer()
    # if quant_mode==None:
    #    return
    return ReLU(*args, **kwargs)


@py_utils.register_quant_op
def gelu(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode == None:
        return torch.nn.GELU(*args, **kwargs)
    return GELU(*args, **kwargs)


@py_utils.register_quant_op
def sigmoid(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode is None:
        return torch.nn.Sigmoid(*args, **kwargs)
    return Sigmoid(*args, **kwargs)


@py_utils.register_quant_op
def tanh(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode == None:
        return torch.nn.Tanh(*args, **kwargs)
    return Tanh(*args, **kwargs)


@py_utils.register_quant_op
def leakyReLU(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode is None or Option.quant_off.value:
        return torch.nn.LeakyReLU(*args, **kwargs)
    return LeakyReLU(*args, **kwargs)


@py_utils.register_quant_op
def prelu(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode == None:
        return torch.nn.PReLU(*args, **kwargs)
    return PReLU(*args, **kwargs)


@py_utils.register_quant_op
def hardsigmoid(*args, **kwargs):
    return Hardsigmoid(*args, **kwargs)


@py_utils.register_quant_op
def hardswish(*args, **kwargs):
    return Hardswish(*args, **kwargs)


@py_utils.register_quant_op
def mish(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode == None:
        return torch.nn.Mish(*args, **kwargs)
    return Mish(*args, **kwargs)


@py_utils.register_quant_op
def softmax(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode == None:
        return torch.nn.Softmax(*args, **kwargs)
    return Softmax(*args, **kwargs)


@py_utils.register_quant_op
def logSoftmax(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode == None:
        return torch.nn.LogSoftmax(*args, **kwargs)
    return LogSoftmax(*args, **kwargs)
