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
"""`torch.var` / `torch.std` 의 op 모듈.

weight standardization(`ConvWS2d`/`ConvAWS2d`)이 `weight.view(c, -1).std(dim=1)` 로
가중치를 정규화한다 — DetectoRS 계열이 이걸 쓴다.

**왜 모듈 파일이 따로 필요한가**: 렌더러(`math/render.py`)만 있으면 C++ 생성은 되지만
ScriptWriter 의 `.py` export 경로가 op 클래스를 못 찾아 실패한다
(`op_class_type of op (unknown) is unknown`). 다른 op 들과 같은 구조로 맞춘다.
"""

import torch
from shared.quantization import maybe_get_quantizer
from shared.quantization import quantize_tensors

__all__ = ["Variance", "Std"]


class _Reduce(torch.nn.Module):
    """var/std 공통 — dim/unbiased/keepdim 을 그대로 torch 에 넘긴다."""

    _fn = None

    def __init__(self):
        super().__init__()
        self.quant_mode, self.quantizer = maybe_get_quantizer()
        self.node = None

    def forward(self, input, dim=None, unbiased=True, keepdim=False):
        [qinput] = quantize_tensors([input], self.node, tensor_type="input")
        if dim is None:
            output = self._fn(qinput, unbiased=unbiased)
        else:
            output = self._fn(qinput, dim=dim, unbiased=unbiased, keepdim=keepdim)
        output = quantize_tensors([output], self.node)[0]
        return output


class Variance(_Reduce):
    _fn = staticmethod(torch.var)


class Std(_Reduce):
    _fn = staticmethod(torch.std)
