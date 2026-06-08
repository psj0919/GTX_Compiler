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

import math
import torch

import utils as py_utils
from shared.quantization import maybe_get_quantizer, quantize_tensors
from shared.utils import Option, ScreenLogger

__all__ = ["adaptiveAvgPool2d"]


class AdaptiveAvgPool2d(torch.nn.modules.AdaptiveAvgPool2d):
    r"""AdaptiveAvgPool2d operation, support float and double"""

    def __init__(self, *args, **kwards):
        super(AdaptiveAvgPool2d, self).__init__(*args, **kwards)
        self.node = None
        self.quant_mode, self.quantizer = maybe_get_quantizer()

    def forward(self, input):
        qinput = quantize_tensors([input], self.node, tensor_type="input")[0]
        output = super().forward(qinput)

        input_size = [int(dim) for dim in input.shape[2:]]
        mod = [input_size[i] % self.output_size[i] for i in range(0, len(input_size))]
        if mod != [0] * len(mod):
            if self.node is not None:
                ScreenLogger().warning_once(
                    f"AdaptiveAvgpool2d op({self.node.name}) is not quantized. Because it's output size {self.output_size} are not factor of input size {input_size}."
                )
            return output
        # During slow trace, the dim of shape will convert to tensor value which is not support in .
        kernel = [
            int(input_size[i] / self.output_size[i]) for i in range(0, len(input_size))
        ]
        # scale to DPU accuracy
        if Option.avg_pool_approximate.value:
            scale = 1.0
            if kernel == [3, 3]:
                scale = 9.0 * 7.0 / 64.0
            elif kernel == [5, 5]:
                scale = 25.0 * 10.0 / 256.0
            elif kernel in [[6, 6], [3, 6], [6, 3]]:
                scale = 36.0 * 7.0 / 256.0
            elif kernel == [7, 7]:
                scale = 49.0 * 21.0 / 1024.0
            elif kernel == [14, 14]:
                scale = 196.0 * 21.0 / 4096.0
            else:
                rec = kernel[0] * kernel[1]
                max_factor = math.ceil(math.log(rec * 128, 2))
                diff = 1.0
                multi_factor = 0.0
                shift_factor = 0.0
                for shift_factor_ in range(max_factor):
                    factor = round((2**shift_factor_) / rec)
                    diff_ = abs(factor / (2**shift_factor_) - 1 / rec)
                    if diff_ < diff:
                        multi_factor = factor
                        diff = diff_
                        shift_factor = shift_factor_
                scale = rec * multi_factor / (2**shift_factor)

            output = output * scale

        output = quantize_tensors([output], self.node)[0]

        return output


@py_utils.register_quant_op
def adaptiveAvgPool2d(*args, **kwargs):
    quant_mode, _ = maybe_get_quantizer()
    if quant_mode is None or Option.quant_off.value:
        return torch.nn.AdaptiveAvgPool2d(*args, **kwargs)
    return AdaptiveAvgPool2d(*args, **kwargs)


# --- ggml/vision.cpp codegen (render) ---
from shared.compile.render_api import register_render as _register_render
from shared.base import OP as _OP


@_register_render(_OP.ADAPTIVEAVGPOOL2D)
def render(node, ctx):
    # output_size 는 그래프 attr 에 없으므로 출력 shape 로 판별한다(NCHW 가정: H,W=마지막 2).
    # output_size==(1,1) 이면 전 spatial 평균 = global avg → ggml_pool_2d(full kernel) 로 정확.
    a = ctx.inp(node)
    is_global = False
    try:
        oshape = list(node.out_tensors[0].shape)
        if len(oshape) >= 2 and int(oshape[-1]) == 1 and int(oshape[-2]) == 1:
            is_global = True
    except Exception:
        pass
    expr = (
        f"ggml_pool_2d(m, {a}, GGML_OP_POOL_AVG, "
        f"{a}->ne[0], {a}->ne[1], {a}->ne[0], {a}->ne[1], 0, 0)"
    )
    if not is_global:
        expr += " /* TODO(ggml): non-(1,1) adaptive avg pool — needs tiled pooling */"
    return ctx.out(node, expr, hint="pool")
