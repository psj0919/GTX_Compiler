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
import utils as py_utils

__all__ = ["interpolate"]


class Interpolate(torch.nn.Module):

    def __init__(self, *args, **kwards):
        super(Interpolate, self).__init__(*args, **kwards)
        self.node = None
        self.quant_mode, self.quantizer = maybe_get_quantizer()

    def forward(
        self, input, size=None, scale_factor=None, mode="nearest", align_corners=None
    ):
        qinput = quantize_tensors([input], self.node, tensor_type="input")[0]

        output = torch.nn.functional.interpolate(
            qinput, size, scale_factor, mode, align_corners
        )

        output = quantize_tensors([output], self.node)[0]

        return output


@py_utils.register_quant_op
def interpolate(*args, **kwargs):
    return Interpolate(*args, **kwargs)


# --- ggml/vision.cpp codegen (render) ---
from shared.compile.render_api import register_render as _rr
from shared.base import OP as _OP


@_rr(_OP.RESIZE, _OP.INTERPOLATE)
def render(node, ctx):
    # YOLO neck 의 upsample 은 nearest, 정수배(보통 2x). ggml_upscale(nearest) 로 매핑.
    s = ctx.scalar(ctx.attr(node, "scale", [2, 2]), 0, 2)
    mode = str(ctx.attr(node, "mode", "nearest")).lower()
    note = "" if "near" in mode else f" /* TODO(ggml): mode={mode} (upscale=nearest) */"
    return ctx.out(
        node,
        f"ggml_upscale(m, {ctx.inp(node)}, {int(s)}, GGML_SCALE_MODE_NEAREST)"
        f" /* TODO(ggml): verify ggml_upscale signature for this build */{note}",
        hint="up",
    )
