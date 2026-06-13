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
from shared.quantization.utils import maybe_get_quantizer
from shared.quantization import quantize_tensors
import utils as py_utils

__all__ = ["matmul"]


class Matmul(torch.nn.Module):

    def __init__(self):
        super(Matmul, self).__init__()
        self.quant_mode, self.quantizer = maybe_get_quantizer()
        self.node = None

    def forward(self, input, other):
        [qinput, qother] = quantize_tensors(
            [input, other], self.node, tensor_type="input"
        )
        output = torch.matmul(input=qinput, other=qother)
        output = quantize_tensors([output], self.node)[0]
        return output


@py_utils.register_quant_op
def matmul(*args, **kwargs):
    return Matmul(*args, **kwargs)


# --- ggml/vision.cpp codegen (render) ---
from shared.compile.render_api import register_render as _register_render
from shared.base import OP as _OP


@_register_render(_OP.MATMUL)
def render(node, ctx):
    # torch matmul(A,B): A=(...,M,K), B=(...,K,N) → C=(...,M,N).
    # ggml_mul_mat(a,b) 는 a.ne0==b.ne0 인 수축축 K 를 요구하고 결과 ne=(a.ne1,b.ne1).
    # in0(A) 는 natural reverse 라 ne0=K 이미 OK. in1(B) 는 ne0=N 이므로 앞 두 ggml축을
    # swap(=마지막 두 torch축 transpose)해 ne0=K 로 만든 뒤, 피연산자를 (Bt, A) 순서로
    # 주면 결과 ne=(N,M)=torch C 의 natural reverse 가 된다 (검증: ggml_backend._op_matmul).
    a = ctx.inp(node, 0)
    b = ctx.inp(node, 1)
    bt = f"ggml_cont(m, ggml_permute(m, {b}, 1, 0, 2, 3))"
    return ctx.out(node, f"ggml_mul_mat(m, {bt}, {a})", hint="mm")
