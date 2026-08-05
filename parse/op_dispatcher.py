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
# Note: input list (number, names and order) must match with torch: torch/_C/_VariableFunctions.pyi

import math
import sys
import torch
import numpy as np
from typing import Callable, Dict
from shared.graph import Node, Tensor, Operation
from shared.base import GLOBAL_MAP, KEYS
from shared.utils import tensor_util, ScreenLogger
from .torch_op_def import *
from .parse_utils import *
from utils import SchemaHelper, TorchOpClassType, convert_type_str
from utils.jit_utils import *
from utils.torch_utils import CmpFlag, compare_torch_version


def _as_int(x, default=None):
    """wrapped 상수(파서 Tensor 래퍼/스칼라/넘파이 등)에서 파이썬 int 를 방어적으로 추출."""
    for get in (lambda v: int(v),
                lambda v: int(v.item()),
                lambda v: int(getattr(v, "data")),
                lambda v: int(getattr(v, "value"))):
        try:
            return get(x)
        except Exception:
            continue
    return default


def _as_int_list(x):
    """int[] 상수를 파이썬 list[int] 로 추출. 실패 원소가 있으면 None 반환(→ render 가 shape 로 추론)."""
    seq = x
    if not isinstance(seq, (list, tuple)):
        for attr in ("data", "value"):
            v = getattr(seq, attr, None)
            if isinstance(v, (list, tuple)):
                seq = v
                break
        else:
            try:
                seq = list(seq)
            except Exception:
                return None
    out = []
    for s in seq:
        iv = _as_int(s)
        if iv is None:
            return None
        out.append(iv)
    return out or None


class OpCreator(object):

    op_convert_map = {
        "__getitem__": "index",
        "TupleUnpack": "tuple_unpack",
        "ListUnpack": "tuple_unpack",
        "__derive_index": "derive_loop_index",
        "_cast_Float": "cast_float",
        "_cast_Int": "cast_int",
        "add_": "add",
        "mul_": "mul",
        "__or__": "logical_or",
        "masked_fill_": "masked_fill",
        "__interpolate": "_interpolate",
        "floordiv": "div",
        "ListConstruct": "list_construct",
        "TupleConstruct": "tuple_construct",
        "TupleIndex": "tuple_index",
        "ScalarImplicit": "item",
        "IntImplicit": "Int",
    }

    def __init__(self):
        self._device_type = GLOBAL_MAP.get_ele(KEYS.DEVICE)

    # torch function should always add 'input' config
    # nn.function, nn.Module and torch.Tensor can ignore 'input'

    def Param(self, *args):
        if "tuple" in args[0].dtype:
            op = TorchUnaryOp(OP.TUPLE_INPUT, "input", force_to_primitive=True)
        else:
            op = TorchUnaryOp(OP.INPUT, "input", force_to_primitive=True)
        input_name = f"args[{args[0].node.name.split('_')[-1]}]"
        op.set_config("input", input_name)
        return op

    def Return(self, *args):
        op = TorchReturn()
        op.set_config("input", args)
        return op

    def _convolution(
        self,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        benchmark,
        deterministic,
        cudnn_enabled,
        *args,
    ):
        """
        input: 1D:(N,Cin,Lin) 2D:(N,Cin,Hin,Win) 3D: (N,Cin,Din,Hin,Win)
        conv2d    weight:
                (out_channels, in_channels/groups, kernel_size_0, kernel_size_1)
        transpose conv2d weight:
                (in_channels, out_channels/groups, kernel_size_0, kernel_size_1)
        """
        if (weight and weight.node != None) or (bias and bias.node != None):
            # 동적 weight conv (예: DetectoRS 의 ConvAWS2d — 표준화된 weight 가 런타임 계산
            # 텐서다). weight/bias 를 param 이 아닌 in_tensor 로 남겨 render 가
            # `ggml_conv_2d(m, W, x, …)` 로 emit 한다. 정규 2D conv(ndim==4, non-transposed,
            # groups==1)만 지원 — 그 외는 기존 raise.
            # ⚠️ 여기서 raise 하면 op 이 통째로 unhandled 로 떨어져 **입력 이미지가 그대로**
            #    다음 BN 으로 흘러간다(detectors 실측: `op149 = x` → batch_norm_2d abort).
            if (weight and weight.node is not None and not transposed
                    and getattr(weight, "ndim", 0) == 4 and int(groups) == 1):
                op = TorchConv2dDynamic()
                op.set_config("stride", _as_int_list(stride) or [1, 1])
                op.set_config("padding", _as_int_list(padding) or [0, 0])
                op.set_config("dilation", _as_int_list(dilation) or [1, 1])
                op.set_config("has_bias", bias is not None)
                return op
            raise ("weight or bias is not a constant param!")
        weight_size = weight.shape
        if transposed:
            weight_size[0], weight_size[1] = weight_size[1], weight_size[0]
            if weight_size[0] == 1 and groups == weight_size[1]:
                if weight.ndim == 4:
                    op = TorchConvTranspose2d(OP.DEPTHWISE_CONVTRANSPOSE2D)
                elif weight.ndim == 5:
                    op = TorchConvTranspose3d(OP.DEPTHWISE_CONVTRANSPOSE3D)
                elif weight.ndim == 3:
                    raise NotImplementedError("Depthwise_ConvTranpose1D is unsupported")
            else:
                if weight.ndim == 4:
                    op = TorchConvTranspose2d(OP.CONVTRANSPOSE2D)
                elif weight.ndim == 5:
                    op = TorchConvTranspose3d(OP.CONVTRANSPOSE3D)
                elif weight.ndim == 3:
                    raise NotImplementedError("ConvTranpose1D is unsupported")

            op.set_config("output_padding", list(output_padding))
            op.set_config("in_channels", weight_size[1])
            op.set_config("out_channels", weight_size[0] * groups)
        else:
            # When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
            # this operation is also known as a “depthwise convolution”.
            input_channel = weight_size[1] * groups
            if (
                groups != 1
                and groups == input_channel
                and weight_size[0] % input_channel == 0
            ):
                if weight.ndim == 4:
                    op = TorchConv2d(OP.DEPTHWISE_CONV2D)
                elif weight.ndim == 5:
                    op = TorchConv3d(OP.DEPTHWISE_CONV3D)
                elif weight.ndim == 3:
                    op = TorchConv1d(OP.DEPTHWISE_CONV1D)
            else:
                if weight.ndim == 4:
                    op = TorchConv2d(OP.CONV2D)
                elif weight.ndim == 5:
                    op = TorchConv3d(OP.CONV3D)
                elif weight.ndim == 3:
                    op = TorchConv1d(OP.CONV1D)

            op.set_config("in_channels", input_channel)
            op.set_config("out_channels", weight_size[0])

        # Should add weight first
        op.set_param(op.ParamName.WEIGHTS, weight)
        if bias is not None:
            op.set_config("bias", True)
            op.set_param(op.ParamName.BIAS, bias)
        else:
            op.set_config("bias", False)

        op.set_config("dilation", list(dilation))
        op.set_config("kernel_size", list(weight_size[2:]))
        op.set_config("stride", list(stride))

        op.set_config("groups", groups)
        op.set_config("padding", list(padding))

        return op

    def batch_norm(
        self,
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        cudnn_enabled,
    ):

        if (weight and weight.node != None) or (bias and bias.node != None):
            raise ("weight or bias is not a constant param!")
        op = TorchBatchNorm()
        op.set_param(op.ParamName.GAMMA, weight)
        op.set_param(op.ParamName.BETA, bias)
        op.set_param(op.ParamName.MOVING_MEAN, running_mean)
        op.set_param(op.ParamName.MOVING_VAR, running_var)
        weight_size = weight.shape

        op.set_config("num_features", weight_size[0])
        op.set_config("eps", eps)
        op.set_config("momentum", momentum)
        op.set_attr(op.AttrName.SCALE, True)
        op.set_attr(op.AttrName.CENTER, True)

        # dims2axis_map = {
        #     2: 1,  # NC -> 1
        #     3: 1,  # NCL -> 1
        #     4: 3,  # NHWC -> 3
        #     5: 1  # NCDHW -> 1
        # }
        # op.set_attr(op.AttrName.AXIS, dims2axis_map[len(input.shape)])
        return op

    def instance_norm(
        self,
        input,
        weight,
        bias,
        running_mean,
        running_var,
        use_input_stats,
        momentum,
        eps,
        cudnn_enabled,
    ):
        # num_features: :math:`C` from an expected input of size
        #     1d: :math:`(N, C, L)`
        #     2d: :math:`(N, C, H, W)`
        #     3d: math:`(N, C, D, H, W)`
        # eps: a value added to the denominator for numerical stability. Default: 1e-5
        # momentum: the value used for the running_mean and running_var computation. Default: 0.1
        # affine: a boolean value that when set to ``True``, this module has learnable affine parameters,
        #    initialized the same way as done for batch normalization. Default: ``False``.
        # track_running_stats: a boolean value that when set to ``True``, this module tracks the running mean and variance,
        #    and when set to ``False``, this module does not track such statistics and always uses batch statistics in both
        #    training and eval modes. Default: ``False``.
        # Input: :math:`(N, C, D, H, W)`, math:`(N, C, H, W)`, or :math:`(N, C, L)`
        # Output: same shape as input
        if (weight and weight.node != None) or (bias and bias.node != None):
            raise ("weight or bias is not a constant param!")
        op = TorchInstanceNorm()
        if weight is not None:
            op.set_param(op.ParamName.GAMMA, weight)
        if bias is not None:
            op.set_param(op.ParamName.BETA, bias)

        op.set_config(
            "num_features", input.shape[1]
        )  # `C` from an expected input of size
        op.set_config("eps", eps)
        op.set_config("momentum", momentum)
        if weight is not None or bias is not None:
            op.set_config("affine", True)
        else:
            op.set_config("affine", False)

        return op

    def group_norm(self, input, num_groups, weight, bias, eps, cudnn_enabled):
        # num_groups (int): number of groups to separate the channels into
        # num_channels (int): number of channels expected in input
        # eps: a value added to the denominator for numerical stability. Default: 1e-5
        # affine: a boolean value that when set to ``True``, this module has learnable per-channel affine parameters
        #     initialized to ones (for weights) and zeros (for biases). Default: ``True``.
        # Shape:
        #    - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        #    - Output: :math:`(N, C, *)` (same shape as input)
        if (weight and weight.node != None) or (bias and bias.node != None):
            raise ("weight or bias is not a constant param!")

        op = TorchGroupNorm()
        if weight is not None:
            op.set_param(op.ParamName.GAMMA, weight)
        if bias is not None:
            op.set_param(op.ParamName.BETA, bias)

        op.set_config(
            "num_groups", num_groups
        )  # number of groups to separate the channels into
        op.set_config(
            "num_channels", input.shape[1]
        )  # `C` from an expected input of size
        op.set_config("eps", eps)
        if weight is not None or bias is not None:
            op.set_config("affine", True)
        else:
            op.set_config("affine", False)

        return op

    @staticmethod
    def _max_pool1d(op, input, kernel_size, stride, padding, dilation, ceil_mode):
        op.set_config("kernel_size", list(kernel_size))
        if not stride:
            op.set_config("stride", list(kernel_size))
        else:
            op.set_config("stride", list(stride))

        if ceil_mode:
            op.set_config("ceil_mode", True)
        else:
            op.set_config("ceil_mode", False)

        op.set_config("padding", list(padding))
        op.set_config("dilation", list(dilation))

        return op

    def max_pool1d(self, *args):
        op = TorchMaxPool1d()
        return self._max_pool1d(op, *args)

    def max_pool1d_with_indices(self, *args):
        op = TorchMaxPool1d()
        op.set_config("return_indices", True)
        return self._max_pool1d(op, *args)

    @staticmethod
    def _max_pool2d(op, input, kernel_size, stride, padding, dilation, ceil_mode):
        op.set_config("kernel_size", list(kernel_size))
        if not stride:
            op.set_config("stride", list(kernel_size))
        else:
            op.set_config("stride", list(stride))

        if ceil_mode:
            op.set_config("ceil_mode", True)
        else:
            op.set_config("ceil_mode", False)

        op.set_config("padding", list(padding))
        op.set_config("dilation", list(dilation))

        return op

    def max_pool2d(self, *args):
        op = TorchMaxPool()
        return self._max_pool2d(op, *args)

    def max_pool2d_with_indices(self, *args):
        op = TorchMaxPool()
        op.set_config("return_indices", True)
        return self._max_pool2d(op, *args)

    def avg_pool2d(
        self,
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override=None,
    ):

        op = TorchAvgPool()

        op.set_config("kernel_size", list(kernel_size))
        if not stride:
            op.set_config("stride", list(kernel_size))
        else:
            op.set_config("stride", list(stride))

        if ceil_mode:
            op.set_config("ceil_mode", True)
        else:
            op.set_config("ceil_mode", False)

        if count_include_pad:
            op.set_config("count_include_pad", True)
        else:
            op.set_config("count_include_pad", False)

        op.set_config("padding", list(padding))
        if divisor_override is not None:
            op.set_config("divisor_override", int(divisor_override))

        return op

    def adaptive_avg_pool2d(self, *args):
        op = TorchAdaptiveAvgPool()
        op.set_config("output_size", args[1])
        return op

    def addmm(self, input, mat1, mat2, beta=None, alpha=None):
        op = TorchBaseOperation(OP.ADDMM, "addmm")
        op.set_config("input", input)
        op.set_config("mat1", mat1)
        op.set_config("mat2", mat2)
        return op

    def t_addmm(self, bias, input, weight, beta=None, alpha=None):
        if (weight and weight.node != None) or (bias and bias.node != None):
            raise ("weight or bias is not a constant param!")
        if len(weight.shape) <= 2:
            op = TorchLinear()
            weight_size = weight.shape
            op.set_param(op.ParamName.WEIGHTS, weight)
            if bias is None:
                op.set_config("bias", False)
            else:
                op.set_config("bias", True)
                op.set_param(op.ParamName.BIAS, bias)

            op.set_config("out_features", weight_size[0])
            op.set_config("in_features", weight_size[1])
        return op

    def linear(self, input, weight, bias):
        if (weight and weight.node != None) or (bias and bias.node != None):
            # 동적 weight/bias (예: nn.MultiheadAttention 이 in_proj_weight 를 split_with_sizes 로
            # 나눈 q/k/v chunk). 상수 FC 가 아니므로 그래프 텐서 matmul 로 처리한다. weight/bias 를
            # param 으로 잡지 않으면 driver 가 이미 채운 in_tensors(input,weight,[bias])로 남는다.
            # ⚠️ 여기서 raise 하면 **op 이 통째로 unhandled 로 떨어져** attention 이 passthrough 가
            #    된다(pvt: aten::linear 32건 + split_with_sizes 32건이 전부 TODO 주석으로 나갔다).
            op = TorchLinearDynamic()
            op.set_config("has_bias", bias is not None)
            return op
        op = TorchLinear()
        weight_size = weight.shape
        op.set_param(op.ParamName.WEIGHTS, weight)
        if bias is None:
            op.set_config("bias", False)
        else:
            op.set_config("bias", True)
            op.set_param(op.ParamName.BIAS, bias)

        op.set_config("out_features", weight_size[0])
        op.set_config("in_features", weight_size[1])
        return op

    def _std_var(self, optype, name, *args):
        # std/var(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False). ConvWS2d 의
        # weight standardization(weight.view(c,-1).std(dim=1)) 등. render 가 mean·sub·sqr·mean·sqrt 로 전개.
        sstr = SchemaHelper(self.cur_node.schema).toString()
        # op 클래스가 있으면 그걸 쓴다 — ScriptWriter 의 `.py` export 가 클래스로 모듈을 찾는다.
        # (없으면 `op_class_type of op (unknown) is unknown` 으로 export 만 실패한다.)
        op = (TorchVariance() if optype == OP.VARIANCE
              else TorchStd() if optype == OP.STD
              else TorchPermuteInvarOp(optype, name))
        op.set_config("input", args[0])
        dim = None
        unbiased = True
        keepdim = False
        if len(args) >= 2 and ("dim" in sstr):
            d = args[1]
            if isinstance(d, int) and not isinstance(d, bool):
                d = [d]
            if isinstance(d, (list, tuple)) and all(
                isinstance(x, int) and not isinstance(x, bool) for x in d
            ):
                dim = list(d)
            for a in args[2:]:
                if isinstance(a, bool):
                    unbiased = a
                    break
            if len(args) >= 4 and isinstance(args[3], bool):
                keepdim = args[3]
        elif len(args) >= 2 and isinstance(args[1], bool):
            unbiased = args[1]
        op.set_config("dim", dim)
        op.set_config("unbiased", unbiased)
        op.set_config("keepdim", keepdim)
        return op

    def var(self, *args):
        return self._std_var(OP.VARIANCE, "variance", *args)

    def std(self, *args):
        return self._std_var(OP.STD, "std", *args)

    def adaptive_max_pool2d(self, *args):
        # BFP(Libra R-CNN) 등: 고해상도 레벨을 gather 크기로 다운샘플. render 가 out/in shape 로
        # kernel=stride=in//out 계산(정수배면 정확). multi-output(값,인덱스)이나 값만 사용.
        # 없으면 unhandled 로 떨어져 **다운샘플이 통째로 생략**되고, BFP 가 128x128 을 그대로
        # 합쳐 뒤 reshape 이 nelements 로 죽는다(libra_rcnn 실측).
        op = TorchAdaptiveMaxPool2d()
        op.set_config("input", args[0])
        if len(args) > 1:
            op.set_config("output_size", args[1])
        return op

    def split_with_sizes(self, input, split_sizes, dim=0):
        # split_with_sizes(x, [s0,s1,...], dim) = dim 축을 s_i 크기로 나눈 N개 텐서(unbind 와 달리
        # 축을 제거하지 않고 크기만 자름). nn.MultiheadAttention 의 in_proj_weight[3E,E]→q/k/v 등.
        # multi-output → render(render_split)가 out_tensors 별 view(누적 offset)를 emit.
        # split_sizes/dim 은 wrapped 상수(Tensor)일 수 있다 → 방어적 추출. 크기 추출 실패해도
        # render 가 out_tensor shape 에서 크기를 추론하므로 dim 만 확실히 넘기면 된다.
        op = TorchBaseOperation(OP.SPLIT, "split")
        op.set_config("input", input)
        op.set_config("dim", _as_int(dim, 0))
        sizes = _as_int_list(split_sizes)
        if sizes is not None:
            op.set_config("split_sizes", sizes)
        return op

    def split(self, input, split_size, dim=0):
        # split(x, size, dim): 균등 크기 size 로 분할. split_with_sizes 로 위임(render 가 크기 추론).
        op = TorchBaseOperation(OP.SPLIT, "split")
        op.set_config("input", input)
        op.set_config("dim", _as_int(dim, 0))
        return op

    def unbind(self, input, dim=0):
        # unbind(x, dim) = dim 축을 따라 N개 텐서로 분리(각 = select(dim, i)). timm attention 의
        # qkv.unbind(0) 등. multi-output → render(render_unbind)가 out_tensors 별 view 를 emit.
        op = TorchBaseOperation(OP.UNBIND, "unbind")
        op.set_config("input", input)
        op.set_config("dim", _as_int(dim, 0))
        return op

    def flatten(self, input, start_dim=0, end_dim=-1):
        op = TorchFlatten()
        op.set_config("input", input)
        op.set_config("start_dim", start_dim)
        op.set_config("end_dim", end_dim)
        return op

    def relu_(self, input):
        op = TorchReLU()
        op.set_config("inplace", True)
        return op

    def relu(self, input):
        op = TorchReLU()
        op.set_config("inplace", False)
        return op

    def silu_(self, input):
        op = TorchSiLU()
        op.set_config("inplace", True)
        return op

    def silu(self, input):
        op = TorchSiLU()
        op.set_config("inplace", False)
        return op

    def leaky_relu_(self, input, negative_slope=0.01):
        op = TorchLeakyReLU()
        op.set_config("negative_slope", negative_slope)
        op.set_config("inplace", True)
        return op

    def leaky_relu(self, input, negative_slope=0.01):
        op = TorchLeakyReLU()
        op.set_config("negative_slope", negative_slope)
        op.set_config("inplace", False)
        return op

    def gelu(self, input, approximate=None):
        op = TorchGELU()
        op.set_config("approximate", f"'{approximate}'")
        return op

    def mish_(self, input):
        op = TorchMish()
        op.set_config("inplace", True)
        return op

    def mish(self, input):
        op = TorchMish()
        op.set_config("inplace", False)
        return op

    def prelu(self, input, weight):
        op = TorchPReLU()
        op.set_config("num_parameters", weight.shape[0])
        op.set_config("init", 0.25)
        op.set_param(op.ParamName.WEIGHT, weight)
        return op

    def sqrt(self, input):
        op = TorchUnaryOp(OP.SQRT, "sqrt")
        op.set_config("input", input)
        return op

    def sqrt_(self, input):
        op = TorchUnaryOp(OP.SQRT, "sqrt")
        op.set_config("input", input)
        return op

    def add(self, *args):
        supported_schemas = [
            "aten::add(Tensor self, Tensor other, Scalar alpha=1) -> Tensor",
            "aten::add(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
            "aten::add(Tensor self, Tensor other, Scalar alpha=1, Tensor out) -> Tensor",
            "aten::add_(Tensor self, Tensor other, Scalar alpha=1) -> Tensor",
            "aten::add_(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
            "aten::add_(Tensor self, Tensor other, Scalar alpha=1, Tensor out) -> Tensor",
        ]

        schema_handler = SchemaHelper(self.cur_node.schema)
        if schema_handler.toString() in supported_schemas:
            op = TorchAdd()
            op.set_config("input", args[0])
            op.set_config("other", args[1])
            if args[2] is not None:
                op.set_config("alpha", args[2])
            return op
        else:
            return self.default(self.cur_node, "aten::add", *args)

    def size(self, input, dim=None):
        op = TorchSize()
        if dim is not None:
            op.set_config("dim", dim)
        return op

    def view(self, input, shape):
        op = TorchView()
        op.set_config("input", input)
        op.set_config("shape", shape)
        return op

    def reshape(self, input, shape):
        return self.view(input, shape)

    def dropout(self, input, p, train):
        op = TorchDropout()
        if train:
            op.set_config("p", p)
        op.set_config("inplace", True)
        return op

    def dropout_(self, input, p, train):
        return self.dropout(input, p, train)

    def cat(self, tensors, dim):
        op = TorchCat()
        op.set_config("dim", dim)
        op.set_config("tensors", tensors)
        return op

    def mean(self, *args):
        """
        unsupport [
         'aten::mean(Tensor self, int? dtype) -> Tensor',
         'aten::mean(Tensor self, str[] dim, bool keepdim=False, int? dtype) -> Tensor',
         'aten::mean(Tensor self, str[] dim, bool keepdim=False, int? dtype, Tensor out) -> Tensor',
        ]
        """
        schema_handler = SchemaHelper(self.cur_node.schema)
        support_schemas = [
            "aten::mean(Tensor self, int[] dim, bool keepdim=False, int? dtype) -> Tensor",
            "aten::mean(Tensor self, int[] dim, bool keepdim=False, int? dtype, Tensor out) -> Tensor",
            # modern `aten::mean.dim` overload: dim is the nullable `int[]?` form
            # (Tensor.mean([2,3]) for global pooling). Without this the schema match
            # fails and a valid reduction falls back to a raw `aten::mean` op.
            "aten::mean(Tensor self, int[]? dim, bool keepdim=False, int? dtype) -> Tensor",
            "aten::mean(Tensor self, int[]? dim, bool keepdim=False, int? dtype, Tensor out) -> Tensor",
        ]

        if schema_handler.toString() not in support_schemas:
            return self.default(self.cur_node, "aten::mean", *args)

        op = TorchPermuteInvarOp(OP.MEAN, "mean")
        op.set_config("input", args[0])
        op.set_config("dim", args[1])
        op.set_config("keepdim", args[2])

        return op

    def relu6(self, input, inplace=False):
        op = TorchUnaryOp(OP.RELU6, "ReLU6")
        op.set_config("input", input)
        op.set_config("inplace", inplace)
        return op

    def relu6_(self, input, inplace=True):
        return self.relu6(input, True)

    def hardtanh(self, input, min_val, max_val):
        if min_val == 0.0 and max_val == 6.0:
            return self.relu6(input, False)
        else:
            op = TorchHardTanh()
            op.set_config("min_val", min_val)
            op.set_config("max_val", max_val)
            op.set_config("inplace", False)
            return op

    def hardtanh_(self, input, min_val, max_val):
        if min_val == 0.0 and max_val == 6.0:
            return self.relu6(input, True)
        else:
            op = TorchHardTanh()
            op.set_config("min_val", min_val)
            op.set_config("max_val", max_val)
            op.set_config("inplace", True)
            return op

    def transpose(self, input, dim0, dim1):
        op = TorchTranspose()
        op.set_config("input", input)
        op.set_config("dim0", dim0)
        op.set_config("dim1", dim1)
        op.set_attr(op.AttrName.ORDER, [dim0, dim1])
        return op

    def contiguous(self, *args):
        op = TorchContiguous()
        return op

    def ConstantChunk(self, input, chunks, dim):
        return self.chunk(input, chunks, dim)

    def chunk(self, input, chunks, dim):
        op = TorchChunk()
        op.set_config("input", input)
        op.set_config("chunks", chunks)
        op.set_config("dim", dim)
        return op

    def _interpolate(
        self,
        input,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=None,
        recompute_scale_factor=None,
    ):

        if input._ndim == 4:
            if mode == "nearest":
                op = TorchInterpolate()
            elif mode == "bilinear":
                op = TorchResizeLinear()
        elif input._ndim == 5:
            if mode == "trilinear":
                op = TorchResizeTrilinear()
            elif mode == "nearest":
                op = TorchBaseOperation(OP.RESIZE_NEAREST_3D, "interpolate")
        else:
            op = TorchBaseOperation(OP.INTERPOLATE, "interpolate")

        op.set_config("input", input)
        op.set_config("mode", f"'{mode}'")
        if size is not None:
            op.set_config("size", size)
        if align_corners is not None:
            op.set_config("align_corners", bool(align_corners))
        if size is None and scale_factor is not None:
            op.set_config("scale_factor", scale_factor)

        if recompute_scale_factor is not None:
            op.set_config("recompute_scale_factor", bool(recompute_scale_factor))
        return op

    def upsample_bilinear2d(
        self, input, tensor_list, align_corners, scale_h=None, scale_w=None
    ):
        input._ndim = input.ndim if input.ndim else 4
        if scale_w is not None and scale_h != scale_w:
            scale = [scale_h, scale_w]
        else:
            scale = scale_h
        return self._interpolate(
            input,
            size=tensor_list,
            scale_factor=scale,
            mode="bilinear",
            align_corners=align_corners,
        )

    def upsample_nearest2d(self, input, tensor_list, scale_h=None, scale_w=None):
        input._ndim = input.ndim if input.ndim else 4
        if scale_w is not None and scale_h != scale_w:
            scale = [scale_h, scale_w]
        else:
            scale = scale_h
        return self._interpolate(input, size=tensor_list, scale_factor=scale)

    def upsample_trilinear3d(
        self,
        input,
        tensor_list,
        align_corners,
        scale_factor_d=None,
        scale_factor_h=None,
        scale_factor_w=None,
    ):
        input._ndim = input.ndim if input.ndim else 5
        if scale_factor_h is not None and scale_factor_w is not None:
            scale = [scale_factor_d, scale_factor_h, scale_factor_w]
        else:
            scale = scale_factor_d
        return self._interpolate(
            input,
            size=tensor_list,
            scale_factor=scale,
            mode="trilinear",
            align_corners=align_corners,
        )

    def upsample_nearest3d(self, input, output_size, scale_factor=None):
        input._ndim = input.ndim if input.ndim else 5
        return self._interpolate(input, size=output_size, scale_factor=scale_factor)

    def NumToTensor(self, input, *args):
        "The device of NumToTensor is cpu"
        op = TorchTensor()
        op.set_config("data", input)
        op.set_config("dtype", input.dtype)
        op.set_config("device", f"'cpu'")
        return op

    def Constant(self, tensor, *args):

        # batter method is spliting op into contant and reshape in raw graph optim
        if (
            isinstance(tensor.data, np.ndarray)
            and tensor.shape != None
            and 0 in tensor.shape[:-1]
        ):
            op = TorchBaseOperation(
                OP.CONSTANT_WITH_RESHAPE, "ConstantWithReshape"
            )
            op.set_config("data_shape", tensor.data.shape)
        else:
            op = TorchConst()
        if not tensor.is_complete_tensor():
            if isinstance(tensor.data, np.ndarray):
                tensor.from_ndarray(tensor.data)
            else:
                dtype = convert_dtype_between_np_and_pytorch(tensor.dtype)
                np_data = np.array(tensor.data, dtype=dtype)
                tensor.from_ndarray(np_data)
        else:
            tensor.from_ndarray(tensor.data)

        op.set_config("data", tensor.data.tolist())
        op.set_config("dtype", convert_dtype_between_np_and_pytorch(tensor.dtype))
        device = (
            self._device_type
            if tensor.device.device_type == TorchDeviceType.UNKOWN
            else tensor.device.device_type
        )
        op.set_config("device", f"'{device}'")

        return op

    def mul(self, *args):
        supported_schemas = [
            "aten::mul(Tensor self, Tensor other) -> Tensor",
            "aten::mul(Tensor self, Scalar other) -> Tensor",
            "aten::mul(Tensor self, Tensor other, Tensor out) -> Tensor",
            "aten::mul_(Tensor self, Tensor other) -> Tensor",
            "aten::mul_(Tensor self, Scalar other) -> Tensor",
            "aten::mul_(Tensor self, Tensor other, Tensor out) -> Tensor",
        ]

        schema_handler = SchemaHelper(self.cur_node.schema)
        if schema_handler.toString() in supported_schemas:
            op = TorchBinaryOp(OP.MULTIPLY, "mul")
            op.set_config("input", args[0])
            op.set_config("other", args[1])
            return op
        else:
            return self.default(self.cur_node, "aten::mul", *args)

    def _to_device(
        self, input, device, dtype, non_blocking=False, copy=False, memory_format=None
    ):
        op = TorchUnaryOp(OP.CAST, "to")
        op.set_config("input", input)
        if isinstance(device, str):
            op.set_config("device", f"'{device}'")
        else:
            op.set_config("device", device)
        op.set_config("dtype", scalar_type_to_pytorch_type[dtype])
        op.set_config("non_blocking", bool(non_blocking))
        op.set_config("copy", bool(copy))
        return op

    def _to_dtype(
        self, input, dtype, non_blocking=False, copy=False, memory_format=None
    ):
        op = TorchUnaryOp(OP.CAST, "to")
        op.set_config("input", input)
        op.set_config("dtype", scalar_type_to_pytorch_type[dtype])
        op.set_config("non_blocking", bool(non_blocking))
        op.set_config("copy", bool(copy))
        return op

    def _to_other(
        self, input, other, non_blocking=False, copy=False, memory_format=None
    ):
        op = TorchUnaryOp(OP.CAST, "to")
        op.set_config("input", input)
        op.set_config("other", other)
        op.set_config("dtype", scalar_type_to_pytorch_type[dtype])
        op.set_config("non_blocking", bool(non_blocking))
        op.set_config("copy", bool(copy))
        return op

    def _to_dtype_layout(
        self,
        input,
        dtype=None,
        layout=None,
        device=None,
        pin_memory=None,
        non_blocking=False,
        copy=False,
        memory_format=None,
    ):
        op = TorchUnaryOp(OP.CAST, "to")
        op.set_config("input", input)
        if dtype is not None:
            op.set_config("dtype", scalar_type_to_pytorch_type[dtype])
        if device is not None:
            if isinstance(device, str):
                op.set_config("device", f"'{device}'")
            else:
                op.set_config("device", device)
        op.set_config("non_blocking", bool(non_blocking))
        op.set_config("copy", bool(copy))
        return op

    def to(self, input, *args):
        if len(args) == 7:
            return self._to_dtype_layout(input, *args)
        if len(args) == 5:
            return self._to_device(input, *args)
        elif len(args) == 4:
            if isinstance(args[0], Tensor):
                return self._to_other(input, *args)
            else:
                return self._to_dtype(input, *args)
        else:
            raise RuntimeError(
                f"'aten::to' schema of {self.cur_node.name} is unkonwn for parser."
            )

    def floor(self, input):
        # op = TorchFloor()
        op = TorchUnaryOp(OP.FLOOR, "floor")
        op.set_config("input", input)
        return op

    def Int(self, *args):
        op = TorchUnaryOp(OP.INT, "int", force_to_primitive=True)
        op.set_config("input", args[0])
        return op

    def permute(self, input, dims):
        op = TorchPermute()
        op.set_config("dims", dims)
        op.set_config("input", input)
        return op

    def floor_divide(self, input, other):
        op = TorchFloorDiv()
        op.set_config("input", input)
        op.set_config("other", other)
        return op

    def div(self, *args):
        supported_schemas = [
            "aten::div(Tensor self, Tensor other) -> Tensor",
            "aten::div(Tensor self, Scalar other) -> Tensor",
            # "aten::div(Tensor self, Tensor other, str? rounding_mode) -> Tensor",
            # "aten::div(Tensor self, Scalar other, str? rounding_mode) -> Tensor",
            "aten::div(Tensor self, Tensor other, Tensor out) -> Tensor",
            # "aten::div(Tensor self, Tensor other, str? rounding_mode, Tensor out) -> Tensor"
        ]
        # schemas = torch._C._jit_get_schemas_for_operator("aten::div")
        # for schema in schemas:
        #   schema_handler = SchemaHelper(schema)
        #   print(schema_handler.toString())

        schema_handler = SchemaHelper(self.cur_node.schema)
        if schema_handler.toString() in supported_schemas:
            op = TorchBinaryOp(OP.DIV, "div")
            op.set_config("input", args[0])
            op.set_config("other", args[1])
            return op
        else:
            return self.default(self.cur_node, "aten::div", *args)

    def softmax(self, input, dim, dtype=None):
        op = TorchSoftmax()
        op.set_config("dim", dim)
        return op

    def sigmoid(self, input):
        op = TorchSigmoid()
        return op

    def hardswish(self, input):
        op = TorchUnaryOp(OP.HSWISH, "Hardswish")
        op.set_config("inplace", False)
        op.set_config("input", input)
        return op

    def hardswish_(self, input):
        op = TorchUnaryOp(OP.HSWISH, "Hardswish")
        op.set_config("inplace", True)
        op.set_config("input", input)
        return op

    def hardsigmoid(self, input):
        op = TorchUnaryOp(OP.HSIGMOID, "Hardsigmoid")
        op.set_config("inplace", False)
        op.set_config("input", input)
        return op

    def hardsigmoid_(self, input):
        op = TorchUnaryOp(OP.HSIGMOID, "Hardsigmoid")
        op.set_config("inplace", True)
        op.set_config("input", input)
        return op

    def strided_slice(self, input, dim, start, end, step):

        op = TorchSlice()
        op.set_config("input", input)
        op.set_config("dim", dim)
        new_start = []
        new_end = []
        for x in start:
            if x is None:
                x = 0
            new_start.append(x)
        for x in end:
            if x is None:
                x = sys.maxsize
            new_end.append(x)

        op.set_config("start", new_start)
        op.set_config("end", new_end)
        op.set_config("step", step)
        return op

    def sub(self, *args):
        op = TorchBinaryOp(OP.SUB, "sub", force_to_primitive=True)
        op.set_config("input", args[0])
        op.set_config("other", args[1])
        if args[2] is not None:
            op.set_config("alpha", args[2])
        return op

    def rsub(self, *args):
        op = TorchBinaryOp(OP.RSUB, "rsub", force_to_primitive=True)
        op.set_config("input", args[0])
        op.set_config("other", args[1])
        if args[2] is not None:
            op.set_config("alpha", args[2])
        return op

    def exp(self, input, *args):
        op = TorchExp()
        op.set_config("input", input)
        return op

    def detach(self, input, *args):
        op = TorchDetach()
        op.set_config("input", input)
        return op

    def select(self, input, dim, index):
        op = TorchSelect()
        op.set_config("input", input)
        op.set_config("dim", dim)
        op.set_config("index", index)
        op.set_attr_by_name("dim", dim)
        op.set_attr_by_name("index", index)
        return op

    def repeat(self, input, sizes):
        op = TorchRepeat()
        op.set_config("repeats", sizes)
        return op

    def copy_(self, input, src, non_blocking):
        op = TorchInplaceCopy()
        op.set_config("other", src)
        op.set_config("non_blocking", bool(non_blocking))
        return op

    def expand(self, input, size, *args):
        op = TorchBaseOperation(OP.EXPAND, "expand", force_to_primitive=True)
        op.set_config("input", input)
        op.set_config("size", size)
        return op

    def t(self, input, *args):
        return self.transpose(input, 0, 1)

    def unsqueeze(self, input, dim):
        op = TorchUnsqueeze()
        op.set_config("input", input)
        op.set_config("dim", dim)
        return op

    def lstm(self, *args):
        return self.default(self.cur_node, "aten::lstm", *args)
        if isinstance(args[3], list):
            raise NotImplementedError("Unimplement packed lstm")
        else:
            return self._lstm_full(*args)

    def _lstm_full(
        self,
        input,
        hidden,
        weight_v,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    ):
        op = TorchLstm()
        op.set_config("bias", bool(has_biases))
        op.set_config("bidirectional", bool(bidirectional))
        op.set_config("input_size", input.shape[-1])
        op.set_config("hidden_size", hidden[0].shape[-1])
        op.set_config("batch_first", bool(batch_first))
        op.set_config("num_layers", num_layers)
        op.set_config("dropout", dropout)
        weight_ih = []
        weight_hh = []
        op.set_param(op.ParamName.WEIGHT_IH, weight_ih)
        op.set_param(op.ParamName.WEIGHT_HH, weight_hh)
        layer_offset = 2

        if bool(has_biases):
            bias = []
            op.set_param(op.ParamName.BIAS, bias)
            layer_offset += 2

        if bool(bidirectional):
            weight_ih_reverse = []
            weight_hh_reverse = []
            op.set_param(op.ParamName.WEIGHT_IH_REVERSE, weight_ih_reverse)
            op.set_param(op.ParamName.WEIGHT_HH_REVERSE, weight_hh_reverse)
            layer_offset += 2
            if bool(has_biases):
                bias_reverse = []
                op.set_param(op.ParamName.BIAS_REVERSE, bias_reverse)
                layer_offset += 2

        for layer_no in range(num_layers):
            i = 0
            for param_name, param in op.params.items():
                step = layer_no * layer_offset
                if param_name in [op.ParamName.BIAS, op.ParamName.BIAS_REVERSE]:
                    param.extend([weight_v[i + step], weight_v[i + 1 + step]])
                    i += 2
                else:
                    param.append(weight_v[i + step])
                    i += 1

        return op

    def gru(self, *args):
        # lstm 과 동일하게 generic(default) 경로 사용 — params(weight_ih_l0 등)가 raw 텐서로
        # 유지돼 GGUF/eager 바인딩이 LSTM 과 균일해진다. (_gru_full 구조화 경로는 params 가
        # GGUF 로 연결되지 않아 미사용; 아래는 dead-code 로 보존.)
        return self.default(self.cur_node, "aten::gru", *args)
        if isinstance(args[3], list):
            raise NotImplementedError("Unimplement packed gri")
        else:
            return self._gru_full(*args)

    def _gru_full(
        self,
        input,
        hidden,
        weight_v,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    ):
        op = TorchGru()
        op.set_config("bias", bool(has_biases))
        op.set_config("bidirectional", bool(bidirectional))
        op.set_config("input_size", input.shape[-1])
        op.set_config("hidden_size", hidden.shape[-1])
        op.set_config("batch_first", bool(batch_first))
        op.set_config("num_layers", num_layers)
        op.set_config("dropout", dropout)

        weight_ih = []
        weight_hh = []
        op.set_param(op.ParamName.WEIGHT_IH, weight_ih)
        op.set_param(op.ParamName.WEIGHT_HH, weight_hh)
        layer_offset = 2

        if bool(has_biases):
            bias_ih = []
            bias_hh = []
            op.set_param(op.ParamName.BIAS_IH, bias_ih)
            op.set_param(op.ParamName.BIAS_HH, bias_hh)
            layer_offset += 2

        if bool(bidirectional):
            weight_ih_reverse = []
            weight_hh_reverse = []
            op.set_param(op.ParamName.WEIGHT_IH_REVERSE, weight_ih_reverse)
            op.set_param(op.ParamName.WEIGHT_HH_REVERSE, weight_hh_reverse)
            layer_offset += 2
            if bool(has_biases):
                bias_ih_reverse = []
                bias_hh_reverse = []
                op.set_param(op.ParamName.BIAS_IH_REVERSE, bias_ih_reverse)
                op.set_param(op.ParamName.BIAS_HH_REVERSE, bias_hh_reverse)
                layer_offset += 2

        for layer_no in range(num_layers):
            i = 0
            for _, param in op.params.items():
                step = layer_no * layer_offset
                param.append(weight_v[i + step])
                i += 1

        return op

    def zeros(self, sizes, dtype, layout, device, pin_memory=False):
        op = TorchZeros()
        op.set_config("size", sizes)
        if isinstance(dtype, Tensor):
            op.set_config("dtype", dtype)
        else:
            dtype = 6 if dtype is None else dtype
            op.set_config("dtype", scalar_type_to_pytorch_type[dtype])
        if isinstance(device, Tensor):
            op.set_config("device", device)
        elif isinstance(device, str):
            op.set_config("device", f"'{device}'")
        else:
            op.set_config("device", f"'{self._device_type}'")
        return op

    def pad(self, *args):
        "aten::pad(Tensor self, int[] pad, str mode='constant', float? value) -> Tensor"
        if args[2] not in ["constant", "reflect", "replicate"] or args[3] in [
            float("inf"),
            float("-inf"),
        ]:
            return self.default(self.cur_node, "aten::pad", *args)
        op = TorchPadNd()
        op.set_config("input", args[0])
        op.set_config("pad", args[1])
        op.set_config("mode", f"'{args[2]}'")
        value = 0.0 if args[3] is None else args[3]
        op.set_config("value", value)
        return op

    def constant_pad_nd(self, input, pad, value):
        op = TorchPad()
        op.set_config("input", input)
        op.set_config("pad", pad)
        op.set_config("value", value)
        op.set_config("mode", "'constant'")
        return op

    def replication_pad2d(self, input, pad):
        for pad_len in pad:
            if not isinstance(pad_len, Tensor):
                if pad_len > 1:
                    raise ValueError(
                        "Only support pad 1 row or 1 col in replication mode"
                    )

        op = TorchPad()
        op.set_config("input", input)
        op.set_config("pad", pad)
        op.set_config("value", 0.0)
        op.set_config("mode", "'replicate'")
        return op

    def matmul(self, input, other, *args):
        if "weight" in other.name:
            op = TorchLinear()
            op.set_param(op.ParamName.WEIGHTS, other)
            op.set_config("bias", False)
            op.set_config("out_features", other.shape[0])
            op.set_config("in_features", other.shape[1])
            op.set_attr(op.AttrName.TRANSPOSE_B, False)
            return op
        else:
            op = TorchMatmul()
            op.set_config("input", input)
            op.set_config("other", other)
            return op

    def clamp(self, input, min=None, max=None):
        op = TorchClamp()
        op.set_config("input", input)
        if min is not None:
            op.set_config("min", min)
        if max is not None:
            op.set_config("max", max)
        return op

    def clamp_min(self, input, min):
        op = TorchClamp()
        op.set_config("input", input)
        op.set_config("min", min)
        # op.set_attr_by_name("min", min)
        return op

    def tanh(self, input, *args):
        op = TorchTanh()
        return op

    def slice_tensor_inplace_copy(self, input, src, non_blocking, dim, index):
        op = TorchBaseOperation(
            OP.SLICE_TENSOR_INPLACE_COPY,
            "slice_tensor_inplace_copy",
            force_to_primitive=True,
        )
        op.set_config("input", input)
        op.set_config("source", src)
        op.set_config("dim", dim)
        op.set_config("index", index)
        return op

    def norm(self, input, p, dim, keepdim):
        if str(p) in ["-inf", "inf"]:
            p = f"float('{p}')"
        op = TorchBaseOperation(OP.NORM, "norm")
        op.set_config("input", input)
        op.set_config("dim", dim)
        op.set_config("keepdim", bool(keepdim))
        op.set_config("p", p)
        op.set_attr_by_name("dim", dim)
        op.set_attr_by_name("keepdim", bool(keepdim))
        op.set_attr_by_name("p", p)
        return op

    def expand_as(self, input, other):
        op = TorchBaseOperation(
            OP.EXPAND_AS, "expand_as", force_to_primitive=True
        )
        op.set_config("input", input)
        op.set_config("other", other)
        return op

    def max(self, input, dim=None, keepdim=None):
        op = TorchPermuteInvarOp(OP.MAX, "max")
        # op = TorchBaseOperation(OP.MAX, "max")
        op.set_config("input", input)
        if dim is not None:
            op.set_config("dim", dim)
        if keepdim is not None:
            op.set_config("keepdim", bool(keepdim))
        return op

    def min(self, input, dim=None, keepdim=None):
        op = TorchPermuteInvarOp(OP.MIN, "min")
        op.set_config("input", input)
        if dim is not None:
            op.set_config("dim", dim)
        if keepdim is not None:
            op.set_config("keepdim", bool(keepdim))
        return op

    def squeeze(self, input, dim=None):
        # op = TorchBaseOperation(OP.SQUEEZE, "squeeze")
        op = TorchSqueeze()
        op.set_config("input", input)
        if dim is not None:
            # if dim == -1:
            #   dim = len(input.shape) - 1
            op.set_config("dim", dim)
        return op

    def quantile(self, *args):
        support_schemas = [
            "aten::quantile(Tensor self, Tensor q, int? dim, bool keepdim=False, str interpolation='linear') -> Tensor",
            "aten::quantile(Tensor self, float q, int? dim, bool keepdim=False, str interpolation='linear') -> Tensor",
            "aten::quantile(Tensor self, Tensor q, int? dim, bool keepdim=False, str interpolation='linear', Tensor out) -> Tensor",
            "aten::quantile(Tensor self, float q, int? dim, bool keepdim=False, str interpolation='linear', Tensor out) -> Tensor",
        ]

        schema_handler = SchemaHelper(self.cur_node.schema)
        op = TorchBaseOperation(OP.QUANTILE, "quantile")
        op.set_config("input", args[0])
        q_tensor = args[1]
        if isinstance(args[1], Tensor):
            q_tensor = args[1]
        else:
            q_tensor = torch.tensor(args[1])

        op.set_config("q", q_tensor)
        op.set_config("dim", args[2])
        op.set_config("keepdim", bool(args[3]))
        op.set_config("interpolation", f"'{args[4]}'")
        return op

    def nanquantile(self, *args):
        support_schemas = [
            "aten::nanquantile(Tensor self, Tensor q, int? dim, bool keepdim=False, str interpolation='linear') -> Tensor",
            "aten::nanquantile(Tensor self, float q, int? dim, bool keepdim=False, str interpolation='linear') -> Tensor",
            "aten::nanquantile(Tensor self, Tensor q, int? dim, bool keepdim=False, str interpolation='linear', Tensor out) -> Tensor",
            "aten::nanquantile(Tensor self, float q, int? dim, bool keepdim=False, str interpolation='linear', Tensor out) -> Tensor",
        ]

        schema_handler = SchemaHelper(self.cur_node.schema)
        op = TorchBaseOperation(OP.QUANTILE, "quantile")
        op.set_config("input", args[0])
        q_tensor = args[1]
        if isinstance(args[1], Tensor):
            q_tensor = args[1]
        else:
            q_tensor = torch.tensor(args[1])

        op.set_config("q", q_tensor)
        op.set_config("dim", args[2])
        op.set_config("keepdim", bool(args[3]))
        op.set_config("interpolation", f"'{args[4]}'")
        return op

    def _compare(self, op_type, torch_op, scalar_op_type, input, other):
        """비교 연산 공용: 한쪽이라도 실제 텐서면 텐서 op, 아니면 스칼라(primitive) op."""
        if (isinstance(input, Tensor) and input.is_real_tensor()) or (
            isinstance(other, Tensor) and other.is_real_tensor()
        ):
            op = TorchBaseOperation(op_type, torch_op)
        else:
            op = TorchBaseOperation(
                scalar_op_type, scalar_op_type, force_to_primitive=False
            )
        op.set_config("input", input)
        op.set_config("other", other)
        return op

    def eq(self, input, other):
        return self._compare(OP.EQUAL, "eq", OP.SCALAR_EQUAL, input, other)

    def ne(self, input, other):
        return self._compare(OP.NOT_EQUAL, "ne", OP.SCALAR_NOT_EQUAL, input, other)

    def gt(self, input, other):
        return self._compare(OP.GREATER, "gt", OP.SCALAR_GREATER_THAN, input, other)

    def logical_or(self, input, other):
        # aten::__or__. bool 마스크 합집합만 처리 — 정수 비트연산이면 기본 경로로 넘긴다.
        if not (isinstance(input, Tensor) and input.is_real_tensor()):
            return self.default(self.cur_node, "aten::__or__", input, other)
        op = TorchBaseOperation(OP.LOGICAL_OR, "logical_or")
        op.set_config("input", input)
        op.set_config("other", other)
        return op

    def where(self, *args):
        # 1항 where(cond) 는 인덱스를 돌려주는 다른 연산 → 기본 경로.
        if len(args) < 3:
            return self.default(self.cur_node, "aten::where", *args)
        op = TorchBaseOperation(OP.WHERE, "where")
        op.set_config("condition", args[0])
        op.set_config("input", args[1])
        op.set_config("other", args[2])
        return op

    def deform_conv2d(self, input, weight, offset, mask, bias, stride_h, stride_w,
                      pad_h, pad_w, dilation_h, dilation_w, groups, offset_groups,
                      use_mask):
        """torchvision::deform_conv2d (DCN v1/v2). 인자 순서는 torchvision 스키마 그대로.

        weight/bias 는 op param 으로 실어 GGUF 에 나가게 한다(conv2d 와 동일 경로).
        """
        # torchvision.ops 소속이라 torch/torch.nn 네임스페이스에 없다 → CUSTOM_FUNCTION.
        op = TorchBaseOperation(OP.DEFORM_CONV2D, "deform_conv2d",
                                class_type=TorchOpClassType.CUSTOM_FUNCTION)
        op.set_config("input", input)
        op.set_config("offset", offset)
        op.set_config("mask", mask)
        op.set_config("stride", [int(stride_h), int(stride_w)])
        op.set_config("padding", [int(pad_h), int(pad_w)])
        op.set_config("dilation", [int(dilation_h), int(dilation_w)])
        op.set_config("groups", int(groups))
        op.set_config("offset_groups", int(offset_groups))
        op.set_config("use_mask", bool(use_mask))
        op.set_config("bias", bias is not None)
        op.set_param("weight", weight)
        if bias is not None:
            op.set_param("bias", bias)
        return op

    def masked_fill(self, input, mask, value):
        op = TorchBaseOperation(OP.MASKED_FILL, "masked_fill")
        op.set_config("input", input)
        op.set_config("mask", mask)
        op.set_config("value", value)
        return op

    def index(self, input, index):
        op = TorchBaseOperation(OP.INDEX, "index", force_to_primitive=True)
        op.set_config("input", input)
        op.set_config("index", index)
        return op

    def index_put_(self, input, indices, value, accumulate, *args):
        op = TorchBaseOperation(
            OP.INDEX_INPUT_INPLACE, "index_put_", force_to_primitive=True
        )
        op.set_config("input", input)
        indices = ["slice(None)" if idx is None else idx for idx in indices]
        op.set_config("indices", indices)
        op.set_config("values", value)
        op.set_config("accumulate", bool(accumulate))
        return op

    def stack(self, input, dim):
        op = TorchBaseOperation(OP.STACK, "stack")
        op.set_config("tensors", input)
        op.set_config("dim", dim)
        return op

    def bmm(self, input, other):
        op = TorchMatmul()
        op.set_config("input", input)
        op.set_config("other", other)
        return op

    def embedding_bag(
        self,
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights=None,
    ):

        if weight.node != None:
            raise ("weight or bias is not a constant param!")

        mode_map = {0: "sum", 1: "mean", 2: "max"}

        op = TorchEmbeddingBag()
        op.set_param(op.ParamName.WEIGHT, weight)

        op.set_config("num_embeddings", weight.shape[0])
        op.set_config("embedding_dim", weight.shape[1])
        op.set_config("scale_grad_by_freq", bool(scale_grad_by_freq))
        op.set_config("mode", f"'{mode_map[mode]}'")
        op.set_config("sparse", bool(sparse))
        return op

    def feature_dropout(self, *args):
        return self.dropout(*args)

    def QuantStubF(self, input, *args):
        op = TorchUnaryOp(OP.QUANT_STUB, "quant_input", force_to_primitive=True)
        op.set_config("input", input)
        return op

    def DeQuantStubF(self, input, *args):
        op = TorchUnaryOp(
            OP.DEQUANT_STUB, "dequant_output", force_to_primitive=True
        )
        op.set_config("input", input)
        return op

    def RelukF(self, input, inplace, channel_max=6.0):
        op = TorchBaseOperation(OP.RELUK, "Reluk", force_to_primitive=True)
        op.set_config("input", input)
        op.set_config("channel_max", channel_max)
        return op

    def ChannelScaleF(self, input, inplace, channel_scale=1.0):
        op = TorchBaseOperation(
            OP.CHANNEL_SCALE, "Channel_Scale", force_to_primitive=True
        )
        op.set_config("input", input)
        op.set_config("channel_scale", channel_scale)
        return op

    def strided_slice_inplace_copy(self, des, dim, start, end, step, src):
        op = TorchBaseOperation(
            OP.STRIDED_SLICE_INPLACE_COPY,
            OP.STRIDED_SLICE_INPLACE_COPY,
            force_to_primitive=False,
        )

        # begin = [0] * len(dim)
        # last = [CONSTANT.INT_MAX] * len(dim)
        # stride = [1] * len(dim)
        # for i, pos in enumerate(dim):
        #  begin[pos] = start[i]
        #  stride[pos] = step[i]
        #  last[pos] = end[i]

        op.set_config("destination", des)
        op.set_config("source", src)
        op.set_config("dim", dim)
        op.set_config("start", start)
        op.set_config("end", end)
        op.set_config("step", step)
        return op

    def pixel_shuffle(self, input, upscale_factor):
        op = TorchPixelShuffle()
        op.set_config("upscale_factor", upscale_factor)
        op.set_attr(op.AttrName.UPSCALE, True)
        return op

    def pixel_unshuffle(self, input, downscale_factor):
        op = TorchPixelUnshuffle()
        op.set_config("downscale_factor", downscale_factor)
        op.set_attr(op.AttrName.UPSCALE, False)
        return op

    def Loop(self, max_trip_count, initial_condition, *args):
        op = TorchBaseOperation(OP.LOOP, OP.LOOP, force_to_primitive=False)
        if isinstance(max_trip_count, int) and max_trip_count == sys.maxsize:
            op.set_config("is_while_loop", True)
        else:
            op.set_config("is_while_loop", False)

        op.set_config("max_trip_count", max_trip_count)
        if isinstance(initial_condition, int):
            op.set_config("initial_condition", bool(initial_condition))
        else:
            op.set_config("initial_condition", initial_condition)

        initial_vars = [arg for arg in args]
        op.set_config("initial_loop_vars", initial_vars)

        return op

    def mm(self, input, other):
        op = TorchMatmul()
        op.set_config("input", input)
        op.set_config("other", other)
        return op

    def layer_norm(self, input, normalized_shape, weight, bias, eps, cudnn_enabled):
        if (weight and weight.node != None) or (bias and bias.node != None):
            raise ("weight or bias is not a constant param!")
        op = TorchLayerNorm()
        op.set_param(op.ParamName.GAMMA, weight)
        op.set_param(op.ParamName.BETA, bias)
        op.set_config("normalized_shape", normalized_shape)
        op.set_config("eps", eps)
        if weight is not None:
            op.set_param(op.ParamName.GAMMA, weight)
        if bias is not None:
            op.set_param(op.ParamName.BETA, bias)
        if weight is not None or bias is not None:
            op.set_config("elementwise_affine", True)
        else:
            op.set_config("elementwise_affine", False)
        return op

    def new_zeros(self, input, *args):
        return self.zeros(*args)

    def neg(self, *args):
        supported_schemas = [
            "aten::neg(Tensor self) -> Tensor",
            "aten::neg(Tensor self, Tensor out) -> Tensor",
        ]
        schema_handler = SchemaHelper(self.cur_node.schema)
        if schema_handler.toString() in supported_schemas:
            op = TorchUnaryOp(OP.NEG, "neg")
            op.set_config("input", args[0])
            return op
        else:
            return self.default(self.cur_node, "aten::neg", *args)

    def grid_sampler(self, input, grid, mode, padding_mode, align_corners):
        mode_map = {
            0: "bilinear",
            1: "nearest",
        }
        padding_mod_map = {0: "zeros", 1: "border", 2: "reflection"}

        op = TorchBaseOperation(OP.GRID_SAMPLE, "grid_sample")
        op.set_config("input", input)
        op.set_config("grid", grid)
        op.set_config("mode", f"'{mode_map[mode]}'")
        op.set_config("padding_mode", f"'{padding_mod_map[padding_mode]}'")
        op.set_config("align_corners", bool(align_corners))
        return op

    def sum(self, input, dim=None, keepdim=False, dtype=None):
        op = TorchPermuteInvarOp(OP.SUM, "sum")
        op.set_config("input", input)
        if dim is not None:
            op.set_config("dim", dim)
            op.set_config("keepdim", bool(keepdim))
        return op

    def tuple_unpack(self, input):
        op = TorchBaseOperation(
            OP.TUPLE_UNPACK, "TupleUnpack", force_to_primitive=True
        )
        # op = TorchBaseOperation(OP.TUPLE_UNPACK, OP.TUPLE_UNPACK)
        op.set_config("input", input)
        return op

    def derive_loop_index(self, index, start, step):
        op = TorchBaseOperation(
            OP.DERIVE_LOOP_INDEX,
            OP.DERIVE_LOOP_INDEX,
            force_to_primitive=False,
        )
        op.set_config("input", index)
        op.set_config("start", start)
        op.set_config("step", step)
        return op

    def cast_float(self, input, non_blocking):
        return self._to_dtype(input, 6)

    def cast_int(self, input, non_blocking):
        return self._to_dtype(input, 3)

    def Bool(self, input):
        return self._to_dtype(input, 11)

    def ceil(self, input):
        op = TorchBaseOperation(OP.CEIL, "ceil")
        op.set_config("input", input)
        return op

    def len(self, input):
        op = TorchBaseOperation(
            OP.LENGTH, OP.LENGTH, force_to_primitive=False
        )
        op.set_config("input", input)
        return op

    def lt(self, input, other):
        # TODO:
        op = TorchBaseOperation(
            OP.SCALAR_LESS_THAN,
            OP.SCALAR_LESS_THAN,
            force_to_primitive=False,
        )
        op.set_config("input", input)
        op.set_config("other", other)
        return op

    def If(self, condition):
        op = TorchBaseOperation(OP.IF, OP.IF, force_to_primitive=False)
        op.set_config("condition", condition)
        return op

    def tensor(self, data, dtype, device, required_grad=False):
        op = TorchTensor()
        op.set_config("data", data)
        if dtype is not None:
            op.set_config("dtype", scalar_type_to_pytorch_type[dtype])
        op.set_config("device", f"'{self._device_type}'")
        return op

    def embedding(self, weight, indices, padding_idx, scale_grad_freq, sparse):
        if weight.node != None:
            raise ("weight or bias is not a constant param!")
        op = TorchEmbedding()
        op.set_config("num_embeddings", weight.shape[0])
        op.set_config("embedding_dim", weight.shape[1])
        op.set_config("padding_idx", padding_idx)
        op.set_param(op.ParamName.WEIGHT, weight)
        return op

    def item(self, input):
        op = TorchBaseOperation(OP.TENSOR_TO_SCALAR, "item")
        return op

    def stft(
        self,
        input,
        n_fft,
        hop_length,
        win_length,
        window,
        normalized,
        onesided,
        return_complex,
    ):
        op = TorchBaseOperation(OP.STFT, "stft")
        op.set_config("input", input)
        op.set_config("n_fft", n_fft)
        op.set_config("hop_length", hop_length)
        op.set_config("win_length", win_length)
        op.set_config("window", window)
        op.set_config("center", False)
        op.set_config("normalized", bool(normalized))
        onesided = onesided if onesided is None else bool(onesided)
        op.set_config("onesided", onesided)
        return_complex = (
            return_complex if return_complex is None else bool(return_complex)
        )
        op.set_config("return_complex", return_complex)

        return op

    def unique_dim(
        self, input, dim, sorted=True, return_inverse=False, return_counts=False
    ):
        """
        inline ::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_dim(const at::Tensor & self, int64_t dim, bool sorted=true, bool return_inverse=false, bool return_counts=false)
        """
        op = TorchBaseOperation(OP.UNIQUE_DIM, "unique")
        op.set_config("input", input)
        op.set_config("dim", dim)
        op.set_config("sorted", bool(sorted))
        # force set return number as max return
        op.set_config("return_inverse", True)
        op.set_config("return_counts", True)
        return op

    def _unique2(self, input, sorted=True, return_inverse=False, return_counts=False):
        """
        inline ::std::tuple<at::Tensor,at::Tensor,at::Tensor> _unique2(const at::Tensor & self, bool sorted=true, bool return_inverse=false, bool return_counts=false)
        """
        op = TorchBaseOperation(OP._UNIQUE2, "unique")
        op.set_config("input", input)
        op.set_config("sorted", bool(sorted))
        # force set return number as max return
        op.set_config("return_inverse", True)
        op.set_config("return_counts", True)
        return op

    def _unique(self, input, sorted=True, return_inverse=False):
        """
        inline ::std::tuple<at::Tensor,at::Tensor> _unique(const at::Tensor & self, bool sorted=true, bool return_inverse=false)
        """
        op = TorchBaseOperation(OP._UNIQUE, "unique")
        op.set_config("input", input)
        op.set_config("sorted", bool(sorted))
        # force set return number as max return
        op.set_config("return_inverse", True)
        return op

    def auto_infer_op(self, node, op_type, args):
        if node.schema is None:
            op = TorchUnknownOperation(op_type)
            return op

        schema_handler = SchemaHelper(node.schema)
        op_caller = get_operation_caller_by_schema_name(node.schema.name)
        node2caller = GLOBAL_MAP.get_ele(KEYS.NODE_CALLER_MAP)
        if node2caller is None:
            node2caller: Dict[str, Callable] = {}
            GLOBAL_MAP.set_map(KEYS.NODE_CALLER_MAP, node2caller)

        if op_caller is None:
            op = TorchUnknownOperation(op_type)
            return op
        else:
            node2caller[node.name] = op_caller

        op = TorchAutoInferOperation(
            node.schema.name,
            node.schema.name,
            schema=node.schema,
            class_type=TorchOpClassType.AUTO_INFER_OP,
        )
        for inp, arg in zip(args, schema_handler.get_arguments()):
            arg_name = schema_handler.arg_name(arg)
            # if arg_name in ["layout", "memory_format", "pin_memory"]:
            #   continue
            config_name = arg_name
            if (
                convert_type_str(schema_handler.arg_type(arg)).replace("?", "")
                == "bool"
            ):
                inp = bool(inp) if inp is not None else inp
            if convert_type_str(schema_handler.arg_type(arg)).replace("?", "") in [
                "str",
                "Device",
            ]:
                inp = f"'{inp}'" if inp is not None and isinstance(inp, str) else inp
            if arg_name == "dtype":
                inp = scalar_type_to_pytorch_type[inp] if inp is not None else inp
            if config_name == "device" and isinstance(inp, str):
                inp = "torch.device(" + inp + ")"
            if str(inp) in ["-inf", "inf"]:
                inp = f"float('{inp}')"
            op.set_config(config_name, inp)
            if config_name == "weight":
                if isinstance(inp, Tensor):
                    inp.requires_grad = False
        return op

    def default(self, node, op_type, *args):
        if compare_torch_version(CmpFlag.GREATER, "1.4.0"):
            return self.auto_infer_op(node, op_type, args)

        if node.schema is None:
            op = TorchUnknownOperation(op_type)
            return op
        schema2torchop = GLOBAL_MAP.get_ele(KEYS.TORCH_SCHEMA_OP_TABLE)
        schema_handler = SchemaHelper(node.schema)
        torchop = schema2torchop.get(schema_handler.toString(), None)

        if torchop is None:
            op_n = self.auto_infer_op(node, op_type, args)
            if op_n is not None:
                return op_n
            op = TorchUnknownOperation(op_type)
            return

        node2caller = GLOBAL_MAP.get_ele(KEYS.NODE_CALLER_MAP)
        if node2caller is None:
            node2caller: Dict[str, Callable] = {}
            GLOBAL_MAP.set_map(KEYS.NODE_CALLER_MAP, node2caller)
        node2caller[node.name] = torchop.caller

        op = TorchBaseOperation(
            schema_handler.op_name, torchop.name, schema=node.schema
        )

        assert len(args) == len(schema_handler.get_arguments())
        if len(args) == 1:
            return op
        arg_name_convertor = {"self": "input"}
        for inp, arg in zip(args, schema_handler.get_arguments()):
            arg_name = schema_handler.arg_name(arg)
            if torchop.op_class_type == TorchOpClassType.TENSOR and arg_name == "self":
                continue
            if arg_name in ["layout", "memory_format", "pin_memory"]:
                continue
            config_name = arg_name_convertor.get(arg_name, arg_name)
            if (
                convert_type_str(schema_handler.arg_type(arg)).replace("?", "")
                == "bool"
            ):
                inp = bool(inp) if inp is not None else inp
            if convert_type_str(schema_handler.arg_type(arg)).replace("?", "") in [
                "str",
                "Device",
            ]:
                inp = f"'{inp}'" if inp is not None else inp
            if arg_name == "dtype":
                inp = scalar_type_to_pytorch_type[inp] if inp is not None else inp
            if str(inp) in ["-inf", "inf"]:
                inp = f"float('{inp}')"
            op.set_config(config_name, inp)
        return op

    def custom_op(self, node, op_type, *args):
        node2caller = GLOBAL_MAP.get_ele(KEYS.NODE_CALLER_MAP)
        if node2caller is None:
            node2caller: Dict[str, Callable] = {}
            GLOBAL_MAP.set_map(KEYS.NODE_CALLER_MAP, node2caller)
        node2caller[node.name] = node.caller
        op = TorchCustomOperation(op_type, op_type)
        for i, arg in enumerate(args):
            op.set_config(str(i), arg)
        attrs = GLOBAL_MAP.get_ele(KEYS.CUSTOM_OP_ATTRS_MAP).get(op_type, None)
        if attrs:
            attr_vals = args[len(args) - len(attrs) :]
            for name, val in zip(attrs, attr_vals):
                op.set_attr_by_name(name, val)
        return op

    def conv2d(self, input, weight, bias, stride, padding, dilation, groups):
        return self._convolution(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            False,
            None,
            groups,
            None,
            None,
            None,
        )

    def conv_transpose2d(
        self, input, weight, bias, stride, padding, output_padding, groups, dilation
    ):
        return self._convolution(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            None,
            None,
            None,
        )

    def Correlation1DElemwiseF(self, input_1, input_2, pad_size):
        # op = TorchBaseOperation(OP.CORRELATION1D_ELEMWISE, "Correlation1d_Elemwise", force_to_primitive=True)
        op = TorchCorrelationOperation(
            OP.CORRELATION1D_ELEMWISE,
            "Correlation1d_Elemwise",
            force_to_primitive=True,
        )
        op.set_config("input_1", input_1)
        op.set_config("input_2", input_2)
        op.set_config("pad_size", pad_size)
        return op

    def Correlation2DElemwiseF(self, input_1, input_2, pad_size):
        # op = TorchBaseOperation(OP.CORRELATION2D_ELEMWISE, "Correlation2d_Elemwise", force_to_primitive=True)
        op = TorchCorrelationOperation(
            OP.CORRELATION2D_ELEMWISE,
            "Correlation2d_Elemwise",
            force_to_primitive=True,
        )
        op.set_config("input_1", input_1)
        op.set_config("input_2", input_2)
        op.set_config("pad_size", pad_size)
        return op

    def CostVolumeF(self, input_1, input_2, maxdisp):
        op = TorchCostVolumeOperation(
            OP.COST_VOLUME, "CostVolume", force_to_primitive=True
        )
        op.set_config("input_1", input_1)
        op.set_config("input_2", input_2)
        op.set_config("maxdisp", maxdisp)
        return op

    def log_softmax(self, input, dim, dtype=None):
        op = TorchLogSoftmax()
        op.set_config("dim", dim)
        return op

    def list_construct(self, *args):
        op = TorchBaseOperation(OP.LIST, OP.LIST)
        if self.cur_node.in_tensors:
            op.set_config("input", list(args))
        else:
            op.set_config("input", [])
        return op

    def tuple_construct(self, *args):
        op = TorchBaseOperation(OP.TUPLE, OP.TUPLE)
        if self.cur_node.in_tensors:
            op.set_config("input", list(args))
        else:
            op.set_config("input", [])
        return op

    def tuple_index(self, input, index):
        op = TorchBaseOperation(OP.TUPLE_INDEX, OP.TUPLE_INDEX)
        op.set_config("input", input)
        op.set_config("index", index)
        return op

    # def device(self, input):
    #  op = TorchBaseOperation(OP.DEVICE, ".device")
    #  op.set_config("input", input)
    #  return op

    def dtype(self, input):
        op = TorchBaseOperation(OP.DTYPE, ".dtype")
        op.set_config("input", input)
        return op

    def argmax(self, input, dim, keepdim=False):
        if dim is not None and keepdim == True:
            op = TorchArgMax_DIM()
        else:
            op = TorchBaseOperation(OP.ARGMAX, "argmax")
        op.set_config("input", input)
        op.set_config("dim", dim)
        op.set_config("keepdim", bool(keepdim))
        return op

    def _shape_as_tensor(self, input):
        op = TorchBaseOperation(OP.SHAPE_AS_TENSOR, "_shape_as_tensor")
        op.set_config("input", input)
        return op

    def arange(self, *args):
        supported_schemas = [
            "aten::arange(Scalar end, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            "aten::arange(Scalar start, Scalar end, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            "aten::arange(Scalar start, Scalar end, Scalar step, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            "aten::arange(Scalar start, Scalar end, Scalar step, Tensor out) -> Tensor",
            "aten::arange(Scalar end, Tensor out) -> Tensor",
        ]

        schema_handler = SchemaHelper(self.cur_node.schema)
        op = TorchArange()
        if schema_handler.toString() == supported_schemas[0]:
            op.set_config("end", args[0])
            if args[1]:
                op.set_config("dtype", scalar_type_to_pytorch_type[args[1]])
            if args[3]:
                if isinstance(args[3], str):
                    op.set_config("device", f"'{args[3]}'")
                else:
                    op.set_config("device", args[3])

        elif schema_handler.toString() == supported_schemas[1]:
            op.set_config("start", args[0])
            op.set_config("end", args[1])
            if args[2]:
                op.set_config("dtype", scalar_type_to_pytorch_type[args[2]])
            if args[4]:
                if isinstance(args[4], str):
                    op.set_config("device", f"'{args[4]}'")
                else:
                    op.set_config("device", args[4])

        elif schema_handler.toString() == supported_schemas[2]:
            op.set_config("start", args[0])
            op.set_config("end", args[1])
            op.set_config("step", args[2])
            if args[3]:
                op.set_config("dtype", scalar_type_to_pytorch_type[args[3]])
            if args[5]:
                if isinstance(args[5], str):
                    op.set_config("device", f"'{args[5]}'")
                else:
                    op.set_config("device", args[5])

        elif schema_handler.toString() == supported_schemas[3]:
            op.set_config("start", args[0])
            op.set_config("end", args[1])
            op.set_config("step", args[2])

        elif schema_handler.toString() == supported_schemas[4]:
            op.set_config("end", args[0])

        else:
            return self.default(self.cur_node, "aten::arange", *args)

        return op

    def sin(self, input, *args):
        op = TorchUnaryOp(OP.SIN, "sin")
        op.set_config("input", input)
        return op

    def cos(self, input, *args):
        op = TorchUnaryOp(OP.COS, "cos")
        op.set_config("input", input)
        return op

    def pow(self, input, exponent, *args):
        # 지수 2 는 ggml_sqr 로 정확히 접힌다. 나머지는 OP.POW(render 가 반복 mul / exp·log).
        if isinstance(exponent, (int, float)) and exponent == 2:
            op = TorchUnaryOp(OP.SQUARE, "square")
            op.set_config("input", input)
            return op
        op = TorchUnaryOp(OP.POW, "pow")
        op.set_config("input", input)
        op.set_config("exponent", exponent)
        return op

    def type_as(self, input, other):
        # dtype 만 맞추는 연산. cast 로 매핑(ggml 그래프는 단일 dtype → render 는 passthrough).
        op = TorchUnaryOp(OP.CAST, "to")
        op.set_config("input", input)
        op.set_config("other", other)
        return op

    def linspace(self, start, end, steps, *args):
        op = TorchBaseOperation(OP.LINSPACE, "linspace")
        op.set_config("start", start)
        op.set_config("end", end)
        op.set_config("steps", steps)
        return op

    def roll(self, input, shifts, dims=None):
        op = TorchBaseOperation(OP.ROLL, "roll")
        op.set_config("input", input)
        op.set_config("shifts", shifts)
        op.set_config("dims", dims)
        return op

    def new_ones(self, input, *args):
        op = TorchBaseOperation(OP.FULL, "full")
        op.set_config("size", args[0] if args else None)
        op.set_config("fill_value", 1.0)
        return op

    def fill_(self, input, value):
        # 텐서 전체를 상수로 덮는 in-place op → 같은 shape 의 상수 텐서와 동치.
        op = TorchBaseOperation(OP.FULL, "full")
        op.set_config(
            "size", list(input.shape) if isinstance(input, Tensor) else None
        )
        op.set_config("fill_value", value)
        return op
