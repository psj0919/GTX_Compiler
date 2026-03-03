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
from gtx_shared.gtx_graph import Tensor
from gtx_shared.quantization import maybe_get_quantizer
from gtx_shared.quantization import quantize_tensors
import gtx_utils as py_utils
from gtx_utils import TorchOpClassType
from gtx_shared.utils import gtx_KEYS, GTX_OP, GLOBAL_MAP, GtxScreenLogger
from gtx_shared.utils import GtxOption
from gtx_shared.gtx_graph import GtxGraphHolder

__all__ = ["Module"]


def creat_module(torch_op_type, torch_op_attr, *args, **kwargs):
    if torch_op_attr.op_class_type == TorchOpClassType.NN_MODULE:
        # creat module for module
        module_cls = getattr(torch.nn, torch_op_type, None)
        if module_cls:
            class Module(module_cls):
                r"""quantizable operation"""

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.node = None
                    self.quant_mode, self.quantizer = maybe_get_quantizer()
                    self.param_quantized = False
                    self.qparams = None
                    self.torch_op_type = torch_op_type

                def extra_repr(self):
                    return f"'{module_cls.__name__}'"

                def forward(self, *args, **kwargs):
                    # quantize input tensor
                    configer = GtxGraphHolder()
                    qinputs = quantize_tensors(
                        list(args), self.node, tensor_type="input"
                    )
                    if configer.node_quantizable_with_params(self.node):
                        qparams = []
                        inplace = (
                            GtxOption.gtx_quant_off.value
                            or self.quantizer is not None
                            and self.quantizer.inplace
                        )
                        # quantize weights/scale and bias for batch norm
                        if not configer.is_conv_like(self.node) or self.node.node_attr(
                            self.node.op.AttrName.BIAS_TERM
                        ):
                            param_names = self.params_name[:2]
                            params = [self.weight, self.bias]
                        else:
                            param_names = [self.params_name[0]]
                            params = [self.weight]
                        if not self.param_quantized:
                            if inplace:
                                _ = quantize_tensors(
                                    params,
                                    self.node,
                                    tensor_names=param_names,
                                    tensor_type="param",
                                )
                                qparams = [p for p in params]
                            else:
                                qparams = quantize_tensors(
                                    params,
                                    self.node,
                                    tensor_names=param_names,
                                    tensor_type="param",
                                )
                            if not GtxOption.gtx_quant_off.value:
                                self.param_quantized = True
                        else:
                            qparams = [p for p in params]

                    output = super().forward(*args, **kwargs)
                    if isinstance(output, (list, tuple)):
                        output = quantize_tensors(output, self.node)
                    else:
                        output = quantize_tensors([output], self.node)[0]
                    return output

            return Module(*args, **kwargs)

    elif torch_op_attr.op_class_type in [
        TorchOpClassType.NN_FUNCTION,
        TorchOpClassType.TORCH_FUNCTION,
        TorchOpClassType.NN_CORE_FUNCTION,
    ]:
        # create module for function
        module_map = {
            TorchOpClassType.NN_FUNCTION: torch.nn.functional,
            TorchOpClassType.TORCH_FUNCTION: torch,
            TorchOpClassType.NN_CORE_FUNCTION: torch._C._nn,
        }
        caller = getattr(
            module_map.get(torch_op_attr.op_class_type, None), torch_op_type
        )
        if caller:

            class FuncModule(torch.nn.Module):
                r"""quantizable operation"""

                def __init__(self, caller, *args, **kwards):
                    super().__init__()
                    self.node = None
                    self.quant_mode, self.quantizer = maybe_get_quantizer()
                    self.caller = caller
                    self.torch_op_type = torch_op_type

                def extra_repr(self):
                    return f"'{caller.__name__}'"

                def forward(self, *args, **kwargs):

                    inputs = []

                    def collect_inputs(inputs, value):
                        if isinstance(value, torch.Tensor):
                            inputs.append(value)
                        elif isinstance(value, (tuple, list)):
                            for i in value:
                                collect_inputs(inputs, i)

                    for _, v in kwargs.items():
                        collect_inputs(inputs, v)

                    inputs = quantize_tensors(inputs, self.node, tensor_type="input")
                    try:
                        output = caller(*args, **kwargs)
                        if (
                            isinstance(output, torch.Tensor)
                            and (self.quantizer is not None)
                            and (not self.quantizer.exporting)
                        ):
                            output = output.clone()
                    except TypeError as e:
                        GtxScreenLogger().warning_once(
                            f"{str(e)}. The arguments of function will convert to positional arguments."
                        )
                        inputs = list(args) + list(kwargs.values())
                        output = caller(*inputs)

                    if isinstance(output, (list, tuple)):
                        output = quantize_tensors(output, self.node)
                    else:
                        output = quantize_tensors([output], self.node)[0]

                    return output

            return FuncModule(caller, *args, **kwargs)

    elif torch_op_attr.op_class_type == TorchOpClassType.TENSOR:
        # create module for method
        if getattr(torch.Tensor, torch_op_type, None):

            class TensorModule(torch.nn.Module):
                r"""quantizable operation"""

                def __init__(self, op_type, *args, **kwards):
                    super().__init__()
                    self.node = None
                    self.quant_mode, self.quantizer = maybe_get_quantizer()
                    self.op_type = op_type
                    self.torch_op_type = torch_op_type

                def extra_repr(self):
                    return f"'{self.op_type}'"

                def forward(self, input, *args, **kwargs):

                    if isinstance(input, (list, tuple)):
                        input = quantize_tensors(input, self.node, tensor_type="input")
                    else:
                        input = quantize_tensors(
                            [input], self.node, tensor_type="input"
                        )[0]

                    if self.torch_op_type == "size" and self._forward_hooks:
                        self.node.in_tensors[0].shape = list(input.size())
                    output = getattr(input, self.op_type, None)(*args, **kwargs)

                    if isinstance(output, (list, tuple)):
                        output = quantize_tensors(output, self.node)
                    else:
                        output = quantize_tensors([output], self.node)[0]

                    return output

            return TensorModule(torch_op_type, *args, **kwargs)
    elif torch_op_attr.op_class_type in [
        TorchOpClassType.TORCH_SCRIPT_BUILTIN_FUNCTION,
        TorchOpClassType.MATH_BUILTIN_FUNCTION,
        TorchOpClassType.GLOBAL_BUILTIN_FUNCTION,
    ]:

        class BuiltinFuncModule(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.node = None
                self.quant_mode, self.quantizer = maybe_get_quantizer()
                self.torch_op_type = torch_op_type

            def extra_repr(self):
                return f"'{self.node.caller.__name__}'"

            def forward(self, *args):

                inputs = []

                def collect_inputs(inputs, value):
                    if isinstance(value, torch.Tensor):
                        inputs.append(value)
                    elif isinstance(value, (tuple, list)):
                        for i in value:
                            collect_inputs(inputs, i)

                for v in args:
                    collect_inputs(inputs, v)

                inputs = quantize_tensors(inputs, self.node, tensor_type="input")

                caller_map = GLOBAL_MAP.get_ele(gtx_KEYS.NODE_CALLER_MAP)
                output = caller_map[self.node.name](*args)

                if isinstance(output, (list, tuple)):
                    output = quantize_tensors(output, self.node)
                else:
                    output = quantize_tensors([output], self.node)[0]

                return output

        return BuiltinFuncModule()

    elif torch_op_attr.op_class_type == TorchOpClassType.CUSTOM_FUNCTION:

        class CustomModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.node = None
                self.quant_mode, self.quantizer = maybe_get_quantizer()
                self.torch_op_type = torch_op_type

            def extra_repr(self):
                return f"'{torch_op_type}'"

            def forward(self, *args):
                caller_map = GLOBAL_MAP.get_ele(gtx_KEYS.NODE_CALLER_MAP)
                output = caller_map[self.node.name](*args)
                if isinstance(output, (list, tuple)):
                    output = quantize_tensors(output, self.node)
                else:
                    output = quantize_tensors([output], self.node)[0]
                return output

        return CustomModule()
    elif torch_op_attr.op_class_type == TorchOpClassType.AUTO_INFER_OP:

        class BuiltinFuncModule(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.node = None
                self.quant_mode, self.quantizer = maybe_get_quantizer()
                self.torch_op_type = torch_op_type

            def extra_repr(self):
                return f"'{torch_op_type}'"

            def forward(self, args):
                caller_map = GLOBAL_MAP.get_ele(gtx_KEYS.NODE_CALLER_MAP)
                output = caller_map[self.node.name](**args)
                if isinstance(output, (list, tuple)):
                    output = quantize_tensors(output, self.node)
                else:
                    output = quantize_tensors([output], self.node)[0]
                return output

        return BuiltinFuncModule()

    else:
        # UNKNOWN op_class_type — passthrough module (INPUT, RETURN 등)
        class PassthroughModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.node = None
                self.quant_mode, self.quantizer = maybe_get_quantizer()
                self.torch_op_type = torch_op_type

            def extra_repr(self):
                return f"'{torch_op_type}'"

            def forward(self, *args, **kwargs):
                if args:
                    return args[0]
                if 'input' in kwargs:
                    return kwargs['input']
                # return first kwarg value
                for v in kwargs.values():
                    return v
                return None

        return PassthroughModule()


def Module(gtx_type, *args, **kwargs):
    torch_op_type = py_utils.get_torch_op_type(gtx_type)
    torch_op_attr = py_utils.get_torch_op_attr(torch_op_type)
    return creat_module(torch_op_type, torch_op_attr, *args, **kwargs)


class CallFunctionModule(torch.nn.Module):
    r"""quantizable operation"""

    def __init__(self, caller):
        super().__init__()
        self.node = None
        self.quant_mode, self.quantizer = maybe_get_quantizer()
        self.caller = caller

    def extra_repr(self):
        return f"'{self.caller.__name__}'"

    def forward(self, *args, **kwargs):

        inputs = []

        def collect_inputs(inputs, value):
            if isinstance(value, torch.Tensor):
                inputs.append(value)
            elif isinstance(value, (tuple, list)):
                for i in value:
                    collect_inputs(inputs, i)

        for _, v in kwargs.items():
            collect_inputs(inputs, v)

        inputs = quantize_tensors(inputs, self.node, tensor_type="input")
        output = self.caller(*args, **kwargs)
        if (
            isinstance(output, torch.Tensor)
            and (self.quantizer is not None)
            and (not self.quantizer.exporting)
        ):
            output = output.clone()

        if isinstance(output, (list, tuple)):
            output = quantize_tensors(output, self.node)
        else:
            output = quantize_tensors([output], self.node)[0]

        return output
