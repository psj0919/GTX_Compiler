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
import itertools
from typing import List, Dict, Any, NoReturn, Tuple
import numpy as np
from functools import partial
from gtx_shared.base import GTX_OP, gtx_KEYS
from gtx_shared.gtx_graph import Tensor, Node
from gtx_graph import XGraph
from gtx_shared.utils import calculate_op_scale, DataXopError

GtxQuantInfo = Dict[str, Dict[str, List[int]]]


class _Converter:
    _torch2ggml_type = {
        np.float32: "FLOAT32",
        np.float64: "FLOAT64",
        np.int64: "INT64",
        np.int32: "INT32",
    }

    _gtx2numpy_type = {
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
    }

    _pad_mode = {"pad_mode": {0: "FLOOR", 1: "CEIL", 2: "SAME", 3: "VALID"}}

    _torch2ggml_value = {
        GTX_OP.CONV2D: _pad_mode,
        GTX_OP.DEPTHWISE_CONV2D: _pad_mode,
        GTX_OP.CONVTRANSPOSE2D: _pad_mode,
        GTX_OP.DEPTHWISE_CONVTRANSPOSE2D: _pad_mode,
        GTX_OP.MAX_POOL: _pad_mode,
        GTX_OP.MAX_POOL1D: _pad_mode,
        GTX_OP.AVG_POOL: _pad_mode,
        GTX_OP.PAD: {"mode": {0: "CONSTANT", 1: "REFLECT", 2: "SYMMETRIC"}},
    }

    @classmethod
    def to_gtx_dtype(cls, numpy_dtype):
        return cls._torch2ggml_type[numpy_dtype]

    @classmethod
    def to_gtx_dtype_by_string(cls, dtype):
        return {
            "float32": "FLOAT32",
            "float64": "FLOAT64",
            "int32": "INT32",
            "int64": "INT64",
        }.get(dtype, dtype)

    @classmethod
    def to_gtx_attr_value(
        cls, node_op_type, gtx_attr_name: str, gtx_attr_value: Any
    ):
        if (
            node_op_type not in cls._torch2ggml_value
            or gtx_attr_name not in cls._torch2ggml_value[node_op_type]
        ):
            return gtx_attr_value
        else:
            return cls._torch2ggml_value[node_op_type][gtx_attr_name][gtx_attr_value]

    @classmethod
    def to_numpy_dtype(cls, gtx_dtype):
        return cls._gtx2numpy_type[gtx_dtype]


def _get_gtx_attr_from_node(node: Node):
    attrs = None
    if len(node.op.attrs) > 0:
        attrs: Dict[str, Any] = {}
        for attr_name, attr_value in node.op.attrs.items():
            if node.op.is_gtx_attr(attr_name):
                attrs[attr_name.value] = _Converter.to_gtx_attr_value(
                    node.op.type, attr_name.value, attr_value.value
                )
    return attrs


def _pack(
    xgraph: XGraph,
    node: Node,
    pack_name: str,
    packed_item: List[Any],
    quant_config: GtxQuantInfo,
) -> Tuple["gtx.Op", List["gtx.Op"]]:
    """
    pack items into stack op
    """
    pack_list = []
    pack_input_ops: Dict[str, List["gtx.Op"]] = {}
    for i, item in enumerate(packed_item):
        if isinstance(item, Tensor):
            pack_list.append(xgraph.get_op_by_name(item.node.name))
        else:
            # dtype = np.int64 if isinstance(item, int) else np.float64
            dtype = np.float32
            const_op = xgraph.create_fixed_const_op(
                name=node.name + f"_{pack_name}_attr[{i}]",
                data=np.array([item], dtype=dtype),
                quant_info=quant_config,
            )
            pack_list.append(const_op)

    pack_input_ops["input"] = pack_list
    attrs: Dict[str, Any] = {}
    attrs["axis"] = 0
    sub_op_pack = xgraph.create_fixed_normal_op(
        node.name + f"_{pack_name}_i0",
        "stack",
        quant_config,
        attrs=attrs,
        input_ops=pack_input_ops,
    )
    return sub_op_pack, pack_list


# def _sub(xgraph, name, input, other, quant_config):
#   if isinstance(input, Tensor) and (not isinstance(other, Tensor)):
#     operand1 = xgraph.get_op_by_name(input.node.name)
#     new_type = other.dtype if isinstance(other, np.ndarray) else type(other)
#     # if input.node.transpose_out_order:
#     #   shape = permute_axes(input.shape, input.node.transpose_out_order)
#     # else:
#     #   shape = input.shape
#     operand2 = np.ones(input.shape, dtype=new_type) * other
#     operand2 = xgraph.create_const_op(f"{name}_other", operand2)
#   elif isinstance(other, Tensor) and (not isinstance(input, Tensor)):
#     new_type = input.dtype if isinstance(input, np.ndarray) else type(input)
#     # if other.node.transpose_out_order:
#     #   shape = permute_axes(other.shape, other.node.transpose_out_order)
#     # else:
#     #   shape = other.shape
#     operand1 = np.ones(other.shape, dtype=new_type) * input
#     operand1 = xgraph.create_const_op(f"{name}_input", operand1)
#     operand2 = xgraph.get_op_by_name(other.node.name)
#   else:
#     operand1 = xgraph.get_op_by_name(input.node.name)
#     operand2 = xgraph.get_op_by_name(other.node.name)

#   input_ops: Dict[str, List["gtx.Op"]] = {}
#   input_ops["input"] = [operand1, operand2]
#   input_ops["input"] = xgraph.create_input_fix_ops(input_ops["input"], name, quant_config)
#   xgraph.create_fixed_normal_op(name, "sub", quant_config, input_ops=input_ops)


def data_iss_op(xgraph: XGraph, node: Node, quant_config: GtxQuantInfo) -> NoReturn:
    shape = node.out_tensors[0].shape
    if not shape:
        shape = [1]

    if shape[0] == 0:
        raise DataXopError("data", shape)
    # shape = permute_axes(shape, node.transpose_out_order)
    try:
        out_tensor = np.zeros(shape, dtype=np.float32)
        attrs: Dict[str, Any] = {}
        attrs["shape"] = shape
        attrs["data_type"] = _Converter.to_gtx_dtype(out_tensor.dtype.type)
        xgraph.create_fixed_normal_op(
            node.name, "data", quant_config, tensor=out_tensor, attrs=attrs
        )
    except:
        raise DataXopError("data", shape)


def const_iss_op(xgraph: XGraph, node: Node, quant_config: GtxQuantInfo) -> NoReturn:
    data = node.node_attr(node.op.AttrName.DATA)
    data_type = np.dtype(node.out_tensors[0].dtype)
    data_type = np.float32 if data_type == np.float64 else data_type
    if not isinstance(data, list) and (not isinstance(data, np.ndarray)):
        data = [data]

    data = np.array(data, dtype=data_type)
    data = (
        np.transpose(data, node.transpose_out_order)
        if node.transpose_out_order
        else data
    )
    xgraph.create_fixed_const_op(name=node.name, data=data, quant_info=quant_config)


def reduction_mean(
    xgraph: XGraph, node: Node, quant_config: GtxQuantInfo
) -> NoReturn:

    attrs = _get_gtx_attr_from_node(node)

    input_ops: Dict[str, List[Op]] = {}
    input_ops["input"] = [xgraph.get_op_by_name(node.in_nodes[0])]
    input_ops["input"] = xgraph.create_input_fix_ops(
        input_ops["input"], node.name, quant_config
    )

    in_tensor_shape = node.in_tensors[0].shape
    if node.node_attr(node.op.AttrName.DIMS) == [None]:
        dim_list = [i for i in range(len(in_tensor_shape))]
    else:
        dim_list = node.node_attr(node.op.AttrName.DIMS)

    rec = 1
    for i in dim_list:
        rec = rec * in_tensor_shape[i]

    if (rec & (rec - 1)) != 0:
        xgraph.create_fixed_normal_op(
            node.name + "_i0",
            "reduction_mean",
            quant_config,
            attrs=attrs,
            input_ops=input_ops,
        )
        scale = calculate_op_scale(rec, node)
        scale = [scale]
        xgraph.create_fixed_const_op(
            name=node.name + "_i1",
            data=np.array(scale, dtype=np.float32),
            quant_info=quant_config,
        )

        input_ops: Dict[str, List[Op]] = {}
        input_ops["input"] = [
            xgraph.get_op_by_name(node.name + "_i0"),
            xgraph.get_op_by_name(node.name + "_i1"),
        ]
        xgraph.create_fixed_normal_op(
            node.name, "mul", quant_config, input_ops=input_ops
        )
    else:
        xgraph.create_fixed_normal_op(
            node.name, "reduction_mean", quant_config, attrs=attrs, input_ops=input_ops
        )


def shape(xgraph: XGraph, node: Node, quant_config: GtxQuantInfo) -> NoReturn:
    r"""gtx shape is a macro operator, including shape, stridedslice"""
    # raise NotImplementedError("shape")
    input_list = []
    shape_input_ops: Dict[str, List["gtx.Op"]] = {}
    for input in node.in_nodes:
        input_op = xgraph.get_op_by_name(input)
        input_list.append(input_op)
    shape_input_ops["input"] = input_list

    sub_op_shape = xgraph.create_fixed_normal_op(
        node.name + "_i0", "shape", quant_config, input_ops=shape_input_ops
    )

    attrs: Dict[str, Any] = {}
    strided_slice_input_ops: Dict[str, List["gtx.Op"]] = {}
    strided_slice_input_ops["input"] = [sub_op_shape]
    dim = node.node_attr(node.op.AttrName.AXIS)
    attrs["begin"] = [dim]
    attrs["end"] = [dim + 1]
    xgraph.create_fixed_normal_op(
        node.name,
        "strided_slice",
        quant_config,
        attrs=attrs,
        input_ops=strided_slice_input_ops,
    )


def reshape(xgraph: XGraph, node: Node, quant_config: GtxQuantInfo) -> NoReturn:
    r"""gtx reshape is a macro operator, including pack, reshape"""
    shape = node.node_attr(node.op.AttrName.SHAPE)
    sub_op_pack, pack_list = _pack(xgraph, node, "shape", shape, quant_config)
    input_ops: Dict[str, List["gtx.Op"]] = {}
    input_ops["shape"] = [sub_op_pack]
    input_ops["input"] = [xgraph.get_op_by_name(node.in_nodes[0])]
    xgraph.create_fixed_normal_op(
        node.name, "reshape", quant_config, input_ops=input_ops
    )


# def rsub(xgraph: XGraph, node: Node, quant_config: GtxQuantInfo) -> NoReturn:
#   operand1, operand2 = node.node_attr(node.op.AttrName.INPUT), node.node_attr(
#       node.op.AttrName.OTHER)

#   _sub(xgraph, node.name, operand1, operand2, quant_config)


# def sub(xgraph: XGraph, node: Node, quant_config: GtxQuantInfo) -> NoReturn:
#   operand1, operand2 = node.node_attr(node.op.AttrName.INPUT), node.node_attr(
#       node.op.AttrName.OTHER)
#   _sub(xgraph, node.name, operand1, operand2, quant_config)


def binary_op(op_type: str, xgraph: XGraph, node: Node, quant_config: GtxQuantInfo):
    input, other = node.node_attr(node.op.AttrName.INPUT), node.node_attr(
        node.op.AttrName.OTHER
    )
    input_name = input.name if input.is_param_tensor() else input.node.name
    operand1 = xgraph.get_op_by_name(input_name)
    other_name = other.name if other.is_param_tensor() else other.node.name
    operand2 = xgraph.get_op_by_name(other_name)

    input_ops: Dict[str, List["gtx.Op"]] = {}
    input_ops["input"] = [operand1, operand2]
    input_ops["input"] = xgraph.create_input_fix_ops(
        input_ops["input"], node.name, quant_config
    )
    xgraph.create_fixed_normal_op(node.name, op_type, quant_config, input_ops=input_ops)


def default_iss_op(
    iss_op_type: str, xgraph: XGraph, node: Node, quant_config: GtxQuantInfo
) -> NoReturn:

    input_ops: Dict[str, List["gtx.Op"]] = {}
    if node.has_bound_params():
        for param_name, param_tensor in node.op.params.items():
            param = xgraph.get_op_by_name(param_tensor.name)
            input_ops[param_name.name.lower()] = [param]

    input_list = []
    for input in node.in_tensors:
        if node.has_bound_params() and input.is_param_tensor():
            continue
        elif input.is_param_tensor():
            input_op = xgraph.get_op_by_name(input.name)
        else:
            input_op = xgraph.get_op_by_name(input.node.name)
        input_list.append(input_op)

    input_ops["input"] = xgraph.create_input_fix_ops(
        input_list, node.name, quant_config
    )

    attrs = _get_gtx_attr_from_node(node)
    xgraph.create_fixed_normal_op(
        node.name, iss_op_type, quant_config, attrs=attrs, input_ops=input_ops
    )


def resize(xgraph: XGraph, node: Node, quant_config: GtxQuantInfo) -> NoReturn:
    """
    resize is a macro operator, including concat , resize
    """
    attrs: Dict[str, Any] = {}
    # attrs["scale"] = node.node_attr(node.op.AttrName.SCALE)

    attrs["align_corners"] = node.node_attr(node.op.AttrName.ALIGN_CORNERS)
    attrs["half_pixel_centers"] = node.node_attr(node.op.AttrName.HALF_PIXEL_CENTERS)
    attrs["mode"] = node.node_attr(node.op.AttrName.MODE)
    # attrs["mode"] = {0: "NEAREST", 3: "BILINEAR"}.get(attrs["mode"])
    size = node.node_attr(node.op.AttrName.SIZE)
    scale = node.node_attr(node.op.AttrName.SCALE)
    # if size[0] == 0 and size[1] == 0:
    if all([s == 0 for s in size]):
        attrs["scale"] = scale
        input_ops: Dict[str, List["gtx.Op"]] = {}
        input_list = []
        for input in node.in_nodes:
            input_op = xgraph.get_op_by_name(input)
            input_list.append(input_op)
        input_ops["input"] = xgraph.create_input_fix_ops(
            input_list, node.name, quant_config
        )
        xgraph.create_fixed_normal_op(
            node.name, "resize", quant_config, attrs=attrs, input_ops=input_ops
        )
    else:
        sub_pack_op, pack_list = _pack(xgraph, node, "size", size, quant_config)
        input_ops: Dict[str, List["gtx.Op"]] = {}
        input_ops["size"] = [sub_pack_op]
        input_list = []
        for input in node.in_nodes:
            input_op = xgraph.get_op_by_name(input)
            input_list.append(input_op)
        input_ops["input"] = input_list
        input_ops["input"] = [
            op
            for op in input_ops["input"]
            if op.get_name() not in [i.get_name() for i in pack_list]
        ]
        input_ops["input"] = xgraph.create_input_fix_ops(
            input_ops["input"], node.name, quant_config
        )
        xgraph.create_fixed_normal_op(
            node.name, "resize", quant_config, attrs=attrs, input_ops=input_ops
        )


def dense(xgraph: XGraph, node: Node, quant_config: GtxQuantInfo) -> NoReturn:

    input_ops: Dict[str, List["gtx.Op"]] = {}
    for param_name, param_tensor in node.op.params.items():
        if param_name == node.op.ParamName.WEIGHTS:
            weights = xgraph.get_op_by_name(param_tensor.name)
        else:
            bias = xgraph.get_op_by_name(param_tensor.name)
            input_ops["bias"] = [bias]

    input_list = []
    for input in node.in_nodes:
        input_op = xgraph.get_op_by_name(input)
        input_list.append(input_op)
    input_ops["input"] = xgraph.create_input_fix_ops(
        input_list, node.name, quant_config
    )
    input_ops["input"].append(weights)

    attrs: Dict[str, Any] = {}
    attrs["transpose_a"] = False
    attrs["transpose_b"] = True

    xgraph.create_fixed_normal_op(
        node.name, "matmul", quant_config, attrs=attrs, input_ops=input_ops
    )


# def permute_invar_op(iss_op_type, xgraph: XGraph, node: Node, quant_config: GtxQuantInfo) -> NoReturn:
#   to_gtx(iss_op_type)(xgraph, node, quant_config)


# def flatten(xgraph: XGraph, node: Node, quant_config: GtxQuantInfo) -> NoReturn:
#   to_gtx("flatten")(xgraph, node, quant_config)


def avgpool(xgraph: XGraph, node: Node, quant_config: GtxQuantInfo) -> NoReturn:
    scale = 1.0
    if node.node_attr(node.op.AttrName.KERNEL) == [3, 3]:
        scale = 9.0 * 7.0 / 64.0
    elif node.node_attr(node.op.AttrName.KERNEL) == [5, 5]:
        scale = 25.0 * 10.0 / 256.0
    elif node.node_attr(node.op.AttrName.KERNEL) in [[6, 6], [3, 6], [6, 3]]:
        scale = 36.0 * 7.0 / 256.0
    elif node.node_attr(node.op.AttrName.KERNEL) == [7, 7]:
        scale = 49.0 * 21.0 / 1024.0
    elif node.node_attr(node.op.AttrName.KERNEL) == [14, 14]:
        scale = 196.0 * 21.0 / 4096.0
    else:
        rec = (
            node.node_attr(node.op.AttrName.KERNEL)[0]
            * node.node_attr(node.op.AttrName.KERNEL)[1]
        )
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

    attrs = _get_gtx_attr_from_node(node)
    # attrs: Dict[str, Any] = {}
    # for attr_name, attr_value in node.op.attrs.items():
    #   attrs[attr_name.value] = _Converter.to_gtx_attr_value(attr_name.value, attr_value.value)

    input_ops: Dict[str, List["gtx.Op"]] = {}
    input_ops["input"] = [xgraph.get_op_by_name(node.in_nodes[0])]
    input_ops["input"] = xgraph.create_input_fix_ops(
        input_ops["input"], node.name, quant_config
    )
    xgraph.create_fixed_normal_op(
        node.name + "_i0", "avgpool2d", quant_config, attrs=attrs, input_ops=input_ops
    )

    scale = [scale]
    xgraph.create_fixed_const_op(
        name=node.name + "_i1",
        data=np.array(scale, dtype=np.float32),
        quant_info=quant_config,
    )

    input_ops: Dict[str, List["gtx.Op"]] = {}
    input_ops["input"] = [
        xgraph.get_op_by_name(node.name + "_i0"),
        xgraph.get_op_by_name(node.name + "_i1"),
    ]
    xgraph.create_fixed_normal_op(node.name, "mul", quant_config, input_ops=input_ops)


# def squeeze(xgraph: XGraph, node: Node, quant_config: GtxQuantInfo) -> NoReturn:
#   to_gtx("squeeze")(xgraph, node, quant_config)


def zeros(xgraph: XGraph, node: Node, quant_config: GtxQuantInfo) -> NoReturn:
    shape = node.node_attr(node.op.AttrName.SHAPE)
    data = np.zeros(shape, dtype=_Converter.to_numpy_dtype(node.out_tensors[0].dtype))
    xgraph.create_fixed_const_op(name=node.name, data=data, quant_info=quant_config)


def conv3d(xgraph: XGraph, node: Node, quant_config: GtxQuantInfo) -> NoReturn:
    attrs = _get_gtx_attr_from_node(node)
    attrs["kernel"] = attrs["kernel"][::-1]
    attrs["stride"] = attrs["stride"][::-1]
    attrs["dilation"] = attrs["dilation"][::-1]
    attrs["pad"] = list(
        itertools.chain.from_iterable([[pad] * 2 for pad in attrs["pad"][::-1]])
    )
    print(attrs)
    input_ops: Dict[str, List["gtx.Op"]] = {}
    if node.has_bound_params():
        for param_name, param_tensor in node.op.params.items():
            param = xgraph.get_op_by_name(param_tensor.name)
            input_ops[param_name.name.lower()] = [param]

    input_list = []
    for input in node.in_tensors:
        if node.has_bound_params() and input.is_param_tensor():
            continue
        elif input.is_param_tensor():
            input_op = xgraph.get_op_by_name(input.name)
        else:
            input_op = xgraph.get_op_by_name(input.node.name)
        input_list.append(input_op)

    input_ops["input"] = xgraph.create_input_fix_ops(
        input_list, node.name, quant_config
    )

    xgraph.create_fixed_normal_op(
        node.name, "conv3d", quant_config, attrs=attrs, input_ops=input_ops
    )


def conv_transpose_3d(
    xgraph: XGraph, node: Node, quant_config: GtxQuantInfo
) -> NoReturn:
    attrs = _get_gtx_attr_from_node(node)
    attrs["kernel"] = attrs["kernel"][::-1]
    attrs["stride"] = attrs["stride"][::-1]
    attrs["dilation"] = attrs["dilation"][::-1]
    attrs["pad"] = list(
        itertools.chain.from_iterable([[pad] * 2 for pad in attrs["pad"][::-1]])
    )
    print(attrs)
    input_ops: Dict[str, List["gtx.Op"]] = {}
    if node.has_bound_params():
        for param_name, param_tensor in node.op.params.items():
            param = xgraph.get_op_by_name(param_tensor.name)
            input_ops[param_name.name.lower()] = [param]

    input_list = []
    for input in node.in_tensors:
        if node.has_bound_params() and input.is_param_tensor():
            continue
        elif input.is_param_tensor():
            input_op = xgraph.get_op_by_name(input.name)
        else:
            input_op = xgraph.get_op_by_name(input.node.name)
        input_list.append(input_op)

    input_ops["input"] = xgraph.create_input_fix_ops(
        input_list, node.name, quant_config
    )

    xgraph.create_fixed_normal_op(
        node.name, "transposed_conv3d", quant_config, attrs=attrs, input_ops=input_ops
    )


def scale(xgraph, node, quant_config):
    attrs: Dict[str, Any] = {}
    input_ops: Dict[str, List["gtx.Op"]] = {}
    if node.has_bound_params():
        for param_name, param_tensor in node.op.params.items():
            if param_name == node.op.ParamName.GAMMA:
                input_ops["scale"] = [xgraph.get_op_by_name(param_tensor.name)]
            if param_name == node.op.ParamName.BETA:
                input_ops["bias"] = [xgraph.get_op_by_name(param_tensor.name)]

    input_list = []
    for input in node.in_tensors:
        if node.has_bound_params() and input.is_param_tensor():
            continue
        elif input.is_param_tensor():
            input_op = xgraph.get_op_by_name(input.name)
        else:
            input_op = xgraph.get_op_by_name(input.node.name)
        input_list.append(input_op)

    input_ops["input"] = xgraph.create_input_fix_ops(
        input_list, node.name, quant_config
    )
    xgraph.create_fixed_normal_op(
        node.name, "scale", quant_config, attrs=attrs, input_ops=input_ops
    )


def hsigmoid(xgraph: XGraph, node: Node, quant_config: GtxQuantInfo) -> NoReturn:
    scale = 6.0 * 2731.0 / 16384.0

    attrs = _get_gtx_attr_from_node(node)

    input_ops: Dict[str, List["gtx.Op"]] = {}
    input_ops["input"] = [xgraph.get_op_by_name(node.in_nodes[0])]
    input_ops["input"] = xgraph.create_input_fix_ops(
        input_ops["input"], node.name, quant_config
    )
    xgraph.create_fixed_normal_op(
        node.name + "_i0",
        "hard-sigmoid",
        quant_config,
        attrs=attrs,
        input_ops=input_ops,
    )

    scale = [scale]
    xgraph.create_fixed_const_op(
        name=node.name + "_i1",
        data=np.array(scale, dtype=np.float32),
        quant_info=quant_config,
    )

    input_ops: Dict[str, List["gtx.Op"]] = {}
    input_ops["input"] = [
        xgraph.get_op_by_name(node.name + "_i0"),
        xgraph.get_op_by_name(node.name + "_i1"),
    ]
    xgraph.create_fixed_normal_op(node.name, "mul", quant_config, input_ops=input_ops)


def hswish(xgraph: XGraph, node: Node, quant_config: GtxQuantInfo) -> NoReturn:

    scale = 6.0 * 2731.0 / 16384.0

    attrs = _get_gtx_attr_from_node(node)
    node_input_op = xgraph.get_op_by_name(node.in_nodes[0])

    input_ops: Dict[str, List["gtx.Op"]] = {}
    input_ops["input"] = [node_input_op]
    input_ops["input"] = xgraph.create_input_fix_ops(
        input_ops["input"], node.name, quant_config
    )
    hsigmoid_op = xgraph.create_fixed_normal_op(
        node.name + "_i0",
        "hard-sigmoid",
        quant_config,
        attrs=attrs,
        input_ops=input_ops,
    )

    scale = [scale]
    const_op = xgraph.create_const_op(
        name=node.name + "_i1", data=np.array(scale, dtype=np.float32)
    )

    input_ops["input"] = [hsigmoid_op, const_op]
    mul_op = xgraph.create_normal_op(node.name + "_mul", "mul", input_ops=input_ops)
    if (
        quant_config
        and node.name in quant_config["output"]
        and quant_config["output"][node.name][0] is not None
    ):
        mul_fp = [8, None]
        mul_fp[0], _ = quant_config["output"][node.name][0]
        mul_fp[1] = mul_fp[0] - 1
        attrs: Dict[str, Any] = {}
        attrs["fix_point"] = mul_fp[1]
        attrs["bit_width"] = mul_fp[0]
        attrs["round_mode"] = "NPU_ROUND"
        attrs["if_signed"] = True
        input_ops["input"] = [mul_op]
        op_name = mul_op.get_name() + gtx_KEYS.FIX_OP_SUFFIX
        mul_fixed_op = xgraph.create_normal_op(
            op_name, "fix", attrs=attrs, input_ops=input_ops
        )

        input_ops["input"] = [mul_fixed_op, node_input_op]
    else:
        input_ops["input"] = [mul_op, node_input_op]

    hswish_fixed_op = xgraph.create_fixed_normal_op(
        node.name, "mul", quant_config, input_ops=input_ops
    )


def custom_iss_op(xgraph: XGraph, node: Node, quant_config: GtxQuantInfo) -> NoReturn:
    shape = node.out_tensors[0].shape
    if not shape:
        shape = [1]

    attrs = _get_gtx_attr_from_node(node)
    attrs = {} if attrs is None else attrs

    attrs["shape"] = shape
    # numpy_type = _Converter.to_numpy_dtype(node.out_tensors[0].dtype)
    # attrs["data_type"] = _Converter.to_gtx_dtype(numpy_type)

    attrs["data_type"] = _Converter.to_gtx_dtype_by_string(node.out_tensors[0].dtype)

    input_ops: Dict[str, List["gtx.Op"]] = {}
    input_list = []
    for input in node.in_tensors:
        if input.is_param_tensor():
            input_op = xgraph.get_op_by_name(input.name)
        else:
            input_op = xgraph.get_op_by_name(input.node.name)
        input_list.append(input_op)

    input_ops["input"] = xgraph.create_input_fix_ops(
        input_list, node.name, quant_config
    )
    xgraph.create_fixed_normal_op(
        node.name, node.op.type, quant_config, attrs=attrs, input_ops=input_ops
    )


def to_gtx(iss_op_type):
    return partial(default_iss_op, iss_op_type)


# def to_permute_invar_op(iss_op_type):
#   return partial(permute_invar_op, iss_op_type)


def to_binary_op(iss_op_type):
    return partial(binary_op, iss_op_type)


ISS_CONVERTOR = {
    # GTX op type: (XIR op type , XIR_CONVERT_FUNCTION)
    GTX_OP.INPUT: ("data", data_iss_op),
    GTX_OP.CONV1D: ("conv1d", to_gtx("conv1d")),
    GTX_OP.CONV2D: ("conv2d", to_gtx("conv2d")),
    GTX_OP.DEPTHWISE_CONV2D: ("depthwise-conv2d", to_gtx("depthwise-conv2d")),
    GTX_OP.CONVTRANSPOSE2D: ("transposed-conv2d", to_gtx("transposed-conv2d")),
    GTX_OP.AVG_POOL: ("avgpool2d", avgpool),
    GTX_OP.MAX_POOL1D: ("maxpool1d", to_gtx("maxpool1d")),
    GTX_OP.MAX_POOL: ("maxpool2d", to_gtx("maxpool2d")),
    GTX_OP.RELU: ("relu", to_gtx("relu")),
    GTX_OP.LEAKY_RELU: ("leaky-relu", to_gtx("leaky-relu")),
    GTX_OP.GELU: ("gelu", to_gtx("gelu")),
    GTX_OP.PRELU: ("prelu", to_gtx("prelu")),
    # GTX_OP.SQRT: ("sqrt", to_gtx("sqrt")),
    GTX_OP.TANH: ("tanh", to_gtx("tanh")),
    GTX_OP.SIGMOID: ("sigmoid", to_gtx("sigmoid")),
    GTX_OP.DENSE: ("matmul", dense),
    GTX_OP.MATMUL: ("matmul", to_gtx("matmul")),
    GTX_OP.RESHAPE: ("reshape", reshape),
    GTX_OP.ADD: ("add", to_binary_op("add")),
    # GTX_OP.SCALAR_ADD: ("add", to_binary_op("add")),
    GTX_OP.FLATTEN: ("flatten", to_gtx("flatten")),
    GTX_OP.CONCAT: ("concat", to_gtx("concat")),
    GTX_OP.MULTIPLY: ("mul", to_binary_op("mul")),
    # GTX_OP.SCALAR_MUL: ("mul", to_binary_op("mul")),
    GTX_OP.STRIDED_SLICE: ("strided_slice", to_gtx("strided_slice")),
    GTX_OP.RSUB: ("sub", to_binary_op("sub")),
    GTX_OP.SUB: ("sub", to_binary_op("sub")),
    GTX_OP.PAD: ("pad", to_gtx("pad")),
    GTX_OP.PAD_ND: ("pad", to_gtx("pad")),
    GTX_OP.RESIZE: ("resize", resize),
    GTX_OP.SOFTMAX: ("softmax", to_gtx("softmax")),
    GTX_OP.PERMUTE: ("transpose", to_gtx("transpose")),
    GTX_OP.CONST: ("const", const_iss_op),
    GTX_OP.TENSOR: ("const", const_iss_op),
    GTX_OP.RELU6: ("relu6", to_gtx("relu6")),
    GTX_OP.MEAN: ("reduction_mean", reduction_mean),
    GTX_OP.BATCH_NORM: ("scale", scale),
    # GTX_OP.LAYER_NORM: ("layernorm", to_gtx("layernorm")),
    GTX_OP.QUANT_STUB: ("data", data_iss_op),
    GTX_OP.MAX: ("reduction_max", to_gtx("reduction_max")),
    GTX_OP.TRANSPOSE: ("transpose", to_gtx("transpose")),
    GTX_OP.SQUEEZE: ("squeeze", to_gtx("squeeze")),
    GTX_OP.ZEROS: ("const", zeros),
    GTX_OP.NEG: ("neg", to_gtx("neg")),
    GTX_OP.DIV: ("div", to_binary_op("div")),
    GTX_OP.SUM: ("reduction_sum", to_gtx("reduction_sum")),
    GTX_OP.HSIGMOID: ("hard-sigmoid", hsigmoid),
    GTX_OP.HSWISH: ("hard-swish", hswish),
    GTX_OP.PIXEL_SHUFFLE: ("pixel-shuffle", to_gtx("pixel-shuffle")),
    GTX_OP.PIXEL_UNSHUFFLE: ("pixel-shuffle", to_gtx("pixel-shuffle")),
    GTX_OP.CONV3D: ("conv3d", to_gtx("conv3d")),
    GTX_OP.DEPTHWISE_CONV3D: ("depthwise-conv3d", to_gtx("depthwise-conv3d")),
    GTX_OP.CONVTRANSPOSE3D: ("transposed-conv3d", to_gtx("transposed-conv3d")),
    GTX_OP.DEPTHWISE_CONVTRANSPOSE3D: (
        "transposed-depthwise-conv3d",
        to_gtx("transposed-depthwise-conv3d"),
    ),
    GTX_OP.RESIZE_3D: ("resize", resize),
    GTX_OP.DEPTHWISE_CONVTRANSPOSE2D: (
        "transposed-depthwise-conv2d",
        to_gtx("transposed-depthwise-conv2d"),
    ),
    GTX_OP.ARGMAX_DIM: ("argmax", to_gtx("argmax")),
    # GTX_OP.MISH: ("mish", to_gtx("mish")),
    # GTX_OP.CLAMP: ("clamp", to_gtx("clamp")),
    # GTX_OP.INSTANCE_NORM: ("instancenorm", to_gtx("instancenorm")),
    # GTX_OP.GROUP_NORM: ("groupnorm", to_gtx("groupnorm"))
}
