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

import copy
from itertools import chain
from functools import reduce

from shared.base import KEYS, OP, PARAM
from shared.utils.log import Debugger
from shared import utils as utils
from shared.utils import Option

from .base_graph import Graph
from .base_node import Node
from .base_operator import Operation

from .utils import *


class GraphHolder(Debugger):

    def __init__(self, graph=None, model_type=None):
        self.__graph = graph
        self.model_type = model_type
        self.scan_commander = {}
        self._QuantGroups = None

        # for quantization
        self.QUANTIZABLE_OPS = [
            OP.AVG_POOL,
            OP.ADAPTIVEAVGPOOL2D,
            OP.CONVTRANSPOSE2D,
            OP.BATCH_NORM,
            OP.LAYER_NORM,
            OP.INSTANCE_NORM,
            OP.GROUP_NORM,
            OP.BIAS_ADD,
            OP.BASIC_LSTM,
            OP.BASIC_GRU,
            OP.CONV2D,
            OP.CONCAT,
            OP.DEPTHWISE_CONV2D,
            OP.CONV1D,
            OP.DENSE,
            OP.ADD,
            OP.MULTIPLY,
            OP.DIV,
            OP.MAX_POOL,
            OP.MAX,
            OP.MEAN,
            OP.MAX_POOL1D,
            OP.MIN,
            OP.RESIZE,
            OP.SUB,
            OP.RSUB,
            OP.PAD,
            OP.QUANT_STUB,
            OP.INPUT,
            OP.TUPLE_INPUT,
            OP.CONV3D,
            OP.DEPTHWISE_CONV3D,
            OP.RESIZE_3D,
            OP.CONVTRANSPOSE3D,
            OP.SUM,
            OP.HSWISH,
            OP.HSIGMOID,
            OP.MATMUL,
            OP.DEPTHWISE_CONVTRANSPOSE2D,
            OP.DEPTHWISE_CONVTRANSPOSE3D,
            OP.CORRELATION1D_ELEMWISE,
            OP.CORRELATION2D_ELEMWISE,
            OP.COST_VOLUME,
            OP.SOFTMAX,
            OP.PRELU,
            OP.SQRT,
            # OP.CLAMP,
            OP.GELU,
            OP.MISH,
        ]
        self.TENSORRT_QUANTIZABLE_OPS = [
            OP.AVG_POOL,
            OP.ADAPTIVEAVGPOOL2D,
            OP.CONV1D,
            OP.CONV2D,
            OP.DEPTHWISE_CONV2D,
            OP.CONV3D,
            OP.DEPTHWISE_CONV3D,
            OP.CONVTRANSPOSE2D,
            OP.CONVTRANSPOSE3D,
            OP.DEPTHWISE_CONVTRANSPOSE2D,
            OP.DEPTHWISE_CONVTRANSPOSE3D,
            OP.DENSE,
        ]

        self.LSTM_QUANTIZABLE_OPS = [
            OP.CONV2D,
            OP.DENSE,
            OP.ADD,
            OP.MULTIPLY,
            OP.SIGMOID,
            OP.TANH,
            OP.SUB,
            OP.RSUB,
            OP.DIV,
            OP.CONCAT,
            OP.STACK,
            OP.INPUT,
            OP.TUPLE_INPUT,
            OP.MATMUL,
            OP.ADDMM,
            OP.SOFTMAX,
            OP.LOG_SOFTMAX,
            OP.LAYER_NORM,
            OP.EMBEDDING,
        ]
        self.QUANTIZABLE_OPS_WITH_PARAMS = [
            OP.DENSE,
            OP.CONV1D,
            OP.CONV2D,
            OP.DEPTHWISE_CONV2D,
            OP.CONVTRANSPOSE2D,
            OP.DEPTHWISE_CONVTRANSPOSE2D,
            OP.CONV3D,
            OP.DEPTHWISE_CONV3D,
            OP.CONVTRANSPOSE3D,
            OP.DEPTHWISE_CONVTRANSPOSE3D,
            OP.BATCH_NORM,
            # OP.INSTANCE_NORM,
        ]
        self.MULTIPLE_OUTPUTS_OPS = [  # OP types where cannot do quantization
            OP.CHUNK,  # has multiple outputs and cannot be deployed
            OP.STRIDED_SLICE,  # copy part of memory from a tensor and create a new tensor
            OP.PERMUTE,  # can not be deployed
            OP.FLATTEN,  # no calculation and only tensor in-place operation
            OP.RESHAPE,  # no calculation and only tensor in-place operation
            OP.PIXEL_SHUFFLE,  # no calculation and only tensor in-place operation
            OP.CONTIGUOUS,  # no calculation and only tensor in-place operation
            OP.SQUEEZE,  # no calculation and only tensor in-place operation
            OP.UNSQUEEZE,
        ]

        self.CONV_LIKE_OPS = [
            OP.CONV2D,
            OP.CONV3D,
            OP.CONV1D,
            OP.CONVTRANSPOSE2D,
            OP.CONVTRANSPOSE3D,
            OP.DEPTHWISE_CONV2D,
            OP.DEPTHWISE_CONV3D,
            OP.DENSE,
            OP.DEPTHWISE_CONVTRANSPOSE2D,
        ]
        self.QUANTIZABLE_DTYPES = ["float16", "float32", "float64"]

    def get_model_type(self):
        return self.model_type or ""

    def get_node(self, node_name):
        return self.__graph.node(node_name)

    def is_node_quantizable(self, node, lstm):
        if Option.tensorrt_strategy.value:
            ret = node.op.type in self.TENSORRT_QUANTIZABLE_OPS
            # check the node is in quant_stub or not:
            ret = ret and node.in_quant_part
            return ret
        elif not lstm:
            # check the node type if it needs to be quantized
            ret = node.op.type in self.QUANTIZABLE_OPS
            # check the node is in quant_stub or not:
            ret = ret and node.in_quant_part
            return ret
        else:
            ret = node.op.type in self.LSTM_QUANTIZABLE_OPS
            ret = ret and node.in_quant_part
            ret = ret and (not self.will_merge_with_table(node, lstm))
            return ret

    def node_output_quantizable(self, node):
        if node.op.type in self.MULTIPLE_OUTPUTS_OPS:
            return False
        else:
            return True

    def node_quantizable_with_params(self, node):
        if node.op.type in self.QUANTIZABLE_OPS_WITH_PARAMS:
            return True
        else:
            return False

    def quant_node_params(self, node_or_name):
        node = self._find_node(node_or_name)
        if node.op.type == OP.BATCH_NORM:
            return {
                k: v
                for k, v in node.op.params.items()
                if k in [node.op.ParamName.GAMMA, node.op.ParamName.BETA]
            }
        else:
            return node.op.params

    def is_concat_input(self, node_or_name):
        node = self._find_node(node_or_name)
        isConcatInput = False  # only ouput tensor to concat
        children = self.graph.children(node)
        if len(children) == 1 and children[0].op.type == OP.CONCAT:
            isConcatInput = True
        return isConcatInput

    def quant_output(self, node_or_name):
        node = self._find_node(node_or_name)
        if Option.only_int_quant.value == False:
            return node
        if not node.in_quant_part:
            return node
        idx = -1
        end_node = self.__graph.node(self._QuantGroups[node.name][idx])

        if (
            end_node.op.type == OP.TUPLE_INPUT
            and len(self.graph.children(end_node)) > 0
            and self.graph.children(end_node)[0].op.type == OP.TUPLE_UNPACK
        ):
            end_node = self.graph.children(end_node)[0]

        if self.is_concat_input(end_node):
            for c in self.graph.children(end_node):
                if c.op.type == OP.CONCAT:
                    return c

        while end_node.op.type in self.MULTIPLE_OUTPUTS_OPS:
            idx = idx - 1
            if -idx > len(self._QuantGroups[node.name]):
                break
            end_node = self.__graph.node(self._QuantGroups[node.name][idx])
        while end_node.op.type in self.MULTIPLE_OUTPUTS_OPS:
            if not end_node.in_nodes:
                break
            up_node = end_node.in_nodes[0]
            end_node = self.__graph.node(up_node)

        return end_node

    def quant_group(self, node_or_name):
        node = self._find_node(node_or_name)
        if not node.in_quant_part:
            return None
        QuantGroupTypes = []
        for node_name in self._QuantGroups[node.name]:
            QuantGroupTypes.append(self.__graph.node(node_name).op.type)
        return self._QuantGroups[node.name], QuantGroupTypes

    # only used in TF RNN case
    def quant_input_names(self, node_or_name, inputs, params=None, validate=True):
        node = self._find_node(node_or_name)
        valid_inputs = None
        for i in inputs:
            try:
                input_node = self.get_node(
                    node_name=utils.node_from_output(i, self.get_model_type())
                )
            except KeyError:
                continue
            if input_node.op.type == OP.INPUT:
                valid_input = input_node.name
                if valid_inputs is None:
                    valid_inputs = []
                valid_inputs.append(valid_input)

        return valid_inputs

    def _find_node(self, node_or_name):
        if isinstance(node_or_name, str):
            return self.get_node(node_name=node_or_name)
        else:
            return node_or_name

    def is_conv_like(self, node_or_name):
        node = self._find_node(node_or_name)
        return node.op.type in self.CONV_LIKE_OPS

    def will_merge_with_table(self, node_or_name, lstm):
        if not lstm:
            return False
        node = self._find_node(node_or_name)
        if node.op.type in [OP.LAYER_NORM]:
            children = self.graph.children(node)
            if len(children) == 1:
                if children[0].op.type in [OP.SIGMOID, OP.TANH]:
                    return True
        return False

    @property
    def quant_groups(self):
        return self._QuantGroups

    @property
    def graph(self):
        return self.__graph
