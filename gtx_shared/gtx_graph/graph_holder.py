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

from gtx_shared.base import gtx_KEYS, GTX_OP, gtx_PARAM
from gtx_shared.utils.log import GtxDebugger
from gtx_shared import utils as gtx_utils
from gtx_shared.utils import GtxOption

from .base_graph import Graph
from .base_node import Node
from .base_operator import Operation

from .utils import *


class GtxGraphHolder(GtxDebugger):

    def __init__(self, graph=None, model_type=None):
        self.__Gtxgraph = graph
        self.model_type = model_type
        self.scan_commander = {}
        self._QuantGroups = None

        # for quantization
        self.QUANTIZABLE_OPS = [
            GTX_OP.AVG_POOL,
            GTX_OP.ADAPTIVEAVGPOOL2D,
            GTX_OP.CONVTRANSPOSE2D,
            GTX_OP.BATCH_NORM,
            GTX_OP.LAYER_NORM,
            GTX_OP.INSTANCE_NORM,
            GTX_OP.GROUP_NORM,
            GTX_OP.BIAS_ADD,
            GTX_OP.BASIC_LSTM,
            GTX_OP.BASIC_GRU,
            GTX_OP.CONV2D,
            GTX_OP.CONCAT,
            GTX_OP.DEPTHWISE_CONV2D,
            GTX_OP.CONV1D,
            GTX_OP.DENSE,
            GTX_OP.ADD,
            GTX_OP.MULTIPLY,
            GTX_OP.DIV,
            GTX_OP.MAX_POOL,
            GTX_OP.MAX,
            GTX_OP.MEAN,
            GTX_OP.MAX_POOL1D,
            GTX_OP.MIN,
            GTX_OP.RESIZE,
            GTX_OP.SUB,
            GTX_OP.RSUB,
            GTX_OP.PAD,
            GTX_OP.QUANT_STUB,
            GTX_OP.INPUT,
            GTX_OP.TUPLE_INPUT,
            GTX_OP.CONV3D,
            GTX_OP.DEPTHWISE_CONV3D,
            GTX_OP.RESIZE_3D,
            GTX_OP.CONVTRANSPOSE3D,
            GTX_OP.SUM,
            GTX_OP.HSWISH,
            GTX_OP.HSIGMOID,
            GTX_OP.MATMUL,
            GTX_OP.DEPTHWISE_CONVTRANSPOSE2D,
            GTX_OP.DEPTHWISE_CONVTRANSPOSE3D,
            GTX_OP.CORRELATION1D_ELEMWISE,
            GTX_OP.CORRELATION2D_ELEMWISE,
            GTX_OP.COST_VOLUME,
            GTX_OP.SOFTMAX,
            GTX_OP.PRELU,
            GTX_OP.SQRT,
            # GTX_OP.CLAMP,
            GTX_OP.GELU,
            GTX_OP.MISH,
        ]
        self.TENSORRT_QUANTIZABLE_OPS = [
            GTX_OP.AVG_POOL,
            GTX_OP.ADAPTIVEAVGPOOL2D,
            GTX_OP.CONV1D,
            GTX_OP.CONV2D,
            GTX_OP.DEPTHWISE_CONV2D,
            GTX_OP.CONV3D,
            GTX_OP.DEPTHWISE_CONV3D,
            GTX_OP.CONVTRANSPOSE2D,
            GTX_OP.CONVTRANSPOSE3D,
            GTX_OP.DEPTHWISE_CONVTRANSPOSE2D,
            GTX_OP.DEPTHWISE_CONVTRANSPOSE3D,
            GTX_OP.DENSE,
        ]

        self.LSTM_QUANTIZABLE_OPS = [
            GTX_OP.CONV2D,
            GTX_OP.DENSE,
            GTX_OP.ADD,
            GTX_OP.MULTIPLY,
            GTX_OP.SIGMOID,
            GTX_OP.TANH,
            GTX_OP.SUB,
            GTX_OP.RSUB,
            GTX_OP.DIV,
            GTX_OP.CONCAT,
            GTX_OP.STACK,
            GTX_OP.INPUT,
            GTX_OP.TUPLE_INPUT,
            GTX_OP.MATMUL,
            GTX_OP.ADDMM,
            GTX_OP.SOFTMAX,
            GTX_OP.LOG_SOFTMAX,
            GTX_OP.LAYER_NORM,
            GTX_OP.EMBEDDING,
        ]
        self.QUANTIZABLE_OPS_WITH_PARAMS = [
            GTX_OP.DENSE,
            GTX_OP.CONV1D,
            GTX_OP.CONV2D,
            GTX_OP.DEPTHWISE_CONV2D,
            GTX_OP.CONVTRANSPOSE2D,
            GTX_OP.DEPTHWISE_CONVTRANSPOSE2D,
            GTX_OP.CONV3D,
            GTX_OP.DEPTHWISE_CONV3D,
            GTX_OP.CONVTRANSPOSE3D,
            GTX_OP.DEPTHWISE_CONVTRANSPOSE3D,
            GTX_OP.BATCH_NORM,
            # GTX_OP.INSTANCE_NORM,
        ]
        self.MULTIPLE_OUTPUTS_OPS = [  # OP types where cannot do quantization
            GTX_OP.CHUNK,  # has multiple outputs and cannot be deployed
            GTX_OP.STRIDED_SLICE,  # copy part of memory from a tensor and create a new tensor
            GTX_OP.PERMUTE,  # can not be deployed
            GTX_OP.FLATTEN,  # no calculation and only tensor in-place operation
            GTX_OP.RESHAPE,  # no calculation and only tensor in-place operation
            GTX_OP.PIXEL_SHUFFLE,  # no calculation and only tensor in-place operation
            GTX_OP.CONTIGUOUS,  # no calculation and only tensor in-place operation
            GTX_OP.SQUEEZE,  # no calculation and only tensor in-place operation
            GTX_OP.UNSQUEEZE,
        ]

        self.CONV_LIKE_OPS = [
            GTX_OP.CONV2D,
            GTX_OP.CONV3D,
            GTX_OP.CONV1D,
            GTX_OP.CONVTRANSPOSE2D,
            GTX_OP.CONVTRANSPOSE3D,
            GTX_OP.DEPTHWISE_CONV2D,
            GTX_OP.DEPTHWISE_CONV3D,
            GTX_OP.DENSE,
            GTX_OP.DEPTHWISE_CONVTRANSPOSE2D,
        ]
        self.QUANTIZABLE_DTYPES = ["float16", "float32", "float64"]

    def get_model_type(self):
        return self.model_type or "Gtx"

    def get_Gtxnode(self, node_name):
        return self.__Gtxgraph.node(node_name)

    def is_node_quantizable(self, node, lstm):
        if GtxOption.gtx_tensorrt_strategy.value:
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
        if node.op.type == GTX_OP.BATCH_NORM:
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
        children = self.Gtxgraph.children(node)
        if len(children) == 1 and children[0].op.type == GTX_OP.CONCAT:
            isConcatInput = True
        return isConcatInput

    def quant_output(self, node_or_name):
        node = self._find_node(node_or_name)
        if GtxOption.gtx_only_int_quant.value == False:
            return node
        if not node.in_quant_part:
            return node
        idx = -1
        end_node = self.__Gtxgraph.node(self._QuantGroups[node.name][idx])

        if (
            end_node.op.type == GTX_OP.TUPLE_INPUT
            and len(self.Gtxgraph.children(end_node)) > 0
            and self.Gtxgraph.children(end_node)[0].op.type == GTX_OP.TUPLE_UNPACK
        ):
            end_node = self.Gtxgraph.children(end_node)[0]

        if self.is_concat_input(end_node):
            for c in self.Gtxgraph.children(end_node):
                if c.op.type == GTX_OP.CONCAT:
                    return c

        while end_node.op.type in self.MULTIPLE_OUTPUTS_OPS:
            idx = idx - 1
            if -idx > len(self._QuantGroups[node.name]):
                break
            end_node = self.__Gtxgraph.node(self._QuantGroups[node.name][idx])
        while end_node.op.type in self.MULTIPLE_OUTPUTS_OPS:
            if not end_node.in_nodes:
                break
            up_node = end_node.in_nodes[0]
            end_node = self.__Gtxgraph.node(up_node)

        return end_node

    def quant_group(self, node_or_name):
        node = self._find_node(node_or_name)
        if not node.in_quant_part:
            return None
        QuantGroupTypes = []
        for node_name in self._QuantGroups[node.name]:
            QuantGroupTypes.append(self.__Gtxgraph.node(node_name).op.type)
        return self._QuantGroups[node.name], QuantGroupTypes

    # only used in TF RNN case
    def quant_input_names(self, node_or_name, inputs, params=None, validate=True):
        node = self._find_node(node_or_name)
        valid_inputs = None
        for i in inputs:
            try:
                input_node = self.get_Gtxnode(
                    node_name=gtx_utils.node_from_output(i, self.get_model_type())
                )
            except KeyError:
                continue
            if input_node.op.type == GTX_OP.INPUT:
                valid_input = input_node.name
                if valid_inputs is None:
                    valid_inputs = []
                valid_inputs.append(valid_input)

        return valid_inputs

    def _find_node(self, node_or_name):
        if isinstance(node_or_name, str):
            return self.get_Gtxnode(node_name=node_or_name)
        else:
            return node_or_name

    def is_conv_like(self, node_or_name):
        node = self._find_node(node_or_name)
        return node.op.type in self.CONV_LIKE_OPS

    def will_merge_with_table(self, node_or_name, lstm):
        if not lstm:
            return False
        node = self._find_node(node_or_name)
        if node.op.type in [GTX_OP.LAYER_NORM]:
            children = self.Gtxgraph.children(node)
            if len(children) == 1:
                if children[0].op.type in [GTX_OP.SIGMOID, GTX_OP.TANH]:
                    return True
        return False

    @property
    def quant_groups(self):
        return self._QuantGroups

    @property
    def Gtxgraph(self):
        return self.__Gtxgraph
