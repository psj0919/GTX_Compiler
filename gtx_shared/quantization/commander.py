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
from gtx_shared.utils import BaseCommander
from gtx_shared.base import GTX_OP
from gtx_shared import gtx_graph as graph_utils


class QuantConfigerCommander(BaseCommander):

    def create_commands(self):

        # def SoftFuseClamp(graph, quant_groups):
        #   return graph_utils.group_up(graph, quant_groups, GTX_OP.CLAMP)

        def SoftFuseHardtanh(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.HARDTANH)

        def SoftFuseRelu(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.RELU)

        def SoftFuseLeakyRelu(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.LEAKY_RELU)

        def SoftFuseRelu6(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.RELU6)

        def SoftFuseReluk(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.RELUK)

        def SoftFuseChannelScale(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.CHANNEL_SCALE)

        def SoftFuseFlatten(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.FLATTEN)

        def SoftFuseSqueeze(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.SQUEEZE)

        def SoftFusePixelShuffle(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.PIXEL_SHUFFLE)

        def SoftFuseReshape(graph, quant_groups):

            def is_reshape_parent(node):
                if node.op.type == GTX_OP.SHAPE:
                    return False
                elif node.op.type in [GTX_OP.MULTIPLY]:
                    for p in graph.parents(node.name):
                        return is_reshape_parent(p)
                else:
                    return True

            for n in graph.all_nodes():
                if not n.in_quant_part or n.blocks:
                    continue
                for p in graph.parents(n.name):
                    if is_reshape_parent(p):
                        if (
                            quant_groups[n.name][0] == n.name
                            and n.op.type == GTX_OP.RESHAPE
                        ):
                            start_node = quant_groups[p.name][0]
                            groups = graph_utils.glue_group_members(
                                graph, quant_groups, start_node, n.name
                            )
            return quant_groups

        def SoftFuseSplit(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.SPLIT)

        def SoftFuseStrideSlice(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.STRIDED_SLICE)

        def SoftFuseTranspose(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.TRANSPOSE)

        def SoftFuseTile(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.TILE)

        def SoftFuseUpSampling(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.UP_SAMPLING)

        def SoftFuseDropout(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.DROPOUT)

        def SoftFuseContiguous(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.CONTIGUOUS)

        def SoftFuseChunk(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.CHUNK)

        def SoftFusePermute(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.PERMUTE)

        def SoftFuseExpand(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.EXPAND)

        def SoftFuseInplaceCopy(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.INPLACE_COPY)

        def SoftFuseRepeat(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.REPEAT)

        # def SoftFuseSelect(graph, quant_groups):
        #   return graph_utils.group_up(graph, quant_groups, GTX_OP.SELECT)

        def SoftFuseUnsqueeze(graph, quant_groups):
            return graph_utils.group_up(graph, quant_groups, GTX_OP.UNSQUEEZE)

        return locals()
