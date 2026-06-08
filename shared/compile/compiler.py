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

from typing import Any, Dict, List, NoReturn, Optional

import numpy as np

from shared.base import OP
from shared.graph import Graph
from shared.utils import (
    AddXopError,
    Option,
    GLOBAL_MAP,
    KEYS,
    ScreenLogger,
)
from shared.utils import QError, QWarning
from .graph import Graph
from .xop_creator import ISS_CONVERTOR, custom_iss_op, to_

QuantInfo = Dict[str, Dict[str, List[int]]]


class Compiler(object):
    @staticmethod
    def do_compile(
        compile_graph: Graph,
        output_file_name=None,
        quant_config_info: Optional[QuantInfo] = None,
        graph_attr_kwargs: Optional[Dict[str, Any]] = None,
    ) -> NoReturn:
        r"""convert  graph to xmodel"""
        # debug
        # for type, bnfp in quant_config_info.items():
        #   print(f"{type}\n")
        #   for name, bnfp_value in bnfp.items():
        #     print(f"{name}:{bnfp_value}\n")
        if Option.quant_off.value:
            quant_config_info = None

        xgraph = Graph(compile_graph.name)

        if graph_attr_kwargs is not None:
            for name, attr in graph_attr_kwargs.items():
                xgraph.graph.set_attr(name, attr)

        for node in compile_graph.nodes:
            for param_type, param_tensor in node.op.params.items():
                if node.op.type == OP.BATCH_NORM and param_type not in [
                    node.op.ParamName.GAMMA,
                    node.op.ParamName.BETA,
                ]:
                    continue
                if xgraph.get_op_by_name(param_tensor.name):
                    continue
                # print(f"{node.name}: {param_tensor.name}, {id(param_tensor)}")
                data = np.copy(param_tensor.data)
                if (
                    node.op.type
                    in [OP.CONVTRANSPOSE2D, OP.DEPTHWISE_CONVTRANSPOSE2D]
                    and param_type == node.op.ParamName.WEIGHTS
                ):
                    # OHWI -> OH'W'I reverse the order of ele in both h and w axis
                    data = np.flip(data, (1, 2))
                    data = np.ascontiguousarray(data)
                elif (
                    node.op.type
                    in [OP.CONVTRANSPOSE3D, OP.DEPTHWISE_CONVTRANSPOSE3D]
                    and param_type == node.op.ParamName.WEIGHTS
                ):
                    # OHWDI -> OH'W'D'I reverse the order of ele in both h and w axis
                    data = np.flip(data, (1, 2, 3))
                    data = np.ascontiguousarray(data)
                try:
                    if data.dtype == np.float16:
                        data = data.astype(np.float32)
                    xgraph.create_fixed_const_op(
                        name=param_tensor.name, data=data, quant_info=quant_config_info
                    )
                except Exception as e:
                    raise AddXopError(param_tensor.name, "const", str(e))

        custom2 = GLOBAL_MAP.get_ele(KEYS.CUSTOM_TO_XIR_LIST)
        if custom2:
            for op_type in custom2:
                ISS_CONVERTOR[op_type] = (op_type, to_(op_type))

        for node in compile_graph.nodes:
            if node.op.type == OP.RETURN:
                continue
            # print("convert...:", node.op.type, node.name, node.out_tensors[0].shape, node.in_quant_part)
            # import sys
            # sys.stdout.flush()
            try:
                ISS_CONVERTOR.get(node.op.type, (node.op.type, custom_iss_op))[1](
                    xgraph, node, quant_config_info
                )
            except Exception as e:
                raise AddXopError(node.name, node.op.type, str(e))

        if output_file_name:
            if quant_config_info is None:
                output_file_name += "_float"
            else:
                output_file_name += "_int"

            xgraph.export_to_xmodel(output_file_name)

        return xgraph

    @staticmethod
    def generate_c_code(
        compile_graph: Graph,
        output_dir: str = "./output",
        model_name: str = "model",
        nest_id: int = 0,
        spu_id: int = 0,
    ) -> dict:
        r""" Graph를 C 소스코드(.c/.h)로 변환.

         intrinsic 함수 호출로 구성된 C 코드를 생성합니다.
        생성된 코드는 RISC-V 크로스 컴파일러로 빌드하여
        ISS 시뮬레이터에서 실행할 수 있습니다.

        Args:
            compile_graph: 파싱/최적화된  Graph (shape/params 포함)
            output_dir: 출력 디렉토리 경로
            model_name: 생성될 C 파일명 (기본: "model")
            nest_id: 타겟 NEST ID (0-3). 기본값 0.
            spu_id: 타겟 SPU ID (0-3). 기본값 0.

        Returns:
            생성된 파일 경로 딕셔너리
        """
        from .c_codegen import CCodeGenerator

        codegen = CCodeGenerator(
            graph=compile_graph,
            output_dir=output_dir,
            model_name=model_name,
            nest_id=nest_id,
            spu_id=spu_id,
        )
        return codegen.generate()

    @staticmethod
    def verify_xmodel(compile_graph: Graph, xgraph: Graph):
        """verify the xmodel by  node shape"""

        for node in compile_graph.nodes:
            if not node.out_tensors:
                continue
            if node.out_tensors[0].ndim and node.out_tensors[0].ndim > 1:
                iss_op_shape = xgraph.get_op_output_shape(node.name)
                if tuple(iss_op_shape) != tuple(node.out_tensors[0].shape):
                    ScreenLogger().error2user(
                        QError.SHAPE_MISMATCH,
                        f"output shape of {node.name}({node.out_tensors[0].shape}) is different from the output shape of XIR ({iss_op_shape}).",
                    )

    @staticmethod
    def verify_graph(compile_graph):
        msg = ""
        for node in compile_graph.nodes:
            if node.op.type == OP.RETURN:
                continue
            if node.blocks:
                msg += (
                    f"XIR don't support control flow op.({node.name}, {node.op.type})\n"
                )
            elif len(node.out_tensors) > 1 and all(
                [len(tensor.uses) > 0 for tensor in node.out_tensors]
            ):
                msg += f"XIR don't support multi-outputs op.({node.name}, {node.op.type})\n"
            elif node.op.type not in ISS_CONVERTOR.keys() and all(
                [tensor.shape is None for tensor in node.out_tensors]
            ):
                msg += f"XIR don't support custom op without shape info.({node.name}, {node.op.type})\n"

        if msg:
            return False, msg

        return True, msg
