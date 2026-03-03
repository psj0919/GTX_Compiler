# Copyright (C) Supergate - All Rights Reserved
# GTX Compiler - C Code Generator
# GTX Graph → C 소스코드 (.c/.h) 생성 백엔드
#
# GTX Context 실행 모델:
#   CPU → __split() → Plan { Shared(DDR↔L2) + Thread(L2↔L1 + 계산) } → __join() → CPU
#
# DMA 2단계:
#   Shared Scope: DDR ↔ L2 SPM (SMU 실행)
#   Thread Scope: L2 SPM ↔ L1 SPM + GTX ISA 계산 (SPU 실행)
#
# Credit 동기화:
#   Shared→Thread: __load_cr(DDR→L2, credit) → Thread의 __load_cr(L2→L1)가 credit 대기
#   Thread→Shared: __store_cr(L1→L2, credit) → Shared의 __credit_chk()가 credit 대기

import os
import re
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from gtx_shared.base.key_names import GTX_OP
from gtx_shared.compile.memory_planner import (
    MemoryPlanner,
    L1SPM_BANK_A_BASE,
    L1SPM_BANK_B_BASE,
    L1SPM_BANK_C_BASE,
    L1SPM_BANK_R_BASE,
    L1SPM_BANK_A_SIZE,
    L1SPM_BANK_B_SIZE,
    L1SPM_BANK_C_SIZE,
    L1SPM_BANK_R_SIZE,
    L2SPM_BASE,
    L2SPM_SIZE,
    L2_INPUT_OFFSET,
    L2_WEIGHT_OFFSET,
    L2_OUTPUT_OFFSET,
    DDR_INPUT_BASE,
    DDR_OUTPUT_BASE,
    DDR_WEIGHT_BASE,
    DDR_TEMP_BASE,
)
from gtx_shared.compile.weight_exporter import WeightExporter


class CCodeGenerator:
    """
    GTX Graph를 C 소스코드로 변환하는 코드 생성기.

    GTX Context 실행 모델을 준수:
      - Plan { Shared(DDR↔L2) ∥ Thread(L2↔L1 + 계산) }
      - Credit 동기화로 Shared↔Thread 간 데이터 의존성 관리
      - __split()/__join()으로 CPU↔NPU 전환
    """

    # L2 SPM 영역 오프셋 (Plan 내에서 사용)
    L2_IN = L2_INPUT_OFFSET
    L2_W = L2_WEIGHT_OFFSET
    L2_OUT = L2_OUTPUT_OFFSET

    def __init__(
        self,
        graph,
        output_dir: str = "./output",
        model_name: str = "model",
        nest_id: int = 0,
        spu_id: int = 0,
    ):
        self.graph = graph
        self.output_dir = output_dir
        self.model_name = model_name

        # NEST/SPU 타겟 설정
        self.nest_id = nest_id
        self.spu_id = spu_id

        # Credit 타겟 비트마스크 (1 << id)
        self.CREDIT_SPU = f"0x{1 << spu_id:X}"
        self.CREDIT_NEST = f"0x{1 << nest_id:X}"

        self.mem_planner = MemoryPlanner(dtype="fp16")
        self.weight_exporter = WeightExporter(output_dir)

        self._ddr_buffers: Dict[str, int] = {}
        self._ddr_buf_offset = 0

        self._header_lines: List[str] = []
        self._impl_lines: List[str] = []
        self._layer_funcs: List[str] = []
        self._inference_calls: List[str] = []
        self._module_map: Dict[int, Dict] = {}

        self._op_handlers = {
            GTX_OP.INPUT: self._emit_input,
            GTX_OP.CONV2D: self._emit_conv2d,
            GTX_OP.DEPTHWISE_CONV2D: self._emit_depthwise_conv2d,
            GTX_OP.BATCH_NORM: self._emit_batch_norm,
            GTX_OP.RELU: self._emit_relu,
            GTX_OP.RELU6: self._emit_relu6,
            GTX_OP.LEAKY_RELU: self._emit_leaky_relu,
            GTX_OP.MAX_POOL: self._emit_maxpool,
            GTX_OP.AVG_POOL: self._emit_avgpool,
            GTX_OP.ADAPTIVEAVGPOOL2D: self._emit_adaptive_avgpool,
            GTX_OP.ADD: self._emit_elemwise_add,
            GTX_OP.MULTIPLY: self._emit_elemwise_mul,
            GTX_OP.DENSE: self._emit_dense,
            GTX_OP.FLATTEN: self._emit_flatten,
            GTX_OP.SOFTMAX: self._emit_softmax,
            GTX_OP.SIGMOID: self._emit_sigmoid,
            GTX_OP.GELU: self._emit_gelu,
            GTX_OP.TANH: self._emit_tanh,
            GTX_OP.CONCAT: self._emit_concat,
            GTX_OP.RESHAPE: self._emit_reshape,
        }

    def generate(self) -> Dict[str, str]:
        """전체 C 코드 생성 진입점."""
        os.makedirs(self.output_dir, exist_ok=True)

        self._analyze_graph()

        self.weight_exporter.export_header()
        self.weight_exporter.export_binary()
        self.weight_exporter.export_address_map()

        self._generate_layer_code()

        header_path = self._write_header()
        impl_path = self._write_implementation()
        modules_dir = self._write_per_module_files()

        return {
            "header": header_path,
            "implementation": impl_path,
            "weights": os.path.join(self.output_dir, "weights.h"),
            "weight_map": os.path.join(self.output_dir, "weight_map.h"),
            "weights_bin": os.path.join(self.output_dir, "weights.bin"),
            "modules_dir": modules_dir,
        }

    # ========================================
    # 그래프 분석
    # ========================================

    def _analyze_graph(self):
        """그래프를 순회하여 가중치를 수집하고 DDR 버퍼를 할당"""
        for node in self.graph.nodes:
            if node.op.type == GTX_OP.RETURN:
                continue

            for param_type, param_tensor in node.op.params.items():
                if param_tensor.data is not None:
                    if node.op.type == GTX_OP.BATCH_NORM:
                        if hasattr(node.op, "ParamName"):
                            if param_type not in [
                                node.op.ParamName.GAMMA,
                                node.op.ParamName.BETA,
                                node.op.ParamName.MEAN,
                                node.op.ParamName.VAR,
                            ]:
                                continue

                    weight_name = self._make_weight_name(node, param_type)
                    self.weight_exporter.add_weight(
                        weight_name,
                        np.copy(param_tensor.data),
                        list(param_tensor.shape)
                        if param_tensor.shape
                        else list(param_tensor.data.shape),
                    )

            if node.out_tensors:
                out_shape = node.out_tensors[0].shape
                if out_shape:
                    out_size = self.mem_planner.calc_tensor_size(list(out_shape))
                    buf_name = f"buf_{node.name}"
                    addr = self.mem_planner.alloc_ddr_temp(buf_name, out_size)
                    self._ddr_buffers[node.name] = addr

    def _make_weight_name(self, node, param_type) -> str:
        node_name = node.name.replace("/", "_").replace(".", "_").replace(" ", "_")
        param_str = str(param_type).replace(".", "_")
        return f"{node_name}_{param_str}"

    def _get_node_output_var(self, node) -> str:
        if node.name in self._ddr_buffers:
            return f"0x{self._ddr_buffers[node.name]:08X}ULL"
        return f"DDR_TEMP_BASE + 0x{self._ddr_buf_offset:08X}"

    def _get_input_addr_expr(self, node, input_idx: int = 0) -> str:
        if not node.in_nodes:
            return f"0x{DDR_INPUT_BASE:08X}ULL"
        in_nodes_list = list(node.in_nodes)
        if input_idx < len(in_nodes_list):
            in_node = in_nodes_list[input_idx]
            if in_node.name in self._ddr_buffers:
                return f"0x{self._ddr_buffers[in_node.name]:08X}ULL"
        return f"0x{DDR_INPUT_BASE:08X}ULL"

    def _get_output_shape(self, node) -> List[int]:
        if node.out_tensors and node.out_tensors[0].shape:
            return list(node.out_tensors[0].shape)
        return [1]

    def _get_input_shape(self, node) -> List[int]:
        if node.in_tensors and node.in_tensors[0].shape:
            return list(node.in_tensors[0].shape)
        return [1]

    # ========================================
    # 코드 생성
    # ========================================

    def _extract_module_id(self, node) -> Optional[int]:
        match = re.match(r"module_(\d+)_", node.name)
        if match:
            return int(match.group(1))
        elif node.name == "input_0":
            return 0
        return None

    def _generate_layer_code(self):
        fallback_idx = 0
        for node in self.graph.nodes:
            if node.op.type == GTX_OP.RETURN:
                continue
            mod_id = self._extract_module_id(node)
            if mod_id is None:
                mod_id = fallback_idx
            handler = self._op_handlers.get(node.op.type)
            if handler:
                handler(node, mod_id)
            else:
                self._layer_funcs.append(
                    f"// WARNING: Unsupported op type '{node.op.type}' for node '{node.name}'\n"
                )
                self._inference_calls.append(
                    f"    // SKIP: {node.name} ({node.op.type}) — not supported"
                )
            fallback_idx += 1

    # ========================================
    # 헬퍼: 올바른 GTX Context Plan 생성
    # ========================================

    def _plan_1d(
        self,
        ddr_in: str,
        ddr_out: str,
        data_bytes: str,
        l2_in_off: str,
        l2_out_off: str,
        l1_in_bank: str,
        l1_out_bank: str,
        compute_lines: str,
        indent: str = "    ",
    ) -> str:
        """
        1D 데이터에 대한 표준 Plan 패턴 생성.
        Shared(DDR→L2, L2→DDR) ∥ Thread(L2→L1, 계산, L1→L2)
        """
        i = indent
        return f"""{i}__start_plan(NEST_ID);
{i}    // === Shared Scope: DDR ↔ L2 SPM ===
{i}    __start_shared();
{i}        __load_cr({ddr_in}, {l2_in_off}, {data_bytes}, {data_bytes}, 1, {data_bytes}, 1, {self.CREDIT_SPU}, {self.CREDIT_NEST});
{i}        __credit_chk({self.CREDIT_SPU});
{i}        __store_cr({l2_out_off}, {ddr_out}, {data_bytes}, {data_bytes}, 1, {data_bytes}, 1, {self.CREDIT_SPU});
{i}    __end_shared();
{i}    // === Thread Scope: L2 ↔ L1 SPM + 계산 ===
{i}    __start_thread(SPU_ID);
{i}        __load_cr({l2_in_off}, {l1_in_bank}, {data_bytes}, {data_bytes}, 1, {data_bytes}, 1, {self.CREDIT_SPU}, {self.CREDIT_NEST});
{compute_lines}
{i}        __store_cr({l1_out_bank}, {l2_out_off}, {data_bytes}, {data_bytes}, 1, {data_bytes}, 1, {self.CREDIT_SPU});
{i}    __end_thread(SPU_ID);
{i}__end_plan(NEST_ID);
"""

    # ========================================
    # Op별 코드 생성 함수 — GTX Context 준수
    # ========================================

    def _emit_input(self, node, idx):
        out_shape = self._get_output_shape(node)
        self._ddr_buffers[node.name] = DDR_INPUT_BASE
        comment = f"    // Module {idx}: INPUT {node.name} — shape {out_shape}"
        addr_comment = f"    // Input data at DDR_INPUT_BASE (0x{DDR_INPUT_BASE:08X})"
        self._inference_calls.append(comment)
        self._inference_calls.append(addr_comment)
        self._module_map[idx] = {
            "func_code": None,
            "call_line": f"{comment}\n{addr_comment}",
            "node": node,
            "func_name": None,
        }

    def _emit_relu(self, node, idx):
        """ReLU — __relu (level3): 입력 R bank → 출력 A bank"""
        shape = self._get_input_shape(node)
        in_addr = self._get_input_addr_expr(node, 0)
        out_addr = self._get_node_output_var(node)
        total_elems = 1
        for s in shape:
            total_elems *= s
        func_name = f"module_{idx}_relu"

        func_code = f"""
// Module {idx}: ReLU — {node.name}
// shape: {shape}, total elements: {total_elems}
static void {func_name}(uint64_t in_ddr, uint64_t out_ddr) {{
    uint32_t total_elems = {total_elems};
    uint32_t max_elems = L1_BANK_R_SIZE / 2;

    for (uint32_t off = 0; off < total_elems; off += max_elems) {{
        uint32_t cur = (off + max_elems > total_elems) ? (total_elems - off) : max_elems;
        uint32_t cur_bytes = cur * 2;

        __start_plan(NEST_ID);
            // Shared: DDR → L2 input, L2 output → DDR
            __start_shared();
                __load_cr(in_ddr + (uint64_t)off * 2, L2_IN, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU, CREDIT_NEST);
                __credit_chk(CREDIT_SPU);
                __store_cr(L2_OUT, out_ddr + (uint64_t)off * 2, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU);
            __end_shared();
            // Thread: L2 → L1, compute, L1 → L2
            __start_thread(SPU_ID);
                __init_spm_addr(0xF);
                __load_cr(L2_IN, BANK_R, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU, CREDIT_NEST);
                __relu(cur, BANK_A, BANK_R);
                __store_cr(BANK_A, L2_OUT, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU);
            __end_thread(SPU_ID);
        __end_plan(NEST_ID);
    }}
}}
"""
        self._layer_funcs.append(func_code)
        # DDR 버퍼는 _analyze_graph()에서 이미 할당됨 — 덮어쓰지 않음
        call_line = f"    {func_name}({in_addr}, {out_addr});"
        self._inference_calls.append(call_line)
        self._module_map[idx] = {
            "func_code": func_code,
            "call_line": call_line,
            "node": node,
            "func_name": func_name,
        }

    def _emit_relu6(self, node, idx):
        """ReLU6 — __relu6 (level3): 입력 R bank → 출력 A bank"""
        shape = self._get_input_shape(node)
        in_addr = self._get_input_addr_expr(node, 0)
        out_addr = self._get_node_output_var(node)
        total_elems = 1
        for s in shape:
            total_elems *= s
        func_name = f"module_{idx}_relu6"
        func_code = f"""
// Module {idx}: ReLU6 — {node.name}
static void {func_name}(uint64_t in_ddr, uint64_t out_ddr) {{
    uint32_t total_elems = {total_elems};
    uint32_t max_elems = L1_BANK_R_SIZE / 2;
    for (uint32_t off = 0; off < total_elems; off += max_elems) {{
        uint32_t cur = (off + max_elems > total_elems) ? (total_elems - off) : max_elems;
        uint32_t cur_bytes = cur * 2;
        __start_plan(NEST_ID);
            __start_shared();
                __load_cr(in_ddr + (uint64_t)off * 2, L2_IN, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU, CREDIT_NEST);
                __credit_chk(CREDIT_SPU);
                __store_cr(L2_OUT, out_ddr + (uint64_t)off * 2, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU);
            __end_shared();
            __start_thread(SPU_ID);
                __init_spm_addr(0xF);
                __load_cr(L2_IN, BANK_R, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU, CREDIT_NEST);
                __relu6(cur, BANK_A, BANK_R);
                __store_cr(BANK_A, L2_OUT, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU);
            __end_thread(SPU_ID);
        __end_plan(NEST_ID);
    }}
}}
"""
        self._layer_funcs.append(func_code)
        call_line = f"    {func_name}({in_addr}, {out_addr});"
        self._inference_calls.append(call_line)
        self._module_map[idx] = {
            "func_code": func_code,
            "call_line": call_line,
            "node": node,
            "func_name": func_name,
        }

    def _emit_leaky_relu(self, node, idx):
        """LeakyReLU — __lrelu (level3): 1 param (vector_size)"""
        shape = self._get_input_shape(node)
        in_addr = self._get_input_addr_expr(node, 0)
        out_addr = self._get_node_output_var(node)
        total_elems = 1
        for s in shape:
            total_elems *= s
        func_name = f"module_{idx}_lrelu"
        func_code = f"""
// Module {idx}: LeakyReLU — {node.name}
static void {func_name}(uint64_t in_ddr, uint64_t out_ddr) {{
    uint32_t total_elems = {total_elems};
    uint32_t max_elems = L1_BANK_R_SIZE / 2;
    for (uint32_t off = 0; off < total_elems; off += max_elems) {{
        uint32_t cur = (off + max_elems > total_elems) ? (total_elems - off) : max_elems;
        uint32_t cur_bytes = cur * 2;
        __start_plan(NEST_ID);
            __start_shared();
                __load_cr(in_ddr + (uint64_t)off * 2, L2_IN, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU, CREDIT_NEST);
                __credit_chk(CREDIT_SPU);
                __store_cr(L2_OUT, out_ddr + (uint64_t)off * 2, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU);
            __end_shared();
            __start_thread(SPU_ID);
                __init_spm_addr(0xF);
                __load_cr(L2_IN, BANK_R, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU, CREDIT_NEST);
                __lrelu(cur);
                __store_cr(BANK_A, L2_OUT, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU);
            __end_thread(SPU_ID);
        __end_plan(NEST_ID);
    }}
}}
"""
        self._layer_funcs.append(func_code)
        call_line = f"    {func_name}({in_addr}, {out_addr});"
        self._inference_calls.append(call_line)
        self._module_map[idx] = {
            "func_code": func_code,
            "call_line": call_line,
            "node": node,
            "func_name": func_name,
        }

    def _emit_sigmoid(self, node, idx):
        """Sigmoid — __sigm(vector_size) macro: Bank A→A (in-place)"""
        self._emit_unary_activation(
            node, idx, "sigmoid", "__sigm", bank_in="BANK_A", bank_out="BANK_A"
        )

    def _emit_gelu(self, node, idx):
        """GELU — __gelu(vector_size) macro: Bank A→A"""
        self._emit_unary_activation(
            node, idx, "gelu", "__gelu", bank_in="BANK_A", bank_out="BANK_A"
        )

    def _emit_tanh(self, node, idx):
        """Tanh — __tanh(vector_size) macro: Bank A→A"""
        self._emit_unary_activation(
            node, idx, "tanh", "__tanh", bank_in="BANK_A", bank_out="BANK_A"
        )

    def _emit_unary_activation(
        self, node, idx, op_name, intrinsic, bank_in="BANK_A", bank_out="BANK_A"
    ):
        """Unary activation (sigm/gelu/tanh) 공통 생성기"""
        shape = self._get_input_shape(node)
        in_addr = self._get_input_addr_expr(node, 0)
        out_addr = self._get_node_output_var(node)
        total_elems = 1
        for s in shape:
            total_elems *= s
        func_name = f"module_{idx}_{op_name}"
        func_code = f"""
// Module {idx}: {op_name.upper()} — {node.name}
static void {func_name}(uint64_t in_ddr, uint64_t out_ddr) {{
    uint32_t total_elems = {total_elems};
    uint32_t max_elems = L1_BANK_A_SIZE / 2;
    for (uint32_t off = 0; off < total_elems; off += max_elems) {{
        uint32_t cur = (off + max_elems > total_elems) ? (total_elems - off) : max_elems;
        uint32_t cur_bytes = cur * 2;
        __start_plan(NEST_ID);
            __start_shared();
                __load_cr(in_ddr + (uint64_t)off * 2, L2_IN, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU, CREDIT_NEST);
                __credit_chk(CREDIT_SPU);
                __store_cr(L2_OUT, out_ddr + (uint64_t)off * 2, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU);
            __end_shared();
            __start_thread(SPU_ID);
                __init_spm_addr(0xF);
                __load_cr(L2_IN, {bank_in}, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU, CREDIT_NEST);
                {intrinsic}(cur);
                __store_cr({bank_out}, L2_OUT, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU);
            __end_thread(SPU_ID);
        __end_plan(NEST_ID);
    }}
}}
"""
        self._layer_funcs.append(func_code)
        call_line = f"    {func_name}({in_addr}, {out_addr});"
        self._inference_calls.append(call_line)
        self._module_map[idx] = {
            "func_code": func_code,
            "call_line": call_line,
            "node": node,
            "func_name": func_name,
        }

    def _emit_batch_norm(self, node, idx):
        """
        BatchNorm — __batchnorm_aff (level3).
        가중치(mean/var/gamma/beta)를 DDR→L2→L1 DMA로 로드.
        채널별: 스칼라 4개를 L1에 로드 후 __batchnorm_aff 호출.
        """
        in_shape = self._get_input_shape(node)
        out_shape = self._get_output_shape(node)
        in_addr = self._get_input_addr_expr(node, 0)
        out_addr = self._get_node_output_var(node)

        eps: float = float(self._get_attr_val(node, "eps", 1e-5))
        channels = in_shape[1] if len(in_shape) > 1 else 1
        spatial = 1
        for s in in_shape[2:]:
            spatial *= s

        mean_name = self._find_weight_name(node, "mean")
        var_name = self._find_weight_name(node, "var")
        gamma_name = self._find_weight_name(node, "gamma")
        beta_name = self._find_weight_name(node, "beta")
        eps_fp16 = self._float_to_fp16_hex(eps)

        func_name = f"module_{idx}_batchnorm"

        # 채널당 spatial 타일링
        max_spatial = L1SPM_BANK_A_SIZE // 2  # Bank A에 맞는 최대 원소 수

        func_code = f"""
// Module {idx}: BatchNorm — {node.name}
// channels: {channels}, spatial: {spatial}, eps: {eps}
static void {func_name}(uint64_t in_ddr, uint64_t out_ddr) {{
    uint16_t channels = {channels};
    uint32_t spatial_size = {spatial};
    uint32_t channel_bytes = spatial_size * 2;
    uint16_t eps_fp16 = 0x{eps_fp16:04X};
    uint32_t max_spatial = L1_BANK_A_SIZE / 2;

    for (uint16_t c = 0; c < channels; c++) {{
        uint64_t ch_in  = in_ddr  + (uint64_t)c * channel_bytes;
        uint64_t ch_out = out_ddr + (uint64_t)c * channel_bytes;

        // BN 파라미터(mean,var,gamma,beta)를 DDR에서 로드
        // 각 파라미터 배열에서 채널 c의 fp16 스칼라 1개 = 2바이트
        uint64_t mean_addr  = {mean_name.upper()}_DDR_ADDR + (uint64_t)c * 2;
        uint64_t var_addr   = {var_name.upper()}_DDR_ADDR  + (uint64_t)c * 2;
        uint64_t gamma_addr = {gamma_name.upper()}_DDR_ADDR + (uint64_t)c * 2;
        uint64_t beta_addr  = {beta_name.upper()}_DDR_ADDR  + (uint64_t)c * 2;

        // spatial 타일링
        for (uint32_t s_off = 0; s_off < spatial_size; s_off += max_spatial) {{
            uint32_t cur_sp = (s_off + max_spatial > spatial_size)
                            ? (spatial_size - s_off) : max_spatial;
            uint32_t cur_bytes = cur_sp * 2;

            // L2 오프셋 배치: input data, 4 scalar params, output data
            uint32_t l2_data_in  = L2_IN;
            uint32_t l2_params   = L2_WEIGHT;   // 8 bytes (4 x fp16)
            uint32_t l2_data_out = L2_OUT;

            __start_plan(NEST_ID);
                // === Shared: DDR → L2 (data + 4 params), L2 → DDR (result) ===
                __start_shared();
                    // 입력 데이터
                    __load_cr(ch_in + (uint64_t)s_off * 2, l2_data_in,
                              cur_bytes, cur_bytes, 1, cur_bytes,
                              1, CREDIT_SPU, CREDIT_NEST);
                    // BN params: mean(2B) + var(2B) + gamma(2B) + beta(2B) = 8B
                    // 4개를 연속으로 L2에 배치 (각각 별도 DMA)
                    __load_cr(mean_addr,  l2_params + 0, 2, 2, 1, 2, 0, 0, 0);
                    __load_cr(var_addr,   l2_params + 2, 2, 2, 1, 2, 0, 0, 0);
                    __load_cr(gamma_addr, l2_params + 4, 2, 2, 1, 2, 0, 0, 0);
                    __load_cr(beta_addr,  l2_params + 6, 2, 2, 1, 2, 0, 0, 0);
                    // Thread 완료 대기 후 결과를 DDR로
                    __credit_chk(CREDIT_SPU);
                    __store_cr(l2_data_out, ch_out + (uint64_t)s_off * 2,
                               cur_bytes, cur_bytes, 1, cur_bytes,
                               1, CREDIT_SPU);
                __end_shared();

                // === Thread: L2 → L1, batchnorm_aff, L1 → L2 ===
                __start_thread(SPU_ID);
                    __init_spm_addr(0xF);
                    // 입력 데이터 → Bank A
                    __load_cr(l2_data_in, BANK_A,
                              cur_bytes, cur_bytes, 1, cur_bytes,
                              1, CREDIT_SPU, CREDIT_NEST);
                    // BN params (8B) → Bank B 시작
                    __load_cr(l2_params, BANK_B,
                              8, 8, 1, 8,
                              0, 0, 0);
                    // Bank B에서 mean/var/gamma/beta 읽기 → SVR로 로드
                    __load_svr(BANK_B + 0, 0);  // SVR[0] = mean
                    __load_svr(BANK_B + 2, 1);  // SVR[1] = var
                    __load_svr(BANK_B + 4, 2);  // SVR[2] = gamma
                    __load_svr(BANK_B + 6, 3);  // SVR[3] = beta

                    // __batchnorm_aff(size, mean, var, scale, shift, A, R, eps)
                    // mean/var/gamma/beta는 SVR에서 직접 스칼라로 전달 불가 →
                    // level3 함수가 직접 스칼라 파라미터를 받으므로,
                    // L1 메모리에서 읽어 레지스터에 넣어야 함.
                    // __batchnorm_aff는 uint16_t 스칼라(mean/var/scale/shift)를 받음.
                    // Thread Scope에서 RISC-V load로 L1 SPM 직접 읽기:
                    //   - GTX_ISS 시뮬레이터에서 검증 완료 (정상 동작)
                    //   - 대안: __load_svr로 SVR에 로드 후, 별도 intrinsic 사용
                    {{
                        volatile uint16_t *params = (volatile uint16_t*)(uintptr_t)BANK_B;
                        uint16_t m  = params[0];
                        uint16_t v  = params[1];
                        uint16_t g  = params[2];
                        uint16_t b  = params[3];
                        __batchnorm_aff(cur_sp, m, v, g, b, BANK_A, BANK_R, eps_fp16);
                    }}

                    // 결과 (Bank R) → L2
                    __store_cr(BANK_R, l2_data_out,
                               cur_bytes, cur_bytes, 1, cur_bytes,
                               1, CREDIT_SPU);
                __end_thread(SPU_ID);
            __end_plan(NEST_ID);
        }}
    }}
}}
"""
        self._layer_funcs.append(func_code)
        call_line = f"    {func_name}({in_addr}, {out_addr});"
        self._inference_calls.append(call_line)
        self._module_map[idx] = {
            "func_code": func_code,
            "call_line": call_line,
            "node": node,
            "func_name": func_name,
        }

    def _emit_conv2d(self, node, idx):
        """
        Conv2d — DDR→L2→L1 im2col + mm.
        간략화: 채널-루프 방식으로 L1에 넣을 수 있는 단위로 처리.
        """
        in_shape = self._get_input_shape(node)
        out_shape = self._get_output_shape(node)
        in_addr = self._get_input_addr_expr(node, 0)
        out_addr = self._get_node_output_var(node)

        kernel_size_raw = self._get_attr_val(node, "kernel_size", [3, 3])
        stride_raw = self._get_attr_val(node, "stride", [1, 1])
        padding_raw = self._get_attr_val(node, "padding", [0, 0])
        dilation_raw = self._get_attr_val(node, "dilation", [1, 1])
        has_bias: bool = bool(self._get_attr_val(node, "bias", False))

        n: int = in_shape[0] if len(in_shape) > 0 else 1
        ic: int = in_shape[1] if len(in_shape) > 1 else 1
        ih: int = in_shape[2] if len(in_shape) > 2 else 1
        iw: int = in_shape[3] if len(in_shape) > 3 else 1
        oc: int = out_shape[1] if len(out_shape) > 1 else 1
        oh: int = out_shape[2] if len(out_shape) > 2 else 1
        ow: int = out_shape[3] if len(out_shape) > 3 else 1
        kh: int = int(
            kernel_size_raw[0]
            if isinstance(kernel_size_raw, (list, tuple))
            else kernel_size_raw
        )
        kw: int = int(
            kernel_size_raw[1]
            if isinstance(kernel_size_raw, (list, tuple))
            else kernel_size_raw
        )
        sh: int = int(
            stride_raw[0] if isinstance(stride_raw, (list, tuple)) else stride_raw
        )
        sw: int = int(
            stride_raw[1] if isinstance(stride_raw, (list, tuple)) else stride_raw
        )
        ph: int = int(
            padding_raw[0] if isinstance(padding_raw, (list, tuple)) else padding_raw
        )
        pw: int = int(
            padding_raw[1] if isinstance(padding_raw, (list, tuple)) else padding_raw
        )
        dh: int = int(
            dilation_raw[0] if isinstance(dilation_raw, (list, tuple)) else dilation_raw
        )
        dw: int = int(
            dilation_raw[1] if isinstance(dilation_raw, (list, tuple)) else dilation_raw
        )

        weight_name = self._find_weight_name(node, "weights")
        bias_name = self._find_weight_name(node, "bias") if has_bias else None
        weight_ddr_macro = weight_name.upper() + "_DDR_ADDR"
        bias_ddr_macro = (bias_name.upper() + "_DDR_ADDR") if bias_name else None

        filt_h: int = kh if dh == 1 else (2 * dh + 1 if dh <= 3 else kh)
        padded_ih: int = sh * (oh - 1) + filt_h
        padded_iw: int = sw * (ow - 1) + (
            kw if dw == 1 else (2 * dw + 1 if dw <= 3 else kw)
        )
        im2col_K: int = ic * kh * kw
        weight_row_bytes: int = im2col_K * 2

        # 출력 행 타일링 계산
        L1_SIZE = (
            L1SPM_BANK_A_SIZE
            + L1SPM_BANK_B_SIZE
            + L1SPM_BANK_C_SIZE
            + L1SPM_BANK_R_SIZE
        )
        avail = L1_SIZE - weight_row_bytes
        if avail < 0:
            avail = L1_SIZE
        oh_tile = oh
        while oh_tile >= 1:
            tile_row_A = sh * (oh_tile - 1) + filt_h
            inp_bytes = tile_row_A * padded_iw * ic * 2
            tile_M = oh_tile * ow
            im2col_bytes = tile_M * im2col_K * 2
            if inp_bytes + im2col_bytes <= avail:
                break
            oh_tile = oh_tile // 2 if oh_tile > 1 else 1
            if oh_tile == 0:
                oh_tile = 1
                break

        tile_row_A = sh * (oh_tile - 1) + filt_h
        tile_M = oh_tile * ow
        needs_padding = (padded_ih != ih) or (padded_iw != iw)

        func_name = f"module_{idx}_conv2d"

        func_code = f"""
// Module {idx}: Conv2d — {node.name}
// in:[{n},{ic},{ih},{iw}] → out:[{n},{oc},{oh},{ow}] k:[{kh},{kw}] s:[{sh},{sw}] p:[{ph},{pw}] d:[{dh},{dw}]
// oh_tile={oh_tile}, tile_M={tile_M}, K={im2col_K}, padding={needs_padding}
static void {func_name}(uint64_t in_ddr, uint64_t out_ddr) {{
    uint16_t in_c = {ic}, out_c = {oc};
    uint16_t out_h = {oh}, out_w = {ow};
    uint16_t oh_tile = {oh_tile};
    uint16_t padded_w = {padded_iw};
    uint32_t weight_row_bytes = {weight_row_bytes};

    for (uint16_t oh_start = 0; oh_start < out_h; oh_start += oh_tile) {{
        uint16_t cur_oh = (oh_start + oh_tile > out_h) ? (out_h - oh_start) : oh_tile;
        uint16_t tile_row_A = {sh} * (cur_oh - 1) + {filt_h};
        uint16_t in_row_start = oh_start * {sh};
        uint16_t tile_M = cur_oh * out_w;
        uint32_t input_tile_bytes = (uint32_t)tile_row_A * padded_w * in_c * 2;
        uint32_t im2col_tile_bytes = (uint32_t)tile_M * {im2col_K} * 2;
"""

        if needs_padding:
            func_code += f"""
        // --- 패딩 입력 준비 (DDR→L2→L1, zero-padding) ---
        // 채널별로 패딩된 데이터를 L2에 구성
        for (uint16_t c = 0; c < in_c; c++) {{
            uint32_t padded_ch_bytes = (uint32_t)tile_row_A * padded_w * 2;
            uint32_t l2_ch_off = L2_IN + c * padded_ch_bytes;

            // Plan 1: zero-fill L2 영역
            __start_plan(NEST_ID);
                __start_shared();
                    __fill(l2_ch_off, padded_w * 2, padded_w * 2, tile_row_A, 0, 0);
                __end_shared();
                __start_thread(SPU_ID);
                __end_thread(SPU_ID);
            __end_plan(NEST_ID);

            // Plan 2: 실제 데이터를 L2 패딩 버퍼 내 올바른 위치에 복사
            int16_t src_row_start = (int16_t)in_row_start - {ph};
            int16_t src_row_end = src_row_start + tile_row_A - 1;
            int16_t actual_start = (src_row_start < 0) ? 0 : src_row_start;
            int16_t actual_end = (src_row_end >= {ih}) ? ({ih} - 1) : src_row_end;
            if (actual_start <= actual_end) {{
                uint16_t copy_rows = actual_end - actual_start + 1;
                uint32_t dst_row_off = (uint32_t)(actual_start - src_row_start) * padded_w * 2;
                uint32_t dst_col_off = (uint32_t){pw} * 2;

                __start_plan(NEST_ID);
                    __start_shared();
                        __load_cr(in_ddr + ((uint64_t)c * {ih} + actual_start) * {iw} * 2,
                                  l2_ch_off + dst_row_off + dst_col_off,
                                  {iw} * 2, {iw} * 2, copy_rows, padded_w * 2,
                                  0, 0, 0);
                    __end_shared();
                    __start_thread(SPU_ID);
                    __end_thread(SPU_ID);
                __end_plan(NEST_ID);
            }}
        }}
"""
        else:
            func_code += f"""
        // --- 입력 타일 DDR → L2 (패딩 불필요) ---
        for (uint16_t c = 0; c < in_c; c++) {{
            uint32_t ch_tile_bytes = (uint32_t)tile_row_A * {iw} * 2;
            __start_plan(NEST_ID);
                __start_shared();
                    __load_cr(in_ddr + ((uint64_t)c * {ih} + in_row_start) * {iw} * 2,
                              L2_IN + c * ch_tile_bytes,
                              {iw} * 2, {iw} * 2, tile_row_A, {iw} * 2,
                              0, 0, 0);
                __end_shared();
                __start_thread(SPU_ID);
                __end_thread(SPU_ID);
            __end_plan(NEST_ID);
        }}
"""

        func_code += f"""
        // --- im2col: L2→L1, im2col, 결과 L1에 유지 ---
        __start_plan(NEST_ID);
            __start_shared();
                // 입력 타일 L2 → L1 (전체)
                __load_cr(L2_IN, 0,
                          input_tile_bytes, input_tile_bytes, 1, input_tile_bytes,
                          1, CREDIT_SPU, CREDIT_NEST);
            __end_shared();
            __start_thread(SPU_ID);
                __set_spm_addr_R(0x00000);
                __set_spm_addr_A(0x00000 + ((input_tile_bytes + 3) & ~3));
                __load_cr(0, BANK_R,
                          input_tile_bytes, input_tile_bytes, 1, input_tile_bytes,
                          1, CREDIT_SPU, CREDIT_NEST);
                __im2col_n(tile_row_A, {padded_iw}, {kh}, {dh}, {sh}, {ic});
            __end_thread(SPU_ID);
        __end_plan(NEST_ID);

        // --- OC 루프: 가중치 DDR→L2→L1, mm, 결과 L1→L2→DDR ---
        for (uint16_t oc_s = 0; oc_s < out_c; oc_s++) {{
            __start_plan(NEST_ID);
                __start_shared();
                    // 가중치 1행 DDR → L2
                    __load_cr({weight_ddr_macro} + (uint64_t)oc_s * weight_row_bytes,
                              L2_WEIGHT,
                              weight_row_bytes, weight_row_bytes, 1, weight_row_bytes,
                              1, CREDIT_SPU, CREDIT_NEST);
                    // Thread 완료 대기 후 결과 L2 → DDR
                    __credit_chk(CREDIT_SPU);
                    {{
                        uint64_t dst_ddr = out_ddr
                            + (uint64_t)oc_s * out_h * out_w * 2
                            + (uint64_t)oh_start * out_w * 2;
                        uint32_t row_bytes = (uint32_t)out_w * 2;
                        __store_cr(L2_OUT, dst_ddr,
                                   row_bytes, row_bytes, cur_oh, row_bytes,
                                   1, CREDIT_SPU);
                    }}
                __end_shared();
                __start_thread(SPU_ID);
                    // 가중치 L2 → L1 Bank B
                    uint32_t spm_B_off = 0x20000;
                    __load_cr(L2_WEIGHT, spm_B_off,
                              weight_row_bytes, weight_row_bytes, 1, weight_row_bytes,
                              1, CREDIT_SPU, CREDIT_NEST);
                    __set_spm_addr_B(spm_B_off);
                    __set_spm_addr_R(0x50000);
                    // mm: A(im2col) x B(가중치) → R
                    __mm(tile_M, {im2col_K}, 1);
                    // 결과 L1 → L2
                    __store_cr(0x50000, L2_OUT,
                               tile_M * 2, tile_M * 2, 1, tile_M * 2,
                               1, CREDIT_SPU);
                __end_thread(SPU_ID);
            __end_plan(NEST_ID);
        }}
    }}
}}
"""
        self._layer_funcs.append(func_code)
        call_line = f"    {func_name}({in_addr}, {out_addr});  // Conv2d [{n},{ic},{ih},{iw}]→[{n},{oc},{oh},{ow}]"
        self._inference_calls.append(call_line)
        self._module_map[idx] = {
            "func_code": func_code,
            "call_line": call_line,
            "node": node,
            "func_name": func_name,
        }

    def _emit_depthwise_conv2d(self, node, idx):
        """
        Depthwise Conv2d — __im2col_d 사용, 채널별 독립 컨볼루션.
        groups == in_channels인 Conv2d. 각 채널에 대해:
          1) 입력 1채널 DDR→L2
          2) L2→L1, __im2col_d, __mm (커널 kh*kw), L1→L2
          3) L2→DDR
        """
        in_shape = self._get_input_shape(node)
        out_shape = self._get_output_shape(node)
        in_addr = self._get_input_addr_expr(node, 0)
        out_addr = self._get_node_output_var(node)

        kernel_size_raw = self._get_attr_val(node, "kernel_size", [3, 3])
        stride_raw = self._get_attr_val(node, "stride", [1, 1])
        padding_raw = self._get_attr_val(node, "padding", [0, 0])
        dilation_raw = self._get_attr_val(node, "dilation", [1, 1])
        has_bias: bool = bool(self._get_attr_val(node, "bias", False))

        n: int = in_shape[0] if len(in_shape) > 0 else 1
        ic: int = in_shape[1] if len(in_shape) > 1 else 1
        ih: int = in_shape[2] if len(in_shape) > 2 else 1
        iw: int = in_shape[3] if len(in_shape) > 3 else 1
        oc: int = out_shape[1] if len(out_shape) > 1 else 1
        oh: int = out_shape[2] if len(out_shape) > 2 else 1
        ow: int = out_shape[3] if len(out_shape) > 3 else 1
        kh: int = int(
            kernel_size_raw[0]
            if isinstance(kernel_size_raw, (list, tuple))
            else kernel_size_raw
        )
        kw: int = int(
            kernel_size_raw[1]
            if isinstance(kernel_size_raw, (list, tuple))
            else kernel_size_raw
        )
        sh: int = int(
            stride_raw[0] if isinstance(stride_raw, (list, tuple)) else stride_raw
        )
        sw: int = int(
            stride_raw[1] if isinstance(stride_raw, (list, tuple)) else stride_raw
        )
        ph: int = int(
            padding_raw[0] if isinstance(padding_raw, (list, tuple)) else padding_raw
        )
        pw: int = int(
            padding_raw[1] if isinstance(padding_raw, (list, tuple)) else padding_raw
        )
        dh: int = int(
            dilation_raw[0] if isinstance(dilation_raw, (list, tuple)) else dilation_raw
        )
        dw: int = int(
            dilation_raw[1] if isinstance(dilation_raw, (list, tuple)) else dilation_raw
        )

        weight_name = self._find_weight_name(node, "weights")
        weight_ddr_macro = weight_name.upper() + "_DDR_ADDR"

        # Depthwise: 채널별 커널 크기 = kh * kw (groups == channels)
        kernel_elems: int = kh * kw
        kernel_bytes: int = kernel_elems * 2

        # 패딩된 입력 크기 (1채널 기준)
        padded_ih: int = sh * (oh - 1) + (
            kh if dh == 1 else (2 * dh + 1 if dh <= 3 else kh)
        )
        padded_iw: int = sw * (ow - 1) + (
            kw if dw == 1 else (2 * dw + 1 if dw <= 3 else kw)
        )
        needs_padding: bool = (padded_ih != ih) or (padded_iw != iw)

        in_ch_bytes: int = ih * iw * 2
        out_ch_bytes: int = oh * ow * 2
        padded_ch_bytes: int = padded_ih * padded_iw * 2
        im2col_elems: int = oh * ow * kh * kw  # im2col_d 결과: M x K (M=oh*ow, K=kh*kw)

        func_name = f"module_{idx}_dwconv2d"

        func_code = f"""
// Module {idx}: Depthwise Conv2d — {node.name}
// in:[{n},{ic},{ih},{iw}] → out:[{n},{oc},{oh},{ow}] k:[{kh},{kw}] s:[{sh},{sw}] p:[{ph},{pw}] d:[{dh},{dw}]
// groups={ic} (depthwise), kernel_elems={kernel_elems}
static void {func_name}(uint64_t in_ddr, uint64_t out_ddr) {{
    uint16_t channels = {ic};
    uint32_t in_ch_bytes  = {in_ch_bytes};
    uint32_t out_ch_bytes = {out_ch_bytes};
    uint32_t kernel_bytes = {kernel_bytes};

    for (uint16_t c = 0; c < channels; c++) {{
        uint64_t ch_in   = in_ddr  + (uint64_t)c * in_ch_bytes;
        uint64_t ch_out  = out_ddr + (uint64_t)c * out_ch_bytes;
        uint64_t ch_wt   = {weight_ddr_macro} + (uint64_t)c * kernel_bytes;
"""

        if needs_padding:
            func_code += f"""
        // --- 패딩 입력 준비 (1채널) ---
        __start_plan(NEST_ID);
            __start_shared();
                __fill(L2_IN, {padded_iw} * 2, {padded_iw} * 2, {padded_ih}, 0, 0);
                {{
                    uint32_t dst_off = ((uint32_t){ph} * {padded_iw} + {pw}) * 2;
                    __load_cr(ch_in, L2_IN + dst_off,
                              {iw} * 2, {iw} * 2, {ih}, {padded_iw} * 2,
                              0, 0, 0);
                }}
            __end_shared();
            __start_thread(SPU_ID);
            __end_thread(SPU_ID);
        __end_plan(NEST_ID);
"""
        else:
            func_code += f"""
        // --- 입력 1채널 DDR → L2 (패딩 불필요) ---
        __start_plan(NEST_ID);
            __start_shared();
                __load_cr(ch_in, L2_IN,
                          in_ch_bytes, in_ch_bytes, 1, in_ch_bytes,
                          0, 0, 0);
            __end_shared();
            __start_thread(SPU_ID);
            __end_thread(SPU_ID);
        __end_plan(NEST_ID);
"""

        l2_input_size = padded_ch_bytes if needs_padding else in_ch_bytes
        func_code += f"""
        // --- im2col_d + mm: L2→L1, compute, L1→L2→DDR ---
        __start_plan(NEST_ID);
            __start_shared();
                // 입력 타일 L2 → L1
                __load_cr(L2_IN, 0,
                          {l2_input_size}, {l2_input_size}, 1, {l2_input_size},
                          1, CREDIT_SPU, CREDIT_NEST);
                // 가중치(kh*kw) DDR → L2
                __load_cr(ch_wt, L2_WEIGHT,
                          kernel_bytes, kernel_bytes, 1, kernel_bytes,
                          0, 0, 0);
                // Thread 완료 대기 후 결과 L2 → DDR
                __credit_chk(CREDIT_SPU);
                __store_cr(L2_OUT, ch_out,
                           out_ch_bytes, out_ch_bytes, 1, out_ch_bytes,
                           1, CREDIT_SPU);
            __end_shared();
            __start_thread(SPU_ID);
                __init_spm_addr(0xF);
                // 입력 → Bank A/R 영역
                __load_cr(0, BANK_R,
                          {l2_input_size}, {l2_input_size}, 1, {l2_input_size},
                          1, CREDIT_SPU, CREDIT_NEST);
                // __im2col_d: 1채널 depthwise im2col
                // (row_A, col_A, kernel, dilation, stride, num_of_channel=1)
                __im2col_d({padded_ih}, {padded_iw}, {kh}, {dh}, {sh}, 1);
                // 가중치 L2 → L1 Bank B
                __load_cr(L2_WEIGHT, BANK_B,
                          kernel_bytes, kernel_bytes, 1, kernel_bytes,
                          0, 0, 0);
                // mm: im2col결과(M x K) x weight(K x 1) → result(M x 1)
                // M = oh*ow, K = kh*kw
                __mm({oh} * {ow}, {kernel_elems}, 1);
                // 결과 (Bank C 또는 R) → L2
                __store_cr(BANK_C, L2_OUT,
                           out_ch_bytes, out_ch_bytes, 1, out_ch_bytes,
                           1, CREDIT_SPU);
            __end_thread(SPU_ID);
        __end_plan(NEST_ID);
    }}
}}
"""
        self._layer_funcs.append(func_code)
        call_line = f"    {func_name}({in_addr}, {out_addr});  // DWConv2d [{n},{ic},{ih},{iw}]→[{n},{oc},{oh},{ow}]"
        self._inference_calls.append(call_line)
        self._module_map[idx] = {
            "func_code": func_code,
            "call_line": call_line,
            "node": node,
            "func_name": func_name,
        }

    def _emit_maxpool(self, node, idx):
        """MaxPool2d — __pool_m: 채널별 L2 경유"""
        in_shape = self._get_input_shape(node)
        out_shape = self._get_output_shape(node)
        in_addr = self._get_input_addr_expr(node, 0)
        out_addr = self._get_node_output_var(node)

        kernel_size_raw = self._get_attr_val(node, "kernel_size", [2, 2])
        stride_raw = self._get_attr_val(node, "stride", [2, 2])
        padding_raw = self._get_attr_val(node, "padding", [0, 0])

        channels: int = in_shape[1] if len(in_shape) > 1 else 1
        ih: int = in_shape[2] if len(in_shape) > 2 else 1
        iw: int = in_shape[3] if len(in_shape) > 3 else 1
        oh: int = out_shape[2] if len(out_shape) > 2 else 1
        ow: int = out_shape[3] if len(out_shape) > 3 else 1
        kh: int = int(
            kernel_size_raw[0]
            if isinstance(kernel_size_raw, (list, tuple))
            else kernel_size_raw
        )
        kw: int = int(
            kernel_size_raw[1]
            if isinstance(kernel_size_raw, (list, tuple))
            else kernel_size_raw
        )
        sh: int = int(
            stride_raw[0] if isinstance(stride_raw, (list, tuple)) else stride_raw
        )
        sw: int = int(
            stride_raw[1] if isinstance(stride_raw, (list, tuple)) else stride_raw
        )
        pad_h: int = int(
            padding_raw[0] if isinstance(padding_raw, (list, tuple)) else padding_raw
        )
        pad_w: int = int(
            padding_raw[1] if isinstance(padding_raw, (list, tuple)) else padding_raw
        )

        padded_h: int = sh * (oh - 1) + kh
        padded_w: int = sw * (ow - 1) + kw
        needs_padding = (padded_h != ih) or (padded_w != iw)

        func_name = f"module_{idx}_maxpool"

        func_code = f"""
// Module {idx}: MaxPool2d — {node.name}
// in:[{in_shape}] → out:[{out_shape}] k:[{kh},{kw}] s:[{sh},{sw}] p:[{pad_h},{pad_w}]
static void {func_name}(uint64_t in_ddr, uint64_t out_ddr) {{
    uint16_t channels = {channels};
    uint16_t in_h = {ih}, in_w = {iw};
    uint16_t out_h = {oh}, out_w = {ow};
    uint32_t in_ch_bytes  = (uint32_t){ih} * {iw} * 2;
    uint32_t out_ch_bytes = (uint32_t){oh} * {ow} * 2;
"""
        if needs_padding:
            func_code += f"""
    uint16_t padded_h = {padded_h}, padded_w = {padded_w};
    uint32_t padded_ch_bytes = (uint32_t)padded_h * padded_w * 2;

    for (uint16_t c = 0; c < channels; c++) {{
        uint64_t ch_in  = in_ddr  + (uint64_t)c * in_ch_bytes;
        uint64_t ch_out = out_ddr + (uint64_t)c * out_ch_bytes;

        // Plan 1: zero-fill L2, copy actual data with padding offset
        __start_plan(NEST_ID);
            __start_shared();
                __fill(L2_IN, padded_w * 2, padded_w * 2, padded_h, 0, 0);
                {{
                    uint32_t dst_off = ((uint32_t){pad_h} * padded_w + {pad_w}) * 2;
                    __load_cr(ch_in, L2_IN + dst_off,
                              {iw} * 2, {iw} * 2, in_h, padded_w * 2,
                              1, CREDIT_SPU, CREDIT_NEST);
                }}
            __end_shared();
            __start_thread(SPU_ID);
            __end_thread(SPU_ID);
        __end_plan(NEST_ID);

        // Plan 2: L2→L1, pool_m, L1→L2→DDR
        __start_plan(NEST_ID);
            __start_shared();
                __load_cr(L2_IN, 0,
                          padded_ch_bytes, padded_ch_bytes, 1, padded_ch_bytes,
                          1, CREDIT_SPU, CREDIT_NEST);
                __credit_chk(CREDIT_SPU);
                __store_cr(L2_OUT, ch_out,
                           out_ch_bytes, out_ch_bytes, 1, out_ch_bytes,
                           1, CREDIT_SPU);
            __end_shared();
            __start_thread(SPU_ID);
                __init_spm_addr(0xF);
                __load_cr(0, BANK_A,
                          padded_ch_bytes, padded_ch_bytes, 1, padded_ch_bytes,
                          1, CREDIT_SPU, CREDIT_NEST);
                __pool_m(padded_h, padded_w, out_h, out_w, {kh}, {kw}, {sh}, {sw});
                __store_cr(BANK_C, L2_OUT,
                           out_ch_bytes, out_ch_bytes, 1, out_ch_bytes,
                           1, CREDIT_SPU);
            __end_thread(SPU_ID);
        __end_plan(NEST_ID);
    }}
}}
"""
        else:
            func_code += f"""
    for (uint16_t c = 0; c < channels; c++) {{
        uint64_t ch_in  = in_ddr  + (uint64_t)c * in_ch_bytes;
        uint64_t ch_out = out_ddr + (uint64_t)c * out_ch_bytes;

        __start_plan(NEST_ID);
            __start_shared();
                __load_cr(ch_in, L2_IN,
                          in_ch_bytes, in_ch_bytes, 1, in_ch_bytes,
                          1, CREDIT_SPU, CREDIT_NEST);
                __credit_chk(CREDIT_SPU);
                __store_cr(L2_OUT, ch_out,
                           out_ch_bytes, out_ch_bytes, 1, out_ch_bytes,
                           1, CREDIT_SPU);
            __end_shared();
            __start_thread(SPU_ID);
                __init_spm_addr(0xF);
                __load_cr(L2_IN, BANK_A,
                          in_ch_bytes, in_ch_bytes, 1, in_ch_bytes,
                          1, CREDIT_SPU, CREDIT_NEST);
                __pool_m(in_h, in_w, out_h, out_w, {kh}, {kw}, {sh}, {sw});
                __store_cr(BANK_C, L2_OUT,
                           out_ch_bytes, out_ch_bytes, 1, out_ch_bytes,
                           1, CREDIT_SPU);
            __end_thread(SPU_ID);
        __end_plan(NEST_ID);
    }}
}}
"""
        self._layer_funcs.append(func_code)
        call_line = f"    {func_name}({in_addr}, {out_addr});"
        self._inference_calls.append(call_line)
        self._module_map[idx] = {
            "func_code": func_code,
            "call_line": call_line,
            "node": node,
            "func_name": func_name,
        }

    def _emit_avgpool(self, node, idx):
        """AvgPool2d — __pool_a: 채널별 L2 경유"""
        in_shape = self._get_input_shape(node)
        out_shape = self._get_output_shape(node)
        in_addr = self._get_input_addr_expr(node, 0)
        out_addr = self._get_node_output_var(node)

        kernel_size_raw = self._get_attr_val(node, "kernel_size", [2, 2])
        stride_raw = self._get_attr_val(node, "stride", [2, 2])
        channels: int = in_shape[1] if len(in_shape) > 1 else 1
        ih: int = in_shape[2] if len(in_shape) > 2 else 1
        iw: int = in_shape[3] if len(in_shape) > 3 else 1
        oh: int = out_shape[2] if len(out_shape) > 2 else 1
        ow: int = out_shape[3] if len(out_shape) > 3 else 1
        kh: int = int(
            kernel_size_raw[0]
            if isinstance(kernel_size_raw, (list, tuple))
            else kernel_size_raw
        )
        kw: int = int(
            kernel_size_raw[1]
            if isinstance(kernel_size_raw, (list, tuple))
            else kernel_size_raw
        )
        sh: int = int(
            stride_raw[0] if isinstance(stride_raw, (list, tuple)) else stride_raw
        )
        sw: int = int(
            stride_raw[1] if isinstance(stride_raw, (list, tuple)) else stride_raw
        )
        k_value = self._float_to_fp16_hex(1.0 / (kh * kw))

        func_name = f"module_{idx}_avgpool"
        func_code = f"""
// Module {idx}: AvgPool2d — {node.name}
static void {func_name}(uint64_t in_ddr, uint64_t out_ddr) {{
    uint16_t channels = {channels};
    uint32_t in_ch_bytes  = (uint32_t){ih} * {iw} * 2;
    uint32_t out_ch_bytes = (uint32_t){oh} * {ow} * 2;
    uint16_t k_value = 0x{k_value:04X};

    for (uint16_t c = 0; c < channels; c++) {{
        uint64_t ch_in  = in_ddr  + (uint64_t)c * in_ch_bytes;
        uint64_t ch_out = out_ddr + (uint64_t)c * out_ch_bytes;

        __start_plan(NEST_ID);
            __start_shared();
                __load_cr(ch_in, L2_IN, in_ch_bytes, in_ch_bytes, 1, in_ch_bytes, 1, CREDIT_SPU, CREDIT_NEST);
                __credit_chk(CREDIT_SPU);
                __store_cr(L2_OUT, ch_out, out_ch_bytes, out_ch_bytes, 1, out_ch_bytes, 1, CREDIT_SPU);
            __end_shared();
            __start_thread(SPU_ID);
                __init_spm_addr(0xF);
                __load_cr(L2_IN, BANK_A, in_ch_bytes, in_ch_bytes, 1, in_ch_bytes, 1, CREDIT_SPU, CREDIT_NEST);
                __pool_a({ih}, {iw}, {oh}, {ow}, {kh}, {kw}, {sh}, {sw}, k_value);
                __store_cr(BANK_C, L2_OUT, out_ch_bytes, out_ch_bytes, 1, out_ch_bytes, 1, CREDIT_SPU);
            __end_thread(SPU_ID);
        __end_plan(NEST_ID);
    }}
}}
"""
        self._layer_funcs.append(func_code)
        call_line = f"    {func_name}({in_addr}, {out_addr});"
        self._inference_calls.append(call_line)
        self._module_map[idx] = {
            "func_code": func_code,
            "call_line": call_line,
            "node": node,
            "func_name": func_name,
        }

    def _emit_adaptive_avgpool(self, node, idx):
        """AdaptiveAvgPool2d — kernel/stride 자동 계산 후 __pool_a"""
        in_shape = self._get_input_shape(node)
        out_shape = self._get_output_shape(node)
        in_addr = self._get_input_addr_expr(node, 0)
        out_addr = self._get_node_output_var(node)

        output_size_raw = self._get_attr_val(node, "output_size", [1, 1])
        channels: int = in_shape[1] if len(in_shape) > 1 else 1
        ih: int = in_shape[2] if len(in_shape) > 2 else 1
        iw: int = in_shape[3] if len(in_shape) > 3 else 1
        oh_t: int = int(
            output_size_raw[0]
            if isinstance(output_size_raw, (list, tuple))
            else output_size_raw
        )
        ow_t: int = int(
            output_size_raw[1]
            if isinstance(output_size_raw, (list, tuple))
            else output_size_raw
        )
        kh: int = ih // oh_t
        kw: int = iw // ow_t
        k_value = self._float_to_fp16_hex(1.0 / (kh * kw))

        func_name = f"module_{idx}_adaptive_avgpool"
        func_code = f"""
// Module {idx}: AdaptiveAvgPool2d — {node.name}
// [{in_shape}] → [{out_shape}], auto kernel=[{kh},{kw}]
static void {func_name}(uint64_t in_ddr, uint64_t out_ddr) {{
    uint16_t channels = {channels};
    uint32_t in_ch_bytes  = (uint32_t){ih} * {iw} * 2;
    uint32_t out_ch_bytes = (uint32_t){oh_t} * {ow_t} * 2;
    uint16_t k_value = 0x{k_value:04X};

    for (uint16_t c = 0; c < channels; c++) {{
        uint64_t ch_in  = in_ddr  + (uint64_t)c * in_ch_bytes;
        uint64_t ch_out = out_ddr + (uint64_t)c * out_ch_bytes;

        __start_plan(NEST_ID);
            __start_shared();
                __load_cr(ch_in, L2_IN, in_ch_bytes, in_ch_bytes, 1, in_ch_bytes, 1, CREDIT_SPU, CREDIT_NEST);
                __credit_chk(CREDIT_SPU);
                __store_cr(L2_OUT, ch_out, out_ch_bytes, out_ch_bytes, 1, out_ch_bytes, 1, CREDIT_SPU);
            __end_shared();
            __start_thread(SPU_ID);
                __init_spm_addr(0xF);
                __load_cr(L2_IN, BANK_A, in_ch_bytes, in_ch_bytes, 1, in_ch_bytes, 1, CREDIT_SPU, CREDIT_NEST);
                __pool_a({ih}, {iw}, {oh_t}, {ow_t}, {kh}, {kw}, {kh}, {kw}, k_value);
                __store_cr(BANK_C, L2_OUT, out_ch_bytes, out_ch_bytes, 1, out_ch_bytes, 1, CREDIT_SPU);
            __end_thread(SPU_ID);
        __end_plan(NEST_ID);
    }}
}}
"""
        self._layer_funcs.append(func_code)
        call_line = f"    {func_name}({in_addr}, {out_addr});"
        self._inference_calls.append(call_line)
        self._module_map[idx] = {
            "func_code": func_code,
            "call_line": call_line,
            "node": node,
            "func_name": func_name,
        }

    def _emit_elemwise_add(self, node, idx):
        """Elementwise Add — __add_vv: A+B→C (via L2)"""
        shape = self._get_output_shape(node)
        in_addr_a = self._get_input_addr_expr(node, 0)
        in_addr_b = self._get_input_addr_expr(node, 1)
        out_addr = self._get_node_output_var(node)
        total_elems = 1
        for s in shape:
            total_elems *= s

        func_name = f"module_{idx}_add"
        func_code = f"""
// Module {idx}: Elementwise Add — {node.name}
// total: {total_elems} elems
static void {func_name}(uint64_t in_ddr_a, uint64_t in_ddr_b, uint64_t out_ddr) {{
    uint32_t total_elems = {total_elems};
    uint32_t max_elems = L1_BANK_B_SIZE / 2;

    for (uint32_t off = 0; off < total_elems; off += max_elems) {{
        uint32_t cur = (off + max_elems > total_elems) ? (total_elems - off) : max_elems;
        uint32_t cur_bytes = cur * 2;

        // L2 layout: L2_IN=operand A, L2_WEIGHT=operand B, L2_OUT=result
        __start_plan(NEST_ID);
            __start_shared();
                __load_cr(in_ddr_a + (uint64_t)off * 2, L2_IN,     cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU, CREDIT_NEST);
                __load_cr(in_ddr_b + (uint64_t)off * 2, L2_WEIGHT, cur_bytes, cur_bytes, 1, cur_bytes, 0, 0, 0);
                __credit_chk(CREDIT_SPU);
                __store_cr(L2_OUT, out_ddr + (uint64_t)off * 2, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU);
            __end_shared();
            __start_thread(SPU_ID);
                __init_spm_addr(0xF);
                __load_cr(L2_IN,     BANK_A, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU, CREDIT_NEST);
                __load_cr(L2_WEIGHT, BANK_B, cur_bytes, cur_bytes, 1, cur_bytes, 0, 0, 0);
                __add_vv(cur);
                __store_cr(BANK_C, L2_OUT, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU);
            __end_thread(SPU_ID);
        __end_plan(NEST_ID);
    }}
}}
"""
        self._layer_funcs.append(func_code)
        call_line = f"    {func_name}({in_addr_a}, {in_addr_b}, {out_addr});"
        self._inference_calls.append(call_line)
        self._module_map[idx] = {
            "func_code": func_code,
            "call_line": call_line,
            "node": node,
            "func_name": func_name,
        }

    def _emit_elemwise_mul(self, node, idx):
        """Elementwise Mul — __mul_vv: A*B→C (via L2)"""
        shape = self._get_output_shape(node)
        in_addr_a = self._get_input_addr_expr(node, 0)
        in_addr_b = self._get_input_addr_expr(node, 1)
        out_addr = self._get_node_output_var(node)
        total_elems = 1
        for s in shape:
            total_elems *= s

        func_name = f"module_{idx}_mul"
        func_code = f"""
// Module {idx}: Elementwise Mul — {node.name}
static void {func_name}(uint64_t in_ddr_a, uint64_t in_ddr_b, uint64_t out_ddr) {{
    uint32_t total_elems = {total_elems};
    uint32_t max_elems = L1_BANK_B_SIZE / 2;
    for (uint32_t off = 0; off < total_elems; off += max_elems) {{
        uint32_t cur = (off + max_elems > total_elems) ? (total_elems - off) : max_elems;
        uint32_t cur_bytes = cur * 2;
        __start_plan(NEST_ID);
            __start_shared();
                __load_cr(in_ddr_a + (uint64_t)off * 2, L2_IN,     cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU, CREDIT_NEST);
                __load_cr(in_ddr_b + (uint64_t)off * 2, L2_WEIGHT, cur_bytes, cur_bytes, 1, cur_bytes, 0, 0, 0);
                __credit_chk(CREDIT_SPU);
                __store_cr(L2_OUT, out_ddr + (uint64_t)off * 2, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU);
            __end_shared();
            __start_thread(SPU_ID);
                __init_spm_addr(0xF);
                __load_cr(L2_IN,     BANK_A, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU, CREDIT_NEST);
                __load_cr(L2_WEIGHT, BANK_B, cur_bytes, cur_bytes, 1, cur_bytes, 0, 0, 0);
                __mul_vv(cur);
                __store_cr(BANK_C, L2_OUT, cur_bytes, cur_bytes, 1, cur_bytes, 1, CREDIT_SPU);
            __end_thread(SPU_ID);
        __end_plan(NEST_ID);
    }}
}}
"""
        self._layer_funcs.append(func_code)
        call_line = f"    {func_name}({in_addr_a}, {in_addr_b}, {out_addr});"
        self._inference_calls.append(call_line)
        self._module_map[idx] = {
            "func_code": func_code,
            "call_line": call_line,
            "node": node,
            "func_name": func_name,
        }

    def _emit_dense(self, node, idx):
        """Dense (Linear) — __mm via L2, with optional bias __add_vv"""
        in_shape = self._get_input_shape(node)
        out_shape = self._get_output_shape(node)
        in_addr = self._get_input_addr_expr(node, 0)
        out_addr = self._get_node_output_var(node)

        in_features: int = int(
            self._get_attr_val(node, "in_features", in_shape[-1] if in_shape else 1)
        )
        out_features: int = int(
            self._get_attr_val(node, "out_features", out_shape[-1] if out_shape else 1)
        )
        has_bias: bool = bool(self._get_attr_val(node, "bias", True))

        weight_name = self._find_weight_name(node, "weights")
        bias_name = self._find_weight_name(node, "bias") if has_bias else None
        weight_ddr_macro = weight_name.upper() + "_DDR_ADDR"
        bias_ddr_macro = (bias_name.upper() + "_DDR_ADDR") if bias_name else None

        func_name = f"module_{idx}_dense"

        bias_code = ""
        if bias_name:
            bias_code = f"""
        // Bias 추가: DDR→L2→L1, add_vv
        __start_plan(NEST_ID);
            __start_shared();
                __load_cr({bias_ddr_macro} + (uint64_t)of_s * 2, L2_WEIGHT,
                          cur_of * 2, cur_of * 2, 1, cur_of * 2, 1, CREDIT_SPU, CREDIT_NEST);
                // 결과(C+bias)를 L2→DDR
                __credit_chk(CREDIT_SPU);
                __store_cr(L2_OUT, out_ddr + (uint64_t)of_s * 2,
                           cur_of * 2, cur_of * 2, 1, cur_of * 2, 1, CREDIT_SPU);
            __end_shared();
            __start_thread(SPU_ID);
                // bias → Bank B
                __load_cr(L2_WEIGHT, BANK_B,
                          cur_of * 2, cur_of * 2, 1, cur_of * 2, 1, CREDIT_SPU, CREDIT_NEST);
                // mm 결과는 Bank C에 남아있음
                __set_spm_addr_A(BANK_C);
                __set_spm_addr_B(BANK_B);
                __add_vv(cur_of);
                // add_vv 결과: C (여기서는 원래 A+B→R? 아니면 C?)
                // add_vv: Bank A + Bank B → Bank R이므로
                // 위에서 set_spm_addr_A(BANK_C) 했으므로 A=C, B=B → R
                __store_cr(BANK_R, L2_OUT,
                           cur_of * 2, cur_of * 2, 1, cur_of * 2, 1, CREDIT_SPU);
            __end_thread(SPU_ID);
        __end_plan(NEST_ID);"""

        no_bias_store = ""
        if not has_bias:
            no_bias_store = f"""
                // 결과 저장 (bias 없음)
                __store_cr(BANK_C, L2_OUT,
                           cur_of * 2, cur_of * 2, 1, cur_of * 2, 1, CREDIT_SPU);"""

        func_code = f"""
// Module {idx}: Dense — {node.name}
// [{in_features}] → [{out_features}]
static void {func_name}(uint64_t in_ddr, uint64_t out_ddr) {{
    uint16_t in_feat  = {in_features};
    uint16_t out_feat = {out_features};
    uint32_t in_bytes = (uint32_t)in_feat * 2;

    // 입력 벡터를 L2로 한 번 로드
    __start_plan(NEST_ID);
        __start_shared();
            __load_cr(in_ddr, L2_IN, in_bytes, in_bytes, 1, in_bytes, 0, 0, 0);
        __end_shared();
        __start_thread(SPU_ID);
        __end_thread(SPU_ID);
    __end_plan(NEST_ID);

    // Bank B에 맞는 최대 output feature 타일
    uint32_t w_row_bytes = in_bytes;
    uint16_t of_tile = L1_BANK_B_SIZE / w_row_bytes;
    if (of_tile < 1) of_tile = 1;
    if (of_tile > out_feat) of_tile = out_feat;

    for (uint16_t of_s = 0; of_s < out_feat; of_s += of_tile) {{
        uint16_t cur_of = (of_s + of_tile > out_feat) ? (out_feat - of_s) : of_tile;

        // mm: input x weight_tile → result_tile
        __start_plan(NEST_ID);
            __start_shared();
                // 입력 (이미 L2에 있음) — 다시 로드 불필요, Thread가 직접 사용
                // 가중치 타일 DDR → L2
                __load_cr({weight_ddr_macro} + (uint64_t)of_s * w_row_bytes, L2_WEIGHT,
                          w_row_bytes, w_row_bytes, cur_of, w_row_bytes, 1, CREDIT_SPU, CREDIT_NEST);
                __credit_chk(CREDIT_SPU);
                {"" if has_bias else f"__store_cr(L2_OUT, out_ddr + (uint64_t)of_s * 2, cur_of * 2, cur_of * 2, 1, cur_of * 2, 1, CREDIT_SPU);"}
            __end_shared();
            __start_thread(SPU_ID);
                __init_spm_addr(0xF);
                // 입력 → Bank A
                __load_cr(L2_IN, BANK_A, in_bytes, in_bytes, 1, in_bytes, 0, 0, 0);
                // 가중치 → Bank B
                __load_cr(L2_WEIGHT, BANK_B,
                          w_row_bytes, w_row_bytes, cur_of, w_row_bytes, 1, CREDIT_SPU, CREDIT_NEST);
                __mm(1, in_feat, cur_of);
{no_bias_store}
            __end_thread(SPU_ID);
        __end_plan(NEST_ID);
{bias_code}
    }}
}}
"""
        self._layer_funcs.append(func_code)
        call_line = f"    {func_name}({in_addr}, {out_addr});"
        self._inference_calls.append(call_line)
        self._module_map[idx] = {
            "func_code": func_code,
            "call_line": call_line,
            "node": node,
            "func_name": func_name,
        }

    def _emit_flatten(self, node, idx):
        """Flatten — no-op, 주소 전달"""
        if node.in_nodes:
            in_node = list(node.in_nodes)[0]
            if in_node.name in self._ddr_buffers:
                self._ddr_buffers[node.name] = self._ddr_buffers[in_node.name]
        call_line = f"    // Module {idx}: Flatten — {node.name} (no-op)"
        self._inference_calls.append(call_line)
        self._module_map[idx] = {
            "func_code": None,
            "call_line": call_line,
            "node": node,
            "func_name": None,
        }

    def _emit_reshape(self, node, idx):
        """Reshape — no-op"""
        if node.in_nodes:
            in_node = list(node.in_nodes)[0]
            if in_node.name in self._ddr_buffers:
                self._ddr_buffers[node.name] = self._ddr_buffers[in_node.name]
        call_line = f"    // Module {idx}: Reshape — {node.name} (no-op)"
        self._inference_calls.append(call_line)
        self._module_map[idx] = {
            "func_code": None,
            "call_line": call_line,
            "node": node,
            "func_name": None,
        }

    def _emit_softmax(self, node, idx):
        """Softmax — __max_vs + __esum + __softmax via L2"""
        shape = self._get_input_shape(node)
        in_addr = self._get_input_addr_expr(node, 0)
        out_addr = self._get_node_output_var(node)
        total_elems = 1
        for s in shape:
            total_elems *= s

        func_name = f"module_{idx}_softmax"
        func_code = f"""
// Module {idx}: Softmax — {node.name}
// total: {total_elems} elems
static void {func_name}(uint64_t in_ddr, uint64_t out_ddr) {{
    uint32_t total_elems = {total_elems};
    uint32_t data_bytes = total_elems * 2;

    // 전체를 L2에 로드
    __start_plan(NEST_ID);
        __start_shared();
            __load_cr(in_ddr, L2_IN, data_bytes, data_bytes, 1, data_bytes, 1, CREDIT_SPU, CREDIT_NEST);
        __end_shared();
        __start_thread(SPU_ID);
        __end_thread(SPU_ID);
    __end_plan(NEST_ID);

    // Pass 1: max 계산
    __start_plan(NEST_ID);
        __start_shared();
            __load_cr(L2_IN, 0, data_bytes, data_bytes, 1, data_bytes, 1, CREDIT_SPU, CREDIT_NEST);
        __end_shared();
        __start_thread(SPU_ID);
            __init_spm_addr(0xF);
            __load_cr(0, BANK_A, data_bytes, data_bytes, 1, data_bytes, 1, CREDIT_SPU, CREDIT_NEST);
            __max_vs(total_elems, 0xFC00, 0, 0);  // max → SVR[0]
        __end_thread(SPU_ID);
    __end_plan(NEST_ID);

    // Pass 2: esum 계산
    __start_plan(NEST_ID);
        __start_shared();
            __load_cr(L2_IN, 0, data_bytes, data_bytes, 1, data_bytes, 1, CREDIT_SPU, CREDIT_NEST);
        __end_shared();
        __start_thread(SPU_ID);
            __init_spm_addr(0xF);
            __load_cr(0, BANK_A, data_bytes, data_bytes, 1, data_bytes, 1, CREDIT_SPU, CREDIT_NEST);
            // esum(size, max_value_from_SVR0, accumulated=0, result_svr=1, r2_sel=0)
            __esum(total_elems, 0, 0, 1, 0);  // uses SVR[0] for max, result→SVR[1]
        __end_thread(SPU_ID);
    __end_plan(NEST_ID);

    // Pass 3: softmax 계산
    __start_plan(NEST_ID);
        __start_shared();
            __load_cr(L2_IN, 0, data_bytes, data_bytes, 1, data_bytes, 1, CREDIT_SPU, CREDIT_NEST);
            __credit_chk(CREDIT_SPU);
            __store_cr(L2_OUT, out_ddr, data_bytes, data_bytes, 1, data_bytes, 1, CREDIT_SPU);
        __end_shared();
        __start_thread(SPU_ID);
            __init_spm_addr(0xF);
            __load_cr(0, BANK_A, data_bytes, data_bytes, 1, data_bytes, 1, CREDIT_SPU, CREDIT_NEST);
            // softmax(size, max_from_SVR0, esum_from_SVR1, r2_sel=0)
            __softmax(total_elems, 0, 0, 0);  // uses SVR[0] max, SVR[1] esum
            __store_cr(BANK_A, L2_OUT, data_bytes, data_bytes, 1, data_bytes, 1, CREDIT_SPU);
        __end_thread(SPU_ID);
    __end_plan(NEST_ID);
}}
"""
        self._layer_funcs.append(func_code)
        call_line = f"    {func_name}({in_addr}, {out_addr});"
        self._inference_calls.append(call_line)
        self._module_map[idx] = {
            "func_code": func_code,
            "call_line": call_line,
            "node": node,
            "func_name": func_name,
        }

    def _emit_concat(self, node, idx):
        """Concat — DDR→DDR copy (via __copy_mem in Shared)"""
        out_addr = self._get_node_output_var(node)
        func_name = f"module_{idx}_concat"

        in_addrs = []
        in_sizes = []
        for i, in_node in enumerate(node.in_nodes):
            in_addrs.append(self._get_input_addr_expr(node, i))
            in_shape = (
                self._get_output_shape(in_node)
                if hasattr(in_node, "out_tensors")
                else [1]
            )
            in_sizes.append(self.mem_planner.calc_tensor_size(in_shape))

        copy_lines = []
        offset = 0
        for i, (addr, size) in enumerate(zip(in_addrs, in_sizes)):
            copy_lines.append(f"    // Copy input {i}: {size} bytes at offset {offset}")
            copy_lines.append(f"    __start_plan(NEST_ID);")
            copy_lines.append(f"        __start_shared();")
            copy_lines.append(
                f"            __copy_mem({addr}, out_ddr + {offset}, {size}, {size}, 1, 0, 0);"
            )
            copy_lines.append(f"        __end_shared();")
            copy_lines.append(f"        __start_thread(SPU_ID);")
            copy_lines.append(f"        __end_thread(SPU_ID);")
            copy_lines.append(f"    __end_plan(NEST_ID);")
            offset += size
        copy_code = "\n".join(copy_lines)

        func_code = f"""
// Module {idx}: Concat — {node.name}
static void {func_name}({", ".join(f"uint64_t in{i}" for i in range(len(in_addrs)))}, uint64_t out_ddr) {{
{copy_code}
}}
"""
        self._layer_funcs.append(func_code)
        args = ", ".join(in_addrs)
        call_line = f"    {func_name}({args}, {out_addr});"
        self._inference_calls.append(call_line)
        self._module_map[idx] = {
            "func_code": func_code,
            "call_line": call_line,
            "node": node,
            "func_name": func_name,
        }

    # ========================================
    # 파일 출력
    # ========================================

    def _write_header(self) -> str:
        filepath = os.path.join(self.output_dir, f"{self.model_name}.h")
        lines = []
        lines.append("// Auto-generated by GTX Compiler — DO NOT EDIT")
        lines.append(f"// Model: {self.graph.name}")
        lines.append("")
        lines.append(f"#ifndef {self.model_name.upper()}_H")
        lines.append(f"#define {self.model_name.upper()}_H")
        lines.append("")
        lines.append("#include <stdint.h>")
        lines.append('#include "gtx/intrin.h"')
        lines.append('#include "gtx/gtx_csr.h"')
        lines.append("")
        lines.append("// DDR 메모리 영역 정의")
        lines.append(f"#define DDR_INPUT_BASE  0x{DDR_INPUT_BASE:08X}ULL")
        lines.append(f"#define DDR_OUTPUT_BASE 0x{DDR_OUTPUT_BASE:08X}ULL")
        lines.append(f"#define DDR_WEIGHT_BASE 0x{DDR_WEIGHT_BASE:08X}ULL")
        lines.append(f"#define DDR_TEMP_BASE   0x{DDR_TEMP_BASE:08X}ULL")
        lines.append("")
        lines.append("// L2 SPM 영역 오프셋 (Shared Scope에서 사용)")
        lines.append(f"#define L2_IN     0x{L2_INPUT_OFFSET:06X}")
        lines.append(f"#define L2_WEIGHT 0x{L2_WEIGHT_OFFSET:06X}")
        lines.append(f"#define L2_OUT    0x{L2_OUTPUT_OFFSET:06X}")
        lines.append("")
        lines.append("// L1 SPM Bank 주소 (Thread Scope에서 사용)")
        lines.append(f"#define BANK_A 0x{L1SPM_BANK_A_BASE:05X}")
        lines.append(f"#define BANK_B 0x{L1SPM_BANK_B_BASE:05X}")
        lines.append(f"#define BANK_C 0x{L1SPM_BANK_C_BASE:05X}")
        lines.append(f"#define BANK_R 0x{L1SPM_BANK_R_BASE:05X}")
        lines.append("")
        lines.append("// L1 SPM Bank 크기")
        lines.append(
            f"#define L1_BANK_A_SIZE  0x{L1SPM_BANK_A_SIZE:05X}  // {L1SPM_BANK_A_SIZE // 1024}KB"
        )
        lines.append(
            f"#define L1_BANK_B_SIZE  0x{L1SPM_BANK_B_SIZE:05X}  // {L1SPM_BANK_B_SIZE // 1024}KB"
        )
        lines.append(
            f"#define L1_BANK_C_SIZE  0x{L1SPM_BANK_C_SIZE:05X}  // {L1SPM_BANK_C_SIZE // 1024}KB"
        )
        lines.append(
            f"#define L1_BANK_R_SIZE  0x{L1SPM_BANK_R_SIZE:05X}  // {L1SPM_BANK_R_SIZE // 1024}KB"
        )
        lines.append("")
        lines.append("// NEST/SPU 타겟 ID")
        lines.append(f"#define NEST_ID     {self.nest_id}")
        lines.append(f"#define SPU_ID      {self.spu_id}")
        lines.append("")
        lines.append("// Credit 타겟 비트마스크")
        lines.append(f"#define CREDIT_SPU  {self.CREDIT_SPU}")
        lines.append(f"#define CREDIT_NEST {self.CREDIT_NEST}")
        lines.append("")
        lines.append("// 추론 함수 프로토타입")
        lines.append(
            f"void {self.model_name}_inference(uint64_t input_ddr_addr, uint64_t output_ddr_addr);"
        )
        lines.append("")
        lines.append(f"#endif // {self.model_name.upper()}_H")
        lines.append("")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))
        return filepath

    def _write_implementation(self) -> str:
        filepath = os.path.join(self.output_dir, f"{self.model_name}.c")
        lines = []
        lines.append("// Auto-generated by GTX Compiler — DO NOT EDIT")
        lines.append(f"// Model: {self.graph.name}")
        lines.append(f"// Total modules: {len(self._module_map)}")
        lines.append("//")
        lines.append("// GTX Context: Plan { Shared(DDR↔L2) ∥ Thread(L2↔L1+계산) }")
        lines.append("// ========================================")
        for mod_id in sorted(self._module_map.keys()):
            entry = self._module_map[mod_id]
            node = entry["node"]
            fn = entry["func_name"] if entry["func_name"] else "(no-op)"
            lines.append(f"//   module_{mod_id:3d}: {node.op.type:<24s} — {fn}")
        lines.append("// ========================================")
        lines.append("")
        lines.append(f'#include "{self.model_name}.h"')
        lines.append('#include "weight_map.h"')
        lines.append("")

        for func_code in self._layer_funcs:
            lines.append(func_code)

        lines.append("")
        lines.append("// ========================================")
        lines.append("// 메인 추론 함수")
        lines.append("// ========================================")
        lines.append(
            f"void {self.model_name}_inference(uint64_t input_ddr_addr, uint64_t output_ddr_addr) {{"
        )
        for call in self._inference_calls:
            lines.append(call)

        # 최종 출력을 DDR_OUTPUT_BASE로 복사 (DDR_TEMP → DDR_OUTPUT)
        final_addr, final_size = self._get_final_output_info()
        if final_addr is not None and final_addr != DDR_OUTPUT_BASE:
            lines.append("")
            lines.append(f"    // 최종 출력을 output_ddr_addr로 복사")
            lines.append(f"    __start_plan(NEST_ID);")
            lines.append(f"        __start_shared();")
            lines.append(
                f"            __load_cr(0x{final_addr:08X}ULL, L2_IN, "
                f"{final_size}, {final_size}, 1, {final_size}, 1, CREDIT_SPU, CREDIT_NEST);"
            )
            lines.append(f"            __credit_chk(CREDIT_SPU);")
            lines.append(
                f"            __store_cr(L2_IN, output_ddr_addr, "
                f"{final_size}, {final_size}, 1, {final_size}, 1, CREDIT_SPU);"
            )
            lines.append(f"        __end_shared();")
            lines.append(f"        __start_thread(SPU_ID);")
            lines.append(f"        __end_thread(SPU_ID);")
            lines.append(f"    __end_plan(NEST_ID);")

        lines.append("}")
        lines.append("")

        # main 함수 — __split()/__join() 포함
        lines.append("// ========================================")
        lines.append("// main 함수 (시뮬레이터 진입점)")
        lines.append("// ========================================")
        lines.append("int main() {")
        lines.append("    // NEST 선택")
        lines.append(f"    __wrspr(NEST_SELECT, 0, NEST_ID, 0xFFFFFFFFFFFFFFFF);")
        lines.append("")
        lines.append("    // CPU → NPU 전환")
        lines.append("    __split();")
        lines.append("")
        lines.append("    // 추론 실행")
        lines.append(
            f"    {self.model_name}_inference(DDR_INPUT_BASE, DDR_OUTPUT_BASE);"
        )
        lines.append("")
        lines.append("    // NPU → CPU 복귀 (Plan 실행 완료 대기)")
        lines.append("    __join();")
        lines.append("")
        lines.append("    // 완료")
        lines.append("    __halt();")
        lines.append("    return 0;")
        lines.append("}")
        lines.append("")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))
        return filepath

    def _write_per_module_files(self) -> str:
        modules_dir = os.path.join(self.output_dir, "modules")
        os.makedirs(modules_dir, exist_ok=True)

        for mod_id in sorted(self._module_map.keys()):
            entry = self._module_map[mod_id]
            func_code = entry["func_code"]
            func_name = entry["func_name"]
            node = entry["node"]

            filepath = os.path.join(modules_dir, f"module_{mod_id}.c")
            lines = []
            lines.append("// Auto-generated by GTX Compiler — DO NOT EDIT")
            lines.append(f"// Standalone module: module_{mod_id}")
            lines.append(f"// Node: {node.name}, Op: {node.op.type}")
            lines.append("")
            lines.append(f'#include "{self.model_name}.h"')
            lines.append('#include "weight_map.h"')
            lines.append("")

            if func_code:
                lines.append(func_code)
                lines.append("")

            lines.append("int main() {")
            lines.append(f"    __wrspr(NEST_SELECT, 0, NEST_ID, 0xFFFFFFFFFFFFFFFF);")
            lines.append("    __split();")
            if func_name:
                op_type = node.op.type
                if op_type in (GTX_OP.ADD, GTX_OP.MULTIPLY):
                    lines.append(
                        f"    {func_name}(DDR_INPUT_BASE, DDR_INPUT_BASE, DDR_OUTPUT_BASE);"
                    )
                else:
                    lines.append(f"    {func_name}(DDR_INPUT_BASE, DDR_OUTPUT_BASE);")
            else:
                lines.append(f"    // module_{mod_id}: {node.name} — no-op")
            lines.append("    __join();")
            lines.append("    __halt();")
            lines.append("    return 0;")
            lines.append("}")
            lines.append("")

            with open(filepath, "w") as f:
                f.write("\n".join(lines))

        return modules_dir

    # ========================================
    # 유틸리티
    # ========================================

    def _get_attr_val(self, node, attr_name: str, default: Any = None) -> Any:
        try:
            if hasattr(node.op, "get_attr"):
                attr = node.op.get_attr(attr_name)
                if attr is not None:
                    return attr.value if hasattr(attr, "value") else attr
        except Exception:
            pass
        try:
            if hasattr(node.op, "attrs"):
                for k, v in node.op.attrs.items():
                    if str(k) == attr_name or (
                        hasattr(k, "name") and k.name == attr_name
                    ):
                        return v.value if hasattr(v, "value") else v
        except Exception:
            pass
        return default

    def _find_weight_name(self, node, param_type_str: str) -> str:
        for param_type, param_tensor in node.op.params.items():
            type_str = str(param_type).lower()
            if param_type_str.lower() in type_str:
                return self._weight_exporter_name(node, param_type)
        node_name = node.name.replace("/", "_").replace(".", "_").replace(" ", "_")
        return f"{node_name}_{param_type_str}"

    def _weight_exporter_name(self, node, param_type) -> str:
        return (
            self._make_weight_name(node, param_type)
            .replace(".", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace(" ", "_")
        )

    def _get_final_output_info(self) -> Tuple[Optional[int], int]:
        """
        최종 레이어 출력의 (DDR 주소, 바이트 크기) 반환.
        DDR_OUTPUT_BASE와 다른 주소인 경우 복사가 필요함.
        """
        # 마지막 모듈(no-op 제외)의 출력 추적
        for mid in sorted(self._module_map.keys(), reverse=True):
            node = self._module_map[mid]["node"]
            if node.name in self._ddr_buffers:
                addr = self._ddr_buffers[node.name]
                out_shape = self._get_output_shape(node)
                size = self.mem_planner.calc_tensor_size(out_shape)
                return addr, size
        return None, 0

    def _float_to_fp16_hex(self, value: float) -> int:
        fp16 = np.float16(value)
        return int(fp16.view(np.uint16))
