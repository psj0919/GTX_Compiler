# Copyright (C) Supergate - All Rights Reserved
# GTX Compiler - NumCpp C++ Code Generator
# GTX Graph → C++ 소스코드 (.cpp/.hpp) 생성 백엔드
#
# NumCpp의 GTX 가속 레이어를 사용하여 NdArray<float> 기반 추론 코드를 생성합니다.
# 내부적으로 NumCpp bridge가 float→FP16 변환, DDR staging, Plan/Shared/Thread 실행을
# 모두 자동 처리하므로, 생성 코드는 고수준 NumCpp API 호출만 포함합니다.

import os
import re
import math
import struct
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from gtx_shared.base.key_names import GTX_OP
from gtx_shared.compile.memory_planner import (
    DDR_INPUT_BASE,
    DDR_OUTPUT_BASE,
    DDR_TEMP_BASE,
)
from gtx_shared.compile.weight_exporter import WeightExporter, float32_to_fp16_hex


class CppCodeGenerator:
    """
    GTX Graph를 NumCpp C++ 소스코드로 변환하는 코드 생성기.

    NumCpp GTX 가속 레이어 사용:
      - NdArray<float> 기반 텐서 연산
      - nc::gtx::activation / norm / ops / pool / conv 네임스페이스
      - 내부적으로 DDR staging + Plan/Shared/Thread + Credit 동기화 자동 처리
    """

    def __init__(
        self,
        graph,
        output_dir: str = "./output",
        model_name: str = "model",
        numcpp_root: Optional[str] = None,
    ):
        self.graph = graph
        self.output_dir = output_dir
        self.model_name = model_name

        # NumCpp 프로젝트 루트 경로 (상대 또는 절대)
        if numcpp_root is None:
            # 기본값: 프로젝트 루트의 NumCpp/ 디렉토리
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            self.numcpp_root = os.path.join(project_root, "NumCpp")
        else:
            self.numcpp_root = numcpp_root

        self.weight_exporter = WeightExporter(output_dir)

        # 텐서 변수 이름 추적
        self._tensor_vars: Dict[str, str] = {}  # node.name → C++ 변수명
        self._tensor_shapes: Dict[str, List[int]] = {}  # node.name → shape
        self._weight_names: Dict[str, str] = {}  # weight_key → C++ 변수명

        self._impl_lines: List[str] = []
        self._weight_load_lines: List[str] = []
        self._inference_lines: List[str] = []
        self._layer_count = 0

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
        """전체 C++ 코드 생성 진입점."""
        os.makedirs(self.output_dir, exist_ok=True)

        self._analyze_graph()

        self.weight_exporter.export_header()
        self.weight_exporter.export_binary()
        self.weight_exporter.export_address_map()

        self._generate_layer_code()

        impl_path = self._write_implementation()

        return {
            "implementation": impl_path,
            "weights": os.path.join(self.output_dir, "weights.h"),
            "weight_map": os.path.join(self.output_dir, "weight_map.h"),
            "weights_bin": os.path.join(self.output_dir, "weights.bin"),
        }

    # ========================================
    # 그래프 분석
    # ========================================

    def _analyze_graph(self):
        """그래프를 순회하여 가중치를 수집"""
        for node in self.graph.nodes:
            if node.op.type == GTX_OP.RETURN:
                continue

            # 가중치 수집
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

            # 출력 shape 기록
            if node.out_tensors:
                out_shape = node.out_tensors[0].shape
                if out_shape:
                    self._tensor_shapes[node.name] = list(out_shape)

    def _make_weight_name(self, node, param_type) -> str:
        node_name = node.name.replace("/", "_").replace(".", "_").replace(" ", "_")
        param_str = str(param_type).replace(".", "_")
        return f"{node_name}_{param_str}"

    def _make_var_name(self, node_name: str) -> str:
        """노드 이름 → C++ 변수명"""
        return re.sub(r"[^a-zA-Z0-9_]", "_", node_name)

    def _get_input_var(self, node, input_idx: int = 0) -> str:
        """노드의 입력 텐서 변수명 반환"""
        if not node.in_nodes:
            return "input"
        in_nodes_list = list(node.in_nodes)
        if input_idx < len(in_nodes_list):
            in_node = in_nodes_list[input_idx]
            if in_node.name in self._tensor_vars:
                return self._tensor_vars[in_node.name]
        return "input"

    def _get_input_shape(self, node) -> List[int]:
        if node.in_tensors and node.in_tensors[0].shape:
            return list(node.in_tensors[0].shape)
        return [1]

    def _get_output_shape(self, node) -> List[int]:
        if node.out_tensors and node.out_tensors[0].shape:
            return list(node.out_tensors[0].shape)
        return [1]

    def _total_elements(self, shape: List[int]) -> int:
        result = 1
        for s in shape:
            result *= s
        return result

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
            handler = self._op_handlers.get(node.op.type)
            if handler:
                handler(node, fallback_idx)
            else:
                self._inference_lines.append(
                    f"    // WARNING: Unsupported op '{node.op.type}' for '{node.name}'"
                )
            fallback_idx += 1

    # ========================================
    # Op별 코드 생성 — NumCpp API 호출
    # ========================================

    def _emit_input(self, node, idx):
        out_shape = self._get_output_shape(node)
        var = self._make_var_name(node.name)
        self._tensor_vars[node.name] = var
        # input은 함수 파라미터로 전달되므로 별도 선언 불필요
        # 추론 함수 시작 시 input_data에서 NdArray 생성
        rows = out_shape[0] if len(out_shape) >= 1 else 1
        cols = self._total_elements(out_shape) // rows if rows > 0 else 1
        self._inference_lines.append(
            f"    // [{idx}] INPUT: {node.name} — shape {out_shape}"
        )
        self._inference_lines.append(
            f"    // Input data loaded from weights.bin at DDR_INPUT_BASE"
        )

    def _emit_relu(self, node, idx):
        in_var = self._get_input_var(node)
        out_var = self._make_var_name(node.name)
        out_shape = self._get_output_shape(node)
        self._tensor_vars[node.name] = out_var
        total = self._total_elements(out_shape)

        self._inference_lines.append(f"    // [{idx}] ReLU: {node.name}")
        self._inference_lines.append(
            f"    nc::gtx::activation::relu("
            f"L2_INPUT, L2_RESULT, {total}, 1, NEST_ID, NUM_SPUS);"
        )

    def _emit_relu6(self, node, idx):
        in_var = self._get_input_var(node)
        out_var = self._make_var_name(node.name)
        out_shape = self._get_output_shape(node)
        self._tensor_vars[node.name] = out_var
        total = self._total_elements(out_shape)

        self._inference_lines.append(f"    // [{idx}] ReLU6: {node.name}")
        self._inference_lines.append(
            f"    nc::gtx::activation::relu6("
            f"L2_INPUT, L2_RESULT, {total}, 1, NEST_ID, NUM_SPUS);"
        )

    def _emit_leaky_relu(self, node, idx):
        in_var = self._get_input_var(node)
        out_var = self._make_var_name(node.name)
        out_shape = self._get_output_shape(node)
        self._tensor_vars[node.name] = out_var
        total = self._total_elements(out_shape)

        attrs = node.op.attrs if hasattr(node.op, "attrs") else {}
        negative_slope = attrs.get("negative_slope", 0.01)
        alpha_fp16 = float32_to_fp16_hex(negative_slope)

        # LeakyReLU(x) = max(0,x) + alpha*min(0,x)
        # Implementation: relu(x) → R1; neg(x) → relu(-x) → R2; alpha*R2 → R3; R1 - R3
        # Simplified: Use relu as approximation for small alpha (common case: 0.01)
        # For correctness, we emit a full Plan that computes:
        #   1) relu(input) → L2_RESULT
        #   2) neg(input) → relu → mul_vs(alpha) → temp
        #   3) result = relu_result - temp (via sub_vv)
        self._inference_lines.append(
            f"    // [{idx}] LeakyReLU: {node.name} — alpha={negative_slope}"
        )
        self._inference_lines.append(f"    {{")
        self._inference_lines.append(f"        // Step 1: relu(x) → L2_RESULT")
        self._inference_lines.append(
            f"        nc::gtx::activation::relu("
            f"L2_INPUT, L2_RESULT, {total}, 1, NEST_ID, NUM_SPUS);"
        )
        self._inference_lines.append(
            f"        // Step 2: neg(x) → relu → scale by alpha → L2_WEIGHT (temp)"
        )
        self._inference_lines.append(f"        __split();")
        self._inference_lines.append(f"        __start_p(NEST_ID);")
        self._inference_lines.append(f"            __start_s();")
        self._inference_lines.append(
            f"                __load_cr(DDR_STAGING_IN, 0, {total * 2}, {total * 2}, 1, {total * 2},"
        )
        self._inference_lines.append(f"                          1, 0x1, 0xDEAD);")
        self._inference_lines.append(f"                __credit_chk(0x1);")
        self._inference_lines.append(
            f"                __store_cr(L2_WEIGHT, DDR_STAGING_B, {total * 2}, {total * 2}, 1, {total * 2},"
        )
        self._inference_lines.append(f"                           1, 0x1);")
        self._inference_lines.append(f"            __end_s();")
        self._inference_lines.append(f"            __start_t(0);")
        self._inference_lines.append(f"                nc::gtx::reset_l1_banks();")
        self._inference_lines.append(
            f"                __load_cr(0, nc::gtx::L1_BANK_A_BASE, {total * 2}, {total * 2}, 1, {total * 2},"
        )
        self._inference_lines.append(f"                          1, 0xDEAD, 0xDEAD);")
        self._inference_lines.append(
            f"                __neg_v({total});"
        )  # neg(x) → Bank R
        self._inference_lines.append(
            f"                // Move neg result to Bank A for relu"
        )
        self._inference_lines.append(
            f"                __set_spm_addr_A(nc::gtx::L1_BANK_R_BASE);"
        )
        self._inference_lines.append(
            f"                __relu({total}, nc::gtx::L1_BANK_A_BASE, nc::gtx::L1_BANK_R_BASE);"
        )
        self._inference_lines.append(
            f"                // relu(-x) in Bank A; scale by alpha"
        )
        self._inference_lines.append(
            f"                __mul_vs({total}, 0x{alpha_fp16:04X});"
        )  # Bank A * alpha → Bank R
        self._inference_lines.append(
            f"                __store_cr(nc::gtx::L1_BANK_R_BASE, L2_WEIGHT, {total * 2}, {total * 2}, 1, {total * 2},"
        )
        self._inference_lines.append(f"                           1, 0xDEAD);")
        self._inference_lines.append(f"            __end_t(0);")
        self._inference_lines.append(f"        __end_p(NEST_ID);")
        self._inference_lines.append(f"        __join();")
        self._inference_lines.append(
            f"        // Step 3: result = relu(x) - alpha*relu(-x)"
        )
        self._inference_lines.append(
            f"        nc::gtx::ops::elementwise_vv(nc::gtx::ops::VVOp::SUB,"
        )
        self._inference_lines.append(
            f"            DDR_STAGING_RESULT, DDR_STAGING_B, DDR_STAGING_RESULT,"
        )
        self._inference_lines.append(f"            L2_RESULT, L2_WEIGHT, L2_RESULT,")
        self._inference_lines.append(f"            {total}, 1, NEST_ID, NUM_SPUS);")
        self._inference_lines.append(f"    }}")

    def _emit_sigmoid(self, node, idx):
        out_var = self._make_var_name(node.name)
        out_shape = self._get_output_shape(node)
        self._tensor_vars[node.name] = out_var
        total = self._total_elements(out_shape)

        self._inference_lines.append(f"    // [{idx}] Sigmoid: {node.name}")
        self._inference_lines.append(
            f"    nc::gtx::activation::sigmoid("
            f"L2_INPUT, L2_RESULT, {total}, 1, NEST_ID, NUM_SPUS);"
        )

    def _emit_gelu(self, node, idx):
        out_var = self._make_var_name(node.name)
        out_shape = self._get_output_shape(node)
        self._tensor_vars[node.name] = out_var
        total = self._total_elements(out_shape)

        self._inference_lines.append(f"    // [{idx}] GELU: {node.name}")
        self._inference_lines.append(
            f"    nc::gtx::activation::gelu("
            f"L2_INPUT, L2_RESULT, {total}, 1, NEST_ID, NUM_SPUS);"
        )

    def _emit_tanh(self, node, idx):
        out_var = self._make_var_name(node.name)
        out_shape = self._get_output_shape(node)
        self._tensor_vars[node.name] = out_var
        total = self._total_elements(out_shape)

        self._inference_lines.append(f"    // [{idx}] Tanh: {node.name}")
        self._inference_lines.append(
            f"    nc::gtx::activation::tanh_act("
            f"L2_INPUT, L2_RESULT, {total}, 1, NEST_ID, NUM_SPUS);"
        )

    def _emit_softmax(self, node, idx):
        out_var = self._make_var_name(node.name)
        out_shape = self._get_output_shape(node)
        self._tensor_vars[node.name] = out_var

        # softmax는 마지막 차원에 대해 수행
        if len(out_shape) >= 2:
            num_rows = self._total_elements(out_shape[:-1])
            num_cols = out_shape[-1]
        else:
            num_rows = 1
            num_cols = self._total_elements(out_shape)

        self._inference_lines.append(f"    // [{idx}] Softmax: {node.name}")
        self._inference_lines.append(
            f"    nc::gtx::activation::softmax("
            f"L2_INPUT, L2_RESULT, {num_cols}, {num_rows}, NEST_ID, NUM_SPUS);"
        )

    def _emit_batch_norm(self, node, idx):
        out_var = self._make_var_name(node.name)
        in_shape = self._get_input_shape(node)
        out_shape = self._get_output_shape(node)
        self._tensor_vars[node.name] = out_var

        # BN 파라미터 (per-channel)
        attrs = node.op.attrs if hasattr(node.op, "attrs") else {}
        eps = attrs.get("eps", 1e-5)
        eps_fp16 = float32_to_fp16_hex(eps)

        # shape: [N, C, H, W] → per-channel BN
        if len(in_shape) == 4:
            n, c, h, w = in_shape
            spatial = h * w
        elif len(in_shape) == 2:
            n, c = in_shape
            spatial = 1
        else:
            c = in_shape[-1] if len(in_shape) > 0 else 1
            spatial = self._total_elements(in_shape) // c

        # 가중치 이름
        mean_name = self._make_weight_name(node, "mean")
        var_name = self._make_weight_name(node, "var")
        gamma_name = self._make_weight_name(node, "gamma")
        beta_name = self._make_weight_name(node, "beta")

        self._inference_lines.append(
            f"    // [{idx}] BatchNorm: {node.name} — C={c}, spatial={spatial}"
        )
        self._inference_lines.append(f"    {{")
        self._inference_lines.append(f"        // Per-channel batch normalization")
        self._inference_lines.append(
            f"        const uint16_t* mean_fp16 = reinterpret_cast<const uint16_t*>("
            f"{mean_name.upper()}_DDR_ADDR);"
        )
        self._inference_lines.append(
            f"        const uint16_t* var_fp16 = reinterpret_cast<const uint16_t*>("
            f"{var_name.upper()}_DDR_ADDR);"
        )
        self._inference_lines.append(
            f"        const uint16_t* gamma_fp16 = reinterpret_cast<const uint16_t*>("
            f"{gamma_name.upper()}_DDR_ADDR);"
        )
        self._inference_lines.append(
            f"        const uint16_t* beta_fp16 = reinterpret_cast<const uint16_t*>("
            f"{beta_name.upper()}_DDR_ADDR);"
        )
        self._inference_lines.append(
            f"        for (uint32_t ch = 0; ch < {c}; ch++) {{"
        )
        self._inference_lines.append(
            f"            nc::gtx::norm::batchnorm("
            f"L2_INPUT + ch * {spatial} * 2, L2_RESULT + ch * {spatial} * 2,"
        )
        self._inference_lines.append(f"                {spatial}, 1,")
        self._inference_lines.append(
            f"                mean_fp16[ch], var_fp16[ch], gamma_fp16[ch], beta_fp16[ch],"
        )
        self._inference_lines.append(
            f"                0x{eps_fp16:04X}, NEST_ID, NUM_SPUS);"
        )
        self._inference_lines.append(f"        }}")
        self._inference_lines.append(f"    }}")

    def _emit_conv2d(self, node, idx):
        out_var = self._make_var_name(node.name)
        in_shape = self._get_input_shape(node)
        out_shape = self._get_output_shape(node)
        self._tensor_vars[node.name] = out_var

        attrs = node.op.attrs if hasattr(node.op, "attrs") else {}
        kernel = attrs.get("kernel_size", [3, 3])
        stride = attrs.get("stride", [1, 1])
        padding = attrs.get("padding", [0, 0])
        dilation = attrs.get("dilation", [1, 1])

        kh = kernel[0] if isinstance(kernel, list) else kernel
        kw = kernel[1] if isinstance(kernel, list) else kernel
        sh = stride[0] if isinstance(stride, list) else stride
        sw = stride[1] if isinstance(stride, list) else stride
        ph = padding[0] if isinstance(padding, list) else padding
        pw = padding[1] if isinstance(padding, list) else padding
        dh = dilation[0] if isinstance(dilation, list) else dilation
        dw = dilation[1] if isinstance(dilation, list) else dilation

        # shape: [N, IC, IH, IW]
        ic = in_shape[1] if len(in_shape) == 4 else 1
        ih = (
            in_shape[2]
            if len(in_shape) == 4
            else in_shape[-2]
            if len(in_shape) >= 2
            else 1
        )
        iw = in_shape[3] if len(in_shape) == 4 else in_shape[-1]
        oc = out_shape[1] if len(out_shape) == 4 else out_shape[-1]
        oh = out_shape[2] if len(out_shape) == 4 else 1
        ow = out_shape[3] if len(out_shape) == 4 else 1

        weight_name = self._make_weight_name(node, "weights")
        has_bias = bool(attrs.get("bias", False))
        bias_name = self._make_weight_name(node, "bias") if has_bias else None

        # Derived constants
        filt_h = kh if dh == 1 else (2 * dh + 1 if dh <= 3 else kh)
        padded_ih = sh * (oh - 1) + filt_h
        padded_iw = sw * (ow - 1) + (kw if dw == 1 else (2 * dw + 1 if dw <= 3 else kw))
        im2col_K = ic * kh * kw
        weight_row_bytes = im2col_K * 2
        needs_padding = (padded_ih != ih) or (padded_iw != iw)

        # Tiling: fit input tile + im2col result into L1
        L1_SIZE = 0x20000 + 0x10000 + 0x20000 + 0x10000  # 384KB total
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

        self._inference_lines.append(
            f"    // [{idx}] Conv2d: {node.name} — "
            f"IC={ic}, OC={oc}, K={kh}x{kw}, S={sh}x{sw}, P={ph}x{pw}, D={dh}x{dw}"
        )
        self._inference_lines.append(
            f"    // oh_tile={oh_tile}, tile_M={tile_M}, im2col_K={im2col_K}, padding={needs_padding}"
        )
        self._inference_lines.append(f"    {{")
        self._inference_lines.append(
            f"        const uint16_t in_c = {ic}, out_c = {oc};"
        )
        self._inference_lines.append(
            f"        const uint16_t out_h = {oh}, out_w = {ow};"
        )
        self._inference_lines.append(f"        const uint16_t oh_tile = {oh_tile};")
        self._inference_lines.append(f"        const uint16_t padded_w = {padded_iw};")
        self._inference_lines.append(
            f"        const uint32_t weight_row_bytes = {weight_row_bytes};"
        )
        self._inference_lines.append(f"")
        self._inference_lines.append(
            f"        for (uint16_t oh_start = 0; oh_start < out_h; oh_start += oh_tile) {{"
        )
        self._inference_lines.append(
            f"            uint16_t cur_oh = (oh_start + oh_tile > out_h) ? (out_h - oh_start) : oh_tile;"
        )
        self._inference_lines.append(
            f"            uint16_t tile_row_A = {sh} * (cur_oh - 1) + {filt_h};"
        )
        self._inference_lines.append(
            f"            uint16_t in_row_start = oh_start * {sh};"
        )
        self._inference_lines.append(f"            uint16_t tile_M = cur_oh * out_w;")
        self._inference_lines.append(
            f"            uint32_t input_tile_bytes = (uint32_t)tile_row_A * padded_w * in_c * 2;"
        )
        self._inference_lines.append(
            f"            uint32_t im2col_tile_bytes = (uint32_t)tile_M * {im2col_K} * 2;"
        )
        self._inference_lines.append(f"")

        if needs_padding:
            self._inference_lines.append(
                f"            // --- Padded input preparation (DDR→L2, zero-padding) ---"
            )
            self._inference_lines.append(
                f"            for (uint16_t c = 0; c < in_c; c++) {{"
            )
            self._inference_lines.append(
                f"                uint32_t padded_ch_bytes = (uint32_t)tile_row_A * padded_w * 2;"
            )
            self._inference_lines.append(
                f"                uint32_t l2_ch_off = L2_INPUT + c * padded_ch_bytes;"
            )
            self._inference_lines.append(f"")
            self._inference_lines.append(f"                // Zero-fill L2 region")
            self._inference_lines.append(f"                __split();")
            self._inference_lines.append(f"                __start_p(NEST_ID);")
            self._inference_lines.append(f"                    __start_s();")
            self._inference_lines.append(
                f"                        __fill(l2_ch_off, padded_w * 2, padded_w * 2, tile_row_A, 0, 0);"
            )
            self._inference_lines.append(f"                    __end_s();")
            self._inference_lines.append(f"                    __start_t(0);")
            self._inference_lines.append(f"                    __end_t(0);")
            self._inference_lines.append(f"                __end_p(NEST_ID);")
            self._inference_lines.append(f"                __join();")
            self._inference_lines.append(f"")
            self._inference_lines.append(
                f"                // Copy actual data to padded position"
            )
            self._inference_lines.append(
                f"                int16_t src_row_start = (int16_t)in_row_start - {ph};"
            )
            self._inference_lines.append(
                f"                int16_t src_row_end = src_row_start + tile_row_A - 1;"
            )
            self._inference_lines.append(
                f"                int16_t actual_start = (src_row_start < 0) ? 0 : src_row_start;"
            )
            self._inference_lines.append(
                f"                int16_t actual_end = (src_row_end >= {ih}) ? ({ih} - 1) : src_row_end;"
            )
            self._inference_lines.append(
                f"                if (actual_start <= actual_end) {{"
            )
            self._inference_lines.append(
                f"                    uint16_t copy_rows = actual_end - actual_start + 1;"
            )
            self._inference_lines.append(
                f"                    uint32_t dst_row_off = (uint32_t)(actual_start - src_row_start) * padded_w * 2;"
            )
            self._inference_lines.append(
                f"                    uint32_t dst_col_off = (uint32_t){pw} * 2;"
            )
            self._inference_lines.append(f"                    __split();")
            self._inference_lines.append(f"                    __start_p(NEST_ID);")
            self._inference_lines.append(f"                        __start_s();")
            self._inference_lines.append(
                f"                            __load_cr(DDR_INPUT_BASE + ((uint64_t)c * {ih} + actual_start) * {iw} * 2,"
            )
            self._inference_lines.append(
                f"                                      l2_ch_off + dst_row_off + dst_col_off,"
            )
            self._inference_lines.append(
                f"                                      {iw} * 2, {iw} * 2, copy_rows, padded_w * 2,"
            )
            self._inference_lines.append(
                f"                                      0, 0, 0);"
            )
            self._inference_lines.append(f"                        __end_s();")
            self._inference_lines.append(f"                        __start_t(0);")
            self._inference_lines.append(f"                        __end_t(0);")
            self._inference_lines.append(f"                    __end_p(NEST_ID);")
            self._inference_lines.append(f"                    __join();")
            self._inference_lines.append(f"                }}")
            self._inference_lines.append(f"            }}")
        else:
            self._inference_lines.append(
                f"            // --- Input tile DDR → L2 (no padding) ---"
            )
            self._inference_lines.append(
                f"            for (uint16_t c = 0; c < in_c; c++) {{"
            )
            self._inference_lines.append(
                f"                uint32_t ch_tile_bytes = (uint32_t)tile_row_A * {iw} * 2;"
            )
            self._inference_lines.append(f"                __split();")
            self._inference_lines.append(f"                __start_p(NEST_ID);")
            self._inference_lines.append(f"                    __start_s();")
            self._inference_lines.append(
                f"                        __load_cr(DDR_INPUT_BASE + ((uint64_t)c * {ih} + in_row_start) * {iw} * 2,"
            )
            self._inference_lines.append(
                f"                                  L2_INPUT + c * ch_tile_bytes,"
            )
            self._inference_lines.append(
                f"                                  {iw} * 2, {iw} * 2, tile_row_A, {iw} * 2,"
            )
            self._inference_lines.append(f"                                  0, 0, 0);")
            self._inference_lines.append(f"                    __end_s();")
            self._inference_lines.append(f"                    __start_t(0);")
            self._inference_lines.append(f"                    __end_t(0);")
            self._inference_lines.append(f"                __end_p(NEST_ID);")
            self._inference_lines.append(f"                __join();")
            self._inference_lines.append(f"            }}")

        self._inference_lines.append(f"")
        self._inference_lines.append(
            f"            // --- im2col: L2→L1, im2col_n, result stays in L1 ---"
        )
        self._inference_lines.append(f"            __split();")
        self._inference_lines.append(f"            __start_p(NEST_ID);")
        self._inference_lines.append(f"                __start_s();")
        self._inference_lines.append(f"                    __load_cr(L2_INPUT, 0,")
        self._inference_lines.append(
            f"                              input_tile_bytes, input_tile_bytes, 1, input_tile_bytes,"
        )
        self._inference_lines.append(f"                              1, 0x1, 0xDEAD);")
        self._inference_lines.append(f"                __end_s();")
        self._inference_lines.append(f"                __start_t(0);")
        self._inference_lines.append(f"                    nc::gtx::reset_l1_banks();")
        self._inference_lines.append(f"                    __set_spm_addr_R(0x00000);")
        self._inference_lines.append(
            f"                    __set_spm_addr_A(0x00000 + ((input_tile_bytes + 3) & ~3));"
        )
        self._inference_lines.append(
            f"                    __load_cr(0, nc::gtx::L1_BANK_R_BASE,"
        )
        self._inference_lines.append(
            f"                              input_tile_bytes, input_tile_bytes, 1, input_tile_bytes,"
        )
        self._inference_lines.append(
            f"                              1, 0xDEAD, 0xDEAD);"
        )
        self._inference_lines.append(
            f"                    nc::gtx::conv::im2col_normal_l1(tile_row_A, {padded_iw}, {kh}, {dh}, {sh}, {ic});"
        )
        self._inference_lines.append(f"                __end_t(0);")
        self._inference_lines.append(f"            __end_p(NEST_ID);")
        self._inference_lines.append(f"            __join();")
        self._inference_lines.append(f"")
        self._inference_lines.append(
            f"            // --- OC loop: weight DDR→L2→L1, mm, result L1→L2→DDR ---"
        )
        self._inference_lines.append(
            f"            for (uint16_t oc_s = 0; oc_s < out_c; oc_s++) {{"
        )
        self._inference_lines.append(f"                __split();")
        self._inference_lines.append(f"                __start_p(NEST_ID);")
        self._inference_lines.append(f"                    __start_s();")
        self._inference_lines.append(
            f"                        __load_cr({weight_name.upper()}_DDR_ADDR + (uint64_t)oc_s * weight_row_bytes,"
        )
        self._inference_lines.append(f"                                  L2_WEIGHT,")
        self._inference_lines.append(
            f"                                  weight_row_bytes, weight_row_bytes, 1, weight_row_bytes,"
        )
        self._inference_lines.append(
            f"                                  1, 0x1, 0xDEAD);"
        )
        self._inference_lines.append(f"                        __credit_chk(0x1);")
        self._inference_lines.append(f"                        {{")
        self._inference_lines.append(
            f"                            uint64_t dst_ddr = DDR_OUTPUT_BASE"
        )
        self._inference_lines.append(
            f"                                + (uint64_t)oc_s * out_h * out_w * 2"
        )
        self._inference_lines.append(
            f"                                + (uint64_t)oh_start * out_w * 2;"
        )
        self._inference_lines.append(
            f"                            uint32_t row_bytes = (uint32_t)out_w * 2;"
        )
        self._inference_lines.append(
            f"                            __store_cr(L2_RESULT, dst_ddr,"
        )
        self._inference_lines.append(
            f"                                       row_bytes, row_bytes, cur_oh, row_bytes,"
        )
        self._inference_lines.append(f"                                       1, 0x1);")
        self._inference_lines.append(f"                        }}")
        self._inference_lines.append(f"                    __end_s();")
        self._inference_lines.append(f"                    __start_t(0);")
        self._inference_lines.append(
            f"                        __load_cr(L2_WEIGHT, nc::gtx::L1_BANK_B_BASE,"
        )
        self._inference_lines.append(
            f"                                  weight_row_bytes, weight_row_bytes, 1, weight_row_bytes,"
        )
        self._inference_lines.append(
            f"                                  1, 0xDEAD, 0xDEAD);"
        )
        self._inference_lines.append(
            f"                        __set_spm_addr_B(nc::gtx::L1_BANK_B_BASE);"
        )
        self._inference_lines.append(
            f"                        __set_spm_addr_R(nc::gtx::L1_BANK_R_BASE);"
        )
        self._inference_lines.append(
            f"                        __mm(tile_M, {im2col_K}, 1);"
        )
        self._inference_lines.append(
            f"                        __store_cr(nc::gtx::L1_BANK_R_BASE, L2_RESULT,"
        )
        self._inference_lines.append(
            f"                                   tile_M * 2, tile_M * 2, 1, tile_M * 2,"
        )
        self._inference_lines.append(f"                                   1, 0xDEAD);")
        self._inference_lines.append(f"                    __end_t(0);")
        self._inference_lines.append(f"                __end_p(NEST_ID);")
        self._inference_lines.append(f"                __join();")
        self._inference_lines.append(f"            }}")
        self._inference_lines.append(f"        }}")
        self._inference_lines.append(f"    }}")

    def _emit_depthwise_conv2d(self, node, idx):
        out_var = self._make_var_name(node.name)
        self._tensor_vars[node.name] = out_var
        in_shape = self._get_input_shape(node)
        out_shape = self._get_output_shape(node)

        attrs = node.op.attrs if hasattr(node.op, "attrs") else {}
        kernel = attrs.get("kernel_size", [3, 3])
        stride = attrs.get("stride", [1, 1])
        padding = attrs.get("padding", [0, 0])
        dilation = attrs.get("dilation", [1, 1])

        kh = kernel[0] if isinstance(kernel, list) else kernel
        kw = kernel[1] if isinstance(kernel, list) else kernel
        sh = stride[0] if isinstance(stride, list) else stride
        sw = stride[1] if isinstance(stride, list) else stride
        ph = padding[0] if isinstance(padding, list) else padding
        pw = padding[1] if isinstance(padding, list) else padding
        dh = dilation[0] if isinstance(dilation, list) else dilation

        n = in_shape[0] if len(in_shape) > 0 else 1
        ic = in_shape[1] if len(in_shape) > 1 else 1
        ih = in_shape[2] if len(in_shape) > 2 else 1
        iw = in_shape[3] if len(in_shape) > 3 else 1
        oc = out_shape[1] if len(out_shape) > 1 else 1
        oh = out_shape[2] if len(out_shape) > 2 else 1
        ow = out_shape[3] if len(out_shape) > 3 else 1

        weight_name = self._make_weight_name(node, "weights")

        kernel_elems = kh * kw
        kernel_bytes = kernel_elems * 2
        padded_ih = sh * (oh - 1) + (kh if dh == 1 else (2 * dh + 1 if dh <= 3 else kh))
        padded_iw = sw * (ow - 1) + (kw if dh == 1 else (2 * dh + 1 if dh <= 3 else kw))
        needs_padding = (padded_ih != ih) or (padded_iw != iw)
        in_ch_bytes = ih * iw * 2
        out_ch_bytes = oh * ow * 2
        padded_ch_bytes = padded_ih * padded_iw * 2
        l2_input_size = padded_ch_bytes if needs_padding else in_ch_bytes

        self._inference_lines.append(
            f"    // [{idx}] DepthwiseConv2d: {node.name} — "
            f"IC={ic}, K={kh}x{kw}, S={sh}x{sw}, P={ph}x{pw}, D={dh}"
        )
        self._inference_lines.append(f"    {{")
        self._inference_lines.append(f"        const uint16_t channels = {ic};")
        self._inference_lines.append(
            f"        const uint32_t in_ch_bytes = {in_ch_bytes};"
        )
        self._inference_lines.append(
            f"        const uint32_t out_ch_bytes = {out_ch_bytes};"
        )
        self._inference_lines.append(
            f"        const uint32_t kernel_bytes = {kernel_bytes};"
        )
        self._inference_lines.append(f"")
        self._inference_lines.append(
            f"        for (uint16_t c = 0; c < channels; c++) {{"
        )
        self._inference_lines.append(
            f"            uint64_t ch_in  = DDR_INPUT_BASE + (uint64_t)c * in_ch_bytes;"
        )
        self._inference_lines.append(
            f"            uint64_t ch_out = DDR_OUTPUT_BASE + (uint64_t)c * out_ch_bytes;"
        )
        self._inference_lines.append(
            f"            uint64_t ch_wt  = {weight_name.upper()}_DDR_ADDR + (uint64_t)c * kernel_bytes;"
        )
        self._inference_lines.append(f"")

        if needs_padding:
            self._inference_lines.append(
                f"            // --- Padded input preparation (1 channel) ---"
            )
            self._inference_lines.append(f"            __split();")
            self._inference_lines.append(f"            __start_p(NEST_ID);")
            self._inference_lines.append(f"                __start_s();")
            self._inference_lines.append(
                f"                    __fill(L2_INPUT, {padded_iw} * 2, {padded_iw} * 2, {padded_ih}, 0, 0);"
            )
            self._inference_lines.append(f"                    {{")
            self._inference_lines.append(
                f"                        uint32_t dst_off = ((uint32_t){ph} * {padded_iw} + {pw}) * 2;"
            )
            self._inference_lines.append(
                f"                        __load_cr(ch_in, L2_INPUT + dst_off,"
            )
            self._inference_lines.append(
                f"                                  {iw} * 2, {iw} * 2, {ih}, {padded_iw} * 2,"
            )
            self._inference_lines.append(f"                                  0, 0, 0);")
            self._inference_lines.append(f"                    }}")
            self._inference_lines.append(f"                __end_s();")
            self._inference_lines.append(f"                __start_t(0);")
            self._inference_lines.append(f"                __end_t(0);")
            self._inference_lines.append(f"            __end_p(NEST_ID);")
            self._inference_lines.append(f"            __join();")
        else:
            self._inference_lines.append(
                f"            // --- Input 1 channel DDR → L2 (no padding) ---"
            )
            self._inference_lines.append(f"            __split();")
            self._inference_lines.append(f"            __start_p(NEST_ID);")
            self._inference_lines.append(f"                __start_s();")
            self._inference_lines.append(
                f"                    __load_cr(ch_in, L2_INPUT,"
            )
            self._inference_lines.append(
                f"                              in_ch_bytes, in_ch_bytes, 1, in_ch_bytes,"
            )
            self._inference_lines.append(f"                              0, 0, 0);")
            self._inference_lines.append(f"                __end_s();")
            self._inference_lines.append(f"                __start_t(0);")
            self._inference_lines.append(f"                __end_t(0);")
            self._inference_lines.append(f"            __end_p(NEST_ID);")
            self._inference_lines.append(f"            __join();")

        self._inference_lines.append(f"")
        self._inference_lines.append(
            f"            // --- im2col_d + mm: L2→L1, compute, L1→L2→DDR ---"
        )
        self._inference_lines.append(f"            __split();")
        self._inference_lines.append(f"            __start_p(NEST_ID);")
        self._inference_lines.append(f"                __start_s();")
        self._inference_lines.append(f"                    __load_cr(L2_INPUT, 0,")
        self._inference_lines.append(
            f"                              {l2_input_size}, {l2_input_size}, 1, {l2_input_size},"
        )
        self._inference_lines.append(f"                              1, 0x1, 0xDEAD);")
        self._inference_lines.append(f"                    __load_cr(ch_wt, L2_WEIGHT,")
        self._inference_lines.append(
            f"                              kernel_bytes, kernel_bytes, 1, kernel_bytes,"
        )
        self._inference_lines.append(f"                              0, 0, 0);")
        self._inference_lines.append(f"                    __credit_chk(0x1);")
        self._inference_lines.append(
            f"                    __store_cr(L2_RESULT, ch_out,"
        )
        self._inference_lines.append(
            f"                               out_ch_bytes, out_ch_bytes, 1, out_ch_bytes,"
        )
        self._inference_lines.append(f"                               1, 0x1);")
        self._inference_lines.append(f"                __end_s();")
        self._inference_lines.append(f"                __start_t(0);")
        self._inference_lines.append(f"                    __init_spm_addr(0xF);")
        self._inference_lines.append(
            f"                    __load_cr(0, nc::gtx::L1_BANK_R_BASE,"
        )
        self._inference_lines.append(
            f"                              {l2_input_size}, {l2_input_size}, 1, {l2_input_size},"
        )
        self._inference_lines.append(
            f"                              1, 0xDEAD, 0xDEAD);"
        )
        self._inference_lines.append(
            f"                    nc::gtx::conv::im2col_depthwise_l1({padded_ih}, {padded_iw}, {kh}, {dh}, {sh}, 1);"
        )
        self._inference_lines.append(
            f"                    __load_cr(L2_WEIGHT, nc::gtx::L1_BANK_B_BASE,"
        )
        self._inference_lines.append(
            f"                              kernel_bytes, kernel_bytes, 1, kernel_bytes,"
        )
        self._inference_lines.append(f"                              0, 0, 0);")
        self._inference_lines.append(
            f"                    __mm({oh} * {ow}, {kernel_elems}, 1);"
        )
        self._inference_lines.append(
            f"                    __store_cr(nc::gtx::L1_BANK_C_BASE, L2_RESULT,"
        )
        self._inference_lines.append(
            f"                               out_ch_bytes, out_ch_bytes, 1, out_ch_bytes,"
        )
        self._inference_lines.append(f"                               1, 0xDEAD);")
        self._inference_lines.append(f"                __end_t(0);")
        self._inference_lines.append(f"            __end_p(NEST_ID);")
        self._inference_lines.append(f"            __join();")
        self._inference_lines.append(f"        }}")
        self._inference_lines.append(f"    }}")

    def _emit_maxpool(self, node, idx):
        out_var = self._make_var_name(node.name)
        in_shape = self._get_input_shape(node)
        out_shape = self._get_output_shape(node)
        self._tensor_vars[node.name] = out_var

        attrs = node.op.attrs if hasattr(node.op, "attrs") else {}
        kernel = attrs.get("kernel_size", [2, 2])
        stride = attrs.get("stride", [2, 2])
        padding = attrs.get("padding", [0, 0])

        kh = kernel[0] if isinstance(kernel, list) else kernel
        kw = kernel[1] if isinstance(kernel, list) else kernel
        sh = stride[0] if isinstance(stride, list) else stride
        sw = stride[1] if isinstance(stride, list) else stride

        c = in_shape[1] if len(in_shape) == 4 else 1
        ih = (
            in_shape[2]
            if len(in_shape) == 4
            else in_shape[-2]
            if len(in_shape) >= 2
            else 1
        )
        iw = in_shape[3] if len(in_shape) == 4 else in_shape[-1]
        oh = out_shape[2] if len(out_shape) == 4 else 1
        ow = out_shape[3] if len(out_shape) == 4 else 1

        in_ch_bytes = ih * iw * 2
        out_ch_bytes = oh * ow * 2

        self._inference_lines.append(
            f"    // [{idx}] MaxPool2d: {node.name} — K={kh}x{kw}, S={sh}x{sw}"
        )
        self._inference_lines.append(f"    {{")
        self._inference_lines.append(f"        const uint16_t channels = {c};")
        self._inference_lines.append(
            f"        const uint32_t in_ch_bytes = {in_ch_bytes};"
        )
        self._inference_lines.append(
            f"        const uint32_t out_ch_bytes = {out_ch_bytes};"
        )
        self._inference_lines.append(f"")
        self._inference_lines.append(
            f"        for (uint16_t ch = 0; ch < channels; ch++) {{"
        )
        self._inference_lines.append(
            f"            uint64_t ch_in  = DDR_INPUT_BASE + (uint64_t)ch * in_ch_bytes;"
        )
        self._inference_lines.append(
            f"            uint64_t ch_out = DDR_OUTPUT_BASE + (uint64_t)ch * out_ch_bytes;"
        )
        self._inference_lines.append(f"")
        self._inference_lines.append(f"            __split();")
        self._inference_lines.append(f"            __start_p(NEST_ID);")
        self._inference_lines.append(f"                __start_s();")
        self._inference_lines.append(f"                    __load_cr(ch_in, L2_INPUT,")
        self._inference_lines.append(
            f"                              in_ch_bytes, in_ch_bytes, 1, in_ch_bytes,"
        )
        self._inference_lines.append(f"                              1, 0x1, 0xDEAD);")
        self._inference_lines.append(f"                    __credit_chk(0x1);")
        self._inference_lines.append(
            f"                    __store_cr(L2_RESULT, ch_out,"
        )
        self._inference_lines.append(
            f"                               out_ch_bytes, out_ch_bytes, 1, out_ch_bytes,"
        )
        self._inference_lines.append(f"                               1, 0x1);")
        self._inference_lines.append(f"                __end_s();")
        self._inference_lines.append(f"                __start_t(0);")
        self._inference_lines.append(f"                    __init_spm_addr(0xF);")
        self._inference_lines.append(
            f"                    __load_cr(L2_INPUT, nc::gtx::L1_BANK_A_BASE,"
        )
        self._inference_lines.append(
            f"                              in_ch_bytes, in_ch_bytes, 1, in_ch_bytes,"
        )
        self._inference_lines.append(
            f"                              1, 0xDEAD, 0xDEAD);"
        )
        self._inference_lines.append(
            f"                    nc::gtx::pool::max_pool_2d_l1("
        )
        self._inference_lines.append(
            f"                        {ih}, {iw}, {oh}, {ow}, {kh}, {kw}, {sh}, {sw});"
        )
        self._inference_lines.append(
            f"                    __store_cr(nc::gtx::L1_BANK_C_BASE, L2_RESULT,"
        )
        self._inference_lines.append(
            f"                               out_ch_bytes, out_ch_bytes, 1, out_ch_bytes,"
        )
        self._inference_lines.append(f"                               1, 0xDEAD);")
        self._inference_lines.append(f"                __end_t(0);")
        self._inference_lines.append(f"            __end_p(NEST_ID);")
        self._inference_lines.append(f"            __join();")
        self._inference_lines.append(f"        }}")
        self._inference_lines.append(f"    }}")

    def _emit_avgpool(self, node, idx):
        out_var = self._make_var_name(node.name)
        self._tensor_vars[node.name] = out_var
        in_shape = self._get_input_shape(node)
        out_shape = self._get_output_shape(node)

        attrs = node.op.attrs if hasattr(node.op, "attrs") else {}
        kernel = attrs.get("kernel_size", [2, 2])
        stride = attrs.get("stride", [2, 2])
        kh = kernel[0] if isinstance(kernel, list) else kernel
        kw = kernel[1] if isinstance(kernel, list) else kernel
        sh = stride[0] if isinstance(stride, list) else stride
        sw = stride[1] if isinstance(stride, list) else stride

        channels = in_shape[1] if len(in_shape) > 1 else 1
        ih = in_shape[2] if len(in_shape) > 2 else 1
        iw = in_shape[3] if len(in_shape) > 3 else 1
        oh = out_shape[2] if len(out_shape) > 2 else 1
        ow = out_shape[3] if len(out_shape) > 3 else 1

        k_value = 1.0 / (kh * kw)
        k_fp16 = float32_to_fp16_hex(k_value)
        in_ch_bytes = ih * iw * 2
        out_ch_bytes = oh * ow * 2

        self._inference_lines.append(
            f"    // [{idx}] AvgPool2d: {node.name} — K={kh}x{kw}, S={sh}x{sw}"
        )
        self._inference_lines.append(f"    {{")
        self._inference_lines.append(f"        const uint16_t channels = {channels};")
        self._inference_lines.append(
            f"        const uint32_t in_ch_bytes = {in_ch_bytes};"
        )
        self._inference_lines.append(
            f"        const uint32_t out_ch_bytes = {out_ch_bytes};"
        )
        self._inference_lines.append(
            f"        const uint16_t k_value = 0x{k_fp16:04X};  // 1/({kh}*{kw}) fp16"
        )
        self._inference_lines.append(f"")
        self._inference_lines.append(
            f"        for (uint16_t c = 0; c < channels; c++) {{"
        )
        self._inference_lines.append(
            f"            uint64_t ch_in  = DDR_INPUT_BASE + (uint64_t)c * in_ch_bytes;"
        )
        self._inference_lines.append(
            f"            uint64_t ch_out = DDR_OUTPUT_BASE + (uint64_t)c * out_ch_bytes;"
        )
        self._inference_lines.append(f"")
        self._inference_lines.append(f"            __split();")
        self._inference_lines.append(f"            __start_p(NEST_ID);")
        self._inference_lines.append(f"                __start_s();")
        self._inference_lines.append(f"                    __load_cr(ch_in, L2_INPUT,")
        self._inference_lines.append(
            f"                              in_ch_bytes, in_ch_bytes, 1, in_ch_bytes,"
        )
        self._inference_lines.append(f"                              1, 0x1, 0xDEAD);")
        self._inference_lines.append(f"                    __credit_chk(0x1);")
        self._inference_lines.append(
            f"                    __store_cr(L2_RESULT, ch_out,"
        )
        self._inference_lines.append(
            f"                               out_ch_bytes, out_ch_bytes, 1, out_ch_bytes,"
        )
        self._inference_lines.append(f"                               1, 0x1);")
        self._inference_lines.append(f"                __end_s();")
        self._inference_lines.append(f"                __start_t(0);")
        self._inference_lines.append(f"                    __init_spm_addr(0xF);")
        self._inference_lines.append(
            f"                    __load_cr(L2_INPUT, nc::gtx::L1_BANK_A_BASE,"
        )
        self._inference_lines.append(
            f"                              in_ch_bytes, in_ch_bytes, 1, in_ch_bytes,"
        )
        self._inference_lines.append(
            f"                              1, 0xDEAD, 0xDEAD);"
        )
        self._inference_lines.append(
            f"                    nc::gtx::pool::avg_pool_2d_l1("
        )
        self._inference_lines.append(
            f"                        {ih}, {iw}, {oh}, {ow}, {kh}, {kw}, {sh}, {sw}, k_value);"
        )
        self._inference_lines.append(
            f"                    __store_cr(nc::gtx::L1_BANK_C_BASE, L2_RESULT,"
        )
        self._inference_lines.append(
            f"                               out_ch_bytes, out_ch_bytes, 1, out_ch_bytes,"
        )
        self._inference_lines.append(f"                               1, 0xDEAD);")
        self._inference_lines.append(f"                __end_t(0);")
        self._inference_lines.append(f"            __end_p(NEST_ID);")
        self._inference_lines.append(f"            __join();")
        self._inference_lines.append(f"        }}")
        self._inference_lines.append(f"    }}")

    def _emit_adaptive_avgpool(self, node, idx):
        out_var = self._make_var_name(node.name)
        self._tensor_vars[node.name] = out_var
        in_shape = self._get_input_shape(node)
        out_shape = self._get_output_shape(node)

        attrs = node.op.attrs if hasattr(node.op, "attrs") else {}
        output_size = attrs.get("output_size", [1, 1])
        channels = in_shape[1] if len(in_shape) > 1 else 1
        ih = in_shape[2] if len(in_shape) > 2 else 1
        iw = in_shape[3] if len(in_shape) > 3 else 1
        oh_t = int(
            output_size[0] if isinstance(output_size, (list, tuple)) else output_size
        )
        ow_t = int(
            output_size[1] if isinstance(output_size, (list, tuple)) else output_size
        )

        # Adaptive: kernel/stride = input_size / output_size
        kh = ih // oh_t
        kw = iw // ow_t
        k_value = 1.0 / (kh * kw)
        k_fp16 = float32_to_fp16_hex(k_value)
        in_ch_bytes = ih * iw * 2
        out_ch_bytes = oh_t * ow_t * 2

        self._inference_lines.append(
            f"    // [{idx}] AdaptiveAvgPool2d: {node.name} — "
            f"[{ih}x{iw}]→[{oh_t}x{ow_t}], auto K={kh}x{kw}"
        )
        self._inference_lines.append(f"    {{")
        self._inference_lines.append(f"        const uint16_t channels = {channels};")
        self._inference_lines.append(
            f"        const uint32_t in_ch_bytes = {in_ch_bytes};"
        )
        self._inference_lines.append(
            f"        const uint32_t out_ch_bytes = {out_ch_bytes};"
        )
        self._inference_lines.append(
            f"        const uint16_t k_value = 0x{k_fp16:04X};  // 1/({kh}*{kw}) fp16"
        )
        self._inference_lines.append(f"")
        self._inference_lines.append(
            f"        for (uint16_t c = 0; c < channels; c++) {{"
        )
        self._inference_lines.append(
            f"            uint64_t ch_in  = DDR_INPUT_BASE + (uint64_t)c * in_ch_bytes;"
        )
        self._inference_lines.append(
            f"            uint64_t ch_out = DDR_OUTPUT_BASE + (uint64_t)c * out_ch_bytes;"
        )
        self._inference_lines.append(f"")
        self._inference_lines.append(f"            __split();")
        self._inference_lines.append(f"            __start_p(NEST_ID);")
        self._inference_lines.append(f"                __start_s();")
        self._inference_lines.append(f"                    __load_cr(ch_in, L2_INPUT,")
        self._inference_lines.append(
            f"                              in_ch_bytes, in_ch_bytes, 1, in_ch_bytes,"
        )
        self._inference_lines.append(f"                              1, 0x1, 0xDEAD);")
        self._inference_lines.append(f"                    __credit_chk(0x1);")
        self._inference_lines.append(
            f"                    __store_cr(L2_RESULT, ch_out,"
        )
        self._inference_lines.append(
            f"                               out_ch_bytes, out_ch_bytes, 1, out_ch_bytes,"
        )
        self._inference_lines.append(f"                               1, 0x1);")
        self._inference_lines.append(f"                __end_s();")
        self._inference_lines.append(f"                __start_t(0);")
        self._inference_lines.append(f"                    __init_spm_addr(0xF);")
        self._inference_lines.append(
            f"                    __load_cr(L2_INPUT, nc::gtx::L1_BANK_A_BASE,"
        )
        self._inference_lines.append(
            f"                              in_ch_bytes, in_ch_bytes, 1, in_ch_bytes,"
        )
        self._inference_lines.append(
            f"                              1, 0xDEAD, 0xDEAD);"
        )
        self._inference_lines.append(
            f"                    nc::gtx::pool::avg_pool_2d_l1("
        )
        self._inference_lines.append(
            f"                        {ih}, {iw}, {oh_t}, {ow_t}, {kh}, {kw}, {kh}, {kw}, k_value);"
        )
        self._inference_lines.append(
            f"                    __store_cr(nc::gtx::L1_BANK_C_BASE, L2_RESULT,"
        )
        self._inference_lines.append(
            f"                               out_ch_bytes, out_ch_bytes, 1, out_ch_bytes,"
        )
        self._inference_lines.append(f"                               1, 0xDEAD);")
        self._inference_lines.append(f"                __end_t(0);")
        self._inference_lines.append(f"            __end_p(NEST_ID);")
        self._inference_lines.append(f"            __join();")
        self._inference_lines.append(f"        }}")
        self._inference_lines.append(f"    }}")

    def _emit_elemwise_add(self, node, idx):
        out_var = self._make_var_name(node.name)
        out_shape = self._get_output_shape(node)
        self._tensor_vars[node.name] = out_var
        total = self._total_elements(out_shape)

        self._inference_lines.append(f"    // [{idx}] Add: {node.name}")
        self._inference_lines.append(
            f"    nc::gtx::ops::elementwise_vv(nc::gtx::ops::VVOp::ADD,"
        )
        self._inference_lines.append(
            f"        DDR_STAGING_A, DDR_STAGING_B, DDR_STAGING_RESULT,"
        )
        self._inference_lines.append(f"        L2_INPUT, L2_WEIGHT, L2_RESULT,")
        self._inference_lines.append(f"        {total}, 1, NEST_ID, NUM_SPUS);")

    def _emit_elemwise_mul(self, node, idx):
        out_var = self._make_var_name(node.name)
        out_shape = self._get_output_shape(node)
        self._tensor_vars[node.name] = out_var
        total = self._total_elements(out_shape)

        self._inference_lines.append(f"    // [{idx}] Multiply: {node.name}")
        self._inference_lines.append(
            f"    nc::gtx::ops::elementwise_vv(nc::gtx::ops::VVOp::MUL,"
        )
        self._inference_lines.append(
            f"        DDR_STAGING_A, DDR_STAGING_B, DDR_STAGING_RESULT,"
        )
        self._inference_lines.append(f"        L2_INPUT, L2_WEIGHT, L2_RESULT,")
        self._inference_lines.append(f"        {total}, 1, NEST_ID, NUM_SPUS);")

    def _emit_dense(self, node, idx):
        out_var = self._make_var_name(node.name)
        in_shape = self._get_input_shape(node)
        out_shape = self._get_output_shape(node)
        self._tensor_vars[node.name] = out_var

        attrs = node.op.attrs if hasattr(node.op, "attrs") else {}
        in_features = int(attrs.get("in_features", in_shape[-1] if in_shape else 1))
        out_features = int(attrs.get("out_features", out_shape[-1] if out_shape else 1))
        has_bias = bool(attrs.get("bias", True))

        weight_name = self._make_weight_name(node, "weights")
        bias_name = self._make_weight_name(node, "bias") if has_bias else ""

        in_bytes = in_features * 2
        # L1 Bank B size = 64KB = 0x10000
        L1_BANK_B_SIZE = 0x10000
        w_row_bytes = in_bytes
        of_tile = L1_BANK_B_SIZE // w_row_bytes
        if of_tile < 1:
            of_tile = 1
        if of_tile > out_features:
            of_tile = out_features

        self._inference_lines.append(
            f"    // [{idx}] Dense: {node.name} — "
            f"in={in_features}, out={out_features}, bias={has_bias}"
        )
        self._inference_lines.append(f"    {{")
        self._inference_lines.append(f"        const uint16_t in_feat = {in_features};")
        self._inference_lines.append(
            f"        const uint16_t out_feat = {out_features};"
        )
        self._inference_lines.append(
            f"        const uint32_t in_bytes = (uint32_t)in_feat * 2;"
        )
        self._inference_lines.append(f"        const uint32_t w_row_bytes = in_bytes;")
        self._inference_lines.append(f"        const uint16_t of_tile = {of_tile};")
        self._inference_lines.append(f"")
        self._inference_lines.append(f"        // Load input vector to L2 once")
        self._inference_lines.append(f"        __split();")
        self._inference_lines.append(f"        __start_p(NEST_ID);")
        self._inference_lines.append(f"            __start_s();")
        self._inference_lines.append(
            f"                __load_cr(DDR_INPUT_BASE, L2_INPUT, in_bytes, in_bytes, 1, in_bytes, 0, 0, 0);"
        )
        self._inference_lines.append(f"            __end_s();")
        self._inference_lines.append(f"            __start_t(0);")
        self._inference_lines.append(f"            __end_t(0);")
        self._inference_lines.append(f"        __end_p(NEST_ID);")
        self._inference_lines.append(f"        __join();")
        self._inference_lines.append(f"")
        self._inference_lines.append(
            f"        for (uint16_t of_s = 0; of_s < out_feat; of_s += of_tile) {{"
        )
        self._inference_lines.append(
            f"            uint16_t cur_of = (of_s + of_tile > out_feat) ? (out_feat - of_s) : of_tile;"
        )
        self._inference_lines.append(f"")
        self._inference_lines.append(
            f"            // mm: input x weight_tile → result_tile"
        )
        self._inference_lines.append(f"            __split();")
        self._inference_lines.append(f"            __start_p(NEST_ID);")
        self._inference_lines.append(f"                __start_s();")
        self._inference_lines.append(
            f"                    __load_cr({weight_name.upper()}_DDR_ADDR + (uint64_t)of_s * w_row_bytes,"
        )
        self._inference_lines.append(f"                              L2_WEIGHT,")
        self._inference_lines.append(
            f"                              w_row_bytes, w_row_bytes, cur_of, w_row_bytes,"
        )
        self._inference_lines.append(f"                              1, 0x1, 0xDEAD);")
        self._inference_lines.append(f"                    __credit_chk(0x1);")

        if not has_bias:
            self._inference_lines.append(
                f"                    __store_cr(L2_RESULT, DDR_OUTPUT_BASE + (uint64_t)of_s * 2,"
            )
            self._inference_lines.append(
                f"                               cur_of * 2, cur_of * 2, 1, cur_of * 2,"
            )
            self._inference_lines.append(f"                               1, 0x1);")

        self._inference_lines.append(f"                __end_s();")
        self._inference_lines.append(f"                __start_t(0);")
        self._inference_lines.append(f"                    __init_spm_addr(0xF);")
        self._inference_lines.append(
            f"                    __load_cr(L2_INPUT, nc::gtx::L1_BANK_A_BASE,"
        )
        self._inference_lines.append(
            f"                              in_bytes, in_bytes, 1, in_bytes, 0, 0, 0);"
        )
        self._inference_lines.append(
            f"                    __load_cr(L2_WEIGHT, nc::gtx::L1_BANK_B_BASE,"
        )
        self._inference_lines.append(
            f"                              w_row_bytes, w_row_bytes, cur_of, w_row_bytes,"
        )
        self._inference_lines.append(
            f"                              1, 0xDEAD, 0xDEAD);"
        )
        self._inference_lines.append(f"                    __mm(1, in_feat, cur_of);")

        if not has_bias:
            self._inference_lines.append(
                f"                    __store_cr(nc::gtx::L1_BANK_C_BASE, L2_RESULT,"
            )
            self._inference_lines.append(
                f"                               cur_of * 2, cur_of * 2, 1, cur_of * 2,"
            )
            self._inference_lines.append(f"                               1, 0xDEAD);")

        self._inference_lines.append(f"                __end_t(0);")
        self._inference_lines.append(f"            __end_p(NEST_ID);")
        self._inference_lines.append(f"            __join();")

        if has_bias:
            self._inference_lines.append(f"")
            self._inference_lines.append(
                f"            // Bias addition: DDR→L2→L1, add_vv"
            )
            self._inference_lines.append(f"            __split();")
            self._inference_lines.append(f"            __start_p(NEST_ID);")
            self._inference_lines.append(f"                __start_s();")
            self._inference_lines.append(
                f"                    __load_cr({bias_name.upper()}_DDR_ADDR + (uint64_t)of_s * 2,"
            )
            self._inference_lines.append(f"                              L2_WEIGHT,")
            self._inference_lines.append(
                f"                              cur_of * 2, cur_of * 2, 1, cur_of * 2,"
            )
            self._inference_lines.append(
                f"                              1, 0x1, 0xDEAD);"
            )
            self._inference_lines.append(f"                    __credit_chk(0x1);")
            self._inference_lines.append(
                f"                    __store_cr(L2_RESULT, DDR_OUTPUT_BASE + (uint64_t)of_s * 2,"
            )
            self._inference_lines.append(
                f"                               cur_of * 2, cur_of * 2, 1, cur_of * 2,"
            )
            self._inference_lines.append(f"                               1, 0x1);")
            self._inference_lines.append(f"                __end_s();")
            self._inference_lines.append(f"                __start_t(0);")
            self._inference_lines.append(
                f"                    __load_cr(L2_WEIGHT, nc::gtx::L1_BANK_B_BASE,"
            )
            self._inference_lines.append(
                f"                              cur_of * 2, cur_of * 2, 1, cur_of * 2,"
            )
            self._inference_lines.append(
                f"                              1, 0xDEAD, 0xDEAD);"
            )
            self._inference_lines.append(
                f"                    __set_spm_addr_A(nc::gtx::L1_BANK_C_BASE);"
            )
            self._inference_lines.append(
                f"                    __set_spm_addr_B(nc::gtx::L1_BANK_B_BASE);"
            )
            self._inference_lines.append(f"                    __add_vv(cur_of);")
            self._inference_lines.append(
                f"                    __store_cr(nc::gtx::L1_BANK_R_BASE, L2_RESULT,"
            )
            self._inference_lines.append(
                f"                               cur_of * 2, cur_of * 2, 1, cur_of * 2,"
            )
            self._inference_lines.append(f"                               1, 0xDEAD);")
            self._inference_lines.append(f"                __end_t(0);")
            self._inference_lines.append(f"            __end_p(NEST_ID);")
            self._inference_lines.append(f"            __join();")

        self._inference_lines.append(f"        }}")
        self._inference_lines.append(f"    }}")

    def _emit_flatten(self, node, idx):
        out_var = self._make_var_name(node.name)
        self._tensor_vars[node.name] = out_var
        # No-op: share input variable
        if node.in_nodes:
            in_node = list(node.in_nodes)[0]
            if in_node.name in self._tensor_vars:
                self._tensor_vars[node.name] = self._tensor_vars[in_node.name]
        self._inference_lines.append(
            f"    // [{idx}] Flatten: {node.name} — no-op (memory reinterpret)"
        )

    def _emit_reshape(self, node, idx):
        out_var = self._make_var_name(node.name)
        self._tensor_vars[node.name] = out_var
        if node.in_nodes:
            in_node = list(node.in_nodes)[0]
            if in_node.name in self._tensor_vars:
                self._tensor_vars[node.name] = self._tensor_vars[in_node.name]
        self._inference_lines.append(
            f"    // [{idx}] Reshape: {node.name} — no-op (memory reinterpret)"
        )

    def _emit_concat(self, node, idx):
        out_var = self._make_var_name(node.name)
        self._tensor_vars[node.name] = out_var
        out_shape = self._get_output_shape(node)

        self._inference_lines.append(f"    // [{idx}] Concat: {node.name}")
        self._inference_lines.append(f"    {{")

        offset = 0
        for i, in_node in enumerate(node.in_nodes):
            in_shape = (
                list(in_node.out_tensors[0].shape)
                if in_node.out_tensors and in_node.out_tensors[0].shape
                else [1]
            )
            in_size = self._total_elements(in_shape) * 2  # bytes (fp16)
            in_var = self._tensor_vars.get(in_node.name, "input")

            self._inference_lines.append(
                f"        // Copy input {i}: {in_size} bytes at offset {offset}"
            )
            self._inference_lines.append(f"        __split();")
            self._inference_lines.append(f"        __start_p(NEST_ID);")
            self._inference_lines.append(f"            __start_s();")
            self._inference_lines.append(
                f"                __load_cr(DDR_INPUT_BASE + {offset}ULL, L2_INPUT,"
            )
            self._inference_lines.append(
                f"                          {in_size}, {in_size}, 1, {in_size}, 0, 0, 0);"
            )
            self._inference_lines.append(
                f"                __store_cr(L2_INPUT, DDR_OUTPUT_BASE + {offset}ULL,"
            )
            self._inference_lines.append(
                f"                           {in_size}, {in_size}, 1, {in_size}, 0, 0);"
            )
            self._inference_lines.append(f"            __end_s();")
            self._inference_lines.append(f"            __start_t(0);")
            self._inference_lines.append(f"            __end_t(0);")
            self._inference_lines.append(f"        __end_p(NEST_ID);")
            self._inference_lines.append(f"        __join();")
            offset += in_size

        self._inference_lines.append(f"    }}")

    # ========================================
    # 파일 생성
    # ========================================

    def _write_implementation(self) -> str:
        """C++ 구현 파일 생성"""
        path = os.path.join(self.output_dir, f"{self.model_name}.cpp")

        lines = []
        lines.append(
            f"// Auto-generated by GTX Compiler (NumCpp backend) — DO NOT EDIT"
        )
        lines.append(f"// Model: {self.graph.name}")
        lines.append(f"//")
        lines.append(f"// NumCpp GTX Acceleration Layer를 사용한 추론 코드")
        lines.append(
            f"// 내부적으로 Plan/Shared/Thread/Credit 동기화를 자동 처리합니다."
        )
        lines.append(f"")

        # Includes
        lines.append(f"#include <cstdint>")
        lines.append(f"#include <cstring>")
        lines.append(f"")
        lines.append(f"// NumCpp GTX headers")
        lines.append(f'#include "NumCpp/Gtx/GtxConfig.hpp"')
        lines.append(f'#include "NumCpp/Gtx/GtxMemory.hpp"')
        lines.append(f'#include "NumCpp/Gtx/GtxOps.hpp"')
        lines.append(f'#include "NumCpp/Gtx/GtxActivation.hpp"')
        lines.append(f'#include "NumCpp/Gtx/GtxNorm.hpp"')
        lines.append(f'#include "NumCpp/Gtx/GtxPoolConv.hpp"')
        lines.append(f"")
        lines.append(f'extern "C" {{')
        lines.append(f'#include "sc_print.h"')
        lines.append(f"}}")
        lines.append(f"")

        # Weight addresses from weight_map.h
        lines.append(f'#include "weight_map.h"')
        lines.append(f"")

        # Constants
        lines.append(f"// =============================================")
        lines.append(f"// Hardware configuration")
        lines.append(f"// =============================================")
        lines.append(f"static constexpr uint8_t  NEST_ID   = 0;")
        lines.append(
            f"static constexpr uint8_t  NUM_SPUS  = 1;  // Single SPU for correctness"
        )
        lines.append(f"")
        lines.append(f"// L2 SPM region offsets")
        lines.append(f"static constexpr uint32_t L2_INPUT  = 0x000000;")
        lines.append(f"static constexpr uint32_t L2_WEIGHT = 0x400000;")
        lines.append(f"static constexpr uint32_t L2_RESULT = 0x800000;")
        lines.append(f"")
        lines.append(f"// DDR staging buffers (for NumCpp bridge pattern)")
        lines.append(f"static constexpr uint64_t DDR_STAGING_IN     = 0x50000000ULL;")
        lines.append(f"static constexpr uint64_t DDR_STAGING_A      = 0x50000000ULL;")
        lines.append(f"static constexpr uint64_t DDR_STAGING_B      = 0x54000000ULL;")
        lines.append(f"static constexpr uint64_t DDR_STAGING_OUT    = 0x58000000ULL;")
        lines.append(f"static constexpr uint64_t DDR_STAGING_RESULT = 0x58000000ULL;")
        lines.append(f"")
        lines.append(f"// DDR weight/input/output addresses")
        lines.append(
            f"static constexpr uint64_t DDR_INPUT_BASE  = 0x{DDR_INPUT_BASE:08X}ULL;"
        )
        lines.append(
            f"static constexpr uint64_t DDR_OUTPUT_BASE = 0x{DDR_OUTPUT_BASE:08X}ULL;"
        )
        # DDR_WEIGHT_BASE is provided by weight_map.h as a #define — do not redefine here
        lines.append(f"")

        # Module table comment
        lines.append(f"// =============================================")
        lines.append(f"// Module table")
        lines.append(f"// =============================================")
        for i, node in enumerate(self.graph.nodes):
            if node.op.type == GTX_OP.RETURN:
                continue
            op_name = (
                str(node.op.type).split(".")[-1]
                if hasattr(node.op.type, "name")
                else str(node.op.type)
            )
            lines.append(f"//   [{i:2d}] {op_name:<20s} {node.name}")
        lines.append(f"")

        # Inference function
        lines.append(f"// =============================================")
        lines.append(f"// Inference function")
        lines.append(f"// =============================================")
        lines.append(f"void {self.model_name}_inference() {{")
        lines.append(f'    sc_printf("=== {self.model_name} inference start ===\\n");')
        lines.append(f"")

        for line in self._inference_lines:
            lines.append(line)
            lines.append(f"")

        lines.append(f'    sc_printf("=== {self.model_name} inference done ===\\n");')
        lines.append(f"}}")
        lines.append(f"")

        # main()
        lines.append(f"// =============================================")
        lines.append(f"// Entry point")
        lines.append(f"// =============================================")
        lines.append(f"int main() {{")
        lines.append(f"    {self.model_name}_inference();")
        lines.append(f"    __halt();")
        lines.append(f"    return 0;")
        lines.append(f"}}")
        lines.append(f"")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        return path
