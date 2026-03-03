#!/usr/bin/env python3
# Copyright (C) Supergate - All Rights Reserved
# GTX Compiler - PyTorch Model → C 소스코드 변환 파이프라인
#
# 사용법:
#   python compile_to_c.py --model export1/ResNet.py --output output/
#
# 전체 흐름:
#   1. PyTorch 모델 로드
#   2. TorchParser → GTX Graph 파싱
#   3. prepare_quantizable_module → 모듈+그래프 동기화
#   4. ModuleHooker.update_blobs_once → shape/param 데이터 채우기
#   5. CCodeGenerator → C 소스코드 생성 (.c/.h)
#   6. (선택) RISC-V 크로스 컴파일 → .elf

import os
import sys
import argparse
import importlib.util
import torch
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _init_op_map():
    """
    gtx_op → torch_op 매핑 테이블을 초기화합니다.
    export1/ResNet.py 같은 사전 내보낸 모델을 로드하기 전에 호출해야 합니다.

    NOTE: TorchParser가 먼저 실행된 경우, 일부 op 매핑이 TorchScript 내부 이름으로
    등록됩니다 (예: "BatchNorm" vs "BatchNorm2d"). add_mapping_item()은 이미
    등록된 키를 건너뛰므로, 여기서 강제 덮어쓰기를 수행합니다.
    """
    from gtx_utils.op_register import op_register
    from gtx_utils.gtx2torch_op_map import _GTX_OP_2_TORCH_OP, _TORCH_OP_2_GTX_OP
    from gtx_utils.torch_op_attr import gen_attr as _gen_attr, _TORCH_OP_ATTR_MAP
    from gtx_shared.base.key_names import GTX_OP

    # (GTX_OP attr name, torch_op name) 쌍으로 정의
    # getattr을 사용하여 존재하지 않는 속성은 건너뜀
    op_defs = [
        ("INPUT", None),  # INPUT→INPUT
        ("RETURN", None),  # RETURN→RETURN
        ("CONV2D", "Conv2d"),
        ("CONVTRANSPOSE2D", "ConvTranspose2d"),
        ("DEPTHWISE_CONV2D", "Conv2d"),
        ("BATCH_NORM", "BatchNorm2d"),
        ("INSTANCE_NORM", "InstanceNorm2d"),
        ("LAYER_NORM", "LayerNorm"),
        ("GROUP_NORM", "GroupNorm"),
        ("RELU", "ReLU"),
        ("RELU6", "ReLU6"),
        ("LEAKY_RELU", "LeakyReLU"),
        ("PRELU", "PReLU"),
        ("GELU", "GELU"),
        ("MISH", "Mish"),
        ("TANH", "Tanh"),
        ("HARDTANH", "Hardtanh"),
        ("SIGMOID", "Sigmoid"),
        ("HSIGMOID", "Hardsigmoid"),
        ("HSWISH", "Hardswish"),
        ("SELU", "SELU"),
        ("SOFTMAX", "Softmax"),
        ("DENSE", "Linear"),
        ("MAX_POOL", "MaxPool2d"),
        ("MAX_POOL1D", "MaxPool1d"),
        ("AVG_POOL", "AvgPool2d"),
        ("ADAPTIVEAVGPOOL2D", "AdaptiveAvgPool2d"),
        ("FLATTEN", "flatten"),
        ("ADD", "add"),
        ("MULTIPLY", "mul"),
        ("DIV", "div"),
        ("CONCAT", "cat"),
        ("RESHAPE", "reshape"),
        ("PERMUTE", "permute"),
        ("TRANSPOSE", "transpose"),
        ("CONTIGUOUS", "contiguous"),
        ("DROPOUT", "Dropout"),
        ("SHAPE", "size"),
        ("CHUNK", "chunk"),
        ("CONST", "tensor"),
        ("TENSOR", "tensor"),
        ("RESIZE", "interpolate"),
        ("PAD", "pad"),
        ("PIXEL_SHUFFLE", "PixelShuffle"),
        ("CHANNEL_SHUFFLE", "channel_shuffle"),
        ("EMBEDDING", "Embedding"),
    ]

    for attr_name, torch_op in op_defs:
        gtx_op = getattr(GTX_OP, attr_name, None)
        if gtx_op is None:
            continue
        if torch_op is None:
            torch_op = gtx_op  # INPUT→INPUT, RETURN→RETURN

        # 강제 덮어쓰기: TorchParser가 먼저 등록한 이름(예: "BatchNorm")을
        # 올바른 이름(예: "BatchNorm2d")으로 교체
        if gtx_op in _GTX_OP_2_TORCH_OP:
            old_torch_op = _GTX_OP_2_TORCH_OP[gtx_op]
            if old_torch_op != torch_op:
                # 이전 역방향 매핑 제거
                _TORCH_OP_2_GTX_OP.pop(old_torch_op, None)
                # 새로운 매핑으로 교체
                _GTX_OP_2_TORCH_OP[gtx_op] = torch_op
                _TORCH_OP_2_GTX_OP[torch_op] = gtx_op
                # torch_op_attr 테이블에도 새 이름 등록
                if torch_op not in _TORCH_OP_ATTR_MAP:
                    try:
                        _gen_attr(
                            torch_op,
                            force_to_primitive=False,
                            schema=None,
                            class_type=None,
                        )
                    except Exception:
                        pass

        try:
            op_register(gtx_op, torch_op)
        except Exception:
            pass


def load_model_from_file(model_path: str) -> torch.nn.Module:
    """
    Python 파일에서 모델 클래스를 동적으로 로드.
    export1/ResNet.py 등의 파일에서 모델을 가져옵니다.
    """
    # op 매핑 테이블 초기화 (모델 로드 전에 필요)
    _init_op_map()

    spec = importlib.util.spec_from_file_location("model_module", model_path)
    module = importlib.util.module_from_spec(spec)  # ty:ignore[invalid-argument-type]
    spec.loader.exec_module(module)  # ty:ignore[unresolved-attribute]

    # 모듈에서 첫 번째 nn.Module 서브클래스 찾기
    model_class = None
    for name, obj in module.__dict__.items():
        if (
            isinstance(obj, type)
            and issubclass(obj, torch.nn.Module)
            and obj is not torch.nn.Module
        ):
            model_class = obj
            break

    if model_class is None:
        raise ValueError(
            f"모델 파일 '{model_path}'에서 nn.Module 서브클래스를 찾을 수 없습니다."
        )

    model = model_class()
    model.eval()
    return model


def parse_model_to_graph(model, input_tensor):
    """
    PyTorch 모델을 GTX Graph로 파싱합니다.
    먼저 TorchParser를 시도하고, 실패하면 직접 모델 구조를 분석합니다.
    """
    try:
        from parse.parser import TorchParser, StandardInputData
        from gtx_shared.base.key_names import gtx_KEYS
        from gtx_shared.utils import GLOBAL_MAP

        parser = TorchParser()
        input_data = StandardInputData((input_tensor,), {})
        graph = parser(model.__class__.__name__, model, input_data)
        return graph
    except Exception as e:
        print(f"  ⚠ TorchParser 실패: {e}")
        print("  → 직접 모델 구조 분석으로 전환합니다...")
        return build_graph_from_model(model, input_tensor)


def build_graph_from_model(model, input_tensor):
    """
    사전 내보낸 GTX 모델(export1/ResNet.py 등)에서 직접 그래프를 구성합니다.
    TorchParser를 사용하지 않고, __init__ 소스에서 gtx_type을 파싱하고,
    forward() 소스에서 data flow를 추출하며,
    shape를 분석적으로 계산하고, 실제 가중치를 추출하여 그래프를 만듭니다.
    """
    import inspect
    import re
    from gtx_shared.base.key_names import GTX_OP

    # ------------------------------------------------------------------
    # Mock 클래스 정의
    # ------------------------------------------------------------------
    class MockTensor:
        def __init__(self, name, shape, data=None):
            self.name = name
            self.shape = shape
            self.data = data
            self.ndim = len(shape) if shape else 0
            self.uses = []

    class MockOp:
        def __init__(self, op_type, params_dict, attrs_dict):
            self.type = op_type
            self._params = {}
            self._attrs = attrs_dict
            self.attrs = attrs_dict
            for pn, pd in params_dict.items():
                self._params[pn] = MockTensor(pn, list(pd.shape), pd)

        @property
        def params(self):
            return self._params

        def get_attr(self, name):
            return self._attrs.get(name)

    class MockNode:
        def __init__(self, name, op_type, in_shape, out_shape, params, attrs):
            self.name = name
            self.op = MockOp(op_type, params, attrs)
            self.in_tensors = [MockTensor(f"{name}_in", in_shape)] if in_shape else []
            self.out_tensors = [MockTensor(f"{name}_out", out_shape)]
            self._in_nodes = set()
            self._out_nodes = set()

        @property
        def in_nodes(self):
            return self._in_nodes

        @property
        def out_nodes(self):
            return self._out_nodes

    class MockGraph:
        def __init__(self, name):
            self.name = name
            self._nodes = []

        @property
        def nodes(self):
            return self._nodes

    # ------------------------------------------------------------------
    # 1) __init__ 소스에서 gtx_type과 kwargs를 직접 파싱
    # ------------------------------------------------------------------
    init_src = inspect.getsource(model.__class__.__init__)
    # 패턴: self.module_N = nn.Module('gtx_xxx' [, kwargs...])
    init_pattern = re.compile(
        r"self\.(module_\d+)\s*=\s*nn\.Module\(\s*'([^']+)'"
        r"(?:\s*,\s*(.*?))?\s*\)"
    )

    # module_name -> { 'gtx_type': str, 'kwargs': dict }
    init_info = {}
    for m in init_pattern.finditer(init_src):
        mod_name = m.group(1)
        gtx_type_str = m.group(2)
        kwargs_str = m.group(3)  # may be None

        parsed_kwargs = {}
        if kwargs_str:
            # Parse key=value pairs from the kwargs string.
            # Values can be lists [...], booleans, numbers, or strings.
            kv_pattern = re.compile(r"(\w+)\s*=\s*(\[[^\]]*\]|True|False|None|[^,\)]+)")
            for kv in kv_pattern.finditer(kwargs_str):
                k = kv.group(1)
                v_str = kv.group(2).strip()
                try:
                    v = eval(v_str)
                except Exception:
                    v = v_str
                parsed_kwargs[k] = v

        init_info[mod_name] = {
            "gtx_type": gtx_type_str,
            "kwargs": parsed_kwargs,
        }

    # ------------------------------------------------------------------
    # 2) forward() 소스 파싱으로 data flow 추출
    # ------------------------------------------------------------------
    fwd_src = inspect.getsource(model.forward)
    fwd_lines = fwd_src.strip().split("\n")
    input_shape = list(input_tensor.shape)

    # forward line pattern:
    #   output_xxx = self.module_N(...)
    assign_pattern = re.compile(r"\s*(output_\w+)\s*=\s*self\.(module_\d+)\((.+)\)")

    # Collect ordered forward calls with parsed arguments
    forward_calls = []  # list of (var_name, mod_name, call_args_str)
    for line in fwd_lines:
        m = assign_pattern.match(line)
        if m:
            forward_calls.append((m.group(1), m.group(2), m.group(3)))

    # ------------------------------------------------------------------
    # Helper: compute output shape analytically
    # ------------------------------------------------------------------
    def _compute_conv_out(in_shape, attrs):
        """Conv2d output: [N, OC, OH, OW]"""
        n = in_shape[0]
        oc = attrs.get("out_channels", in_shape[1])
        ih, iw = in_shape[2], in_shape[3]
        kh, kw = (
            attrs["kernel_size"]
            if isinstance(attrs.get("kernel_size"), list)
            else [attrs.get("kernel_size", 3)] * 2
        )
        sh, sw = (
            attrs["stride"]
            if isinstance(attrs.get("stride"), list)
            else [attrs.get("stride", 1)] * 2
        )
        ph, pw = (
            attrs["padding"]
            if isinstance(attrs.get("padding"), list)
            else [attrs.get("padding", 0)] * 2
        )
        dh, dw = (
            attrs["dilation"]
            if isinstance(attrs.get("dilation"), list)
            else [attrs.get("dilation", 1)] * 2
        )
        oh = (ih + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        ow = (iw + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        return [n, oc, oh, ow]

    def _compute_pool_out(in_shape, attrs):
        """MaxPool/AvgPool output: [N, C, OH, OW]"""
        n, c = in_shape[0], in_shape[1]
        ih, iw = in_shape[2], in_shape[3]
        ks = attrs.get("kernel_size", [3, 3])
        kh, kw = ks if isinstance(ks, list) else [ks, ks]
        st = attrs.get("stride", ks)  # default stride == kernel_size
        sh, sw = st if isinstance(st, list) else [st, st]
        pd = attrs.get("padding", [0, 0])
        ph, pw = pd if isinstance(pd, list) else [pd, pd]
        dl = attrs.get("dilation", [1, 1])
        dh, dw = dl if isinstance(dl, list) else [dl, dl]
        oh = (ih + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        ow = (iw + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        return [n, c, oh, ow]

    def _compute_adaptive_avgpool_out(in_shape, attrs):
        """AdaptiveAvgPool2d output: [N, C, OH, OW]"""
        n, c = in_shape[0], in_shape[1]
        os = attrs.get("output_size", [1, 1])
        oh, ow = os if isinstance(os, list) else [os, os]
        return [n, c, oh, ow]

    def _compute_flatten_out(in_shape, start_dim, end_dim):
        """Flatten output shape."""
        ndim = len(in_shape)
        if start_dim < 0:
            start_dim += ndim
        if end_dim < 0:
            end_dim += ndim
        # product of dims in [start_dim, end_dim]
        flat = 1
        for i in range(start_dim, end_dim + 1):
            flat *= in_shape[i]
        return list(in_shape[:start_dim]) + [flat] + list(in_shape[end_dim + 1 :])

    def _compute_dense_out(in_shape, attrs):
        """Dense/Linear output: [N, out_features]"""
        out_f = attrs.get("out_features", 1)
        if len(in_shape) == 1:
            return [out_f]
        return list(in_shape[:-1]) + [out_f]

    def _compute_out_shape(gtx_type, in_shape, attrs, call_args_str):
        """Dispatch shape computation based on op type."""
        if in_shape is None:
            return None
        if gtx_type in (GTX_OP.CONV2D, GTX_OP.DEPTHWISE_CONV2D, GTX_OP.CONVTRANSPOSE2D):
            return _compute_conv_out(in_shape, attrs)
        elif gtx_type in (GTX_OP.MAX_POOL, GTX_OP.AVG_POOL):
            return _compute_pool_out(in_shape, attrs)
        elif gtx_type == GTX_OP.ADAPTIVEAVGPOOL2D:
            return _compute_adaptive_avgpool_out(in_shape, attrs)
        elif gtx_type == GTX_OP.FLATTEN:
            sd = attrs.get("start_dim", 1)
            ed = attrs.get("end_dim", -1)
            return _compute_flatten_out(in_shape, sd, ed)
        elif gtx_type == GTX_OP.DENSE:
            return _compute_dense_out(in_shape, attrs)
        elif gtx_type in (
            GTX_OP.BATCH_NORM,
            GTX_OP.RELU,
            GTX_OP.RELU6,
            GTX_OP.LEAKY_RELU,
            GTX_OP.SIGMOID,
            GTX_OP.TANH,
            GTX_OP.GELU,
            GTX_OP.MISH,
            GTX_OP.HSWISH,
            GTX_OP.HSIGMOID,
            GTX_OP.SELU,
            GTX_OP.SOFTMAX,
            GTX_OP.PRELU,
            GTX_OP.HARDTANH,
            GTX_OP.DROPOUT,
            GTX_OP.INSTANCE_NORM,
            GTX_OP.LAYER_NORM,
            GTX_OP.GROUP_NORM,
        ):
            return list(in_shape)
        elif gtx_type in (GTX_OP.ADD, GTX_OP.MULTIPLY):
            return list(in_shape)
        elif gtx_type == GTX_OP.RESHAPE:
            # try to parse target shape from call args
            shape_match = re.search(r"shape=\[([^\]]+)\]", call_args_str)
            if shape_match:
                try:
                    return [int(x.strip()) for x in shape_match.group(1).split(",")]
                except Exception:
                    pass
            return list(in_shape)
        else:
            # default: passthrough
            return list(in_shape)

    # ------------------------------------------------------------------
    # Helper: extract params and attrs from submodule for a given gtx_type
    # ------------------------------------------------------------------
    def _extract_params_and_attrs(submod, gtx_type, init_kwargs, call_args_str):
        """Extract numpy weight arrays and attribute dicts from a live submodule."""
        params = {}
        attrs = {}

        if gtx_type in (GTX_OP.CONV2D, GTX_OP.CONVTRANSPOSE2D, GTX_OP.DEPTHWISE_CONV2D):
            if hasattr(submod, "weight") and submod.weight is not None:
                params["weights"] = submod.weight.data.cpu().numpy()
            if hasattr(submod, "bias") and submod.bias is not None:
                params["bias"] = submod.bias.data.cpu().numpy()
            # Prefer init_kwargs (from parsed source) for attributes;
            # fall back to submodule attributes.
            attrs = {
                "kernel_size": init_kwargs.get(
                    "kernel_size",
                    list(submod.kernel_size)
                    if hasattr(submod, "kernel_size")
                    else [3, 3],
                ),
                "stride": init_kwargs.get(
                    "stride",
                    list(submod.stride) if hasattr(submod, "stride") else [1, 1],
                ),
                "padding": init_kwargs.get(
                    "padding",
                    list(submod.padding) if hasattr(submod, "padding") else [0, 0],
                ),
                "dilation": init_kwargs.get(
                    "dilation",
                    list(submod.dilation) if hasattr(submod, "dilation") else [1, 1],
                ),
                "groups": init_kwargs.get(
                    "groups", submod.groups if hasattr(submod, "groups") else 1
                ),
                "in_channels": init_kwargs.get(
                    "in_channels",
                    submod.in_channels if hasattr(submod, "in_channels") else 0,
                ),
                "out_channels": init_kwargs.get(
                    "out_channels",
                    submod.out_channels if hasattr(submod, "out_channels") else 0,
                ),
                "bias": init_kwargs.get(
                    "bias",
                    submod.bias is not None if hasattr(submod, "bias") else False,
                ),
            }

        elif gtx_type == GTX_OP.BATCH_NORM:
            if hasattr(submod, "weight") and submod.weight is not None:
                params["gamma"] = submod.weight.data.cpu().numpy()
            if hasattr(submod, "bias") and submod.bias is not None:
                params["beta"] = submod.bias.data.cpu().numpy()
            if hasattr(submod, "running_mean") and submod.running_mean is not None:
                params["mean"] = submod.running_mean.data.cpu().numpy()
            if hasattr(submod, "running_var") and submod.running_var is not None:
                params["var"] = submod.running_var.data.cpu().numpy()
            attrs = {
                "eps": init_kwargs.get(
                    "eps", submod.eps if hasattr(submod, "eps") else 1e-5
                ),
                "momentum": init_kwargs.get(
                    "momentum", submod.momentum if hasattr(submod, "momentum") else 0.1
                ),
                "num_features": init_kwargs.get(
                    "num_features",
                    submod.num_features if hasattr(submod, "num_features") else 0,
                ),
            }

        elif gtx_type == GTX_OP.DENSE:
            if hasattr(submod, "weight") and submod.weight is not None:
                params["weights"] = submod.weight.data.cpu().numpy()
            if hasattr(submod, "bias") and submod.bias is not None:
                params["bias"] = submod.bias.data.cpu().numpy()
            attrs = {
                "in_features": init_kwargs.get(
                    "in_features",
                    submod.in_features if hasattr(submod, "in_features") else 0,
                ),
                "out_features": init_kwargs.get(
                    "out_features",
                    submod.out_features if hasattr(submod, "out_features") else 0,
                ),
                "bias": init_kwargs.get(
                    "bias",
                    submod.bias is not None if hasattr(submod, "bias") else False,
                ),
            }

        elif gtx_type in (GTX_OP.MAX_POOL, GTX_OP.AVG_POOL):
            attrs = {
                "kernel_size": init_kwargs.get(
                    "kernel_size",
                    list(submod.kernel_size)
                    if hasattr(submod, "kernel_size")
                    else [3, 3],
                ),
                "stride": init_kwargs.get(
                    "stride",
                    list(submod.stride) if hasattr(submod, "stride") else [2, 2],
                ),
                "padding": init_kwargs.get(
                    "padding",
                    list(submod.padding) if hasattr(submod, "padding") else [0, 0],
                ),
                "dilation": init_kwargs.get(
                    "dilation",
                    list(submod.dilation) if hasattr(submod, "dilation") else [1, 1],
                ),
                "ceil_mode": init_kwargs.get("ceil_mode", False),
            }

        elif gtx_type == GTX_OP.ADAPTIVEAVGPOOL2D:
            attrs = {
                "output_size": init_kwargs.get(
                    "output_size",
                    list(submod.output_size)
                    if hasattr(submod, "output_size")
                    else [1, 1],
                ),
            }

        elif gtx_type == GTX_OP.FLATTEN:
            # Extract start_dim/end_dim from the forward() call args
            sd_match = re.search(r"start_dim=(\S+)", call_args_str)
            ed_match = re.search(r"end_dim=(\S+)", call_args_str)
            attrs = {
                "start_dim": int(sd_match.group(1).rstrip(",)")) if sd_match else 1,
                "end_dim": int(ed_match.group(1).rstrip(",)")) if ed_match else -1,
            }

        elif gtx_type in (
            GTX_OP.RELU,
            GTX_OP.RELU6,
            GTX_OP.LEAKY_RELU,
            GTX_OP.SIGMOID,
            GTX_OP.TANH,
            GTX_OP.GELU,
            GTX_OP.SOFTMAX,
            GTX_OP.MISH,
            GTX_OP.HSWISH,
            GTX_OP.HSIGMOID,
            GTX_OP.SELU,
            GTX_OP.PRELU,
            GTX_OP.HARDTANH,
            GTX_OP.DROPOUT,
        ):
            attrs = dict(init_kwargs)
            # PReLU has a learnable weight
            if gtx_type == GTX_OP.PRELU:
                if hasattr(submod, "weight") and submod.weight is not None:
                    params["weights"] = submod.weight.data.cpu().numpy()

        elif gtx_type == GTX_OP.LAYER_NORM:
            if hasattr(submod, "weight") and submod.weight is not None:
                params["gamma"] = submod.weight.data.cpu().numpy()
            if hasattr(submod, "bias") and submod.bias is not None:
                params["beta"] = submod.bias.data.cpu().numpy()
            attrs = dict(init_kwargs)

        elif gtx_type == GTX_OP.GROUP_NORM:
            if hasattr(submod, "weight") and submod.weight is not None:
                params["gamma"] = submod.weight.data.cpu().numpy()
            if hasattr(submod, "bias") and submod.bias is not None:
                params["beta"] = submod.bias.data.cpu().numpy()
            attrs = dict(init_kwargs)

        elif gtx_type == GTX_OP.INSTANCE_NORM:
            if hasattr(submod, "weight") and submod.weight is not None:
                params["gamma"] = submod.weight.data.cpu().numpy()
            if hasattr(submod, "bias") and submod.bias is not None:
                params["beta"] = submod.bias.data.cpu().numpy()
            if hasattr(submod, "running_mean") and submod.running_mean is not None:
                params["mean"] = submod.running_mean.data.cpu().numpy()
            if hasattr(submod, "running_var") and submod.running_var is not None:
                params["var"] = submod.running_var.data.cpu().numpy()
            attrs = dict(init_kwargs)

        elif gtx_type == GTX_OP.EMBEDDING:
            if hasattr(submod, "weight") and submod.weight is not None:
                params["weights"] = submod.weight.data.cpu().numpy()
            attrs = dict(init_kwargs)

        elif gtx_type in (GTX_OP.ADD, GTX_OP.MULTIPLY):
            attrs = dict(init_kwargs)

        elif gtx_type == GTX_OP.CONCAT:
            # axis/dim may be in call args
            dim_match = re.search(r"dim=(\S+)", call_args_str)
            attrs = {
                "dim": int(dim_match.group(1).rstrip(",)")) if dim_match else 1,
            }

        else:
            # Generic: store any kwargs from __init__
            attrs = dict(init_kwargs)

        return params, attrs

    # ------------------------------------------------------------------
    # Helper: resolve input node from call args
    # ------------------------------------------------------------------
    def _resolve_input_node(call_args_str, var_map, input_node):
        """Return (primary_input_node, secondary_input_node_or_None) from call_args."""
        primary = None
        secondary = None

        # Check for kwarg form: input=<var>
        inp_match = re.search(r"input=(\w+)", call_args_str)
        if inp_match:
            vname = inp_match.group(1)
            if vname == "args[0]":
                primary = input_node
            elif vname in var_map:
                primary = var_map[vname]

            # Check for other=<var> (elemwise_add)
            other_match = re.search(r"other=(\w+)", call_args_str)
            if other_match:
                oname = other_match.group(1)
                if oname in var_map:
                    secondary = var_map[oname]
        else:
            # Positional form: first arg is the input variable
            first_arg = call_args_str.split(",")[0].strip()
            if first_arg == "args[0]" or first_arg == "*args":
                primary = input_node
            elif first_arg in var_map:
                primary = var_map[first_arg]

        return primary, secondary

    # ------------------------------------------------------------------
    # 3) Build graph
    # ------------------------------------------------------------------
    graph = MockGraph(model.__class__.__name__)

    # INPUT node
    input_node = MockNode("input_0", GTX_OP.INPUT, None, input_shape, {}, {})
    graph._nodes.append(input_node)

    var_map = {}  # variable_name -> MockNode
    var_map["args[0]"] = input_node
    node_map = {}  # module_name -> MockNode

    for var_name, mod_name, call_args_str in forward_calls:
        submod = getattr(model, mod_name, None)
        if submod is None:
            continue

        # --- Determine gtx_type ---
        info = init_info.get(mod_name)
        if info is not None:
            gtx_type = info["gtx_type"]
            init_kwargs = info["kwargs"]
        else:
            # Fallback: read from submodule's torch_op_type attribute
            gtx_type = getattr(submod, "torch_op_type", type(submod).__name__)
            init_kwargs = {}

        # --- Skip INPUT nodes (passthrough) ---
        if gtx_type == GTX_OP.INPUT:
            # INPUT just passes through args[0]
            inp_node, _ = _resolve_input_node(call_args_str, var_map, input_node)
            var_map[var_name] = inp_node if inp_node is not None else input_node
            continue

        # --- Resolve input connections ---
        primary_in, secondary_in = _resolve_input_node(
            call_args_str, var_map, input_node
        )

        # Determine input shape from the primary input node
        if primary_in is not None and primary_in.out_tensors:
            in_shape = list(primary_in.out_tensors[0].shape)
        else:
            in_shape = list(input_shape)

        # --- Extract params and attrs ---
        params, attrs = _extract_params_and_attrs(
            submod, gtx_type, init_kwargs, call_args_str
        )

        # --- Compute output shape analytically ---
        out_shape = _compute_out_shape(gtx_type, in_shape, attrs, call_args_str)
        if out_shape is None:
            out_shape = list(in_shape)

        # --- Create node ---
        node = MockNode(
            f"{mod_name}_{gtx_type}",
            gtx_type,
            in_shape,
            out_shape,
            params,
            attrs,
        )

        # --- Wire connections ---
        if primary_in is not None:
            node._in_nodes.add(primary_in)
            primary_in._out_nodes.add(node)

        if secondary_in is not None:
            node._in_nodes.add(secondary_in)
            secondary_in._out_nodes.add(node)

        # --- Register ---
        graph._nodes.append(node)
        node_map[mod_name] = node
        var_map[var_name] = node

    return graph


def connect_and_fill_graph(model, graph, input_tensor):
    """
    모델과 그래프를 연결하고, forward 실행으로 shape/param 데이터를 채웁니다.
    """
    from qproc.ModuleHooker import ModuleHooker
    from qproc.utils import connect_module_with_graph

    try:
        # 모듈-그래프 연결
        connect_module_with_graph(model, graph)

        # Forward 실행으로 shape/param 채우기
        ModuleHooker.update_blobs_once(model, graph, update_shape_only=False)
    except Exception as e:
        print(f"경고: 모듈-그래프 동기화 중 오류 발생: {e}")
        print("shape/param 데이터 없이 계속 진행합니다.")

        # Fallback: 직접 shape 정보 추론
        _fill_shapes_from_model(model, graph, input_tensor)


def _fill_shapes_from_model(model, graph, input_tensor):
    """
    ModuleHooker가 실패할 경우 직접 forward 실행으로 shape 추론.
    """
    try:
        with torch.no_grad():
            # 중간 출력을 캡처하기 위한 hook 등록
            shapes = {}
            hooks = []

            def make_hook(name):
                def hook_fn(module, input, output):
                    if isinstance(output, torch.Tensor):
                        shapes[name] = list(output.shape)
                    elif isinstance(output, (tuple, list)) and len(output) > 0:
                        if isinstance(output[0], torch.Tensor):
                            shapes[name] = list(output[0].shape)

                return hook_fn

            for name, mod in model.named_modules():
                hooks.append(mod.register_forward_hook(make_hook(name)))

            model(input_tensor)

            # hook 제거
            for h in hooks:
                h.remove()

            # 그래프 노드에 shape 할당
            for node in graph.nodes:
                if node.name in shapes and node.out_tensors:
                    node.out_tensors[0].shape = shapes[node.name]
    except Exception as e:
        print(f"경고: shape 추론 실패: {e}")


def generate_c_code(
    graph, output_dir: str, model_name: str = "model", nest_id: int = 0, spu_id: int = 0
):
    """
    GTX Graph → C 소스코드 생성.

    Args:
        nest_id: 타겟 NEST ID (0-3). 기본값 0.
        spu_id: 타겟 SPU ID (0-3). 기본값 0.
    """
    from gtx_shared.compile.c_codegen import CCodeGenerator

    codegen = CCodeGenerator(
        graph,
        output_dir=output_dir,
        model_name=model_name,
        nest_id=nest_id,
        spu_id=spu_id,
    )
    files = codegen.generate()

    return files


def generate_cpp_code(graph, output_dir: str, model_name: str = "model"):
    """GTX Graph → NumCpp C++ 소스코드 생성."""
    from gtx_shared.compile.cpp_codegen import CppCodeGenerator

    codegen = CppCodeGenerator(
        graph,
        output_dir=output_dir,
        model_name=model_name,
    )
    files = codegen.generate()
    return files


def generate_numcpp_makefile(output_dir: str, model_name: str = "model"):
    """NumCpp C++ 백엔드용 RISC-V 크로스 컴파일 Makefile 생성."""
    # NumCpp 루트는 이 스크립트(compile_to_c.py)가 위치한 프로젝트 루트의 NumCpp/ 디렉토리
    project_root_abs = os.path.dirname(os.path.abspath(__file__))
    numcpp_root = os.path.join(project_root_abs, "NumCpp")

    makefile_content = f"""# Auto-generated by GTX Compiler (NumCpp backend)
# RISC-V Cross-compilation Makefile for {model_name} (C++)

# Toolchain
CC      = riscv64-unknown-elf-gcc
CXX     = riscv64-unknown-elf-g++
OBJDUMP = riscv64-unknown-elf-objdump
OBJCOPY = riscv64-unknown-elf-objcopy
SIZE    = riscv64-unknown-elf-size

# Architecture
ARCH_FLAGS = -march=rv64g -mabi=lp64d -mcmodel=large

# Include paths
INC_PATHS = -I{numcpp_root}/inc \\
            -I{numcpp_root}/include \\
            -I{os.path.abspath(output_dir)}

# Common flags
COMMON_FLAGS = $(ARCH_FLAGS) -O0 -g -nostartfiles \\
               -ffunction-sections -fdata-sections \\
               $(INC_PATHS)

# Defines
DEFINES = -DNUMCPP_USE_GTX -DGTX_USE_GTX

# C flags (for intrinsic .c sources)
CFLAGS = $(COMMON_FLAGS) $(DEFINES) -std=c11 \\
         -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function

# C++ flags
CXXFLAGS = $(COMMON_FLAGS) $(DEFINES) -std=c++20 \\
           -fno-exceptions -fno-rtti \\
           -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function

# Linker flags
LDFLAGS = -T {numcpp_root}/examples/numcpp_gtx/linker.ld -nostdlib -Wl,--gc-sections -lm -lc -lgcc

# GTX intrinsic sources (C)
GTX_C_SRCS = {numcpp_root}/src/intrin_level1.c \\
             {numcpp_root}/src/intrin_level2.c \\
             {numcpp_root}/src/intrin_level3.c \\
             {numcpp_root}/src/gtx_utils.c \\
             {numcpp_root}/src/sc_print.c

# C++ runtime and startup
STARTUP_SRCS = {numcpp_root}/examples/numcpp_gtx/startup.c \\
               {numcpp_root}/examples/numcpp_gtx/soft_math.c
CXXRT_SRC    = {numcpp_root}/examples/numcpp_gtx/cxxrt.cpp

# Model source
MODEL_SRC = {model_name}.cpp

# Build directory
BUILD_DIR = build

# Object files
GTX_OBJS    = $(patsubst %.c,$(BUILD_DIR)/%.o,$(notdir $(GTX_C_SRCS)))
STARTUP_OBJS = $(patsubst %.c,$(BUILD_DIR)/%.o,$(notdir $(STARTUP_SRCS)))
CXXRT_OBJ   = $(BUILD_DIR)/cxxrt.o
MODEL_OBJ   = $(BUILD_DIR)/{model_name}.o

TARGET = {model_name}.elf

all: $(TARGET)

$(TARGET): $(MODEL_OBJ) $(GTX_OBJS) $(STARTUP_OBJS) $(CXXRT_OBJ)
\t@echo "[LINK] $@"
\t$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
\t@$(SIZE) $@

# Model C++ source
$(MODEL_OBJ): $(MODEL_SRC)
\t@mkdir -p $(BUILD_DIR)
\t@echo "[CXX]  $<"
\t$(CXX) $(CXXFLAGS) -c $< -o $@

# C++ runtime
$(CXXRT_OBJ): $(CXXRT_SRC)
\t@mkdir -p $(BUILD_DIR)
\t@echo "[CXX]  $<"
\t$(CXX) $(CXXFLAGS) -c $< -o $@

# GTX intrinsic C sources
$(BUILD_DIR)/intrin_level1.o: {numcpp_root}/src/intrin_level1.c
\t@mkdir -p $(BUILD_DIR)
\t$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/intrin_level2.o: {numcpp_root}/src/intrin_level2.c
\t@mkdir -p $(BUILD_DIR)
\t$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/intrin_level3.o: {numcpp_root}/src/intrin_level3.c
\t@mkdir -p $(BUILD_DIR)
\t$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/gtx_utils.o: {numcpp_root}/src/gtx_utils.c
\t@mkdir -p $(BUILD_DIR)
\t$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/sc_print.o: {numcpp_root}/src/sc_print.c
\t@mkdir -p $(BUILD_DIR)
\t$(CC) $(CFLAGS) -c $< -o $@

# Startup C sources
$(BUILD_DIR)/startup.o: {numcpp_root}/examples/numcpp_gtx/startup.c
\t@mkdir -p $(BUILD_DIR)
\t$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/soft_math.o: {numcpp_root}/examples/numcpp_gtx/soft_math.c
\t@mkdir -p $(BUILD_DIR)
\t$(CC) $(CFLAGS) -c $< -o $@

dump: $(TARGET)
\t$(OBJDUMP) -d $(TARGET) > {model_name}.dump

bin: $(TARGET)
\t$(OBJCOPY) -O binary $(TARGET) {model_name}.bin

clean:
\trm -rf $(BUILD_DIR) $(TARGET) {model_name}.dump {model_name}.bin

.PHONY: all dump bin clean
"""

    filepath = os.path.join(output_dir, "Makefile")
    with open(filepath, "w") as f:
        f.write(makefile_content)

    return filepath


def generate_makefile(output_dir: str, model_name: str = "model"):
    """
    RISC-V 크로스 컴파일용 Makefile 생성.
    """
    project_root_abs = os.path.abspath(os.path.join(output_dir, ".."))

    makefile_content = f"""# Auto-generated by GTX Compiler
# RISC-V Cross-compilation Makefile for {model_name}

# RISC-V 크로스 컴파일러 설정
CC = riscv64-unknown-elf-gcc
OBJDUMP = riscv64-unknown-elf-objdump
OBJCOPY = riscv64-unknown-elf-objcopy

# 컴파일 옵션
CFLAGS = -O2 -march=rv64gc -mabi=lp64d -I{project_root_abs}/nn/include -I{project_root_abs}/nn/include/gtx
LDFLAGS = -T linker.ld -nostdlib -static

# 소스 파일
STARTUP_SRC = crt0.S
DATA_SRC = data_embed.S
MODEL_SRC = {model_name}.c
INTRIN_SRC = {project_root_abs}/nn/src/gtx/intrin_level1.c \\
             {project_root_abs}/nn/src/gtx/intrin_level2.c \\
             {project_root_abs}/nn/src/gtx/intrin_level3.c \\
             {project_root_abs}/nn/src/gtx/gtx_utils.c

# 출력 파일
TARGET = {model_name}.elf

# 기본 타겟
all: $(TARGET)

$(TARGET): $(STARTUP_SRC) $(DATA_SRC) $(MODEL_SRC) $(INTRIN_SRC) linker.ld weights.bin
\t$(CC) $(CFLAGS) -o $@ $(STARTUP_SRC) $(DATA_SRC) $(MODEL_SRC) $(INTRIN_SRC) $(LDFLAGS)
\t@echo "Build complete: $(TARGET)"

# 역어셈블
dump: $(TARGET)
\t$(OBJDUMP) -d $(TARGET) > {model_name}.dump

# 바이너리 추출
bin: $(TARGET)
\t$(OBJCOPY) -O binary $(TARGET) {model_name}.bin

# 정리
clean:
\trm -f $(TARGET) {model_name}.dump {model_name}.bin

.PHONY: all dump bin clean
"""

    filepath = os.path.join(output_dir, "Makefile")
    with open(filepath, "w") as f:
        f.write(makefile_content)

    return filepath


def generate_modules_makefile(output_dir: str, model_name: str = "model"):
    """
    Per-module RISC-V 크로스 컴파일용 Makefile 생성.
    output/modules/ 디렉토리에 각 module_N.elf를 빌드하는 Makefile을 만듭니다.
    """
    modules_dir = os.path.join(output_dir, "modules")
    if not os.path.isdir(modules_dir):
        return None

    # modules 디렉토리에서 module_N.c 파일 목록을 스캔
    module_files = sorted(
        [
            f
            for f in os.listdir(modules_dir)
            if f.startswith("module_") and f.endswith(".c")
        ]
    )

    if not module_files:
        return None

    # module_N.c → module_N.elf 타겟 목록
    module_targets = [f.replace(".c", ".elf") for f in module_files]
    module_names = [f.replace(".c", "") for f in module_files]

    project_root_abs = os.path.abspath(os.path.join(output_dir, ".."))
    parent_dir = os.path.abspath(output_dir)

    targets_str = " ".join(module_targets)

    # 개별 빌드 규칙
    individual_rules = []
    for mf, mt in zip(module_files, module_targets):
        individual_rules.append(
            f"{mt}: {mf}\n"
            f"\t$(CC) $(CFLAGS) -o $@ $(STARTUP_SRC) $(DATA_SRC) $< $(INTRIN_SRC) $(LDFLAGS)\n"
            f'\t@echo "Built: $@"'
        )

    rules_str = "\n\n".join(individual_rules)

    makefile_content = f"""# Auto-generated by GTX Compiler
# Per-module RISC-V Cross-compilation Makefile
# Each module_N.c is compiled into a standalone module_N.elf

# RISC-V 크로스 컴파일러 설정
CC = riscv64-unknown-elf-gcc
OBJDUMP = riscv64-unknown-elf-objdump
OBJCOPY = riscv64-unknown-elf-objcopy

# 컴파일 옵션
# Include paths: project headers + parent output dir (for model.h, weights.h, weight_map.h)
CFLAGS = -O2 -march=rv64gc -mabi=lp64d \\
         -I{project_root_abs}/nn/include \\
         -I{project_root_abs}/nn/include/gtx \\
         -I{parent_dir}
LDFLAGS = -T {parent_dir}/linker.ld -nostdlib -static

# Startup 코드 및 DDR 데이터 임베딩
STARTUP_SRC = {parent_dir}/crt0.S
DATA_SRC = {parent_dir}/data_embed.S

# Intrinsic 소스 파일
INTRIN_SRC = {project_root_abs}/nn/src/gtx/intrin_level1.c \\
             {project_root_abs}/nn/src/gtx/intrin_level2.c \\
             {project_root_abs}/nn/src/gtx/intrin_level3.c \\
             {project_root_abs}/nn/src/gtx/gtx_utils.c

# 모든 모듈 타겟
MODULES = {targets_str}

# 기본 타겟: 모든 모듈 빌드
all: $(MODULES)
\t@echo "All modules built successfully."

# 개별 모듈 빌드 규칙
{rules_str}

# 정리
clean:
\trm -f $(MODULES)
\t@echo "Cleaned all module ELFs."

.PHONY: all clean
"""

    filepath = os.path.join(modules_dir, "Makefile")
    with open(filepath, "w") as f:
        f.write(makefile_content)

    return filepath


def generate_linker_script(output_dir: str):
    """
    RISC-V 링커 스크립트 생성.
    GTX 하드웨어 메모리 맵에 맞춤.
    _start 심볼은 crt0.S에서 정의되며, .text.init 섹션에 배치됨.
    """
    linker_content = """/* Auto-generated by GTX Compiler */
/* GTX RISC-V Linker Script */

OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    /* 명령어 메모리 (ITCM) */
    IMEM (rx)  : ORIGIN = 0x00000000, LENGTH = 1M
    
    /* 데이터 메모리 (DTCM) */  
    DMEM (rw)  : ORIGIN = 0x00100000, LENGTH = 1M
    
    /* L2 SPM */
    L2SPM (rw) : ORIGIN = 0x10000000, LENGTH = 16M
    
    /* DDR */
    DDR (rw)   : ORIGIN = 0x80000000, LENGTH = 2048M
}

SECTIONS
{
    .text : {
        KEEP(*(.text.init))     /* _start 진입점 (crt0.S) */
        *(.text._start)
        *(.text*)
    } > IMEM
    
    .rodata : {
        *(.rodata*)
    } > IMEM
    
    .data : {
        *(.data*)
        *(.sdata*)
    } > DMEM
    
    .bss (NOLOAD) : {
        __bss_start = .;
        *(.bss*)
        *(.sbss*)
        *(COMMON)
        __bss_end = .;
    } > DMEM
    
    /* 스택 (BSS 이후에 배치) */
    .stack (NOLOAD) : {
        . = ALIGN(16);
        __stack_bottom = .;
        . += 0x10000;  /* 64KB 스택 */
        __stack_top = .;
    } > DMEM
    
    /* DDR 영역: ELF PT_LOAD 세그먼트로 시뮬레이터 메모리에 직접 배치 */
    .ddr_input 0x80000000 : {
        *(.ddr_input)
    } > DDR

    .ddr_weights 0xA0000000 : {
        *(.ddr_weights)
    } > DDR

    /* gp 심볼 (small data 접근용) */
    __global_pointer$ = ADDR(.data) + 0x800;
}
"""

    filepath = os.path.join(output_dir, "linker.ld")
    with open(filepath, "w") as f:
        f.write(linker_content)

    return filepath


def generate_startup_code(output_dir: str):
    """
    RISC-V bare-metal startup 코드(crt0.S) 생성.
    - _start 심볼 정의
    - gp (global pointer) 설정
    - sp (스택 포인터) 설정
    - BSS 영역 zero 초기화
    - main() 호출
    - main 반환 후 halt
    """
    crt0_content = r"""/* Auto-generated by GTX Compiler */
/* RISC-V Bare-Metal Startup Code (crt0.S) */
/* _start → sp/gp 설정 → BSS 클리어 → main() → halt */

    .section .text.init, "ax", @progbits
    .global _start
    .type   _start, @function

_start:
    /* ---- 1) Global Pointer 설정 ---- */
    .option push
    .option norelax
    la      gp, __global_pointer$
    .option pop

    /* ---- 2) Stack Pointer 설정 (16바이트 정렬) ---- */
    la      sp, __stack_top
    andi    sp, sp, -16

    /* ---- 3) BSS 영역 zero 초기화 ---- */
    la      t0, __bss_start
    la      t1, __bss_end
    bgeu    t0, t1, .Lbss_done
.Lbss_loop:
    sd      zero, 0(t0)
    addi    t0, t0, 8
    bltu    t0, t1, .Lbss_loop
.Lbss_done:

    /* ---- 4) main() 호출 ---- */
    /* a0 = argc = 0, a1 = argv = NULL */
    li      a0, 0
    li      a1, 0
    call    main

    /* ---- 5) main 반환 후 무한 루프 (안전장치) ---- */
    /* main()에서 __halt()를 호출하므로 여기 도달하지 않아야 함 */
.Lhalt_loop:
    j       .Lhalt_loop

    .size   _start, . - _start
"""

    filepath = os.path.join(output_dir, "crt0.S")
    with open(filepath, "w") as f:
        f.write(crt0_content)

    return filepath


def generate_data_embed(
    output_dir: str, weights_bin: str = "weights.bin", input_bin=None
):
    """
    DDR 데이터 임베딩용 어셈블리 파일 생성.

    .incbin 디렉티브로 weights.bin (및 선택적 input.bin)을
    DDR 주소에 배치되는 ELF 섹션에 내장한다.
    시뮬레이터가 ELF PT_LOAD 세그먼트로 직접 로드하므로
    -L 플래그 없이도 DDR 메모리에 데이터가 배치된다.
    """
    lines = []
    lines.append("/* Auto-generated by GTX Compiler */")
    lines.append("/* DDR data embedding via .incbin */")
    lines.append("")

    # weights.bin → .ddr_weights section (placed at 0xA0000000 by linker)
    lines.append('    .section .ddr_weights, "a", @progbits')
    lines.append("    .balign 4")
    lines.append("    .global _weights_start")
    lines.append("_weights_start:")
    lines.append(f'    .incbin "{weights_bin}"')
    lines.append("    .global _weights_end")
    lines.append("_weights_end:")
    lines.append("")

    # input.bin → .ddr_input section (placed at 0x80000000 by linker)
    if input_bin:
        lines.append('    .section .ddr_input, "a", @progbits')
        lines.append("    .balign 4")
        lines.append("    .global _input_start")
        lines.append("_input_start:")
        lines.append(f'    .incbin "{input_bin}"')
        lines.append("    .global _input_end")
        lines.append("_input_end:")
        lines.append("")

    filepath = os.path.join(output_dir, "data_embed.S")
    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")

    return filepath


def load_torch_model(model_spec: str):
    """
    다양한 형태의 PyTorch 모델을 로드합니다.

    지원 형태:
      - "resnet18" → torchvision.models.resnet18(pretrained=False)
      - "torchvision.models.resnet18" → 동적 임포트
      - "path/to/model.pt" → torch.load()
    """
    import importlib

    # 1) torchvision 내장 모델 이름 (resnet18, vgg16, etc.)
    try:
        import torchvision.models as tv_models

        if hasattr(tv_models, model_spec):
            model_fn = getattr(tv_models, model_spec)
            model = model_fn(weights=None)
            model.eval()
            print(f"  → torchvision 모델 로드: {model_spec}")
            return model
    except Exception:
        pass

    # 2) 전체 Python 경로 (예: torchvision.models.resnet18)
    if (
        "." in model_spec
        and not model_spec.endswith(".py")
        and not model_spec.endswith(".pt")
    ):
        try:
            parts = model_spec.rsplit(".", 1)
            module = importlib.import_module(parts[0])
            model_fn = getattr(module, parts[1])
            model = model_fn(weights=None) if callable(model_fn) else model_fn
            if isinstance(model, torch.nn.Module):
                model.eval()
                print(f"  → 동적 임포트 모델: {model_spec}")
                return model
        except Exception:
            pass

    # 3) .pt/.pth 파일
    if model_spec.endswith(".pt") or model_spec.endswith(".pth"):
        model = torch.load(model_spec, map_location="cpu")
        if isinstance(model, torch.nn.Module):
            model.eval()
            print(f"  → 저장된 모델 로드: {model_spec}")
            return model

    return None


def generate_export_python(graph, model, output_dir: str, model_name: str = "model"):
    """
    export1/ResNet.py 스타일의 Python 모듈 설명 파일을 생성합니다.
    TorchParser 그래프가 있으면 get_script_writer를 사용하고,
    MockGraph인 경우에는 직접 생성합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    export_file = os.path.join(output_dir, f"{model_name}.py")

    # TorchParser 그래프인 경우 기존 exporter 사용 시도
    if not hasattr(graph, "_nodes"):
        try:
            from qproc.export import get_script_writer

            exporter = get_script_writer(enable_quant=True)
            exporter.write(graph, file_path=export_file)
            print(f"  → Python 모듈 파일 (ScriptWriter): {export_file}")
            return export_file
        except Exception as e:
            print(f"  ⚠ ScriptWriter 실패: {e}, 직접 생성으로 전환")

    # MockGraph 또는 ScriptWriter 실패 시 직접 생성
    _write_export_python_from_graph(graph, model, export_file, model_name)
    print(f"  → Python 모듈 파일 (직접 생성): {export_file}")
    return export_file


def _write_export_python_from_graph(graph, model, filepath: str, model_name: str):
    """
    MockGraph에서 직접 export1/ResNet.py 스타일 파일을 생성합니다.
    """
    import re
    from gtx_shared.base.key_names import GTX_OP

    # GTX_OP → gtx_type 문자열 매핑
    op_to_gtx_type = {
        GTX_OP.INPUT: "gtx_input",
        GTX_OP.CONV2D: "gtx_conv2d",
        GTX_OP.DEPTHWISE_CONV2D: "gtx_depthwise_conv2d",
        GTX_OP.BATCH_NORM: "gtx_batch_norm",
        GTX_OP.RELU: "gtx_relu",
        GTX_OP.RELU6: "gtx_relu6",
        GTX_OP.LEAKY_RELU: "gtx_leaky_relu",
        GTX_OP.MAX_POOL: "gtx_maxpool",
        GTX_OP.AVG_POOL: "gtx_avgpool",
        GTX_OP.ADAPTIVEAVGPOOL2D: "gtx_adaptive_avg_pool2d",
        GTX_OP.ADD: "gtx_elemwise_add",
        GTX_OP.MULTIPLY: "gtx_elemwise_mul",
        GTX_OP.DENSE: "gtx_dense",
        GTX_OP.FLATTEN: "gtx_flatten",
        GTX_OP.SOFTMAX: "gtx_softmax",
        GTX_OP.SIGMOID: "gtx_sigmoid",
        GTX_OP.GELU: "gtx_gelu",
        GTX_OP.TANH: "gtx_tanh",
        GTX_OP.CONCAT: "gtx_concat",
        GTX_OP.RESHAPE: "gtx_reshape",
        GTX_OP.LAYER_NORM: "gtx_layer_norm",
        GTX_OP.GROUP_NORM: "gtx_group_norm",
        GTX_OP.INSTANCE_NORM: "gtx_instance_norm",
        GTX_OP.DROPOUT: "gtx_dropout",
        GTX_OP.CONVTRANSPOSE2D: "gtx_conv_transpose2d",
    }

    class_name = model.__class__.__name__ if model else model_name.capitalize()

    lines = []
    lines.append("# GENETARED BY SuperGate GTX Compiler, DO NOT EDIT!")
    lines.append("")
    lines.append("import torch")
    lines.append("from torch import tensor")
    lines.append("import nn")
    lines.append("")
    lines.append(f"class {class_name}(nn.GTX_QuantModel):")
    lines.append("    def __init__(self):")
    lines.append(f"        super({class_name}, self).__init__()")

    # 노드를 순회하며 module 등록
    node_list = list(graph.nodes)
    mod_id_map = {}  # node_name → mod_id
    mod_nodes = []  # (mod_id, node) 순서대로

    for node in node_list:
        match = re.match(r"module_(\d+)_", node.name)
        if match:
            mod_id = int(match.group(1))
        elif node.name == "input_0":
            mod_id = 0
        else:
            continue

        mod_id_map[node.name] = mod_id
        mod_nodes.append((mod_id, node))

    # __init__ 부분: self.module_N = nn.Module(...)
    for mod_id, node in sorted(mod_nodes, key=lambda x: x[0]):
        gtx_type = op_to_gtx_type.get(node.op.type, str(node.op.type))
        attrs = node.op.attrs if hasattr(node.op, "attrs") else {}

        # kwargs 문자열 생성
        kwargs_parts = []
        for k, v in attrs.items():
            if isinstance(v, bool):
                kwargs_parts.append(f"{k}={v}")
            elif isinstance(v, (list, tuple)):
                kwargs_parts.append(f"{k}={list(v)}")
            elif isinstance(v, float):
                kwargs_parts.append(f"{k}={v}")
            elif isinstance(v, int):
                kwargs_parts.append(f"{k}={v}")
            elif isinstance(v, str):
                kwargs_parts.append(f"{k}='{v}'")
            else:
                kwargs_parts.append(f"{k}={v}")

        if kwargs_parts:
            kwargs_str = ", " + ", ".join(kwargs_parts)
        else:
            kwargs_str = ""

        comment = f" #{node.name}"
        lines.append(
            f"        self.module_{mod_id} = nn.Module('{gtx_type}'{kwargs_str}){comment}"
        )

    # forward() 부분
    lines.append("")
    lines.append("    @nn.forward_processor")
    lines.append("    def forward(self, *args):")

    # forward 호출 순서를 그래프 엣지에서 재구성
    # 각 노드의 출력 변수명과 입력 연결 정보를 추적
    node_to_var = {}  # node_name → 변수명

    for mod_id, node in sorted(mod_nodes, key=lambda x: x[0]):
        op_type = node.op.type

        # 입력 노드들 추출
        in_nodes_list = list(node.in_nodes) if node.in_nodes else []

        if op_type == GTX_OP.INPUT:
            var_name = "output_module_0"
            lines.append(f"        {var_name} = self.module_{mod_id}(input=args[0])")
            node_to_var[node.name] = var_name
            continue

        # 출력 변수명 결정 — residual block 시작 기준
        if in_nodes_list:
            first_in = in_nodes_list[0]
            first_in_var = node_to_var.get(first_in.name, "output_module_0")
            # 변수명을 입력의 변수명과 동일하게 유지 (ResNet.py 스타일)
            var_name = first_in_var
        else:
            var_name = "output_module_0"

        # Conv2d가 분기점에서 새 변수를 시작해야 하는 경우 판단
        # 새 residual block이 시작되면 새 변수명 사용
        if op_type in (GTX_OP.CONV2D, GTX_OP.DEPTHWISE_CONV2D) and in_nodes_list:
            first_in = in_nodes_list[0]
            # 입력 노드가 여러 출력을 가지면 (분기점) 새 변수명 시작
            if len(first_in.out_nodes) > 1:
                var_name = f"output_module_{mod_id}"

        # elemwise_add/mul은 두 입력을 받음
        if op_type in (GTX_OP.ADD, GTX_OP.MULTIPLY):
            if len(in_nodes_list) >= 2:
                in_a_var = node_to_var.get(in_nodes_list[0].name, "output_module_0")
                in_b_var = node_to_var.get(in_nodes_list[1].name, "output_module_0")
                lines.append(
                    f"        {var_name} = self.module_{mod_id}(input={in_a_var}, other={in_b_var}, alpha=1)"
                )
            else:
                lines.append(f"        {var_name} = self.module_{mod_id}({var_name})")
        elif op_type == GTX_OP.FLATTEN:
            attrs = node.op.attrs if hasattr(node.op, "attrs") else {}
            sd = attrs.get("start_dim", 1)
            ed = attrs.get("end_dim", -1)
            lines.append(
                f"        {var_name} = self.module_{mod_id}(input={var_name}, start_dim={sd}, end_dim={ed})"
            )
        else:
            lines.append(f"        {var_name} = self.module_{mod_id}({var_name})")

        node_to_var[node.name] = var_name

    # return 문
    # 마지막 노드의 변수명을 반환
    if mod_nodes:
        last_var = node_to_var.get(mod_nodes[-1][1].name, "output_module_0")
        lines.append(f"        return {last_var}")

    lines.append("")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="GTX Compiler: PyTorch Model → C Source Code + Python Export"
    )

    # 입력 모델 (두 가지 방식 지원)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--model", type=str, help="사전 내보낸 모델 파일 경로 (예: export1/ResNet.py)"
    )
    input_group.add_argument(
        "--torch-model",
        type=str,
        help="PyTorch 모델 이름 또는 경로 (예: resnet18, torchvision.models.vgg16, model.pt)",
    )

    parser.add_argument(
        "--output", type=str, default="./output", help="출력 디렉토리 (기본: ./output)"
    )
    parser.add_argument(
        "--name", type=str, default="model", help="생성 모델 이름 (기본: model)"
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default="1,3,224,224",
        help="입력 텐서 shape (기본: 1,3,224,224)",
    )
    parser.add_argument(
        "--no-makefile", action="store_true", help="Makefile 생성 건너뛰기"
    )
    parser.add_argument(
        "--nest-id", type=int, default=0, help="타겟 NEST ID (0-3, 기본: 0)"
    )
    parser.add_argument(
        "--spu-id", type=int, default=0, help="타겟 SPU ID (0-3, 기본: 0)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["intrinsic", "numcpp"],
        default="numcpp",
        help="코드 생성 백엔드 (intrinsic: raw GTX intrinsic C, intrinsic, 기본: numcpp: NumCpp C++ )",
    )

    args = parser.parse_args()

    # 입력 shape 파싱
    input_shape = [int(x) for x in args.input_shape.split(",")]

    print("=" * 60)
    print("GTX Compiler — PyTorch → C Code + Python Export Generator")
    print("=" * 60)

    if args.torch_model:
        print(f"PyTorch model: {args.torch_model}")
    else:
        print(f"Model file: {args.model}")
    print(f"Output dir: {args.output}")
    print(f"Input shape: {input_shape}")
    print()

    # Step 1: 모델 로드
    print("[1/6] 모델 로드 중...")
    if args.torch_model:
        # 방식 1: PyTorch 모델 직접 로드
        model = load_torch_model(args.torch_model)
        if model is None:
            print(f"  ✗ '{args.torch_model}'를 로드할 수 없습니다.")
            sys.exit(1)
        # 모델 이름 자동 설정
        if args.name == "model":
            args.name = model.__class__.__name__.lower()
    else:
        # 방식 2: 사전 내보낸 모델 파일 로드
        model = load_model_from_file(args.model)
    print(f"  → 모델 클래스: {model.__class__.__name__}")

    # Step 2: 입력 텐서 생성
    input_tensor = torch.randn(*input_shape)

    # Step 3: 모델 파싱 → GTX Graph (TorchParser)
    #   --torch-model의 경우: TorchParser로 파싱 → ScriptWriter로 Python 내보내기
    #                          → 다시 MockGraph 생성 (C 코드 생성용)
    #   --model의 경우: 직접 MockGraph 분석
    print("[2/6] 모델 파싱 중 (TorchParser)...")

    export_py_path = None
    parser_graph = None

    if args.torch_model:
        # PyTorch 모델에서 직접: TorchParser → ScriptWriter로 Python export 생성
        try:
            from parse.parser import TorchParser
            from parse.rich_in_out_helper import StandardInputData

            tp_parser = TorchParser()
            input_data = StandardInputData((input_tensor,), {})
            parser_graph = tp_parser(model.__class__.__name__, model, input_data)
            print(f"  → TorchParser 그래프 노드 수: {len(list(parser_graph.nodes))}")

            # ScriptWriter로 Python export 파일 생성
            print("[3/6] Python 모듈 파일 생성 중 (ScriptWriter)...")
            try:
                from qproc.export import get_script_writer

                os.makedirs(args.output, exist_ok=True)
                export_py_path = os.path.join(args.output, f"{args.name}.py")
                exporter = get_script_writer(enable_quant=True)
                exporter.write(parser_graph, file_path=export_py_path)
                print(f"  → Python 모듈 파일: {export_py_path}")
            except Exception as e:
                print(f"  ⚠ ScriptWriter 실패: {e}")
        except Exception as e:
            print(f"  ⚠ TorchParser 실패: {e}")

        # 생성된 Python export 파일을 다시 로드하여 MockGraph 생성 (C 코드 생성용)
        if export_py_path and os.path.exists(export_py_path):
            print("  → 생성된 Python 모듈 파일에서 MockGraph 생성 중...")
            _init_op_map()
            export_model = load_model_from_file(export_py_path)
            graph = build_graph_from_model(export_model, input_tensor)
        else:
            # ScriptWriter 실패 시 직접 MockGraph 분석
            print("  → 직접 MockGraph 분석으로 전환...")
            graph = build_graph_from_model(model, input_tensor)
    else:
        # 사전 내보낸 모델 파일 로드 시: 직접 MockGraph 분석
        graph = parse_model_to_graph(model, input_tensor)

    print(f"  → C 코드용 그래프 노드 수: {len(list(graph.nodes))}")

    # Step 4: shape/param 데이터 확인
    if hasattr(graph, "_nodes"):
        print("[4/6] shape/param 데이터: MockGraph에서 이미 포함됨 ✓")
    else:
        print("[4/6] shape/param 데이터 동기화 중...")
        connect_and_fill_graph(model, graph, input_tensor)

    # Step 5: Python 모듈 파일 생성 (아직 생성되지 않은 경우)
    if export_py_path is None:
        print("[4.5/6] Python 모듈 파일 생성 중...")
        export_py_path = generate_export_python(graph, model, args.output, args.name)

    # Step 6: 코드 생성 (백엔드에 따라 분기)
    if args.backend == "numcpp":
        print("[5/6] C++ 소스코드 생성 중 (NumCpp 백엔드)...")
        files = generate_cpp_code(graph, args.output, args.name)
        for ftype, fpath in files.items():
            print(f"  → {ftype}: {fpath}")

        if not args.no_makefile:
            print("[6/6] C++ 빌드 파일 생성 중...")
            makefile_path = generate_numcpp_makefile(args.output, args.name)
            print(f"  -> Makefile: {makefile_path}")
    else:
        print("[5/6] C 소스코드 생성 중...")
        files = generate_c_code(
            graph, args.output, args.name, nest_id=args.nest_id, spu_id=args.spu_id
        )
        for ftype, fpath in files.items():
            print(f"  → {ftype}: {fpath}")

        if not args.no_makefile:
            print("[6/6] 빌드 파일 생성 중...")
            makefile_path = generate_makefile(args.output, args.name)
            linker_path = generate_linker_script(args.output)
            startup_path = generate_startup_code(args.output)
            data_embed_path = generate_data_embed(args.output)
            print(f"  -> Makefile: {makefile_path}")
            print(f"  -> Linker script: {linker_path}")
            print(f"  -> Startup code: {startup_path}")
            print(f"  -> Data embed: {data_embed_path}")

            modules_makefile_path = generate_modules_makefile(args.output, args.name)
            if modules_makefile_path:
                print(f"  -> Modules Makefile: {modules_makefile_path}")

    print()
    print("=" * 60)
    print("완료!")
    print(f"생성된 파일들: {args.output}/")
    if export_py_path:
        print(f"Python 모듈 파일: {export_py_path}")
    print()
    ext = "cpp" if args.backend == "numcpp" else "c"
    print("크로스 컴파일:")
    print(f"  cd {args.output} && make")
    print()
    print("시뮬레이터 실행:")
    print(f"  ./simulator/GTX_ISS {args.output}/{args.name}.elf")
    print("=" * 60)


if __name__ == "__main__":
    main()
