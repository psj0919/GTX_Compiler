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
"""ONNX export — Vitis-AI pytorch_nndct/qproc/onnx.py 에서 이식.

두 경로:
  1) **quant-free** `export_onnx(model, ...)` : 일반 nn.Module 을 `torch.onnx.export` 로
     내보낸다. 양자화 불필요 — Netron 시각화/배포용 표준 ONNX. (지금 쓰는 경로)
  2) **양자화(QDQ)** `export_onnx_runable_model`/`export_onnx_model_for_lstm` : VAI
     `fix_neuron` op 을 ONNX QuantizeLinear/DequantizeLinear 로 변환해 quant_model 을
     내보낸다. **향후 GGUF 양자화 배포에 재사용** 하려고 원본 로직을 그대로 보존한다.
     (모델에 fix_neuron op 이 없으면 symbolic 등록은 무해하게 무시된다.)

원본 대비 변경: nndct_shared 심볼을 이 프로젝트 것으로 매핑
(NNDCT_OP→OP, NNDCT_KEYS→KEYS, GLOBAL_MAP=shared.base) + 로거/옵션/에러코드는 경량 스텁.
NndctRound(C++) 의존인 QuantizeLinear/DequantizeLinear autograd 클래스는 export 함수가
쓰지 않아 이식에서 제외했다.
"""
import copy
import os

import numpy as np
import torch

from shared.base import GLOBAL_MAP, KEYS, OP
from utils.module_util import get_module_name, to_device
from utils.onnx_utils import get_opset_version
from utils.torch_utils import CmpFlag, compare_torch_version


# --- nndct_shared 경량 스텁 (원본 로거/옵션/에러코드 대체) --------------------
class _ScreenLogger:
    def info(self, msg):
        print(f"[onnx] {msg}", flush=True)

    def warning(self, msg):
        print(f"[onnx][warn] {msg}", flush=True)

    def error(self, code, msg):
        print(f"[onnx][error] {msg}", flush=True)

    def error2user(self, code, msg):
        print(f"[onnx][error] {msg}", flush=True)


def NndctScreenLogger():
    return _ScreenLogger()


class _Opt:
    def __init__(self, value):
        self.value = value


class _NndctOption:
    # 원본은 전역 옵션 테이블. 여기선 export 가 참조하는 두 개만.
    nndct_native_onnx = _Opt(True)          # QDQ 를 표준 Quantize/DequantizeLinear 로
    nndct_onnx_opset_version = _Opt(-1)     # -1 → get_opset_version()


NndctOption = _NndctOption()


class QError:
    TORCH_VERSION = "TORCH_VERSION"
    EXPORT_ONNX = "EXPORT_ONNX"
    FIX_INPUT_TYPE = "FIX_INPUT_TYPE"
    NO_FORWARD = "NO_FORWARD"


# ==========================================================================
# quant-free export (양자화 없이 일반 모델 → ONNX)
# ==========================================================================
def export_onnx(model, example_inputs, output_dir, model_name=None,
                input_names=None, opset_version=None, dynamic_batch=False,
                verbose=False):
    """일반 nn.Module 을 표준 ONNX 로 내보낸다(양자화 불필요).

    model 은 `torch.onnx.export` 가 trace 가능한 eager torch 모델이어야 한다
    (ggml 백엔드 커스텀 커널은 trace 불가 → pytorch 백엔드/원본 torch 모델 사용).
    반환: 생성 .onnx 경로.
    """
    if not compare_torch_version(CmpFlag.GREATER_EQUAL, "1.7.0"):
        NndctScreenLogger().error2user(
            QError.TORCH_VERSION, "ONNX export 는 pytorch 1.7 이상 필요.")
        return None
    os.makedirs(output_dir, exist_ok=True)
    if opset_version is None:
        opset_version = get_opset_version()
    name = model_name or get_module_name(model)
    output_file = os.path.join(output_dir, f"{name}.onnx")
    args = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)
    dynamic_axes = None
    if dynamic_batch and input_names:
        dynamic_axes = {n: [0] for n in input_names}
    try:
        torch.onnx.export(model.eval(), args, output_file,
                          input_names=input_names, verbose=verbose,
                          opset_version=opset_version, dynamic_axes=dynamic_axes,
                          do_constant_folding=False)
    except Exception as e:
        NndctScreenLogger().error2user(
            QError.EXPORT_ONNX,
            f"{name} can not be exported for onnx. PyTorch reason:\n{str(e)}")
        return None
    NndctScreenLogger().info(f"{name}.onnx is generated. ({output_file})")
    return output_file


# ==========================================================================
# 양자화(QDQ) export — 향후 GGUF 양자화 배포용으로 원본 보존
# ==========================================================================
def export_onnx_model_for_lstm(quantizer, example_inputs, input_tensors_name,
                               return_tensors_name, convert_script_to_qscript,
                               output_dir, verbose=False, dynamic_batch=False,
                               opset_version=None, native_onnx=True,
                               dump_layers=False, check_model=False, opt_graph=False):
    from torch.onnx import register_custom_op_symbolic
    from torch.onnx.symbolic_helper import parse_args

    @parse_args("v", "v", "v", "v", "v", "v", "v", "v")
    def symbolic_fix_neuron(g, input, valmin, valmax, valamp, zero_point, method,
                            device_id, inplace):
        return g.op("vai::fix_neuron", input, valmax, valamp, method, device_id,
                    inplace).setType(input.type())

    register_custom_op_symbolic("vai::fix_neuron", symbolic_fix_neuron, 9)

    opset_version = NndctOption.nndct_onnx_opset_version.value if opset_version is None else opset_version
    opset_version = get_opset_version() if opset_version == -1 else opset_version
    device = GLOBAL_MAP.get_ele(KEYS.QUANT_DEVICE)

    script_models = quantizer.scripts if len(quantizer.scripts) > 0 else None
    quantizer.reset_status_for_exporting()
    if script_models is None:
        output_file = os.path.join(output_dir, f"{quantizer.quant_model._get_name()}_int.onnx")
        model, input_args = to_device(quantizer.quant_model, example_inputs, device)
        try:
            torch.onnx.export(quantizer.quant_model, input_args, output_file,
                              verbose=verbose, input_names=input_tensors_name,
                              opset_version=opset_version, custom_opsets={'vai': 2})
        except Exception as e:
            NndctScreenLogger().error2user(
                QError.EXPORT_ONNX,
                f"The {get_module_name(quantizer.quant_model)} can not be exported for onnx. "
                f"The PyTorch internal failed reason is:\n{str(e)}.")
        else:
            NndctScreenLogger().info(
                f"{get_module_name(quantizer.quant_model)}_int.onnx is generated.({output_file})")
    else:
        if len(script_models) == 1:
            inputs = [example_inputs]
            inputs_name = [input_tensors_name]
            outputs_name = [return_tensors_name]
        else:
            inputs = example_inputs
            inputs_name = input_tensors_name
            outputs_name = return_tensors_name

        for script, inputs, input_name, output_name in zip(script_models, inputs, inputs_name, outputs_name):
            q_model = convert_script_to_qscript(script, verbose=verbose)
            _, inputs = to_device(None, inputs, device)
            output_file = os.path.join(output_dir, f"{get_module_name(q_model)}_int.onnx")
            try:
                torch.onnx.export(q_model, inputs, output_file, verbose=verbose,
                                  input_names=input_name, opset_version=opset_version,
                                  custom_opsets={'vai': 2})
            except Exception as e:
                NndctScreenLogger().error2user(
                    QError.EXPORT_ONNX,
                    f"The {get_module_name(q_model)} can not be exported for onnx. "
                    f"The PyTorch internal failed reason is:\n{str(e)}.")
            NndctScreenLogger().info(
                f"{get_module_name(q_model)}_int.onnx is generated.({output_file})")


def export_onnx_runable_model(quantizer, example_inputs, input_tensors_name,
                              return_tensors_name, convert_script_to_qscript,
                              output_dir, verbose=False, dynamic_batch=False,
                              opset_version=None, native_onnx=True,
                              dump_layers=False, check_model=False, opt_graph=False):
    from torch.onnx import register_custom_op_symbolic
    from torch.onnx.symbolic_helper import parse_args
    import sys

    if compare_torch_version(CmpFlag.LESS, "1.7.0"):
        NndctScreenLogger().error2user(
            QError.TORCH_VERSION,
            'Only supprt exporting onnx model with pytorch 1.7 and later version.')
        return

    if quantizer.contain_channel_quantize():
        if compare_torch_version(CmpFlag.LESS, "1.10.0"):
            NndctScreenLogger().error2user(
                QError.TORCH_VERSION,
                'Only supprt exporting per_channel quantization onnx model with '
                'pytorch 1.10 and later version.')
            return

    @parse_args("v", "i", "i", "f", "i", "i", "i", "i")
    def symbolic_fix_neuron(g, input, valmin, valmax, valamp, zero_point, method,
                            device_id, inplace):
        if valamp < sys.float_info.min:
            scale = torch.tensor(sys.float_info.max).float()   # double 방지
        else:
            scale = torch.tensor(1.0 / valamp).float()
        zero_point = torch.tensor(0, dtype=torch.int8)          # ONNX zero_point 는 tensor
        if not isinstance(input, torch._C.Value) or not isinstance(valmin, int) or not isinstance(valmax, int) \
                or not isinstance(valamp, float) or zero_point.dtype != torch.int8 or not isinstance(method, int) \
                or not isinstance(device_id, int) or not isinstance(inplace, int) or valamp <= 0.:
            NndctScreenLogger().error2user(
                QError.FIX_INPUT_TYPE,
                'Data type or value illegal fix neuron in when exporting onnx model.')

        if compare_torch_version(CmpFlag.GREATER_EQUAL, "2.0.0"):
            NndctOption.nndct_native_onnx.value = True

        if NndctOption.nndct_native_onnx.value:
            return g.op("DequantizeLinear", g.op("QuantizeLinear", input, scale, zero_point), scale, zero_point)
        else:
            return g.op("vai::DequantizeLinear", g.op("vai::QuantizeLinear", input, torch.tensor(valmin, dtype=torch.int32), torch.tensor(valmax, dtype=torch.int32), scale, zero_point, torch.tensor(method, dtype=torch.int8), torch.tensor(1, dtype=torch.int8)), scale, zero_point, torch.tensor(1, dtype=torch.int8))

    # per-channel quantization
    @parse_args("v", "i", "i", "v", "v", "i", "i", "i", "i")
    def symbolic_fix_neuron_per_channel(g, input, valmin, valmax, scale, zero_point,
                                        axis, method, device_id, inplace):
        if not isinstance(input, torch._C.Value) or not isinstance(valmin, int) or not isinstance(valmax, int) \
           or not isinstance(scale, torch._C.Value) or not isinstance(zero_point, torch._C.Value) or not isinstance(method, int) \
           or not isinstance(device_id, int) or not isinstance(inplace, int):
            NndctScreenLogger().error2user(
                QError.FIX_INPUT_TYPE,
                'Data type or value illegal fix neuron in when exporting onnx model.')

        if compare_torch_version(CmpFlag.GREATER_EQUAL, "2.0.0"):
            NndctOption.nndct_native_onnx.value = True

        if NndctOption.nndct_native_onnx.value:
            return g.op("DequantizeLinear", g.op("QuantizeLinear", input, scale, zero_point, axis_i=axis), scale, zero_point, axis_i=axis)
        else:
            return g.op("vai::DequantizeLinear", g.op("vai::QuantizeLinear", input, torch.tensor(valmin, dtype=torch.int32), torch.tensor(valmax, dtype=torch.int32), scale, zero_point, torch.tensor(method, dtype=torch.int8), torch.tensor(axis, dtype=torch.int8)), scale, zero_point, torch.tensor(axis, dtype=torch.int8))

    NndctOption.nndct_native_onnx.value = native_onnx    # 이하 전역값 사용
    register_custom_op_symbolic("vai::fix_neuron", symbolic_fix_neuron, 9)
    register_custom_op_symbolic("vai::fix_neuron_per_channel", symbolic_fix_neuron_per_channel, 10)
    opset_version = NndctOption.nndct_onnx_opset_version.value if opset_version is None else opset_version
    opset_version = get_opset_version() if opset_version == -1 else opset_version
    device = GLOBAL_MAP.get_ele(KEYS.QUANT_DEVICE)
    script_models = quantizer.scripts if len(quantizer.scripts) > 0 else None
    quantizer.reset_status_for_exporting()
    if dynamic_batch:    # dynamic batch N in [N, C, L, H, W]
        dynamic_axes = {}
        for i in range(len(input_tensors_name)):
            dynamic_axes[input_tensors_name[i]] = [0]
    else:
        dynamic_axes = None

    # check onnx: backup batchnorm eps
    batchnorm_eps_bak = {}
    if check_model:
        for module in quantizer.quant_model.modules():
            if hasattr(module, "node") and module.node.op.type == OP.BATCH_NORM:
                batchnorm_eps_bak[module.name] = module.eps
                module.eps = 1.0e-5

    if script_models is None:
        output_file = os.path.join(output_dir, f"{quantizer.quant_model._get_name()}_int.onnx")
        model, input_args = to_device(quantizer.quant_model, example_inputs, device)
        try:
            torch.onnx.export(quantizer.quant_model.eval(), input_args, output_file,
                              input_names=input_tensors_name, verbose=verbose,
                              opset_version=opset_version, dynamic_axes=dynamic_axes,
                              do_constant_folding=False)
        except Exception as e:
            NndctScreenLogger().error2user(
                QError.EXPORT_ONNX,
                f"The {get_module_name(quantizer.quant_model)} can not be exported for onnx. "
                f"The PyTorch internal failed reason is:\n{str(e)}")
        else:
            NndctScreenLogger().info(
                f"{get_module_name(quantizer.quant_model)}_int.onnx is generated.({output_file})")

        if opt_graph:
            optimize_onnx_graph(output_file)
        if dump_layers:
            dump_onnx_layers(quantizer, output_dir, output_file, native_onnx, check_model)
    else:
        if len(script_models) == 1:
            inputs = [example_inputs]
            inputs_name = [input_tensors_name]
            outputs_name = [return_tensors_name]
        else:
            inputs = example_inputs
            inputs_name = input_tensors_name
            outputs_name = return_tensors_name

        for script, inputs, input_name, output_name in zip(script_models, inputs, inputs_name, outputs_name):
            q_model = convert_script_to_qscript(script, verbose=verbose)
            _, inputs = to_device(None, inputs, device)
            output_file = os.path.join(output_dir, f"{get_module_name(q_model)}_int.onnx")
            try:
                torch.onnx.export(q_model, inputs, output_file, input_names=input_name,
                                  verbose=verbose, opset_version=opset_version,
                                  dynamic_axes=dynamic_axes)
            except Exception as e:
                NndctScreenLogger().error2user(
                    QError.EXPORT_ONNX,
                    f"The {get_module_name(q_model)} can not be exported for onnx. "
                    f"The PyTorch internal failed reason is:\n{str(e)}.")
            else:
                NndctScreenLogger().info(
                    f"{get_module_name(q_model)}_int.onnx is generated.({output_file}).")

            if opt_graph:
                optimize_onnx_graph(output_file)
            if NndctOption.nndct_native_onnx.value:
                return
            elif dump_layers:
                dump_onnx_layers(quantizer, output_dir, output_file, native_onnx, check_model)

    # restore batchnorm eps
    if len(batchnorm_eps_bak) > 0:
        for module in quantizer.quant_model.modules():
            if module.name in batchnorm_eps_bak.keys():
                module.eps = batchnorm_eps_bak[module.name]


def optimize_onnx_graph(onnx_path):
    import onnx
    from onnxsim import simplify
    model_org = onnx.load(onnx_path)
    model_simp, check = simplify(model_org)
    if check:
        NndctScreenLogger().info("Optimize onnx graph successfully.")
        onnx.save(model_simp, onnx_path)
    else:
        NndctScreenLogger().info("Optimize onnx graph failed.")


def dump_onnx_layers(quantizer, output_dir, onnx_model, native_onnx, check_model):
    import onnx

    onnx.checker.check_model(onnx_model)
    inputs = get_blob_input(quantizer)
    if inputs is None:
        NndctScreenLogger().warning("ONNX model layers are not dumped since input data is None!")
        return

    ep_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_inputs = {}
    if native_onnx:
        import onnxruntime as ort
        ort_session = ort.InferenceSession(onnx_model, providers=ep_list)
        org_outputs = [x for x in ort_session.get_outputs()]
        for i in range(len(inputs)):
            ort_inputs[ort_session.get_inputs()[i].name] = inputs[i]
    else:
        from onnxruntime_extensions import PyOrtFunction
        from pytorch_nndct.apis import load_vai_ops
        ort_session = PyOrtFunction.from_model(onnx_model)
        org_outputs = [x for x in ort_session.output_names]
        for i in range(len(inputs)):
            ort_inputs[ort_session.input_names[i]] = inputs[i]

    model = onnx.load(onnx_model)
    for node in model.graph.node:
        for output in node.output:
            if output not in org_outputs:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    output_names = [x.name for x in model.graph.output]

    onnx_model_tmp = output_dir + f"{quantizer.quant_model._get_name()}_int_tmp.onnx"
    onnx.save(model, onnx_model_tmp)

    if native_onnx:
        run_ort = ort.InferenceSession(onnx_model_tmp, providers=ep_list)
        ort_outs = run_ort.run(output_names, ort_inputs)
    else:
        from pytorch_nndct.apis import load_vai_ops
        load_vai_ops()
        run_ort = PyOrtFunction.from_model(onnx_model_tmp)
        run_ort._ensure_ort_session()
        ort_outs = run_ort.ort_session.run(output_names, ort_inputs)

    layer_dir = output_dir + "/onnx_layer_output/"
    if not os.path.exists(layer_dir):
        os.mkdir(layer_dir)
    debug = False
    for i in range(len(output_names)):
        layer_name = output_names[i]
        layer_data = ort_outs[i].astype("float32")
        if debug:
            layer_data = permute(layer_data)
            dump_path = layer_dir + layer_name + '.txt'
            if not os.path.exists(os.path.dirname(dump_path)):
                os.makedirs(os.path.dirname(dump_path))
            layer_data.flatten().tofile(dump_path, sep='\n', format="%.6f")
        else:
            dump_path = layer_dir + layer_name + '.bin'
            if not os.path.exists(os.path.dirname(dump_path)):
                os.makedirs(os.path.dirname(dump_path))
            layer_data.flatten().tofile(dump_path)
    os.remove(onnx_model_tmp)
    NndctScreenLogger().info(f"ONNX model layers are dumped successfully.({layer_dir})")

    if not native_onnx and check_model:
        check_dump_onnx(quantizer, ort_outs, inputs)


def get_blob_input(quantizer):
    from qproc.utils import update_blob_data
    update_blob_data(quantizer.quant_model, quantizer.graph, only_update_shape=False)
    input_dic = {}
    input_names = []
    for node in quantizer.graph.all_nodes():
        if node.op.type == OP.INPUT:
            input_idx = int(node.name.split('_')[-1])       # input_0, input_1
            input_data = np.expand_dims(node.out_tensors[0].data[0], axis=0)
            input_dic[input_idx] = input_data
            input_names.append(input_idx)
    input_names.sort()      # input_0, input_1, ... 순서 보장
    inputs = tuple(input_dic[idx] for idx in input_names)
    return inputs


def check_dump_onnx(quantizer, ort_outs, input):
    rtol = 1e-4
    atol = 1e-5
    device = GLOBAL_MAP.get_ele(KEYS.QUANT_DEVICE)
    input_torch = []
    for i in range(len(input)):
        input_torch.append(torch.from_numpy(input[i]).to(device))
    quant_out = quantizer.quant_model(input_torch)

    if isinstance(quant_out, (tuple, list)):
        for i in range(len(quant_out)):
            np.testing.assert_allclose(quant_out[i].detach().cpu().numpy(), ort_outs[i], rtol=rtol, atol=atol)
        NndctScreenLogger().info("ONNX model is checked valid.")
    elif isinstance(quant_out, torch.Tensor):
        np.testing.assert_allclose(quant_out.detach().cpu().numpy(), ort_outs[0], rtol=rtol, atol=atol)
        NndctScreenLogger().info("ONNX model is checked valid.")
    else:
        NndctScreenLogger().warning("ONNX model is checked failed: quant output is not tensor, tuple or list!")


def permute(input):
    dim = input.ndim
    if dim == 3:
        output = input.transpose(0, 2, 1)
    elif dim == 4:
        output = input.transpose(0, 2, 3, 1)
    elif dim == 5:
        output = input.transpose(0, 2, 3, 4, 1)
    else:
        output = input
    return output


def round(x, method):
    if method == 2:      # half_up
        y = copy.deepcopy(x)
        y = np.where(y - np.floor(y) == 0.5, np.ceil(y), np.round(y))
    elif method == 3:    # c++ std::round: negative half_down, positive half_up
        y = copy.deepcopy(x)
        y = np.where(y < 0, np.ceil(y - 0.5), np.floor(y + 0.5))
    elif method == 4:    # floor
        y = np.floor(x)
    elif method == 5:    # negative half_up, positive half_even
        y = copy.deepcopy(x)
        y = np.where((y < 0) & (y - np.floor(y) == 0.5), np.ceil(y), np.round(y))
    elif method == 6:    # negative half_up, positive half_down (vs method 3)
        y = copy.deepcopy(x)
        y = np.where((y < 0) & (y - np.floor(y) == 0.5), np.ceil(y), y)
        y = np.where((y > 0) & (y - np.floor(y) == 0.5), np.floor(y), np.round(y))
    elif method == 7:    # up
        y = np.ceil(x)
    else:                # half_even
        y = np.round(x)
    return y
