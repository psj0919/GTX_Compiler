#!/usr/bin/env python3
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
"""PyTorch 모델 → vision.cpp(ggml) arch C++ + GGUF 컴파일 파이프라인.

콘솔 스크립트 ``g2c`` 의 구현(pyproject ``[project.scripts]``). 과거 루트
``compile_to_c.py`` 의 로직을 재사용 가능한 함수로 추출한 것이며, ``test/compile_to_c.py``
와 ``g2c`` 가 공통으로 사용한다.

CLI 예시::

    g2c --model "ultralytics.YOLO('yolo11n')" --output output/
    g2c --model resnet18 --output output/resnet18
    g2c --model "torchvision.models.resnet50" --pth weights.pth --output output/

흐름: 모델 로드 → TorchParser 그래프 → ScriptWriter export(.py) →
VispCodeGenerator(.cpp/.h) + GGUF(.gguf) + ggml 실행 진입점 append.
설계: docs/ggml_codegen.md
"""

import os
import sys

import numpy as np
import torch

# 프로젝트 루트(= shared/ 의 부모의 부모)를 Python 경로에 추가 (스크립트 직접 실행 대비).
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# --------------------------------------------------------------------------
# codegen / weight 직렬화
# --------------------------------------------------------------------------
def generate_ggml_code(graph, output_dir: str, model_name: str = "model", quant_plan=None,
                       baked=None, skip=None):
    """Graph → vision.cpp(ggml) arch 스타일 C++ 소스코드 생성. 설계: docs/ggml_codegen.md

    quant_plan: build_quant_plan 결과. 양자 conv 노드는 conv_2d_q 로 emit + 헬퍼 주입.
    baked/skip: const_fold.fold_constants 결과 (상수 subgraph → baked weight 로드).
    """
    from shared.compile.ggml_codegen import VispCodeGenerator

    codegen = VispCodeGenerator(graph, output_dir=output_dir, model_name=model_name,
                                quant_plan=quant_plan, baked=baked, skip=skip)
    return codegen.generate()


# ── 양자화 두 갈래 ──────────────────────────────────────────────────────────
# (A) in-process: gguf-py 가 순수 파이썬으로 quantize 를 구현한 legacy 타입(block=32).
#     arch 무관, 외부 의존 없음. 자격되는 2D Linear 가중치에만 적용.
_GGUF_QUANTS = {"q8_0", "q4_0", "q4_1", "q5_0", "q5_1"}
_GGUF_QUANT_BLOCK = 32
# (B) 외부 llama-quantize 경유: k-quant / IQ-quant (gguf-py 에 quantize 미구현).
#     fp16 GGUF 를 먼저 쓰고 `llama-quantize <in> <out> <TYPE>` 로 변환. 바이너리 필요.
#     주의: llama-quantize 는 general.architecture 로 모델을 로드 → vision arch
#     ('resnet'/'yolo')를 llama.cpp 가 모르면 실패할 수 있음(LLM 전용 도구).
_LLAMA_QUANTS = {
    "q2_k", "q2_k_s", "q3_k_s", "q3_k_m", "q3_k_l", "q4_k_s", "q4_k_m",
    "q5_k_s", "q5_k_m", "q6_k", "q8_k",
    "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq2_m",
    "iq3_xxs", "iq3_xs", "iq3_s", "iq3_m", "iq4_xs", "iq4_nl",
}


def _resolve_quant(gguf, quant):
    """quant 문자열 → (kind, payload, block).

      kind 'py'    : payload=GGMLQuantizationType (in-process), block=32
      kind 'llama' : payload=대문자 타입명(예 'Q4_K_M'), block=None
      kind 'none'  : fp16 (payload/block=None)
    """
    if not quant:
        return "none", None, None
    q = quant.lower()
    if q in ("none", "fp16", "f16"):
        return "none", None, None
    if q in _GGUF_QUANTS:
        return "py", getattr(gguf.GGMLQuantizationType, q.upper()), _GGUF_QUANT_BLOCK
    if q in _LLAMA_QUANTS:
        return "llama", quant.upper(), None
    print(f"  ⚠ --quantize '{quant}' 미지원 → fp16. "
          f"(in-process: {sorted(_GGUF_QUANTS)} / llama-quantize: {sorted(_LLAMA_QUANTS)})")
    return "none", None, None


def _find_llama_quantize():
    """llama-quantize 바이너리 탐색: $LLAMA_QUANTIZE → PATH → vision.cpp 빌드 디렉토리."""
    import shutil

    env = os.environ.get("LLAMA_QUANTIZE")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    for name in ("llama-quantize", "quantize"):
        p = shutil.which(name)
        if p:
            return p
    for cand in (
        os.path.join(project_root, "vision.cpp", "build", "bin", "llama-quantize"),
        os.path.join(project_root, "vision.cpp", "depend", "llama", "build", "bin", "llama-quantize"),
    ):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _llama_quantize_file(src_gguf, dst_gguf, type_name):
    """fp16 GGUF(src)를 llama-quantize 로 type_name(k/IQ-quant) 변환 → dst. 성공 시 True."""
    import subprocess

    binpath = _find_llama_quantize()
    if not binpath:
        print(f"  ⚠ k-quant '{type_name}' 요청이지만 llama-quantize 바이너리를 못 찾음 "
              f"→ fp16 유지.\n"
              f"      llama.cpp 빌드 후 PATH 또는 $LLAMA_QUANTIZE 로 지정하세요:\n"
              f"      llama-quantize {src_gguf} <out.gguf> {type_name}")
        return False
    try:
        print(f"  → llama-quantize ({binpath}) {type_name} …")
        subprocess.run([binpath, src_gguf, dst_gguf, type_name], check=True)
        return os.path.isfile(dst_gguf)
    except subprocess.CalledProcessError as e:
        print(f"  ⚠ llama-quantize 실패(exit {e.returncode}) → fp16 유지. "
              f"vision arch 를 llama.cpp 가 인식 못 했을 수 있음(--pure 또는 arch 지원 필요).")
        return False


# conv/linear op 식별 (build_quant_plan 용). depthwise conv 는 row=KH*KW 가 작아 제외.
_PLAN_CONV = {"conv2d"}
_PLAN_LINEAR = {"linear", "dense", "addmm", "matmul"}


def _node_weight_tensor(node):
    """노드의 .weight 파라미터 텐서 반환(없으면 None)."""
    params = getattr(getattr(node, "op", None), "params", None) or {}
    try:
        items = list(params.values())
    except AttributeError:
        return None
    for t in items:
        nm = (getattr(t, "name", "") or "").split("::")[-1]
        if nm.endswith(".weight"):
            return t
    return None


def build_quant_plan(graph, quant):
    """양자화 단일 진실원천: (kind, payload, plan).

    plan = {weight_key: {"qtype", "kind"('conv'|'linear'), "kh", "kw"}} — gguf(저장 형태)와
    codegen(conv_2d_q emit 여부) 가 **같은 plan** 을 본다. conv-as-matmul 양자화(커널을
    2D [OC, IC*KH*KW] 로 보고 mul_mat)는 in-process legacy quant('py') 에서만. 'llama'/'none'
    은 plan 비움(conv 4D 유지 + 후처리/fp16).

    자격: conv 는 row=IC*KH*KW % block==0, linear 는 in_features % block==0.
    그래프 conv weight 레이아웃은 [OC,KH,KW,IC] (generate_gguf 의 OIHW transpose 전).
    """
    import numpy as _np

    gguf_py = os.path.join(project_root, "vision.cpp", "depend", "llama", "gguf-py")
    if gguf_py not in sys.path:
        sys.path.insert(0, gguf_py)
    try:
        import gguf
    except Exception:
        return "none", None, {}

    from shared.compile.render_api import weight_key

    kind, payload, qblock = _resolve_quant(gguf, quant)
    plan = {}
    if kind != "py":
        return kind, payload, plan

    for node in getattr(graph, "nodes", []):
        ot = str(getattr(getattr(node, "op", None), "type", "")).lower()
        is_conv = ot in _PLAN_CONV
        is_lin = ot in _PLAN_LINEAR
        if not (is_conv or is_lin):
            continue
        wt = _node_weight_tensor(node)
        if wt is None:
            continue
        shp = tuple(int(d) for d in _np.asarray(wt.data).shape)
        wk = weight_key(node)
        if is_conv and len(shp) == 4:
            _oc, kh, kw, ic = shp                 # 그래프 conv weight = [OC,KH,KW,IC]
            if (ic * kh * kw) % qblock == 0:
                plan[wk] = {"qtype": payload, "kind": "conv", "kh": kh, "kw": kw}
        elif is_lin and len(shp) == 2:
            if shp[1] % qblock == 0:               # [out, in]
                plan[wk] = {"qtype": payload, "kind": "linear"}
    return kind, payload, plan


def generate_gguf(graph, output_dir: str, model_name: str, arch_id: str,
                  quant=None, plan=None, baked=None):
    """그래프의 (Conv-BN folded) 파라미터를 GGUF 로 직렬화한다(기본 fp16).

    weight 는 `model.state_dict()` 가 아니라 **그래프 node.op.params** 에서 모은다 →
    `fold_conv_bn_graph` 의 fold 결과(folded conv weight/bias, BN 제거)가 GGUF 에 반영된다.
    텐서명은 param 이름에서 `<GraphName>::` prefix 를 떼 state_dict 키(`layer1.0.conv1.weight`)
    로 만든다 → 생성 .cpp 의 weight_key(`m["layer1.0.conv1"]`)와 정합. metadata
    `general.architecture = arch_id`. GGUF writer 는 vision.cpp 에 vendor 된 gguf-py 사용.

    quant: None/'fp16' → 전부 fp16. 'q8_0'/'q4_0'/'q4_1'/'q5_0'/'q5_1' → ggml 블록 양자화를
    **자격 텐서에만** 적용(per-tensor 혼합). 자격: 2D weight(Linear) & 마지막 차원 % 32 == 0.
    conv 4D 커널은 GGUF 연속 차원이 KW 라 블록 정합 불가 + ggml conv 가 양자 커널 미소비 →
    제외(fp16 유지). 나머지(bias/BN/scale)도 fp16. 자격 안 되면 조용히 fp16 폴백 후 리포트.
    """
    gguf_py = os.path.join(project_root, "vision.cpp", "depend", "llama", "gguf-py")
    if gguf_py not in sys.path:
        sys.path.insert(0, gguf_py)
    try:
        import gguf
        from gguf import quants as gguf_quants
    except Exception as e:
        print(f"  ⚠ gguf 로드 실패 ({e}) — GGUF 생략")
        return None

    kind, payload, qblock = _resolve_quant(gguf, quant)
    if plan is None:
        _, _, plan = build_quant_plan(graph, quant)
    from shared.compile.render_api import weight_key

    # 그래프 파라미터 수집(이름 dedup). conv/bn/linear 등 weight-bearing 노드의 param.
    # TorchParser 는 conv weight 를 PyTorch 와 다른 레이아웃으로 저장 → GGUF/.cpp/eager 가
    # 기대하는 PyTorch 레이아웃으로 4D weight 만 되돌린다:
    #   regular conv : graph [OC,KH,KW,IC] → OIHW [OC,IC,KH,KW]      = (0,3,1,2)
    #   depthwise    : graph [1,KH,KW,C]   → [C,1,KH,KW]             = (3,0,1,2)
    _CONV_REG = {"conv2d", "conv1d", "conv3d"}
    _CONV_DW = {"depthwise_conv2d", "depthwise_conv1d", "depthwise_conv3d"}
    weights = {}        # gguf_key -> (arr, weight_key)
    for node in getattr(graph, "nodes", []):
        op = getattr(node, "op", None)
        op_type = getattr(op, "type", None)
        params = getattr(op, "params", None) or {}
        try:
            items = list(params.values())
        except AttributeError:
            continue
        wk = weight_key(node)
        for tensor in items:
            name = getattr(tensor, "name", None)
            data = getattr(tensor, "data", None)
            if not name or data is None:
                continue
            key = name.split("::")[-1]            # "<GraphName>::layer1.0.conv1.weight" → 뒤만
            if key.endswith("num_batches_tracked"):
                continue
            arr = np.asarray(data)
            if arr.ndim == 4 and key.endswith(".weight"):
                if op_type in _CONV_REG:
                    arr = np.transpose(arr, (0, 3, 1, 2))   # OHWI → OIHW
                elif op_type in _CONV_DW:
                    arr = np.transpose(arr, (3, 0, 1, 2))   # [1,KH,KW,C] → [C,1,KH,KW]
            if key not in weights:
                weights[key] = (arr, wk)   # raw(보통 f32) 유지 — 양자화/ f16 은 write 시 결정

    path = os.path.join(output_dir, f"{model_name}.gguf")
    writer = gguf.GGUFWriter(path, arch=arch_id)
    writer.add_architecture()  # general.architecture = arch_id

    n_quant = 0
    for key, (arr, wk) in weights.items():
        entry = plan.get(wk) if key.endswith(".weight") else None
        if entry is not None:
            a = np.ascontiguousarray(arr, dtype=np.float32)
            if entry["kind"] == "conv":
                # conv 커널 [OC,IC,KH,KW] → 2D [OC, IC*KH*KW] (conv-as-matmul). ggml 의
                # im2col 컬럼 순서(kw 가장 빠름)와 numpy row-major 가 일치 → mul_mat 정합.
                a = a.reshape(a.shape[0], -1)
            qbytes = gguf_quants.quantize(a, entry["qtype"])
            writer.add_tensor(key, qbytes, raw_dtype=entry["qtype"])
            n_quant += 1
        else:
            writer.add_tensor(key, np.ascontiguousarray(arr).astype(np.float16))

    # const-fold baked 텐서(anchor/stride/scalar) — 정적 상수라 f32 로 보존(.cpp 의
    # m.weights("const.foldN") 와 정합). 작은 텐서라 용량 영향 미미.
    n_baked = 0
    for _tid, (bkey, barr) in (baked or {}).items():
        writer.add_tensor(bkey, np.ascontiguousarray(barr, dtype=np.float32))
        n_baked += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if kind == "py":
        n_f16 = len(weights) - n_quant
        n_conv = sum(1 for e in plan.values() if e["kind"] == "conv")
        n_lin = sum(1 for e in plan.values() if e["kind"] == "linear")
        print(f"  → gguf: {path} ({len(weights)} tensors: {n_quant}×{quant.lower()}"
              f"(conv {n_conv}+linear {n_lin}) + {n_f16}×fp16, {size_mb:.1f} MB, "
              f"general.architecture='{arch_id}')")
        if n_quant == 0:
            print(f"      ⚠ 양자화 자격(conv row%{qblock}==0 / linear in%{qblock}==0) 텐서 없음.")
    elif kind == "llama":
        # fp16 으로 쓴 GGUF 를 llama-quantize 로 k/IQ-quant 변환(in-place 교체).
        print(f"  → gguf(fp16 중간): {path} ({len(weights)} tensors, {size_mb:.1f} MB)")
        tmp = path + ".kq.tmp"
        if _llama_quantize_file(path, tmp, payload):
            os.replace(tmp, path)
            print(f"  → gguf: {path} ({payload} via llama-quantize, "
                  f"{os.path.getsize(path)/(1024*1024):.1f} MB, "
                  f"general.architecture='{arch_id}')")
        else:
            if os.path.exists(tmp):
                os.remove(tmp)
    else:
        print(f"  → gguf: {path} ({len(weights)} tensors, fp16, {size_mb:.1f} MB, "
              f"general.architecture='{arch_id}')")
    return path


def append_ggml_runner(export_path: str, class_name: str, gguf_name: str,
                       input_shape=(1, 3, 224, 224)):
    """export 파일 끝에 ggml 실행 진입점을 추가한다.

    `python output/<Model>.py` 직접 실행 시 ggml(libggml.so) 커널로 forward 가 돈다:
    set_backend('ggml') → nn.Module → GgmlModule → run_gguf 가 옆 .gguf 를 이름 바인딩 후 실행.
    """
    marker = "# ----- ggml 실행 진입점 (libggml.so) -----"
    src = open(export_path).read()
    if marker in src:
        return

    boot = (
        "import os as _bos, sys as _bsys\n"
        "_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname("
        "_bos.path.abspath(__file__))))\n"
    )
    if "_bsys.path.insert" not in src:
        src = src.replace("import nn\n", boot + "import nn\n", 1)

    shape = ", ".join(str(int(s)) for s in input_shape)
    block = f'''

{marker}
if __name__ == "__main__":
    import os as _os
    import numpy as _np
    import nn as _nn

    _nn.set_backend("ggml")
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _gguf = _os.path.join(_here, "{gguf_name}")
    _x = _np.random.randn({shape}).astype("float32")
    _out = _nn.run_gguf({class_name}(), _gguf, __file__, _x)
    _main = _out[0] if isinstance(_out, (list, tuple)) else _out
    _main = _np.asarray(_main)
    print("[ggml] output:", _main.shape)
'''
    if not src.endswith("\n"):
        src += "\n"
    with open(export_path, "w") as f:
        f.write(src + block)


# --------------------------------------------------------------------------
# 모델 로드 (--model 표현식 / named / .pt + --pth state_dict)
# --------------------------------------------------------------------------
def _unwrap_module(obj):
    """ultralytics.YOLO 등 래퍼에서 실제 nn.Module(백본)을 꺼낸다.

    ultralytics 의 YOLO 는 그 자체가 nn.Module 이지만 추론 백본은 `.model`
    (DetectionModel)이므로 ultralytics 래퍼는 `.model` 을 우선한다.
    """
    inner = getattr(obj, "model", None)
    if isinstance(inner, torch.nn.Module) and type(obj).__module__.startswith("ultralytics"):
        return inner
    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(inner, torch.nn.Module):
        return inner
    return obj


def _eval_model_expr(expr: str):
    """`ultralytics.YOLO('yolo11n')` / `torchvision.models.resnet50` 같은 표현식 평가."""
    import importlib

    ns = {}
    for mod in ("torch", "torchvision", "torchvision.models", "ultralytics"):
        try:
            ns[mod.split(".")[0]] = importlib.import_module(mod.split(".")[0])
            importlib.import_module(mod)  # 서브모듈 로드 (torchvision.models 등)
        except Exception:
            pass
    obj = eval(expr, ns)              # noqa: S307 — 신뢰된 CLI 입력(사용자 본인 표현식)
    if callable(obj) and not isinstance(obj, torch.nn.Module):
        obj = obj()                  # `torchvision.models.resnet50` → 호출
    return obj


def _default_input_shape(model_spec: str, model):
    spec = (model_spec or "").lower()
    is_yolo = "yolo" in spec or model.__class__.__name__.lower().endswith("detectionmodel")
    return (1, 3, 640, 640) if is_yolo else (1, 3, 224, 224)


def build_model(model_spec: str, pth: str = None, input_shape=None):
    """`--model` 스펙을 (model, name, input_shape) 로 해석한다.

    - 표현식(`(` 포함): `_eval_model_expr` 로 평가 후 nn.Module unwrap.
    - 그 외: yolo* → ultralytics, 아니면 torchvision 분류 모델 이름.
    - `--pth`: state_dict 또는 전체 모델 파일을 모델에 로드(override).
    """
    from utils.module_util import get_module_name

    if model_spec and model_spec.endswith((".pt", ".pth")):
        obj = torch.load(model_spec, map_location="cpu", weights_only=False)
        model = _unwrap_module(obj)
    elif model_spec and ("(" in model_spec or "." in model_spec):
        # 표현식: `ultralytics.YOLO('yolo11n')` / `torchvision.models.resnet50`
        model = _unwrap_module(_eval_model_expr(model_spec))
    elif model_spec and "yolo" in model_spec.lower():
        from ultralytics import YOLO

        model = YOLO(model_spec).model
    else:
        import torchvision.models as tvm

        fn = getattr(tvm, model_spec)
        try:
            model = fn(weights="DEFAULT")   # 기본: pretrained 가중치
            print(f"  → torchvision {model_spec}: pretrained(DEFAULT) 가중치 로드")
        except Exception as e:
            print(f"  ⚠ {model_spec} pretrained 로드 실패({e}) → random 가중치")
            model = fn()

    model = model.cpu().eval()

    if pth:
        sd = torch.load(pth, map_location="cpu", weights_only=False)
        if isinstance(sd, torch.nn.Module):
            sd = sd.state_dict()
        elif isinstance(sd, dict) and "model" in sd and hasattr(sd["model"], "state_dict"):
            sd = sd["model"].state_dict()  # ultralytics .pt 체크포인트
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  → --pth 가중치 로드: {pth} (missing={len(missing)}, unexpected={len(unexpected)})")

    name = get_module_name(model)
    shape = tuple(input_shape) if input_shape else _default_input_shape(model_spec, model)
    return model, name, shape


# --------------------------------------------------------------------------
# 컴파일 (parse → export → ggml cpp/h + gguf + runner)
# --------------------------------------------------------------------------
def fold_conv_bn_graph(graph):
    """Conv-BN folding (그래프 레벨, repo 자체 패스 — FX/model.fuse() 불요).

    vision.cpp 의 `batch_norm_2d` 는 BN 이 conv 로 fused 됐다고 가정하고
    `running_mean`/`running_var` 가 GGUF 에 있으면 ASSERT 실패한다. repo 의
    `OptimizeCommander` 로 그래프에서 직접 fold:
    - `FuseBnToConv`: conv→BN 패턴을 찾아 conv params(weight/bias)에 BN 을 흡수하고
      BN 노드 제거(ConvBnHandler 가 numpy 데이터 직접 갱신).
    - `ConvertBNParams`: 잔여 standalone BN 을 weight/bias affine 으로 변환(running stats 제거)
      → vision.cpp `batch_norm_2d`(mul+add) 와 정합.
    folded weight 는 graph param 에 들어가며 generate_gguf 가 거기서 GGUF 를 쓴다.
    실패해도 best-effort(원본 그래프 유지). 모든 모델(resnet/yolo) 일관 적용.
    """
    try:
        from shared.optimization.commander import OptimizeCommander
        cmd = OptimizeCommander(graph=graph)
        cmd.FuseBnToConv()
        cmd.ConvertBNParams()
        n_bn = sum(getattr(n.op, "type", None) == "batch_norm" for n in graph.nodes)
        print(f"[g2c] Conv-BN fold (graph, repo pass) — 남은 batch_norm 노드={n_bn}", flush=True)
    except Exception as e:
        print(f"[g2c] Conv-BN fold 생략 ({type(e).__name__}: {e}); 원본 그래프 사용", flush=True)
    return graph


# xmodel 백엔드(DevGraphOptimizer)의 **topology-only 안전 패스**를 ggml codegen 앞에 재사용.
# 데이터/quant-태깅/HW-layout 의존 패스(constant_folding/broadcast/update_node_data/
# layout_tranform/partition)는 제외 — ggml 은 런타임 브로드캐스트·자체 layout 이라 불요/무의미.
# DevGraphOptimizer 는 입력 그래프를 clone(param data 보존 검증됨) → dev_graph 반환.
# 패스마다 점증 추가하며 ggml cos 회귀 검증 후 커밋한다.
_SAFE_DEV_OPTS = [
    "strip_redundant_ops",       # CONTIGUOUS 등 잉여 op 제거
]


def apply_dev_graph_opts(graph):
    """_SAFE_DEV_OPTS 를 순차 적용(best-effort). 반환: 최적화된 dev_graph(clone) 또는 원본."""
    if not _SAFE_DEV_OPTS:
        return graph
    try:
        from shared.compile.deploy_optimizer import DevGraphOptimizer
        opt = DevGraphOptimizer(graph)
        applied = []
        for name in _SAFE_DEV_OPTS:
            getattr(opt, name)()
            applied.append(name)
        print(f"[g2c] dev-graph opts: {', '.join(applied)} "
              f"(nodes {len(list(graph.nodes))}→{len(list(opt.dev_graph.nodes))})", flush=True)
        return opt.dev_graph
    except Exception as e:
        print(f"[g2c] dev-graph opts 생략 ({type(e).__name__}: {e}); 원본 그래프 사용", flush=True)
        return graph


def compile_model(model, name: str, input_shape, output_dir: str, quant=None):
    """모델을 vision.cpp(ggml) arch C++ + GGUF 로 컴파일한다.

    반환: 생성 파일 dict (source/header/weights_manifest) 또는 None(파싱 실패).
    """
    import traceback

    from parse import TorchParser
    from parse.rich_in_out_helper import StandardInputData
    from utils import TorchSymbol

    os.makedirs(output_dir, exist_ok=True)
    inputs = torch.randn(*input_shape).cpu()
    print(f"[g2c] Input shape: {tuple(inputs.shape)}", flush=True)

    graph = None
    try:
        print("[g2c] Parsing graph (TorchParser)...", flush=True)
        graph = TorchParser()(name, model, StandardInputData((inputs,), {}))
        fold_conv_bn_graph(graph)   # vision.cpp 정합: BN 을 conv 로 흡수(export·.cpp·GGUF 일관)
        graph = apply_dev_graph_opts(graph)   # topology-only 안전 패스(xmodel 재사용)
        print(f"[g2c] Graph nodes: {len(list(getattr(graph, 'nodes', [])))}", flush=True)

        from qproc.export import get_script_writer

        export_name = name + TorchSymbol.SCRIPT_SUFFIX
        get_script_writer(enable_quant=True).write(
            graph, file_path=os.path.join(output_dir, export_name)
        )
        print(f"[g2c] Export written: {output_dir}/{export_name}", flush=True)
    except Exception:
        print("[g2c] Parse/export 실패:")
        traceback.print_exc()
        return None

    # 양자화 단일 진실원천 — codegen(conv_2d_q emit)·gguf(2D 양자 저장) 가 같은 plan 사용.
    _, _, quant_plan = build_quant_plan(graph, quant)

    # 입력 무관 상수 subgraph(anchor/stride/scalar) → numpy 평가 후 GGUF baking.
    # codegen 과 gguf 가 같은 fold 결과(키)를 공유해야 정합 → 여기서 한 번만 계산.
    from shared.compile.const_fold import fold_constants

    baked, skip = fold_constants(graph, (input_shape[-2], input_shape[-1]))
    if baked:
        print(f"[g2c] const-fold: {len(baked)} baked tensor(s), {len(skip)} node(s) skipped",
              flush=True)

    print("[g2c] Generating visp/ggml arch C++...", flush=True)
    files = generate_ggml_code(graph, output_dir, name, quant_plan=quant_plan,
                               baked=baked, skip=skip)
    for ftype, fpath in files.items():
        print(f"  → {ftype}: {fpath}")

    generate_gguf(graph, output_dir, name, name.lower(), quant=quant, plan=quant_plan,
                  baked=baked)

    from utils import TorchSymbol as _TS

    append_ggml_runner(
        os.path.join(output_dir, name + _TS.SCRIPT_SUFFIX),
        name,
        f"{name}.gguf",
        input_shape,
    )
    print(f"  → 실행: python {output_dir}/{name}{_TS.SCRIPT_SUFFIX} (ggml/libggml.so 커널)")
    return files


# --------------------------------------------------------------------------
# CLI 진입점 (g2c)
# --------------------------------------------------------------------------
def main(argv=None):
    import argparse
    from utils.module_util import get_module_name  # noqa: F401 (지연 import 검증)

    ap = argparse.ArgumentParser(
        prog="g2c",
        description="GTX Compiler: PyTorch 모델 → vision.cpp(ggml) arch C++ + GGUF",
    )
    ap.add_argument(
        "--model", required=True,
        help="모델 스펙: 표현식(\"ultralytics.YOLO('yolo11n')\"), 이름(resnet18/yolov8n), "
             "또는 .pt/.pth 경로",
    )
    ap.add_argument("--pth", default=None, help="state_dict/.pth 가중치 (모델에 로드)")
    ap.add_argument("--output", default="output", help="출력 디렉터리")
    ap.add_argument("--name", default=None, help="아키텍처/파일 이름 (기본: 모델 클래스명)")
    ap.add_argument(
        "--input-shape", default=None,
        help="입력 shape (예: 1,3,640,640). 미지정 시 yolo→640, 그 외→224",
    )
    ap.add_argument(
        "--quantize", default=None, metavar="TYPE",
        help="GGUF 가중치 양자화. (A) in-process(외부 의존 없음): q8_0/q4_0/q4_1/q5_0/q5_1 — "
             "자격되는 2D Linear 가중치만, 나머지 fp16. (B) llama-quantize 경유: q4_k_m/q6_k/iq4_xs "
             "등 — $LLAMA_QUANTIZE/PATH 의 바이너리 필요(vision arch 인식 가능해야). 미지정=fp16.",
    )
    args = ap.parse_args(argv)

    shape = None
    if args.input_shape:
        shape = tuple(int(s) for s in args.input_shape.replace(" ", "").split(","))

    print("[g2c] Start", flush=True)
    model, name, shape = build_model(args.model, pth=args.pth, input_shape=shape)
    name = args.name or name
    print(f"[g2c] Model: {args.model} → {model._get_name()} (name={name})", flush=True)

    compile_model(model, name, shape, args.output, quant=args.quantize)
    print("=" * 60)
    print("완료!", flush=True)


if __name__ == "__main__":
    main()
