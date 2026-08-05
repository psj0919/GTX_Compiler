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
from rich import box
from rich.console import Console
from rich.table import Table

_console = Console()

# 그래프 최적화 Action → 색상 (컴파일 로그 가독성)
_ACTION_COLOR = {
    "REMOVED": "red", "FUSED": "cyan", "FOLDED": "yellow",
    "CONVERTED": "magenta", "NORMALIZED": "blue",
}


def _action_cell(action):
    return f"[{_ACTION_COLOR.get(action, 'white')}]{action}[/]"

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
            # vision.cpp batch_norm_2d(nn.cpp:150)는 running_mean/var 가 GGUF 에 있으면 ASSERT
            # 실패(affine weight/bias 만 기대). ConvertBNParams 가 이미 BN 을 affine(weight=scale,
            # bias=offset, mean=0/var=1)으로 변환했으므로 running stats 는 불필요 → GGUF 에서 제외.
            # (fold 된 BN 은 노드 자체가 없어 무관. g2c 는 ensure_bn_affine 로 변환 보장.)
            if key.endswith(("num_batches_tracked", "running_mean", "running_var")):
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
        elif not np.issubdtype(np.asarray(arr).dtype, np.floating):
            # 정수 버퍼(예: Swin 의 `relative_position_index`)를 float 으로 캐스팅하면 안 된다 —
            # `ggml_get_rows` 는 인덱스 텐서가 **I32** 여야 하고, 아니면
            # `GGML_ASSERT(b->type == GGML_TYPE_I32)` 로 죽는다(swin·glip·grounding_dino 공통).
            writer.add_tensor(key, np.ascontiguousarray(arr).astype(np.int32))
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
                       input_shape=(1, 3, 224, 224), profile=False, profile_reps=1):
    """export 파일 끝에 ggml 실행 진입점을 추가한다.

    `python output/<Model>.py` 직접 실행 시 ggml(libggml.so) 커널로 forward 가 돈다:
    set_backend('ggml') → nn.Module → GgmlModule → run_gguf 가 옆 .gguf 를 이름 바인딩 후 실행.
    profile: True 면 런타임 프로파일 env 를 파일 최상단에 baking(nn import 전에 set →
    ggml_profiler 가 계측 활성화). setdefault 라 실행 시 env 로 오버라이드 가능.
    """
    marker = "# ----- ggml 실행 진입점 (libggml.so) -----"
    src = open(export_path).read()
    if marker in src:
        return

    # 프로파일 env 는 nn(→ggml_profiler) import 전에 설정돼야 계측이 켜진다 → boot 최상단.
    prof = ""
    if profile:
        prof = (f'os.environ.setdefault("GTX_PROFILE", "1")\n'
                f'os.environ.setdefault("GTX_PROFILE_REPS", "{int(profile_reps)}")\n')
    # `import nn` 이 되도록 g2c 를 import 한다 — 설치된 g2c(g2c/__init__.py)가 토킷 루트를
    # sys.path 에 올리고 nn 등 top-level 패키지를 노출한다(uv 설치 전제). 프로파일 env 는
    # g2c→nn(→ggml_profiler) import 전에 설정돼야 하므로 g2c import 앞에 둔다.
    boot = "import g2c  # noqa: F401 — 루트 sys.path 등록 + nn 노출\n"
    if prof:
        boot = "import os\n" + prof + boot
    if "import g2c" not in src:
        src = src.replace("import nn\n", boot + "import nn\n", 1)

    shape = ", ".join(str(int(s)) for s in input_shape)
    block = f'''

{marker}
if __name__ == "__main__":
    import os
    import numpy as np
    import nn

    nn.set_backend("ggml")
    here = os.path.dirname(os.path.abspath(__file__))
    gguf_path = os.path.join(here, "{gguf_name}")
    x = np.random.randn({shape}).astype("float32")
    reps = int(os.environ.get("GTX_PROFILE_REPS", "1") or "1")   # 프로파일 N회 반복
    for r in range(max(1, reps)):
        out = nn.run_gguf({class_name}(), gguf_path, __file__, x)
    main = out[0] if isinstance(out, (list, tuple)) else out
    main = np.asarray(main)
    print("[ggml] output:", main.shape)
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
# 그래프 최적화 Pass 별 로그 (PDF p5: Layer|Op|Action|Note + Pass 헤더 N→M)
# --------------------------------------------------------------------------
# 옵티마이저 내부를 건드리지 않고, 각 pass 전후 노드 스냅샷(name→op)을 diff 해
# 제거/타입변환 노드를 뽑는다. 계측은 report(list) 누적 → _render_graph_opt_report.
_PASS_META = {
    # pass_name: (removed 노드에 붙일 Action, Note)
    "fold_conv_bn":                  ("FUSED",      "BN 을 conv 로 흡수"),
    "strip_redundant_ops":           ("REMOVED",    "잉여 op(contiguous 등)"),
    "fuse_pad":                      ("FUSED",      "pad → conv/pool attr"),
    "fuse_transpose_matmul":         ("FUSED",      "transpose+matmul → matmul"),
    "fuse_redundant_transpose":      ("REMOVED",    "상쇄되는 연속 transpose"),
    "merge_permute_to_linear":       ("FUSED",      "permute → linear weight 재배열"),
    "merge_consecutive_reshape":     ("FUSED",      "연속 reshape 병합"),
    "convert_shape_tensor_to_const": ("FOLDED",     "shape 텐서 → const"),
    "convert_rsub_to_sub":           ("CONVERTED",  "rsub → sub"),
    "normalize_pad_nd":              ("NORMALIZED", "pad N-D 정규화"),
    "const_fold":                    ("FOLDED",     "입력-무관 상수 subgraph → baked"),
}


def _node_snapshot(graph):
    """현재 노드 {name: op_type}. pass 전후 diff 용."""
    return {n.name: str(getattr(n.op, "type", "?")) for n in list(graph.nodes)}


def _pass_changes(before, after, pass_name):
    """before/after 스냅샷 diff → 변경 노드 레코드 리스트 [(layer, op, action, note)].

    removed = before 에만 존재(제거/fuse), converted = 이름 동일·op 타입 변경.
    """
    action, note = _PASS_META.get(pass_name, ("REMOVED", pass_name))
    rows = []
    for name, op in before.items():
        if name not in after:
            rows.append((name, op, action, note))
        elif after[name] != op:
            rows.append((name, op, "CONVERTED", f"{op} → {after[name]}"))
    return rows


def _pass_record(pass_name, before, after, changes):
    return {"pass": pass_name,
            "before": len(before), "after": len(after), "changes": changes}


def _render_graph_opt_report(report, init_ops, final_ops):
    """Pass 별 표 렌더 (PDF p5) — Rich 테이블."""
    if not report:
        return
    pct = (100.0 * (init_ops - final_ops) / init_ops) if init_ops else 0.0
    _console.rule(f"[bold cyan]GRAPH OPTIMIZATION[/]  "
                  f"{init_ops} ops → {final_ops} ops [bold green](-{pct:.0f}%)[/]",
                  align="left")
    for rec in report:
        d = rec["before"] - rec["after"]
        sign = f"-{d}" if d >= 0 else f"+{-d}"
        title = (f"▸ [bold]{rec['pass']}[/]  "
                 f"{rec['before']} → {rec['after']} [yellow]({sign} ops)[/]")
        if not rec["changes"]:
            _console.print(f"  {title}   [dim](변경 노드 없음)[/]")
            continue
        table = Table(title=title, title_justify="left", box=box.SIMPLE_HEAD,
                      show_edge=False, pad_edge=False, header_style="bold")
        table.add_column("Layer", style="cyan", no_wrap=True, max_width=40)
        table.add_column("Op", style="white")
        table.add_column("Action")
        table.add_column("Note", style="dim")
        for layer, op, action, note in rec["changes"]:
            # IR 노드명은 계층적이라 구분되는 부분이 뒤쪽 → 길면 꼬리를 남긴다.
            lyr = layer if len(layer) <= 40 else "…" + layer[-39:]
            table.add_row(lyr, op[:15], _action_cell(action), note)
        _console.print(table)
    _console.print()


def _render_compile_summary(graph, skip, init_ops):
    """컴파일 최종 요약 (PDF p4,p7): Ops 표 + Operator breakdown + Dispatch.

    현재 실행 backend 는 ggml CPU → dispatch fallback·CAST 없음(전면 fp16). 따라서
    Target 은 전부 CPU(ggml), cast 0. (NPU 는 목표 아키텍처.)
    """
    from collections import Counter

    skip = skip or set()
    emitted = [n for n in graph.nodes if id(n) not in skip]   # codegen 이 실제 emit
    final = len(emitted)
    removed = init_ops - final
    brk = Counter(str(getattr(n.op, "type", "?")) for n in emitted)

    _console.rule("[bold cyan]COMPILE SUMMARY[/]", align="left")

    ops_t = Table(box=box.ROUNDED, show_edge=True, pad_edge=False, header_style="bold")
    ops_t.add_column("Ops", style="white")
    ops_t.add_column("Original", justify="right")
    ops_t.add_column("After opt", justify="right", style="cyan")
    ops_t.add_column("After dispatch", justify="right", style="green")
    ops_t.add_row("total", str(init_ops), str(final), str(final))
    ops_t.add_row("removed", "", f"[red]-{removed}[/]", f"[red]-{removed}[/]")
    ops_t.add_row("cast inserted", "", "", "0")
    _console.print(ops_t)

    brk_t = Table(title="Operator breakdown (final)", title_justify="left",
                  box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
    brk_t.add_column("Op", style="cyan")
    brk_t.add_column("Count", justify="right")
    for op, c in brk.most_common():
        brk_t.add_row(op, str(c))
    brk_t.add_section()
    brk_t.add_row("[bold]TOTAL[/]", f"[bold]{final}[/]")
    _console.print(brk_t)

    disp_t = Table(title="Dispatch (현재 backend = ggml CPU · NPU 는 목표 arch)",
                   title_justify="left",
                   box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
    disp_t.add_column("Target")
    disp_t.add_column("Ops", justify="right")
    disp_t.add_column("Ratio", justify="right")
    disp_t.add_row("[green]CPU (ggml)[/]", str(final), "100.0%")
    disp_t.add_row("[dim]NPU (목표)[/]", "0", "0.0%")
    _console.print(disp_t)
    _console.print()


def _render_dispatch_table(graph, skip, quant_plan):
    """레이어별 DISPATCH & FALLBACK 표 (PDF p6): Layer|Op|Target|DType|Note.

    현재 실행 backend 는 **ggml CPU** (libggml.so CPU 커널) → Target 전부 CPU(ggml).
    NPU 는 목표 아키텍처. fallback/CAST 삽입 없음. DType 은 `--quantize` 시 해당
    conv/linear 만 INT8, 그 외 FP16.
    """
    from shared.compile.render_api import weight_key

    skip = skip or set()
    quant_plan = quant_plan or {}
    emitted = [n for n in graph.nodes if id(n) not in skip]

    _console.rule("[bold cyan]DISPATCH & FALLBACK[/]", align="left")
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
    table.add_column("Layer", style="cyan", no_wrap=True, max_width=40)
    table.add_column("Op", style="white")
    table.add_column("Target")
    table.add_column("DType")
    table.add_column("Note", style="dim")
    for n in emitted:
        name = getattr(n, "name", "?") or "?"
        lyr = name if len(name) <= 40 else "…" + name[-39:]
        op = str(getattr(n.op, "type", "?"))
        int8 = weight_key(n) in quant_plan
        dtype = "[yellow]INT8[/]" if int8 else "[cyan]FP16[/]"
        table.add_row(lyr, op[:15], "[green]CPU (ggml)[/]", dtype, "")
    _console.print(table)
    _console.print("  [dim]현재 backend = ggml CPU (NPU 는 목표 arch). "
                   "fallback/CAST 삽입 없음[/]")
    _console.print()


# --------------------------------------------------------------------------
# 컴파일 (parse → export → ggml cpp/h + gguf + runner)
# --------------------------------------------------------------------------
def fold_conv_bn_graph(graph, report=None):
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
        before = _node_snapshot(graph)
        cmd = OptimizeCommander(graph=graph)
        cmd.FuseBnToConv()
        cmd.ConvertBNParams()
        n_bn = sum(getattr(n.op, "type", None) == "batch_norm" for n in graph.nodes)
        if report is not None:
            after = _node_snapshot(graph)
            report.append(_pass_record(
                "fold_conv_bn", before, after,
                _pass_changes(before, after, "fold_conv_bn")))
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
    "fuse_pad",                  # 명시적 Pad 노드 → conv/pool pad attr 흡수
    "fuse_transpose_matmul",     # transpose+matmul → 단일 matmul(transpose flag)
    "fuse_redundant_transpose",  # 상쇄되는 연속 transpose 제거
    "merge_permute_to_linear",   # linear 앞 permute 를 weight 축 재배열로 흡수
    "merge_consecutive_reshape", # 연속 reshape 를 하나로 병합
    # convert_reshapelike_to_reshape — 제외: reshape 로 바꾼 노드의 shape attr 이 None
    #   (xmodel 은 blob/동적으로 채움) → ggml _op_reshape 실행 실패. ggml 부적합.
    "convert_shape_tensor_to_const",   # shape 텐서(정적) → const 노드
    "convert_rsub_to_sub",       # rsub(a-x) → sub 정규화
    # convert_adaptive_pool_to_pool — 제외: 이 패스는 shape[1:3]=[H,W](NHWC/xmodel
    #   레이아웃) 가정으로 kernel 계산. g2c 그래프는 NCHW(shape[1:3]=[C,H])라 kernel 이
    #   엉뚱한 축([7,1])으로 나와 pool 출력 오류→mul_mat 크래시. avgpool 자체는 정상
    #   (네이티브 nn.AvgPool2d 는 ggml/vision.cpp end-to-end 동작). adaptive 도 이미 cos=1.0.
    "normalize_pad_nd",          # pad 표현 정규화(N-D)
    # update_op_attrs — 제외: permute 매핑을 등록해도 ggml _op_permute 가 'dims' kwarg 를
    #   못 받아 KeyError. transpose 는 이미 cos=1.0 → 변환 이득 없음.
]


def apply_dev_graph_opts(graph, report=None):
    """_SAFE_DEV_OPTS 를 순차 적용(best-effort). 반환: 최적화된 dev_graph(clone) 또는 원본."""
    if not _SAFE_DEV_OPTS:
        return graph
    try:
        from shared.compile.deploy_optimizer import DevGraphOptimizer
        opt = DevGraphOptimizer(graph)
        applied = []
        for name in _SAFE_DEV_OPTS:
            before = _node_snapshot(opt.dev_graph)
            getattr(opt, name)()
            applied.append(name)
            if report is not None:
                after = _node_snapshot(opt.dev_graph)
                changes = _pass_changes(before, after, name)
                if changes:   # 변경 없는 pass 는 노이즈라 표에서 생략
                    report.append(_pass_record(name, before, after, changes))
        print(f"[g2c] dev-graph opts: {', '.join(applied)} "
              f"(nodes {len(list(graph.nodes))}→{len(list(opt.dev_graph.nodes))})", flush=True)
        return opt.dev_graph
    except Exception as e:
        print(f"[g2c] dev-graph opts 생략 ({type(e).__name__}: {e}); 원본 그래프 사용", flush=True)
        return graph


def _convert_bn_affine(graph, report=None):
    """BN op 을 vision.cpp `batch_norm_2d`(affine mul+add) 형태로 변환(fold 없이).

    `ConvertBNParams` 로 running_mean/var 를 weight/bias affine 에 흡수해 제거한다. BN 노드는
    **그래프에 남지만**(nn.cpp:150 batch_norm_2d 가 처리) running stats 가 없어 GGUF/ASSERT
    정합. fold(BN→conv 흡수)와 달리 노드 토폴로지는 유지 — g2c 가 레벨 무관 항상 적용해야
    --Opt 0/2 에서도 유효한 GGUF 를 쓴다."""
    try:
        from shared.optimization.commander import OptimizeCommander
        before = _node_snapshot(graph)
        OptimizeCommander(graph=graph).ConvertBNParams()
        if report is not None:
            after = _node_snapshot(graph)
            report.append(_pass_record("bn_affine", before, after,
                                       _pass_changes(before, after, "bn_affine")))
    except Exception as e:
        print(f"[opt] BN affine 변환 생략 ({type(e).__name__}: {e})", flush=True)
    return graph


def apply_graph_opts(graph, level=0, report=None, ensure_bn_affine=False):
    """--Opt 레벨별 그래프 최적화 — g2c(compile_model)·graph_visualizer 공유.

      0: 없음 (raw 파싱 그래프)
      1: Memory 최적화 — DevGraphOptimizer (topology-only 안전 패스, apply_dev_graph_opts)
      2: OP 최적화   — OptimizeCommander (`FuseBnToConv` + `ConvertBNParams`, BN→conv 흡수)
      3: 1 + 2

    ensure_bn_affine=True (g2c 컴파일): 레벨과 무관하게 BN 을 vision.cpp affine 형태로 변환
      (`_convert_bn_affine`) → --Opt 0/1 에서도 running stats 없는 유효 GGUF. 시각화(False)는
      level 0 을 진짜 raw(BN+running stats 그대로)로 둔다.
    const-fold 와 act-fuse(`--fuse-activation`)는 여기 미포함 — caller 별 opt-in.
    반환: 최적화된 graph(dev-opts 는 clone 반환).
    """
    if level in (2, 3):
        fold_conv_bn_graph(graph, report=report)      # FuseBnToConv + ConvertBNParams
    elif ensure_bn_affine:
        _convert_bn_affine(graph, report=report)      # ConvertBNParams 만 (fold 없이)
    if level in (1, 3):
        graph = apply_dev_graph_opts(graph, report=report)
    return graph


def compile_model(model, name: str, input_shape, output_dir: str, quant=None,
                  profile=False, profile_reps=1, opt_level=0,
                  visualize=None, visualize_fmt="svg"):
    """모델을 vision.cpp(ggml) arch C++ + GGUF 로 컴파일한다.

    profile: True 면 생성 runner 에 런타임 프로파일(GTX_PROFILE)을 baking.
    opt_level: --Opt (0 없음/1 Memory최적화/2 OP최적화/3 둘다). g2c 는 레벨과 무관하게
      BN affine 변환·const-fold(컴파일 필수)를 항상 적용하고 그 위에 레벨 최적화를 얹는다.
    visualize: 경로(문자열)면 최종 컴파일 그래프를 그 경로에 DOT(+이미지)로 시각화.
      "__auto__" 면 `<output_dir>/<name>_graph`. None 이면 생략.
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
        print("[g2c] Parsing graph...", flush=True)
        graph = TorchParser()(name, model, StandardInputData((inputs,), {}))
        opt_report = []             # 그래프 최적화 pass 별 변경 노드 로그(PDF p5)
        init_ops = len(list(getattr(graph, "nodes", [])))
        # 공용 최적화(--Opt 레벨). g2c 는 BN affine 변환을 항상(ensure_bn_affine) → GGUF 정합.
        graph = apply_graph_opts(graph, level=opt_level, report=opt_report,
                                 ensure_bn_affine=True)
        print(f"[g2c] Graph nodes: {len(list(getattr(graph, 'nodes', [])))}", flush=True)
    except Exception:
        print("[g2c] Parse 실패:")
        traceback.print_exc()
        return None

    # .py(ggml 러너) export 는 **보조 산출물**이다. ScriptWriter 가 모르는 동적 op
    # (linear_dynamic / variance / conv2d_dynamic 등)이 있으면 여기서 죽는데, 주 산출물인
    # cpp/gguf 는 그 op 을 렌더할 수 있다. 같이 죽이면 op 하나 때문에 계열 전체가 날아간다
    # (pvt: MultiheadAttention 의 linear_dynamic 하나로 컴파일 전체 실패).
    # → 실패해도 삼키고 cpp/gguf 는 계속 만든다. 중간까지 쓰인 부분 .py 는 무효라 지운다.
    export_name = name + TorchSymbol.SCRIPT_SUFFIX
    export_path = os.path.join(output_dir, export_name)
    try:
        from qproc.export import get_script_writer

        get_script_writer(enable_quant=True).write(graph, file_path=export_path)
        print(f"[g2c] Export written: {export_path}", flush=True)
    except Exception:
        print("[g2c] .py export 건너뜀(ScriptWriter 미지원 op) — cpp/gguf 는 계속:")
        traceback.print_exc()
        if os.path.exists(export_path):
            os.remove(export_path)

    # 양자화 단일 진실원천 — codegen(conv_2d_q emit)·gguf(2D 양자 저장) 가 같은 plan 사용.
    _, _, quant_plan = build_quant_plan(graph, quant)

    # 입력 무관 상수 subgraph(anchor/stride/scalar) → numpy 평가 후 GGUF baking.
    # codegen 과 gguf 가 같은 fold 결과(키)를 공유해야 정합 → 여기서 한 번만 계산.
    from shared.compile.const_fold import fold_constants

    # const-fold 는 **기본 off** 다. skip 된 노드의 출력이 baked 에 안 들어가는 경우가 있어
    # 하류 `ctx.inp()` 가 그래프 입력 x(= 이미지)로 폴백한다 — PVT 는 pos_embed 슬라이스가
    # 접히면서 `ggml_add(ln4, permute(x))` 가 나와 can_repeat 로 죽었다.
    # (실측: pvt 에서 410 노드 skip / 16 텐서만 bake. deploy 는 이 기능 자체가 없고 100/100.)
    # G2C_CONST_FOLD=1 로 켤 수 있다.
    import os as _cf_os
    if _cf_os.environ.get("G2C_CONST_FOLD"):
        baked, skip = fold_constants(graph, (input_shape[-2], input_shape[-1]))
    else:
        baked, skip = {}, set()
    if baked:
        print(f"[g2c] const-fold: {len(baked)} baked tensor(s), {len(skip)} node(s) skipped",
              flush=True)
    # const-fold 는 노드를 그래프에서 지우지 않고 skip(id) 로 codegen 제거 → 표에 반영.
    cur_ops = len(list(getattr(graph, "nodes", [])))
    if skip:
        action, note = _PASS_META["const_fold"]
        rows = [(n.name, str(getattr(n.op, "type", "?")), action, note)
                for n in graph.nodes if id(n) in skip]
        opt_report.append({"pass": "const_fold", "before": cur_ops,
                           "after": cur_ops - len(skip), "changes": rows})
    _render_graph_opt_report(opt_report, init_ops, cur_ops - len(skip))
    _render_dispatch_table(graph, skip, quant_plan)
    _render_compile_summary(graph, skip, init_ops)

    # Conv+Activation fuse(--Opt 2/3, OP 최적화)는 **export(.py) 이후** 적용 → 생성 .py 는 relu 노드를
    # 유지(eager 검증 무손상), .cpp codegen 은 fused 마커에서 활성화를 emit(수치 동일).
    if opt_level in (2, 3):
        from shared.optimization.commander import OptimizeCommander
        n_act = OptimizeCommander(graph=graph).FuseConvActivation()
        if n_act:
            print(f"[g2c] Conv+Activation fuse — {n_act} activation fused", flush=True)

    # 그래프 시각화(--visualize): 최종 컴파일 그래프(opt·fold·act-fuse 반영)를 DOT+이미지로.
    if visualize:
        from shared.compile.graph_viz import visualize_graph
        graph.viz_skip_ids = skip                 # const-fold 로 제거된 노드는 뷰에서도 제외
        viz_out = (os.path.join(output_dir, f"{name}_graph")
                   if visualize == "__auto__" else visualize)
        visualize_graph(graph, viz_out, fmt=visualize_fmt)

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
        profile=profile,
        profile_reps=profile_reps,
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
    ap.add_argument(
        "--profile", action="store_true",
        help="컴파일 후 생성 모델을 실행해 런타임 프로파일(op 별 latency/메모리/Time Share)을 "
             "계측·출력(콘솔 표 + CSV/JSON/Perfetto). env `GTX_PROFILE`/`GTX_PROFILE_REPS` 로도 오버라이드.",
    )
    ap.add_argument(
        "--profile-reps", type=int, default=None, metavar="N",
        help="프로파일 반복 횟수 N (1회차 warmup 제외 avg/max/min). 지정 시 --profile 자동 on. 기본 1.",
    )
    ap.add_argument(
        "--Opt", type=int, default=3, choices=[0, 1, 2, 3],
        help="그래프 최적화 수준. 0: 없음 / 1: Memory 최적화(DevGraphOptimizer: topology-only) / "
             "2: OP 최적화(OptimizeCommander: Conv-BN fold) / 3: 1+2. "
             "g2c 는 레벨과 무관하게 BN affine 변환·const-fold(컴파일 필수)를 항상 적용.",
    )
    ap.add_argument(
        "--visualize", nargs="?", const="__auto__", default=None, metavar="PATH",
        help="최종 컴파일 그래프를 graphviz DOT(+이미지)로 시각화. PATH 생략 시 "
             "`<output>/<name>_graph`. Netron/graphviz 로 op·attr·shape 확인.",
    )
    ap.add_argument(
        "--visualize-format", default="svg", choices=["png", "svg", "pdf"],
        help="--visualize 이미지 포맷 (기본 svg).",
    )
    args = ap.parse_args(argv)

    shape = None
    if args.input_shape:
        shape = tuple(int(s) for s in args.input_shape.replace(" ", "").split(","))

    print("[g2c] Start", flush=True)
    model, name, shape = build_model(args.model, pth=args.pth, input_shape=shape)
    name = args.name or name
    print(f"[g2c] Model: {args.model} → {model._get_name()} (name={name})", flush=True)

    profile_on = args.profile or (args.profile_reps is not None)
    reps = args.profile_reps or 1
    files = compile_model(model, name, shape, args.output, quant=args.quantize,
                          profile=profile_on, profile_reps=reps,
                          opt_level=args.Opt,
                          visualize=args.visualize, visualize_fmt=args.visualize_format)
    # --profile: 생성 모델을 실제 실행해 프로파일(콘솔+CSV/JSON/Perfetto)을 산출.
    if profile_on and files is not None:
        _run_profile(args.output, name, reps)
    print("=" * 60)
    print("완료!", flush=True)


def _run_profile(output_dir, name, reps):
    """생성된 runner 를 GTX_PROFILE 로 실행 → 프로파일 콘솔/CSV/JSON/Perfetto 산출."""
    import subprocess
    from utils import TorchSymbol as _TS

    runner = os.path.join(output_dir, name + _TS.SCRIPT_SUFFIX)
    env = dict(os.environ, GTX_PROFILE="1", GTX_PROFILE_REPS=str(int(reps)),
               GTX_PROFILE_OUT=output_dir)
    print(f"[g2c] Profiling: {runner} (reps={reps}) …", flush=True)
    subprocess.run([sys.executable, runner], env=env, check=False)


if __name__ == "__main__":
    main()
