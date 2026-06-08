# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

PyTorch 모델을 **vision.cpp(ggml) arch 스타일 C++ + GGUF 가중치**로 변환하는 컴파일러. 자세한 사용법은 `README.md` 참조.

## 명령어

패키지 매니저는 **uv** (`pyproject.toml`). 정식 CLI 는 콘솔 스크립트 **`g2c`**
(= `shared.compile.pipeline:main`, pyproject `[project.scripts]`).

```bash
git submodule update --init --recursive                    # vision.cpp (+ depend/llama)
uv sync                                                     # 의존성 + g2c CLI 설치 (torch cu124 인덱스)

# 컴파일: PyTorch 모델 → output/<Model>.{cpp,h,gguf,py}
uv run g2c --model "ultralytics.YOLO('yolo11n')" --output output/yolo11n
uv run g2c --model resnet18 --output output/resnet18
uv run g2c --model torchvision.models.resnet50 --pth W.pth --name r50 --output output/r50
#   --model(필수: 표현식/이름/.pt) --pth(가중치) --output --name --input-shape(미지정 시 yolo→640,그외→224)
#   ultralytics 래퍼는 .model(백본) 자동 추출.

# 생성물을 ggml(libggml.so) 커널로 직접 실행 (검증용)
uv run python output/yolo11n/DetectionModel.py             # [ggml] output: (1, ...)

# 과거 진입점(테스트 shim, 동일 파이프라인)
uv run python test/compile_to_c.py --model yolov8n --output output/yolov8n
```

> **머신 주의:** 무거운 trace(`g2c`/parse) 실행 시 thread 폭주로 hang 가능 →
> `OMP_NUM_THREADS=1 ... torch.set_num_threads(1)` + 한 번에 하나씩 순차 실행.

## 컴파일 파이프라인 (big picture)

```
PyTorch Module
  └─ parse/TorchParser: torch.jit.trace → Torch Graph(aten) → op_dispatcher → Graph IR
        ↓
 Graph IR  (shared/graph/ — Node=Operation+in/out Tensor, Operation=type+params(numpy)+attrs)
        ↓  optimization/ (Conv-BN fusion 등), quantization/
ScriptWriter (qproc/export/) → export1/*.py / output/<Model>.py  (op-chain, ggml 실행 진입점 포함)
        ↓
shared/compile/  VispCodeGenerator(ggml_codegen.py) + op별 render(node,ctx) (render_api 레지스트리)
        ↓
output/<Model>.cpp/.h  (visp arch 스타일 ggml 그래프)  +  output/<Model>.gguf (state_dict→fp16)
```

### 핵심 설계 (여러 파일을 읽어야 이해되는 것들)

- **op render 레지스트리:** `shared/compile/render_api.py` 의 `RENDERERS` + `@register_render(OP.X)`.
  각 op 의 C++ 렌더링은 `nn/modules/<op>.py` 의 module-level `render(node, ctx)` 에 있고,
  `VispCodeGenerator` 가 Graph IR 를 순회하며 디스패치한다. `RenderContext`: `inp`/`attr`/
  `weight`/`out`/`out_shape`/`ggml_axis`/`bind_outputs`. 새 op 지원 = 그 모듈에 render 추가.

- **detection-head / 비전 op render:** `nn/modules/head_render.py`(anchor/DFL/NMS-free 디코드:
  permute/transpose/stack/strided_slice/gather/max/topk/meshgrid/const/full/floor_divide…) +
  `nn/modules/vision_ops_render.py`(gelu/tanh/leaky_relu/clamp/layer_norm/group_norm/sqrt/mean…).
  전부 **실제 ggml 그래프 op** 으로 emit (passthrough 없음). torch dim → ggml ne 축은
  `ggml_axis()`(ne 는 torch shape 역순)로 변환.

- **연산자 enum:** `shared/base/key_names.py` 의 `OP`(plain str 클래스, `OP.CONV2D=='conv2d'`).
  PyTorch↔op 매핑은 `utils/op_register.py`, torch op 변환은 `parse/op_dispatcher.py` + `parse/torch_op_def.py`.

- **가중치(GGUF):** `pipeline.generate_gguf` 가 state_dict 키(`layer1.0.conv1.weight`)를 그대로
  fp16 GGUF 텐서명으로 쓴다 → 생성 .cpp 의 `m["layer1.0.conv1"]` 와 정합. arch 메타=`general.architecture`.

- **ggml 런타임 백엔드:** `nn/modules/ggml_backend.py` (`set_backend('ggml')`). 생성 export 를
  ggml(libggml.so) 커널로 eager 실행 — codegen render 의 수치 기준(host numpy, torch dim 의미).

## 디렉토리 역할

- `g2c` (= `shared/compile/pipeline.py`) — E2E CLI: 모델 로드 → 그래프 → export → ggml cpp/h + gguf
- `shared/compile/` — `pipeline.py`(g2c), `ggml_codegen.py`(VispCodeGenerator), `render_api.py`(render 프레임워크)
- `shared/` — `graph/`(IR), `optimization/`, `quantization/`, `base/`(OP·GlobalMap 상수), `utils/`
- `parse/` — TorchParser (PyTorch→Graph)
- `qproc/` — ModuleHooker(activation 캡처·param 동기화), adaquant 양자화, export(ScriptWriter)
- `quantization/`, `utils/` — 양자화 알고리즘 / op 등록·타입 매핑·hw dtype(fp16/bf16/fp8/int8)
- `nn/modules/` — op별 PyTorch 모듈 + `render()`, `head_render.py`/`vision_ops_render.py`, `ggml_backend.py`
- `vision.cpp/` — (git submodule) ggml/GGUF 추론 라이브러리. `depend/llama/gguf-py` 가 GGUF writer.
- `test/compile_to_c.py` — 과거 진입점(테스트 shim). `nn/include`·`src`(legacy intrinsic), `simulator/`(ISS) 는 HW 백엔드용.

## 알려진 제약 / 진행 중

- 생성 .cpp 는 vision.cpp arch 로 빌드해야 ggml 수치 검증 가능 — 현재 일부 head op(strided_slice
  offset / group_norm affine / floor_divide 등)은 정적 정보 기반 **best-effort** (TODO 주석).
- ggml 런타임 백엔드(`python output/<Model>.py`)는 ResNet18 / YOLO v8~v12 PyTorch 대비 cosine 1.0 검증됨.
- 향후: best-effort render 정밀화, 모델 지원 확대(Transformer/세그멘테이션).

## 주의

- `tests/test_config.h` 식의 생성물처럼, `output/` 와 `export1/` 는 컴파일러가 매 실행마다 덮어쓰는 산출물이다 — 수동 편집이 보존될 것을 기대하지 말 것.
- 상위 `/mnt/e/7_RISCV/ppp/CLAUDE.md` 는 **다른 프로젝트**(dsppp CMSIS-DSP 템플릿 라이브러리)에 대한 것이다. 이 디렉토리(Compiler)와 혼동하지 말 것.
