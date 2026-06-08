#  Compiler

PyTorch 모델을 **vision.cpp(ggml) arch 스타일 C++ + GGUF 가중치**로 변환하는 컴파일러.
생성물은 ggml(libggml.so) 위에서 그대로 실행/검증된다.

## 컴파일 파이프라인

```
PyTorch Model
    │  TorchParser (parse/) — torch.jit.trace → Graph IR
    ▼
 Graph IR (shared/graph/) — Conv-BN fusion 등 최적화
    │  ScriptWriter → export1/<Model>.py (op-chain)
    ▼
 VispCodeGenerator (shared/compile/ggml_codegen.py)
    │  op 별 render(node, ctx) 레지스트리 (render_api)
    ▼
 output/<Model>.{cpp,h}  (visp arch 스타일 ggml 그래프)
 output/<Model>.gguf     (state_dict → fp16, 텐서명=state_dict 키)
 output/<Model>.py       (+ggml 실행 진입점)
```

## 사전 요구사항

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) (패키지 매니저)
- `vision.cpp` (git submodule, ggml/GGUF 백엔드) — 아래 설치 참조

## 설치

```bash
git clone <repository-url>
cd Compiler
git submodule update --init --recursive      # vision.cpp (+ depend/llama)
uv sync                                       # 의존성 + g2c CLI 설치
```

## 사용법

### `g2c` — PyTorch 모델 → vision.cpp(ggml) C++ + GGUF

`g2c` 는 pip 콘솔 스크립트(= `shared.compile.pipeline:main`)다. `uv sync`/`uv pip install -e .`
후 사용한다.

```bash
# 모델 표현식 (ultralytics 래퍼는 .model 백본 자동 추출)
uv run g2c --model "ultralytics.YOLO('yolo11n')" --output output/yolo11n

# torchvision 분류 모델 (이름 또는 점경로)
uv run g2c --model resnet18 --output output/resnet18
uv run g2c --model torchvision.models.resnet50 --name r50 --output output/r50

# .pt/.pth 가중치 로드 / 입력 shape 지정
uv run g2c --model "ultralytics.YOLO('yolov8n')" --pth weights.pth \
           --input-shape 1,3,640,640 --output output/

# 옵션: --model(필수, 표현식/이름/.pt) --pth --output --name --input-shape
#   input shape 미지정 시 yolo→(1,3,640,640), 그 외→(1,3,224,224)
```

생성된 `output/<Model>.cpp/.h` 는 vision.cpp 의 `src/visp/arch/` 에 두면 ggml 로 빌드/실행된다.

### ggml 로 바로 실행 (검증용)

```bash
# 생성된 export 를 ggml(libggml.so) 커널로 직접 실행 (PyTorch 참조 없이 forward)
uv run python output/yolo11n/DetectionModel.py     # [ggml] output: (1, ...)
```

### 테스트 / 예제 스크립트

```bash
# 과거 진입점(테스트용 shim). 기본 resnet18, g2c 와 동일 파이프라인.
uv run python test/compile_to_c.py --model yolov8n --output output/yolov8n
```

##  실행 모델

```
CPU → __split() → Plan { Shared ∥ Thread } → __join() → CPU

DMA 2단계:
  Shared Scope (SMU): DDR ↔ L2 SPM
  Thread Scope (SPU): L2 SPM ↔ L1 SPM +  ISA 계산

Credit 동기화:
  Shared → Thread: __load_cr(DDR→L2, credit)  →  Thread가 credit 대기 후 L2→L1 로드
  Thread → Shared: __store_cr(L1→L2, credit)  →  Shared가 __credit_chk()로 대기
```

## 메모리 맵

### DDR (External Memory)

| 영역 | 주소 | 용도 |
|------|------|------|
| Input | `0x80000000` | 입력 데이터 |
| Output | `0x90000000` | 추론 결과 |
| Weight | `0xA0000000` | 모델 가중치 |
| Temp | `0xB0000000` | 중간 버퍼 |

### L1 SPM (Scratchpad Memory)

| Bank | 주소 | 크기 | 용도 |
|------|------|------|------|
| A | `0x00000` | 128KB | 입력 |
| B | `0x20000` | 64KB | 가중치 |
| C | `0x30000` | 128KB | 출력 |
| R | `0x50000` | 64KB | 임시 |

## 디렉토리 구조

```
Compiler/
├── pyproject.toml               # g2c 콘솔 스크립트 등록 ([project.scripts])
├── resnet.py                    # 예제: ResNet-18 파싱 → export 생성
├── test/
│   └── compile_to_c.py          # 과거 진입점(테스트 shim) → pipeline 재사용
│
├── shared/                      # 컴파일러 핵심 인프라
│   ├── compile/
│   │   ├── pipeline.py          # g2c 구현: 모델로드→parse→export→cpp/h+gguf
│   │   ├── ggml_codegen.py      # VispCodeGenerator (visp/ggml arch C++ 생성)
│   │   ├── render_api.py        # op render 레지스트리/RenderContext/헬퍼
│   │   ├── memory_planner.py    # (legacy intrinsic) L1/L2/DDR 타일링
│   │   └── weight_exporter.py   # 가중치 fp16 변환
│   ├── graph/                   # Graph IR (노드, 텐서, 연산)
│   ├── optimization/            # 그래프 최적화 (Conv-BN fusion 등)
│   ├── quantization/            # 양자화 설정 및 연산
│   ├── base/                    # 상수 정의 (OP enum, 디버그 레벨)
│   └── utils/                   # 공통 유틸리티
│
├── nn/                          # NN 모듈 + ggml 백엔드
│   ├── modules/
│   │   ├── <op>.py              # op별 PyTorch 모듈 + render(node,ctx) (ggml emit)
│   │   ├── head_render.py       # detection-head/anchor/DFL/meta op render
│   │   ├── vision_ops_render.py # 활성화/정규화/math op render
│   │   └── ggml_backend.py      # set_backend('ggml') 런타임 실행 백엔드
│   └── include//, src//         # (legacy) intrinsic 헤더/구현
│
├── parse/                       # TorchParser: PyTorch → Graph IR
├── qproc/                       # ModuleHooker, adaquant, export(ScriptWriter)
├── quantization/, utils/        # 양자화 알고리즘 / op 등록·타입 매핑
└── vision.cpp/                  # (submodule) ggml/GGUF 추론 라이브러리
```

## ggml operator 커버리지

codegen render + 런타임 백엔드가 사용하는 ggml op(추론 전용; 학습 op 제외): conv_2d,
mul_mat, pool_2d, add/sub/mul/div, relu/silu/sigmoid/gelu/tanh/leaky_relu/clamp,
soft_max/log_softmax, norm/group_norm, concat/cont/transpose/permute/reshape/view,
repeat/upscale/arange/top_k/argmax/get_rows, sqrt/sum/mean/scale. detection head 의
meta op(strided_slice/gather/index/max/meshgrid/stack/full/const/floor_divide 등)도
실제 ggml 그래프 op 으로 emit한다(passthrough 없음).

## 프로젝트 상태

### 완료

- PyTorch 모델 파싱 → Graph IR → visp/ggml arch C++ + GGUF 생성 (`g2c`)
- ResNet18 / YOLO v8~v12 codegen: detection head 포함 **0 unhandled op**
- ggml(libggml.so) 런타임 백엔드 검증: ResNet18 / YOLO v8~v12 vs PyTorch cosine 1.0

### 진행 중 / 향후

- 생성 .cpp 를 vision.cpp arch 로 빌드해 ggml 수치 검증 (현재 일부 head op 은 best-effort)
- group_norm affine / strided_slice offset 등 best-effort render 정밀화
- 추가 모델 지원 확대 (Transformer/세그멘테이션)

## 라이선스

MIT License - Copyright (c) 2025 Sudo42b
