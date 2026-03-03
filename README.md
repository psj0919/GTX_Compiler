# GTX Compiler

PyTorch 모델을 GTX NPU 하드웨어에서 실행 가능한 C 소스코드로 변환하는 컴파일러.

## 컴파일 파이프라인

```
PyTorch Model
    │
    ▼
TorchParser (parse/)
    │  FX 기반 그래프 추출
    ▼
GTX Graph (gtx_shared/gtx_graph/)
    │  최적화 (Conv-BN fusion 등)
    ▼
CCodeGenerator (gtx_shared/compile/c_codegen.py)
    │
    ├──→ model.c / model.h       GTX intrinsic 추론 코드
    ├──→ weights.bin              fp16 바이너리 가중치
    ├──→ weight_map.h             DDR 절대 주소 매핑
    ├──→ data_embed.S             .incbin으로 ELF에 데이터 임베딩
    ├──→ crt0.S / linker.ld       Bare-metal 스타트업 + 메모리 맵
    └──→ Makefile                 RISC-V 크로스 컴파일 빌드
            │
            ▼
    riscv64-unknown-elf-gcc
            │
            ▼
      model.elf  →  GTX_ISS 시뮬레이터
```

## 사전 요구사항

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) (패키지 매니저)
- RISC-V 크로스 컴파일러: `riscv64-unknown-elf-gcc` (RV64GC)
- SystemC 2.3+ (`/opt/systemc/lib` — 시뮬레이터 실행 시)

## 설치

```bash
git clone <repository-url>
cd GTX_Compiler
uv sync
```

## 사용법

### 1. PyTorch 모델 → C 코드 변환

```bash
# torchvision 모델 직접 변환
uv run python compile_to_c.py --torch-model resnet18 --output output/ --name resnet18

# 사전 추출된 IR 파일 사용
uv run python resnet.py                    # IR 생성 (export1/ResNet.py)
uv run python compile_to_c.py --model export1/ResNet.py --output output/ --name resnet18

# 옵션
#   --input-shape 1,3,224,224    입력 텐서 shape (기본값: 1,3,224,224)
#   --nest-id 0                  타겟 NEST ID (0-3)
#   --spu-id 0                   타겟 SPU ID (0-3)
#   --no-makefile                빌드 파일 생성 생략
```

### 2. 크로스 컴파일

```bash
cd output && make
# 생성: model.elf (RISC-V bare-metal 바이너리)
```

### 3. 시뮬레이터 실행

```bash
export LD_LIBRARY_PATH=/opt/systemc/lib:$LD_LIBRARY_PATH
../simulator/GTX_ISS -I output/resnet18.elf -M -W 0
#   -I    입력 ELF 파일
#   -M    모니터링 모드 (코드 무결성 검증)
#   -l 3  로그 레벨 (선택)
```

### 4. 테스트

```bash
# pytest 기반 수치 검증 (코드 생성 + ELF 빌드 + 시뮬레이터 무결성 + PyTorch 참조값)
uv run pytest tests/test_numerical.py -v

# 코드 생성 테스트 (Mock 그래프)
uv run python test_codegen.py

# 미니 모델 E2E 테스트 (크로스 컴파일 + 시뮬레이터)
uv run python test_codegen.py --mini

# 개별 연산자 빌드 + 시뮬레이터 테스트
uv run python test_codegen.py --op-test

# 전체 모델 ELF 빌드 + 시뮬레이터 테스트
uv run python test_codegen.py --compile-all
```

## GTX 실행 모델

```
CPU → __split() → Plan { Shared ∥ Thread } → __join() → CPU

DMA 2단계:
  Shared Scope (SMU): DDR ↔ L2 SPM
  Thread Scope (SPU): L2 SPM ↔ L1 SPM + GTX ISA 계산

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

## 지원 연산

| 카테고리 | Op | GTX Intrinsic |
|----------|-----|---------------|
| **Convolution** | Conv2d | `__im2col_n` + `__mm` |
| | ConvTranspose2d | `__col2im` + `__mm` |
| | DepthwiseConv2d | `__dw_conv` |
| **Normalization** | BatchNorm | `__batchnorm_aff` |
| | LayerNorm, GroupNorm, InstanceNorm | `__layernorm` / `__groupnorm` |
| **Activation** | ReLU / ReLU6 | `__relu` / `__relu6` |
| | LeakyReLU / PReLU | `__lrelu` |
| | GELU / Mish | `__gelu` / `__mish` |
| | Sigmoid / Tanh | `__sigm` / `__tanh` |
| | Hardswish / Hardsigmoid | `__hswish` / `__hsigm` |
| | Softmax | `__esum` + `__softmax` |
| **Pooling** | MaxPool2d | `__pool_m` |
| | AvgPool2d / AdaptiveAvgPool2d | `__pool_a` |
| **Linear** | Dense (Linear) | `__mm` |
| **Elementwise** | Add / Mul / Div | `__add_vv` / `__mul_vv` / `__div_vv` |
| **Reshape** | Flatten / Reshape / Permute | No-op (메모리 재해석) |
| **기타** | Concat, Pad, Resize, PixelShuffle 등 | 메모리 조작 기반 |

## 디렉토리 구조

```
GTX_Compiler/
├── compile_to_c.py              # E2E 컴파일 파이프라인 CLI
├── resnet.py                    # 예제: ResNet-18 파싱 → IR 생성
├── test_codegen.py              # C 코드 생성 + 빌드 + 시뮬레이터 테스트
├── tests/
│   └── test_numerical.py        # pytest 수치 검증 테스트
│
├── gtx_shared/                  # 컴파일러 핵심 인프라
│   ├── compile/
│   │   ├── c_codegen.py         # C 코드 생성기 (CCodeGenerator)
│   │   ├── memory_planner.py    # L1/L2/DDR 메모리 할당 + 타일링
│   │   ├── weight_exporter.py   # 가중치 fp16 변환 + 바이너리 내보내기
│   │   ├── gtx_compiler.py      # GTX 컴파일러 API
│   │   └── op_test_runner.py    # 개별 연산자 시뮬레이터 테스트 러너
│   ├── gtx_graph/               # GTX Graph IR (노드, 텐서, 연산)
│   ├── optimization/            # 그래프 최적화 (Conv-BN fusion 등)
│   ├── quantization/            # 양자화 설정 및 연산
│   ├── base/                    # 상수 정의 (GTX_OP, 디버그 레벨)
│   └── utils/                   # 공통 유틸리티
│
├── nn/                          # GTX 하드웨어 인터페이스
│   ├── include/gtx/             # GTX intrinsic 헤더
│   │   ├── intrin_level1.h      # Level 1: 기본 ISA (DMA, 연산)
│   │   ├── intrin_level2.h      # Level 2: 편의 매크로
│   │   ├── intrin_level3.h      # Level 3: 고수준 연산 (BN, ReLU 등)
│   │   ├── gtx_csr.h            # CSR 레지스터 정의
│   │   └── gtx_utils.h          # 유틸리티 함수
│   ├── src/gtx/                 # GTX intrinsic C 구현체
│   └── modules/                 # NN 레이어 PyTorch 모듈 구현
│
├── parse/                       # TorchParser: PyTorch → GTX Graph
├── fx/                          # FX 기반 그래프 변환 유틸리티
├── qproc/                       # 양자화 프로세서, ModuleHooker, export
├── quantization/                # 양자화 알고리즘
├── gtx_utils/                   # 유틸리티 (op 등록, 타입 매핑 등)
│
└── simulator/                   # GTX ISS (SystemC TLM 2.0 기반)
    ├── GTX_ISS                  # 시뮬레이터 바이너리
    ├── src/                     # RISC-V CPU + GTX 확장 유닛 (SPU, NSU, TMU)
    └── inc/                     # 시뮬레이터 헤더
```

## 프로젝트 상태

### 완료

- PyTorch 모델 파싱 → GTX Graph IR 변환
- GTX intrinsic 기반 C 코드 생성 (18+ 연산자)
- L1/L2/DDR 메모리 타일링 및 DMA 코드 자동 생성
- Credit 동기화 코드 자동 생성 (Shared ↔ Thread)
- NEST/SPU ID 설정 가능화 (매크로 기반)
- ELF 데이터 임베딩 (`.incbin` + 링커 스크립트)
- 시뮬레이터 통합 (PTY 기반 실행, 메모리 덤프)
- pytest 기반 수치 검증 테스트 (24 tests passing)

### 진행 중

- 시뮬레이터 비-monitoring 모드에서의 실제 연산 결과 추출
  - 현재 `-M` 모드: 코드 무결성만 검증, DMA/연산 미실행
  - 비-M 모드: credit 동기화 문제로 hang (시뮬레이터 제한)

### 향후 계획

- 시뮬레이터 수치 검증 활성화 (output 메모리 덤프 vs PyTorch 참조값 비교)
- `c_codegen.py`의 `_emit_*` 메서드를 `nn/modules/` 구조로 리팩터링
- 추가 모델 지원 확대 (YOLO, Transformer 등)

## 라이선스

MIT License - Copyright (c) 2025 Sudo42b
