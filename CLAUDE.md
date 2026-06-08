# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

PyTorch 모델을  가속기에서 실행 가능한 C/C++ 소스코드로 변환하는 컴파일러. 자세한 사용법·연산자 표·메모리 맵은 `README.md` 참조.

## 명령어

패키지 매니저는 **uv** (`pyproject.toml`). 모든 실행은 `uv run` 으로.

```bash
uv sync                                                    # 의존성 설치 (torch cu124 인덱스 사용)

# 컴파일 (입력 방식 2가지 중 하나 필수 — mutually exclusive)
uv run python compile_to_c.py --torch-model resnet18 --output output/ --name resnet18
uv run python compile_to_c.py --model export1/ResNet.py --output output/ --name resnet18

cd output && make                                          # 크로스 컴파일 → model.elf (RISC-V bare-metal)

export LD_LIBRARY_PATH=/opt/systemc/lib:$LD_LIBRARY_PATH    # 시뮬레이터 실행
../simulator/ISS -I output/resnet18.elf -M -W 0        # -M: monitoring(무결성만), -l 3: 로그레벨

# 테스트
uv run pytest tests/test_numerical.py -v                   # pytest 수치 검증 (codegen+build+sim+PyTorch ref)
uv run pytest tests/test_numerical.py::<name> -v           # 단일 테스트
uv run python test_codegen.py                              # Mock 그래프 codegen 테스트
uv run python test_codegen.py --mini                       # 미니 모델 E2E (빌드+sim)
uv run python test_codegen.py --op-test                    # 개별 연산자 sim 테스트
uv run python test_codegen.py --compile-all                # 전체 모델 ELF 빌드+sim
```

**비자명한 기본값:** `compile_to_c.py --backend` 의 기본값은 **`numcpp`** (NumCpp C++ 생성), `intrinsic` 이 아니다. raw  intrinsic C 를 원하면 `--backend intrinsic` 을 명시해야 한다. README 사용 예시에는 `--backend` 가 빠져 있으니 주의.

## 컴파일 파이프라인 (big picture)

```
PyTorch Module
  └─ [경로 A] parse/TorchParser: torch.jit.trace → Torch Graph(aten) → op_dispatcher →  Graph
  └─ [경로 B] compile_to_c.py:build_graph_from_model: 직접 모듈 분석 → MockGraph
        ↓
 Graph IR  (shared/graph/ — Node=Operation+in/out Tensor, Operation=type+params(numpy)+attrs)
        ↓  optimization/ (Conv-BN fusion 등), quantization/
ScriptWriter (qproc/export/) → export1/*.py  로 Python 재구성 후 재파싱 (디버깅 용이)
        ↓
shared/compile/  CCodeGenerator(intrinsic) | CppCodeGenerator(numcpp)
        ↓
model.c/.h + weights.h/.bin + Makefile + linker script + startup
```

### 핵심 설계 (여러 파일을 읽어야 이해되는 것들)

- **연산자 디스패치:** `shared/base/key_names.py` 의 `OP` enum(200+ op) 이 중심. CCodeGenerator 가 op-type → `_emit_<op>()` 핸들러 dict 로 매핑. PyTorch↔ op 매핑은 `utils/op_register.py` / `2torch_op_map.py`, torch op 변환 로직은 `parse/op_dispatcher.py`(대형) + `parse/torch_op_def.py`.

- ** 실행 모델 (codegen 의 핵심):** `CPU → __split() → Plan{ Shared ∥ Thread } → __join() → CPU`. **DMA 2단계** — Shared Scope(SMU): DDR↔L2 SPM, Thread Scope(SPU): L2↔L1 SPM +  ISA 계산. 두 스코프는 **credit 동기화** (`__load_cr`/`__store_cr`/`__credit_chk`, 비트마스크 `1<<id`) 로 데이터 의존성을 맞춘다. CCodeGenerator 가 이 credit 코드를 자동 생성.

- **메모리 플래닝 (`shared/compile/memory_planner.py`):** L1 SPM 4뱅크(A 입력/B 가중치/C 출력/R 임시, 고정 base 주소)에 맞게 Conv2D·MatMul 타일 크기를 계산하고 2-D DMA 명령을 생성. L2 는 Input/Weight/Output 존으로 분할, 큰 가중치/임시는 DDR. DMA band 효율을 위해 64바이트 정렬.

- **가중치 임베딩 (`weight_exporter.py`):** fp32 → **fp16(uint16)** 변환. 작은 모델은 `weights.h`(static const), 큰 모델은 `weights.bin` + `.incbin` + 링커 스크립트로 DDR 주소에 배치. 전 경로 기본 dtype 은 fp16.

- **하드웨어 intrinsic 구현 (`nn/`):** `nn/include//intrin_level{1,2,3}.h` (Level1 기본 ISA, Level2 편의/credit 매크로, Level3 고수준 BN/ReLU/타일 연산) + `nn/src//` C 구현. intrinsic 백엔드가 생성하는 C 가 이 헤더를 참조.

## 디렉토리 역할

- `compile_to_c.py` — E2E CLI 오케스트레이터 (모델 로드 → 그래프 → codegen → 빌드파일)
- `shared/` — 컴파일러 핵심: `compile/`(codegen·memory·weight), `graph/`(IR), `optimization/`, `quantization/`, `base/`(OP·GlobalMap 상수), `utils/`
- `parse/` — TorchParser (PyTorch→ Graph), `fx/` — FX 기반 변환(보조/일부 미완)
- `qproc/` — ModuleHooker(forward hook 로 activation 캡처·param 동기화), adaquant 양자화, export(ScriptWriter)
- `quantization/`, `utils/` — 양자화 알고리즘 / op 등록·타입 매핑·hw dtype(fp16/bf16/fp8/int8)
- `nn/` —  하드웨어 인터페이스(intrinsic 헤더/구현 + NN 모듈)
- `simulator/` —  ISS (SystemC TLM 2.0, 별도 빌드 바이너리)

## 알려진 제약 / 진행 중

- 시뮬레이터 `-M` (monitoring) 모드는 **코드 무결성만** 검증하고 DMA/연산은 실제 실행하지 않음. 비-`-M` 모드는 credit 동기화 문제로 hang (시뮬레이터 제한). 따라서 수치 검증은 현재 PyTorch 참조값 비교까지만, 실 연산 결과 추출은 미완.
- `x86` 으로 동작하는 컴파일러 완성, 시뮬레이터 실연산 검증, 문서화/코드 정리가 남은 주요 작업.
- 향후: `c_codegen.py` 의 `_emit_*` 를 `nn/modules/` 구조로 리팩터링, 모델 지원 확대(YOLO/Transformer).

## 주의

- `tests/test_config.h` 식의 생성물처럼, `output/` 와 `export1/` 는 컴파일러가 매 실행마다 덮어쓰는 산출물이다 — 수동 편집이 보존될 것을 기대하지 말 것.
- 상위 `/mnt/e/7_RISCV/ppp/CLAUDE.md` 는 **다른 프로젝트**(dsppp CMSIS-DSP 템플릿 라이브러리)에 대한 것이다. 이 디렉토리(Compiler)와 혼동하지 말 것.
