# 갈래2 스케치: ggml 그래프 →  커널 → ELF

> 상태: 부분 검증됨.
> - 갈래1(ggml 검증 lane): ✅ ResNet18/YOLOv8n fp32/fp16/fp8/int8 (`tools/ggml_validate_*.py`).
> - **실행 루프 (tape→ intrinsic C→x86→수치)**: ✅ matmul 슬라이스, ggml과 rel 2e-7 (`tools/x86_dense.py`).
> - **c_codegen full ResNet C → x86 컴파일**: ✅ (intrinsic은 매크로라 `using namespace` 불필요, `/csr.h` 셰임만).
> - **c_codegen full 실행**: ⚠️ x86 credit 가드가 c_codegen 버그 검출(credit을 16 SPU 발행/SPU0만 소비). c_codegen credit 발행 수정 + host weight 하니스가 다음 작업.
> 아래는 전체 설계.

## 0. 무엇을 푸는가

지금 ggml 백엔드는 **검증 lane**이다 (eager per-op, fp16 data/fp32 compute → PyTorch와 대조).
갈래2는 같은 ggml 그래프를 **배포물(ELF)** 로 내린다. 두 하위경로가 있다:

- **2a. per-op RPC 디스패치** — 이미 인프라 존재(`ggml-rpc-server/op_registry.py`가
  `ggml_op enum → pyspike .elf 커널` 매핑). 빠른 bring-up·op별 검증용. fusion 없음·DDR 왕복.
  **배포물이 아니라 검증 도구.**
- **2b. whole-graph AOT** — ggml 그래프 → split/plan/shared/thread/credit 구조의 단일 C 커널
  → `riscv64-unknown-elf-g++` → 단일 ELF. **실제 배포물.** x86으로 호스트 함수검증.

## 1. 재사용 가능한 기존 자산

| 자산 | 위치 | 역할 |
|---|---|---|
| ggml op→커널 매핑 | `host/ggml-rpc-server/op_registry.py` | `GGML_OP_*` → `test/<OP>/n1s16/*.elf` 인벤토리 |
| op numpy 레퍼런스 | `host/ggml-rpc-server/ggml_compute.py` | per-op golden (검증 대조용) |
| firmware 템플릿 | `host/ggml-rpc-server/firmware_templates/*.c.tpl` | x86 참조하는 커널 소스 (mul_mat 등) |
|  intrinsic C | `nn/src//intrin_level{1,2,3}.c`, `nn/include//` | DMA/credit/compute ISA |
| x86 함수 모델 | `host/x86/include` (헤더온리) | 동일 커널이 호스트=riscv 동일결과, ISS 대조 검증됨 |
| AOT codegen | `shared/compile/c_codegen.py` + `memory_planner.py` |  Graph→C (split/plan/shared/thread/credit, 메모리 타일링) |
| ggml 프론트엔드 | `gadget/` (gg_torch) | PyTorch→ggml 그래프, GGUF, bit-exact |

핵심: **AOT codegen(c_codegen)은 이미  Graph에서 ELF를 만든다.** 갈래2의 일은
"ggml 그래프"를 그 codegen의 입력 IR로 연결하는 것 — 새 컴파일러가 아니라 **프론트엔드 어댑터**.

## 2. 권장 경로: 2b (whole-graph AOT)

```
PyTorch ─(gadget)→ ggml 그래프(GGUF) ─[adapter]→  Graph IR ─(c_codegen)→  C
                         │                                              │
                         └→ ggml-CPU 실행 = golden                       └→ riscv g++ → model.elf
                                                                         └→ x86 include → host 함수검증
```

### 2.1 ggml 그래프 →  Graph IR 어댑터 (신규, 핵심 작업)
- ggml 그래프를 walk (`ggml_graph` 노드 순회; gadget의 compute 그래프에서 추출).
- 각 `ggml_tensor`(op) → `shared/graph` 의 `Node`/`Operation`/`Tensor` 로 변환.
  매핑 테이블은 `op_registry.py`의 역방향 + 우리 `OP` enum (이미 ggml_op↔op 지식 존재).
- 가중치는 GGUF tensor →  Graph param (fp16). 레이아웃은 검증 lane에서 확정된 규약 재사용
  (conv weight OIHW native, 입력 NCHW, im2col 열순서는 x86 README의 `(ch,kh,kw)` 가정).
- 결과  Graph를 기존 `optimization/`(Conv-BN fuse 등) → `c_codegen` 에 그대로 투입.

> 대안: ggml→ Graph 대신 **검증 lane의 tape(② lazy 모드의 op 기록)** 를 IR로 써도 된다.
> tape는 이미 (op, attrs, inputs, weights) 의 위상정렬 리스트 → c_codegen 어댑터의 입력으로 자연스럽다.

### 2.2 커널 매핑 (op →  intrinsic)
검증 lane에서 op별 ggml 매핑이 이미 1:1로 정리됨 (`nn/modules/ggml_backend.py`의 `_OPS`).
같은 op들이 c_codegen의 `_emit_*` 핸들러로 간다:
- conv2d → `__im2col_n` + `__mm`
- depthwise → `__dw_conv`
- matmul/dense → `__mm`
- batch_norm → Conv-BN fold(이미 구현) 후 conv에 흡수, 또는 `__batchnorm_aff`
- pool → `__pool_m`/`__pool_a`
- elementwise → `__add_vv`/`__mul_vv`/...
- activation → `__relu`/`__silu`/`__gelu`/... (level3)
- conv_transpose → `__col2im` + `__mm`

### 2.3 검증 사다리 (각 단계가 다음의 golden)
1. **PyTorch fp32** — 기능 정답.
2. **ggml-CPU fp16** (검증 lane, ✅ 동작) — fp16 data/fp32 compute, HW 수치 근사. argmax/rel 대조.
3. **x86** — AOT C를 `-I host/x86/include`로 호스트 컴파일·실행. credit/scope 가드(`exit 70`).
   ggml-CPU(2)와 대조.
4. **riscv ELF on spike+pk / HW** — 동일 C를 riscv로. x86(3)과 bit 대조.

각 단계는 인접 단계만 대조하면 되므로 회귀 위치가 좁혀진다.

## 3. 단계별 작업 (제안)

1. **tape→ Graph 어댑터 프로토타입**: ②의 lazy tape(또는 ggml 그래프)에서 `OP` 노드 그래프 생성. 1개 모델(ResNet18)로 c_codegen 투입 → C 생성 확인.
2. **x86 호스트 실행**: 생성 C를 x86 include로 컴파일·실행 → ggml-CPU와 대조 (단계3).
3. **op 커버리지 정렬**: 검증 lane `_OPS` ↔ c_codegen `_emit_*` ↔ x86 지원 op 의 교집합 표 작성, 갭(conv_transpose 등 long-tail)을 명시.
4. **riscv ELF**: `riscv64-unknown-elf-g++` 빌드 + spike/pk 실행 (단계4). 메모리맵/링커는 기존 compile_to_c 자산.
5. **(선택) 2a RPC 경로**: bring-up 가속용. gadget `rpc:host:port` + `ggml-rpc-server` 그대로.

## 4. 리스크 / 확인 필요

- **레이아웃**: im2col 열순서 `(ch,kh,kw)` ch-outermost (x86 README "확인 필요"). 다채널 conv bit 대조를 ISS로.
- **fp8/int8 양자화**: 검증 lane은 fp16. HW의 fp8(E4M3)/int8 경로는 별도 — quant 정보(qproc)와 codegen 연동 필요.
- **op 갭**: ggml에 1급으로 없는 op(일부 norm/resize/pixelshuffle)은 분해 또는 custom 커널.
- **credit 정확성**: c_codegen이 생성하는 credit 동기화를 x86 credit 가드로 pre-silicon 검출(이미 지원).

## 5. 요약

ELF은 **새로 만들 게 없다** — c_codegen이  Graph→ELF를 한다. 갈래2의 실제 작업은
**(ggml 그래프 또는 검증 lane tape) →  Graph IR 어댑터** 하나와, **검증 사다리 2→3→4 배선**이다.
검증 lane(✅)이 단계2 golden을 공짜로 준다.
