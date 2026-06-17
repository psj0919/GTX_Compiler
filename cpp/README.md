# GTX Compiler — C++ / libtorch 포트

Python `g2c` 컴파일러를 **Modern C++ + libtorch** 로 1:1 포팅하는 sub-project.
브랜치 `feat/cpp-libtorch-compiler`. 먼저 **파서**(PyTorch → Graph IR)부터 옮기고,
이후 codegen/GGUF, 그 다음 최적화로 확장한다.

## 설계 결론: 트레이싱 경로

`torch::jit::trace` 는 C++ 함수/Module 을 트레이싱한다. 그러나 resnet18·yolo 처럼
**Python 으로 정의된 nn.Module 은 C++ 에서 직접 트레이스할 수 없다**. 따라서 경로는:

```
[Python]  torch.jit.trace(model, x).save("m.pt")     # 트레이싱 = 언어 무관(eager)
            └ cpp/tools/export_traced.py
                          ↓  (.pt = TorchScript)
[C++]     torch::jit::load("m.pt")
            → freeze_module()        # 파라미터를 그래프 상수로 인라인(BN 보존)
            → optimize_graph_19()    # JIT 패스 재현
            → torch::jit::Graph 순회 → Graph IR
```

즉 Python `TorchParser` 가 **트레이스 이후** 하는 일(`_convert_graph`/`op_dispatcher`/
`_load_data`)을 C++ 로 옮기는 것이 본질이다. 트레이스 자체는 PyTorch eager 라 Python 에
두는 것이 정확하고(동일 그래프 보장), 파싱·IR·코드젠이 C++ 작업 대상이다.

### freeze 주의 (parser parity 핵심)

- 상위 API `torch::jit::freeze()` 는 `OptimizeFrozenGraph`(conv-BN fold 등)까지 수행 →
  BatchNorm 이 사라진다.
- Python 파서는 `freeze_graph_wo_opt`(= `_freeze_module`, 최적화 없음)로 **BN 을 별도
  노드로 유지**한다(vision.cpp 용 fold 는 이후 자체 패스에서).
- 따라서 C++ 도 `torch::jit::freeze_module()` 을 직접 호출한다.

### optimize_graph 패스 (torch≥1.9, `_optimize_graph_19` 재현)

`Inline → InlineForkWait → LowerAllTuples → ConstantPropagation → DCE →
CanonicalizeOps → PeepholeOptimize(addmm) → FuseLinear(addmm→linear) → Canonicalize`

(Python onnx 전용 `_split_tensor_list_constants` / `onnx_remove_print` 는 trace 그래프에
영향이 없어 제외. ScriptModule 경로의 `is_jit_graph=True` 분기 차이는 파서 마일스톤에서 반영.)

## 빌드

pip `torch`(=`.venv`) 가 libtorch 헤더/lib/CMake config 를 모두 포함 → 별도 libtorch 설치 불필요.

```bash
# 의존 환경: uv sync 로 .venv 준비됨 (torch 2.6.0+cu124)
bash cpp/build.sh                       # → cpp/build/gtxc-parse
```

빌드 환경 메모:
- pip torch 는 **pre-cxx11 ABI** (`_GLIBCXX_USE_CXX11_ABI=0`). `find_package(Torch)` 가
  `${TORCH_CXX_FLAGS}` 로 자동 주입.
- CUDA 12.x 는 `nvToolsExt` 를 header-only(nvtx3)로 대체 → torch CMake 가 참조하는
  `CUDA::nvToolsExt` IMPORTED 타깃 부재로 generate 가 깨진다. `find_package(Torch)` 전에
  빈 INTERFACE 스텁을 정의해 우회(파서는 CPU 만 사용).
- RPATH 로 torch/lib 를 박아 실행 시 `LD_LIBRARY_PATH` 불필요.

## 사용

```bash
bash cpp/build.sh                          # 빌드

# Python IR(ground truth) vs C++ IR 1:1 parity 한 방에:
cpp/run_parity.sh resnet18                 # → ✅ 구조·shape·config/attr/param 완전 일치
cpp/run_parity.sh resnet50 1,3,224,224

# GGUF 출력(fp16) + 검증:
cpp/build/gtxc-parse cpp/assets/resnet18.pt --graph-name resnet18 --gguf cpp/assets/resnet18.gguf --out /dev/null
.venv/bin/python cpp/tools/verify_gguf.py resnet18 cpp/assets/resnet18.gguf   # → ✅ 102 텐서 일치

# E2E 컴파일: PyTorch → vision.cpp arch C++(.cpp/.h) + folded GGUF (Conv-BN fold):
cpp/build/gtxc-parse cpp/assets/resnet18.pt --graph-name resnet18 --compile cpp/out/resnet18
#   → cpp/out/resnet18/resnet18.{cpp,h,gguf}  (생성 .cpp 는 output/resnet18/ResNet.cpp 와 op 단위 일치)

# 개별 단계:
.venv/bin/python cpp/tools/export_traced.py --model resnet18 --out cpp/assets/resnet18   # trace→.pt
.venv/bin/python cpp/tools/dump_ir.py       --model resnet18 --out cpp/assets/resnet18.ir.py.json
cpp/build/gtxc-parse cpp/assets/resnet18.pt --graph-name resnet18 --out cpp/assets/resnet18.ir.cpp.json
.venv/bin/python cpp/tools/compare_ir.py cpp/assets/resnet18.ir.py.json cpp/assets/resnet18.ir.cpp.json
```

### parity 결과 (현재)

분류·RNN 은 1:1 완전 일치, yolo 는 백본 완전 파싱(헤드는 OptPass 포팅 필요).

| model | nodes | params | 결과 |
|-------|------:|------:|------|
| resnet18/34/50 | 71/127/177 | 102/182/267 | ✅ 구조·shape·config/attr/param 완전 일치 |
| lstm·gru (단/양방향·멀티레이어) | 5~6 | 4~8 | ✅ 완전 일치 |
| yolo11n | 334 | — | 백본 완전 파싱(UNPARSED 3/334, shape 296/345). 헤드 best-effort |

```bash
cpp/run_parity.sh resnet18            # 분류/RNN: 완전 일치
cpp/run_parity.sh lstm_bi 1,8,16
# yolo (노드 정렬 불가 → 커버리지 리포트):
.venv/bin/python cpp/tools/export_traced.py --model yolo11n --out cpp/assets/yolo11n
cpp/build/gtxc-parse cpp/assets/yolo11n.pt --graph-name yolo11n --input-shape 1,3,640,640 --out cpp/assets/yolo11n.ir.cpp.json
.venv/bin/python cpp/tools/coverage.py cpp/assets/yolo11n.ir.cpp.json
```

- op_type 시퀀스, in/out 텐서 shape, params(이름+shape), configs, attrs 가 Python TorchParser 와 동일.
- **활성 텐서의 value 이름**(ret.1 / 3217 …)만 다르다 — Python 저수준 트레이서 vs 저장된
  trace 의 debugName 차이로, trace-export 방식의 본질적 한계(파라미터 이름·구조·수치는 일치).
- shape 는 C++ 파서가 per-op 추론(Python 은 트레이서가 기록) — 결과는 동일.

### op 핸들러 (현재 지원)

- **전용 핸들러**: conv2d / depthwise_conv2d / batch_norm / relu / maxpool / adaptive_avg_pool2d /
  elemwise_add / flatten / dense(linear) / shape(size) / zeros / input / return.
- **제네릭 default**(op_dispatcher `auto_infer_op` 포트): schema 인자→config, free param 바인딩,
  dtype/device/bool/inf 변환. lstm/gru(단·양방향·멀티), silu/sigmoid 등 activation, cat/chunk/
  split/upsample 등이 이 경로 + shape 추론으로 처리된다.
- **구조 처리**: prim::ListUnpack 을 producer(chunk/split)에 병합(OptPass unpack_ListUnpack_op 대응),
  prim::GetAttr 체인 → state_dict full name, prim::Constant Tensor[] fold 회피(preserveParameters).

### yolo 헤드 — 트레이스 메커니즘 경계 (본질적)

yolo 는 전 노드 파싱(UNPARSED 0)되고 백본은 Python 과 정확 일치한다. 검출 헤드의 **노드 구조
차이**는 OptPass 미포팅이 아니라 **트레이스 메커니즘 차이**에서 온다:

- 이 포트는 저장된 `torch.jit.trace`(.pt)를 소비한다. trace 는 고정 입력으로 평가하므로 헤드의
  **anchor 생성(arange/meshgrid/stack)과 입력-shape 파생 슬라이싱을 상수로 fold** 한다 →
  C++ 그래프에는 slice/arange/meshgrid/stack 노드가 아예 없다(전부 baked 상수).
- Python TorchParser 는 Python 전용 저수준 트레이서(`_get_trace_graph`)를 써서 이들을 live op
  로 유지한다(strided_slice 23 / arange 6 / meshgrid 3 / stack 3 …). 이 트레이서는 libtorch
  C++ 에서 접근 불가 → 활성 텐서 value 이름 차이와 **동일한 경계**다.
- 단, trace 의 상수 fold 결과는 Python 파이프라인의 `const_fold.py`(anchor/stride→baked) 출력과
  **동등**하다 — 즉 codegen 이 쓰는 최종(fold 후) 형태로는 정합한다.

slice/select→strided_slice OptPass 핸들러와 헤드 op 네이밍은 포팅되어 있어, 상수 fold 되지
않는 슬라이스/헤드 op 가 있는 모델에서는 그대로 동작한다.

## 디렉토리

```
cpp/
  CMakeLists.txt        libtorch 링크 + gtxc-parse 빌드
  build.sh              torch cmake prefix 자동 주입 빌드 helper
  include/              (예정) Graph IR / parser 헤더
  src/
    main.cpp                 CLI: --out(IR JSON) / --gguf / --compile
    ir/graph_dump.cpp        Graph IR → JSON
    parse/parser.cpp         torch::jit::Graph → Graph IR (op_dispatcher 포트, Parser 클래스)
    parse/parse_internal.hpp do_parse/write_gguf_core 내부 공유 선언
    compile/gguf.cpp         GGUF(fp16) writer
    compile/codegen.cpp      Conv-BN fold + vision.cpp arch C++ codegen
  tools/  export_traced.py(trace→.pt) dump_ir.py(ground truth) compare_ir.py
          coverage.py verify_gguf.py _modelkit.py
  assets/ out/            생성물 (gitignore)
```

## 로드맵 (파서 우선)

1. [x] 툴체인 증명: libtorch CMake 빌드 + traced 그래프 순회
2. [x] Graph IR(Node/Operation/Tensor/Graph) 포팅 — `include/gtxc/ir.hpp`, JSON 덤프
3. [x] TorchParser + op_dispatcher(resnet op셋) — `src/parse/parser.cpp`, per-op shape, OIHW→OHWI
4. [x] parity 하네스 — resnet18/34/50 완전 일치
5. [x] op 확장 — RNN(lstm/gru 전 변형) 완전 일치, 제네릭 default 핸들러(auto_infer_op)
6. [x] yolo — **100% 파싱(UNPARSED 0/332, shape 336/343)**, slice/select→strided_slice OptPass
       포팅, 헤드 op clean 네이밍(resize/stack/arange/meshgrid/…), 스칼라정수(NumToTensor/Int/
       mul) 추적, ListUnpack 병합. 백본·다수 op Python 과 정확 일치. 헤드 차이는 트레이스 경계(아래)
7. [x] **GGUF writer**(fp16) — `write_gguf`, 텐서명=state_dict키. resnet/rnn/yolo 전부
       state_dict 와 값 일치 검증(`cpp/tools/verify_gguf.py`)
8. [x] **codegen** — `compile_model`: parse→**Conv-BN fold**→vision.cpp arch C++(.cpp/.h)+folded
       GGUF. resnet18/34/50 생성물이 Python 파이프라인과 op 단위 일치. **yolo11n 전체 forward
       완전 렌더(254 ggml 호출, TODO(head) 0)**: 백본(conv/silu/chunk view/concat/add/maxpool/
       resize/depthwise) + 검출 헤드 DFL·C2PSA 어텐션 디코드(reshape→qkv split→permute→mul_mat
       →softmax→permute→mul_mat→reshape). output/v10d 패턴과 동일.
       (남음: matmul operand order/attention scale 등 수치 best-effort 정밀화, vision.cpp 직접 빌드 verify)

## parity 정책

`1:1 매핑`은 **관찰 가능한 IR 표면**(op type, params 이름·shape·dtype·값, configs/attrs)
수준에서의 동등성을 의미한다. Python 의 `IrAttr` typed-memory·use-def 추적 같은 reflection
머신은 최적화 패스 전용이라, 파서 단계 C++ 에서는 variant 기반으로 단순화한다(동작 동일,
구현 idiomatic). 최적화 패스를 옮길 때 use-def 를 도입한다.
