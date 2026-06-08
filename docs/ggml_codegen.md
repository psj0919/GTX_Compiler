# ggml / vision.cpp 백엔드 codegen 설계

## 목적

`compile_to_c.py` 의 코드 생성을 **naive C / NumCpp / intrinsic** 이 아니라,
`vision.cpp` (ggml 기반 C++ 추론 라이브러리)의 **arch 모듈 컨벤션**에 맞는
C++ 소스로 바꾼다. 즉 PyTorch 모델 → `vision.cpp/src/visp/arch/<model>.cpp` 형태로 생성하여
ggml 위에서 그대로 실행/검증 가능하게 한다.

```
resnet.py            → export1/ResNet.py   (nn.Module op-chain = Graph IR 소스)
compile_to_c.py      → output/<model>.cpp  (visp arch 스타일, ggml 그래프 빌드)
                     → output/<model>.h    (params 구조체 + forward 선언)
                     → output/<model>.gguf (state_dict 명명 가중치 + arch 메타데이터)
```

## 참고한 vision.cpp 컨벤션 (arch/esrgan.cpp, arch/dino.cpp, nn.h)

- forward 함수 시그니처: `tensor <arch>_<fn>(model_ref m, tensor x, <arch>_params const& p)`
- 가중치 접근: `m["conv1"]`(named) / `m[i]`(sequential) → 해당 서브모듈의 GGUF 텐서.
  `conv_2d(m["conv1"], x, ...)` 는 GGUF 의 `conv1.weight` / `conv1.bias` 를 사용.
- 레이어 빌딩블록(nn.h): `conv_2d(m, x, stride, pad)`, `conv_2d_depthwise`,
  `batch_norm_2d(m, x)`, `linear(m, x)`, `layer_norm`, `attention`, …
- 기본 ggml op: `ggml_relu`, `ggml_add`, `ggml_mul`, `ggml_gelu`, `ggml_leaky_relu`,
  `ggml_scale_inplace`, `interpolate`, `concat`, `ggml_reshape_*` …
- 레이아웃: 입력 경계에서 `cwhn_to_contiguous_2d(m, x)` / `contiguous_2d_to_cwhn(m, x)`.
- 종료: `compute_graph_output(m, x, "result")`.
- 메타: `<arch>_params`, `<arch>_detect_params(model_file const&)`(GGUF 메타 읽기),
  `<arch>_estimate_graph_size(params)`.

## Graph IR → visp op 매핑 표

| Graph OP (`shared/base/key_names.OP`) | visp / ggml 호출 | 가중치 |
|---|---|---|
| `INPUT` (`"input"`) | 그래프 입력 텐서(파라미터 x) | – |
| `CONV2D` (`"conv2d"`) | `conv_2d(m["<k>"], x, stride, pad)` | `<k>.weight`, `<k>.bias?` |
| `BATCH_NORM` (`"batch_norm"`) | `batch_norm_2d(m["<k>"], x)` | `<k>.{weight,bias,running_mean,running_var}` |
| `RELU` (`"relu"`) | `ggml_relu(m, x)` (inplace 시 `_inplace`) | – |
| `ADD` (`"elemwise_add"`) | `ggml_add(m, a, b)` | – |
| `MAX_POOL` (`"maxpool"`) | `ggml_pool_2d(m, x, GGML_OP_POOL_MAX, k, k, s, s, p, p)` | – |
| `AVG_POOL` / adaptive avg | `ggml_pool_2d(..., GGML_OP_POOL_AVG, ...)` | – |
| `FLATTEN` (`"flatten"`) | `ggml_reshape_2d(m, x, n, 1)` | – |
| `DENSE` (`"dense"`) | `linear(m["<k>"], x)` | `<k>.weight`, `<k>.bias?` |
| `MATMUL` (`"matmul"`) | `ggml_mul_mat(m, a, b)` | – |

> stride/padding 등은 `node.op.attrs` (예: `attrs["stride"] = [2,2]`)에서 가져온다.
> ResNet 컨볼루션은 대칭이라 `stride[0]`, `padding[0]` 만 사용한다(비대칭은 TODO).

## 가중치 명명 (GGUF)

vision.cpp 의 `m["conv1"]` ↔ GGUF 텐서 `conv1.weight` 가 일치해야 한다.
PyTorch state_dict 경로(`layer1.0.conv1.weight` 등)를 그대로 GGUF 텐서명으로 쓴다.
codegen 은 노드별 **weight key**(점 경로)를 산출하고, `.cpp` 의 `m["<key>"]` 와
`.gguf` 의 `<key>.<param>` 텐서를 같은 key 로 맞춘다. (현재 스캐폴드: `node.name`
기반 best-effort 산출 + manifest 출력. 정밀 state_dict 매핑은 weight_exporter 연동 TODO.)

dtype: vision.cpp 기본 F16. fp32 weight → fp16 변환 후 GGUF 기록(`weight_exporter` 재사용).

## 구현 범위 (이번 스캐폴드)

### 아키텍처: op 별 render(node, ctx) (분산형 레지스트리)

op 의 C++ 렌더링은 각 **`nn/modules/<op>.py`** 의 module-level `render(node, ctx)` 함수에
두고 `@register_render(OP.X)` 로 등록한다. 새 op 지원 = 그 op 의 모듈 파일에 render 추가.

- **`shared/compile/render_api.py`** — 렌더 프레임워크(leaf, nn 의존 없음):
  `RENDERERS` 레지스트리, `register_render(*op_types)` 데코레이터, `RenderContext`,
  헬퍼(`weight_key`/`attr`/`scalarize`). `RenderContext` API: `inp(node,i)` /
  `attr(node,k,d)` / `scalar(v)` / `weight(node,[suffix])`→`m["key"]` / `out(node,expr)`.
- **`shared/compile/ggml_codegen.py`** 의 `VispCodeGenerator` — thin dispatcher:
  `import nn.modules` 로 render 등록을 트리거하고 `RENDERERS` 로 디스패치. 구조적 op
  (`INPUT`=x 바인딩, `RETURN`=무시, `FLATTEN`=reshape)만 내장. `.cpp/.h/.weights.txt`
  3종 emit, `compute_graph_output` 으로 종료.
- **render 보유 op (9):** `CONV2D`(conv.py) `BATCH_NORM`(batch_norm.py) `RELU`(relu.py)
  `ADD`(add.py) `MAX_POOL`(maxpool.py) `AVG_POOL`(avgpool.py)
  `ADAPTIVEAVGPOOL2D`(adaptive_avg_pool.py) `DENSE`(linear.py) `MATMUL`(matmul.py).
  CONV2D/BATCH_NORM/RELU/ADD 는 완전 구현, 나머지는 파라미터/레이아웃 검증 TODO 주석 포함.
- `compile_to_c.py` 는 TorchParser 그래프를 codegen 에 직접 사용(텐서 identity 공유로
  데이터 흐름·weight key 정확). 기존 `numcpp`/`intrinsic` 은 삭제된 codegen 참조 레거시.

## GGUF 직렬화 + 양자화 파라미터 (확인 결과)

**현재 상태 (gap 확인됨):** GGUF 변환 시 양자화 파라미터(qparams: scale/zero-point)가
**전혀 직렬화되지 않는다.**
- `shared/compile/weight_exporter.py`(`WeightExporter`): **fp16 변환만** — C 헤더/바이너리/
  주소맵 생성. scale·zero-point 없음.
- `tools/ggml_weight_binder.py`: export 주석의 모듈 경로를 `node_path_to_key` 로 파싱해
  (`...Conv2d[conv1]/ret.5` → `layer1.0.conv1`) torch `state_dict` 에서 **이름 기반으로
  fp32 weight 만 바인딩**. quant param 직렬화 없음. (소비처였던 in-process ggml validation
  lane 은 삭제됨.)
- `gguf` 파이썬 라이브러리 미설치 — 실제 `.gguf` writer 부재.
- qparams 자체는 `nn/modules/module_template.py` quant forward 에서 매 forward 마다
  계산되는 **휘발성 fake-quant 값**이었고(아래 수정 전엔 적용조차 안 됨), 어디에도 영속화되지 않음.
  실제 scale 은 `shared/quantization/{base_quantizer,fix_pos_adjust,quant_config_imp}.py` 에서
  calibration 시 산출되어 quantizer/node 에 부착된다.

**필요 작업 (deploy 시 양자화 재현):** GGUF 에 weight + qparams 를 함께 기록해야 한다.
- 텐서 명명: `<key>.weight`(양자화 or fp16) + `<key>.scale`(+ 비대칭이면 `.zero_point`),
  또는 ggml 네이티브 블록양자(Q8_0/Q4_K 등 — 블록당 scale 내장) 포맷 사용.
- calibration 으로 산출된 per-tensor/per-channel scale 을 캡처하여 GGUF 텐서/메타로 기록.
- weight key 는 codegen 의 `m["<key>"]` 와 동일해야 하므로 `ggml_weight_binder.node_path_to_key`
  방식(export 주석 파싱)을 codegen·GGUF 양쪽의 단일 소스로 채택.

## 남은 작업 (TODO)

1. **weight-key 단일화:** codegen 이 현재 MockGraph 노드명(`module_N_TYPE`)을 쓰는데,
   `ggml_weight_binder.node_path_to_key`(→ `layer1.0.conv1`)로 교체해 GGUF 텐서명과 일치.
2. **GGUF writer + qparams 직렬화** (위 절): `gguf` lib 도입 또는 vision.cpp gguf 툴 사용.
3. 레이아웃(CWHN/WHCN) 경계 변환 자동 삽입.
4. BasicBlock/Sequential 등 계층 구조를 `m["layer1"][0]["conv1"]` 식 중첩 model_ref 로 재현.
5. `vision.cpp` 측 arch 등록(`<arch>_detect_params`, c-api, CMake).
6. residual add 의 두 입력(피연산자) 추적 — 현재 in_tensors 순서 의존.
7. MockGraph in/out_tensor 가 객체 identity 로 연결되지 않아 codegen 데이터 흐름이
   `/*?*/ x` 로 끊김 — `in_nodes` 기반 연결로 교체 필요.

## quant forward 수정 (2026-06-08)

`nn/modules/module_template.py` NN_MODULE quant forward 의 fake-quant 가 **no-op** 이던
버그 수정: `qinputs`(양자화 입력)·`qparams`(양자화 weight/bias)를 계산만 하고
`super().forward(*args)` 에 원본을 넘겨 무시하고 있었다. → `qinputs` 를 forward 에 사용하고,
비-inplace `qparams` 를 forward 동안만 weight/bias 에 주입 후 복원하도록 변경.
