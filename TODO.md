# TODO — AI 모델 검증용 기능 구현 (dev/260703_AI_모델_검증용_기능_구현_요청.pdf)

출처: `dev/260703_AI_모델_검증용_기능_구현_요청.pdf` (supergate, 12p).
목표: **검증 사항 확인용** 기능 3종 — ① 그래프 시각화, ② 컴파일 로그 구체화, ③ 런타임 프로파일링.

> **백엔드 주의:** PDF 예시(NPU/CPU dispatch, INT8/FP16, CAST, xmodel `.mblt`, MBLT Netron)는
> **NPU/xmodel(DPU) 배포 백엔드** 지향이다. 이 repo 는 두 백엔드가 있다 —
> (A) **ggml/vision.cpp**(`g2c`, 최근 작업 대부분), (B) **xmodel/DPU**(`shared/compile/deploy_optimizer.py`,
> `DevGraphOptimizer`, `get_xmodel_and_dump_infos`, `QuantOptimizer` — NPU/INT8 dispatch·CAST 개념 보유).
> 각 기능이 어느 백엔드를 대상으로 하는지 먼저 확정할 것. 아래는 그 매핑을 표기했다.

---

## 1. 그래프 시각화 (PDF p3)

**요구:** GGUF 그래프 시각화 툴로 연산자의 **hyperparameter(attributes) + parameter + input + output 값·사이즈를
한 번에** 확인. (현 graphviz 는 input/output·size 만) — MBLT Netron 처럼 NODE PROPERTIES(type/ATTRIBUTES/
INPUTS/OUTPUTS + weight/bias tensor dtype·shape) 노출.

**현재 상태 (부분 완료):**
- `tools/graph_visualizer.py` 가 Netron `dot.js` record 규칙으로 DOT emit → op 가 노드 **type**, hyperparameter
  (kernel/stride/pad/dilation/group/in_dim/out_dim…)와 out_shape 이 **개별 attribute 필드**로 뜬다. (이미 구현·검증)
- 컴파일 그래프 반영(Conv-BN fold + const-fold), `--extra-opt`.

**갭 / 할 일:**
- [x] **parameter(weight/bias) 텐서 정보 추가** — `_param_pairs`(`tools/graph_visualizer.py`)가 `op._params`
      의 weight/bias 를 `weight: float32[64, 7, 7, 3]`, `bias: float32[64]` 처럼 dtype·shape 필드로 추가.
      dot.js 규칙 파싱 검증 + `tools/test_dot_netron.py` param 케이스. (conv shape 는 그래프 IR 레이아웃
      [OC,KH,KW,IC] — OIHW/GGUF-정확 표기는 아래 후속 항목 참고)
- [x] input/output **텐서 이름·dtype** 도 필드로(현재 out_shape 만). in_tensors dtype/shape 노출.
      `_io_pairs`/`_tensor_desc`(graph_visualizer.py) → 노드에 `in{i}`/`out{i}: [name ]dtype[shape]`.
      DOT 이미터만 수정(다른 백엔드 무관). test_dot_netron.py in0/out0 케이스 추가.
- [ ] (선택) GGUF 를 직접 읽어 시각화하는 경로 — 현재는 Graph IR 기반. "GGUF 파일 그래프 시각화"를 문자
      그대로 요구하면 gguf tensor 메타(dtype/shape)를 노드에 조인.
- [ ] Netron 실기(브라우저) 확인은 사용자 몫(하네스는 `dot.js` 규칙 재현으로 검증). 스크린샷 첨부 여부 확인.

---

## 2. 컴파일 로그 구체화 (PDF p4–7)

**요구:** 컴파일 과정의 **그래프 최적화 + Fallback 구간을 레이어별로 기록** → 누적 오차·문제 원인 파악.

### 2-1. [1] GRAPH OPTIMIZATION — Pass 별 상세 (p5)
**요구:** Pass(Dead Node Elimination, Constant Folding …)별 표 `Layer | Op | Action(REMOVED/FOLDED) | Note(사유)`
+ Pass 헤더 `35 → 31 (-4 ops)`, 최상단 `35 ops → 13 ops (-62%)`.

**현재 상태 (부분):** dev-graph opts 는 노드 총계만 로깅(`[g2c] dev-graph opts: … (nodes 319→319)`),
const-fold 는 `N baked / M skipped`. **Pass별 레이어 단위 기록 없음.**

**할 일:**
- [x] 각 그래프 패스(`fold_conv_bn_graph`, `apply_dev_graph_opts` 의 각 `_SAFE_DEV_OPTS`, `fold_constants`)가
      **변경된 노드 리스트**(name, op, action, 사유)를 수집하도록 계측. `shared/compile/pipeline.py`.
      (pass 전후 노드 스냅샷 diff → `_node_snapshot`/`_pass_changes`, 옵티마이저 내부 무수정)
- [x] 표 렌더러(레이어별 REMOVED/FOLDED/FUSED/CONVERTED + Note) + Pass 헤더 전/후 op 수·감소율
      + 최상단 `X ops → Y ops (-Z%)`. `_render_graph_opt_report`. resnet18/yolo11n e2e 검증.
- [x] `DevGraphOptimizer` 계측 — GTX(ggml/vision.cpp) deploy 최적화기(`apply_dev_graph_opts` 가 g2c
      경로에서 호출, `_SAFE_DEV_OPTS` 실행). 위 계측에 이미 각 dev pass 별로 포함됨.

### 2-2. [2] DISPATCH & FALLBACK (p6)
**요구:** 레이어별 **실행 디바이스(NPU/CPU) + DType(INT8/FP16/FP32)** + 강제 **CAST 삽입 위치** 추적.
표 `Layer | Op | Target | DType | Note`, 범례 `! fallback`, `-- cast inserted`.

**백엔드 확정:** **GTX Architecture = 전면 NPU 구동**(모든 op 을 자체 arch NPU 에서 fp16 실행) →
미지원 op·CPU fallback·CAST 삽입 **없음**. dispatch 는 단일 타깃이라 Target=NPU(GTX) 고정.

**할 일:**
- [x] **대상 백엔드 결정** — GTX(ggml/vision.cpp), 전면 NPU/fp16. fallback/CAST 개념 없음.
- [x] 레이어별 target·dtype 태깅 — `_render_dispatch_table`(pipeline.py). Target=NPU(GTX),
      DType=FP16(기본)/INT8(`--quantize` 자격 conv·linear, quant_plan 조회). resnet18 fp16/q8_0 검증.
- [x] CAST·`! fallback`·precision violation — GTX 전면 NPU/fp16 이라 **해당 없음**(범례만 표기).
- [x] 표 렌더러 + 범례 — `Layer|Op|Target|DType|Note` + 범례.

### 2-3. 컴파일 최종 요약 (Summary) (p4, p7)
**요구:** Ops 표(Original/After graph opt/After dispatch, removed, cast inserted, **Operator breakdown**(종류별 빈도), TOTAL)
+ Dispatch 표(Target별 Ops·Ratio%, NPU total/CPU total 점유율).

**할 일:**
- [x] 컴파일 종료 시 집계 렌더러: 최적화 전/후 연산자 총량·종류별 빈도, cast 수.
      `_render_compile_summary`(pipeline.py) — Ops 표(Original/After opt/After dispatch, removed,
      cast=0) + Operator breakdown(op별 빈도, TOTAL). resnet18/yolo11n 검증.
- [x] Target별 op 수·비율(%), NPU vs CPU 점유율. **GTX = 전면 NPU**(fallback/CAST 없음, fp16)
      → NPU(GTX) 100% / CPU 0%. (dispatch 개념이 단일 타깃이라 표는 확정값.)

---

## 3. 런타임 프로파일링 제공 (PDF p8–12)

**요구:** 런타임에 **연산자별 지연시간(Latency) + 사용 메모리** 기록 → 병목 체크. (TensorFlow Profiler 참고)

**구현:** `nn/modules/ggml_profiler.py` — `GgmlModule.__call__`(op forward dispatch 단일 지점)에
**데코레이터**(`@profile`) 로 op 별 계측. `GTX_PROFILE=1` 로 on(미설정 시 무오버헤드). atexit dump.

### 3-1. 측정 항목
- [x] **Latency**: op 단위 `time.perf_counter` (입력~출력).
- [x] **메모리**: Activation(입력+출력, fp16), Weight(param), Peak(상주 weight + 최대 activation).
      ※ Scratch(ggml 내부 mem-pool)는 비노출이라 생략(콘솔에 명시).
- [x] **Time Share(%)**: op latency / 전체 추론 시간.

### 3-2. 실행 제어
- [x] on/off — **컴파일 옵션** `g2c --profile [--profile-reps N]`(생성 .py 최상단에 env baking) +
      **런타임 env** `GTX_PROFILE=1`. baking 은 setdefault 라 실행 시 env 로 오버라이드(끄기 포함) 가능.
- [x] **횟수 기반 통계** — `--profile-reps N` / `GTX_PROFILE_REPS=N` → 1회차 warmup 제외 후 Average/Max/Min.
      (runner 가 forward 를 N회 반복; profiler 가 rep 별 집계.)

### 3-3. 출력 형식 (다중)
- [x] **터미널/콘솔 로그** (p10): `Idx | Op | Target | DType | Lat | Share | Activation | Peak`
      + TOTAL INFERENCE TIME / WEIGHT MEMORY / GLOBAL PEAK MEMORY. (Target 전부 NPU(GTX),
      fallback/cast 없음 → Note 열 생략. Layer 명은 런타임에 없어 Idx+Op 로.)
- [x] **파일 저장**: CSV, JSON (`<Model>.profile.csv/.json`, `GTX_PROFILE_OUT` 로 경로 지정).
- [ ] **호스트 전송** (타깃 디바이스 → 호스트) — GTX ggml 은 로컬 실행이라 해당 없음(원격 NPU 실측 시 필요).
- [x] **Perfetto JSON** (p11–12): `<Model>.profile.perfetto.json` — Duration(ph=X, op latency)
      + Counter(ph=C, "Peak Memory" Allocated Bytes) + process/thread_name(ph=M). perfetto.dev 로드 검증.

### 3-4. 계측 위치
- [x] ggml 백엔드 op 단위 timing 훅 — `GgmlModule.__call__` 데코레이터 + 텐서 크기 기반 메모리(fp16).
      resnet18 검증(50 ops, weight 22.29MB = fp16 GGUF 일치).

---

## 우선순위 제안 (초안 — 확정 필요)
1. **그래프 시각화 parameter 추가** — 이미 80% 됨, 잔여 작업 작음. 빠른 완결.
2. **[1] GRAPH OPTIMIZATION Pass별 로그** — dev-graph opts 계측과 직결, ggml 경로에서 바로 가능.
3. **런타임 프로파일링(콘솔 + CSV/JSON + Perfetto)** — 신규 구현 큼, 독립 가치 높음.
4. **[2] DISPATCH & FALLBACK + Summary dispatch** — **백엔드(NPU/xmodel vs ggml) 확정이 선행**되어야 함.

## 확정이 필요한 질문
- 이 기능들의 **대상 백엔드**가 ggml/vision.cpp 인가, NPU/xmodel(DPU) 인가? (dispatch·INT8·CAST 예시는 후자 지향)
- 런타임 프로파일링의 **실행 타깃**(NPU 실측 vs ggml CPU 실측)은?
- 그래프 시각화는 현 DOT/Netron 경로 유지인가, 별도 GGUF 직접 리더인가?
