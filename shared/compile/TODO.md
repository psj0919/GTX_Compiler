# ggml codegen — operator coverage TODO

ggml 정규 op 97개(106 − 제외 9) 대상 codegen 커버리지의 남은 작업.
현황 검증: `uv run python tools/ggml_codegen_coverage.py`. 단일 소스: `shared/compile/ggml_ops.py`.

## ✅ Done

- **빌더 가용성 97/97** — 모든 대상 op 이 현재 ggml 라이브러리에 빌더 존재(conv_transpose_2d 는 `_p0` 변형).
- **render 와이어링 64/97** — codegen 이 실제 ggml 호출을 emit:
  - 기본/활성화: conv_2d, conv_2d_dw, mul_mat, pool_2d, add/sub/mul/div, relu/silu/sigmoid/gelu(+erf/quick)/tanh/leaky_relu/elu/clamp/relu6/softplus/hardsigmoid/hardswish, soft_max/log_softmax, exp/log/neg/floor/ceil, sqrt/sum/mean, scale, norm/group_norm, concat/cont/repeat/upscale/arange/top_k/argmax/get_rows/view, pad, reshape/unsqueeze 등
  - **B그룹 단일 op(완료)**: sin/cos/abs/sgn/step/round/sqr/expm1/trunc/xielu, argsort/sum_rows/cumsum, pool_1d(max/avg), conv_3d, conv_transpose_1d/2d, pad_reflect_1d (OP enum 추가 + render). conv_3d/conv_transpose 는 인자 시그니처 best-effort(TODO 주석).
  - **fused(op 시퀀스 융합, `ggml_fusion.py`)**: swiglu/geglu/reglu, rms_norm, l2_norm, flash_attn_ext
- **fusion 프레임워크**: `shared/compile/ggml_fusion.py` (`shared/inspector` SubgraphMatcher 활용, 패턴 6종). backbone(YOLO/ResNet) 무회귀 확인.
- **검증 도구**: `tools/ggml_codegen_coverage.py`.

## ⏳ 남은 33개 — 분류별 구현 방안

### A. fused op — 실제 모델 trace 패턴 필요 (pattern-match 로 wiring 가능)
`rope`, `add_rel_pos`/`get_rel_pos`, `timestep_embedding`, `flash_attn_ext`(정교화), GLU 변형(`geglu_erf`/`geglu_quick`/`swiglu_oai`).
- 현재 `flash_attn` 패턴은 **직접 인접** `matmul→softmax→matmul` 만 매칭 → 실제 attention 은 중간에 `scale`/`transpose` 가 끼어 미매칭. 가변 패턴(옵션 노드 허용) 필요.
- **구현 전제**: transformer 계열 비전 모델(ViT/DETR/RT-DETR/SAM 등) 입력을 trace 해 실제 op 분해를 확인해야 패턴이 맞다(이 repo 엔 아직 해당 아키텍처 입력 없음 → 합성 그래프로만 검증됨).

### B. 단일 op — ✅ 완료 (OP enum 추가 + render wiring)
`sin/cos/abs/sgn/step/round/sqr/expm1/trunc/xielu`, `argsort/sum_rows/cumsum`,
`pool_1d(max/avg)`, `conv_3d`, `conv_transpose_1d/2d`, `pad_reflect_1d` → `vision_ops_render.py`.
- 남은 잔여: `im2col`/`im2col_3d` (conv 내부 연산, 사용자 PyTorch op 아님 → 보류).
- **주의(parse-side)**: render(codegen)는 준비됐으나, 실제 트리거되려면 `parse/op_dispatcher`/
  `utils/op_register` 의 **torch aten op → OP enum 매핑**이 있어야 한다(sin/cos 등 신규 OP 는
  현재 매핑 미등록). conv_3d/conv_transpose 는 인자 시그니처 vision.cpp 빌드 시 정밀화 필요.

### C. 아키텍처 전용 — 해당 모델 컴파일 시에만 의미
`rwkv_wkv6/7`, `ssm_conv`, `ssm_scan`, `gated_linear_attn`. RWKV/Mamba(SSM) 모델 지원 시 구현.

### D. 메모리/내부 op — PyTorch trace 에 거의 안 나옴 (낮은 우선순위)
`acc`, `add1`, `add_id`, `cpy`, `dup`, `fill`, `set`/`set_rows`, `out_prod`, `count_equal`, `mul_mat_id`, `roll`, `diag`, `diag_mask_inf/zero`, `tri`, `solve_tri`, `win_part/unpart`.

## 🔧 활용 가능한 기존 모듈 (구현 시)

### `shared/graph/graph_searcher.py` — **권장 fusion 매처** (네이티브 chain 매칭)
메인 Graph IR 에서 직접 동작하는 선형(chain) 패턴 매처. `deploy_optimizer`/`optimization/commander`/
`parse`/`qproc/adaquant` 에서 **이미 실사용 중**. networkx/inspector 어댑터 불필요 + 매치별 `action` 콜백 + 다중 패턴 동시 매칭.

```python
from shared.graph.graph_searcher import GraphSearcher
from shared.utils.pattern_matcher import PatternType   # PatternType([op_type...], action)

def fuse_rms_norm(pattern, node_set, graph):
    ...  # node_set(매치된 노드 리스트)을 fused 노드로 치환 / codegen 마킹
patterns = [PatternType(["square", "mean", "elemwise_add", "sqrt", "elemwise_div"], fuse_rms_norm)]
node_sets = GraphSearcher(graph).find_nodes_from_type(patterns)
```

→ 현재 `ggml_fusion.py` 의 inspector `SubgraphMatcher`(networkx subgraph isomorphism)는 분기 패턴까지
되지만 graph 변환 비용이 있다. **선형 체인(대부분의 fused 분해)은 GraphSearcher 로 이관**하면 더 가볍고
실 Graph IR 와 정합. 분기/다이아몬드 패턴(flash_attn 등)만 SubgraphMatcher 유지하는 하이브리드 권장.

### `shared/graph/operator_definition.py` — op/attr 정의 + fused 노드 생성
op 별 `AttrName` enum + `IrAttr`(예: Conv2d 의 KERNEL/STRIDE/PAD/GROUP/BIAS_TERM) 의 **권위 소스**.
- render 의 attr 키를 추정 별칭(`render_api._ATTR_ALIASES`) 대신 여기서 정확히 가져오면 견고.
- **graph-rewrite 방식 fusion**(매치 서브그래프를 단일 노드로 치환) 시, 새 fused `Operation` 을 이 모듈의
  `Operation` 베이스로 생성 → 일반 walk+render 경로로 처리(현재의 walk-내 anchor/skip 방식보다 깔끔).

## 검증 공백

- fusion 패턴은 합성 그래프로만 검증됨. **실모델 trace 의 op 분해와 일치하는지 미검증** → transformer
  비전 모델 입력 확보 후 A 그룹 패턴을 실측 보정해야 함.
- 생성 `.cpp` 는 vision.cpp arch 로 빌드해야 ggml 수치 검증 가능(현재 미컴파일). `m_ctx(m)` 등 일부
  헬퍼는 vision.cpp 측 존재 가정.
