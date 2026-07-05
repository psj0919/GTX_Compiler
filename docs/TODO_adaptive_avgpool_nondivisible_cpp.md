# TODO: non-divisible adaptive avg pool — C++ (vision.cpp) render

상태: **보류(별도 작업 필요)**. 알고리즘·검증 하네스는 준비됨. 레이아웃만 맞추면 완성.

## 배경 / 무엇이 되고 무엇이 안 되나

`AdaptiveAvgPool2d((Ho,Wo))` 는 입력 (Hi,Wi) 가 출력의 정수배일 때(`Hi%Ho==0 && Wi%Wo==0`)
고정 커널 avg pool 과 동치라 `ggml_pool_2d(GGML_OP_POOL_AVG, ...)` 로 정확히 낮춰진다.
**비-정수배(진짜 가변커널)** 는 단일 pool 로 표현 불가.

| 경로 | divisible (1×1 global 포함) | non-divisible |
|---|---|---|
| **eager** (`python output/<Model>.py`, `nn/modules/ggml_backend.py`) | ✅ 정확 | ✅ **정확(수정 완료, 커밋됨)** — `_adaptive_avg_pool2d` 윈도우 평균 |
| **C++ render** (`nn/modules/adaptive_avg_pool.py` → vision.cpp) | ✅ 정확(`ggml_pool_2d`) | ❌ **global 폴백(부정확, shape도 1×1로 틀림)** ← 본 TODO |

## 정확 알고리즘 (이미 검증됨)

가변커널 adaptive avg pool = **분리가능 평균행렬 2개**로 정확 계산:

    Y = Mh · X · Mw^T      (Mh: Ho×Hi,  Mw: Wo×Wi)
    M[o,i] = 1/win  if  floor(o*I/O) <= i < ceil((o+1)*I/O)  else 0

- `nn/modules/adaptive_avg_pool.py` 의 `adaptive_avg_matrix(I,O)` 가 M 을 만든다(구현 존재).
- `tools/test_adaptive_pool.py` 가 `Mh·X·Mw^T == torch.adaptive_avg_pool2d` 를 단위검증(통과).
  → **수학은 맞다.** 남은 건 이 행렬 matmul 을 vision.cpp 그래프로 정확히 emit 하는 것뿐.

## 왜 아직 안 되나 — 근본 원인

시도한 render: 평균행렬 2개를 GGUF 로 baking 하고 아래처럼 emit.

    t1 = ggml_mul_mat(m, Mw, x)              // W 축 pool
    t2 = ggml_cont(m, ggml_transpose(m, t1))
    t3 = ggml_mul_mat(m, Mh, t2)             // H 축 pool
    out = ggml_cont(m, ggml_transpose(m, t3))

**문제:** vision.cpp 의 텐서 레이아웃이 **런타임 플래그(`model_build_flag::cwhn`) 의존**이다
(`vision.cpp/src/visp/nn.cpp` 의 `cwhn_to_contiguous_2d`/`contiguous_2d_to_cwhn`):
- cwhn 플래그 ON → forward 도메인이 **CWHN [C,W,H,N]** (ne0=C)
- OFF → **WHCN [W,H,C,N]** (ne0=W)

정적 생성 `.cpp` 는 어느 축이 spatial 인지 모른 채 raw `ggml_mul_mat` 로 ne0 을 contract 하므로
축을 잘못 잡아 **결과가 틀린다(측정 cos≈0.16)**. `ggml_pool_2d` 는 vision.cpp 가 레이아웃을
알고 처리하지만, adaptive 용 primitive 는 없다.

## 재개 방법 (택1)

1. **레이아웃 인지 emit** — cwhn 플래그에 따라 spatial 축(ne0,ne1 vs ne1,ne2)을 골라 contract.
   생성 .cpp 에서 `m.flags & model_build_flag::cwhn` 분기 또는 `permute_*_to_contiguous_2d` 로
   먼저 WHCN 로 정규화 후 matmul, 다시 되돌리기.
2. **vision.cpp 에 adaptive_avg_pool primitive 추가** (평균행렬 matmul 을 C++ 헬퍼로) — 레이아웃을
   그 안에서 처리. `ggml_pool_2d` 와 같은 레벨의 재사용 함수.

baking 통로(render→GGUF)는 미구현 상태로 되돌렸으니, 재개 시 `RenderContext` 에 상수 bake 채널
(`extra_baked` {key:ndarray} → `generate_gguf` 병합)부터 다시 추가해야 한다(과거 시도 참고: git
history 의 이 TODO 도입 커밋 직전 revert).

## 검증 하네스 (준비됨)

C++ render 를 vision.cpp 로 빌드해 torch 와 수치 대조 가능(이 루프로 resnet18 cos=0.999999 확인):

    # 1) 모델 생성 (non-divisible adaptive 포함)
    uv run g2c --model <spec> --output <dir>
    # 2) 입력/ref bin: x[0].permute(1,2,0).ravel() (CWHN 선형순서), torch 출력 ravel
    # 3) vision.cpp 빌드
    bash tools/build_yolo_cpp.sh <dir> <Arch>
    # 4) 실행 (SZ=정사각 입력변)  → out.bin
    <dir>/run_yolo_cpp <dir>/<Arch>.gguf <dir>/in.bin <dir>/out.bin <SZ>
    # 5) out.bin 은 CWHN [C,Wo,Ho,N] flat(C fastest) → torch (N,C,Ho,Wo) 와 축 재배열 후 cos

참고: 검증 모델은 실제로 non-divisible adaptive 를 쓰는 것이 이상적(예: adaptive_avg_pool2d((3,3))
을 7×7 에 적용). eager 는 이미 정확하므로 eager 출력을 중간 기준으로도 쓸 수 있다.
