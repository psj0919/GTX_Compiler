# 빌드 & 실행 가이드

PyTorch 모델을 **vision.cpp(ggml) arch C++ + GGUF** 로 변환하는 컴파일러 `g2c` 의 빌드/실행
방법. 배포(소스 보호) 빌드와 검증 절차를 포함한다.

- 컴파일러 동작 원리: [`ggml_codegen.md`](ggml_codegen.md)
- Cython 배포 상세: [`../deploy/DISTRIBUTION.md`](../deploy/DISTRIBUTION.md)

---

## 0. 사전 준비

```bash
git submodule update --init --recursive   # vision.cpp (+ depend/ggml, llama)
uv sync                                    # 의존성 + g2c 콘솔 스크립트 설치 (torch cu124)
```

> **머신 주의:** 무거운 trace 시 thread 폭주로 hang 가능 → 모든 실행에
> `OMP_NUM_THREADS=1` 권장, 한 번에 하나씩 순차 실행.

---

## 1. 빌드 방법

세 가지 형태가 있다. 일반 사용은 **(A)** 면 충분하고, 소스 비노출 배포는 **(B)** 또는 **(C)**.

### (A) 소스 모드 — 빌드 불필요

`uv sync` 하면 `g2c` 콘솔 스크립트(`= shared.compile.pipeline:main`)가 설치된다. 별도 빌드 없이
바로 실행한다(§2-A).

### (B) 단일 `.so` — Nuitka `--module` (권장: 배포 간결)

핵심 6개 패키지(shared/parse/qproc/quantization/utils/nn)를 **하나의 `.so`** 에 임베드한다.
torch/ultralytics 등 외부 의존성은 번들하지 않고 런타임 pip 의존성으로 둔다(onefile 처럼 torch
전체 수 GB 를 끌어오지 않음).

```bash
OMP_NUM_THREADS=1 uv run python deploy/build_so.py            # 단일 스레드 빌드(--jobs=1)
#   --out <빌드 디렉토리>   기본 /tmp/g2c_so
#   --final-dir <복사 위치> 기본 deploy/
```

- 산출: `deploy/g2c_entry.cpython-3XX-x86_64-linux-gnu.so` (~22 MB). import 명 `g2c_entry`.
- **빌드는 단일 스레드**(`--jobs=1`, `OMP_NUM_THREADS=1`) — thread 폭주 hang 방지. C 링크 ~10분.
- **타겟 Python 고정**: `.so` 는 ABI(`cpython-3XX`)에 묶임 → 배포 서버와 동일 Python minor 버전.
- 실행은 §2-B 의 런처(`deploy/g2c_run.py`) 사용.

> **제약:** 단일 `.so` 는 `__file__` 기반 데이터/동적 import 코드가 깨질 수 있다. 현재
> `nn/load_kernels.py`(디렉토리 부재 시 `kernels=None`)는 대응됨. 새 `__file__` 의존 코드 추가
> 시 주의. 그 점이 중요하면 (C) 사용.

### (C) Cython 파일별 `.so` — 소스 보호 배포본

각 `.py` 를 같은 위치의 `.so` 로 in-place 컴파일(193개). `__file__` 경로가 보존돼 동적 import/
데이터 접근이 안전하다. 상세·소스 패치·제외 목록은 [`DISTRIBUTION.md`](../deploy/DISTRIBUTION.md).

```bash
# 단일스레드 빌드 권장 (WSL 등에서 thread 폭주 hang 방지)
CYTHON_JOBS=1 OMP_NUM_THREADS=1 bash deploy/build_cython.sh   # → deploy/dist_cython/ 배포본
STAGE=/tmp/dist PY=python bash deploy/build_cython.sh         # 출력/인터프리터 지정
```

- **`CYTHON_JOBS`** — cythonize/`build_ext` 동시 작업 수(미지정 시 `cpu_count`). 무거운 머신에서
  hang 이 나면 `CYTHON_JOBS=1` 로 순차 빌드. 빌드 시간↑ 이지만 안정적.
- **슬림화(g2c ggml 경로 전용):** 빌드 스크립트가 런타임 미사용 항목을 자동 제외한다 —
  HW 백엔드 C 헤더/소스(`nn/include`·`nn/src`), DPU/xmodel 전용 모듈(`dpu_pattern_handle`·
  `dpu_pattern_transform`·`device_allocator`), RNN/LSTM 전용 `nn/modules/rnn_builder`(jit.script 평문
  소스 → 유출 차단), 문서 잔재(`README.md`·`TODO.md`). 결과: `.so` 193 + 정적 컴파일 불가 평문 5개.
- **격리 실행 검증** (소스 트리 영향 없이 `.so` 만으로 동작 확인):

  ```bash
  cd deploy/dist_cython
  PYTHONPATH= OMP_NUM_THREADS=1 <repo>/.venv/bin/python g2c.py --model resnet18 --output /tmp/v
  PYTHONPATH= OMP_NUM_THREADS=1 <repo>/.venv/bin/python g2c.py --model "ultralytics.YOLO('yolo11n')" --output /tmp/v
  ```

- **메일 전달용 압축:** `tar czf deploy/compiler_dist.tar.gz -C deploy dist_cython` (≈15 MB).

---

## 2. 실행 방법

`g2c` 는 **PyTorch 모델 → `output/<Model>.{cpp,h,gguf,py}`** 를 생성한다.

### 공통 옵션 (CLI)

| 옵션 | 설명 |
|---|---|
| `--model` (필수) | 표현식 `"ultralytics.YOLO('yolo11n')"` / 이름 `resnet18`,`yolov8n` / `.pt` 경로 |
| `--output` | 출력 디렉터리 (기본 `output`) |
| `--pth` | state_dict/`.pth` 가중치 |
| `--name` | 아키텍처/파일 이름 (기본: 모델 클래스명) |
| `--input-shape` | 예 `1,3,640,640` (미지정 시 yolo→640, 그 외→224) |
| `--quantize` | GGUF 가중치 양자화 `q8_0`/`q4_0`/`q4_1`/`q5_0`/`q5_1` 등 |

### (A) 소스/설치 모드

```bash
OMP_NUM_THREADS=1 uv run g2c --model resnet18 --output output/resnet18
OMP_NUM_THREADS=1 uv run g2c --model "ultralytics.YOLO('yolo11n')" --output output/yolo11n
OMP_NUM_THREADS=1 uv run g2c --model torchvision.models.resnet50 --pth W.pth --name r50 --output output/r50
# 과거 진입점(동일 파이프라인): uv run python test/compile_to_c.py --model yolov8n --output output/yolov8n
```

### (B) 단일 `.so` 런처 — 소스 없이 실행

`deploy/build_so.py` 산출 `.so` 와 런처 `deploy/g2c_run.py` 만 있으면 된다(소스 `.py` 불필요,
torch 등은 pip 설치):

```bash
PYTHONPATH=deploy OMP_NUM_THREADS=1 uv run python deploy/g2c_run.py \
    --model resnet18 --output out

# 완전 격리(배포 검증): .so + 런처만 둔 디렉토리에서, 소스 path 배제
mkdir -p /tmp/dist && cp deploy/g2c_entry.*.so deploy/g2c_run.py /tmp/dist/
cd /tmp/dist && PYTHONPATH= OMP_NUM_THREADS=1 <python> g2c_run.py --model resnet18 --output out
```

> 검증됨: 격리 실행한 단일 `.so` 산출물(`ResNet.cpp`/`DetectionModel.cpp`)이 소스 빌드와
> **바이트 동일**.

### (C) 생성 산출물 실행 — ggml 백엔드 수치 검증

생성된 `output/<Model>.py` 는 GGUF 를 ggml(libggml.so) 커널로 eager 실행하는 진입점이다
(codegen 의 수치 기준). PyTorch 대비 cosine 검증용.

```bash
OMP_NUM_THREADS=1 uv run python output/yolo11n/DetectionModel.py     # [ggml] output: (1, 84, 8400)
```

### (D) 생성 C++ 를 vision.cpp 빌드로 실행 (E2E 검증)

생성 `.cpp` 를 vision.cpp 빌드로 컴파일·실행해 torch 와 수치 비교한다(검증된 모델:
yolov8/11/12 cos 1.0, yolov10 dense, resnet18).

```bash
# 1) vision.cpp 빌드 (최초 1회): cmake -G Ninja -DVISP_VULKAN=OFF -DVISP_TESTS=OFF
# 2) 참조 입력/출력 생성 → 컴파일 → 실행 → 비교
uv run python tools/yolo_cpp_ref.py yolov8n tools/v8n          # torch 참조 (.bin)
bash tools/build_yolo_cpp.sh output/yolov8n                    # g++ + vision.cpp 링크
./output/yolov8n/run_yolo_cpp output/yolov8n/DetectionModel.gguf tools/v8n_input.bin tools/v8n_out.bin
uv run python tools/yolo_cpp_cmp.py tools/v8n_out.bin tools/v8n_ref.bin     # cos / max|diff|
```

NMS-free(yolov10)는 그래프가 dense 까지만 출력하고 top-k 는 CPU 후처리:
`run_yolo_cpp ... 640 v10` → (300,6). 비교: `tools/yolo_v10_cmp.py`, 실이미지: `tools/yolo_v10_img_ref.py`.

### (E) 중간 텐서 탭 디버거 — emission 버그 국소화

생성 C++ ggml 그래프와 eager 백엔드를 **노드별** 비교해 첫 발산 op 를 자동 특정.

```bash
GTX_DEBUG_TAPS=all OMP_NUM_THREADS=1 uv run g2c --model "ultralytics.YOLO('yolo12n')" --output output/yolo12n
bash tools/build_yolo_cpp.sh output/yolo12n
./output/yolo12n/run_yolo_cpp <gguf> <input.bin> <out.bin> 640 none tools/taps_cpp
uv run python tools/yolo_tap_ref.py output/yolo12n <input.bin> tools/taps_ref
uv run python tools/yolo_tap_cmp.py tools/taps_cpp tools/taps_ref     # FIRST DIVERGENCE: tapN (op=...)
```

---

## 3. 트러블슈팅

| 증상 | 원인 / 대응 |
|---|---|
| trace 중 hang / 멈춤 | thread 폭주 → `OMP_NUM_THREADS=1`, 순차 실행. 빌드도 `--jobs=1`. |
| 단일 `.so` 가 `FileNotFoundError: .../nn` | `__file__` 로 디스크 디렉토리 스캔하는 코드 → 임베드 .so 엔 부재. 견디게 수정 필요(예: `load_kernels.py`). |
| `.so` import 실패 (ABI) | 배포 서버 Python minor 가 빌드와 다름. 동일 `cpython-3XX` 로 재빌드. |
| `import gguf` 실패 | `pip install gguf` (단일 `.so`/Cython 배포 시 런타임 pip 의존성). |
| 생성 `.cpp` 빌드 실패 | vision.cpp 빌드(`vision.cpp/build/`) + `depend/ggml` 체크아웃 필요. |
