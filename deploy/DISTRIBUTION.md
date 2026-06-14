# 배포 가이드 — Cython 소스 보호 빌드 (Linux)

사내 서버에 **소스 코드를 노출하지 않고** g2c 컴파일러를 공유하기 위한 가이드.
우리 핵심 로직(`shared/parse/qproc/quantization/utils/nn`)을 **Cython 으로 네이티브
`.so` 로 컴파일**하고, torch/ultralytics 등 외부 의존성은 일반 pip 패키지로 둔다.
배포 형태는 **CLI 실행 + 라이브러리 import 둘 다** 지원한다.

> **Nuitka 대비:** Nuitka onefile 은 torch 전체를 정적 분석·번들해 20~40분/수 GB 가 든다.
> Cython 은 **모듈별 독립 컴파일**이라 빠르고(수 분), torch 는 번들하지 않는다(pip 설치).
> 보호 강도가 최우선이며 성능 최적화(cdef 타이핑)는 목표가 아니다 — `.py` 를 "있는
> 그대로" `language_level=3` + **`annotation_typing=False`** 로 컴파일한다(아래 §4 주의).

- 빌드 스크립트: [`build_cython.sh`](build_cython.sh) (빌드 + strip + .c 정리 일괄)
- 컴파일 setup: [`setup_cython.py`](setup_cython.py)

---

## 1. 보안 모델 — 무엇이 보호되는가

| 항목 | 배포본 형태 | 보호 수준 |
|---|---|---|
| 핵심 6개 패키지의 로직 모듈 | `*.so` (Cython→C→머신코드, **strip 됨**) | **소스 복원 불가** — `.py`/`.pyx`/`.c` 미동봉, 심볼 제거 |
| 코드젠/fusion/양자화 알고리즘 | 위 `.so` 에 포함 | 보호됨 |
| `__init__.py`, 진입 런처 `g2c.py` | 평문 `.py` (import 집계·인자 파싱만) | 노출(IP 없음) |
| 컴파일 제외 모듈(§3) | 평문 `.py` | 노출(불가피 — jit.script/예제) |
| torch / torchvision / ultralytics / gguf | pip 설치 (오픈소스) | 보호 대상 아님 |

> **정직한 한계:** Cython `.so` 는 C 로 변환 후 컴파일·strip 되어 바이트코드 디컴파일이
> 불가능하다. 다만 네이티브 디스어셈블은 이론상 가능(모든 컴파일 SW 동일). "소스를
> 안 보여주고 사용하게 한다" 목적에는 충분하다.

---

## 2. 의존성 분류

| 패키지 | 용도 | 처리 |
|---|---|---|
| **torch / torchvision** | 본체에서 사용(파싱·텐서) | 컴파일 대상이 **호출**만 함 → 본체 `.so` 컴파일, torch 자체는 pip |
| **ultralytics** | `--model yolo*` 분기에서만(조건부) | 컴파일 안 함. pip(선택) — YOLO 컴파일 시에만 필요 |
| **matplotlib** | `shared/inspector/graph.py`·`shared/utils/plot.py` 의 **지연 import**(디버그 시각화) | 컴파일 OK(지연), 번들·설치 안 함 → 해당 plot 함수는 미사용 |
| networkx / tqdm / numpy / gguf | 본체 사용 | pip |

torch 모델 저장은 전부 `torch.save(state_dict)` 또는 텐서 저장만 사용 — **전체 모델
pickle 없음**(rule 5 충족, 변경 불필요).

---

## 3. 컴파일 제외 목록 (평문 `.py` 유지)

| 대상 | 이유 (규칙) |
|---|---|
| `nn/modules/rnn_builder/` (전체) | `stacked_lstm` 이 `torch.jit.script` 대상(`qproc/utils.py:550,558`). jit.script 는 Python 소스 필요 → 컴파일 시 RNN-T `jit_script` 경로 깨짐 [3c] |
| 모든 `__init__.py` | 패키지 import 집계만(IP 없음). 패키지 인식 안정성 위해 평문 유지 |
| `g2c.py` (런처) | CLI 진입점 얇은 래퍼 — 인자 파싱·`main()` 호출만 [4] |
| `resnet.py`(루트), `test/`, `tools/` | 예제·데모·개발 스크립트. 배포본 미포함(ultralytics/matplotlib 사용처 포함) [3a] |

**안전 확인됨(컴파일 OK):**
- `utils/torch_op_attr.py` 의 `inspect.signature(...)`·`torch.jit.script(attr)` →
  대상이 **torch 외부 객체**(torch.nn.Conv2d / torch.nn.functional 함수)이고 try/except
  로 감싸져 우리 컴파일 코드와 무관 [3b/3c 비해당]
- `pipeline.py`·`code_template.py` 의 `if __name__=="__main__"` 가드 → 모듈 import 시
  비실행이므로 컴파일해도 무해(직접 실행은 `g2c.py` 런처가 대신)

**소스 잠재 버그 수정 → 전부 컴파일됨(200/200):** 아래 5개는 원래 소스의 잠재 버그/오타
(CPython 에선 해당 분기 실행 시에만 드러남)라 Cython 정적 분석을 통과 못 했으나, **버그를
바로잡아 이제 모두 `.so` 로 컴파일·보호된다**(동작 보존 수정).

| 파일 | 원래 버그 | 수정 |
|---|---|---|
| `nn/quantization/ops/bfp_ops.py` | `f(...)`(f-string 오타); `_to_bfp_prime`(미정의) 호출 | `f"..."` 로; 미사용이던 `BFPPrimeQuantize.apply` 로(형제 v1/v2/v3 와 동일 패턴) |
| `nn/quantization/ops/round_ops.py` | `return linear`(미정의) | `linear` 항등 함수 정의(get(None) 의 pass-through) |
| `shared/compile/xcompiler.py` | `graph = graph.Graph...`(미정의/shadow) | Vitis AI `xir.Graph.deserialize` 지역 import(xcompiler 와 동일 외부 생태계) |
| `shared/utils/parameters.py` | comprehension 내 미정의 `node` | `param_list[0]`(center 분기와 대칭 — gamma) |
| `utils/profiler.py` | 미정의 `m` | `module`(함수 인자) |

추가로 `bfp_ops.py:21` 의 `from utils import onnx_utils`(저장소에 없는 모듈, 파일 내 미사용
**죽은 import**)를 제거해 `nn.quantization` 서브패키지가 정상 import 된다. 이제
`BFPQuantizer`·`bfp_ops`·`round_ops` 가 양자화에서 실제로 쓰일 수 있다(소스/배포 모두 검증:
`round_ops.get(None)`=항등, `quantize_to_bfp_prime`/`_v1` 동작).

**Cython 호환을 위한 소스 패치(3곳, 동작 보존):** 컴파일된 함수/모듈은 순수 Python 과
런타임 introspection 결과가 달라 아래만 조정했다(소스-실행 경로도 동일하게 정상).
- `utils/op_register.py` — 데코레이터의 `inspect.isfunction(func)` → `callable(func)`
  (Cython 함수는 `types.FunctionType` 이 아님)
- `nn/load_kernels.py` — AOT 커널 감지를 "아무 `.so`" → **`_kernels*.so` 만** 으로 특정화
  (배포본 `nn/` 가 우리 `.so` 로 가득 차 오탐하던 문제)
- `utils/torch_op_attr.py` — `__builtins__[fn]` → `getattr(builtins, fn)`
  (Cython 모듈에선 `__builtins__` 가 dict 가 아니라 module 객체)

---

## 4. 빌드 (Linux, 타겟 Python 고정)

`.so` 는 Python ABI(`cpython-3XX`)에 묶인다 → **배포 서버와 동일 Python minor 버전**으로
빌드해야 한다. (타겟 버전: **Python 3.12** — `.python-version` / 프로젝트 `.venv`. 빌드
산출 .so 는 `*.cpython-312-x86_64-linux-gnu.so`)

### 전제조건
```bash
pip install "cython>=3.0" build
pip install torch torchvision networkx tqdm numpy gguf   # 빌드 분석/검증용
#   torch CPU: --index-url https://download.pytorch.org/whl/cpu
```

### 실행
```bash
bash deploy/build_cython.sh                 # → deploy/dist_cython (스테이징 배포본)
STAGE=/tmp/dist PY=python bash deploy/build_cython.sh   # 출력/인터프리터 지정
```
`build_cython.sh` 가 하는 일(스테이징 디렉토리 = `deploy/dist_cython` 기본):
0. 스테이징 초기화 후 6개 패키지를 복사(test/tools/resnet.py 등 예제는 미포함), stale
   `__pycache__` 제거
1. `setup_cython.py` 로 `.py`(§3 제외목록 제외)를 `.so` 로 in-place 컴파일
   - `compiler_directives={"language_level":"3","emit_code_comments":False,`
     **`"annotation_typing":False"`**`}` — 어노테이션 타입 강제 끄기(없으면 duck-typed
     인자 전달부가 런타임에 `expected str, got ...` 로 깨짐)
   - **파일별 cythonize** — 정적 컴파일 불가 파일이 있으면 건너뛰고 `.cython_skipped.txt`
     로 보고(현재는 0개, §3 의 잠재 버그 수정으로 200/200 컴파일)
2. 생성된 `.so` 의 심볼 제거: `strip --strip-all`
3. 중간 산출 `.c` 전부 삭제
4. 컴파일된 모듈의 원본 `.py` 제거(배포본에서 `.so` 만 남김), `build/`·`*.egg-info`·모든
   `__pycache__` 제거(.pyc 유출 방지)
5. 평문 진입물 생성: `g2c.py` 런처 / `requirements.txt` / `README.md`

> **검증됨(2026-06):** Python 3.12 / Cython 3.2.5 / gcc 11.4 에서 **200/200 `.so` 생성**, resnet18
> 컴파일이 소스 빌드와 **노드 수(51)·`ResNet.cpp`·`ResNet.h`·`.gguf` 바이트 동일**.

> Cython 은 `.py` 를 같은 위치 `.so` 로 만들므로 **실제 `__file__` 경로가 보존**된다
> → `pipeline.project_root = dirname^3(__file__)`, `nn/load_kernels.py` 의 `os.listdir`
> 등 경로 의존 코드가 그대로 동작(Nuitka 가상화 불필요).

---

## 5. 배포 패키징

```
배포본/
├── shared/ parse/ qproc/ quantization/ utils/ nn/   # *.so + __init__.py + 제외 .py
│     └── (각 모듈 .py → .so, .c·원본.py 제거됨)
├── nn/modules/rnn_builder/*.py                       # 제외(평문) — jit.script 용
├── g2c.py                                            # CLI 런처(평문)
├── requirements.txt                                  # torch 등 외부 의존성
└── README.md
```
wheel 배포 시: `python -m build --wheel` (sdist 금지). MANIFEST/pyproject 에서 컴파일
모듈의 `.py/.pyx/.c` 제외, `.so` 만 포함.

---

## 6. 수령측(사내 서버) 설치 / 실행

```bash
python -m venv .venv && source .venv/bin/activate     # Python 은 빌드와 동일 minor
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CLI
python g2c.py --model resnet18 --output output/resnet18
# 라이브러리
python -c "from shared.compile.pipeline import compile_model; ..."
```
생성물: `output/<Model>.{cpp,h,gguf,py,weights.txt}`

> **Python 버전 일치 필수:** `.so` 는 빌드 시점 Python minor 버전에서만 로드된다.

---

## 7. 검증 (빌드 후 반드시 수행) — 2026-06 통과

배포본 디렉토리(`deploy/dist_cython`)에서 소스 트리 격리 실행:
`cd deploy/dist_cython && env -u PYTHONPATH python ...`

- [x] **import 스모크**: `import shared.compile.pipeline, nn; len(RENDERERS)` → 90 정상
- [x] **CLI 스모크**: `python g2c.py --model resnet18 --output out_smoke` → `.cpp/.h/.gguf/.py/.weights.txt` 생성
- [x] **기능 동등성**: 소스 빌드와 노드 수(51)·`ResNet.cpp`·`ResNet.h` 동일, `.gguf` 바이트 동일(42 tensors)
- [x] **모델 저장/로드**: `state_dict` → fp16 GGUF 정상(rnn 제외경로 영향 없음)
- [x] **보호 검증**: `.py/.pyx/.c` = 0(컴파일 대상), `.so` = 200, `__pycache__` = 0. 잔여 `.py` 27개는
      런처+`__init__`(22)+`rnn_builder`(4)뿐
