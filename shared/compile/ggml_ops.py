#
# Copyright 2025 Supergate.cc, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""GGML operator 정규 목록(106개) + codegen 대상(97개) 단일 소스.

분류는 ggml 의 GGML_OP / GGML_GLU_OP / GGML_UNARY_OP 를 따른다. codegen 은 이 중
**제외 9개**(view/메타데이터 5 + 학습 3 + SWIGLU 점검)를 뺀 **97개**를 대상으로 한다.
커버리지 검증: ``tools/ggml_op_coverage.py``.
"""

# --- GLU OP (6) ---
GLU_OPS = ["GEGLU", "GEGLU_ERF", "GEGLU_QUICK", "REGLU", "SWIGLU", "SWIGLU_OAI"]

# --- 기본 OP (78) ---
BASIC_OPS = [
    "ACC", "ADD", "ADD_ID", "ADD_REL_POS", "ADD1", "ARANGE", "ARGMAX", "ARGSORT",
    "CLAMP", "CONCAT", "CONT", "CONV_2D", "CONV_2D_DW", "CONV_3D",
    "CONV_TRANSPOSE_1D", "CONV_TRANSPOSE_2D", "COS", "COUNT_EQUAL", "CPY",
    "CROSS_ENTROPY_LOSS", "CUMSUM", "DIAG", "DIAG_MASK_INF", "DIAG_MASK_ZERO",
    "DIV", "DUP", "FILL", "FLASH_ATTN_EXT", "GATED_LINEAR_ATTN", "GET_REL_POS",
    "GET_ROWS", "GROUP_NORM", "IM2COL", "IM2COL_3D", "L2_NORM", "LEAKY_RELU",
    "LOG", "MEAN", "MUL", "MUL_MAT", "MUL_MAT_ID", "NORM", "OPT_STEP_ADAMW",
    "OPT_STEP_SGD", "OUT_PROD", "PAD", "PAD_REFLECT_1D", "PERMUTE", "POOL_1D",
    "POOL_2D", "REPEAT", "RESHAPE", "RMS_NORM", "ROLL", "ROPE", "RWKV_WKV6",
    "RWKV_WKV7", "SCALE", "SET", "SET_ROWS", "SIN", "SOFT_MAX", "SOLVE_TRI",
    "SQR", "SQRT", "SSM_CONV", "SSM_SCAN", "SUB", "SUM", "SUM_ROWS",
    "TIMESTEP_EMBEDDING", "TOP_K", "TRANSPOSE", "TRI", "UPSCALE", "VIEW",
    "WIN_PART", "WIN_UNPART",
]

# --- Unary OP (22) ---
UNARY_OPS = [
    "CEIL", "ELU", "EXP", "EXPM1", "FLOOR", "GELU", "GELU_ERF", "GELU_QUICK",
    "HARDSIGMOID", "HARDSWISH", "NEG", "RELU", "ROUND", "SGN", "SIGMOID",
    "SILU", "SOFTPLUS", "STEP", "TANH", "TRUNC", "XIELU", "ABS",
]

ALL_OPS = GLU_OPS + BASIC_OPS + UNARY_OPS  # 106

# 제외 9개: view/메타데이터(5) + 학습(3) + GLU 점검(1)
EXCLUDED = {
    "VIEW": "view/메타데이터",
    "RESHAPE": "view/메타데이터",
    "PERMUTE": "view/메타데이터",
    "TRANSPOSE": "view/메타데이터",
    "CONT": "view 보조",
    "CROSS_ENTROPY_LOSS": "학습",
    "OPT_STEP_ADAMW": "학습",
    "OPT_STEP_SGD": "학습",
    "SWIGLU": "GLU(점검 필요)",
}

TARGET_OPS = [o for o in ALL_OPS if o not in EXCLUDED]  # 97


def builder_name(op, ggml_module=None):
    """GGML_OP 이름 → ggml 빌더 함수명. ggml_module 주면 실제 존재하는 변형을 반환."""
    base = f"ggml_{op.lower()}"
    if ggml_module is None:
        return base
    avail = set(dir(ggml_module))
    for cand in (base, base + "_inplace", base + "_ext", base + "_p0",
                 base + "_1d", base + "_2d", base + "_3d", base + "_4d"):
        if cand in avail:
            return cand
    return None


def emitted_builders(roots=None):
    """render 코드(nn/modules, shared/compile)가 emit 하는 ggml_* 빌더 집합을 스캔."""
    import glob
    import os
    import re

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # render 는 카테고리 서브폴더(activations/render.py, math/render.py 등)에 있으므로
    # 재귀(**)로 스캔한다. deprecated/ 는 inert(활성 경로에서 import 안 함)이라 제외.
    roots = roots or [
        os.path.join(here, "nn", "modules", "**", "*.py"),
        os.path.join(here, "shared", "compile", "**", "*.py"),
    ]
    out = set()
    for pat in roots:
        for f in glob.glob(pat, recursive=True):
            if os.path.basename(f) == "ggml_ops.py":
                continue
            if "deprecated" in f.replace("\\", "/").split("/"):
                continue
            try:
                txt = open(f, encoding="utf-8").read()
            except Exception:
                continue
            for m in re.findall(r"\bggml_[a-z0-9_]+", txt):
                out.add(m)
                out.add(re.sub(r"_(1|2|3|4)d$", "", m))  # reshape_2d → reshape (별칭도)
                out.add(re.sub(r"_split$", "", m))       # geglu_split → geglu (GLU 융합)
    return out
