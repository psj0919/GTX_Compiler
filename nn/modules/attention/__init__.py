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
"""어텐션 op 카테고리 (placeholder).

현재는 전용 어텐션 op 이 없다 — YOLO C2PSA 등의 어텐션은 matmul + softmax 조합으로
head_render 에서 처리된다. 향후 scaled-dot-product-attention 등 어텐션 전용 op 을 추가할 때
다른 카테고리(activations/layers/pooling/normalize)와 동일하게:

    attention/{op}.py   — operator (nn.Module 클래스)
    attention/quant.py  — register_quant_op 팩토리
    attention/render.py — ggml C++ render

구조로 채우고, 이 __init__ 에서 per-op 파일 + quant + render 를 import 한 뒤
nn/modules/__init__.py 에 `from .attention import *` 를 추가한다.
"""
