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
"""op 시퀀스 → 단일 fused ggml op 융합 (shared/inspector SubgraphMatcher 활용).

단일 PyTorch op 이 아니라 **op 시퀀스**로만 표현되는 fused ggml op
(SwiGLU/GEGLU/REGLU, RMS_NORM, L2_NORM, FLASH_ATTN_EXT)을 Graph IR 에서 서브그래프
패턴 매칭으로 찾아 하나의 ggml 호출로 emit 한다. VispCodeGenerator 가 walk 전에
:func:`detect_fusions` 를 호출해 (anchor 노드 → emit) 와 (스킵할 내부 노드)를 얻는다.

⚠️ 패턴은 transformer 계열 모델의 일반적 분해를 가정한 **시작 세트**다. 실제 모델
trace 의 op 분해와 일치하는지(특히 RMS_NORM/FLASH_ATTN) 검증이 필요하며, backbone
(YOLO/ResNet)에는 매칭되지 않도록 보수적으로 정의했다.
"""

from shared.inspector.dpu_pattern_match import SubgraphMatcher
from shared.inspector.graph import Graph, Node

# silu 는 정규화 전(aten::silu_)/후(silu) 둘 다 허용
_SILU = {"aten::silu_", "silu"}
_MUL = {"elemwise_mul", "mul"}
_DIV = {"elemwise_div"}
_ADD = {"elemwise_add", "add"}
_SQR = {"square"}


class FusionSpec:
    """fused op 1종: 패턴 서브그래프 + anchor(출력 노드) + emit.

    pattern: list[(role, {op_types})], edges: list[(role_src, role_dst)].
    anchor: 출력 역할(패턴에서 자식 없는 노드). emit(ctx, roles) -> ggml 식 문자열.
    """

    def __init__(self, name, ggml_op, pattern, edges, anchor, emit):
        self.name = name
        self.ggml_op = ggml_op
        self.pattern = pattern
        self.edges = edges
        self.anchor = anchor
        self.emit = emit

    def to_graph(self):
        g = Graph(self.name)
        for role, types in self.pattern:
            g.add_node(role, Node(set(types)))
        for a, b in self.edges:
            g.add_edge(a, b)
        return g


def _var(ctx, tensor, default="x"):
    return ctx._var.get(id(tensor), default)


def _other_input(node, exclude_node):
    """node 의 in_tensors 중 exclude_node 가 만들지 않은 텐서(외부 입력)를 반환."""
    excl_out = set(id(t) for t in (exclude_node.out_tensors or []))
    for t in node.in_tensors:
        if t is not None and id(t) not in excl_out:
            return t
    return node.in_tensors[0] if node.in_tensors else None


# ---------------------------------------------------------------- emit 함수
def _emit_glu(builder):
    def emit(ctx, roles):
        act, mul = roles["act"], roles["mul"]
        gate = _var(ctx, act.in_tensors[0])          # act 입력 = gate 분기
        up = _var(ctx, _other_input(mul, act))       # mul 의 다른 입력 = up 분기
        return f"{builder}(m, {gate}, {up})"
    return emit


def _emit_rms_norm(ctx, roles):
    x = _var(ctx, roles["sqr"].in_tensors[0])
    return f"ggml_rms_norm(m, {x}, 1e-6f) /* fused RMS_NORM */"


def _emit_l2_norm(ctx, roles):
    x = _var(ctx, roles["sqr"].in_tensors[0])
    return f"ggml_l2_norm(m, {x}, 1e-12f) /* fused L2_NORM */"


def _emit_flash_attn(ctx, roles):
    mm1, mm2 = roles["qk"], roles["av"]
    q = _var(ctx, mm1.in_tensors[0])
    k = _var(ctx, mm1.in_tensors[1] if len(mm1.in_tensors) > 1 else mm1.in_tensors[0])
    v = _var(ctx, _other_input(mm2, roles["softmax"]))
    return (f"ggml_flash_attn_ext(m, {q}, {k}, {v}, NULL, 1.0f, 0.0f, 0.0f)"
            " /* fused attention; TODO(ggml): scale/mask/operand order 확인 */")


# ---------------------------------------------------------------- 패턴 정의
FUSIONS = [
    FusionSpec("swiglu", "ggml_swiglu_split",
               [("act", _SILU), ("mul", _MUL)], [("act", "mul")],
               "mul", _emit_glu("ggml_swiglu_split")),
    FusionSpec("geglu", "ggml_geglu_split",
               [("act", {"GELU", "gelu"}), ("mul", _MUL)], [("act", "mul")],
               "mul", _emit_glu("ggml_geglu_split")),
    FusionSpec("reglu", "ggml_reglu_split",
               [("act", {"relu"}), ("mul", _MUL)], [("act", "mul")],
               "mul", _emit_glu("ggml_reglu_split")),
    FusionSpec("rms_norm", "ggml_rms_norm",
               [("sqr", _SQR), ("mean", {"mean"}), ("add", _ADD),
                ("sqrt", {"sqrt", "rsqrt"}), ("div", _DIV)],
               [("sqr", "mean"), ("mean", "add"), ("add", "sqrt"), ("sqrt", "div")],
               "div", _emit_rms_norm),
    FusionSpec("l2_norm", "ggml_l2_norm",
               [("sqr", _SQR), ("sum", {"sum"}), ("sqrt", {"sqrt"}), ("div", _DIV)],
               [("sqr", "sum"), ("sum", "sqrt"), ("sqrt", "div")],
               "div", _emit_l2_norm),
    # flash_attn fusion 은 **의도적으로 비활성화**한다. best-effort emitter 가
    # q/k/v 레이아웃·scale·mask 의미를 보존하지 못해 PVT/Libra/empirical-attention
    # 그래프에서 틀린다. 게다가 융합으로 내부 노드가 skip 되면 그 출력이 바인딩되지 않아
    # 하류 `ctx.inp()` 가 그래프 입력 x(= 이미지)로 폴백한다 —
    # `ggml_add(ln4, permute(x))` 같은 코드가 나와 can_repeat 로 죽는다.
    # 명시적 matmul → softmax → matmul 시퀀스를 그대로 둔다.
]


def detect_fusions(graph):
    """graph 에서 모든 fused 패턴을 찾아 (anchor 노드 → (FusionSpec, roles)) 와
    (스킵할 비-anchor 멤버 노드 id 집합) 을 반환.

    겹치는 매치는 멤버가 이미 다른 매치에 쓰였으면 건너뛴다.
    """
    def node_match(n1, n2):
        return n1 is not None and str(n1.op.type) in n2.get_types()

    try:
        matcher = SubgraphMatcher(graph)
    except Exception:
        return {}, set()

    anchor_emit = {}   # id(anchor_node) -> (spec, roles)
    skip = set()       # id(non-anchor member)
    used = set()       # id(any member) — 중복 융합 방지

    for spec in FUSIONS:
        try:
            matches = matcher.findPatternMatches(spec.to_graph(), node_match)
        except Exception:
            continue
        for sub in matches:
            roles = {tmpl: node for node, tmpl in sub.items()}
            member_ids = [id(n) for n in roles.values()]
            if any(mid in used for mid in member_ids):
                continue
            anchor = roles.get(spec.anchor)
            if anchor is None:
                continue
            used.update(member_ids)
            anchor_emit[id(anchor)] = (spec, roles)
            for n in roles.values():
                if id(n) != id(anchor):
                    skip.add(id(n))
    return anchor_emit, skip
