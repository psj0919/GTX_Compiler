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
"""입력 무관(input-independent) 상수 subgraph 를 codegen 시점에 numpy 로 평가해 GGUF
텐서로 baking 하는 패스.

detection head 의 anchor grid / stride tensor / scalar 상수(0.5, 2.0, attn scale)는
입력 해상도가 고정되면 전부 정적값이다. ggml 그래프에 arange/meshgrid/full/const 로
emit 하면 (a) `full` 의 채움값(stride)이 trace 에 안 남고 (b) meshgrid 다중출력/지연할당
등 깨지기 쉬운 best-effort 가 된다. 대신 이 패스가 그 subgraph 를 numpy 로 접어 결과를
GGUF weight 로 굽고, codegen 은 경계 텐서를 `m.weights("<key>")` 로 로드한다
(vision.cpp-native: anchor 등을 precompute/weight 로 다루는 방식과 동일).

반환:
- baked: {id(out_tensor): (gguf_key, np.ndarray(torch-shape))}  — codegen 이 m.weights emit,
  generate_gguf 가 f32 텐서로 write.
- skip:  {id(node)}  — 완전히 상수-내부(경계 아님)라 codegen 이 건너뛸 노드.
"""

import math

import numpy as np

from shared.compile.render_api import attr as _attr, op_value, out_shape


def _infer_concat_axis(out_sh, in_shs):
    """concat 결과/입력 shape 로 axis 추론(합쳐진 차원 = 크기가 다른 축)."""
    for d in range(len(out_sh)):
        if any(s[d] != out_sh[d] for s in in_shs):
            return d
    return 0


def _infer_perm(in_sh, out_sh):
    """transpose: in→out 으로 가는 축 순열 추론(중복 크기 있으면 greedy)."""
    used = [False] * len(in_sh)
    perm = []
    for od in out_sh:
        for j, sd in enumerate(in_sh):
            if not used[j] and sd == od:
                used[j] = True
                perm.append(j)
                break
        else:
            return None
    return perm


def _eval_node(node, vals, img_h):
    """상수 노드 1개를 numpy 로 평가. 출력 텐서별 값 리스트 반환(없으면 None)."""
    ov = op_value(node)
    ins = [vals.get(id(t)) for t in (node.in_tensors or []) if t is not None]
    osh = out_shape(node)

    if ov == "const":
        d = getattr(node.out_tensors[0], "data", None)
        if d is None:
            d = _attr(node, "data", None)
        return [np.asarray(d, dtype=np.float32)]

    if ov == "arange":
        n = osh[0] if osh else 0
        return [np.arange(0, n, dtype=np.float32)]

    if ov in ("elemwise_add",):
        if any(v is None for v in ins[:2]):
            return None
        return [np.asarray(ins[0]) + np.asarray(ins[1])]

    if ov in ("elementwise_sub", "elemwise_sub"):
        if any(v is None for v in ins[:2]):
            return None
        return [np.asarray(ins[0]) - np.asarray(ins[1])]

    if ov in ("elemwise_mul",):
        if any(v is None for v in ins[:2]):
            return None
        return [np.asarray(ins[0]) * np.asarray(ins[1])]

    if ov == "aten::meshgrid":
        if any(v is None for v in ins[:2]):
            return None
        a, b = np.asarray(ins[0]), np.asarray(ins[1])
        ga, gb = np.meshgrid(a, b, indexing="ij")
        return [ga.astype(np.float32), gb.astype(np.float32)]

    if ov == "stack":
        if any(v is None for v in ins):
            return None
        arrs = [np.asarray(v) for v in ins]
        axis = next((i for i, d in enumerate(osh) if d == len(arrs)), -1) if osh else -1
        return [np.stack(arrs, axis=axis)]

    if ov == "reshape":
        if ins and ins[0] is not None and osh:
            return [np.asarray(ins[0]).reshape(osh)]
        return None

    if ov == "aten::full":
        # 채움값(stride)은 trace 에 없음 → 정사각 grid 가정으로 stride = img_h / feat_h.
        if not osh:
            return None
        n = int(np.prod(osh))
        feat = int(round(math.isqrt(n))) if n > 0 else 0
        stride = float(img_h) / feat if feat else 0.0
        return [np.full(osh, stride, dtype=np.float32)]

    if ov == "concat":
        if any(v is None for v in ins) or not osh:
            return None
        arrs = [np.asarray(v) for v in ins]
        axis = _infer_concat_axis(osh, [a.shape for a in arrs])
        return [np.concatenate(arrs, axis=axis)]

    if ov == "transpose":
        if not ins or ins[0] is None or not osh:
            return None
        a = np.asarray(ins[0])
        perm = _infer_perm(list(a.shape), list(osh))
        return [np.transpose(a, perm) if perm else a.T]

    if ov == "unsqueeze":
        if ins and ins[0] is not None and osh:
            return [np.asarray(ins[0]).reshape(osh)]
        return None

    if ov == "shape":
        return None  # non-tensor (정적 정수) — 소비처는 out_shape 사용

    return None  # 평가 불가 → 상수 아님으로 취급


def fold_constants(graph, img_hw):
    """입력 무관 상수 subgraph 를 평가해 baked/skip 맵을 만든다.

    img_hw: (H, W) 입력 해상도 — `full` stride 유도용.
    """
    img_h = int(img_hw[0])
    nodes = list(getattr(graph, "nodes", []))

    # 1) taint: INPUT 출력에서 도달 가능한 노드 = 입력 의존(=非상수).
    #    ★ `shape` 는 taint 배리어: 입력 의존 텐서에서 차원을 뽑지만, 입력 해상도 고정이
    #    이 codegen 의 전제라 그 차원값은 정적 상수다 → 출력을 tainted 로 보지 않는다.
    tainted_t = set()   # id(tensor)
    for n in nodes:
        ov = op_value(n)
        if ov == "shape":
            continue  # 배리어: shape 출력은 정적 상수
        ins = [t for t in (n.in_tensors or []) if t is not None]
        dep_input = (ov == "input") or any(id(t) in tainted_t for t in ins)
        if dep_input:
            for t in (n.out_tensors or []):
                if t is not None:
                    tainted_t.add(id(t))

    # 2) 상수 노드 평가(topo 순서 = graph.nodes 순서).
    vals = {}           # id(tensor) -> np.ndarray
    const_nodes = []    # 평가 성공한 상수 노드(경계/내부 판정용)
    for n in nodes:
        ov = op_value(n)
        if ov in ("input", "return"):
            continue
        outs = [t for t in (n.out_tensors or [])]
        # 입력 의존 출력이 하나라도 있으면 비상수
        if any(t is not None and id(t) in tainted_t for t in outs):
            continue
        res = _eval_node(n, vals, img_h)
        if res is None:
            continue
        ok = True
        for t, v in zip(outs, res):
            if t is not None and v is not None:
                vals[id(t)] = v
            elif t is not None:
                ok = False
        if ok:
            const_nodes.append(n)

    # 3) 경계 텐서: 상수값을 가지면서 tainted 노드의 입력으로 쓰이는 텐서 → bake.
    consumed_by_tainted = set()
    crossing = set()    # 평가 여부와 무관하게 상수→tainted 경계를 넘는 텐서
    for n in nodes:
        outs = [t for t in (n.out_tensors or [])]
        if any(t is not None and id(t) in tainted_t for t in outs):  # tainted 노드
            for t in (n.in_tensors or []):
                if t is None or id(t) in tainted_t:
                    continue
                crossing.add(id(t))
                if id(t) in vals:
                    consumed_by_tainted.add(id(t))

    baked = {}
    counter = {"n": 0}

    def _key(arr):
        counter["n"] += 1
        return f"const.fold{counter['n']}"

    tid_to_key = {}
    for n in const_nodes:
        for t in (n.out_tensors or []):
            if t is None:
                continue
            if id(t) in consumed_by_tainted and id(t) not in baked:
                arr = np.ascontiguousarray(vals[id(t)], dtype=np.float32)
                key = _key(arr)
                baked[id(t)] = (key, arr)
                tid_to_key[id(t)] = key

    # 4) skip: 상수영역 노드(출력이 하나도 tainted 가 아님) 는 codegen 에서 전부 제거.
    #    경계 텐서는 baked 로 따로 로드(m.weights)하고, 그 외 상수 출력은 상수영역
    #    내부에서만 소비되므로(=tainted 가 참조하는 상수는 모두 baked 됨) 안전하게 제거.
    #    단, 평가에 실패해 bake 되지 못한 **텐서** 출력이 경계를 넘어가면 제거할 수 없다 —
    #    소비처가 바인딩을 잃고 조용히 다른 텐서로 대체된다. 그래프에 남겨 render 로 emit.
    #    스칼라(shape 산술 등)는 소비처가 정적 값으로 쓰므로 그대로 제거한다.
    def _is_tensor(t):
        try:
            return len(t.shape) > 0
        except Exception:
            return False

    skip = set()
    for n in nodes:
        ov = op_value(n)
        if ov in ("input", "return"):
            continue
        outs = [t for t in (n.out_tensors or [])]
        if not outs or any(t is not None and id(t) in tainted_t for t in outs):
            continue
        if any(t is not None and id(t) in crossing and id(t) not in baked and _is_tensor(t)
               for t in outs):
            continue
        skip.add(id(n))

    return baked, skip
