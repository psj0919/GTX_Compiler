"""to_dot 의 record 라벨이 Netron source/dot.js 규칙으로 파싱되는지 자체검증.

torch 트레이스 없이 fake Graph IR 로 to_dot 을 돌리고, dot.js(92-160행)의
label→type/attr 승격 규칙을 그대로 재현해 op type·파라미터가 살아나는지 assert.
  uv run python tools/test_dot_netron.py   # OK 출력 = 통과
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from tools.graph_visualizer import to_dot


class _Attr:
    def __init__(self, v): self.value = v


class _Name:
    def __init__(self, v): self.value = v


class _Op:
    def __init__(self, type, attrs):
        self.type = type
        self._attrs = {_Name(k): _Attr(v) for k, v in attrs.items()}


class _T:
    def __init__(self, shape): self.shape = shape


class _Node:
    def __init__(self, name, op, out_shape, out_nodes):
        self.name = name
        self.op = op
        self.out_tensors = [_T(out_shape)]
        self.out_nodes = out_nodes


class _Graph:
    name = "g"
    def __init__(self, nodes): self.nodes = nodes


def _parse_like_netron(nid, label):
    """dot.js 규칙 재현: (type, attrs). type==nid 면 id 폴백(=실패)."""
    ty = None; attrs = {}
    if label.startswith('{') and label.endswith('}'):
        lines = label[1:-1].split('|')
        if len(lines) > 1 and nid == lines[0] and lines[1].startswith('op_code='):
            ty = lines[1].split('\\l')[0].split('=').pop()
            if len(lines) > 2:
                for a in lines[2].split('\\l'):
                    p = a.split(':')
                    if len(p) == 2:
                        k, v = p[0].strip(), p[1].strip()
                        if v.startswith('(') and v.endswith(')'):
                            v = json.loads('[' + v[1:-1] + ']')
                        attrs[k] = v
    return (ty or nid), attrs


def main():
    # conv → add(두 입력: conv, skip)  — residual 분기 포함.
    conv = _Node("m::conv", _Op("conv2d", {
        "kernel": [3, 3], "stride": [1, 1], "pad": [1, 1, 1, 1],
        "bias_term": False, "in_dim": 8, "out_dim": 8,
        "weight": _T([8]),          # Tensor attr → 표시 제외돼야 함
    }), [1, 8, 4, 4], [])
    skip = _Node("m::skip", _Op("relu", {}), [1, 8, 4, 4], [])
    add = _Node("m::add", _Op("elemwise_add", {}), [1, 8, 4, 4], [])
    conv.out_nodes = [add.name]
    skip.out_nodes = [add.name]
    dot = to_dot(_Graph([conv, skip, add]))

    got = {}
    for line in dot.splitlines():
        m = re.match(r'\s+(n\d+) \[label="(.*?)", fillcolor', line)
        if m:
            got[m.group(1)] = _parse_like_netron(m.group(1), m.group(2))

    # 전부 op type 으로 승격 (id 폴백 0).
    fell = [nid for nid, (ty, _) in got.items() if ty == nid]
    assert not fell, f"id 폴백: {fell}"
    # conv type + 파라미터 필드(튜플→배열), Tensor attr 은 빠짐.
    ty, at = got["n0"]
    assert ty == "conv2d", ty
    assert at["kernel"] == [3, 3] and at["pad"] == [1, 1, 1, 1], at
    assert at["out_shape"] == [1, 8, 4, 4], at
    assert "weight" not in at, "Tensor attr 이 새어나감"
    # 엣지 = residual 두 입력.
    ins = {}
    for line in dot.splitlines():
        e = re.match(r'\s+(n\d+) -> (n\d+);', line)
        if e:
            ins.setdefault(e.group(2), []).append(e.group(1))
    add_id = next(nid for nid, (ty, _) in got.items() if ty == "elemwise_add")
    assert len(ins[add_id]) == 2, ins
    print("OK — record 라벨이 dot.js 규칙으로 type+attr 승격, residual 엣지 정상.")


if __name__ == "__main__":
    main()
