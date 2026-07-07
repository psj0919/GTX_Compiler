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
    def __init__(self, type, attrs, params=None):
        self.type = type
        self._attrs = {_Name(k): _Attr(v) for k, v in attrs.items()}
        self._params = {_Name(k): v for k, v in (params or {}).items()}


class _Data:
    def __init__(self, dtype, shape): self.dtype = dtype; self.shape = shape


class _T:
    def __init__(self, shape, dtype=None, name=None, has_data=False):
        self.shape = shape
        self.dtype = dtype          # activation dtype (Tensor.dtype 프로퍼티 모사)
        self.name = name
        # param 만 .data(상수 weight) 보유 → 뷰어가 param(f16) vs activation(f32) 구분.
        self.data = _Data(dtype, tuple(shape)) if has_data else None


class _Node:
    def __init__(self, name, op, out_shape, out_nodes, in_tensors=None):
        self.name = name
        self.op = op
        self.in_tensors = in_tensors or []
        self.out_tensors = [_T(out_shape, dtype="float32", name="y")]  # activation out
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
    }, params={                     # op._params → 배포 dtype(f16)[shape] 필드로 표시돼야 함
        "weight": _T([8, 8, 3, 3], dtype="float32", has_data=True),
        "bias": _T([8], dtype="float32", has_data=True),
    }), [1, 8, 4, 4], [],
        in_tensors=[_T([1, 8, 4, 4], dtype="float32", name="x")])  # activation in → in0
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
    # conv type + hyperparameter 필드(튜플→배열).
    ty, at = got["n0"]
    assert ty == "conv2d", ty
    assert at["kernel"] == [3, 3] and at["pad"] == [1, 1, 1, 1], at
    assert at["out_shape"] == [1, 8, 4, 4], at
    # GTX 는 전면 fp16 배포 → weight/bias·activation 모두 float16[shape](torch f32 아님).
    assert at["weight"] == "float16[8, 8, 3, 3]", at.get("weight")
    assert at["bias"] == "float16[8]", at.get("bias")
    # activation in/out 텐서 → in{i}/out{i} 필드(이름 + 배포 dtype f16[shape]).
    assert at["in0"] == "x float16[1, 8, 4, 4]", at.get("in0")
    assert at["out0"] == "y float16[1, 8, 4, 4]", at.get("out0")
    # 엣지 = residual 두 입력.
    ins = {}
    for line in dot.splitlines():
        e = re.match(r'\s+(n\d+) -> (n\d+);', line)
        if e:
            ins.setdefault(e.group(2), []).append(e.group(1))
    add_id = next(nid for nid, (ty, _) in got.items() if ty == "elemwise_add")
    assert len(ins[add_id]) == 2, ins
    print("OK — record 라벨이 dot.js 규칙으로 type+attr+param(dtype[shape]) 승격, residual 엣지 정상.")


if __name__ == "__main__":
    main()
