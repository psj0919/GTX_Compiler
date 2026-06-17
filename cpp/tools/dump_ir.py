#!/usr/bin/env python
"""Python TorchParser 의 Graph IR 을 정규화 JSON 으로 덤프 — C++ 파서 parity ground truth.

C++ gtxc-parse 의 IR JSON 과 동일 스키마로 출력하여 1:1 대조한다.
스키마(노드 리스트, topo 순):
  {"model","input_shape","nodes":[
     {"idx","name","scope_name","op_type",
      "in_nodes":[...], "out_nodes":[...],
      "in_tensors":[{"name","shape","dtype"}...],
      "out_tensors":[...],
      "params":{pname:{"name","shape","dtype"}...},
      "configs":{cname: value...},
      "attrs":{aname: value...}}]}

사용:
  python cpp/tools/dump_ir.py --model resnet18 --out cpp/assets/resnet18.ir.py.json
"""
import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _modelkit import build, default_input_shape  # noqa: E402


def tensor_desc(t):
    return {"name": t.name, "shape": list(t.shape) if t.shape else None, "dtype": t.dtype}


def jsonable(v):
    """numpy/torch/enum 값을 JSON 직렬화 가능 형태로."""
    import numpy as np
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return {"__ndarray__": True, "shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    # Tensor (shared.graph) — 파라미터가 config 에 들어간 경우
    if hasattr(v, "name") and hasattr(v, "shape") and hasattr(v, "dtype"):
        return {"__tensor__": v.name, "shape": list(v.shape) if v.shape else None}
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def op_params(op):
    out = {}
    for name, tensor in op.params.items():
        key = name if isinstance(name, str) else name.value
        if isinstance(tensor, list):
            out[key] = [tensor_desc(t) for t in tensor]
        else:
            out[key] = tensor_desc(tensor)
    return out


def op_configs(op):
    out = {}
    for cname in op.configs:
        try:
            out[cname] = jsonable(op.get_config(cname))
        except Exception as e:  # noqa
            out[cname] = f"<err:{e}>"
    return out


def op_attrs(op):
    out = {}
    for name, attr in op.attrs.items():
        key = name.value if hasattr(name, "value") else str(name)
        try:
            out[key] = jsonable(attr.value)
        except Exception as e:  # noqa
            out[key] = f"<err:{e}>"
    return out


def node_desc(node):
    op = node.op
    return {
        "idx": node.idx,
        "name": node.name,
        "scope_name": node.scope_name,
        "op_type": op.type,
        "in_nodes": list(node.in_nodes),
        "out_nodes": list(node.out_nodes),
        "in_tensors": [tensor_desc(t) for t in node.in_tensors],
        "out_tensors": [tensor_desc(t) for t in node.out_tensors],
        "params": op_params(op),
        "configs": op_configs(op),
        "attrs": op_attrs(op),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-shape", default=None)
    args = ap.parse_args()

    shape = (tuple(int(s) for s in args.input_shape.split(","))
             if args.input_shape else default_input_shape(args.model))
    model = build(args.model)  # randomize 포함(C++ 와 동일 가중치)
    inputs = torch.randn(*shape)

    from parse import TorchParser
    from parse.rich_in_out_helper import StandardInputData

    graph = TorchParser()(args.model, model, StandardInputData((inputs,), {}))

    nodes = [node_desc(n) for n in sorted(graph.nodes, key=lambda n: n.idx)]
    doc = {"model": args.model, "input_shape": list(shape), "nodes": nodes}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"[dump_ir] {len(nodes)} nodes → {args.out}")
    # op_type 히스토그램(빠른 점검)
    from collections import Counter
    hist = Counter(n["op_type"] for n in nodes)
    for k, v in sorted(hist.items()):
        print(f"    {v:4d}  {k}")


if __name__ == "__main__":
    main()
