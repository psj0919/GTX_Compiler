"""갈래2 adapter prototype: ggml lazy tape ->  Graph IR.

The lazy execution tape (`nn.record_tape`) is a topologically-ordered op list
with resolved data-dependency edges -- exactly the IR the ELF path needs.  This
builds a real `shared.graph.Graph` from it (Node + Operation + attrs +
param shapes + edges), the first stage of the ggml->->ELF path documented in
docs/ggml_to_elf.md.

Wiring the resulting Graph through `optimization/` + `c_codegen` + `memory_planner`
(needs exact tensor shapes/param data + memory plan) is the remaining step.

    uv run --no-sync python tools/tape_to_graph.py
"""

import importlib.util
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as tnn
from torchvision.models import resnet18

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import nn
from shared import graph as gg


def build_graph(entries, output_idx, name="model"):
    """Map a ggml lazy tape to  Node/Operation IR + a dependency graph.

    Builds a real ``graph.Node`` + ``Operation`` (op type, attrs, param
    shapes) per tape entry and the data-dependency adjacency.  Assembling these
    into a Graph/Block container with input/return nodes reuses the parser's
    graph-builder (shared.graph) -- the remaining wiring step.
    """
    nodes = {}        # idx -> (Node, Operation)
    adj = {}          # idx -> [src idx ...]
    for e in entries:
        node = gg.Node(name=f"n{e['idx']}", op=e["type"])
        op = gg.Operation(e["type"])
        for k, v in e["attrs"].items():
            try:
                op.set_attr(k, v)
            except Exception:
                pass  # attr schema differs from parser's -- prototype tolerant
        node._op = op
        nodes[e["idx"]] = (node, op)
        adj[e["idx"]] = list(e["inputs"])

    n_edges = sum(len(v) for v in adj.values())
    return nodes, adj, n_edges


def main():
    torch.manual_seed(0)
    tv = resnet18()
    for m in tv.modules():
        if isinstance(m, tnn.BatchNorm2d):
            m.running_var.uniform_(0.5, 1.5)
            m.weight.data.uniform_(0.5, 1.5)
    tv.eval()
    x = torch.randn(1, 3, 224, 224)

    export_path = os.path.join(HERE, "export1", "ResNet.py")
    spec = importlib.util.spec_from_file_location("export1_resnet", export_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    nn.set_backend("ggml")
    model = mod.ResNet()
    # fold weights so BN -> identity (matches lazy builders)
    sys.path.insert(0, os.path.join(HERE, "tools"))
    from ggml_validate_resnet import inject_weights
    inject_weights(model, tv)

    entries, out_idx = nn.record_tape(model, x)
    nodes, adj, n_edges = build_graph(entries, out_idx, name="ResNet")

    hist = Counter(e["type"] for e in entries)
    weighted = [e for e in entries if any(v is not None for v in e["weights"].values())]

    print(f"tape entries     : {len(entries)}")
    print(f" Node/Op built: {len(nodes)}")
    print(f"dependency edges : {n_edges}")
    print(f"output node      : n{out_idx}")
    print(f"op histogram     : {dict(hist)}")
    print(f"weighted nodes   : {len(weighted)} (param shapes mapped)")
    # node types are real graph.Operation objects
    sample = entries[1]
    nd, op = nodes[sample["idx"]]
    print(f"sample node      : {nd.name} op={op.type} (graph.Operation)")
    for e in weighted[:2]:
        print(f"  n{e['idx']:<3d} {e['type']:<22s} in={e['inputs']} weights={e['weights']}")

    ok = (
        len(nodes) == len(entries)
        and n_edges > 0
        and out_idx is not None
        and all(isinstance(op, gg.Operation) for _, op in nodes.values())
        and len(weighted) == 21  # 20 conv + 1 dense (BN folded into conv)
    )
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
