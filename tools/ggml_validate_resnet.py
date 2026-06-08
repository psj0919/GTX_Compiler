"""Validate the ggml execution backend against PyTorch for ResNet-18.

Runs the generated export (``export1/ResNet.py``) with the ggml backend
(``nn.set_backend('ggml')``), injecting Conv-BN-folded weights from a reference
torchvision ResNet-18, and compares the output logits against the same model
run in PyTorch.

    uv run --no-sync python tools/ggml_validate_resnet.py
"""

import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn as tnn
from torchvision.models import resnet18

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import nn


def load_export_model():
    path = os.path.join(HERE, "export1", "ResNet.py")
    spec = importlib.util.spec_from_file_location("export1_resnet", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ResNet


def fold_conv_bn(conv, bn):
    w = conv.weight.detach().numpy().astype(np.float32)          # (OC,IC,KH,KW)
    bconv = (
        conv.bias.detach().numpy()
        if conv.bias is not None
        else np.zeros(w.shape[0], dtype=np.float32)
    )
    gamma = bn.weight.detach().numpy()
    beta = bn.bias.detach().numpy()
    mean = bn.running_mean.detach().numpy()
    var = bn.running_var.detach().numpy()
    scale = gamma / np.sqrt(var + bn.eps)
    w2 = w * scale[:, None, None, None]
    b2 = (bconv - mean) * scale + beta
    return w2.astype(np.float32), b2.astype(np.float32)


def ordered_ggml_ops(model):
    ops = []
    i = 0
    while hasattr(model, f"module_{i}"):
        ops.append(getattr(model, f"module_{i}"))
        i += 1
    return ops


def inject_weights(ggml_model, tv_model):
    torch_layers = [
        m for m in tv_model.modules()
        if isinstance(m, (tnn.Conv2d, tnn.BatchNorm2d, tnn.Linear))
    ]
    weighted = [
        op for op in ordered_ggml_ops(ggml_model)
        if op.type in ("conv2d", "batch_norm", "dense")
    ]
    assert len(torch_layers) == len(weighted), (
        f"layer count mismatch: torch={len(torch_layers)} ggml={len(weighted)}"
    )

    pending_conv = None
    for tl, op in zip(torch_layers, weighted):
        if isinstance(tl, tnn.Conv2d):
            assert op.type == "conv2d", (tl, op.type)
            pending_conv = (tl, op)
        elif isinstance(tl, tnn.BatchNorm2d):
            assert op.type == "batch_norm", (tl, op.type)
            conv_tl, conv_op = pending_conv
            w2, b2 = fold_conv_bn(conv_tl, tl)
            conv_op.weights = {"w": w2, "b": b2}
            pending_conv = None
        elif isinstance(tl, tnn.Linear):
            assert op.type == "dense", (tl, op.type)
            op.weights = {
                "w": tl.weight.detach().numpy().astype(np.float32),
                "b": tl.bias.detach().numpy().astype(np.float32),
            }


def main():
    torch.manual_seed(0)
    tv = resnet18()
    # randomise BN running stats so Conv-BN folding is non-trivial
    for m in tv.modules():
        if isinstance(m, tnn.BatchNorm2d):
            m.running_mean.normal_(0, 1)
            m.running_var.uniform_(0.5, 1.5)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.normal_(0, 0.5)
    tv.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ref = tv(x).numpy()

    import time
    dtype = sys.argv[1] if len(sys.argv) > 1 else "fp32"
    exec_mode = sys.argv[2] if len(sys.argv) > 2 else "eager"
    nn.set_backend("ggml")
    nn.set_dtype(dtype)
    ResNet = load_export_model()
    model = ResNet()
    inject_weights(model, tv)

    t0 = time.time()
    if exec_mode == "lazy":
        out = nn.lazy_forward(model, x)
    else:
        out = np.asarray(model(x))
    dt = time.time() - t0
    out = out.reshape(ref.shape)
    print(f"dtype        : {dtype}")
    print(f"exec         : {exec_mode}  ({dt * 1000:.0f} ms)")

    max_err = np.abs(out - ref).max()
    mean_err = np.abs(out - ref).mean()
    rel = max_err / (np.abs(ref).max() + 1e-9)
    top1_ggml = int(out.argmax())
    top1_ref = int(ref.argmax())
    print(f"output shape : ggml {out.shape}  ref {ref.shape}")
    print(f"max_err      : {max_err:.4e}")
    print(f"mean_err     : {mean_err:.4e}")
    print(f"rel_err      : {rel:.4e}")
    print(f"argmax       : ggml {top1_ggml}  ref {top1_ref}  {'MATCH' if top1_ggml == top1_ref else 'DIFFER'}")
    if dtype in ("fp8", "int8"):
        ok = top1_ggml == top1_ref            # quant: require classification preserved
    else:
        ok = top1_ggml == top1_ref and rel < (5e-2 if dtype == "fp16" else 1e-2)
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
