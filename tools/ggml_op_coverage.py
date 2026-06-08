"""Per-op coverage test for the ggml backend, in fp32 and fp16 modes.

    uv run --no-sync python tools/ggml_op_coverage.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import nn


def run(dtype):
    nn.set_backend("ggml")
    nn.set_dtype(dtype)
    tol = 3e-2 if dtype == "fp16" else 1e-3
    np.random.seed(0)
    results = []

    def chk(name, ggml_out, ref):
        g = np.asarray(ggml_out).astype(np.float32)
        r = np.asarray(ref).astype(np.float32)
        ok = g.shape == r.shape and np.allclose(g, r, atol=tol, rtol=tol)
        err = np.abs(g - r).max() if g.shape == r.shape else float("nan")
        results.append((name, ok, err))

    x = np.random.randn(1, 8, 16, 16).astype(np.float32)
    y = np.random.randn(1, 8, 16, 16).astype(np.float32)
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)

    # activations
    chk("relu", nn.Module("relu")(x), F.relu(tx).numpy())
    chk("relu6", nn.Module("relu6")(x), F.hardtanh(tx, 0, 6).numpy())
    chk("leaky_relu", nn.Module("leaky_relu", negative_slope=0.1)(x),
        F.leaky_relu(tx, 0.1).numpy())
    chk("sigmoid", nn.Module("sigmoid")(x), torch.sigmoid(tx).numpy())
    chk("tanh", nn.Module("tanh")(x), torch.tanh(tx).numpy())
    chk("gelu", nn.Module("GELU")(x), F.gelu(tx, approximate="tanh").numpy())
    chk("silu", nn.Module("silu")(x), F.silu(tx).numpy())
    chk("hardsigmoid", nn.Module("hardsigmoid")(x), F.hardsigmoid(tx).numpy())
    chk("hardswish", nn.Module("hardswish")(x), F.hardswish(tx).numpy())

    # binary
    chk("add", nn.Module("elemwise_add")(input=x, other=y, alpha=1),
        (tx + ty).numpy())
    chk("sub", nn.Module("elementwise_sub")(input=x, other=y, alpha=1),
        (tx - ty).numpy())
    chk("mul", nn.Module("elemwise_mul")(input=x, other=y), (tx * ty).numpy())
    chk("div", nn.Module("elemwise_div")(input=x, other=y), (tx / ty).numpy())

    # shape / movement
    chk("flatten", nn.Module("flatten")(input=x, start_dim=1, end_dim=-1),
        torch.flatten(tx, 1).numpy())
    chk("concat", nn.Module("concat", dim=1)(tensors=[x, y]),
        torch.cat([tx, ty], 1).numpy())

    # pooling
    chk("maxpool", nn.Module("maxpool", kernel_size=[2, 2], stride=[2, 2],
                             padding=[0, 0])(x),
        F.max_pool2d(tx, 2, 2).numpy())
    chk("avgpool", nn.Module("avgpool", kernel_size=[2, 2], stride=[2, 2],
                             padding=[0, 0])(x),
        F.avg_pool2d(tx, 2, 2).numpy())
    chk("adaptive_avg", nn.Module("adaptive_avg_pool2d", output_size=[1, 1])(x),
        F.adaptive_avg_pool2d(tx, 1).numpy())

    # depthwise conv
    C = 8
    wdw = np.random.randn(C, 1, 3, 3).astype(np.float32)
    bdw = np.random.randn(C).astype(np.float32)
    m = nn.Module("depthwise_conv2d", stride=[1, 1], padding=[1, 1])
    m.weights = {"w": wdw, "b": bdw}
    chk("dwconv", m(x),
        F.conv2d(tx, torch.from_numpy(wdw), torch.from_numpy(bdw),
                 stride=1, padding=1, groups=C).numpy())

    # dense
    win = np.random.randn(10, 8 * 16 * 16).astype(np.float32)
    bin_ = np.random.randn(10).astype(np.float32)
    md = nn.Module("dense")
    md.weights = {"w": win, "b": bin_}
    xf = x.reshape(1, -1)
    chk("dense", md(xf),
        (torch.from_numpy(xf) @ torch.from_numpy(win).T + torch.from_numpy(bin_)).numpy())

    # group_norm / instance_norm
    gw = np.random.randn(C).astype(np.float32)
    gb = np.random.randn(C).astype(np.float32)
    mg = nn.Module("group_norm", num_groups=2)
    mg.weights = {"w": gw, "b": gb}
    chk("group_norm", mg(x),
        F.group_norm(tx, 2, torch.from_numpy(gw), torch.from_numpy(gb)).numpy())
    mi = nn.Module("instance_norm")
    mi.weights = {"w": gw, "b": gb}
    chk("instance_norm", mi(x),
        F.instance_norm(tx, weight=torch.from_numpy(gw), bias=torch.from_numpy(gb)).numpy())

    # conv_transpose_2d (padding 0)
    IC, OC = 8, 6
    wt = np.random.randn(IC, OC, 2, 2).astype(np.float32)
    bt = np.random.randn(OC).astype(np.float32)
    mct = nn.Module("conv_transpose_2d", stride=[2, 2], padding=[0, 0])
    mct.weights = {"w": wt, "b": bt}
    chk("conv_transpose", mct(x),
        F.conv_transpose2d(tx, torch.from_numpy(wt), torch.from_numpy(bt), stride=2).numpy())

    print(f"\n=== dtype={dtype} (tol={tol}) ===")
    npass = 0
    for name, ok, err in results:
        print(f"  {name:14s} {'PASS' if ok else 'FAIL':4s}  max_err={err:.3e}")
        npass += ok
    print(f"  {npass}/{len(results)} passed")
    return npass == len(results)


def main():
    ok32 = run("fp32")
    ok16 = run("fp16")
    print(f"\nRESULT: {'PASS' if ok32 and ok16 else 'FAIL'}")
    return 0 if (ok32 and ok16) else 1


if __name__ == "__main__":
    sys.exit(main())
