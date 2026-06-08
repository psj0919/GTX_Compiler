"""Validate the expanded lazy (whole-graph) op set: lazy == eager == torch.

Builds a small MobileNet-ish chain exercising the newly added lazy builders
(silu, depthwise conv, relu6, avgpool, elemwise mul) and checks that the fused
single-graph result matches both eager ggml and a torch reference.

    uv run --no-sync python tools/ggml_lazy_ops.py
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


class TinyNet(nn.QuantModel):
    def __init__(self):
        super().__init__()
        self.module_0 = nn.Module("input")
        self.module_1 = nn.Module("conv2d", stride=[1, 1], padding=[1, 1])
        self.module_2 = nn.Module("silu")
        self.module_3 = nn.Module("depthwise_conv2d", stride=[1, 1], padding=[1, 1])
        self.module_4 = nn.Module("relu6")
        self.module_5 = nn.Module("avgpool", kernel_size=[2, 2], stride=[2, 2], padding=[0, 0])
        self.module_6 = nn.Module("adaptive_avg_pool2d", output_size=[1, 1])
        self.module_7 = nn.Module("flatten")
        self.module_8 = nn.Module("dense")

    @nn.forward_processor
    def forward(self, *args):
        o = self.module_0(input=args[0])
        o = self.module_1(o)
        o = self.module_2(o)
        o = self.module_3(o)
        o = self.module_4(o)
        o = self.module_5(o)
        o = self.module_6(o)
        o = self.module_7(input=o, start_dim=1, end_dim=-1)
        o = self.module_8(o)
        return o


def main():
    np.random.seed(0)
    C, OC = 3, 8
    w1 = np.random.randn(OC, C, 3, 3).astype(np.float32)
    b1 = np.random.randn(OC).astype(np.float32)
    wdw = np.random.randn(OC, 1, 3, 3).astype(np.float32)
    bdw = np.random.randn(OC).astype(np.float32)
    wfc = np.random.randn(4, OC).astype(np.float32)
    bfc = np.random.randn(4).astype(np.float32)
    x = np.random.randn(1, C, 8, 8).astype(np.float32)
    tx = torch.from_numpy(x)

    # torch reference
    with torch.no_grad():
        t = F.conv2d(tx, torch.from_numpy(w1), torch.from_numpy(b1), padding=1)
        t = F.silu(t)
        t = F.conv2d(t, torch.from_numpy(wdw), torch.from_numpy(bdw), padding=1, groups=OC)
        t = F.hardtanh(t, 0, 6)
        t = F.avg_pool2d(t, 2, 2)
        t = F.adaptive_avg_pool2d(t, 1)
        t = torch.flatten(t, 1)
        t = t @ torch.from_numpy(wfc).T + torch.from_numpy(bfc)
        ref = t.numpy()

    def build_model():
        m = TinyNet()
        m.module_1.weights = {"w": w1, "b": b1}
        m.module_3.weights = {"w": wdw, "b": bdw}
        m.module_8.weights = {"w": wfc, "b": bfc}
        return m

    nn.set_backend("ggml")
    nn.set_dtype("fp32")

    eager = np.asarray(build_model()(x)).reshape(ref.shape)
    lazy = np.asarray(nn.lazy_forward(build_model(), x)).reshape(ref.shape)

    e_err = np.abs(eager - ref).max()
    l_err = np.abs(lazy - ref).max()
    el_err = np.abs(eager - lazy).max()
    print(f"eager vs torch : max_err {e_err:.3e}")
    print(f"lazy  vs torch : max_err {l_err:.3e}")
    print(f"lazy  vs eager : max_err {el_err:.3e}")
    ok = e_err < 1e-3 and l_err < 1e-3
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
