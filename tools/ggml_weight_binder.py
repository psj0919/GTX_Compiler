"""Name-based weight binding for the ggml backend (arch-agnostic).

The generated export carries each op's traced module path in a comment:

    self.module_5 = nn.Module('conv2d', ...) #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv1]/ret.5()

We parse that path into a state_dict prefix (``layer1.0.conv1``) and bind each
weighted / norm module directly from a torch ``state_dict`` by its own name --
no Conv-BN structural pairing, so it generalises to YOLO and other non-ResNet
graphs.  BatchNorm is executed as a real affine op (see
``ggml_backend._op_batch_norm``) rather than folded, which keeps binding purely
name-driven.

    uv run --no-sync python tools/ggml_weight_binder.py     # self-test on resnet18
"""

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

_INIT_RE = re.compile(
    r"self\.module_(\d+)\s*=\s*nn\.Module\('([^']+)'.*?\)\s*#[^:]+::(.+?)\(\)\s*$"
)
_BRACKET_RE = re.compile(r"\w+\[([\w.]+)\]")


def node_path_to_key(node_path):
    """'ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv1]/ret.5' -> 'layer1.0.conv1'."""
    return ".".join(_BRACKET_RE.findall(node_path))


def parse_export(export_path):
    """Return {module_idx: (type, state_dict_prefix)} from the export file."""
    info = {}
    with open(export_path) as f:
        for line in f:
            m = _INIT_RE.search(line.strip())
            if m:
                idx, type, node_path = int(m.group(1)), m.group(2), m.group(3)
                info[idx] = (type, node_path_to_key(node_path))
    return info


_WEIGHT_OPS = (
    "conv2d",
    "depthwise_conv2d",
    "conv_transpose_2d",
    "dense",
)
_NORM_OPS = ("group_norm", "instance_norm")


def bind_from_state_dict(ggml_model, state_dict, export_path):
    """Inject weights into a ggml-backed export model from a torch state_dict."""
    info = parse_export(export_path)

    def arr(key):
        t = state_dict.get(key)
        return None if t is None else t.detach().cpu().numpy().astype(np.float32)

    bound, missing = 0, []
    for idx, (type, prefix) in info.items():
        mod = getattr(ggml_model, f"module_{idx}", None)
        if mod is None or not hasattr(mod, "weights"):
            continue
        if type in _WEIGHT_OPS:
            w = arr(f"{prefix}.weight")
            if w is None:
                missing.append(f"{prefix}.weight ({type})")
                continue
            mod.weights = {"w": w}
            b = arr(f"{prefix}.bias")
            if b is not None:
                mod.weights["b"] = b
            bound += 1
        elif type == "batch_norm":
            if arr(f"{prefix}.running_mean") is None:
                missing.append(f"{prefix}.running_mean")
                continue
            mod.weights = {
                "w": arr(f"{prefix}.weight"),
                "b": arr(f"{prefix}.bias"),
                "mean": arr(f"{prefix}.running_mean"),
                "var": arr(f"{prefix}.running_var"),
            }
            bound += 1
        elif type in _NORM_OPS:
            w = arr(f"{prefix}.weight")
            if w is not None:
                mod.weights = {"w": w, "b": arr(f"{prefix}.bias")}
            bound += 1
    return bound, missing


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _selftest():
    import importlib.util
    import torch
    import torch.nn as tnn
    from torchvision.models import resnet18
    import nn

    torch.manual_seed(0)
    tv = resnet18()
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

    export_path = os.path.join(HERE, "export1", "ResNet.py")
    spec = importlib.util.spec_from_file_location("export1_resnet", export_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    nn.set_backend("ggml")
    nn.set_dtype("fp32")
    model = mod.ResNet()
    n, _missing = bind_from_state_dict(model, tv.state_dict(), export_path)
    out = np.asarray(model(x)).reshape(ref.shape)

    rel = np.abs(out - ref).max() / (np.abs(ref).max() + 1e-9)
    ok = int(out.argmax()) == int(ref.argmax()) and rel < 1e-2
    print(f"bound {n} modules by name")
    print(f"rel_err {rel:.3e}  argmax ggml {int(out.argmax())} ref {int(ref.argmax())}")
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest())
