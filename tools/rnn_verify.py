"""RNN(LSTM/GRU) eager ggml 백엔드 ↔ PyTorch cosine 검증.

사용: python tools/rnn_verify.py output/lstm1  [--gru]
  output/<dir>/<Name>.py + .gguf 를 eager 실행하고, 같은 GGUF weight 를 PyTorch
  nn.LSTM/GRU 에 주입해 동일 입력으로 비교한다.
"""
import os
import sys
import glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nn as _nn  # noqa: E402


def _load_gguf(path):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gp = os.path.join(here, "vision.cpp", "depend", "llama", "gguf-py")
    if gp not in sys.path:
        sys.path.insert(0, gp)
    import gguf
    r = gguf.GGUFReader(path)
    return {t.name: np.array(t.data, dtype=np.float32).reshape(tuple(reversed(t.shape)))
            for t in r.tensors}


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    out_dir = sys.argv[1]
    is_gru = "--gru" in sys.argv
    py = glob.glob(os.path.join(out_dir, "*.py"))[0]
    gguf_path = glob.glob(os.path.join(out_dir, "*.gguf"))[0]
    name = os.path.splitext(os.path.basename(py))[0]

    tensors = _load_gguf(gguf_path)

    # 모델 형상 추론 (weight_ih_l0: [gate*H, in], weight_hh_l0: [gate*H, H])
    wih0 = tensors["weight_ih_l0"]
    gate = 3 if is_gru else 4
    H = wih0.shape[0] // gate
    in_size = wih0.shape[1]
    num_layers = sum(1 for k in tensors if k.startswith("weight_ih_l") and "_reverse" not in k)
    bidir = any("_reverse" in k for k in tensors)
    has_bias = any(k.startswith("bias_ih_l") for k in tensors)

    # 입력
    rng = np.random.RandomState(0)
    T = 8
    x = rng.randn(1, T, in_size).astype("float32")

    # eager (ggml 백엔드) — export .py 의 모듈 로드해 run_gguf
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, py)
    mod = importlib.util.module_from_spec(spec)
    _nn.set_backend("ggml")
    spec.loader.exec_module(mod)
    ModelCls = getattr(mod, name)
    eager_out = _nn.run_gguf(ModelCls(), gguf_path, py, x)
    eager_y = np.asarray(eager_out[0] if isinstance(eager_out, (list, tuple)) else eager_out)

    # PyTorch reference
    Cls = torch.nn.GRU if is_gru else torch.nn.LSTM
    ref = Cls(in_size, H, num_layers=num_layers, batch_first=True,
              bidirectional=bidir, bias=has_bias).eval()
    sd = {}
    for k, v in tensors.items():
        sd[k] = torch.from_numpy(v.copy())
    missing, unexpected = ref.load_state_dict(sd, strict=False)
    with torch.no_grad():
        ref_y, _ = ref(torch.from_numpy(x))
    ref_y = ref_y.numpy()

    c = _cos(eager_y, ref_y)
    md = float(np.max(np.abs(eager_y.ravel() - ref_y.ravel())))
    print(f"[{name}] eager{eager_y.shape} vs torch{ref_y.shape}  cos={c:.6f}  max|diff|={md:.3e}")
    print(f"  layers={num_layers} bidir={bidir} bias={has_bias} H={H} in={in_size}")
    if missing or unexpected:
        print(f"  state_dict missing={missing} unexpected={unexpected}")
    print("PASS" if c > 0.9999 else "FAIL")


if __name__ == "__main__":
    main()
