"""RNN(LSTM/GRU) 생성 C++(vision.cpp 빌드) ↔ PyTorch cosine 검증.

사용: python tools/rnn_cpp_verify.py output/lstm1 [--gru]
전제: tools/build_rnn_cpp.sh 로 output/<dir>/run_rnn_cpp 가 빌드돼 있어야 함.
"""
import os
import sys
import glob
import subprocess
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_gguf(path):
    gp = os.path.join(ROOT, "vision.cpp", "depend", "llama", "gguf-py")
    if gp not in sys.path:
        sys.path.insert(0, gp)
    import gguf
    r = gguf.GGUFReader(path)
    return {t.name: np.array(t.data, dtype=np.float32).reshape(tuple(reversed(t.shape)))
            for t in r.tensors}


def _cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    out_dir = sys.argv[1]
    is_gru = "--gru" in sys.argv
    gguf_path = glob.glob(os.path.join(out_dir, "*.gguf"))[0]
    runner = os.path.join(out_dir, "run_rnn_cpp")
    assert os.path.exists(runner), f"빌드 필요: bash tools/build_rnn_cpp.sh {out_dir}"

    tensors = _load_gguf(gguf_path)
    gate = 3 if is_gru else 4
    H = tensors["weight_ih_l0"].shape[0] // gate
    in_size = tensors["weight_ih_l0"].shape[1]
    num_layers = sum(1 for k in tensors if k.startswith("weight_ih_l") and "_reverse" not in k)
    bidir = any("_reverse" in k for k in tensors)
    has_bias = any(k.startswith("bias_ih_l") for k in tensors)

    rng = np.random.RandomState(0)
    T, B = 8, 1
    x = rng.randn(B, T, in_size).astype("float32")  # [batch, seq, feat]

    # PyTorch reference
    Cls = torch.nn.GRU if is_gru else torch.nn.LSTM
    ref = Cls(in_size, H, num_layers=num_layers, batch_first=True,
              bidirectional=bidir, bias=has_bias).eval()
    ref.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in tensors.items()},
                        strict=False)
    with torch.no_grad():
        ref_y, _ = ref(torch.from_numpy(x))
    ref_y = ref_y.numpy()  # [B, T, H*dir]

    # C++ 실행: input.bin(= torch contiguous ravel == ggml [feat,seq,batch] 메모리)
    in_bin = os.path.join(out_dir, "rnn_in.bin")
    out_bin = os.path.join(out_dir, "rnn_out.bin")
    x.astype("float32").ravel().tofile(in_bin)
    r = subprocess.run([runner, gguf_path, in_bin, out_bin,
                        str(in_size), str(T), str(B)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr); print("FAIL (run)"); return
    cpp_y = np.fromfile(out_bin, dtype=np.float32)
    ref_flat = ref_y.ravel()
    if cpp_y.size != ref_flat.size:
        print(f"size mismatch cpp={cpp_y.size} ref={ref_flat.size}"); print(r.stdout)
        print("FAIL (size)"); return

    c = _cos(cpp_y, ref_flat)
    md = float(np.max(np.abs(cpp_y - ref_flat)))
    print(f"[{os.path.basename(out_dir)}] cpp vs torch  cos={c:.6f}  max|diff|={md:.3e}"
          f"  (layers={num_layers} bidir={bidir} H={H} in={in_size})")
    print("PASS" if c > 0.9999 else "FAIL")


if __name__ == "__main__":
    main()
