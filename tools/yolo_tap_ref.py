"""중간 텐서 탭 — eager(ggml) 백엔드 레퍼런스 캡처.
생성 .py 의 module_N(=IR 노드 N) 출력을 forward hook 으로 잡아 tapN.bin 으로 저장.
usage: yolo_tap_ref.py <gen_dir> <input_cwhn.bin> <out_dir>
"""
import os
import re
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import importlib.util
import numpy as np
import torch
torch.set_num_threads(1)

gen_dir, inp_bin, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True)
arch = [f[:-3] for f in os.listdir(gen_dir) if f.endswith(".py")][0]
py = os.path.join(gen_dir, arch + ".py")
gguf = os.path.join(gen_dir, arch + ".gguf")

# C++ 와 동일 입력: CWHN(H,W,C) ravel → (1,3,H,W)
cwhn = np.fromfile(inp_bin, dtype=np.float32)
SZ = int(round((cwhn.size / 3) ** 0.5))
x = cwhn.reshape(SZ, SZ, 3).transpose(2, 0, 1)[None]  # (1,3,H,W)

spec = importlib.util.spec_from_file_location("genmod", py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import nn as _nn

# 모델 인스턴스화는 set_backend('ggml') 이후라야 op_register 가 op→torch_op 맵을 채운다.
# ggml 백엔드에서 module_N 은 GgmlModule(torch.nn.Module 아님) → torch hook 불가.
# GgmlModule.__call__ 을 클래스 패치해 id(instance)→노드 N 매핑으로 출력 캡처.
_nn.set_backend("ggml")
from nn.modules.ggml_backend import GgmlModule

model = mod.__dict__[arch]()
cap = {}


def _np(o):
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().float().numpy()
    return np.asarray(o, dtype=np.float32)


id2n = {}
for k, v in vars(model).items():
    mm = re.fullmatch(r"module_(\d+)", k)
    if mm and isinstance(v, GgmlModule):
        id2n[id(v)] = (int(mm.group(1)), str(v.type))

_orig_call = GgmlModule.__call__


def _patched(self, *a, **k):
    out = _orig_call(self, *a, **k)
    ent = id2n.get(id(self))
    if ent is not None:
        try:
            cap[ent[0]] = (_np(out), ent[1])
        except Exception:
            pass
    return out


GgmlModule.__call__ = _patched
try:
    _nn.run_gguf(model, gguf, py, x)
finally:
    GgmlModule.__call__ = _orig_call

man = open(os.path.join(out_dir, "manifest.txt"), "w")
for N in sorted(cap):
    arr, mtype = cap[N]
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.ravel().tofile(os.path.join(out_dir, f"tap{N}.bin"))
    man.write(f"tap{N} {mtype} {arr.shape}\n")
man.close()
print(f"captured {len(cap)} node taps → {out_dir}")
