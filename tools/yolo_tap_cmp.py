"""중간 텐서 탭 노드별 비교 (C++ ggml 그래프 vs eager 백엔드 레퍼런스).
첫 갈라지는 노드를 자동 특정 → 그 op 의 C++ emission 버그.
usage: yolo_tap_cmp.py <cpp_tap_dir> <ref_tap_dir> [cos_thresh=0.99]
"""
import os
import re
import sys
import numpy as np

cpp_dir, ref_dir = sys.argv[1], sys.argv[2]
THR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.99

# ref manifest: tapN <op_type> <shape>
optype = {}
for line in open(os.path.join(ref_dir, "manifest.txt")):
    p = line.split(maxsplit=2)
    if len(p) >= 2:
        optype[p[0]] = p[1]


def nodes(d):
    return {f[:-4] for f in os.listdir(d) if re.fullmatch(r"tap\d+\.bin", f)}


common = nodes(cpp_dir) & nodes(ref_dir)
order = sorted(common, key=lambda t: int(t[3:]))
print(f"common taps: {len(common)} (cpp {len(nodes(cpp_dir))}, ref {len(nodes(ref_dir))})")


def cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 1.0


first_bad = None
rows = []
for t in order:
    a = np.fromfile(os.path.join(cpp_dir, t + ".bin"), dtype=np.float32)
    b = np.fromfile(os.path.join(ref_dir, t + ".bin"), dtype=np.float32)
    if a.size != b.size:
        rows.append((t, optype.get(t, "?"), float("nan"), -1, a.size, b.size))
        if first_bad is None:
            first_bad = t
        continue
    c = cos(a, b)
    md = float(np.abs(a - b).max())
    rows.append((t, optype.get(t, "?"), c, md, a.size, b.size))
    if c < THR and first_bad is None:
        first_bad = t

# 요약: 나쁜 노드만(있으면) + 처음/끝 몇 개
bad = [r for r in rows if not (r[2] >= THR)]
print(f"\n{'node':>8} {'op':<16} {'cos':>10} {'max|diff|':>10}  size")
for r in (bad if bad else rows[:8]):
    cs = f"{r[2]:.6f}" if r[2] == r[2] else "SIZE!"
    print(f"{r[0]:>8} {r[1]:<16} {cs:>10} {r[3]:>10.4f}  {r[4]}vs{r[5]}")

if first_bad:
    fb = next(r for r in rows if r[0] == first_bad)
    print(f"\n>>> FIRST DIVERGENCE: {first_bad} (op={fb[1]}) cos={fb[2]:.6f} "
          f"size {fb[4]}vs{fb[5]} — C++ emission of this op is likely wrong.")
else:
    print(f"\n>>> ALL {len(order)} taps agree (cos≥{THR}). C++ graph matches eager backend.")
