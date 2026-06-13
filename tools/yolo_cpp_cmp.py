"""C++ 출력 vs torch 참조 비교(cos + max abs/rel diff). usage: cmp.py <out.bin> <ref.bin>"""
import sys
import numpy as np

out = np.fromfile(sys.argv[1], dtype=np.float32)
ref = np.fromfile(sys.argv[2], dtype=np.float32)
print(f"out={out.size} ref={ref.size}")
n = min(out.size, ref.size)
o, r = out[:n].astype(np.float64), ref[:n].astype(np.float64)


def cos(a, b):
    a = a.ravel(); b = b.ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0


def report(tag, a, b):
    ad = np.abs(a - b)
    rel = ad / (np.abs(b) + 1e-9)
    print(f"{tag:10s} cos={cos(a, b):.6f}  max|diff|={ad.max():.4f}  "
          f"mean|diff|={ad.mean():.5f}  max_rel={rel.max():.4f}")


report("FULL", o, r)
if n % 8400 == 0 and (n // 8400) >= 5:
    C = n // 8400
    O = o.reshape(C, 8400); R = r.reshape(C, 8400)
    report("bbox(0:4)", O[:4], R[:4])
    report("cls(4:)", O[4:], R[4:])
    cc = np.array([cos(O[i], R[i]) for i in range(C)])
    md = np.abs(O - R).max(axis=1)
    worst = np.argsort(cc)[:5]
    print("worst-cos ch:", [(int(i), round(float(cc[i]), 5), round(float(md[i]), 3)) for i in worst])
    print(f"channel with max|diff|: ch{int(md.argmax())} = {md.max():.4f}")
