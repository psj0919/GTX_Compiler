"""v10 (300,6) C++ top-k vs torch 비교. 행은 score 내림차순 정렬이라 행별 정합.
usage: yolo_v10_cmp.py <cpp_out.bin> <torch_ref.bin>"""
import sys
import numpy as np

o = np.fromfile(sys.argv[1], dtype=np.float32)
r = np.fromfile(sys.argv[2], dtype=np.float32)
no, nr = o.size // 6, r.size // 6
O = o[:no * 6].reshape(no, 6)
R = r[:nr * 6].reshape(nr, 6)
n = min(no, nr)
O, R = O[:n], R[:n]
print(f"cpp dets={no} torch dets={nr} compared={n}")

box_o, box_r = O[:, :4], R[:, :4]
conf_o, conf_r = O[:, 4], R[:, 4]
lab_o, lab_r = O[:, 5].astype(int), R[:, 5].astype(int)


def cos(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0


print(f"box   cos={cos(box_o, box_r):.6f}  max|diff|={np.abs(box_o-box_r).max():.4f}")
print(f"conf  cos={cos(conf_o, conf_r):.6f}  max|diff|={np.abs(conf_o-conf_r).max():.5f}")
print(f"label match(all)={int((lab_o==lab_r).sum())}/{n} ({100*(lab_o==lab_r).mean():.1f}%)")


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# 의미있는 지표: conf>thr 인 torch 검출을 cpp 검출과 IoU 매칭.
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
tr = R[R[:, 4] > thr]
ious, lab_ok = [], 0
for d in tr:
    best = max(range(no), key=lambda j: iou(d[:4], O[j, :4]))
    ious.append(iou(d[:4], O[best, :4]))
    lab_ok += int(int(O[best, 5]) == int(d[5]))
if len(tr):
    print(f"[conf>{thr}] torch dets={len(tr)}  matched mean_IoU={np.mean(ious):.4f}  "
          f"min_IoU={np.min(ious):.4f}  label match={lab_ok}/{len(tr)}")
# 상위 5개 검출 표
print("  cpp[top5 x1y1x2y2 conf lab] vs torch:")
for i in range(min(5, n)):
    print(f"   #{i} cpp={O[i,:4].round(1).tolist()} c={O[i,4]:.3f} l={int(O[i,5])}"
          f"  | torch={R[i,:4].round(1).tolist()} c={R[i,4]:.3f} l={int(R[i,5])}")
