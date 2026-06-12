"""ggml_quant_conv_proof.py — conv-as-matmul 양자화 설계의 수치 증명(엔진 무수정).

배경: ggml 의 conv 는 im2col + mul_mat 로 분해되고, mul_mat 은 양자 가중치를 소비한다.
즉 conv 커널 [OC,IC,KH,KW] 를 2D [OC, IC*KH*KW] 로 보면 Linear 가중치와 똑같이 ggml
블록 양자화(Q8_0/Q4_0 …)가 가능하다. 이 스크립트는 vision.cpp 엔진을 고치기 전에 그
설계가 수치적으로 옳은지 파이썬에서 증명한다:

  증명1) im2col+matmul == F.conv2d            (재구성 정확성, max_abs_err≈0)
  증명2) 자격 conv/linear 를 양자화→복원했을 때 모델 출력 보존 (cosine, top-1)

자격: conv 는 row=IC*KH*KW 가 블록(32)의 배수, linear 는 in_features 가 블록의 배수.
(ggml mul_mat 은 양자 행렬의 연속 차원=row 를 블록 단위로 양자화 → 위 row 가 그 축)

  uv run python tools/ggml_quant_conv_proof.py --model resnet18 --type q8_0
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

import gguf.quants as quants
from gguf import GGMLQuantizationType

_QT = {
    "q8_0": GGMLQuantizationType.Q8_0,
    "q4_0": GGMLQuantizationType.Q4_0,
    "q5_0": GGMLQuantizationType.Q5_0,
    "q4_1": GGMLQuantizationType.Q4_1,
    "q5_1": GGMLQuantizationType.Q5_1,
}
_BLOCK = 32


def _qdq(w2d: np.ndarray, qt) -> np.ndarray:
    """numpy 2D [OC, row] → ggml quantize→dequantize (저장 양자화 효과 시뮬)."""
    qb = quants.quantize(np.ascontiguousarray(w2d, np.float32), qt)
    return quants.dequantize(qb, qt).astype(np.float32)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    return float(a @ b / (a.norm() * b.norm()))


def prove_reconstruction() -> None:
    """conv 커널을 2D 로 reshape 해 mul_mat 하면 F.conv2d 와 동일함을 보인다."""
    torch.manual_seed(0)
    OC, IC, KH, KW, stride, pad = 128, 64, 3, 3, 1, 1
    W = torch.randn(OC, IC, KH, KW)
    x = torch.randn(1, IC, 16, 16)
    ref = F.conv2d(x, W, stride=stride, padding=pad)
    cols = F.unfold(x, (KH, KW), stride=stride, padding=pad)        # [1, IC*KH*KW, L]
    out = (W.reshape(OC, IC * KH * KW) @ cols).reshape(1, OC, *ref.shape[2:])
    err = (out - ref).abs().max().item()
    print(f"[증명1] im2col+matmul vs F.conv2d : max_abs_err={err:.2e} "
          f"→ conv 커널 2D[OC,IC*KH*KW]=({OC},{IC * KH * KW}) 로 mul_mat = conv")


def prove_model(model_name: str, qname: str) -> None:
    """자격 conv/linear 를 양자화→복원 후 모델 출력 보존을 측정."""
    qt = _QT[qname]
    m = getattr(torchvision.models, model_name)(weights="DEFAULT").eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        base = m(x)

    mq = getattr(torchvision.models, model_name)().eval()
    mq.load_state_dict(m.state_dict())
    n_q = n_tot = 0
    skipped = []
    with torch.no_grad():
        for name, p in mq.named_parameters():
            if not name.endswith(".weight"):
                continue
            w = p.data
            if w.ndim == 4:                       # conv: row=IC*KH*KW
                n_tot += 1
                row = int(np.prod(w.shape[1:]))
                if row % _BLOCK == 0:
                    dq = _qdq(w.reshape(w.shape[0], row).numpy(), qt)
                    p.data = torch.from_numpy(dq).reshape(w.shape)
                    n_q += 1
                else:
                    skipped.append(name)
            elif w.ndim == 2 and w.shape[1] % _BLOCK == 0:   # linear
                n_tot += 1
                p.data = torch.from_numpy(_qdq(w.numpy(), qt))
                n_q += 1
        outq = mq(x)
    top1 = "일치" if base.argmax() == outq.argmax() else "불일치"
    print(f"[증명2 {qname}] {model_name}: conv+linear {n_q}/{n_tot} 양자화 "
          f"(skip={[s.split('.')[0] for s in skipped]})  "
          f"출력 cosine={_cos(base, outq):.5f}  top1={top1}")


def main() -> None:
    ap = argparse.ArgumentParser(description="conv-as-matmul 양자화 수치 증명")
    ap.add_argument("--model", default="resnet18", help="torchvision 모델명")
    ap.add_argument("--type", default="q8_0", choices=sorted(_QT), help="양자화 타입")
    args = ap.parse_args()
    prove_reconstruction()
    prove_model(args.model, args.type)


if __name__ == "__main__":
    main()
