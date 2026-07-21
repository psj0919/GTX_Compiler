"""TODO.md A/B/C 그룹 신규 op 의 ggml codegen 스모크 테스트.

각 그룹의 op 을 모은 모델을 컴파일하고, 생성 .cpp 에
 - `TODO(ggml): unhandled op` 가 없고
 - raw `aten::` op 타입이 남지 않았고 (= dispatcher 가 전부 받았고)
 - 각 op 이 기대하는 ggml 빌더 호출로 emit 되는지
를 확인한다.

    uv run --no-sync python tools/ggml_new_ops_smoke.py
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn
import torchvision

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

torch.set_num_threads(1)


class NewOps(nn.Module):
    """A+B 그룹: mul_ / sin / cos / pow / rsub / new_ones / zeros / fill_ /
    linspace / roll / type_as."""

    def forward(self, x):
        x = x.mul_(1.5)                                   # aten::mul_
        x = x + torch.sin(x) + torch.cos(x)               # aten::sin / aten::cos
        x = x + torch.pow(x, 3)                           # aten::pow  (지수 != 2)
        x = x * torch.pow(x, 2)                           # aten::pow  -> OP.SQUARE
        x = 1.0 - x                                       # aten::rsub (스칼라)
        x = x + x.new_ones(x.shape)                       # aten::new_ones
        x = x + torch.zeros(x.shape)                      # aten::zeros
        x = x + torch.empty(x.shape).fill_(0.25)          # aten::fill_
        x = x + torch.linspace(0.0, 1.0, x.shape[-1])     # aten::linspace
        x = x + torch.roll(x, shifts=1, dims=-1)          # aten::roll
        return x.type_as(x)                               # aten::type_as


class CmpOps(nn.Module):
    """C 그룹: 비교(gt/ne/eq) · 논리(__or__) · 마스크 선택(where/masked_fill)."""

    def forward(self, x):
        gt = x > 0.0                                      # aten::gt (스칼라 other)
        ne = x != torch.sin(x)                            # aten::ne (텐서 other)
        both = gt | ne                                    # aten::__or__
        y = torch.where(both, x, -x)                      # aten::where
        y = y.masked_fill(gt, float("-inf"))              # aten::masked_fill
        return y + (x == x).float()                       # aten::eq


class DeformOps(nn.Module):
    """D 그룹: grid_sample(ggml 커널 부재 → custom op) · deformable conv(ggml_conv_2d_deform)."""

    def __init__(self):
        super().__init__()
        self.dcn = torchvision.ops.DeformConv2d(3, 4, 3, padding=1)
        self.off = nn.Conv2d(3, 2 * 3 * 3, 3, padding=1)
        self.msk = nn.Conv2d(3, 3 * 3, 3, padding=1)

    def forward(self, x):
        y = self.dcn(x, self.off(x), self.msk(x).sigmoid())      # torchvision::deform_conv2d
        grid = x[:, :2].permute(0, 2, 3, 1).tanh()               # 입력 의존 grid (N,H,W,2)
        return torch.nn.functional.grid_sample(y, grid, align_corners=False)


# 모델 -> {검사 이름: 생성 .cpp 에 반드시 나와야 하는 문자열}
CASES = [
    (NewOps, {
        "mul_": "ggml_mul(",
        "sin": "ggml_sin(",
        "cos": "ggml_cos(",
        "pow(x,3)": "ggml_mul(",
        "pow(x,2)": "ggml_sqr(",
        "rsub": "ggml_scale(m, ",        # other 는 상수텐서 → other + (-alpha*x)
        "new_ones/zeros/fill_": "ggml_fill(",
        "linspace": "ggml_arange(",
        "roll": "ggml_roll(",
        # type_as 는 단일 dtype 그래프에서 no-op(라인 없음)이 정답 → aten:: 누출 검사로 확인.
    }),
    (CmpOps, {
        "gt (스칼라)": "ggml_step(m, ggml_scale_bias(",
        "ne (텐서)": "ggml_step(m, ggml_abs(",
        "eq": "ggml_scale_bias(m, ggml_step(m, ggml_abs(",
        "__or__": "ggml_step(m, ggml_add(",
        "where": "ggml_add(m, ggml_mul(",
        "masked_fill(-inf)": "-1e+30f",  # 0*inf=NaN 회피용 유한 대체값
    }),
    (DeformOps, {
        "grid_sample 호출": "grid_sample_2d(m, ",
        "grid_sample 커널": "ggml_custom_4d(m, GGML_TYPE_F32",
        "deform_conv2d": "ggml_conv_2d_deform(m, ",
    }),
]


def check(cls, expect, out_dir):
    from shared.compile.pipeline import compile_model

    name = cls.__name__
    compile_model(cls().eval(), name, (1, 3, 8, 8), out_dir, opt_level=0)
    src_path = os.path.join(out_dir, f"{name}.cpp")
    if not os.path.exists(src_path):
        print(f"  {name}: FAIL — {name}.cpp 가 생성되지 않았다")
        return False
    with open(src_path, encoding="utf8") as f:
        src = f.read()

    ok = True
    print(f"\n=== {name} ===")
    for label, needle in expect.items():
        hit = needle in src
        ok &= hit
        print(f"  {label:22s} {'PASS' if hit else 'FAIL':4s}  ({needle})")

    unhandled = [l.strip() for l in src.splitlines() if "unhandled op" in l]
    ok &= not unhandled
    print(f"  {'unhandled op':22s} {'PASS' if not unhandled else 'FAIL':4s}  ({len(unhandled)}건)")
    for l in unhandled:
        print(f"    {l}")

    # dispatcher 에 def 가 없으면 op.type 이 raw schema 이름('aten::type_as')으로 남는다.
    leaked = sorted({w for w in src.split() if w.startswith("aten::")})
    ok &= not leaked
    print(f"  {'aten:: 누출':21s} {'PASS' if not leaked else 'FAIL':4s}  ({leaked or '없음'})")
    return ok


def main():
    with tempfile.TemporaryDirectory() as tmp:      # SMOKE_OUT=<dir> 로 산출물 보존 가능
        out = os.environ.get("SMOKE_OUT") or tmp
        ok = all([check(cls, expect, out) for cls, expect in CASES])

    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
