"""export_traced.py / dump_ir.py 공유: 모델 빌드 + 결정적 파라미터 randomize.

randomize 이유: weights=None(기본 init)은 BN weight=running_var=1, bias=running_mean=0 처럼
값이 동일 → C++ freeze_module 의 상수 dedup 이 서로 다른 파라미터를 병합해버린다.
모든 파라미터/버퍼를 distinct 한 값으로 채우면 dedup 이 사라지고, 동일 seed 로 Python/C++
양쪽이 같은 가중치를 갖게 되어 향후 수치 parity 도 가능.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch

torch.set_num_threads(1)


# rnn 스펙: lstm/gru [+ _ml(멀티레이어) / _bi(양방향)]. in=16, hidden=24.
_RNN = {"lstm": False, "gru": True}


def _build_rnn(spec):
    is_gru = spec.startswith("gru")
    layers = 2 if "_ml" in spec else 1
    bidir = "_bi" in spec
    cls = torch.nn.GRU if is_gru else torch.nn.LSTM
    return cls(16, 24, num_layers=layers, batch_first=True, bidirectional=bidir).eval()


def is_yolo(spec: str) -> bool:
    return "yolo" in spec.lower()


def default_input_shape(spec: str):
    if is_yolo(spec):
        return (1, 3, 640, 640)
    if spec.split("_")[0] in ("lstm", "gru"):
        return (1, 8, 16)  # [batch, seq, feat]
    return (1, 3, 224, 224)


def build_model(spec: str):
    if spec.split("_")[0] in ("lstm", "gru"):
        return _build_rnn(spec)
    if is_yolo(spec):
        from ultralytics import YOLO
        return YOLO(spec).model.cpu().eval()
    if spec in ("resnet18", "resnet34", "resnet50"):
        import torchvision.models as M
        return getattr(M, spec)(weights=None).eval()
    import torchvision  # noqa: F401
    obj = eval(spec, {"torch": torch, "torchvision": torchvision})  # noqa: S307
    return (obj() if isinstance(obj, type) else obj).eval()


def randomize_params(model, seed: int = 1234):
    """모든 float 파라미터/버퍼를 distinct 한 값으로 채움(결정적)."""
    g = torch.Generator().manual_seed(seed)
    sd = model.state_dict()
    for k, v in sd.items():
        if not torch.is_floating_point(v):
            continue
        if "running_var" in k:
            v.copy_(torch.rand(v.shape, generator=g) + 0.5)  # 양수
        else:
            v.copy_(torch.randn(v.shape, generator=g))
    return model


def build(spec: str, seed: int = 1234):
    model = build_model(spec)
    # yolo 는 pretrained(가중치 distinct) → randomize 시 stride 버퍼 등 head 의미가 바뀌므로 보존.
    if is_yolo(spec):
        return model
    return randomize_params(model, seed)
