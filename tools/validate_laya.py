#!/usr/bin/env python3
"""Compare prepared GTX Laya tensors with the original PyTorch checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--laya-source", type=Path, required=True, help="checkout containing the laya package")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.laya_source.resolve()))
    from laya.agent import Agent

    with args.batch.open(encoding="utf-8") as f:
        batch = json.load(f)
    row = batch["rows"][args.row]
    root = args.batch.parent
    ids = np.fromfile(root / row["tokens"], dtype=np.int32).astype(np.int64)
    markers = np.fromfile(root / row["markers"], dtype=np.int32).astype(np.int64)

    agent = Agent(str(args.checkpoint), compile=False, device="cpu")
    model = agent.model.eval()
    input_ids = torch.from_numpy(ids)[None, :]
    attention_mask = torch.ones_like(input_ids)
    marker_pos = torch.from_numpy(markers)[None, :]
    marker_mask = torch.ones_like(marker_pos, dtype=torch.bool)
    qtype = torch.tensor([row["qtype"]], dtype=torch.long)
    with torch.no_grad():
        reference_hidden = model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state.float().numpy()[0]
        reference_logits, _ = model(
            input_ids, attention_mask, marker_pos, marker_mask, qtype
        )
        reference_logits = reference_logits.float().numpy()[0]

    actual_hidden = np.fromfile(args.hidden, dtype=np.float32).reshape(len(ids), -1)
    actual_logits = np.fromfile(args.logits, dtype=np.float32)
    if actual_hidden.shape != reference_hidden.shape:
        raise ValueError(f"hidden shape mismatch: {actual_hidden.shape} != {reference_hidden.shape}")
    if actual_logits.shape != reference_logits.shape:
        raise ValueError(f"logit shape mismatch: {actual_logits.shape} != {reference_logits.shape}")

    report = {
        "hidden_cosine": _cosine(reference_hidden, actual_hidden),
        "hidden_max_abs": float(np.max(np.abs(reference_hidden - actual_hidden))),
        "logits_cosine": _cosine(reference_logits, actual_logits),
        "logits_max_abs": float(np.max(np.abs(reference_logits - actual_logits))),
        "reference_logits": reference_logits.tolist(),
        "actual_logits": actual_logits.tolist(),
        "argmax_equal": int(reference_logits.argmax()) == int(actual_logits.argmax()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
