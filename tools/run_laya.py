#!/usr/bin/env python3
"""Run every prepared Laya question through vision.cpp and decode answers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from laya_io import decode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--runner-dir", type=Path, help="Unified build bin directory (defaults to model-dir)")
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.batch.open(encoding="utf-8") as f:
        batch = json.load(f)
    runner_dir = args.runner_dir or args.model_dir
    encoder = (runner_dir / "run_laya_encoder").resolve()
    head = (runner_dir / "run_laya_head").resolve()
    encoder_gguf = args.model_dir / "laya-encoder.gguf"
    head_gguf = args.model_dir / "laya-head.gguf"
    for path in (encoder, head, encoder_gguf, head_gguf):
        if not path.exists():
            raise FileNotFoundError(path)

    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    root = args.batch.parent
    for row in batch["rows"]:
        hidden = root / f"hidden-{row['index']:03d}.bin"
        subprocess.run(
            [
                str(encoder), str(encoder_gguf), str(root / row["tokens"]),
                str(row["token_count"]), str(hidden),
            ],
            check=True,
            env=env,
        )
        subprocess.run(
            [
                str(head), str(head_gguf), str(hidden), str(row["token_count"]),
                str(row["qtype"]), str(root / row["markers"]),
                str(row["marker_count"]), str(root / row["logits"]), "768",
            ],
            check=True,
            env=env,
        )
    decode(args.batch, args.output)


if __name__ == "__main__":
    main()
