#!/usr/bin/env python3
"""Export laya-multilingual as a ModernBERT encoder and a GTX decision head.

The encoder remains a standard llama.cpp ``modern-bert`` GGUF.  The custom
Laya tensors are written to a second GGUF consumed by ``tools/run_laya_head``.
This keeps the vendored llama.cpp unmodified and gives both graphs to the same
GGML backend used by vision.cpp.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


DEFAULT_REPO = "convaiinnovations/laya-multilingual"
DEFAULT_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
EXPECTED_ENCODER = "jhu-clsp/mmBERT-base"


def _require_laya_dependencies():
    try:
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise SystemExit(
            "Laya export dependencies are missing. Run `uv sync --extra laya`."
        ) from exc
    return snapshot_download, safe_open, save_file


def _checkpoint_path(checkpoint: str, revision: str | None, local_files_only: bool) -> Path:
    path = Path(checkpoint)
    if path.is_dir():
        return path.resolve()
    snapshot_download, _, _ = _require_laya_dependencies()
    return Path(
        snapshot_download(
            checkpoint,
            revision=revision,
            local_files_only=local_files_only,
            allow_patterns=[
                "model.safetensors",
                "rl_agent_config.json",
                "encoder/config.json",
                "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json",
            ],
        )
    )


def _load_configs(root: Path) -> tuple[dict, dict]:
    with (root / "rl_agent_config.json").open(encoding="utf-8") as f:
        laya = json.load(f)
    with (root / "encoder" / "config.json").open(encoding="utf-8") as f:
        encoder = json.load(f)
    if laya.get("encoder") != EXPECTED_ENCODER:
        raise ValueError(
            f"expected multilingual encoder {EXPECTED_ENCODER!r}, got {laya.get('encoder')!r}"
        )
    if encoder.get("model_type") != "modernbert":
        raise ValueError(f"expected modernbert encoder, got {encoder.get('model_type')!r}")
    if int(encoder.get("hidden_size", 0)) != 768 or int(encoder.get("num_attention_heads", 0)) != 12:
        raise ValueError("unsupported mmBERT shape; expected hidden_size=768 and 12 heads")
    if int(laya.get("head_layers", 0)) != 2:
        raise ValueError("unsupported Laya head; expected exactly two transformer layers")
    return laya, encoder


def _head_key(source: str) -> str:
    # vision.cpp's split_qkv helper expects the packed projection directly at
    # <module>.weight/.bias and its output projection below .out_proj.
    return source.replace(".self_attn.in_proj_weight", ".self_attn.weight").replace(
        ".self_attn.in_proj_bias", ".self_attn.bias"
    )


def _write_head_gguf(source: Path, output: Path, laya_cfg: dict, encoder_cfg: dict) -> int:
    _, safe_open, _ = _require_laya_dependencies()
    gguf_py = Path(__file__).parents[2] / "vision.cpp" / "depend" / "llama" / "gguf-py"
    sys.path.insert(0, str(gguf_py))
    import gguf  # type: ignore

    writer = gguf.GGUFWriter(str(output), arch="laya-head")
    writer.add_string("laya.encoder", laya_cfg["encoder"])
    writer.add_uint32("laya.context_length", int(laya_cfg.get("max_len", 1024)))
    writer.add_uint32("laya.head_length", int(laya_cfg.get("head_max_len", 256)))
    writer.add_uint32("laya.embedding_length", int(encoder_cfg["hidden_size"]))
    writer.add_uint32("laya.attention.head_count", int(encoder_cfg["num_attention_heads"]))
    writer.add_uint32("laya.block_count", int(laya_cfg["head_layers"]))
    temperatures = laya_cfg.get("temperature", [1.0, 1.0, 1.0])
    writer.add_array("laya.temperature", [float(v) for v in temperatures])

    count = 0
    with safe_open(str(source), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("encoder.") or key == "temperature":
                continue
            tensor = f.get_tensor(key).detach().cpu()
            arr = tensor.float().numpy() if not tensor.is_floating_point() else tensor.numpy()
            writer.add_tensor(_head_key(key), np.ascontiguousarray(arr).astype(np.float16))
            count += 1
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return count


def _make_encoder_view(source_root: Path, target: Path) -> int:
    _, safe_open, save_file = _require_laya_dependencies()
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / "encoder" / "config.json", target / "config.json")
    shutil.copy2(source_root / "tokenizer" / "tokenizer.json", target / "tokenizer.json")
    shutil.copy2(source_root / "tokenizer" / "tokenizer_config.json", target / "tokenizer_config.json")

    encoder = {}
    with safe_open(str(source_root / "model.safetensors"), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("encoder."):
                encoder[key[len("encoder.") :]] = f.get_tensor(key).contiguous()
    save_file(encoder, str(target / "model.safetensors"))
    return len(encoder)


def export_laya(
    checkpoint: str,
    output_dir: str,
    revision: str | None = DEFAULT_REVISION,
    local_files_only: bool = False,
    keep_encoder_view: bool = False,
) -> tuple[Path, Path]:
    root = _checkpoint_path(checkpoint, revision, local_files_only)
    laya_cfg, encoder_cfg = _load_configs(root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_weights = root / "model.safetensors"
    if not source_weights.is_file():
        raise FileNotFoundError(source_weights)

    head_path = output / "laya-head.gguf"
    n_head = _write_head_gguf(source_weights, head_path, laya_cfg, encoder_cfg)

    owned_tmp = None
    if keep_encoder_view:
        encoder_view = output / "encoder-hf"
        if encoder_view.exists():
            shutil.rmtree(encoder_view)
    else:
        owned_tmp = tempfile.TemporaryDirectory(prefix="g2c-laya-")
        encoder_view = Path(owned_tmp.name) / "encoder"
    n_encoder = _make_encoder_view(root, encoder_view)

    converter = Path(__file__).parents[2] / "tools" / "laya_convert_encoder.py"
    encoder_path = output / "laya-encoder.gguf"
    subprocess.run(
        [sys.executable, str(converter), str(encoder_view), "--outfile", str(encoder_path), "--outtype", "f16"],
        check=True,
    )
    if owned_tmp is not None:
        owned_tmp.cleanup()

    shutil.copy2(root / "rl_agent_config.json", output / "rl_agent_config.json")
    shutil.copytree(root / "tokenizer", output / "tokenizer", dirs_exist_ok=True)
    manifest = {
        "source": checkpoint,
        "revision": revision,
        "encoder": encoder_path.name,
        "head": head_path.name,
        "encoder_tensors": n_encoder,
        "head_tensors": n_head,
        "max_len": int(laya_cfg.get("max_len", 1024)),
        "head_max_len": int(laya_cfg.get("head_max_len", 256)),
        "hidden_size": int(encoder_cfg["hidden_size"]),
        "attention_heads": int(encoder_cfg["num_attention_heads"]),
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"encoder: {encoder_path} ({n_encoder} tensors)")
    print(f"head:    {head_path} ({n_head} tensors)")
    return encoder_path, head_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_REPO, help="HF repo id or local checkpoint directory")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="pinned HF revision")
    parser.add_argument("--output", default="output/laya-multilingual")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--keep-encoder-view", action="store_true")
    args = parser.parse_args(argv)
    export_laya(
        args.checkpoint,
        args.output,
        revision=args.revision or None,
        local_files_only=args.local_files_only,
        keep_encoder_view=args.keep_encoder_view,
    )


if __name__ == "__main__":
    main()
