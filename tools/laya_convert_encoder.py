#!/usr/bin/env python3
"""Run llama.cpp's HF converter for mmBERT without patching the submodule.

The bundled converter recognizes answerdotai/ModernBERT's tokenizer hash but
not mmBERT's Metaspace tokenizer hash.  GTX Laya tokenizes on the host and
passes token IDs directly, so llama.cpp's pre-tokenizer is not used at runtime.
The vocabulary and IDs still come from mmBERT; only the metadata enum is set to
the existing ModernBERT value so conversion can proceed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).parents[1]
    converter = root / "vision.cpp" / "depend" / "llama" / "convert_hf_to_gguf.py"
    spec = importlib.util.spec_from_file_location("g2c_llama_convert_hf", converter)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {converter}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # ModernBertModel inherits through BertModel/TextModel, whose class body
    # carries the generated function independently of ModelBase in this llama
    # revision. Override the concrete converter class to avoid relying on MRO.
    module.ModernBertModel.get_vocab_base_pre = lambda self, tokenizer: "modern-bert"
    module.main()


if __name__ == "__main__":
    main()
