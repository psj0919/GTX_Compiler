"""qproc package

Lightweight __init__ to avoid heavy side-effect imports.
Submodules should be imported explicitly, e.g.:
  from qproc.export import get_script_writer
  from qproc.utils import prepare_quantizable_module
"""

__all__ = [
    "export",
    "utils",
    "base",
    "rnn",
    "adaquant",
    "onnx",
]
