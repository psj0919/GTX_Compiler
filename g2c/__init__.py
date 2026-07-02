"""g2c — the GTX-to-C compiler toolkit, as one importable root.

The toolkit's packages (``parse``, ``shared``, ``qproc``, ``quantization``,
``utils``, ``nn``) historically live at the repo root and are imported top-level
(``from parse.parser import TorchParser``).  That works, but there is no single
name that says "this is the compiler".  This package is that name:

    import g2c
    from g2c.parse.parser import TorchParser        # same modules, one root
    parser = g2c.TorchParser()                       # common entry, re-exported

It is a *thin alias* — no files were moved.  On import it puts the toolkit root
on ``sys.path`` and re-exposes the existing top-level packages under ``g2c.*``,
so both the old (``from parse...``) and new (``from g2c.parse...``) imports work.
Deeper modules (e.g. ``g2c.parse.parser``) load on demand via the real package's
search path.
"""
import importlib
import os
import sys

# the toolkit root (this file lives at <root>/g2c/__init__.py)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# the toolkit's top-level packages, re-exposed under the g2c.* namespace
_SUBPACKAGES = ("parse", "shared", "qproc", "quantization", "utils", "nn")
for _name in _SUBPACKAGES:
    try:
        _mod = importlib.import_module(_name)
    except Exception:                       # optional / heavy deps absent -> skip
        continue
    sys.modules[f"{__name__}.{_name}"] = _mod   # makes `import g2c.<name>` resolve
    globals()[_name] = _mod

# convenience: the most common entry point at the root
try:
    from parse.parser import TorchParser     # noqa: F401  (re-export as g2c.TorchParser)
except Exception:
    TorchParser = None

__all__ = [*_SUBPACKAGES, "TorchParser"]
