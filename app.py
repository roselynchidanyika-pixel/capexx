"""CapEx AI Agent - deployment entry point (Streamlit Cloud / repo root).

The verified application lives in ``ui/app.py``. Streamlit Cloud launches the
file named ``app.py`` at the repo root, so this thin shim only has to:
  1. put the repository root on ``sys.path`` (so ``import core.data_model`` and
     ``import ui.app`` resolve regardless of the working directory), and
  2. execute ``ui/app.py`` (the real, versioned application) as __main__.

All model, finance, risk and UI logic stays in the verified ``core/`` and
``ui/`` packages; nothing is re-implemented here.
"""
from __future__ import annotations

import os
import runpy
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_REAL_APP = os.path.join(_ROOT, "ui", "app.py")

if __name__ == "__main__" or __name__ == "__mp_main__":
    runpy.run_path(_REAL_APP, run_name="__main__")
