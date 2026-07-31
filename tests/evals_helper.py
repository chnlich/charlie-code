"""Shared loader so the evals plain-scripts are importable from the test suite.

`evals/` is intentionally not an installable package; each script is loaded by
path so the unit tests can exercise its functions without touching the network.
"""

import importlib.util
from pathlib import Path

_EVALS = Path(__file__).resolve().parent.parent / "evals"


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, _EVALS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
