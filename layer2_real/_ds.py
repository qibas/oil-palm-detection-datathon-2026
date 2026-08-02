"""Dataset switch for `layer2_real/`.

Prefers Agent 1's real `dataset.py` (the contract implementation over the frozen
`../data_clean/layer2_*.csv`). Falls back to `_stub_dataset.py` — SYNTHETIC — so
the model/harness work could proceed in parallel.

`IS_REAL` is threaded into every printed banner and into every row of
`results_real.csv`, so a stub run can never be mistaken for a finding.
"""
import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def get():
    """-> (module, is_real)."""
    if os.path.exists(os.path.join(_HERE, "dataset.py")):
        try:
            m = importlib.import_module("dataset")
            missing = [f for f in ("load", "census", "node_features", "adjacency",
                                   "build_examples", "folds") if not hasattr(m, f)]
            if missing:
                raise ImportError("dataset.py is missing contract functions: " + ", ".join(missing))
            return m, True
        except Exception as exc:                       # noqa: BLE001 - report, do not hide
            print(f"[_ds] real dataset.py present but unusable ({exc!r}); falling back to STUB",
                  flush=True)
    return importlib.import_module("_stub_dataset"), False


DS, IS_REAL = get()
TAG = "REAL" if IS_REAL else "STUB"

WINDOW = int(getattr(DS, "WINDOW", 3))
HORIZONS = tuple(getattr(DS, "HORIZONS", (1, 2, 3, 4)))
N_REL = int(getattr(DS, "N_REL", 1))
