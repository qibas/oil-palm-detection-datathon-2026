"""Minimal example: load stgnn_final.pt from other code and score palms.

    python example_inference.py

This is the whole integration surface. A downstream demo, service or notebook
needs exactly four things:

    stgnn_final.pt        the weights + metadata
    data_clean/*.csv      the three frozen CSVs
    dataset.py            features, adjacency, risk set
    models_real.py        the architecture

Nothing else — not run_real.py, not train_final.py, not this file. Copy the
`score_palms()` body into your own code and it will work.

READ BEFORE USING THE OUTPUT
    The return value is a RELATIVE score for ranking. It is not a probability and
    the model is not calibrated. Rank palms by it, colour maps by its quantiles,
    and never present sigmoid(score) as a percentage. The checkpoint carries the
    full caveat in its `scope_warning` field; print it if you are unsure.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import dataset as ds          # noqa: E402
import models_real as M       # noqa: E402


def score_palms(ckpt_path, census_idx=None):
    """-> (palm_ids, scores, census_t). One score per palm in the risk set.

    palm_ids  (n,) str    identifiers, aligned with `scores`
    scores    (n,) float  relative risk logits; higher = higher risk
    census_t  float       the census date scored, in years
    """
    # 1. load the checkpoint
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    h = ck["task"]["horizon"]

    # 2. rebuild the architecture from the stored metadata and load the weights
    model = M.build(ck["model_class"], ck["arch"]["in_dim"], horizon=h)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()

    # 3. rebuild the inputs from the frozen CSVs
    T = len(ds.census())
    F = np.asarray(ds.node_features(np.arange(T)), np.float32)   # (T, N, d)
    A = np.asarray(ds.adjacency("true"), np.float32)             # (N, N)
    D = ds.diffuse(F, A)                                         # (T, N, 1, d)

    # 4. pick the census and build the risk set for it
    t = ck["probe"]["census_idx"] if census_idx is None else int(census_idx)
    tree, t_idx, _ = ds.build_examples(h, np.array([t]))
    F_seq, D_seq = ds.make_windows(tree, t_idx, F, D)             # [B,W,d], [B,W,1,d]

    # 5. forward pass
    with torch.no_grad():
        scores = model(torch.as_tensor(F_seq), torch.as_tensor(D_seq)).numpy()

    ids = np.asarray(ck["data"]["palm_ids"])[tree]   # palm_ids maps row index -> id
    return ids, scores, float(ds.census()[t])


if __name__ == "__main__":
    ids, scores, t = score_palms(os.path.join(HERE, "stgnn_final.pt"))

    print(f"scored {len(ids)} at-risk palms at t = {t:.1f} yr")
    print(f"scores: shape {scores.shape}, dtype {scores.dtype}, "
          f"range {scores.min():+.4f} .. {scores.max():+.4f}\n")

    top = np.argsort(-scores)[:5]
    print("highest-risk palms (ranking only — NOT a probability):")
    for r, i in enumerate(top, 1):
        pct = 100.0 * (scores < scores[i]).sum() / (len(scores) - 1)
        print(f"  {r}. {ids[i]:<10} score {scores[i]:+.4f}   {pct:.1f}th percentile")
