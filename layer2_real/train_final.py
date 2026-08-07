"""Layer 2 — train ONE final STGNN and freeze it to a checkpoint.

    python train_final.py

This is NOT an experiment. It produces no number that may enter the paper. It
produces one artifact: `stgnn_final.pt`, the inference model the demo loads.

---------------------------------------------------------------------------
How this differs from run_real.py, and why
---------------------------------------------------------------------------
`run_real.py` trains 5 configs x 2 folds x 20 seeds and DISCARDS every model
once its AUC-PR is recorded. That is correct for evaluation, because what is
being measured is the difference between configurations, not the weights of any
one model.

This file trains **one** model over **all 1,200 palms** (both parcels) and saves
it. The consequence is stated up front rather than hidden:

    *** THIS MODEL HAS NO HELD-OUT SET. ***

    Every palm it scores at inference time is a palm it saw during training. No
    performance number (AUC-PR, precision, recall) may be quoted from it.
    Performance numbers come ONLY from run_real.py / run_v2.py, which use
    leave-one-parcel-out. This warning is stored inside the checkpoint as the
    `scope_warning` field, so it cannot be lost when the file changes hands.

Why all 1,200 and not a single fold: this artifact is used to RANK palms within
the same plantation, not to generalise to a different one. A fold model has only
ever seen 480 or 720 palms, so it would score the remaining 720/480 through a
graph that is half unfamiliar. The effect size differs 2.6x between blocks
(44A +0.0084 vs 44B +0.0219), so picking one fold means picking one of two
plantations. For the demo, both must be represented.

---------------------------------------------------------------------------
Reused, not rewritten
---------------------------------------------------------------------------
Architecture, diffusion, window assembly and loss are taken VERBATIM from
`run_real.py` and `models_real.py` via ordinary imports (both are import-safe --
all their execution sits under `if __name__ == "__main__"`). Not one line in
either file is touched. If the architecture changes there, this file changes
with it; it cannot drift silently.

    R.Bundle          features (T,N,d) + adjacency + examples, with its assertions
    R.diffuse         D = A_scaled @ F
    R._gather_window  (tree_idx, t_idx) -> F_seq [B,W,d], D_seq [B,W,1,d]
    R._focal          focal loss gamma=2.0 alpha=0.75
    M.build("STGNN")  exactly the architecture behind the `true` view in the
                      decomposition

---------------------------------------------------------------------------
Checkpoint contents
---------------------------------------------------------------------------
A `state_dict` alone is not enough to reload honestly, so the checkpoint also
carries: architecture shape, task definition (h, WINDOW, risk-set status), node
ordering (`palm_ids` -- tensor row indices are MEANINGLESS without it), feature
column names, adjacency scale, SHA-256 fingerprints of the feature and adjacency
tensors, and a probe set of logits computed at save time.

Those fingerprints and the probe are what let `verify_checkpoint.py` prove the
reload is FAITHFUL rather than merely error-free: if the frozen CSVs shift or the
weights are corrupted, the probe logits will not match and verification fails
hard.
"""
import hashlib
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import models_real as M                        # noqa: E402
import run_real as R                           # noqa: E402

DS, IS_REAL, TAG = R.DS, R.IS_REAL, R.TAG

# Task: the primary horizon, same as PRIMARY_H in run_real.py.
HORIZON = int(os.environ.get("H", R.PRIMARY_H))
SEED = int(os.environ.get("SEED", 0))
EPOCHS = int(os.environ.get("EPOCHS", R.EPOCHS))
VIEW = "true"                                  # the actual proximity graph
MODEL_NAME = "STGNN"                           # not SI(D): that head is NEG at all horizons
DEVICE = R.DEVICE                              # CPU, same as the decomposition

OUT = os.path.join(HERE, os.environ.get("CKPT", "stgnn_final.pt"))
FORMAT_VERSION = 1

SCOPE_WARNING = (
    "Trained on ALL 1,200 palms with no held-out set. This is an INFERENCE "
    "artifact, not an evaluation artifact. No performance number may be quoted "
    "from this model; performance numbers come only from run_real.py / run_v2.py "
    "(leave-one-parcel-out). Its output is a RELATIVE score for ranking, not a "
    "calibrated probability -- never present sigmoid(logit) as a '% chance of "
    "infection'."
)


def log(*a):
    print(*a, flush=True)


def sha(x):
    """Deterministic fingerprint of an array."""
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()

    if not IS_REAL:
        log("!" * 78)
        log("!! dataset.py is NOT active -- this would train on the SYNTHETIC STUB.")
        log("!! The resulting checkpoint would be meaningless. Aborting.")
        log("!" * 78)
        sys.exit(1)

    log(f"dataset: {TAG} | device: {DEVICE} | model: {MODEL_NAME} | view: {VIEW}")
    log(f"h={HORIZON} | WINDOW={R.WINDOW} | epochs={EPOCHS} | seed={SEED} | "
        f"lr={R.LR} | wd={R.WD}")

    # ---- data: the exact same path as the decomposition, assertions included
    b = R.Bundle()
    log(f"  features (T,N,d) = {b.Fnp.shape} | mean degree "
        f"{b.A_true.sum(1).mean():.2f} | edges {int(b.A_true.sum()//2)} | "
        f"adjacency scale {b.scale:.4f}")

    # ---- examples: ALL palms, NO fold split ----------------------------------
    # This is the only substantive difference from run_real.py::split_examples,
    # and it is deliberate. split_examples() filters `tree` by the fold mask; here
    # there is no filtering at all.
    tree, t_idx, y = b.ex[HORIZON]
    n_palms = int(np.unique(tree).size)
    log(f"  training examples: {y.size} (WHOLE plantation, no fold) | "
        f"positives {int(y.sum())} ({100*y.mean():.2f}%) | "
        f"distinct palms touched: {n_palms}/{b.N}")
    assert y.size > 0, "no examples -- invalid horizon/WINDOW?"
    if n_palms < b.N:
        # Not lost data. A query example requires status 'A' at a valid census
        # (t >= WINDOW-1). Palms already symptomatic/dead before the first valid
        # census never enter the risk set -- that is prohibition #5 working, not a
        # bug. They still train the model through the graph, as neighbours in
        # D = A @ F.
        miss = np.setdiff1d(np.arange(b.N), np.unique(tree))
        deg = b.A_true[miss].sum(1).astype(int)
        log(f"    ({b.N - n_palms} palms are never a QUERY example: they left the "
            f"risk set before the first valid census (t_idx={R.WINDOW-1}). They "
            f"still contribute through the graph, degree {deg.min()}-{deg.max()}.)")

    # ---- train ---------------------------------------------------------------
    D = b.view(VIEW, SEED)
    torch.manual_seed(1234 + SEED)             # run_real.py::train_eval seed convention
    model = M.build(MODEL_NAME, b.d, horizon=HORIZON).to(DEVICE)
    n_par = M.count_params(model)
    log(f"  {MODEL_NAME} params={n_par}")

    opt = torch.optim.Adam(model.parameters(), lr=R.LR, weight_decay=R.WD)
    F_seq, D_seq = R._gather_window(tree, t_idx, b.Ft, D)
    y_t = torch.as_tensor(y, device=DEVICE)

    model.train()
    for ep in range(EPOCHS):
        opt.zero_grad()
        loss = R._focal(model(F_seq, D_seq), y_t)
        loss.backward()
        opt.step()
        if ep == 0 or (ep + 1) % 20 == 0 or ep == EPOCHS - 1:
            log(f"    epoch {ep+1:3d}/{EPOCHS}  focal_loss={float(loss.detach()):.6f}")
    log(f"  [{time.time()-t0:.1f}s to end of training]")

    # ---- probe: reference logits that prove a FAITHFUL reload -----------------
    # The last census valid for this horizon -- also the census the demo will use,
    # since it is the most recent one.
    t_star = int(b.T - 1 - HORIZON)
    p_tree, p_t, _ = DS.build_examples(HORIZON, np.array([t_star]))
    n_probe = int(min(8, p_tree.size))
    assert n_probe > 0, f"risk set is empty at census t={t_star}"
    pf, pd_ = R._gather_window(p_tree[:n_probe], p_t[:n_probe], b.Ft, D)

    model.eval()
    with torch.no_grad():
        probe_logits = model(pf, pd_).cpu().numpy().astype(np.float64)
    log(f"  probe: census t_idx={t_star} (t={b.census[t_star]:.1f} yr), "
        f"risk set {p_tree.size} palms, first {n_probe} examples taken as reference")

    # ---- save ----------------------------------------------------------------
    nodes, _, _ = DS.load()
    ckpt = {
        "format_version": FORMAT_VERSION,
        "scope_warning": SCOPE_WARNING,
        "model_class": MODEL_NAME,
        "arch": {"in_dim": int(b.d), "hidden": int(M.HIDDEN), "n_rel": int(M.N_REL)},
        "state_dict": model.state_dict(),
        "task": {
            "horizon": HORIZON,
            "window": int(R.WINDOW),
            "risk_status": "A",
            "pos_status": ["S", "D"],
            "output": "relative logit for ranking; NOT a calibrated probability",
        },
        "train": {
            "dataset_tag": TAG,
            "fold": "NONE -- all 1,200 palms, both parcels",
            "n_examples": int(y.size),
            "n_positives": int(y.sum()),
            "pos_rate": float(y.mean()),
            "n_palms_touched": n_palms,
            "epochs": EPOCHS,
            "lr": R.LR,
            "weight_decay": R.WD,
            "seed": SEED,
            "loss": "focal(gamma=2.0, alpha=0.75)",
            "final_loss": float(loss.detach()),
            "n_params": n_par,
            "device": str(DEVICE),
        },
        "data": {
            "source_csv": ["layer2_nodes.csv", "layer2_panel.csv", "layer2_edges.csv"],
            "n_nodes": int(b.N),
            "n_censuses": int(b.T),
            "feature_names": list(DS.feature_names()),
            "palm_ids": nodes.palm_id.astype(str).tolist(),   # node ordering -- REQUIRED
            "adjacency_view": VIEW,
            "adjacency_scale": float(b.scale),
            "mean_degree": float(b.A_true.sum(1).mean()),
            "n_edges": int(b.A_true.sum() // 2),
            "feature_sha256": sha(b.Fnp),
            "adjacency_sha256": sha(b.A_true),
        },
        "probe": {
            "census_idx": t_star,
            "census_t": float(b.census[t_star]),
            "tree_idx": p_tree[:n_probe].astype(int).tolist(),
            "t_idx": p_t[:n_probe].astype(int).tolist(),
            "logits": probe_logits.tolist(),
        },
    }
    torch.save(ckpt, OUT)
    size_kb = os.path.getsize(OUT) / 1024

    log(f"\n  saved: {OUT}  ({size_kb:.1f} KB)")
    log(f"  feature sha256   {ckpt['data']['feature_sha256'][:16]}...")
    log(f"  adjacency sha256 {ckpt['data']['adjacency_sha256'][:16]}...")
    log(f"  probe logits (first 8): "
        + " ".join(f"{v:+.4f}" for v in probe_logits))
    log("\n  " + "-" * 74)
    log("  " + SCOPE_WARNING.replace(". ", ".\n  "))
    log("  " + "-" * 74)
    log(f"\nTOTAL {time.time()-t0:.1f}s")
    log("Verify the reload in a separate process: python verify_checkpoint.py")


if __name__ == "__main__":
    main()
