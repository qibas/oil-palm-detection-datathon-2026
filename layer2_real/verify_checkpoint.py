"""Layer 2 — proof that `stgnn_final.pt` reloads STANDALONE and runs.

    python verify_checkpoint.py [checkpoint_path]

Runs as a SEPARATE PROCESS and deliberately does **not** import `train_final.py`
or `run_real.py`. It touches only:

    dataset.py       frozen CSVs -> features, adjacency, risk set (+ diffuse/make_windows)
    models_real.py   architecture

If this file passes, the checkpoint genuinely stands on its own: a downstream
demo needs nothing more than `stgnn_final.pt` + `data_clean/*.csv` + those two
modules.

---------------------------------------------------------------------------
Five checks, weakest to most binding
---------------------------------------------------------------------------
  V1  file loads               torch.load succeeds, required fields present
  V2  architecture matches     checkpoint in_dim/hidden/n_rel == models_real as
                               it is today; load_state_dict(strict=True) with no
                               missing or unexpected keys
  V3  data has not shifted     SHA-256 of the feature and adjacency tensors
                               RECOMPUTED from the frozen CSVs == the stored ones
  V4  reload is FAITHFUL       recomputed probe logits == the logits stored at
                               training time (atol 1e-5)
                               <- THIS is the check that actually binds. V1-V2
                               only prove "no error"; V4 proves the loaded
                               weights produce the SAME numbers.
  V5  full forward pass        one forward pass over the entire risk set at the
                               most recent census -> per-palm scores, shapes
                               reported

Any failure -> exit 1. There are no soft warnings.

HONESTY NOTE: the model's output is a **relative logit for ranking**. This file
reports logits and rank percentiles. It does NOT print sigmoid(logit) as a
percentage, because this model was never calibrated and such a number would read
as a "% chance of infection" -- which honesty constraint #1 in context_layer2.md
calls a bug, not a feature.
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

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "stgnn_final.pt")
ATOL = 1e-5

_fails = []


def hdr(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


def check(ok, label, detail=""):
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        _fails.append(label)
    return ok


def sha(x):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def load_ckpt(path):
    """Version-tolerant torch.load: weights_only=True where supported, else False."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # ---- V1 -----------------------------------------------------------------
    hdr("V1 — FILE LOADS")
    if not os.path.exists(CKPT):
        print(f"  [FAIL] checkpoint not found: {CKPT}")
        print("         run this first: python train_final.py")
        sys.exit(1)
    ck = load_ckpt(CKPT)
    print(f"  file     : {CKPT} ({os.path.getsize(CKPT)/1024:.1f} KB)")
    need = ("format_version", "model_class", "arch", "state_dict", "task",
            "train", "data", "probe", "scope_warning")
    missing = [k for k in need if k not in ck]
    check(not missing, "required fields present", f"missing: {missing}" if missing else
          f"{len(need)} fields present")
    print(f"  class    : {ck['model_class']} | arch {ck['arch']}")
    print(f"  task     : h={ck['task']['horizon']} window={ck['task']['window']} "
          f"risk_status={ck['task']['risk_status']} pos={ck['task']['pos_status']}")
    print(f"  training : fold={ck['train']['fold']}")
    print(f"             {ck['train']['n_examples']} examples, "
          f"{ck['train']['n_positives']} positives ({100*ck['train']['pos_rate']:.2f}%), "
          f"{ck['train']['epochs']} epochs, seed {ck['train']['seed']}")

    # ---- V2 -----------------------------------------------------------------
    hdr("V2 — ARCHITECTURE MATCHES models_real.py AS IT IS TODAY")
    a = ck["arch"]
    check(a["hidden"] == M.HIDDEN, "hidden matches",
          f"checkpoint {a['hidden']} vs models_real.HIDDEN {M.HIDDEN}")
    check(a["n_rel"] == M.N_REL, "n_rel matches",
          f"checkpoint {a['n_rel']} vs models_real.N_REL {M.N_REL}")
    model = M.build(ck["model_class"], a["in_dim"], horizon=ck["task"]["horizon"])
    incompat = model.load_state_dict(ck["state_dict"], strict=True)
    check(not incompat.missing_keys and not incompat.unexpected_keys,
          "load_state_dict(strict=True)",
          f"missing={list(incompat.missing_keys)} unexpected={list(incompat.unexpected_keys)}")
    model.eval()
    n_par = M.count_params(model)
    check(n_par == ck["train"]["n_params"], "parameter count matches",
          f"{n_par} == {ck['train']['n_params']}")

    # ---- V3 -----------------------------------------------------------------
    hdr("V3 — FROZEN DATA HAS NOT SHIFTED (recomputed from CSV)")
    T = len(ds.census())
    Fnp = np.asarray(ds.node_features(np.arange(T)), np.float32)
    A = np.asarray(ds.adjacency(ck["data"]["adjacency_view"]), np.float32)
    check(Fnp.shape == (T, ck["data"]["n_nodes"], a["in_dim"]), "feature shape",
          f"{Fnp.shape}")
    f_ok = sha(Fnp) == ck["data"]["feature_sha256"]
    a_ok = sha(A) == ck["data"]["adjacency_sha256"]
    check(f_ok, "feature tensor sha256", f"{sha(Fnp)[:16]}... vs "
          f"{ck['data']['feature_sha256'][:16]}...")
    check(a_ok, "adjacency sha256", f"{sha(A)[:16]}... vs "
          f"{ck['data']['adjacency_sha256'][:16]}...")
    check(list(ds.feature_names()) == ck["data"]["feature_names"],
          "feature column names match", f"{len(ck['data']['feature_names'])} columns")
    nodes, _, _ = ds.load()
    check(nodes.palm_id.astype(str).tolist() == ck["data"]["palm_ids"],
          "node ordering (palm_ids) matches", f"{len(ck['data']['palm_ids'])} palms")

    # diffusion uses dataset.py's own helper (dataset.py:474) -- not a copy
    D = ds.diffuse(Fnp, A)
    check(D.shape == (T, ck["data"]["n_nodes"], M.N_REL, a["in_dim"]),
          "diffusion tensor shape", f"{D.shape}")

    # ---- V4 -----------------------------------------------------------------
    hdr("V4 — FAITHFUL RELOAD (probe logits reproduced)")
    pr = ck["probe"]
    tree = np.asarray(pr["tree_idx"], int)
    t_idx = np.asarray(pr["t_idx"], int)
    Fs, Dz = ds.make_windows(tree, t_idx, Fnp, D)
    with torch.no_grad():
        got = model(torch.as_tensor(Fs), torch.as_tensor(Dz)).numpy().astype(np.float64)
    want = np.asarray(pr["logits"], np.float64)
    dmax = float(np.abs(got - want).max())
    print(f"  probe census : idx {pr['census_idx']} (t = {pr['census_t']:.1f} yr), "
          f"{len(tree)} reference examples")
    print(f"  stored       : " + " ".join(f"{v:+.6f}" for v in want))
    print(f"  recomputed   : " + " ".join(f"{v:+.6f}" for v in got))
    check(dmax <= ATOL, "probe logits reproduced", f"max |diff| {dmax:.2e} <= {ATOL:.0e}")

    # ---- V5 -----------------------------------------------------------------
    hdr("V5 — ONE FORWARD PASS OVER THE CURRENT RISK SET")
    t_star = pr["census_idx"]
    tr, tt, yy = ds.build_examples(ck["task"]["horizon"], np.array([t_star]))
    Fs, Dz = ds.make_windows(tr, tt, Fnp, D)
    F_t, D_t = torch.as_tensor(Fs), torch.as_tensor(Dz)
    with torch.no_grad():
        logits = model(F_t, D_t).numpy()

    N = ck["data"]["n_nodes"]
    full = np.full(N, np.nan, np.float64)
    full[tr] = logits
    rank = np.full(N, np.nan, np.float64)
    order = np.argsort(np.argsort(logits))                  # 0 = lowest risk
    rank[tr] = 100.0 * order / max(len(logits) - 1, 1)      # percentile within risk set

    print(f"  census       : idx {t_star} (t = {pr['census_t']:.1f} yr) — last census "
          f"valid for h={ck['task']['horizon']}")
    print(f"  input        : F_seq {tuple(F_t.shape)}   D_seq {tuple(D_t.shape)}")
    print(f"  output       : logits {logits.shape}  dtype {logits.dtype}")
    print(f"  risk set     : {len(tr)} of {N} palms "
          f"({100*len(tr)/N:.1f}% — the rest are symptomatic/dead/censored)")
    print(f"  lattice vec  : {full.shape} (NaN for {int(np.isnan(full).sum())} "
          f"palms outside the risk set)")
    print(f"  logit range  : {logits.min():+.4f} .. {logits.max():+.4f}  "
          f"(median {np.median(logits):+.4f})")
    check(np.isfinite(logits).all(), "all scores finite (no NaN/Inf)")
    check(logits.shape == (len(tr),), "output shape = (n_risk_set,)", f"{logits.shape}")
    check(len(tr) > 0, "risk set is not empty")

    ids = np.asarray(ck["data"]["palm_ids"])
    top = tr[np.argsort(-logits)][:5]
    print("\n  top 5 palms by risk rank (RELATIVE SCORE, not a probability):")
    print(f"    {'palm_id':>10} {'idx':>5} {'logit':>10} {'percentile':>11}")
    for i in top:
        print(f"    {ids[i]:>10} {i:>5} {full[i]:>+10.4f} {rank[i]:>10.1f}%")

    # ---- summary ------------------------------------------------------------
    hdr("SUMMARY")
    print("  " + ck["scope_warning"].replace(". ", ".\n  "))
    print()
    if _fails:
        print(f"  {len(_fails)} check(s) FAILED:")
        for f in _fails:
            print(f"    - {f}")
        sys.exit(1)
    print("  All checks passed. The checkpoint loads standalone and runs.")


if __name__ == "__main__":
    main()
