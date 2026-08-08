"""Layer 2 — export the final model's output: ranked CSV + quantile risk map.

    python export_risk.py [--census IDX] [--out-dir DIR]

Consumes `stgnn_final.pt` and produces the artifacts that `context_layer2.md` §8
calls output 3:

    layer2_real/risk_ranked.csv        one row per palm, ranked
    layer2_real/risk_ranked.meta.json  provenance sidecar (see below)
    figures/fig_layer2_risk_map.png    lattice coloured by risk quantile
    figures/fig_layer2_score_dist.png  score distribution

---------------------------------------------------------------------------
Honesty constraint #1 is enforced here, not merely mentioned
---------------------------------------------------------------------------
`context_layer2.md` §1.1: "Output is RELATIVE risk (rank / quantile), NOT
calibrated probability. Any code that emits a '% chance of infection' is a bug.
Risk maps use quantile colouring."

This script therefore emits, per palm:

    logit             the raw model output — monotone in risk, unitless
    risk_percentile   0-100 rank within the risk set
    risk_decile       1-10 quantile band (10 = highest risk)

and never `sigmoid(logit)`. The model was never calibrated: the pos-rate at h=3
is 4.45%, and a focal-loss-trained network's sigmoid does not estimate that rate.
A number that looks like a probability would be read as one.

The map is coloured by **decile**, not by logit, for the same reason — a
continuous colour ramp over an uncalibrated score invites reading the gaps as
meaningful, and they are not. Only the ordering is meaningful.

---------------------------------------------------------------------------
Why every palm is in the CSV, not just the risk set
---------------------------------------------------------------------------
Writing only the 672 scored palms would silently drop 528 rows and make the file
look like a complete census of the plantation. Each palm therefore gets a row
with `in_risk_set` and its `status`, and the unscored ones carry empty score
fields rather than a zero. A zero would sort as "lowest risk", which is exactly
wrong for a palm that is already dead.

---------------------------------------------------------------------------
The sidecar
---------------------------------------------------------------------------
A CSV cannot carry provenance without corrupting itself for naive readers, so
`risk_ranked.meta.json` holds: the checkpoint's scope warning, the census used,
the model's training summary, the data fingerprints, and the column dictionary.
If the CSV travels, the sidecar should travel with it.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import dataset as ds          # noqa: E402
import models_real as M       # noqa: E402

CKPT = os.path.join(HERE, "stgnn_final.pt")
FIGDIR = os.path.join(ROOT, "figures")


def log(*a):
    print(*a, flush=True)


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def load_ckpt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--census", type=int, default=None,
                    help="census index to score (default: the checkpoint's probe census, "
                         "i.e. the most recent one valid for this horizon)")
    ap.add_argument("--out-dir", default=HERE, help="where to write the CSV")
    ap.add_argument("--fig-dir", default=FIGDIR, help="where to write the PNGs")
    a = ap.parse_args()

    if not os.path.exists(CKPT):
        log(f"checkpoint not found: {CKPT}\nrun this first: python train_final.py")
        sys.exit(1)
    ck = load_ckpt(CKPT)
    h = ck["task"]["horizon"]

    # ---- rebuild model + data, with the same hard fingerprint check as verify ----
    T = len(ds.census())
    F = np.asarray(ds.node_features(np.arange(T)), np.float32)
    A = np.asarray(ds.adjacency(ck["data"]["adjacency_view"]), np.float32)
    if sha(F) != ck["data"]["feature_sha256"] or sha(A) != ck["data"]["adjacency_sha256"]:
        log("FATAL: the frozen CSVs have changed since this checkpoint was trained.")
        log("       Scores would be attached to different data. Retrain, do not export.")
        sys.exit(1)

    model = M.build(ck["model_class"], ck["arch"]["in_dim"], horizon=h)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()

    D = ds.diffuse(F, A)
    t_star = ck["probe"]["census_idx"] if a.census is None else int(a.census)
    assert ds.WINDOW - 1 <= t_star < T - h, (
        f"census {t_star} is not valid for h={h}: need {ds.WINDOW-1} <= t < {T-h}")
    census_t = float(ds.census()[t_star])

    # ---- score the risk set ---------------------------------------------------
    tree, t_idx, _ = ds.build_examples(h, np.array([t_star]))
    Fs, Dz = ds.make_windows(tree, t_idx, F, D)
    with torch.no_grad():
        logits = model(torch.as_tensor(Fs), torch.as_tensor(Dz)).numpy().astype(np.float64)

    log(f"checkpoint : {os.path.basename(CKPT)}  ({ck['model_class']}, h={h})")
    log(f"census     : idx {t_star} (t = {census_t:.1f} yr)")
    log(f"risk set   : {len(tree)} of {ck['data']['n_nodes']} palms")
    log(f"logit range: {logits.min():+.4f} .. {logits.max():+.4f}")

    # ---- assemble one row per palm -------------------------------------------
    nodes, _, _ = ds.load()
    N = len(nodes)
    st = ds.status_matrix()
    status_now = st[t_star]
    deg = A.sum(1).astype(int)
    n_sick = (A @ np.isin(status_now, ("S", "D")).astype(float)).astype(int)

    order_desc = np.argsort(-logits)                 # highest risk first
    rank = np.full(N, 0, int)
    rank[tree[order_desc]] = np.arange(1, len(tree) + 1)

    pct = np.full(N, np.nan)
    asc = np.argsort(np.argsort(logits))             # 0 = lowest risk
    pct[tree] = 100.0 * asc / max(len(logits) - 1, 1)

    dec = np.full(N, 0, int)
    # Decile by rank, so bands are equal-sized by construction (10 = highest risk).
    dec[tree[order_desc]] = 10 - np.minimum(
        (np.arange(len(tree)) * 10) // max(len(tree), 1), 9)

    score = np.full(N, np.nan)
    score[tree] = logits
    in_risk = np.zeros(N, bool)
    in_risk[tree] = True

    cols = ["rank", "palm_id", "parcel", "plot", "progeny", "xm", "ym",
            "in_risk_set", "status", "logit", "risk_percentile", "risk_decile",
            "n_neighbours", "n_sick_neighbours"]

    # risk-set palms first, ranked; then the unscored ones, stable by palm order
    idx_sorted = list(tree[order_desc]) + [i for i in range(N) if not in_risk[i]]

    out_csv = os.path.join(a.out_dir, "risk_ranked.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        import csv
        w = csv.writer(fh)
        w.writerow(cols)
        for i in idx_sorted:
            r = nodes.iloc[i]
            w.writerow([
                rank[i] if in_risk[i] else "",
                # NOTE: r["plot"], never r.plot -- on a Series, `.plot` resolves to
                # the pandas plotting accessor, not the column of that name.
                r.palm_id, r.parcel, r["plot"], r.progeny,
                f"{r.xm:.4f}", f"{r.ym:.4f}",
                int(in_risk[i]), status_now[i],
                f"{score[i]:.6f}" if in_risk[i] else "",
                f"{pct[i]:.2f}" if in_risk[i] else "",
                dec[i] if in_risk[i] else "",
                deg[i], n_sick[i],
            ])
    log(f"\nwrote {out_csv}  ({N} rows: {int(in_risk.sum())} scored, "
        f"{int((~in_risk).sum())} outside the risk set)")

    # ---- provenance sidecar ---------------------------------------------------
    meta = {
        "produced_by": "layer2_real/export_risk.py",
        "checkpoint": os.path.basename(CKPT),
        "scope_warning": ck["scope_warning"],
        "census": {"index": t_star, "t_years": census_t,
                   "note": "most recent census valid for this horizon"},
        "task": ck["task"],
        "model": {"class": ck["model_class"], "arch": ck["arch"],
                  "n_params": ck["train"]["n_params"]},
        "training_summary": ck["train"],
        "data_fingerprint": {"feature_sha256": ck["data"]["feature_sha256"],
                             "adjacency_sha256": ck["data"]["adjacency_sha256"],
                             "source_csv": ck["data"]["source_csv"]},
        "rows": {"total": N, "scored": int(in_risk.sum()),
                 "outside_risk_set": int((~in_risk).sum())},
        "columns": {
            "rank": "1 = highest risk, within the risk set only; empty otherwise",
            "in_risk_set": "1 if status was 'A' at this census and the palm is scorable",
            "status": "A asymptomatic | S symptomatic | D dead | C censored",
            "logit": "raw model output, monotone in risk, unitless. NOT a probability",
            "risk_percentile": "0-100 rank within the risk set",
            "risk_decile": "1-10 quantile band, 10 = highest risk",
            "n_neighbours": "graph degree at r = 1.5 x planting spacing",
            "n_sick_neighbours": "neighbours with status S or D at this census",
        },
        "prohibited_uses": [
            "Do not quote any performance metric from this file; the model has no held-out set.",
            "Do not convert `logit` to a probability. It is uncalibrated.",
            "Do not treat an empty score as low risk; those palms are already symptomatic, dead or censored.",
        ],
    }
    out_meta = os.path.join(a.out_dir, "risk_ranked.meta.json")
    with open(out_meta, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    log(f"wrote {out_meta}")

    # ---- figures --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    os.makedirs(a.fig_dir, exist_ok=True)
    xy = nodes[["xm", "ym"]].to_numpy()

    # -- risk map, coloured by DECILE (discrete), never by raw logit
    # Figure height is derived from the DATA aspect: the lattice is ~51 x ~31
    # spacings, so with aspect="equal" a square figure would leave the axes box
    # short and strand a full-height colorbar beside it.
    span_x = float(xy[:, 0].max() - xy[:, 0].min())
    span_y = float(xy[:, 1].max() - xy[:, 1].min())
    fig_w = 10.0
    fig_h = max(3.6, fig_w * 0.76 * (span_y / span_x) + 1.5)   # axes + title/labels
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = ListedColormap(plt.get_cmap("YlOrRd")(np.linspace(0.08, 0.96, 10)))
    norm = BoundaryNorm(np.arange(0.5, 11.5, 1.0), cmap.N)
    ax.scatter(xy[~in_risk, 0], xy[~in_risk, 1], s=13, c="#D9D9D9",
               linewidths=0, label=f"outside risk set (n={int((~in_risk).sum())})")
    sc = ax.scatter(xy[in_risk, 0], xy[in_risk, 1], s=20, c=dec[in_risk],
                    cmap=cmap, norm=norm, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, ticks=range(1, 11), pad=0.015, fraction=0.035,
                      aspect=22)
    cb.set_label("risk decile within the risk set  (10 = highest)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("x (planting spacings, cos30-corrected)")
    ax.set_ylabel("y (planting spacings)")
    ax.set_title(f"Eg9PP relative risk, h={h} censuses ahead\n"
                 f"census {t_star} (t = {census_t:.1f} yr) — ranking, not probability",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    p1 = os.path.join(a.fig_dir, "fig_layer2_risk_map.png")
    fig.savefig(p1, dpi=300, facecolor="white")
    plt.close(fig)

    # -- score distribution
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(logits, bins=45, color="#4C72B0", edgecolor="white", linewidth=0.6)
    ax.axvline(float(np.median(logits)), color="#C44E52", ls="--", linewidth=1.5,
               label=f"median {np.median(logits):+.3f}")
    ax.set_xlabel("model output (logit) — monotone in risk, unitless")
    ax.set_ylabel("number of palms")
    ax.set_title(f"Score distribution over the risk set (n={len(logits)}), "
                 f"h={h}, t={census_t:.1f} yr", fontsize=11)
    ax.legend()
    fig.tight_layout()
    p2 = os.path.join(a.fig_dir, "fig_layer2_score_dist.png")
    fig.savefig(p2, dpi=300, facecolor="white")
    plt.close(fig)

    log(f"wrote {p1}")
    log(f"wrote {p2}")

    # ---- console summary ------------------------------------------------------
    log("\ntop 10 by risk rank (relative score, not a probability):")
    log(f"  {'#':>3} {'palm_id':>10} {'parcel':>7} {'logit':>9} {'pct':>7} "
        f"{'decile':>7} {'sick nb':>9}")
    for i in tree[order_desc][:10]:
        r = nodes.iloc[i]
        log(f"  {rank[i]:>3} {r.palm_id:>10} {r.parcel:>7} {score[i]:>+9.4f} "
            f"{pct[i]:>6.1f}% {dec[i]:>7} {n_sick[i]:>6}/{deg[i]}")

    log("\n  " + "-" * 74)
    log("  " + ck["scope_warning"].replace(". ", ".\n  "))
    log("  " + "-" * 74)


if __name__ == "__main__":
    main()
