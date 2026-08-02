"""SawitGuard-GNN — experiment driver for the REAL Eg9PP panel (`layer2_real/`).

    python run_real.py [n_seeds] [n_perm]

Design, and why each piece is the way it is:

* **Fold = parcel, leave-one-parcel-out, 2 folds.** Never a random split. The
  frozen build asserts 0 of 3354 root-contact edges cross a parcel boundary, so
  splitting by parcel does not cut a single edge: no neighbour information
  crosses the fold. All 14 families are present in both parcels, so the fold is
  not confounded by genotype either.

* **The split is spatial, NOT temporal.** Both folds see all 45 censuses. That is
  deliberate and it is what the `prevalence` component of the decomposition
  measures: a model can learn "the epidemic is at level p at census t" from the
  training parcel and carry it to the held-out parcel. Calling that leakage
  would be wrong — it is a real, deployable signal — but calling it *spatial*
  would also be wrong, which is exactly why the decomposition separates it from
  STRUCTURE.

* **The paired unit is (fold, seed), n = 2 x SEEDS.** Honest caveat, stated here
  because it must not be lost downstream: the real panel is ONE fixed dataset.
  Re-seeding re-initialises the network and redraws the degree-preserving random
  graph — it does NOT resample the epidemic, as it does on the synthetic side.
  So the across-seed std is an *optimisation* std, not a data std, and the only
  data-level replication available is the 2 parcels. Treat a small mean with a
  tiny std with more suspicion here than in the synthetic Layer 2.

* **Metric is AUC-PR.** Pos-rate is 1.6-5.7%; accuracy is meaningless. The
  no-skill AUC-PR equals the test-fold pos-rate and is printed next to every
  number so the reader can see how far above chance a model actually is.

* `paired()` is copied VERBATIM from repo-root `run_experiment.py` (it is not
  importable without executing that module's argv-dependent top level). Same
  mean/std/sign-count/INCONCLUSIVE rule, deliberately not re-invented.

Decomposition, identical in form to `run_experiment.py::decomposition()`:

    temporal    = nograph - MLP        value of modelling time (+ genotype ctx)
    prevalence  = random  - nograph    value of having ANY graph
    STRUCTURE   = true    - random     value of the CORRECT contact map
"""
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _ds                                     # noqa: E402
import models_real as M                        # noqa: E402
import perm_null                               # noqa: E402

DS, IS_REAL, TAG = _ds.DS, _ds.IS_REAL, _ds.TAG
WINDOW, HORIZONS, N_REL = _ds.WINDOW, _ds.HORIZONS, _ds.N_REL

DEVICE = torch.device("cpu")                   # 1200 nodes x 45 censuses: CPU is the right tool
# 60 full-batch epochs, identical to repo-root train.py. Verified, not inherited:
# on fold 44A / h=3 the held-out AUC-PR of STGNN(true) reads
#   ep20 .168  ep40 .164  ep60 .173  ep100 .173  ep150 .157  ep200 .138  ep400 .106
# i.e. the peak is a plateau at 60-100 epochs and everything past ~150 is pure
# overfitting on 16k examples with ~750 positives. Every model gets the same
# budget, so the comparison stays fair.
EPOCHS = int(os.environ.get("EPOCHS", 60))
LR = 5e-3
WD = 1e-4

OUT = os.path.join(HERE, "results_real.csv")
ROWS = []


# --------------------------------------------------------------------------- #
# significance harness — VERBATIM from repo-root run_experiment.py::paired()
def paired(delta, label="STGNN"):
    d = np.asarray(delta, float); d = d[~np.isnan(d)]
    if d.size == 0:
        return dict(mean=np.nan, std=np.nan, n=0, pos=0, verdict="n/a")
    mean, std = float(d.mean()), float(d.std())
    verdict = "INCONCLUSIVE" if abs(mean) < std else ("POS" if mean > 0 else "NEG")
    return dict(mean=mean, std=std, n=int(d.size), pos=int((d > 0).sum()), verdict=verdict)


def log(*a):
    print(*a, flush=True)


def rec(**k):
    ROWS.append(dict(dataset=TAG, **k))


# --------------------------------------------------------------------------- #
def adjacency_scale(A_true):
    """One global scale, shared by every view, taken from the TRUE map.

    Same rule as root `train.py::adjacency_scale`: normalise by the mean row-sum
    so the diffused magnitudes are comparable across views. Fixing it to the true
    map (never to the swapped view) is what makes `random` a fair control — the
    degree-preserving rewire has the same row-sums, so only structure differs.
    """
    return 1.0 / (float(A_true.sum(1).mean()) + 1e-8)


def diffuse(Ft, A_scaled):
    """Ft [T,N,d], A [N,N] -> D [T,N,N_REL,d] with N_REL = 1 (root contact only)."""
    D = torch.einsum("ij,tjd->tid", A_scaled, Ft)
    return D.unsqueeze(2)


def _focal(logits, y, gamma=2.0, alpha=0.75):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = torch.where(y > 0.5, p, 1 - p)
    w = torch.where(y > 0.5, alpha, 1 - alpha)
    return (w * (1 - pt) ** gamma * ce).mean()


def _gather_window(idx_tree, idx_t, Ft, D):
    tt = torch.as_tensor(np.asarray(idx_t), device=Ft.device, dtype=torch.long)
    ii = torch.as_tensor(np.asarray(idx_tree), device=Ft.device, dtype=torch.long)
    Fs, Ds = [], []
    for w in range(WINDOW):
        cyc = tt - (WINDOW - 1) + w
        Fs.append(Ft[cyc, ii])
        Ds.append(D[cyc, ii])
    return torch.stack(Fs, 1), torch.stack(Ds, 1)


# --------------------------------------------------------------------------- #
class Bundle:
    """Everything that does not depend on (model, seed, fold): features, examples,
    and the seed-independent diffusions. Built once."""

    def __init__(self):
        self.census = np.asarray(DS.census())
        self.T = len(self.census)
        self.folds = DS.folds()
        # Rule 2 ("fit the scaler on training data only") under a purely SPATIAL fold:
        # there is no held-out census, so every census is a training census; what must
        # be held out is the test PARCEL. dataset.py exposes `train_nodes` for exactly
        # that, so the scaler is fit on the fold's training palms. The tensor is then
        # computed ONCE because the fit is provably fold-invariant -- the only scaled
        # block (SELF: t_years, census_idx, dt_prev, log1p_t) is constant across palms
        # within a census, so its mean/std cannot depend on which palms are used. That
        # invariance is asserted below rather than assumed.
        train_t = np.arange(self.T)
        try:
            X = [np.asarray(DS.node_features(train_t, train_nodes=np.asarray(trm, bool)),
                            np.float32) for trm, _ in self.folds]
            assert all(np.allclose(X[0], x) for x in X[1:]), \
                "scaler is NOT fold-invariant -- features must be rebuilt per fold"
            self.Fnp = X[0]
        except TypeError:                     # contract signature only: node_features(train_t)
            self.Fnp = np.asarray(DS.node_features(train_t), np.float32)
        assert self.Fnp.ndim == 3, f"node_features must be (T,N,d), got {self.Fnp.shape}"
        assert self.Fnp.shape[0] == self.T, (self.Fnp.shape, self.T)
        assert np.isfinite(self.Fnp).all(), "node_features contains NaN/Inf"
        self.T, self.N, self.d = self.Fnp.shape
        self.Ft = torch.as_tensor(self.Fnp, device=DEVICE)

        self.A_true = np.asarray(DS.adjacency("true"), np.float32)
        assert self.A_true.shape == (self.N, self.N)
        assert np.allclose(self.A_true, self.A_true.T), "true adjacency is not symmetric"
        assert np.allclose(np.diag(self.A_true), 0), "true adjacency has a non-zero diagonal"
        self.scale = adjacency_scale(self.A_true)

        self.fold_names = self._fold_names()
        self._check_folds()

        self.D = {"true": diffuse(self.Ft, torch.as_tensor(self.A_true * self.scale, device=DEVICE)),
                  "zero": torch.zeros(self.T, self.N, N_REL, self.d, device=DEVICE)}
        self._Drandom = {}
        self.ex = {h: DS.build_examples(h, np.arange(self.T)) for h in HORIZONS}

    def _fold_names(self):
        try:
            nodes, _, _ = DS.load()
            par = nodes.parcel.to_numpy()
            return [str(np.unique(par[te])[0]) if len(np.unique(par[te])) == 1 else f"fold{k}"
                    for k, (_, te) in enumerate(self.folds)]
        except Exception:                                   # noqa: BLE001
            return [f"fold{k}" for k in range(len(self.folds))]

    def _check_folds(self):
        for k, (trm, tem) in enumerate(self.folds):
            trm, tem = np.asarray(trm, bool), np.asarray(tem, bool)
            assert trm.size == self.N and tem.size == self.N
            assert not (trm & tem).any(), f"fold {k}: train/test node masks overlap"
            cross = float(self.A_true[np.ix_(trm, tem)].sum())
            if cross > 0:
                log(f"  !! WARNING fold {k}: {int(cross)} edges cross the fold boundary — "
                    f"leave-one-parcel-out is supposed to cut zero")
            else:
                log(f"  fold {k} ({self.fold_names[k]}): train {int(trm.sum())} / "
                    f"test {int(tem.sum())} palms, 0 edges cut")

    def D_random(self, seed):
        if seed not in self._Drandom:
            Ar = np.asarray(DS.adjacency("random", seed=seed), np.float32)
            dd = np.abs(Ar.sum(1) - self.A_true.sum(1)).max()
            if dd > 1e-6:
                log(f"  !! WARNING random view (seed {seed}) does NOT preserve degree "
                    f"(max |delta deg| = {dd:.2f})")
            for k, (trm, tem) in enumerate(self.folds):
                x = float(Ar[np.ix_(np.asarray(trm, bool), np.asarray(tem, bool))].sum())
                if x > 0 and seed == 0:
                    log(f"  !! WARNING random view creates {int(x)} cross-fold edges in fold {k}; "
                        f"the `prevalence` component will partly measure fold leakage, not "
                        f"delocalised smoothing. Rewiring should be restricted to within-parcel.")
            self._Drandom[seed] = diffuse(self.Ft, torch.as_tensor(Ar * self.scale, device=DEVICE))
        return self._Drandom[seed]

    def view(self, name, seed):
        return self.D_random(seed) if name == "random" else self.D[name]


# --------------------------------------------------------------------------- #
def split_examples(bundle, h, fold):
    tree, t, y = bundle.ex[h]
    trm, tem = (np.asarray(m, bool) for m in bundle.folds[fold])
    a, b = trm[tree], tem[tree]
    return (tree[a], t[a], y[a]), (tree[b], t[b], y[b])


def train_eval(bundle, fold, seed, h, model_name, view, base_hazard=None):
    tr, te = split_examples(bundle, h, fold)
    if tr[2].size == 0 or te[2].size == 0 or te[2].sum() == 0:
        return float("nan"), float("nan")
    D = bundle.view(view, seed)

    torch.manual_seed(1234 + seed)
    model = M.build(model_name, bundle.d, horizon=h, base_hazard=base_hazard).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    Ftr, Dtr = _gather_window(tr[0], tr[1], bundle.Ft, D)
    ytr = torch.as_tensor(tr[2], device=DEVICE)
    Fte, Dte = _gather_window(te[0], te[1], bundle.Ft, D)

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        _focal(model(Ftr, Dtr), ytr).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Fte, Dte)).cpu().numpy()
    return float(average_precision_score(te[2], p)), float(te[2].mean())


# --------------------------------------------------------------------------- #
CONFIGS = [("MLP", "MLPBaseline", "zero"),
           ("nograph", "STGNN", "zero"),
           ("random", "STGNN", "random"),
           ("true", "STGNN", "true"),
           ("SID", "STGNN_SID", "true")]

COMPONENTS = [("temporal (nograph-MLP)", "nograph", "MLP"),
              ("prevalence (random-nograph)", "random", "nograph"),
              ("STRUCTURE (true-random)", "true", "random"),
              ("total graph (true-nograph)", "true", "nograph"),
              ("total STGNN-MLP (true-MLP)", "true", "MLP"),
              ("SI(D) head (SID-true) *NOT vs synth SEIR*", "SID", "true")]


def decomposition(bundle, seeds, horizons):
    nF = len(bundle.folds)
    ap = {(h, f, k): [] for h in horizons for f in range(nF) for k, _, _ in CONFIGS}
    posrate = {}
    t0 = time.time()
    for h in horizons:
        for f in range(nF):
            for s in range(seeds):
                for k, mname, view in CONFIGS:
                    a, pr = train_eval(bundle, f, s, h, mname, view)
                    ap[(h, f, k)].append(a)
                    posrate[(h, f)] = pr
    log(f"  [{time.time()-t0:.1f}s of training]")

    for h in horizons:
        log(f"\n  ----- horizon h={h} census steps "
            f"(no-skill AUC-PR = test pos-rate) -----")
        for f in range(nF):
            log(f"    fold {bundle.fold_names[f]} (test pos-rate {100*posrate[(h,f)]:.2f}%): " +
                "  ".join(f"{k} {np.mean(ap[(h,f,k)]):.3f}" for k, _, _ in CONFIGS))
            rec(part="AP", fold=bundle.fold_names[f], h=h, model="posrate",
                metric="no-skill AUCPR", mean=posrate[(h, f)], std=0.0, n=1)
            for k, _, _ in CONFIGS:
                v = np.asarray(ap[(h, f, k)], float)
                rec(part="AP", fold=bundle.fold_names[f], h=h, model=k, metric="AUCPR",
                    mean=float(np.nanmean(v)), std=float(np.nanstd(v)), n=int(v.size))

        for name, hi, lo in COMPONENTS:
            per_fold = []
            pooled = []
            for f in range(nF):
                d = np.asarray(ap[(h, f, hi)], float) - np.asarray(ap[(h, f, lo)], float)
                per_fold.append(paired(d))
                pooled.append(d)
            g = paired(np.concatenate(pooled))
            log(f"    {name:44s}: {g['mean']:+.4f}+/-{g['std']:.4f}  sign {g['pos']}/{g['n']}  "
                f"[{g['verdict']}]   per-fold " +
                " | ".join(f"{bundle.fold_names[f]} {per_fold[f]['mean']:+.4f} "
                           f"{per_fold[f]['pos']}/{per_fold[f]['n']}" for f in range(nF)))
            rec(part="DECOMP", fold="pooled", h=h, model=name, metric="pairedAUCPR", **g)
            for f in range(nF):
                rec(part="DECOMP", fold=bundle.fold_names[f], h=h, model=name,
                    metric="pairedAUCPR", **per_fold[f])
    return ap, posrate


def leak_sentinel(ap, posrate, horizons, nF):
    """A model that scores far above the no-skill line on a 2%-positive early-warning
    task is more likely to be reading a leak than seeing the future. Flag it."""
    bad = []
    for h in horizons:
        for f in range(nF):
            for k in ("MLP", "true"):
                m = float(np.nanmean(ap[(h, f, k)]))
                if m > 0.60 and m > 8 * posrate[(h, f)]:
                    bad.append((h, f, k, m, posrate[(h, f)]))
    if bad:
        log("\n  !! LEAKAGE SENTINEL TRIPPED — AUC-PR is implausibly far above the no-skill line:")
        for h, f, k, m, pr in bad:
            log(f"     h={h} fold={f} {k}: AUC-PR {m:.3f} vs no-skill {pr:.3f}. "
                f"Check that node_features carries no future information and no "
                f"pre-aggregated neighbour counts.")
    else:
        log("\n  leakage sentinel: clean (no model is implausibly far above the no-skill line)")


def sid_init_probe(bundle, seeds, horizons, base_hazard):
    """Is the SI(D) head's deficit MECHANISTIC or just its initialisation?

    Root's convention `rates = zeros` is not neutral: softplus(0) = 0.693, so the
    head opens at a force of infection of 0.69 per census and predicts
    p = 1 - exp(-h * 0.69), i.e. 0.50 / 0.75 / 0.88 / 0.94 for h = 1..4, against a
    true event rate of 1.6 / 3.0 / 4.5 / 5.7%. The offset the optimiser has to
    unlearn therefore GROWS WITH h — and the measured deficit grows with h too.
    That is a testable prediction, so it gets tested: re-initialise lambda0 at the
    observed per-census hazard (the mechanistic equivalent of setting a logistic
    bias to the base rate) and re-run. Nothing else changes.

    The headline row keeps root's init. This probe is reported separately so that
    an initialisation change can never be mistaken for a modelling gain.

    ANSWER (measured, 4 seeds x 2 folds = n 8, run 2026-07-23):
        h=1  root-init -0.0115 0/8 [NEG]   neutral-init -0.0001 2/8 [INCONCLUSIVE]
        h=2  root-init -0.0230 0/8 [NEG]   neutral-init -0.0256 0/8 [NEG]
        h=3  root-init -0.0362 0/8 [NEG]   neutral-init -0.0459 0/8 [NEG]
        h=4  root-init -0.0426 0/8 [NEG]   neutral-init -0.0388 0/8 [NEG]
    The hypothesis is REFUTED for h >= 2: neutralising lambda0 removes the deficit
    only at h=1 and makes it no better (h=3: worse) further out. So the SI(D)
    deficit is NOT an initialisation artifact — the mechanistic head genuinely
    hurts on this panel. The likeliest mechanism is saturation: the head emits
    logit(1 - exp(-h * force)), which flattens as h grows, so it adds a
    strongly saturating monotone function of neighbour pressure to the logit and
    compresses the ranking among high-pressure palms. AUC-PR is rank-based, so
    compression at the top of the ranking is exactly what it punishes.
    """
    log(f"\n===== SI(D) INITIALISATION PROBE — lambda0 started at the observed "
        f"per-census hazard {base_hazard:.4f} instead of softplus(0)=0.693 "
        f"({seeds} seeds) =====")
    for h in horizons:
        d_root, d_neut, ref = [], [], []
        for f in range(len(bundle.folds)):
            for s in range(seeds):
                a_true, _ = train_eval(bundle, f, s, h, "STGNN", "true")
                a_root, _ = train_eval(bundle, f, s, h, "STGNN_SID", "true")
                a_neut, _ = train_eval(bundle, f, s, h, "STGNN_SID", "true",
                                       base_hazard=base_hazard)
                ref.append(a_true); d_root.append(a_root - a_true); d_neut.append(a_neut - a_true)
        gr, gn = paired(d_root), paired(d_neut)
        log(f"  h={h}: STGNN(true) {np.mean(ref):.3f} | SID-STGNN root-init "
            f"{gr['mean']:+.4f}+/-{gr['std']:.4f} {gr['pos']}/{gr['n']} [{gr['verdict']}]"
            f" | neutral-init {gn['mean']:+.4f}+/-{gn['std']:.4f} {gn['pos']}/{gn['n']} "
            f"[{gn['verdict']}]")
        rec(part="SIDPROBE", fold="pooled", h=h, model="SID(root init)-STGNN",
            metric="pairedAUCPR", note="rates=zeros, root convention", **gr)
        rec(part="SIDPROBE", fold="pooled", h=h, model="SID(neutral init)-STGNN",
            metric="pairedAUCPR", note=f"lambda0 init at base hazard {base_hazard:.4f}", **gn)


def load_existing():
    """Append mode: keep the rows the full run already wrote."""
    if not os.path.exists(OUT):
        return
    import csv as _csv
    with open(OUT, newline="") as fh:
        for r in _csv.DictReader(fh):
            ROWS.append({k: v for k, v in r.items() if v != ""})


def save():
    keys = ["part", "dataset", "fold", "h", "model", "metric", "mean", "std", "n", "pos",
            "verdict", "strata", "observed", "null_mean", "null_std", "excess", "z",
            "n_ge", "n_perm", "note"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in ROWS:
            w.writerow({k: r.get(k, "") for k in keys})
    log(f"\nsaved {OUT} ({len(ROWS)} rows)")


# --------------------------------------------------------------------------- #
PRIMARY_H = 3        # root run_experiment.py::decomposition() is h=3 only; mirror it

if __name__ == "__main__":
    # Budget note (measured, not guessed): one train_eval = ~3.3 s on this CPU, and the
    # grid is 5 configs x 2 folds. So the full 4 horizons x 20 seeds would be 800 calls
    # ~= 44 min. The budget is ~15 min, so the seeds are spent where the paper spends
    # them: the PRIMARY decomposition at h=3 gets the full seed count (mirroring
    # run_experiment.py::decomposition(), which is h=3 only), and h=1/2/4 run as a
    # horizon-robustness check at a smaller, EXPLICITLY REPORTED seed count.
    PROBE = len(sys.argv) > 1 and sys.argv[1] == "probe"
    if PROBE:
        sys.argv.pop(1)
    SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    SEEDS_SEC = int(os.environ.get("SEEDS_SEC", max(2, SEEDS // 5)))
    t0 = time.time()

    banner = ("REAL Eg9PP data (dataset.py)" if IS_REAL else
              "*** STUB SYNTHETIC DATA (_stub_dataset.py) — NOT A FINDING ***")
    log(f"dataset: {banner}")
    log(f"device: {DEVICE} | seeds={SEEDS} | epochs={EPOCHS} | WINDOW={WINDOW} | "
        f"N_REL={N_REL} | horizons={HORIZONS}")
    if not IS_REAL:
        log("!" * 78)
        log("!! Agent 1's dataset.py has not landed. Every AUC-PR below is a STUB number.")
        log("!" * 78)

    b = Bundle()
    log(f"  features (T,N,d) = {b.Fnp.shape} | mean degree {b.A_true.sum(1).mean():.2f} | "
        f"edges {int(b.A_true.sum()//2)} | adjacency scale {b.scale:.4f}")
    if hasattr(DS, "censored_in_horizon"):
        # Honest label limit: a palm that leaves observation inside (t, t+h] is labelled 0
        # although its fate is unknown. Same convention as DATASET_CARD.md's pos-rate table,
        # so the numbers stay comparable — but the size of the compromise is reported.
        log("  label caveat — negatives that are actually CENSORED inside the horizon: " +
            " | ".join(f"h={h}: {DS.censored_in_horizon(h)[0]}/{DS.censored_in_horizon(h)[1]}"
                       f" ({100*DS.censored_in_horizon(h)[0]/DS.censored_in_horizon(h)[1]:.2f}%)"
                       for h in HORIZONS))
    for n, mn, _ in CONFIGS:
        log(f"  {n:8s} = {mn:12s} params={M.count_params(M.build(mn, b.d))}")

    if PROBE:
        # per-census hazard implied by the h=1 task (events per at-risk palm-census)
        _, _, y1 = DS.build_examples(1, np.arange(b.T))
        bh = float(y1.mean())
        load_existing()
        sid_init_probe(b, SEEDS, HORIZONS, bh)
        save()
        log(f"\nTOTAL {time.time()-t0:.1f}s on {DEVICE}  [dataset = {TAG}, probe only]")
        sys.exit(0)

    hs_sec = tuple(h for h in HORIZONS if h != PRIMARY_H)
    log(f"\n===== PRIMARY DECOMPOSITION (h={PRIMARY_H}) — leave-one-parcel-out, "
        f"paired over (fold, seed), n = {len(b.folds)} x {SEEDS} = {len(b.folds)*SEEDS} =====")
    ap, pr = decomposition(b, SEEDS, (PRIMARY_H,))
    leak_sentinel(ap, pr, (PRIMARY_H,), len(b.folds))

    if hs_sec:
        log(f"\n===== HORIZON ROBUSTNESS (h={hs_sec}) — REDUCED seed count "
            f"{SEEDS_SEC} (n = {len(b.folds)*SEEDS_SEC} paired units), NOT {SEEDS}. "
            f"Read these as a shape check, not as the primary table. =====")
        ap2, pr2 = decomposition(b, SEEDS_SEC, hs_sec)
        leak_sentinel(ap2, pr2, hs_sec, len(b.folds))

    obs, prows = perm_null.run(n_perm=NPERM)
    for r in prows:
        rec(**r)

    save()
    log(f"\nTOTAL {time.time()-t0:.1f}s on {DEVICE}  [dataset = {TAG}]")
