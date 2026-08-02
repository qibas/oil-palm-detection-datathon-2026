"""Genotype-controlled permutation null for local contagion in the Eg9PP panel.

The question: an at-risk palm whose root-contact neighbours are already
symptomatic — is it really at higher risk, or does it only *look* that way
because Eg9PP plants full-sib families in contiguous plot blocks, so a
susceptible palm's neighbours are its own susceptible relatives?

Statistic (no model, pure data):

    RR = P(event in (t, t+h] | >=1 symptomatic neighbour at t)
       / P(event in (t, t+h] | 0 symptomatic neighbours at t)

    over the risk set (status 'A' at census t). "symptomatic neighbour" means
    status S or D — a dead palm's stump remains an inoculum source, and the
    frozen panel's `n_nb_sympt` uses the same S-or-D definition (this module
    recomputes it from the status matrix and ASSERTS agreement with the panel
    column, so the recomputation is verified, not assumed).

    Reported pooled and Mantel-Haenszel-stratified by census, because both
    exposure and hazard rise over the 25 years and a pooled ratio alone is
    open to a Simpson reversal.

Null: permute whole per-palm trajectories (the entire status column, censoring
included) between palms **within a stratum**, holding the lattice fixed.

    global          naive null. Wrong here, and included precisely to show how
                    wrong: it destroys the family blocks along with the local
                    structure, so all of the family-clustering signal is
                    credited to space.
    progeny         *** the prescribed null *** — permute only among palms of
                    the same family. Family composition of every neighbourhood
                    is preserved, so genotype susceptibility is held fixed and
                    only the fine-grained spatial arrangement is destroyed.
    progeny+parcel  stricter: also holds the parcel fixed, so a trajectory
                    cannot move between two blocks with different observation
                    coverage.
    plot            strictest: permute only inside a planted plot. Since most
                    root contacts are within-plot this removes most of the
                    signal by construction; it is a floor, not the headline.

A column permutation preserves the global epidemic curve EXACTLY (the multiset
of trajectories is unchanged), and a within-family permutation additionally
preserves the per-family per-census counts. So the null and the observation
share their temporal marginals by construction; no extra time adjustment is
needed for the comparison to be fair.

Runs standalone (`python perm_null.py`) and is also called by `run_real.py`.
Reads the FROZEN `../data_clean/layer2_*.csv` directly: `load()` in the contract
is defined as "the DataFrames as they are", so there is no design freedom here
and no dependency on the feature engineering in `dataset.py`.
"""
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(os.path.dirname(HERE), "data_clean")

WINDOW = 3          # keep the census range identical to the forecasting task
N_PERM = 200
STRATA = ("global", "progeny", "progeny_parcel", "plot")
PRIMARY = "progeny"


# ---------------------------------------------------------------------------
def load_frozen():
    nodes = pd.read_csv(os.path.join(CLEAN, "layer2_nodes.csv"))
    panel = pd.read_csv(os.path.join(CLEAN, "layer2_panel.csv"))
    edges = pd.read_csv(os.path.join(CLEAN, "layer2_edges.csv"))
    return nodes, panel, edges


def build(nodes, panel, edges):
    ids = nodes.palm_id.to_numpy()
    pos = {p: i for i, p in enumerate(ids)}
    N = len(ids)

    census = np.sort(panel.t.unique())
    T = len(census)
    piv = panel.pivot(index="palm_id", columns="t", values="status").reindex(ids)
    st = piv.to_numpy().T                                   # (T, N) of 'A'/'S'/'D'/'C'

    A = np.zeros((N, N), np.float32)
    ia = np.array([pos[a] for a in edges.a])
    ib = np.array([pos[b] for b in edges.b])
    A[ia, ib] = 1.0
    A[ib, ia] = 1.0

    sym = ((st == "S") | (st == "D")).astype(np.float32)     # (T, N)

    # verify the recomputed neighbour count against the frozen panel column
    nb_chk = sym @ A
    pan_chk = panel.pivot(index="palm_id", columns="t", values="n_nb_sympt")\
                   .reindex(ids).to_numpy().T.astype(np.float32)
    assert np.allclose(nb_chk, pan_chk), "recomputed n_nb_sympt disagrees with the frozen panel"
    return dict(ids=ids, N=N, T=T, census=census, st=st, A=A, sym=sym, nodes=nodes)


# ---------------------------------------------------------------------------
def _rates(st, sym, A, h, window=WINDOW):
    """Per-census 2x2 counts. Returns (n_exp, ev_exp, n_unexp, ev_unexp) arrays."""
    T, N = st.shape
    nb = sym @ A                                            # (T, N) symptomatic neighbours
    risk = st == "A"
    ts = [t for t in range(T) if t >= window - 1 and t + h < T]
    n1 = np.zeros(len(ts)); a1 = np.zeros(len(ts))
    n0 = np.zeros(len(ts)); a0 = np.zeros(len(ts))
    for k, t in enumerate(ts):
        ev = sym[t + 1:t + 1 + h].max(0) > 0
        m = risk[t]
        e = m & (nb[t] >= 1)
        u = m & (nb[t] < 1)
        n1[k], a1[k] = e.sum(), ev[e].sum()
        n0[k], a0[k] = u.sum(), ev[u].sum()
    return n1, a1, n0, a0


def _rr(n1, a1, n0, a0):
    """(pooled RR, Mantel-Haenszel RR)."""
    p1 = a1.sum() / max(n1.sum(), 1)
    p0 = a0.sum() / max(n0.sum(), 1)
    pooled = p1 / p0 if p0 > 0 else np.nan
    n = n1 + n0
    ok = n > 0
    num = (a1[ok] * n0[ok] / n[ok]).sum()
    den = (a0[ok] * n1[ok] / n[ok]).sum()
    mh = num / den if den > 0 else np.nan
    return float(pooled), float(mh)


def observed(d, horizons=(1, 2, 3, 4)):
    out = {}
    for h in horizons:
        n1, a1, n0, a0 = _rates(d["st"], d["sym"], d["A"], h)
        pooled, mh = _rr(n1, a1, n0, a0)
        out[h] = dict(RR=pooled, RR_MH=mh,
                      n_exp=int(n1.sum()), ev_exp=int(a1.sum()),
                      n_unexp=int(n0.sum()), ev_unexp=int(a0.sum()),
                      p_exp=float(a1.sum() / max(n1.sum(), 1)),
                      p_unexp=float(a0.sum() / max(n0.sum(), 1)))
    return out


def _strata_key(nodes, kind):
    if kind == "global":
        return np.zeros(len(nodes), int)
    if kind == "progeny":
        k = nodes.progeny.astype(str)
    elif kind == "progeny_parcel":
        k = nodes.progeny.astype(str) + "|" + nodes.parcel.astype(str)
    elif kind == "plot":
        k = nodes.parcel.astype(str) + "|" + nodes["plot"].astype(str)
    else:
        raise ValueError(kind)
    return pd.factorize(k)[0]


def permute(d, kind, rng):
    """Permute whole trajectories within strata; the lattice never moves."""
    key = _strata_key(d["nodes"], kind)
    perm = np.arange(d["N"])
    for g in np.unique(key):
        idx = np.flatnonzero(key == g)
        perm[idx] = rng.permutation(idx)
    return d["st"][:, perm], d["sym"][:, perm]


def run(n_perm=N_PERM, horizons=(1, 2, 3, 4), strata=STRATA, seed=0, verbose=True):
    t0 = time.time()
    d = build(*load_frozen())
    obs = observed(d, horizons)
    key_sizes = {k: np.bincount(_strata_key(d["nodes"], k)) for k in strata}

    if verbose:
        print(f"\n===== GENOTYPE-CONTROLLED PERMUTATION NULL (REAL Eg9PP, "
              f"{d['N']} palms x {d['T']} censuses) =====")
        print(f"  strata sizes: " + " | ".join(
            f"{k}: {len(v)} strata, median {int(np.median(v))}" for k, v in key_sizes.items()))
        print("  observed neighbour risk ratio (>=1 symptomatic neighbour vs 0):")
        for h in horizons:
            o = obs[h]
            print(f"    h={h}: RR {o['RR']:.2f}x  (MH {o['RR_MH']:.2f}x)  "
                  f"exposed {o['ev_exp']}/{o['n_exp']} = {100*o['p_exp']:.2f}%  "
                  f"unexposed {o['ev_unexp']}/{o['n_unexp']} = {100*o['p_unexp']:.2f}%")

    rows, null = [], {}
    for kind in strata:
        rng = np.random.default_rng(1000 + seed)
        draws = {h: {"RR": [], "RR_MH": []} for h in horizons}
        for _ in range(n_perm):
            stp, symp = permute(d, kind, rng)
            for h in horizons:
                pooled, mh = _rr(*_rates(stp, symp, d["A"], h))
                draws[h]["RR"].append(pooled)
                draws[h]["RR_MH"].append(mh)
        null[kind] = draws
        if verbose:
            print(f"  --- null: permute within {kind} ({n_perm} permutations) ---")
        for h in horizons:
            for metric in ("RR", "RR_MH"):
                v = np.asarray(draws[h][metric], float)
                v = v[np.isfinite(v)]
                o = obs[h][metric]
                mu, sd = float(v.mean()), float(v.std())
                excess = o / mu if mu > 0 else np.nan
                z = (o - mu) / sd if sd > 0 else np.nan
                ge = int((v >= o).sum())
                rows.append(dict(part="PERMNULL", strata=kind, h=h, metric=metric,
                                 observed=o, null_mean=mu, null_std=sd,
                                 excess=excess, z=z, n_ge=ge, n_perm=int(v.size)))
                if verbose and metric == "RR":
                    print(f"    h={h} RR   : obs {o:.2f}x vs null {mu:.2f}+/-{sd:.2f}  "
                          f"excess {excess:.2f}x  z={z:+.1f}  perms>=obs {ge}/{v.size}")
                elif verbose:
                    print(f"    h={h} RR_MH: obs {o:.2f}x vs null {mu:.2f}+/-{sd:.2f}  "
                          f"excess {excess:.2f}x  z={z:+.1f}  perms>=obs {ge}/{v.size}")

    if verbose:
        print(f"  [{time.time()-t0:.1f}s]  PRIMARY null = within-{PRIMARY}. The `global` row is "
              f"the naive null and is expected to be too small — it is a control, not a result.")
    return obs, rows


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_PERM
    run(n_perm=n)
