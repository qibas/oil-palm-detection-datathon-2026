"""STUB dataset for `layer2_real/` — **SYNTHETIC, NOT THE REAL Eg9PP DATA**.

Purpose: let Agent 3 (models + harness) develop and smoke-test `models_real.py`
and `run_real.py` against the LOCKED contract in `INTERFACE.md` while Agent 1
implements the real `dataset.py` in parallel.

Every number produced with this module is a **stub number**. It exists to prove
the harness runs and the shapes line up. It must NEVER be reported as a finding.
`_ds.py` prefers the real `dataset.py` and only falls back here.

Shapes reproduced from `../data_clean/DATASET_CARD.md`:
    N = 1200 palms, 2 parcels x 600, triangular lattice (6 NN at distance 1.0)
    T = 45 censuses, irregular dates
    root-contact graph at r = 1.5 x planting distance, mean degree ~5.6,
    ZERO cross-parcel edges (this is what makes leave-one-parcel-out valid)
    14 families planted in plot-sized blocks, all 14 present in both parcels
    ever-symptomatic ~58.5%, dead ~30.5%, ~498 never symptomatic (censored)

Feature-design note that the REAL dataset.py must also satisfy (see the report):
`node_features` has to carry a per-tree, past-only STATUS indicator (symptomatic
at t / dead at t). Within the risk set that column is constant 0 for the tree
itself, so it cannot leak its own label — but it is the ONLY thing the diffusion
D = A @ F can carry. Without it the neighbour tensor holds no epidemic
information at all and the entire graph ablation is vacuous. Conversely, the
features must NOT contain pre-aggregated neighbour counts (`n_nb_sympt`,
`n_nb_dead` from the panel), or the MLP gets the graph for free and every graph
component of the decomposition collapses to ~0 for the wrong reason.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

IS_STUB = True

# ---- contract constants ---------------------------------------------------
WINDOW = 3
HORIZONS = (1, 2, 3, 4)
N_REL = 1

# ---- stub geometry --------------------------------------------------------
_COLS, _ROWS = 50, 12          # per parcel -> 600 palms
_PARCELS = ("44A", "44B")
_N_FAM = 14
_T = 45
_COS30 = np.cos(np.pi / 6)
_R_CONTACT = 1.5

_CACHE = {}


# ---------------------------------------------------------------------------
def _build():
    if "nodes" in _CACHE:
        return _CACHE
    rng = np.random.default_rng(20260723)

    # ---- lattice: columns spaced cos30 in x, alternating 0.5 offset in y ---
    rows = []
    for p, pname in enumerate(_PARCELS):
        for c in range(_COLS):
            for r in range(_ROWS):
                x = c * _COS30
                y = r * 1.0 + (0.5 if c % 2 else 0.0) + p * 20.0
                plot = p * 40 + (c // 5) * 4 + (r // 3)      # 15 palms per plot
                rows.append((pname, plot, x, y))
    nodes = pd.DataFrame(rows, columns=["parcel", "plot", "xm", "ym"])
    # families planted in plot blocks (the genotype/space confound this stub exists to mimic)
    fam_of_plot = {pl: int(pl % _N_FAM) for pl in nodes["plot"].unique()}
    nodes["fam"] = nodes["plot"].map(fam_of_plot)
    nodes["progeny"] = ["FAM%02d" % f for f in nodes.fam]
    nodes["palm_id"] = ["stub_%04d" % i for i in range(len(nodes))]
    nodes["fold"] = nodes.parcel
    N = len(nodes)

    # ---- root-contact graph, r = 1.5 x planting distance -------------------
    from scipy.spatial import cKDTree
    xy = nodes[["xm", "ym"]].to_numpy()
    pairs = cKDTree(xy).query_pairs(_R_CONTACT, output_type="ndarray")
    par = nodes.parcel.to_numpy()
    assert (par[pairs[:, 0]] != par[pairs[:, 1]]).sum() == 0, "stub leaked a cross-parcel edge"
    A = np.zeros((N, N), np.float32)
    A[pairs[:, 0], pairs[:, 1]] = 1.0
    A[pairs[:, 1], pairs[:, 0]] = 1.0
    nb = [np.flatnonzero(A[i]) for i in range(N)]
    deg = A.sum(1)

    # ---- irregular census grid --------------------------------------------
    census = np.sort(rng.choice(np.arange(0.5, 26.0, 0.25), size=_T, replace=False))

    # ---- epidemic: family susceptibility + local contagion -----------------
    fam_s = np.linspace(0.5, 1.6, _N_FAM)[nodes.fam.to_numpy()]
    a0, a1 = 0.0043, 0.021                     # baseline / neighbour force per census
    #                                            (calibrated to ~58.5% ever-symptomatic)
    # censoring: most palms observed to the end, a tail leaves early
    cens = np.full(N, _T - 1)
    early = rng.random(N) < 0.28
    cens[early] = rng.integers(10, _T - 1, early.sum())

    t1s = np.full(N, 10 ** 9)
    sympt = np.zeros(N, bool)
    for t in range(_T):
        alive_obs = (t <= cens)
        nb_s = A @ sympt.astype(np.float32)
        hz = 1.0 - np.exp(-(a0 * fam_s + a1 * nb_s))
        hit = (~sympt) & alive_obs & (rng.random(N) < hz)
        t1s[hit] = t
        sympt |= hit
    ever = t1s < 10 ** 9
    dies = ever & (rng.random(N) < 0.52)
    td = np.where(dies, t1s + rng.integers(1, 5, N), 10 ** 9)

    # ---- status matrix (T, N): 0=A 1=S 2=D 3=C ----------------------------
    st = np.zeros((_T, N), np.int8)
    for t in range(_T):
        s = np.zeros(N, np.int8)
        s[(t < t1s)] = 0
        s[(t >= t1s) & (t < td)] = 1
        s[(t >= td)] = 2
        s[t > cens] = 3                        # out of observation -> NOT healthy
        st[t] = s
    _A_, _S_, _D_, _C_ = 0, 1, 2, 3

    # ---- panel (long) ------------------------------------------------------
    ids = nodes.palm_id.to_numpy()
    symp_m = (st == _S_) | (st == _D_)
    obs_m = st != _C_
    frames = []
    for t in range(_T):
        frames.append(pd.DataFrame({
            "palm_id": ids, "t": census[t],
            "status": np.array(["A", "S", "D", "C"])[st[t]],
            "at_risk": (st[t] == _A_).astype(int),
            "deg": deg.astype(int),
            "n_nb_obs": [int(obs_m[t][v].sum()) for v in nb],
            "n_nb_sympt": [int(symp_m[t][v].sum()) for v in nb],
            "n_nb_dead": [int((st[t][v] == _D_).sum()) for v in nb],
            "prev_global": float(symp_m[t][obs_m[t]].mean()) if obs_m[t].any() else np.nan,
        }))
    panel = pd.concat(frames, ignore_index=True)
    edges = pd.DataFrame({"a": ids[pairs[:, 0]], "b": ids[pairs[:, 1]],
                          "parcel": par[pairs[:, 0]]})

    _CACHE.update(nodes=nodes, panel=panel, edges=edges, A=A, st=st,
                  census=census, N=N, deg=deg, pairs=pairs, par=par)
    return _CACHE


# ---------------------------------------------------------------------------
def load():
    c = _build()
    return c["nodes"], c["panel"], c["edges"]


def census():
    return _build()["census"]


def node_features(train_t):
    """(T, N, d) float32, past-only. Scaler fit on `train_t` censuses only."""
    c = _build()
    st, N, T = c["st"], c["N"], _T
    nodes, deg = c["nodes"], c["deg"]
    sym = ((st == 1) | (st == 2)).astype(np.float32)
    dead = (st == 2).astype(np.float32)
    cen = (st == 3).astype(np.float32)
    obs = (st != 3).astype(np.float32)
    prev = np.array([sym[t][obs[t] > 0].mean() if (obs[t] > 0).any() else 0.0
                     for t in range(T)], np.float32)
    dprev = np.diff(prev, prepend=prev[0])
    date = c["census"].astype(np.float32)

    fam = nodes.fam.to_numpy()
    onehot = np.zeros((N, _N_FAM), np.float32)
    onehot[np.arange(N), fam] = 1.0
    parcel = (nodes.parcel.to_numpy() == "44B").astype(np.float32)
    vigor = np.random.default_rng(7).normal(size=N).astype(np.float32)

    cols = []
    for t in range(T):
        tm1 = max(t - 1, 0)
        base = np.stack([
            sym[t], dead[t], cen[t],                       # own state at t (const 0 in risk set)
            sym[tm1], dead[tm1],                           # own state at t-1
            np.full(N, prev[t], np.float32),
            np.full(N, dprev[t], np.float32),
            np.full(N, date[t] / 25.0, np.float32),
            np.full(N, t / (T - 1), np.float32),
            (deg / 6.0).astype(np.float32),
            parcel, vigor,
        ], axis=1)
        cols.append(np.concatenate([base, onehot], axis=1))
    X = np.stack(cols).astype(np.float32)                  # (T, N, d)

    tr = np.asarray(train_t, int)
    sc = StandardScaler().fit(X[tr].reshape(-1, X.shape[2]))
    return sc.transform(X.reshape(-1, X.shape[2])).reshape(X.shape).astype(np.float32)


def _rewire_within_parcel(pairs, par, N, rng, n_swap_mult=12):
    """Degree-preserving double-edge swap, restricted to WITHIN-parcel edges.

    Keeping the swap inside a parcel is deliberate: the `random` view must
    destroy local structure WITHOUT creating cross-parcel edges, otherwise the
    leave-one-parcel-out fold stops being a clean block and the `prevalence`
    component would be measuring fold leakage instead of delocalised smoothing.
    """
    E = pairs.copy()
    have = {(min(a, b), max(a, b)) for a, b in E}
    grp = {}
    for k, (a, b) in enumerate(E):
        grp.setdefault(par[a], []).append(k)
    for _, idx in grp.items():
        idx = np.asarray(idx)
        for _ in range(n_swap_mult * idx.size):
            i, j = rng.choice(idx, 2, replace=False)
            a, b = E[i]
            c, d = E[j]
            if rng.random() < 0.5:
                c, d = d, c
            if len({a, b, c, d}) < 4:
                continue
            n1, n2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
            if n1 in have or n2 in have:
                continue
            have.discard((min(a, b), max(a, b)))
            have.discard((min(c, d), max(c, d)))
            have.add(n1)
            have.add(n2)
            E[i] = (a, d)
            E[j] = (c, b)
    A = np.zeros((N, N), np.float32)
    A[E[:, 0], E[:, 1]] = 1.0
    A[E[:, 1], E[:, 0]] = 1.0
    return A


def adjacency(view, seed=0):
    c = _build()
    N = c["N"]
    if view == "zero":
        return np.zeros((N, N), np.float32)
    if view == "true":
        return c["A"].copy()
    rng = np.random.default_rng(700 + seed)
    R = _rewire_within_parcel(c["pairs"], c["par"], N, rng)
    if view == "random":
        return R
    raise ValueError(view)


def perturb(eps, seed=0):
    c = _build()
    return ((1 - eps) * c["A"] + eps * adjacency("random", seed)).astype(np.float32)


def build_examples(h, cycles):
    """(tree_idx, t_idx, y). Risk set = status 'A' at t. y=1 if the palm turns
    S or D within (t, t+h]. Censored ('C') palms are OUT of the risk set and are
    NOT counted as negatives."""
    c = _build()
    st, N = c["st"], c["N"]
    T = _T
    trees, ts, ys = [], [], []
    for t in np.asarray(cycles, int):
        if t < WINDOW - 1 or t + h >= T:
            continue
        risk = st[t] == 0
        fut = ((st[t + 1:t + h + 1] == 1) | (st[t + 1:t + h + 1] == 2)).any(0)
        idx = np.flatnonzero(risk)
        trees.append(idx)
        ts.append(np.full(idx.size, t))
        ys.append(fut[idx].astype(np.float32))
    if not trees:
        return np.array([], int), np.array([], int), np.array([], np.float32)
    return np.concatenate(trees), np.concatenate(ts), np.concatenate(ys)


def folds():
    """Leave-one-parcel-out, 2 folds. Fold k HOLDS OUT parcel k."""
    c = _build()
    par = c["nodes"].parcel.to_numpy()
    out = []
    for p in _PARCELS:
        te = par == p
        out.append((~te, te))
    return out


if __name__ == "__main__":
    n, p, e = load()
    c = _build()
    st = c["st"]
    ever = ((st == 1) | (st == 2)).any(0)
    print(f"[STUB] N={len(n)} T={_T} edges={len(e)} mean_deg={c['deg'].mean():.2f}")
    print(f"[STUB] ever-symptomatic {ever.mean()*100:.1f}%  dead {(st==2).any(0).mean()*100:.1f}%"
          f"  never-sympt {int((~ever).sum())}")
    for h in HORIZONS:
        _, _, y = build_examples(h, np.arange(_T))
        print(f"[STUB] h={h}: {y.size} examples, {int(y.sum())} pos ({100*y.mean():.2f}%)")
    print("[STUB] feature dim =", node_features(np.arange(_T)).shape)
