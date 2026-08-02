"""Premise test: are the real Unhealthy palms spatially CLUSTERED?

The whole spread-modelling premise assumes an unhealthy palm raises its neighbours'
risk. With one survey date we cannot test transmission over time, but we CAN test the
spatial fingerprint it would leave: contact-driven disease clusters, random stress does
not.

Statistic = join count, i.e. the number of graph edges whose BOTH endpoints are
Unhealthy, on the real contact graph (root radius 13 m over reconstructed positions).
Null = random relabelling holding the Unhealthy count fixed. Also reports Moran's I.

Read-only. Consumes out/crowns_B_*.npy written by stitch_probe.py.
"""
import os, glob
import numpy as np
from grids import OUT

R_ROOT_M = 13.0
N_PERM = 9999
RNG = np.random.default_rng(0)


def adjacency(xy_m, radius=R_ROOT_M):
    """Binary symmetric contact graph, no self loops."""
    n = len(xy_m)
    A = np.zeros((n, n), bool)
    CH = 2000
    for s in range(0, n, CH):
        e = min(s + CH, n)
        d = np.sqrt(((xy_m[s:e, None, :] - xy_m[None, :, :]) ** 2).sum(-1))
        A[s:e] = d <= radius
    np.fill_diagonal(A, False)
    return A


def join_count(A, z):
    """Number of edges with both endpoints labelled 1 (z is 0/1)."""
    return float(z @ A @ z / 2.0)


def morans_i(A, z):
    w = A.astype(float)
    n = len(z)
    x = z - z.mean()
    denom = (x ** 2).sum()
    if denom == 0 or w.sum() == 0:
        return np.nan
    return (n / w.sum()) * float(x @ w @ x) / denom


def perm_test(A, z, stat, n_perm=N_PERM):
    obs = stat(A, z)
    null = np.empty(n_perm)
    zc = z.copy()
    for i in range(n_perm):
        RNG.shuffle(zc)
        null[i] = stat(A, zc)
    # one-sided (clustering => larger than null), +1 correction
    p = (1.0 + (null >= obs).sum()) / (n_perm + 1.0)
    return obs, float(null.mean()), float(null.std()), p


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(OUT, "crowns_B_*.npy")))
    if not files:
        raise SystemExit("run stitch_probe.py first")

    print(f"contact radius = {R_ROOT_M} m | permutations = {N_PERM}\n")
    for f in files:
        d = np.load(f, allow_pickle=True).item()
        ortho = os.path.basename(f)[len("crowns_B_"):-len(".npy")]
        xy = d["xy_px"] * d["gsd"]
        z = (d["label"] == "Unhealthy").astype(float)
        A = adjacency(xy)

        deg = A.sum(1)
        obs, mu, sd, p = perm_test(A, z, join_count)
        mi_obs, mi_mu, mi_sd, mi_p = perm_test(A, z, morans_i)

        print(f"ortho {ortho}:  n={len(z):5d}  Unhealthy={int(z.sum()):3d} "
              f"({100*z.mean():.2f}%)  mean_degree={deg.mean():.2f}")
        print(f"    join count (UU edges) : obs={obs:6.1f}  null={mu:6.2f}+/-{sd:5.2f}"
              f"   p={p:.4f}")
        print(f"    Moran's I             : obs={mi_obs:+.4f} null={mi_mu:+.4f}+/-{mi_sd:.4f}"
              f"   p={mi_p:.4f}")
        z_sc = (obs - mu) / sd if sd > 0 else np.nan
        print(f"    -> join count is {z_sc:+.2f} sd from random"
              f"  [{'CLUSTERED' if p < 0.05 else 'no detectable clustering'}]\n")
