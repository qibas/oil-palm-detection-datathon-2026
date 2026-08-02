"""Eg9PP v2 — tangga lokalitas + ablasi fitur. Menulis results_v2.csv.

Dua pertanyaan, keduanya muncul dari kelemahan yang terukur pada baseline.

EKSPERIMEN A — TANGGA LOKALITAS
  `STRUKTUR = true - random` pada baseline membandingkan 6 tetangga langsung
  (panjang sisi 1,0) lawan 6 pohon berjarak median 13,2 jarak tanam. Itu bisa jadi
  mengukur LOKAL vs GLOBAL, bukan PETA BENAR vs PETA SALAH. Kita naikkan kontrolnya
  bertingkat, derajat selalu dipertahankan:

      random        panjang sisi med 13,2   sisa sisi asli  1,0%   (global)
      random_local6 panjang sisi med  4,0   sisa sisi asli  6,7%
      random_local3 panjang sisi med  2,0   sisa sisi asli 22,0%   (paling ketat)

  Kalau selisih terhadap `true` bertahan sampai anak tangga terketat, model memang
  butuh peta yang BENAR. Kalau meluruh jadi nol, yang kita ukur adalah lokalitas —
  dan itu dilaporkan apa adanya.

EKSPERIMEN B — ABLASI FITUR (view selalu `true`)
  v1        24 kolom, difusi 1-hop            (baseline beku)
  +umur     28 kolom (umur inokulum + paparan kumulatif), 1-hop
  +2hop     28 kolom, difusi 1-hop DAN 2-hop  (n_rel 1 -> 2)

Metrik AUC-PR. Setiap komponen dipasangkan per (fold, seed) dan divonis dengan
harness yang sama seperti run_real.py: mean +/- std, sign-count, INCONCLUSIVE bila
di dalam 1 std.

    SEEDS=5 python run_v2.py     # penyaringan cepat
    SEEDS=20 python run_v2.py    # pelaporan
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
sys.path.insert(0, HERE)
import dataset as ds          # noqa: E402
import dataset_v2 as v2       # noqa: E402
import models_real as M       # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = int(os.environ.get("SEEDS", 5))
HORIZON = int(os.environ.get("H", 3))
EPOCHS = int(os.environ.get("EPOCHS", 60))
LR, WD = 5e-3, 1e-4
WINDOW = ds.WINDOW
OUT = os.environ.get("OUT") or os.path.join(HERE, "results_v2.csv")
ROWS = []


class STGNN_MR(M.STGNN):
    """STGNN multi-relasi. `models_real.STGNN` sengaja menolak n_rel != 1 (softmax
    atas satu relasi konstan 1,0 dan parameternya tak pernah dapat gradien). Untuk
    varian 2-hop relasinya jadi dua, jadi perhatian-relasi hidup kembali. Baseline
    beku tidak disentuh — kelas ini hanya hidup di runner v2."""

    def __init__(self, in_dim, hidden=M.HIDDEN, n_rel=2):
        super().__init__(in_dim, hidden, n_rel=1)
        self.attn = torch.nn.Parameter(torch.zeros(n_rel))
        self.n_rel = n_rel

    def encode(self, F_seq, D_seq):
        a = torch.softmax(self.attn, 0)
        h = torch.zeros(F_seq.shape[0], self.hidden, device=F_seq.device)
        for t in range(F_seq.shape[1]):
            agg = (D_seq[:, t] * a[None, :, None]).sum(1)
            h = self.gru(F.relu(self.w_self(F_seq[:, t]) + self.w_agg(agg)), h)
        return h


def log(*a):
    print(*a, flush=True)


def paired(delta):
    """Harness signifikansi — identik dengan run_real.py::paired."""
    d = np.asarray([x for x in delta if np.isfinite(x)], float)
    if d.size == 0:
        return float("nan"), float("nan"), 0, 0, "NA"
    m, s = float(d.mean()), float(d.std())
    pos = int((d > 0).sum())
    if abs(m) < s:
        v = "INCONCLUSIVE"
    else:
        v = "POS" if m > 0 else "NEG"
    return m, s, pos, d.size, v


def _focal(logits, y, gamma=2.0, alpha=0.75):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = torch.where(y > 0.5, p, 1 - p)
    w = torch.where(y > 0.5, alpha, 1 - alpha)
    return (w * (1 - pt) ** gamma * ce).mean()


def _windows(tree, t, Ft, D):
    tt = torch.as_tensor(t, device=Ft.device)
    ii = torch.as_tensor(tree, device=Ft.device)
    Fs, Dz = [], []
    for w in range(WINDOW):
        c = tt - (WINDOW - 1) + w
        Fs.append(Ft[c, ii])
        Dz.append(D[c, ii])
    return torch.stack(Fs, 1), torch.stack(Dz, 1)


def diffuse(Ft, mats, scale):
    """[T,N,d] x list of [N,N] -> [T,N,n_rel,d]."""
    return torch.stack(
        [torch.einsum("ij,tjd->tid", torch.as_tensor(A * scale, device=DEVICE), Ft)
         for A in mats], dim=2)


class Feat:
    """Satu varian fitur: tensor F, difusi per-view, contoh, dan lipatan."""

    def __init__(self, tag, use_v2, two_hop):
        self.tag, self.two_hop = tag, two_hop
        T = len(ds.census())
        self.folds = ds.folds()
        fn = v2.node_features_v2 if use_v2 else ds.node_features
        Xs = [np.asarray(fn(np.arange(T), train_nodes=np.asarray(m, bool)), np.float32)
              for m, _ in self.folds]
        assert all(np.allclose(Xs[0], x) for x in Xs[1:]), "scaler bocor antar-lipatan"
        self.Fnp = Xs[0]
        assert np.isfinite(self.Fnp).all()
        self.T, self.N, self.d = self.Fnp.shape
        self.Ft = torch.as_tensor(self.Fnp, device=DEVICE)
        A = np.asarray(ds.adjacency("true"), np.float32)
        tot = A.sum(1)
        self.scale = 1.0 / (tot.mean() + 1e-8)
        self.n_rel = 2 if two_hop else 1
        self._D = {}
        self.ex = {HORIZON: ds.build_examples(HORIZON, np.arange(self.T))}

    def D(self, view, seed):
        key = (view, seed if view.startswith("random") else 0)
        if key not in self._D:
            if view == "zero":
                self._D[key] = torch.zeros(self.T, self.N, self.n_rel, self.d,
                                           device=DEVICE)
            else:
                A = np.asarray(v2.view(view, seed=seed), np.float32)
                self._D[key] = diffuse(self.Ft, v2.relations(A, self.two_hop), self.scale)
        return self._D[key]


def train_eval(ft, fold, seed, model_name, view):
    tree, t, y = ft.ex[HORIZON]
    trm, tem = (np.asarray(m, bool) for m in ft.folds[fold])
    a, b = trm[tree], tem[tree]
    tr = (tree[a], t[a], y[a])
    te = (tree[b], t[b], y[b])
    if te[2].sum() == 0:
        return float("nan")
    D = ft.D(view, seed)
    torch.manual_seed(1234 + seed)
    if model_name == "MLPBaseline":
        model = M.MLPBaseline(in_dim=ft.d).to(DEVICE)
    else:
        model = (STGNN_MR(ft.d, n_rel=ft.n_rel) if ft.n_rel > 1
                 else M.STGNN(ft.d, n_rel=1)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    Ftr, Dtr = _windows(tr[0], tr[1], ft.Ft, D)
    ytr = torch.as_tensor(tr[2], device=DEVICE)
    Fte, Dte = _windows(te[0], te[1], ft.Ft, D)
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        _focal(model(Ftr, Dtr), ytr).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Fte, Dte)).cpu().numpy()
    return float(average_precision_score(te[2], p))


def sweep(ft, configs, seeds):
    """AP[cfg][(fold,seed)] untuk semua konfigurasi pada satu varian fitur."""
    ap = {c[0]: {} for c in configs}
    for fold in range(len(ft.folds)):
        for seed in seeds:
            for name, mdl, view in configs:
                ap[name][(fold, seed)] = train_eval(ft, fold, seed, mdl, view)
    return ap


def report(ap, comps, block, tag):
    keys = sorted(next(iter(ap.values())).keys())
    for label, hi, lo in comps:
        d = [ap[hi][k] - ap[lo][k] for k in keys]
        m, s, pos, n, v = paired(d)
        log(f"  {label:34s} {m:+.4f} +/- {s:.4f}  {pos:2d}/{n:2d}  {v}")
        ROWS.append(dict(block=block, varian=tag, h=HORIZON, komponen=label,
                         mean=round(m, 5), std=round(s, 5), n=n,
                         sign=f"{pos}/{n}", vonis=v))
    for name, vals in ap.items():
        a = np.asarray([vals[k] for k in keys], float)
        ROWS.append(dict(block=block + "_AP", varian=tag, h=HORIZON, komponen=name,
                         mean=round(float(a.mean()), 5), std=round(float(a.std()), 5),
                         n=a.size, sign="", vonis=""))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    seeds = list(range(SEEDS))
    t0 = time.time()
    log(f"device={DEVICE} | seeds={SEEDS} | h={HORIZON} | epochs={EPOCHS} | WINDOW={WINDOW}")

    log("\n=== diagnostik kontrol (derajat SELALU dipertahankan) ===")
    log(f"  {'view':>14} | {'sisa sisi asli':>14} | {'panjang med':>11} | maks")
    for nm in ("true", "random", "random_local6", "random_local3"):
        s = v2.edge_stats(v2.view(nm, seed=0))
        log(f"  {nm:>14} | {100*s['kept_frac']:13.1f}% | {s['len_med']:11.2f} | {s['len_max']:.1f}")

    # ---------------- EKSPERIMEN A — tangga lokalitas ----------------------
    log("\n=== A. TANGGA LOKALITAS (fitur v1, 1-hop) ===")
    base = Feat("v1", use_v2=False, two_hop=False)
    log(f"  fitur {base.Fnp.shape} | n_rel={base.n_rel}")
    cfgA = [("MLP", "MLPBaseline", "zero"), ("nograph", "STGNN", "zero"),
            ("random", "STGNN", "random"),
            ("rlocal6", "STGNN", "random_local6"),
            ("rlocal3", "STGNN", "random_local3"),
            ("true", "STGNN", "true")]
    apA = sweep(base, cfgA, seeds)
    report(apA, [
        ("temporal (nograph-MLP)", "nograph", "MLP"),
        ("prevalensi (random-nograph)", "random", "nograph"),
        ("STRUKTUR vs random GLOBAL", "true", "random"),
        ("STRUKTUR vs local r<=6", "true", "rlocal6"),
        ("STRUKTUR vs local r<=3  (paling ketat)", "true", "rlocal3"),
        ("total STGNN-MLP", "true", "MLP"),
    ], "TANGGA", "v1")

    # ---------------- EKSPERIMEN B — ablasi fitur --------------------------
    log("\n=== B. ABLASI FITUR (view=true) ===")
    variants = [("v1", False, False), ("+umur", True, False), ("+2hop", True, True)]
    apB = {}
    for tag, uv2, th in variants:
        ft = base if tag == "v1" else Feat(tag, uv2, th)
        log(f"  {tag}: fitur d={ft.d} n_rel={ft.n_rel}")
        cfg = [("MLP", "MLPBaseline", "zero"), ("true", "STGNN", "true"),
               ("rlocal3", "STGNN", "random_local3")]
        a = apA if tag == "v1" else sweep(ft, cfg, seeds)
        apB[tag] = a
        report({k: a[k] for k in ("MLP", "true", "rlocal3")}, [
            ("STRUKTUR vs local r<=3", "true", "rlocal3"),
            ("total STGNN-MLP", "true", "MLP"),
        ], "FITUR", tag)

    keys = sorted(apB["v1"]["true"].keys())
    log("\n  AP view=true, dipasangkan terhadap v1:")
    for tag, _, _ in variants:
        a = np.asarray([apB[tag]["true"][k] for k in keys], float)
        b = np.asarray([apB["v1"]["true"][k] for k in keys], float)
        m, s, pos, n, v = paired(a - b)
        log(f"    {tag:7s} AP {a.mean():.4f} +/- {a.std():.4f}"
            + ("" if tag == "v1" else f"   delta {m:+.4f} +/- {s:.4f}  {pos}/{n}  {v}"))
        if tag != "v1":
            ROWS.append(dict(block="FITUR_delta", varian=tag, h=HORIZON,
                             komponen=f"AP({tag}) - AP(v1), view=true",
                             mean=round(m, 5), std=round(s, 5), n=n,
                             sign=f"{pos}/{n}", vonis=v))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ROWS[0].keys()))
        w.writeheader()
        w.writerows(ROWS)
    log(f"\n-> {OUT} ({len(ROWS)} baris) | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
