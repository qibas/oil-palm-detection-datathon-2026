"""Null permutasi dalam-famili untuk varian foto-tunggal (v3).

    python run_v3_perm.py [n_perm] [seeds]      # default 50 permutasi x 2 seed

Menulis `results_v3_perm.csv`.

PERTANYAANNYA.

`run_v3.py` menunjukkan 77% kemampuan model foto datang khusus dari PETA KONTAK
YANG BENAR. Tetapi v3 membuang genotipe, dan Eg9PP menanam famili sekandung dalam
blok petak yang bersambung. Jadi ada tafsir tandingan: bukan penularan, melainkan
seekor pohon rentan dikelilingi kerabatnya yang sama rentannya.

NULL YANG DIPAKAI, DAN CARA MEMBACANYA.

Lintasan per-pohon (seluruh kolom status, termasuk penyensoran) dipertukarkan
**hanya antar pohon sefamili**; kisinya tidak pernah bergerak. Komposisi famili
setiap ketetanggaan karena itu DIPERTAHANKAN - kerentanan genotipe dipegang tetap -
dan yang dihancurkan hanya susunan spasial halusnya.

    kemampuan RUNTUH di bawah null   -> yang diukur adalah susunan spasial,
                                        yaitu penularan.  BUKTI MENDUKUNG.
    kemampuan BERTAHAN di bawah null -> model hanya membaca "ketetanggaan ini
                                        famili rentan". Efek graf v3 tercemar.

Perhatikan arah bacanya: di sini kemampuan yang bertahan adalah kabar BURUK.
Itu kebalikan dari kebanyakan uji permutasi, dan gampang salah baca.

PENGAMAN.

Modul ini membangun ulang blok STATE dan daftar contoh dari matriks status, bukan
memanggil `dataset.py` - karena status yang dipermutasi tidak ada di CSV beku.
Rekonstruksi itu WAJIB bit-identik dengan `dataset.py` pada data tak-terpermutasi;
`_verify_reimplementation()` mengasersinya sebelum satu permutasi pun dijalankan.
Kalau gagal, seluruh null tidak sah dan skrip berhenti.
"""
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import dataset as ds            # noqa: E402
import dataset_v3 as v3         # noqa: E402
import models_real as M         # noqa: E402
import perm_null as PN          # noqa: E402
import run_real as R            # noqa: E402
import run_v3 as V3             # noqa: E402

H = V3.H
EPOCHS = V3.EPOCHS
MIN_POS = V3.MIN_POS
STRATA = os.environ.get("STRATA", "progeny")
# Nama berkas memuat strata: dua strata adalah dua uji berbeda, dan menulis
# keduanya ke satu nama akan membuat yang kedua diam-diam menimpa yang pertama.
OUT = os.path.join(HERE, "results_v3_perm_%s.csv" % STRATA)
DEVICE = R.DEVICE


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
def state_from_status(st):
    """(T,N) status -> (T,N,6) blok STATE. Cermin `dataset.raw_features`."""
    is_sympt = np.isin(st, ds.POS_STATUS).astype(np.float64)
    is_dead = (st == "D").astype(np.float64)
    is_cens = (st == "C").astype(np.float64)
    lvl = np.stack([is_sympt, is_dead, is_cens], axis=-1)
    dlt = np.zeros_like(lvl)
    dlt[1:] = lvl[1:] - lvl[:-1]                # beda hanya-masa-lalu
    return np.concatenate([lvl, dlt], axis=-1).astype(np.float32)


def examples_from_status(st, h, window=ds.WINDOW):
    """(tree, t, y). Cermin `dataset.build_examples`, termasuk aturan t yang sah."""
    T, N = st.shape
    trees, ts, ys = [], [], []
    for t in range(T):
        if t < window - 1 or t + h >= T:
            continue
        at_risk = st[t] == ds.RISK_STATUS
        fut = np.isin(st[t + 1:t + h + 1], ds.POS_STATUS).any(0)
        idx = np.flatnonzero(at_risk)
        trees.append(idx)
        ts.append(np.full(idx.size, t))
        ys.append(fut[idx].astype(np.int64))
    return (np.concatenate(trees), np.concatenate(ts),
            np.concatenate(ys).astype(np.float32))


def _verify_reimplementation(st):
    """Tanpa ini, null-nya tidak berarti apa-apa."""
    mine = state_from_status(st)
    theirs = v3.node_features_v3(np.arange(len(ds.census())))
    assert mine.shape == theirs.shape, (mine.shape, theirs.shape)
    assert np.array_equal(mine, theirs), \
        "blok STATE hasil rekonstruksi BERBEDA dari dataset.py — null tidak sah"
    a = examples_from_status(st, H)
    b = ds.build_examples(H, np.arange(len(ds.census())))
    for k, (x, y) in enumerate(zip(a, b)):
        assert np.array_equal(np.asarray(x), np.asarray(y)), \
            "daftar contoh berbeda pada elemen %d — null tidak sah" % k
    return mine.shape, a[0].shape


# --------------------------------------------------------------------------- #
def score_once(state, ex, A_scaled, folds, seeds):
    """Rata-rata AP dalam-sensus atas (lipatan x seed) untuk satu matriks status."""
    Ft = torch.as_tensor(state, device=DEVICE)
    D = R.diffuse(Ft, A_scaled)
    tree, t_idx, y = ex
    out = []
    for fold, (trm, tem) in enumerate(folds):
        trm, tem = np.asarray(trm, bool), np.asarray(tem, bool)
        a, b = trm[tree], tem[tree]
        if y[a].size == 0 or y[b].sum() == 0:
            continue
        ii_tr = torch.as_tensor(tree[a], device=DEVICE, dtype=torch.long)
        tt_tr = torch.as_tensor(t_idx[a], device=DEVICE, dtype=torch.long)
        ii_te = torch.as_tensor(tree[b], device=DEVICE, dtype=torch.long)
        tt_te = torch.as_tensor(t_idx[b], device=DEVICE, dtype=torch.long)
        Ftr = Ft[tt_tr, ii_tr].unsqueeze(1)          # window = 1
        Dtr = D[tt_tr, ii_tr].unsqueeze(1)
        Fte = Ft[tt_te, ii_te].unsqueeze(1)
        Dte = D[tt_te, ii_te].unsqueeze(1)
        ytr = torch.as_tensor(y[a], device=DEVICE)
        for seed in range(seeds):
            torch.manual_seed(1234 + seed)
            model = M.build("STGNN", state.shape[2], horizon=H).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=R.LR, weight_decay=R.WD)
            model.train()
            for _ in range(EPOCHS):
                opt.zero_grad()
                R._focal(model(Ftr, Dtr), ytr).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(Fte, Dte)).cpu().numpy()
            wc, _, _ = V3.within_census_ap(t_idx[b], y[b], p)
            if np.isfinite(wc):
                out.append(wc)
    return float(np.mean(out)) if out else float("nan")


def main():
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    t0 = time.time()

    log("=" * 74)
    log("NULL PERMUTASI DALAM-FAMILI untuk v3  |  h=%d  |  %d permutasi x %d seed"
        % (H, n_perm, seeds))
    log("strata = %s  (lintasan ditukar hanya antar pohon sefamili)" % STRATA)
    log("=" * 74)

    d = PN.build(*PN.load_frozen())
    shp = _verify_reimplementation(d["st"])
    log("pengaman: rekonstruksi STATE %s dan %d contoh IDENTIK dengan dataset.py"
        % (shp[0], shp[1][0]))

    A = np.asarray(ds.adjacency("true"), np.float32)
    A_scaled = torch.as_tensor(A * R.adjacency_scale(A), device=DEVICE)
    folds = ds.folds()

    obs = score_once(state_from_status(d["st"]), examples_from_status(d["st"], H),
                     A_scaled, folds, seeds)
    log("\nteramati (data asli)  AP dalam-sensus = %.4f" % obs)
    log("  pembanding run_v3.py foto:true = 0.1015  (selisih kecil = derau seed)")

    key = PN._strata_key(d["nodes"], STRATA)
    log("  %d famili, ukuran %d-%d palm\n"
        % (len(np.unique(key)), np.bincount(key).min(), np.bincount(key).max()))

    rng = np.random.default_rng(0)
    null = []
    for i in range(n_perm):
        st_p, _ = PN.permute(d, STRATA, rng)
        v = score_once(state_from_status(st_p), examples_from_status(st_p, H),
                       A_scaled, folds, seeds)
        null.append(v)
        if (i + 1) % 10 == 0:
            log("  %2d/%d permutasi  |  null sejauh ini %.4f +/- %.4f"
                % (i + 1, n_perm, np.nanmean(null), np.nanstd(null, ddof=1)))

    null = np.array(null, float)
    nm, nsd = float(np.nanmean(null)), float(np.nanstd(null, ddof=1))
    z = (obs - nm) / nsd if nsd > 0 else float("nan")
    n_ge = int((null >= obs).sum())

    log("\n" + "-" * 74)
    log("teramati        %.4f" % obs)
    log("null            %.4f +/- %.4f   (%d permutasi)" % (nm, nsd, len(null)))
    log("kelebihan       %+.4f   z = %+.2f   permutasi >= teramati: %d/%d"
        % (obs - nm, z, n_ge, len(null)))
    log("-" * 74)

    if n_ge == 0 and z > 3:
        log("BACAAN: kemampuan RUNTUH ketika susunan spasial dihancurkan sementara")
        log("        komposisi famili dipertahankan. Efek graf v3 adalah SPASIAL,")
        log("        bukan artefak kekerabatan. Keraguan genotipe TERTUTUP.")
    elif z < 2:
        log("BACAAN: kemampuan BERTAHAN di bawah null. Model sebagian besar membaca")
        log("        'ketetanggaan ini famili rentan', bukan penularan. Efek graf v3")
        log("        TERCEMAR - klaim foto-tunggal tidak boleh dibuat tanpa genotipe.")
    else:
        log("BACAAN: di antara keduanya. Perbesar n_perm sebelum menyimpulkan.")

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["blok", "strata", "h", "n_perm", "seeds", "observed",
                    "null_mean", "null_std", "excess", "z", "n_ge"])
        w.writerow(["v3-perm", STRATA, H, len(null), seeds, obs, nm, nsd,
                    obs - nm, z, n_ge])
    log("\nditulis: %s   (%.1f menit)" % (OUT, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
