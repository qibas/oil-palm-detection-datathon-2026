"""Agregasi BLOK: apakah memeringkat petak lebih kuat daripada memeringkat pohon?

    python run_v3_blocks.py [seeds]        # default 10 seed x 2 lipatan

Menulis `results_v3_blocks.csv`.

ALASANNYA OPERASIONAL, BUKAN STATISTIK.

Mandor tidak memeriksa satu pohon; ia mengirim regu ke satu blok. Metrik per-pohon
(AP 0,10) menjawab pertanyaan yang tidak pernah ditanyakan siapa pun. Kalau unitnya
petak, merata-ratakan puluhan pohon memangkas derau - dan pertanyaannya menjadi
"petak mana yang akan paling banyak kasus baru", yang memang keputusan sebenarnya.

TIGA PENGAMAN, KARENA "ANGKANYA NAIK" ITU MUDAH DIPALSUKAN.

1. LEAVE-ONE-PARCEL-OUT, sama dengan seluruh paket. Blok yang dinilai TIDAK PERNAH
   ikut melatih.
2. GARIS ACAK DIHITUNG PADA METRIK YANG SAMA. Skor pohon diacak DI DALAM sensus,
   lalu diagregasi dengan cara yang sama persis. Kalau metrik blok memang "terlalu
   mudah", garis acak ini akan ikut tinggi - dan kenaikan modelnya palsu. Kalau
   garis acak tetap di sekitar nol sementara model naik, kenaikannya nyata.
3. JUMLAH BLOK DILAPORKAN. Blok dengan risk set terlalu kecil dikeluarkan dan
   jumlahnya dicetak; korelasi atas 3 blok bukan hasil, itu derau.

DUA METRIK.

    Spearman   korelasi peringkat petak-diprediksi lawan kasus-baru-sesungguhnya,
               dihitung PER SENSUS lalu dirata-rata (di dalam satu sensus, waktu
               konstan - alasan yang sama dengan AP dalam-sensus)
    tangkapan  kalau regu dikirim ke K petak teratas, berapa persen kasus baru
               sensus itu yang tertangkap
"""
import csv
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import dataset as ds            # noqa: E402
import dataset_v3 as v3         # noqa: E402
import models_real as M         # noqa: E402
import run_real as R            # noqa: E402
import run_v3 as V3             # noqa: E402

OUT = os.path.join(HERE, "results_v3_blocks.csv")
H, EPOCHS = V3.H, V3.EPOCHS
MIN_PALMS = int(os.environ.get("MIN_PALMS", 5))    # risk set minimum per blok-sensus
MIN_BLOCKS = int(os.environ.get("MIN_BLOCKS", 5))  # blok minimum agar sensus dihitung
TOPK = [1, 3, 5]


def block_key():
    """Blok = parcel|plot. Itu unit kerja regu di lapangan."""
    nodes, _, _ = ds.load()
    k = nodes.parcel.astype(str) + "|" + nodes["plot"].astype(str)
    import pandas as pd
    return pd.factorize(k)[0], k.to_numpy()


def evaluate(t_idx, y, p, blk, rng=None):
    """-> (spearman rata2, tangkapan@K, n blok-sensus, n sensus dipakai, n dibuang).

    `rng` bukan None => skor diacak DI DALAM sensus lebih dulu (garis acak).
    """
    from scipy.stats import spearmanr

    rho, cap = [], {k: [] for k in TOPK}
    nb, nk, nd = [], 0, 0
    for t in np.unique(t_idx):
        m = t_idx == t
        pt, yt, bt = p[m].copy(), y[m], blk[m]
        if rng is not None:
            pt = pt[rng.permutation(len(pt))]
        ub = np.unique(bt)
        pred, obs, cnt = [], [], []
        for b in ub:
            mb = bt == b
            if mb.sum() < MIN_PALMS:
                continue
            pred.append(pt[mb].mean())
            obs.append(yt[mb].mean())
            cnt.append(yt[mb].sum())
        if len(pred) < MIN_BLOCKS or np.sum(cnt) == 0:
            nd += 1
            continue
        nk += 1
        nb.append(len(pred))
        pred, obs, cnt = np.array(pred), np.array(obs), np.array(cnt, float)
        r = spearmanr(pred, obs).statistic
        if np.isfinite(r):
            rho.append(r)
        order = np.argsort(-pred)
        for k in TOPK:
            kk = min(k, len(order))
            cap[k].append(cnt[order[:kk]].sum() / max(1e-9, cnt.sum()))
    return (float(np.mean(rho)) if rho else float("nan"),
            {k: (float(np.mean(v)) if v else float("nan")) for k, v in cap.items()},
            float(np.mean(nb)) if nb else 0.0, nk, nd)


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print("=" * 74)
    print("AGREGASI BLOK  |  h=%d  |  %d seed x %d lipatan  |  leave-one-parcel-out"
          % (H, seeds, len(ds.folds())))
    print("=" * 74)

    T = len(ds.census())
    arm = V3.Arm("foto6", v3.node_features_v3(np.arange(T)), 1)
    tree, t_idx, y = ds.build_examples(H, np.arange(T))
    bid, bname = block_key()
    blk_all = bid[tree]
    print("blok didefinisikan sebagai parcel|plot  ->  %d blok di seluruh kebun"
          % len(np.unique(bid)))
    print("syarat: >= %d sawit risk set per blok-sensus, >= %d blok per sensus\n"
          % (MIN_PALMS, MIN_BLOCKS))

    rng = np.random.default_rng(0)
    acc = {"model": {"rho": [], "cap": {k: [] for k in TOPK}},
           "acak": {"rho": [], "cap": {k: [] for k in TOPK}}}
    palm_ap, nblocks, ncen, ndrop = [], [], [], []

    for fold, (trm, tem) in enumerate(ds.folds()):
        trm, tem = np.asarray(trm, bool), np.asarray(tem, bool)
        a, b = trm[tree], tem[tree]
        if y[a].size == 0 or y[b].sum() == 0:
            continue
        for s in range(seeds):
            arm._D_active = arm.view("true", s)
            torch.manual_seed(1234 + s)
            m = M.build("STGNN", arm.d, horizon=H).to(R.DEVICE)
            opt = torch.optim.Adam(m.parameters(), lr=R.LR, weight_decay=R.WD)
            Ftr, Dtr = arm.gather(tree[a], t_idx[a])
            Fte, Dte = arm.gather(tree[b], t_idx[b])
            ytr = torch.as_tensor(y[a], device=R.DEVICE)
            m.train()
            for _ in range(EPOCHS):
                opt.zero_grad()
                R._focal(m(Ftr, Dtr), ytr).backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                p = torch.sigmoid(m(Fte, Dte)).cpu().numpy()

            wc, _, _ = V3.within_census_ap(t_idx[b], y[b], p)
            palm_ap.append(wc)
            for lbl, rg in (("model", None), ("acak", rng)):
                r, c, nb, nk, nd = evaluate(t_idx[b], y[b], p, blk_all[b], rg)
                acc[lbl]["rho"].append(r)
                for k in TOPK:
                    acc[lbl]["cap"][k].append(c[k])
                if lbl == "model":
                    nblocks.append(nb); ncen.append(nk); ndrop.append(nd)
        print("  lipatan %d selesai" % fold)

    print("\nblok-sensus yang dievaluasi: %.1f blok/sensus, %.1f sensus dipakai, "
          "%.1f sensus dibuang (positif/blok terlalu sedikit)"
          % (np.mean(nblocks), np.mean(ncen), np.mean(ndrop)))
    print("total unit blok-sensus per lari: %.0f\n" % (np.mean(nblocks) * np.mean(ncen)))

    def line(lbl, d):
        r = np.array(d["rho"], float)
        s = "%-8s Spearman %+.3f +/- %.3f" % (lbl, np.nanmean(r), np.nanstd(r, ddof=1))
        for k in TOPK:
            v = np.array(d["cap"][k], float)
            s += "   top%d %.1f%%" % (k, 100 * np.nanmean(v))
        return s

    print(line("model", acc["model"]))
    print(line("acak", acc["acak"]))
    pa = np.array(palm_ap, float)
    print("\npembanding per-POHON (metrik lama): AP dalam-sensus %.4f +/- %.4f, "
          "lift %.2fx atas 0,0632" % (pa.mean(), pa.std(ddof=1), pa.mean() / 0.0632))

    rm = np.array(acc["model"]["rho"], float)
    ra = np.array(acc["acak"]["rho"], float)
    d = rm - ra
    print("\n--- putusan (aturan run_real.py::paired) ---")
    print("  Spearman model - acak   %+.3f +/- %.3f  %d/%d  %s"
          % (d.mean(), d.std(ddof=1), int((d > 0).sum()), len(d),
             "TIDAK KONKLUSIF" if abs(d.mean()) < d.std(ddof=1)
             else ("POS" if d.mean() > 0 else "NEG")))
    for k in TOPK:
        cm = np.array(acc["model"]["cap"][k], float)
        ca = np.array(acc["acak"]["cap"][k], float)
        dd = cm - ca
        print("  tangkapan top%d  model %.1f%% lawan acak %.1f%%   selisih %+.1f pp  %d/%d"
              % (k, 100 * np.nanmean(cm), 100 * np.nanmean(ca), 100 * np.nanmean(dd),
                 int((dd > 0).sum()), len(dd)))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["blok", "h", "unit", "metrik", "model", "acak", "selisih",
                    "blok_per_sensus", "sensus_dipakai", "n"])
        w.writerow(["v3-blocks", H, "blok", "spearman", np.nanmean(rm), np.nanmean(ra),
                    d.mean(), np.mean(nblocks), np.mean(ncen), len(d)])
        for k in TOPK:
            cm = np.nanmean(acc["model"]["cap"][k]); ca = np.nanmean(acc["acak"]["cap"][k])
            w.writerow(["v3-blocks", H, "blok", "tangkapan_top%d" % k, cm, ca, cm - ca,
                        np.mean(nblocks), np.mean(ncen), len(acc["model"]["cap"][k])])
        w.writerow(["v3-blocks", H, "pohon", "AP dalam-sensus", pa.mean(), 0.0632,
                    pa.mean() - 0.0632, "", "", len(pa)])
    print("\nditulis: %s" % OUT)


if __name__ == "__main__":
    main()
