"""Kurva kepekaan: berapa sinyal Lapisan 2 yang bertahan sebagai fungsi MUTU DETEKTOR.

    python sweep_v3_noisy.py [seeds] [n_draw]     # default 10 seed x 20 undian
    RECALLS=0.3,0.5,0.7 FPR=0.0094 python sweep_v3_noisy.py

Menulis `results_v3_noisy_sweep.csv`.

MENGAPA MODUL INI ADA.

`run_v3_noisy.py` menjawab satu titik: pada laju detektor ds_B yang terukur
(recall 0,446 / fpr 0,0094) AP dalam-sensus turun 0,0916 -> 0,0800, yaitu 59%
sinyal bertahan. Itu menjawab "berapa ongkosnya sekarang" tetapi TIDAK menjawab
"berapa nilainya kalau detektornya diperbaiki" - padahal justru pertanyaan kedua
yang menentukan apakah pekerjaan anotasi berikutnya layak dikerjakan.

Modul ini menyapu recall dan mengubah satu titik itu menjadi kurva. Hasilnya
sebuah harga: "menaikkan recall detektor dari 0,446 ke 0,60 mengembalikan sekian
persen sinyal". Itu peta jalan yang terukur, bukan dugaan.

MENGAPA SAPUANNYA MURAH, dan mengapa itu BUKAN kecurangan.

Modelnya dilatih pada status BERSIH dan hanya DIUJI pada status berderau -
rancangan yang sama persis dengan `run_v3_noisy.py`, dan alasannya sama: di kebun
tujuan tidak ada label lapangan untuk melatih ulang. Karena recall dan fpr hanya
menyentuh sisi UJI, model yang sama dapat dinilai ulang pada seluruh titik sapuan.
Jadi latih SEKALI per (lipatan, seed), lalu nilai di semua titik. Yang mahal
adalah pelatihannya, dan pelatihannya tidak bergantung pada laju detektor.

Yang TIDAK boleh disimpulkan dari kurva ini: bahwa detektor dengan recall 0,9
memang bisa dibangun. Kurva ini memberi harga sebuah perbaikan, bukan bukti bahwa
perbaikan itu tercapai.

BATAS. Sama dengan run_v3_noisy.py: laju acuannya diukur di ds_B, kebun yang
BERBEDA dari Eg9PP. Ini simulasi ongkos, bukan pengukuran lapangan.
"""
import csv
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import dataset as ds              # noqa: E402
import dataset_v3 as v3           # noqa: E402
import models_real as M           # noqa: E402
import run_real as R              # noqa: E402
import run_v3 as V3               # noqa: E402
import run_v3_noisy as NZ         # noqa: E402

OUT = os.path.join(HERE, "results_v3_noisy_sweep.csv")
H, EPOCHS, DEVICE = V3.H, V3.EPOCHS, R.DEVICE
COL = NZ.COL
NO_SKILL = 0.0632                 # garis tanpa-graf dalam-sensus (results_v3.csv)

# Titik sapuan. 1,000 = detektor sempurna (batas atas), dan laju ds_B yang
# terukur disisipkan otomatis supaya kurva selalu memuat titik nyata.
RECALLS = [float(x) for x in os.environ.get(
    "RECALLS", "0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00").split(",")]

# MODE=recall     sapu recall dengan fpr DITAHAN TETAP. Menjawab "berapa nilainya
#                 kalau detektor jadi lebih baik", yaitu harga sebuah perbaikan.
# MODE=operating  sapu KURVA OPERASI detektor yang SESUNGGUHNYA: tiap titik adalah
#                 satu ambang tau, dengan recall DAN fpr yang benar-benar
#                 dihasilkannya. Menjawab pertanyaan yang berbeda dan lebih tajam:
#                 "ambang mana yang terbaik kalau dinilai dari AP Lapisan 2, bukan
#                 dari F1 Tahap 1?" Ambang 0,75 sekarang dipilih dengan
#                 memaksimalkan F1 Tahap 1 - padahal fungsi ongkos hilirnya
#                 ASIMETRIS: pohon sakit yang terlewat menghilangkan muatan difusi,
#                 sedangkan alarm palsu hanya mengencerkannya.
MODE = os.environ.get("MODE", "recall")


def operating_points():
    """(label, recall, fpr) dari kurva ambang Unhealthy ds_B yang sesungguhnya."""
    thr = json.load(open(NZ.THR))
    n_gt = {f: r["n_gt"] for f, r in json.load(open(NZ.CEN))["folds"].items()}
    taus = sorted({t for c in thr["curves"].values() for t in c}, key=float)
    out = []
    for t in taus:
        Rs, F1s, FP, NEG = [], [], 0, 0
        for f, c in thr["curves"].items():
            if t not in c:
                continue
            Rs.append(c[t]["R"]); F1s.append(c[t]["F1"]); FP += c[t]["FP"]
            NEG += n_gt[f] - (c[t]["TP"] + c[t]["FN"])
        if Rs and NEG:
            out.append((t, float(np.mean(Rs)), FP / NEG, float(np.mean(F1s))))
    return out


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    ndraw = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    rec0, fpr0, src = NZ.detector_rates()
    fpr = float(os.environ["FPR"]) if os.environ.get("FPR") else fpr0

    if MODE == "operating":
        pts = [(("tau %s" % t), r, fp, f1) for t, r, fp, f1 in operating_points()]
        banner = "KURVA OPERASI DETEKTOR SESUNGGUHNYA (recall DAN fpr ikut berubah)"
    else:
        grid = sorted(set([round(r, 4) for r in RECALLS] + [round(rec0, 4)]))
        pts = [("recall %.3f" % r, r, fpr, float("nan")) for r in grid]
        banner = "HARGA PERBAIKAN DETEKTOR (fpr ditahan tetap di %.4f)" % fpr

    print("=" * 78)
    print("KEPEKAAN MUTU DETEKTOR  |  h=%d  |  %d seed x %d undian x %d titik"
          % (H, seeds, ndraw, len(pts)))
    print(banner)
    print("titik terukur ds_B: recall %.3f  fpr %.4f  (%s)" % (rec0, fpr0, src))
    print("=" * 78)

    T = len(ds.census())
    clean = v3.node_features_v3(np.arange(T))[:, :, [COL]]
    A = np.asarray(ds.adjacency("true"), np.float32)
    A_s = torch.as_tensor(A * R.adjacency_scale(A), device=DEVICE)
    tree, t_idx, y = ds.build_examples(H, np.arange(T))
    folds = ds.folds()

    Ft_c = torch.as_tensor(clean, device=DEVICE)
    D_c = R.diffuse(Ft_c, A_s)

    # Undian derau dibuat SEKALI per titik recall dan dipakai ulang di semua
    # (lipatan, seed), supaya perbedaan antar-titik adalah recall - bukan undian.
    rng = np.random.default_rng(0)
    draws = {}
    for lab, rc, fp, _ in pts:
        draws[lab] = [(lambda z: (torch.as_tensor(z, device=DEVICE),
                                  R.diffuse(torch.as_tensor(z, device=DEVICE), A_s)))
                      (NZ.corrupt(clean, rc, fp, rng)) for _ in range(ndraw)]

    # res      : semua undian (untuk mean/std marjinal)
    # res_unit : SATU nilai per (lipatan, seed) - undian dirata-ratakan lebih dulu.
    #            Inilah unit yang boleh dipasangkan: model, lipatan dan seed-nya
    #            sama persis di semua titik ambang, jadi selisihnya berpasangan dan
    #            std marjinal (yang didominasi ragam lipatan) BUKAN pita deraunya.
    res_clean, res = [], {lab: [] for lab, _, _, _ in pts}
    res_unit = {lab: [] for lab, _, _, _ in pts}
    for fold, (trm, tem) in enumerate(folds):
        trm, tem = np.asarray(trm, bool), np.asarray(tem, bool)
        a, b = trm[tree], tem[tree]
        if y[a].size == 0 or y[b].sum() == 0:
            continue
        ii_tr = torch.as_tensor(tree[a], dtype=torch.long)
        tt_tr = torch.as_tensor(t_idx[a], dtype=torch.long)
        ii_te = torch.as_tensor(tree[b], dtype=torch.long)
        tt_te = torch.as_tensor(t_idx[b], dtype=torch.long)
        ytr = torch.as_tensor(y[a], device=DEVICE)
        Ftr, Dtr = Ft_c[tt_tr, ii_tr].unsqueeze(1), D_c[tt_tr, ii_tr].unsqueeze(1)

        for s in range(seeds):
            torch.manual_seed(1234 + s)                 # seed IDENTIK run_v3_noisy
            m = M.build("STGNN", 1, horizon=H).to(DEVICE)
            opt = torch.optim.Adam(m.parameters(), lr=R.LR, weight_decay=R.WD)
            m.train()
            for _ in range(EPOCHS):
                opt.zero_grad()
                R._focal(m(Ftr, Dtr), ytr).backward()
                opt.step()
            m.eval()

            with torch.no_grad():
                p = torch.sigmoid(m(Ft_c[tt_te, ii_te].unsqueeze(1),
                                    D_c[tt_te, ii_te].unsqueeze(1))).cpu().numpy()
            res_clean.append(V3.within_census_ap(t_idx[b], y[b], p)[0])

            for lab, _, _, _ in pts:                    # inferensi saja - murah
                per_draw = []
                for Ft_n, D_n in draws[lab]:
                    with torch.no_grad():
                        pn = torch.sigmoid(m(Ft_n[tt_te, ii_te].unsqueeze(1),
                                             D_n[tt_te, ii_te].unsqueeze(1))).cpu().numpy()
                    per_draw.append(V3.within_census_ap(t_idx[b], y[b], pn)[0])
                res[lab].extend(per_draw)
                res_unit[lab].append(float(np.nanmean(per_draw)))
        print("  lipatan %d selesai" % fold)

    c = float(np.nanmean(res_clean))
    print("\nmasukan BERSIH (status lapangan): AP %.4f  (lift %.2fx)  n=%d"
          % (c, c / NO_SKILL, len(res_clean)))
    print("\n%-13s %7s %8s %8s %-18s %-7s %s"
          % ("titik", "recall", "fpr", "F1_thp1", "AP dalam-sensus", "lift", "bertahan"))
    rows = []
    for lab, rc, fp, f1 in pts:
        v = np.array(res[lab], float)
        mn, sd = float(np.nanmean(v)), float(np.nanstd(v, ddof=1))
        keep = (mn - NO_SKILL) / max(1e-9, c - NO_SKILL)
        mark = "  <- dipakai sekarang" if abs(rc - rec0) < 1e-3 else ""
        print("%-13s %7.3f %8.4f %8s %.4f +/- %.4f  %-7.2f %3.0f%%%s"
              % (lab, rc, fp, ("%.3f" % f1) if f1 == f1 else "—",
                 mn, sd, mn / NO_SKILL, 100 * keep, mark))
        rows.append((lab, rc, fp, f1, mn, sd, len(v), mn / NO_SKILL, keep))

    best = max(rows, key=lambda r: r[4])
    cur = min(rows, key=lambda r: abs(r[1] - rec0))
    if MODE == "operating":
        print("\n== AMBANG TERBAIK MENURUT HILIR, bukan menurut F1 Tahap 1 ==")
        print("   dipakai sekarang : %-12s recall %.3f fpr %.4f -> AP %.4f (%.0f%%)"
              % (cur[0], cur[1], cur[2], cur[4], 100 * cur[8]))
        print("   terbaik hilir    : %-12s recall %.3f fpr %.4f -> AP %.4f (%.0f%%)"
              % (best[0], best[1], best[2], best[4], 100 * best[8]))
        print("   selisih          : %+.4f AP  (%+.0f poin sinyal bertahan)"
              % (best[4] - cur[4], 100 * (best[8] - cur[8])))
        print("\n   Ambang Tahap 1 dipilih dengan memaksimalkan F1 - metrik SIMETRIS.")
        print("   Ongkos hilirnya TIDAK simetris, jadi optimum keduanya tidak wajib sama.")
        print("   Ini BUKAN izin menyetel ambang di atas Eg9PP: kurva ambangnya diukur")
        print("   di ds_B silang-lipatan, dan selisih di dalam pita derau tetap TIDAK")
        print("   KONKLUSIF. Bandingkan std-nya sebelum mengklaim apa pun.")
    else:
        print("\n== HARGA PERBAIKAN DETEKTOR (dari titik terukur %.3f) ==" % rec0)
        for lab, rc, fp, f1, mn, sd, n, lf, keep in rows:
            if rc > rec0:
                print("   recall %.3f -> %.3f : bertahan %.0f%% -> %.0f%% (+%.0f poin), "
                      "lift %.2fx -> %.2fx"
                      % (rec0, rc, 100 * cur[8], 100 * keep,
                         100 * (keep - cur[8]), cur[7], lf))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["blok", "mode", "h", "titik", "recall", "fpr", "f1_tahap1", "n",
                    "ap_mean", "ap_std", "lift", "sinyal_bertahan", "ap_clean",
                    "no_skill", "titik_terukur"])
        for lab, rc, fp, f1, mn, sd, n, lf, keep in rows:
            w.writerow(["v3-noisy-sweep", MODE, H, lab, rc, fp, f1, n, mn, sd, lf,
                        keep, c, NO_SKILL, int(abs(rc - rec0) < 1e-3)])
    print("\nditulis: %s" % OUT)
    print("\nBATAS. Kurva ini memberi HARGA sebuah perbaikan detektor, bukan bukti")
    print("bahwa perbaikan itu tercapai. Laju acuannya diukur di ds_B, kebun yang")
    print("BERBEDA dari Eg9PP; recall 1,0 adalah batas atas teoretis, bukan target.")


if __name__ == "__main__":
    main()
