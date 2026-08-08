"""Ongkos substitusi detektor->status: v3 dilatih bersih, diuji dengan masukan berderau.

    python run_v3_noisy.py [seeds] [n_draw]      # default 10 seed x 20 undian derau
    RECALL=0.55 FPR=0.004 python run_v3_noisy.py

Menulis `results_v3_noisy.csv`.

PERTANYAANNYA.

`run_v3.py` melatih DAN menguji pada `is_sympt` Eg9PP - status S/D yang
TERVERIFIKASI LAPANGAN. Saat dipakai pada citra, kolom itu diisi kelas `Unhealthy`
detektor: kesehatan tajuk generik, dengan recall dan presisi yang jauh dari
sempurna. Selama ini ongkos substitusi itu tercatat sebagai "belum diukur siapa
pun". Modul ini mengukurnya.

RANCANGANNYA MENIRU PENYERAHAN SESUNGGUHNYA.

    latih   pada status BERSIH   (model memang dilatih pada panel lapangan)
    uji     pada status BERDERAU (di lapangan ia hanya menerima keluaran detektor)

Bukan latih-berderau-uji-berderau. Yang ditanyakan adalah "berapa yang hilang saat
model yang baik diberi masukan yang buruk", bukan "bisakah model beradaptasi pada
derau" - yang kedua mengandaikan kita punya label lapangan untuk melatih ulang di
kebun tujuan, dan justru itu yang tidak ada.

MODEL DERAUNYA.

    lewat   dengan peluang (1 - recall) pohon bergejala TIDAK terdeteksi
    palsu   dengan peluang fpr pohon sehat DILAPORKAN bergejala

Derau ditebarkan ke SELURUH matriks status per undian, lalu difusi dihitung ulang -
karena yang dilihat model bukan status pohon itu sendiri (selalu 0 di risk set)
melainkan status TETANGGANYA lewat graf.

BATAS. Recall dan fpr diambil dari pengukuran ds_B, kebun yang berbeda dari Eg9PP.
Ini simulasi ongkos, bukan pengukuran lapangan di kebun tujuan. Ia mengganti sebuah
"tidak diketahui" dengan sebuah "diperkirakan sekian", dan itu saja.
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

import dataset as ds            # noqa: E402
import dataset_v3 as v3         # noqa: E402
import models_real as M         # noqa: E402
import run_real as R            # noqa: E402
import run_v3 as V3             # noqa: E402

OUT = os.path.join(HERE, "results_v3_noisy.csv")
THR = os.path.join(os.path.dirname(HERE), "layer1_build", "yolo12_results",
                   "unhealthy_threshold.json")
H, EPOCHS, DEVICE = V3.H, V3.EPOCHS, R.DEVICE
COL = 0                                     # is_sympt


def detector_rates():
    """Recall/FPR kelas Unhealthy. Dari pengukuran ds_B kalau ada, kalau tidak default."""
    rec = os.environ.get("RECALL")
    fpr = os.environ.get("FPR")
    if rec and fpr:
        return float(rec), float(fpr), "env"
    if os.path.isfile(THR):
        d = json.load(open(THR))
        tau = str(d["tau_unhealthy"])
        Ps, Rs, N, FP = [], [], 0, 0
        for f, c in d["curves"].items():
            if tau in c:
                Ps.append(c[tau]["P"]); Rs.append(c[tau]["R"])
                N += c[tau]["TP"] + c[tau]["FP"] + c[tau]["FN"]
                FP += c[tau]["FP"]
        if Rs:
            return float(np.mean(Rs)), float(FP) / max(1, N), "ds_B tau=%s" % tau
    return 0.50, 0.005, "default (pengukuran ds_B belum ada)"


def corrupt(state, recall, fpr, rng):
    """(T,N,1) is_sympt bersih -> berderau. Meniru detektor yang lewat & salah alarm."""
    s = state[:, :, 0].copy()
    sick = s > 0.5
    miss = sick & (rng.random(s.shape) > recall)
    false = (~sick) & (rng.random(s.shape) < fpr)
    s[miss] = 0.0
    s[false] = 1.0
    return s[:, :, None].astype(np.float32)


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    ndraw = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    recall, fpr, src = detector_rates()

    print("=" * 74)
    print("ONGKOS SUBSTITUSI DETEKTOR->STATUS  |  h=%d  |  %d seed x %d undian"
          % (H, seeds, ndraw))
    print("laju detektor: recall %.3f  fpr %.4f   (sumber: %s)" % (recall, fpr, src))
    print("=" * 74)

    T = len(ds.census())
    clean = v3.node_features_v3(np.arange(T))[:, :, [COL]]
    A = np.asarray(ds.adjacency("true"), np.float32)
    A_s = torch.as_tensor(A * R.adjacency_scale(A), device=DEVICE)
    tree, t_idx, y = ds.build_examples(H, np.arange(T))
    folds = ds.folds()

    Ft_c = torch.as_tensor(clean, device=DEVICE)
    D_c = R.diffuse(Ft_c, A_s)
    rng = np.random.default_rng(0)
    noisy = [(lambda z: (torch.as_tensor(z, device=DEVICE),
                         R.diffuse(torch.as_tensor(z, device=DEVICE), A_s)))
             (corrupt(clean, recall, fpr, rng)) for _ in range(ndraw)]

    res_clean, res_noisy = [], []
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
        Ftr = Ft_c[tt_tr, ii_tr].unsqueeze(1)
        Dtr = D_c[tt_tr, ii_tr].unsqueeze(1)

        for s in range(seeds):
            torch.manual_seed(1234 + s)
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
            wc, _, _ = V3.within_census_ap(t_idx[b], y[b], p)
            res_clean.append(wc)
            for Ft_n, D_n in noisy:
                with torch.no_grad():
                    pn = torch.sigmoid(m(Ft_n[tt_te, ii_te].unsqueeze(1),
                                         D_n[tt_te, ii_te].unsqueeze(1))).cpu().numpy()
                w2, _, _ = V3.within_census_ap(t_idx[b], y[b], pn)
                res_noisy.append(w2)
        print("  lipatan %d selesai" % fold)

    c = np.array(res_clean, float); nz = np.array(res_noisy, float)
    no_skill = 0.0632
    print("\n%-34s %s" % ("", "AP dalam-sensus"))
    print("%-34s %.4f +/- %.4f  (n=%d)" % ("masukan BERSIH (status lapangan)",
                                           np.nanmean(c), np.nanstd(c, ddof=1), len(c)))
    print("%-34s %.4f +/- %.4f  (n=%d)" % ("masukan BERDERAU (detektor)",
                                           np.nanmean(nz), np.nanstd(nz, ddof=1), len(nz)))
    print("%-34s %.4f" % ("garis tanpa-skill", no_skill))
    keep = (np.nanmean(nz) - no_skill) / max(1e-9, np.nanmean(c) - no_skill)
    print("\nsinyal di atas tanpa-skill yang BERTAHAN: %.0f%%" % (100 * keep))
    print("lift terhadap tanpa-skill: bersih %.2fx  ->  berderau %.2fx"
          % (np.nanmean(c) / no_skill, np.nanmean(nz) / no_skill))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["blok", "h", "recall", "fpr", "sumber_laju", "n_clean", "n_noisy",
                    "ap_clean", "ap_clean_std", "ap_noisy", "ap_noisy_std",
                    "no_skill", "sinyal_bertahan"])
        w.writerow(["v3-noisy", H, recall, fpr, src, len(c), len(nz),
                    np.nanmean(c), np.nanstd(c, ddof=1),
                    np.nanmean(nz), np.nanstd(nz, ddof=1), no_skill, keep])
    print("\nditulis: %s" % OUT)
    print("\nBATAS: laju detektor diukur di ds_B, kebun yang BERBEDA dari Eg9PP. "
          "Ini simulasi ongkos, bukan pengukuran lapangan di kebun tujuan.")


if __name__ == "__main__":
    main()
