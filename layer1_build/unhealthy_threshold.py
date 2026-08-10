"""Ambang kelas Unhealthy — dipilih SILANG-LIPATAN, terpisah dari ambang lokalisasi.

    python unhealthy_threshold.py

Menulis `yolo12_results/unhealthy_threshold.json`.

MASALAHNYA.

`centre_eval_folds.py` memilih conf = 0,75 untuk memaksimalkan F1 PUSAT TAJUK.
Metrik itu didominasi kelas Healthy (5.011 dari 5.077 pohon), jadi ambangnya
dioptimalkan untuk LOKALISASI. Dipakai apa adanya untuk kelas Unhealthy, hasilnya
0-1 gejala terdeteksi per ubin - dan tanpa sumber, model risiko Lapisan 2 buta.

Satu ambang tidak bisa melayani dua keputusan yang berbeda: "adakah pohon di sini"
dan "apakah pohon ini sakit". Modul ini memisahkannya.

RANCANGAN.

1. Deteksi pada conf RENDAH (0,10) supaya kotak Unhealthy berkeyakinan sedang tidak
   terbuang sebelum sempat dinilai.
2. Gabungkan ke pohon unik memakai prosedur yang sama dengan `detect_centres.py`.
3. Untuk tiap pohon, ambil keyakinan Unhealthy TERTINGGI di antara seluruh deteksi
   yang jatuh ke pohon itu. Ini penting: penggabungan biasa mempertahankan deteksi
   paling yakin, yang hampir selalu Healthy, sehingga sinyal Unhealthy tertimpa.
4. Cocokkan ke pohon GT, lalu sapu ambang dan hitung F1 kelas Unhealthy.
5. Pilih ambang untuk tiap lipatan dari kurva lipatan LAIN saja - aturan yang sama
   dengan `centre_eval_folds.py`, supaya ortomosaik uji tidak ikut memilih apa pun
   tentang dirinya sendiri.

BATAS. Positif Unhealthy hanya 66 pohon unik di seluruh dataset (17/31/18 per
ortomosaik). Angka apa pun di sini bersandar pada belasan pohon per lipatan, jadi
pita deraunya lebar. Ini memperbaiki ambang yang jelas salah; ia TIDAK membuat
deteksi penyakit menjadi kuat.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import detect_centres as dc     # noqa: E402
import y12                      # noqa: E402

OUT = os.path.join(y12.RESDIR, "unhealthy_threshold.json")
SEED = 42
TAG = "yolo12n_base"
LOW_CONF = 0.10
TAUS = [round(x, 2) for x in np.arange(0.15, 0.86, 0.05)]


def per_tree_scores(fold, weights, imgsz=640):
    """-> (xy pohon unik, skor Unhealthy tertinggi per pohon)."""
    import torch
    from scipy.spatial import cKDTree
    from ultralytics import YOLO

    dev = y12.pick_device()
    model = YOLO(weights)
    paths = [l.strip() for l in open(os.path.join(y12.ROOT, f"{fold}_val.txt")) if l.strip()]
    det, _ = dc.detect(model, paths, LOW_CONF, imgsz, 32, dev, stitch=True)
    if not det:
        return np.zeros((0, 2)), np.zeros(0)

    box = float(np.median([d[6] for d in det]))
    kept, scale = dc.merge_two_pass(det, box)
    xy = np.array([[d[0], d[1]] for d in kept], float)

    # Skor Unhealthy per pohon = MAKS di antara semua deteksi yang jatuh ke pohon itu.
    # Tanpa langkah ini sinyal Unhealthy hilang ditimpa deteksi Healthy yang lebih yakin.
    all_xy = np.array([[d[0], d[1]] for d in det], float)
    unh = np.array([d[2] if d[3] == 1 else 0.0 for d in det], float)
    kd = cKDTree(xy)
    score = np.zeros(len(xy))
    _, idx = kd.query(all_xy, k=1)
    for i, s in zip(idx, unh):
        if s > score[i]:
            score[i] = s
    return xy, score


def curve(fold, weights):
    ortho = dc.__dict__  # noqa: F841  (jaga import tetap eksplisit di bawah)
    first = open(os.path.join(y12.ROOT, f"{fold}_val.txt")).readline().strip()
    ortho = y12._tile_offset(os.path.basename(first))[0]
    gt_xy, gt_lab = y12.gt_trees(ortho)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(gt_xy).query(gt_xy, k=2)
    spacing = float(np.median(d[:, 1]))

    xy, score = per_tree_scores(fold, weights)
    mt = y12.centre_match(xy, gt_xy, 0.5 * spacing)
    if len(mt) == 0:
        return {}, 0
    s_m, y_m = score[mt[:, 0]], (gt_lab[mt[:, 1]] == 1).astype(int)
    out = {}
    for tau in TAUS:
        pred = (s_m >= tau).astype(int)
        tp = int((pred & y_m).sum()); fp = int((pred & ~y_m.astype(bool)).sum())
        fn = int(((1 - pred) & y_m).sum())
        pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
        out[tau] = {"P": pr, "R": rc, "F1": 2 * pr * rc / max(1e-9, pr + rc),
                    "TP": tp, "FP": fp, "FN": fn}
    return out, int(y_m.sum())


def main():
    folds = ["fold0", "fold1", "fold2"]
    have = {f: os.path.join(y12.RUNS, f"{TAG}_{f}_s{SEED}", "weights", "best.pt")
            for f in folds}
    folds = [f for f in folds if os.path.isfile(have[f])]
    if not folds:
        raise SystemExit("bobot tidak ada — jalankan train_folds_gpu.py")

    print("ambang lokalisasi tetap 0,75 (dari centre_eval_folds.py).")
    print("yang dipilih di sini HANYA ambang kelas Unhealthy.\n")

    cur, npos = {}, {}
    for f in folds:
        cur[f], npos[f] = curve(f, have[f])
        best = max(cur[f], key=lambda t: cur[f][t]["F1"])
        print("%s: %d pohon Unhealthy GT  |  F1 terbaik %.3f di tau=%.2f"
              % (f, npos[f], cur[f][best]["F1"], best))

    print("\n== ambang dipilih SILANG-LIPATAN ==")
    chosen, rows = {}, []
    for f in folds:
        others = [o for o in folds if o != f and cur[o]]
        if not others:
            continue
        tau = max(TAUS, key=lambda t: np.mean([cur[o][t]["F1"] for o in others]))
        chosen[f] = tau
        r = cur[f][tau]
        rows.append(r)
        print("  %s: tau=%.2f (dari %s)  ->  P=%.3f R=%.3f F1=%.3f  TP=%d FP=%d FN=%d"
              % (f, tau, "+".join(others), r["P"], r["R"], r["F1"], r["TP"], r["FP"], r["FN"]))

    if rows:
        for k in ("P", "R", "F1"):
            v = np.array([r[k] for r in rows])
            print("  rata-rata %-3s %.3f +/- %.3f" % (k, v.mean(), v.std(ddof=1)))

    base = {f: cur[f][0.75] for f in folds if 0.75 in cur[f]}
    if base:
        print("\n== pembanding: memakai ambang lokalisasi 0,75 apa adanya ==")
        for f, r in base.items():
            print("  %s: P=%.3f R=%.3f F1=%.3f  TP=%d FP=%d FN=%d"
                  % (f, r["P"], r["R"], r["F1"], r["TP"], r["FP"], r["FN"]))

    tau_final = float(np.median(list(chosen.values()))) if chosen else 0.75
    os.makedirs(y12.RESDIR, exist_ok=True)
    json.dump({"tau_unhealthy": tau_final, "chosen_per_fold": chosen,
               "n_pos_gt": npos, "taus": TAUS,
               "curves": {f: {str(k): v for k, v in c.items()} for f, c in cur.items()},
               "note": ("Ambang KELAS Unhealthy, terpisah dari ambang LOKALISASI 0,75. "
                        "Dipilih silang-lipatan. Bersandar pada 17-31 pohon positif per "
                        "ortomosaik — pita derau lebar.")},
              open(OUT, "w"), indent=2)
    print("\ntau_unhealthy final (median antar lipatan) = %.2f" % tau_final)
    print("ditulis: %s" % OUT)


if __name__ == "__main__":
    main()
