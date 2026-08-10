"""Metrik UTAMA Tahap 1 di ketiga lipatan + angka derajat untuk uji jembatan.

    python centre_eval_folds.py

Dijalankan SETELAH `train_folds_gpu.py`. Menulis `yolo12_results/centre_eval.json`.

MENGAPA AMBANG conf DIPILIH SILANG-LIPATAN.

Pada lari fold1 pertama, ambang conf dipilih dengan melihat kurva presisi/recall
lipatan yang DITAHAN. Itu memilih hiperparameter di atas data uji - bentuk halus
dari kebocoran yang justru dijaga `region()` + leave-one-ortho-out.

Di sini ambang untuk lipatan f dipilih dari kurva lipatan LAIN saja:

    conf*(f) = argmax_c  rata-rata F1 lipatan != f pada ambang c

sehingga ortomosaik yang ditahan tidak pernah ikut memilih apa pun tentang
dirinya sendiri. Ambang yang dipilih dicetak per lipatan; kalau ketiganya
mendarat di nilai yang sama, itu tanda pilihannya stabil, bukan kebetulan.

MENGAPA INFERENSI DIULANG PER AMBANG, BUKAN DISARING SETELAHNYA.

Menyaring hasil yang sudah digabung pada ambang rendah TIDAK sama dengan
menggabung ulang pada ambang tinggi: penggabungan bersifat serakah menurut
keyakinan, jadi anggota gugus yang bertahan bisa berbeda. Karena satu lintasan
di GPU murah, `predict_global()` dipanggil ulang untuk tiap ambang. Lebih lambat,
tetapi angkanya angka yang sesungguhnya.

DERAJAT DILAPORKAN DUA CARA.

`semua` memasukkan pohon tepi, yang derajatnya rendah karena kebunnya habis -
bukan karena grafnya beda. `dalam` hanya pohon yang jaraknya > r dari tepi
kotak pembatas, dan ITULAH yang sebanding dengan angka 5,74 Eg9PP (juga pohon
bagian dalam). Jangan sandingkan `semua` dengan 5,74.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import y12  # noqa: E402

SEED = int(os.environ.get("SEED", 42))
# TAG dapat ditimpa lewat env supaya varian lain dinilai oleh KODE YANG SAMA,
# bukan salinannya - termasuk pemilihan ambang silang-lipatan, yang paling mudah
# menyimpang kalau diduplikasi. Dua pemakaiannya sejauh ini:
#   yolo12n_base_1fold   lari verifikasi satu lipatan (run_1fold_yolov12n.ipynb)
#   yolo12n_base_i1024   lengan kandidat imgsz 1024 (run_arms_gpu.py)
# Default tetap lari 3 lipatan, yaitu yang angkanya masuk naskah.
TAG = os.environ.get("TAG", "yolo12n_base")
IMGSZ = int(os.environ.get("IMGSZ", 640))
RADIUS_FRAC = 0.5          # ambang cocok, kelipatan jarak tanam
R_GRAPH = 1.5              # radius graf kontak, kelipatan jarak tanam
CONFS = [float(x) for x in os.environ.get(
    "CONFS", "0.25,0.35,0.45,0.55,0.65,0.75").split(",")]
FOLDS = [f"fold{int(x)}" for x in os.environ.get("FOLDS", "0,1,2").split(",")]


def weights_of(fold, tag=None):
    p = os.path.join(y12.RUNS, f"{tag or TAG}_{fold}_s{SEED}", "weights", "best.pt")
    return p if os.path.isfile(p) else None


def ortho_of(fold):
    first = open(os.path.join(y12.ROOT, f"{fold}_val.txt")).readline().strip()
    return y12._tile_offset(os.path.basename(first))[0]


def spacing_of(gt_xy):
    d, _ = cKDTree(gt_xy).query(gt_xy, k=2)
    return float(np.median(d[:, 1]))


def score(p_xy, gt_xy, spacing):
    mt = y12.centre_match(p_xy, gt_xy, RADIUS_FRAC * spacing)
    tp = len(mt)
    pr = tp / max(1, len(p_xy))
    rc = tp / max(1, len(gt_xy))
    rmse = (float(np.sqrt(((p_xy[mt[:, 0]] - gt_xy[mt[:, 1]]) ** 2).sum(1).mean()))
            if tp else float("nan"))
    return dict(n_pred=len(p_xy), n_gt=len(gt_xy), tp=tp, fp=len(p_xy) - tp,
                fn=len(gt_xy) - tp, precision=pr, recall=rc,
                f1=2 * pr * rc / max(1e-9, pr + rc),
                rmse_px=rmse, rmse_frac_spacing=rmse / spacing)


def degree(xy, r):
    """Derajat tiap titik pada radius r. -> (rata2 semua, rata2 pohon dalam, n_dalam)."""
    kd = cKDTree(xy)
    deg = np.array([len(kd.query_ball_point(p, r)) - 1 for p in xy], float)
    lo, hi = xy.min(0), xy.max(0)
    inner = ((xy[:, 0] > lo[0] + r) & (xy[:, 0] < hi[0] - r)
             & (xy[:, 1] > lo[1] + r) & (xy[:, 1] < hi[1] - r))
    return (float(deg.mean()),
            float(deg[inner].mean()) if inner.any() else float("nan"),
            int(inner.sum()))


def main():
    have = {f: weights_of(f) for f in FOLDS}
    missing = [f for f, w in have.items() if w is None]
    if missing:
        print("bobot belum ada untuk: %s" % ", ".join(missing))
        print("jalankan dulu:  FOLDS=%s python train_folds_gpu.py"
              % ",".join(m[-1] for m in missing))
        if all(w is None for w in have.values()):
            raise SystemExit(1)
    folds = [f for f in FOLDS if have[f]]

    # ---- 1. kurva P/R/F1 vs conf per lipatan --------------------------------
    curves, gts = {}, {}
    for f in folds:
        ortho = ortho_of(f)
        gt_xy, gt_lab = y12.gt_trees(ortho)
        sp = spacing_of(gt_xy)
        gts[f] = (ortho, gt_xy, gt_lab, sp)
        curves[f] = {}
        print("\n== %s (tahan %s, jarak tanam %.1f px, %d pohon GT) ==="
              % (f, ortho, sp, len(gt_xy)))
        for c in CONFS:
            p_xy, p_conf, p_cls = y12.predict_global(have[f], f, conf=c,
                                                     imgsz=IMGSZ, verbose=False)
            s = score(p_xy, gt_xy, sp)
            s["pred_xy"] = p_xy
            s["pred_cls"] = p_cls
            curves[f][c] = s
            print("   conf %.2f  n_pred=%5d  P=%.3f R=%.3f F1=%.3f  RMSE=%.2f x jarak tanam"
                  % (c, s["n_pred"], s["precision"], s["recall"], s["f1"],
                     s["rmse_frac_spacing"]))

    # ---- 2. pilih conf SILANG-LIPATAN --------------------------------------
    print("\n== pemilihan ambang (silang-lipatan, lipatan uji tidak ikut memilih) ==")
    chosen = {}
    for f in folds:
        others = [o for o in folds if o != f]
        if not others:
            chosen[f] = None
            print("   %s: hanya satu lipatan tersedia -> ambang TIDAK dapat dipilih bersih" % f)
            continue
        best = max(CONFS, key=lambda c: np.mean([curves[o][c]["f1"] for o in others]))
        chosen[f] = best
        print("   %s: conf=%.2f  (dipilih dari %s)" % (f, best, "+".join(others)))
    if len({c for c in chosen.values() if c is not None}) == 1:
        print("   ketiganya sepakat -> pilihan stabil")

    # ---- 3. metrik utama pada ambang terpilih -------------------------------
    print("\n== METRIK UTAMA Tahap 1: pusat tajuk pada pohon unik ==")
    rows, out = [], {}
    for f in folds:
        c = chosen[f]
        if c is None:
            continue
        s = curves[f][c]
        ortho, gt_xy, gt_lab, sp = gts[f]
        d_gt = degree(gt_xy, R_GRAPH * sp)
        d_pr = degree(s["pred_xy"], R_GRAPH * sp)
        rec = dict(fold=f, ortho=ortho, conf=c, spacing_px=sp,
                   **{k: v for k, v in s.items() if k not in ("pred_xy", "pred_cls")},
                   deg_gt_all=d_gt[0], deg_gt_inner=d_gt[1], n_inner_gt=d_gt[2],
                   deg_pred_all=d_pr[0], deg_pred_inner=d_pr[1], n_inner_pred=d_pr[2])
        rows.append(rec)
        out[f] = rec
        print("   %-6s %-13s conf=%.2f  P=%.3f R=%.3f F1=%.3f  RMSE=%.3f x jarak tanam"
              % (f, ortho, c, s["precision"], s["recall"], s["f1"], s["rmse_frac_spacing"]))

    if len(rows) > 1:
        print("\n   rata-rata %d ortomosaik (std = antar-situs, n kecil):" % len(rows))
        for k in ("precision", "recall", "f1", "rmse_frac_spacing"):
            v = np.array([r[k] for r in rows])
            print("     %-18s %.3f +/- %.3f" % (k, v.mean(), v.std(ddof=1)))

    # ---- 4. angka jembatan --------------------------------------------------
    print("\n== UJI JEMBATAN: derajat graf @ r = %.1f x jarak tanam ==" % R_GRAPH)
    print("   %-6s %-13s %-9s %-9s %-9s %-9s" % ("", "ortho", "GT semua", "GT dalam",
                                                 "pred semua", "pred dalam"))
    for r in rows:
        print("   %-6s %-13s %9.2f %9.2f %9.2f %9.2f"
              % (r["fold"], r["ortho"], r["deg_gt_all"], r["deg_gt_inner"],
                 r["deg_pred_all"], r["deg_pred_inner"]))
    if len(rows) > 1:
        gi = np.array([r["deg_gt_inner"] for r in rows])
        pi = np.array([r["deg_pred_inner"] for r in rows])
        print("\n   pohon DALAM, rata-rata %d ortomosaik:" % len(rows))
        print("     kotak kebenaran-dasar : %.2f +/- %.2f" % (gi.mean(), gi.std(ddof=1)))
        print("     prediksi detektor     : %.2f +/- %.2f" % (pi.mean(), pi.std(ddof=1)))
        print("     selisih               : %+.2f (%.1f%%)"
              % (pi.mean() - gi.mean(), 100 * (pi.mean() - gi.mean()) / gi.mean()))
    print("\n   pembanding Lapisan 2 (Eg9PP, pohon bagian dalam) = 5,74")
    print("   CATATAN: hanya kolom 'dalam' yang sebanding dengan 5,74.")

    # Garis dasar menulis `centre_eval.json`; lengan lain menulis berkasnya sendiri.
    # WAJIB dipisah: `demo_core.CENTRE_JSON` membaca `centre_eval.json` dan
    # menampilkan F1-nya di layar. Lengan eksperimen yang menimpanya akan membuat
    # demo memamerkan angka yang belum tervalidasi, tanpa satu pun galat muncul.
    p = os.path.join(y12.RESDIR, "centre_eval.json" if TAG == "yolo12n_base"
                     else "centre_eval_%s.json" % TAG)
    json.dump(dict(tag=TAG, seed=SEED, imgsz=IMGSZ, radius_frac=RADIUS_FRAC,
                   r_graph=R_GRAPH, confs=CONFS, chosen_conf=chosen,
                   conf_selection="silang-lipatan; lipatan uji tidak ikut memilih",
                   folds=out), open(p, "w"), indent=2)
    print("\nditulis: %s" % p)


if __name__ == "__main__":
    main()
