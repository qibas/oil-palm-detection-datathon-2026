"""Berapa nilainya kalau detektor bisa membedakan MATI dari BERGEJALA?

    python run_v3_cols.py [seeds]        # default 10 seed x 2 lipatan

Menulis `results_v3_cols.csv`.

PERTANYAANNYA, DAN KENAPA IA BUKAN SOAL DETEKTOR.

Detektor ds_B hanya punya dua kelas: Healthy / Unhealthy. Ia TIDAK dapat
membedakan sawit mati dari sawit yang tajuknya merana, sehingga jalur foto
terpaksa memakai satu kolom biner - dan pada ubin khas itu menghasilkan hanya
DUA tingkat skor.

Menambah kelas ke detektor itu pekerjaan berbulan: anotasi baru, latih ulang,
validasi. Sebelum membayarnya, pertanyaan yang benar adalah: BERAPA NILAINYA?
Itu bisa dijawab hari ini tanpa menyentuh detektor sama sekali, karena Eg9PP
memang punya status terpisah S / D / C.

VARIAN YANG DIUJI (semua W=1, graf benar, leave-one-parcel-out):

    1 kolom   is_sympt                  yang bisa diberi detektor SEKARANG
    2 kolom   + is_dead                 kalau detektor punya kelas "mati"   <- target
    3 kolom   + is_cens                 MUSTAHIL dari foto; sebagai plafon
    6 kolom   + selisih antar sensus    butuh dua kunjungan; plafon kedua

Selisih 1->2 kolom adalah angka keputusannya. Selisih 2->3 dan 3->6 menunjukkan
berapa yang tetap di luar jangkauan foto tunggal, apa pun yang dilakukan pada
detektor.

JUGA DILAPORKAN: jumlah TINGKAT skor yang berbeda per varian. Itu yang langsung
terlihat di peta demo - satu kolom memberi dua pita, dan pertanyaan pengguna
"kenapa tidak lima" dijawab di sini dengan angka, bukan penjelasan.
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

OUT = os.path.join(HERE, "results_v3_cols.csv")
H, EPOCHS = V3.H, V3.EPOCHS
NO_SKILL = 0.0632

VARIANTS = [
    ("1 kolom  is_sympt", [0], "yang bisa diberi detektor SEKARANG"),
    ("2 kolom  + is_dead", [0, 1], "kalau detektor punya kelas mati  <- TARGET"),
    ("3 kolom  + is_cens", [0, 1, 2], "is_cens mustahil dari foto"),
    ("6 kolom  + selisih", [0, 1, 2, 3, 4, 5], "butuh dua kunjungan"),
]


def levels_on_riskset(cols):
    """Berapa TINGKAT skor berbeda yang mungkin, dari kombinasi tetangga yang ada.

    Dihitung dari data, bukan dari model: model 1-kolom monoton pada satu skalar,
    jadi jumlah tingkat = jumlah nilai difusi yang berbeda di risk set.
    """
    T = len(ds.census())
    X = v3.node_features_v3(np.arange(T))[:, :, cols]
    A = np.asarray(ds.adjacency("true"), np.float32)
    D = np.einsum("ij,tjd->tid", A * R.adjacency_scale(A), X)
    tree, t_idx, _ = ds.build_examples(H, np.arange(T))
    vals = np.round(D[t_idx, tree], 6)
    return len(np.unique(vals, axis=0))


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print("=" * 74)
    print("NILAI MEMBEDAKAN MATI DARI BERGEJALA  |  h=%d  |  %d seed x %d lipatan"
          % (H, seeds, len(ds.folds())))
    print("=" * 74)

    T = len(ds.census())
    Xfull = v3.node_features_v3(np.arange(T))
    names = v3.feature_names_v3()
    rows, res = [], {}

    for label, cols, note in VARIANTS:
        arm = V3.Arm(label, Xfull[:, :, cols], 1)
        vals = []
        for fold in range(len(ds.folds())):
            for s in range(seeds):
                r = V3.train_eval(arm, fold, s, "STGNN", "true")
                if r:
                    vals.append(r["within"])
        v = np.array(vals, float)
        nlev = levels_on_riskset(cols)
        res[label] = v
        print("%-22s AP %.4f +/- %.4f   lift %.2fx   tingkat mungkin: %d"
              % (label, v.mean(), v.std(ddof=1), v.mean() / NO_SKILL, nlev))
        print("%-22s %s  [%s]" % ("", note, ", ".join(names[c] for c in cols)))
        rows.append(dict(blok="v3-cols", varian=label, h=H, n=len(v),
                         kolom=len(cols), fitur="|".join(names[c] for c in cols),
                         ap_mean=v.mean(), ap_std=v.std(ddof=1),
                         lift=v.mean() / NO_SKILL, tingkat=nlev, catatan=note))

    print("\n--- selisih berpasangan (aturan run_real.py::paired) ---")
    pairs = [("NILAI KELAS MATI      (2 kolom - 1 kolom)", 1, 0),
             ("+ penyensoran         (3 kolom - 2 kolom)", 2, 1),
             ("+ selisih antar waktu (6 kolom - 3 kolom)", 3, 2)]
    for lbl, a, b in pairs:
        va, vb = res[VARIANTS[a][0]], res[VARIANTS[b][0]]
        n = min(len(va), len(vb))
        d = va[:n] - vb[:n]
        m, sd = d.mean(), d.std(ddof=1)
        sign = int((d > 0).sum())
        verdict = "TIDAK KONKLUSIF" if abs(m) < sd else ("POS" if m > 0 else "NEG")
        print("  %-42s %+.4f +/- %.4f  %2d/%d  %s" % (lbl, m, sd, sign, n, verdict))
        rows.append(dict(blok="v3-cols-paired", varian=lbl, h=H, n=n,
                         ap_mean=m, ap_std=sd, sign="%d/%d" % (sign, n), vonis=verdict))

    keys = sorted({k for r in rows for k in r})
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\nditulis: %s" % OUT)
    print("\nCATATAN: 'tingkat mungkin' dihitung dari data Eg9PP yang sebaran gejalanya "
          "40,6%. Pada ubin drone yang laju gejalanya 1,1%, jumlah tingkat jauh lebih "
          "kecil apa pun jumlah kolomnya - keterbatasannya ada pada berapa banyak "
          "penyakit yang terlihat, bukan pada model.")


if __name__ == "__main__":
    main()
