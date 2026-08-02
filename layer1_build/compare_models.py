"""Bandingkan beberapa backbone detektor pada lipatan leave-one-ortho-out yang sama.

Membaca setiap `yolo_results_<tag>.json` yang ditulis `yolo_train.py`, lalu
melaporkan selisih BERPASANGAN per lipatan terhadap satu garis dasar.

Aturan keputusannya sengaja disamakan dengan Layer 2 (`run_experiment.py::paired`):
selisih yang berada di dalam satu simpangan baku dinyatakan TIDAK KONKLUSIF.
Simpangan bakunya memakai ddof=1, bukan ddof=0, supaya pita derau tidak
menyempit palsu pada n yang sangat kecil.

PERINGATAN YANG MELEKAT: hanya ada 3 ortomosaik, jadi n = 3. Itu bukan
sampel untuk menyatakan satu detektor lebih baik dari yang lain. Skrip ini
ada untuk mencegah pembacaan berlebih, bukan untuk memberi izin mengklaim.

Pakai:
    set MODEL=yolo11n.pt && python yolo_train.py
    set MODEL=yolo12n.pt && python yolo_train.py
    set MODEL=yolo26n.pt && python yolo_train.py
    python compare_models.py                  # garis dasar = yolo11n
    python compare_models.py yolo12n          # garis dasar = yolo12n
"""
import glob
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
METRICS = [("map", "mAP50-95"), ("map50", "mAP50"), ("mp", "Precision"), ("mr", "Recall")]


def load():
    runs = {}
    for path in sorted(glob.glob(os.path.join(BASE, "yolo_results_*.json"))):
        d = json.load(open(path))
        if "folds_result" not in d:
            print("LEWATI (format lama, jalankan ulang): %s" % os.path.basename(path))
            continue
        runs[d["tag"]] = d
    return runs


def check_comparable(runs):
    """Perbandingan hanya sah kalau setelan dan lipatannya identik."""
    ref = None
    for tag, d in sorted(runs.items()):
        key = (d["epochs"], d["imgsz"], tuple(d["folds"]), d.get("seed"))
        if ref is None:
            ref, ref_tag = key, tag
        elif key != ref:
            raise SystemExit(
                "SETELAN BERBEDA, perbandingan dibatalkan.\n"
                "  %-10s epochs=%d imgsz=%d folds=%s seed=%s\n"
                "  %-10s epochs=%d imgsz=%d folds=%s seed=%s\n"
                "Jalankan ulang dengan EPOCHS/IMGSZ/FOLDS yang sama."
                % (ref_tag, ref[0], ref[1], list(ref[2]), ref[3],
                   tag, key[0], key[1], list(key[2]), key[3]))
    return ref


def main():
    runs = load()
    if len(runs) < 2:
        raise SystemExit("Perlu minimal dua berkas yolo_results_<tag>.json. Ditemukan: %s"
                         % (sorted(runs) or "tidak ada"))

    epochs, imgsz, folds, seed = check_comparable(runs)
    folds = list(folds)
    base = sys.argv[1] if len(sys.argv) > 1 else ("yolo11n" if "yolo11n" in runs
                                                  else sorted(runs)[0])
    if base not in runs:
        raise SystemExit("Garis dasar '%s' tidak ada. Tersedia: %s" % (base, sorted(runs)))

    print("setelan  : epochs=%d imgsz=%d folds=%s seed=%s" % (epochs, imgsz, folds, seed))
    print("model    : %s" % ", ".join(sorted(runs)))
    print("garis dasar: %s\n" % base)

    def val(tag, i, k):
        return runs[tag]["folds_result"][str(i)][k]

    for key, label in METRICS:
        print("=" * 78)
        print("%s  (leave-one-ortho-out, n = %d lipatan)" % (label, len(folds)))
        head = "  %-10s" % "model" + "".join("  fold%-7d" % i for i in folds)
        print(head + "  %9s" % "mean+/-std")
        for tag in sorted(runs):
            v = [val(tag, i, key) for i in folds]
            print("  %-10s" % tag + "".join("  %11.3f" % x for x in v)
                  + "  %.3f+/-%.3f" % (np.mean(v), np.std(v, ddof=1)))

        print("  -- selisih berpasangan terhadap %s --" % base)
        for tag in sorted(runs):
            if tag == base:
                continue
            d = np.array([val(tag, i, key) - val(base, i, key) for i in folds])
            mean, std = float(d.mean()), float(d.std(ddof=1))
            wins = int((d > 0).sum())
            # "<=" bukan "<": selisih persis nol (model identik, atau metrik yang
            # tidak bergerak) harus jatuh sebagai TIDAK KONKLUSIF, bukan NEGATIF.
            verdict = ("TIDAK KONKLUSIF" if abs(mean) <= std
                       else ("POSITIF" if mean > 0 else "NEGATIF"))
            print("  %-10s %+0.4f +/- %0.4f   tanda %d/%d   [%s]"
                  % (tag, mean, std, wins, len(folds), verdict)
                  + "   per-lipatan " + " ".join("%+0.4f" % x for x in d))
        print()

    print("=" * 78)
    print("CATATAN. n = %d ortomosaik. Pada ukuran ini pita deraunya lebar dan hampir"
          % len(folds))
    print("setiap selisih akan jatuh sebagai TIDAK KONKLUSIF. Itu jawaban yang benar,")
    print("bukan kegagalan skrip. Jangan menuliskan satu detektor unggul atas dasar ini;")
    print("yang boleh ditulis hanyalah bahwa keduanya setara dalam pita derau.")


if __name__ == "__main__":
    main()
