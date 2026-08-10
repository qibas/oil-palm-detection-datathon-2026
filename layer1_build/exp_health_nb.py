"""Apakah PENAMPAKAN TETANGGA menolong klasifikasi kesehatan tajuk Lapisan 1?

    python exp_health_nb.py [seeds]        # default 5 seed x 3 lipatan = 15 pasangan

PERTANYAANNYA, dan mengapa ia bukan sekadar menambah fitur.

Temuan utama Lapisan 2 adalah bahwa yang membawa sinyal justru KONDISI TETANGGA
lewat graf kontak: 77% kemampuan varian foto datang khusus dari peta kontak yang
BENAR. Tetapi klasifikator kesehatan Lapisan 1 (`exp_health.py`) sama sekali tidak
melihat tetangga - seluruh 20 fiturnya dihitung dari potongan tajuk pohon itu
sendiri. Kedua lapisan karena itu menceritakan hal yang berbeda tentang hal yang
sama.

Modul ini menguji analog Lapisan 1 dari temuan Lapisan 2: pada radius kontak yang
SAMA (r = 1,5 x jarak tanam, radius yang dipakai uji jembatan), apakah penampakan
tetangga menambah kemampuan memisahkan tajuk sakit?

Fitur yang ditambahkan ada dua jenis, dan yang kedua yang menarik:

    NB_*   agregat penampakan tetangga  (rata-rata, minimum, sebaran)
    d_*    KONTRAS terhadap tetangga    (nilai sendiri - rata-rata tetangga)

`d_*` adalah bentuk yang benar secara fisik. Klorosis itu relatif: tajuk yang
agak kusam di kebun yang kusam adalah normal, sedangkan tajuk yang sama di kebun
yang subur adalah anomali. Fitur mutlak tidak dapat menyatakan perbedaan itu;
fitur kontras bisa.

LARANGAN KEBOCORAN YANG DIJAGA DI SINI.

Fitur tetangga dihitung SEMATA dari PENAMPAKAN tetangga - tidak pernah dari
labelnya. Memakai label tetangga akan membocorkan target lewat pintu belakang,
karena tajuk sakit memang mengelompok (lihat `premise_test.py`). Penjaga
`_assert_no_label_leak()` menegakkan itu secara struktural: pembangun fitur
tetangga tidak pernah menerima `y` sebagai argumen sama sekali.

Tetangga selalu dicari DI DALAM ortomosaik yang sama, dan lipatan = ortomosaik,
jadi tidak ada tetangga yang menyeberangi batas lipatan.

HARAPAN YANG JUJUR. Positif unik hanya 66 (17/31/18 per lipatan). Pita deraunya
lebar dan hasil TIDAK KONKLUSIF adalah keluaran yang sangat mungkin - dan itu
jawaban yang benar, bukan kegagalan. Aturan putusnya sama dengan seluruh paket:
|mean| <= 1 std => TIDAK KONKLUSIF.
"""
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import exp_health as EH  # noqa: E402

R_CONTACT = 1.5          # radius graf kontak, kelipatan jarak tanam - SAMA dgn uji jembatan

# Fitur penampakan yang diagregasi dari tetangga. Dipilih karena inilah yang
# dominan pada model dasar (GmR, exg_std, GmB teratas) - yaitu greenness/klorosis.
NB_SRC = ["exg_mean", "GmR", "GmB", "green_frac", "bright", "crown_cov"]


def _assert_no_label_leak(fn):
    """Penjaga struktural: pembangun fitur tetangga tidak boleh punya parameter label."""
    import inspect
    bad = {"y", "label", "labels", "target"} & set(inspect.signature(fn).parameters)
    if bad:
        raise SystemExit("KEBOCORAN: %s menerima %s - fitur tetangga tidak boleh "
                         "melihat label." % (fn.__name__, sorted(bad)))


def neighbour_features(X, feat_names, xy, regs, view="true", seed=0, verbose=True):
    """Agregat + kontras penampakan tetangga. TIDAK menerima label, sengaja.

    `view="true"`   tetangga sesungguhnya dalam radius kontak.
    `view="random"` KONTROL: jumlah tetangga tiap pohon DIPERTAHANKAN, tetapi
                    pasangannya diundi acak dari ortomosaik yang sama. Ini analog
                    Lapisan 1 dari view `random` di `run_real.py` - ia menjawab
                    "apakah yang menolong itu PETA KONTAK yang benar, atau sekadar
                    'penampakan kebun ini secara umum'?" Kalau perolehan bertahan
                    terhadap kontrol ini, yang diukur bukan struktur melainkan
                    normalisasi tingkat-kebun, dan klaimnya harus ditulis begitu.
    """
    idx = {k: i for i, k in enumerate(feat_names)}
    src = [k for k in NB_SRC if k in idx]
    n = len(X)
    rng = np.random.default_rng(seed)
    n_nb = np.zeros(n, np.float32)
    agg_mean = np.zeros((n, len(src)), np.float32)
    agg_min = np.zeros((n, len(src)), np.float32)
    agg_std = np.zeros((n, len(src)), np.float32)

    for reg in sorted(set(regs)):
        m = np.flatnonzero(regs == reg)
        P = xy[m]
        kd = cKDTree(P)
        d, _ = kd.query(P, k=2)
        spacing = float(np.median(d[:, 1]))
        r = R_CONTACT * spacing
        if verbose:
            print("  %-13s jarak tanam %.1f px -> radius kontak %.1f px"
                  % (reg, spacing, r))
        V = X[np.ix_(m, [idx[k] for k in src])]
        for a, p in enumerate(P):
            nb = [b for b in kd.query_ball_point(p, r) if b != a]
            if view == "random" and nb:
                # derajat dipertahankan, identitas tetangga dihancurkan
                pool = rng.choice(len(m), size=len(nb) + 1, replace=False)
                nb = [b for b in pool if b != a][:len(nb)]
            n_nb[m[a]] = len(nb)
            if nb:
                W = V[nb]
                agg_mean[m[a]] = W.mean(0)
                agg_min[m[a]] = W.min(0)
                agg_std[m[a]] = W.std(0)
            else:                       # pohon terisolasi: kontras = 0, bukan NaN
                agg_mean[m[a]] = V[a]
                agg_min[m[a]] = V[a]

    own = X[:, [idx[k] for k in src]]
    contrast = own - agg_mean           # <- bentuk yang benar secara fisik
    NB = np.hstack([n_nb[:, None], agg_mean, agg_min, agg_std, contrast])
    names = (["n_nb"]
             + ["nb_%s_mean" % k for k in src]
             + ["nb_%s_min" % k for k in src]
             + ["nb_%s_std" % k for k in src]
             + ["d_%s" % k for k in src])
    return NB.astype(np.float32), names, len(src)


def paired(a, b, label):
    """Aturan putus yang sama dengan y12.paired()/run_real.paired(): ddof=1,
    dan |mean| <= 1 std => TIDAK KONKLUSIF."""
    d = np.asarray(b, float) - np.asarray(a, float)
    mean = float(d.mean())
    std = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    wins = int((d > 0).sum())
    verdict = ("TIDAK KONKLUSIF" if not (abs(mean) > std)
               else ("POSITIF" if mean > 0 else "NEGATIF"))
    print("  %-34s d=%+.4f +/- %.4f  tanda %d/%d  [%s]"
          % (label, mean, std, wins, len(d), verdict))
    return mean, std, wins, verdict


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    t0 = time.time()
    _assert_no_label_leak(neighbour_features)

    d = EH.extract()
    X, y, regs, names = d["X"], d["y"], d["regs"], d["feat_names"]

    print("\n== membangun fitur tetangga (r = %.1f x jarak tanam) ==" % R_CONTACT)
    NB, nb_names, nsrc = neighbour_features(X, names, d["xy"], regs, view="true")
    print("  %d fitur tetangga ditambahkan (%d sumber x 4 bentuk + n_nb)"
          % (NB.shape[1], nsrc))
    print("  derajat rata-rata: %.2f  (pembanding Eg9PP pohon dalam = 5,74)"
          % NB[:, 0].mean())
    print("\n== KONTROL: tetangga acak, derajat dipertahankan ==")
    NBr, _, _ = neighbour_features(X, names, d["xy"], regs, view="random",
                                   seed=7, verbose=False)

    d_only = [i for i, k in enumerate(nb_names) if k.startswith("d_")]
    X_nb = np.hstack([X, NB]).astype(np.float32)
    X_ct = np.hstack([X, NB[:, d_only]]).astype(np.float32)
    X_ctr = np.hstack([X, NBr[:, d_only]]).astype(np.float32)

    variants = [("dasar (20 fitur tajuk)", X, names),
                ("+ KONTRAS acak (kontrol)", X_ctr, names + [nb_names[i] for i in d_only]),
                ("+ KONTRAS saja", X_ct, names + [nb_names[i] for i in d_only]),
                ("+ semua fitur tetangga", X_nb, names + nb_names)]

    print("\n== block-CV leave-one-ortho-out, %d seed x 3 lipatan ==" % seeds)
    per = {}
    for tag, XX, _ in variants:
        folds = []
        for s in range(seeds):
            _, _, pf = EH.run(XX, y, regs, {**EH.COMMON, "seed": 42 + s},
                              tag, verbose=False)
            folds.extend(pf)
        per[tag] = folds
        arr = np.array(folds)
        print("  %-26s PR-AUC %.4f  (ddof1 %.4f | ddof0 %.4f)  n=%d"
              % (tag, arr.mean(), arr.std(ddof=1), arr.std(), len(arr)))

    print("\n== putusan berpasangan (unit = lipatan x seed) ==")
    base = per["dasar (20 fitur tajuk)"]
    print("  -- nilai punya tetangga SAMA SEKALI --")
    paired(base, per["+ KONTRAS acak (kontrol)"], "kontras ACAK - dasar")
    paired(base, per["+ KONTRAS saja"], "kontras BENAR - dasar")
    paired(base, per["+ semua fitur tetangga"], "semua tetangga - dasar")
    print("  -- STRUKTUR: nilai peta kontak yang BENAR (analog run_real.py) --")
    paired(per["+ KONTRAS acak (kontrol)"], per["+ KONTRAS saja"], "benar - acak")

    print("\n== per lipatan, model dasar lawan + semua tetangga (seed pertama) ==")
    for i, reg in enumerate(sorted(set(regs))):
        print("  %-13s %.4f -> %.4f  (%+.4f)"
              % (reg, base[i], per["+ semua fitur tetangga"][i],
                 per["+ semua fitur tetangga"][i] - base[i]))

    print("\nCATATAN std. Angka beku 0,182 +/- 0,059 di 00_RINGKASAN.csv memakai")
    print("ddof=0. `y12.paired()` memakai ddof=1 dengan alasan eksplisit bahwa pada")
    print("n kecil ddof=0 menyempitkan pita derau secara palsu. Pada n=3 selisihnya")
    print("besar: 0,059 lawan 0,073. Kedua angka dicetak di atas; yang dipakai untuk")
    print("PUTUSAN adalah ddof=1, konsisten dengan sisa paket.")
    print("\nBATAS. 66 positif unik (17/31/18 per lipatan). Seed menambah pasangan")
    print("tetapi TIDAK menambah situs: ia mempersempit derau optimisasi, bukan")
    print("ketidakpastian antar-kebun. Klaim generalisasi tetap dibatasi n=3.")
    print("\nTOTAL %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
