"""Pusat tajuk + graf kontak dari citra UAV sembarang, memakai bobot ds_B.

    python detect_centres.py <citra-atau-folder> [-o keluaran.csv] [--fold 0|1|2]
    python detect_centres.py ds_B_tiles/ --stitch      # ubin ds_B: gabungkan antar-ubin

Keluaran: satu baris per pohon terdeteksi -

    image, tree_id, cx, cy, conf, class, deg

`cx`,`cy` adalah PUSAT TAJUK. Itulah bentuk keluaran Tahap 1 yang dikonsumsi
Lapisan 2; kotaknya sendiri hanya perantara (lihat LABEL_QUALITY_AUDIT.md).

MENGAPA BUKAN `use_stage1.py`.

`use_stage1.py` juga menerima citra apa pun, tetapi memuat model PERU
(`stage1_model.pkl`). `limits` model itu menyatakan sendiri bahwa ia hanya
mengotaki sawit yang DIPILIH penganotasi - sekitar 1,27 kotak per citra padahal
bingkainya memuat 3-6 sawit. Dipakai membangun graf, ia akan kurang-mendeteksi
secara sistematis dan derajatnya bias ke bawah, TANPA memunculkan galat.
Untuk kotaki-setiap-sawit, bobot ds_B-lah yang benar, dan itu yang dipakai di sini.

AMBANG conf DEFAULT 0,75, BUKAN 0,25.

Nilai itu bukan selera. Ia dipilih SILANG-LIPATAN di `centre_eval_folds.py`:
ambang untuk tiap lipatan diambil dari kurva F1 lipatan lain saja, ketiganya
sepakat di 0,75, dan sapuan sampai 0,90 menunjukkan itu optimum interior.
Di 0,25 presisi jatuh ke 0,82 karena deteksi duplikat lolos penggabungan, dan
derajat graf menggelembung ~19%.

TIGA HAL YANG DICETAK SEBAGAI PERINGATAN, DAN MENGAPA.

1. F1 0,960 +/- 0,024 diukur leave-one-ortho-out DI DALAM SATU KEBUN. Citra dari
   kebun, sensor, atau ketinggian terbang lain adalah domain KEEMPAT yang belum
   pernah diukur. Angka itu tidak berpindah ke sana, dan skrip ini tidak akan
   berpura-pura sebaliknya.
2. Kelas `Unhealthy` adalah kesehatan tajuk generik tanpa verifikasi lapangan.
   BUKAN BSR, BUKAN Ganoderma.
3. Radius graf memakai jarak tanam yang DIESTIMASI dari citra itu sendiri
   (median jarak tetangga terdekat). Estimasi itu rapuh bila pohon yang
   terdeteksi sedikit, dan tidak berarti apa-apa bila skala citranya jauh dari
   acuan ds_B - karena itu keduanya diperiksa dan dilaporkan.
"""
import argparse
import csv
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import y12  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Acuan ds_B: jarak tanam terukur 101-106 px pada ubin 1024^2 (LABEL_QUALITY_AUDIT.md).
SPACING_REF = (101.1, 105.8)
SCALE_WINDOW = (0.8, 1.25)       # di luar ini, ubin harus di-resample dulu
MIN_TREES_FOR_SPACING = 20       # di bawah ini median NN tidak dapat dipercaya
R_GRAPH = 1.5                    # radius graf kontak, kelipatan jarak tanam

# F1 per lipatan pada ortomosaik yang DITAHAN (centre_eval_folds.py, conf 0,75).
FOLD_F1 = {"fold0": 0.969, "fold1": 0.977, "fold2": 0.933}


def images_in(path):
    if os.path.isfile(path):
        return [path]
    out = []
    for root, _, files in os.walk(path):
        out += [os.path.join(root, f) for f in sorted(files)
                if f.lower().endswith(IMG_EXT)]
    return out


def weights_for(fold):
    p = os.path.join(y12.RUNS, f"yolo12n_base_{fold}_s42", "weights", "best.pt")
    return p if os.path.isfile(p) else None


def detect(model, paths, conf, imgsz, batch, device, stitch):
    """-> (det, unparsed). det = daftar (x, y, conf, cls, src, GRUP).

    GRUP adalah kunci BIDANG KOORDINAT, dan ia wajib ada.

    Tanpa --stitch, tiap citra punya bidang pikselnya sendiri: (100,100) di dua
    foto berbeda bukan tempat yang sama. Dengan --stitch, offset ubin hanya
    berarti DI DALAM satu ortomosaik - ubin `44000_4000_512_0` dan
    `52000_20000_512_0` sama-sama menjadi (512,0) padahal berjarak berkilometer.

    Menggabungkan bidang yang berbeda menghasilkan tetangga palsu, jadi derajat
    grafnya menggelembung - dan tidak ada galat yang muncul. Karena itu grup
    dipisahkan di sini dan setiap grup diproses sendiri-sendiri.
    """
    det, unparsed = [], 0
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        res = model.predict(chunk, imgsz=imgsz, conf=conf, device=device, verbose=False)
        # WAJIB zip terhadap `chunk`. Ketika sumbernya DAFTAR path, Ultralytics 8.4
        # menamai ulang hasilnya 'image0.jpg', 'image1.jpg', ... sehingga r.path
        # tidak dapat dipakai. Kesalahan ini gagal diam-diam, bukan dengan galat.
        for src, r in zip(chunk, res):
            off, grp = (0, 0), os.path.basename(src)
            if stitch:
                p = y12._tile_offset(os.path.basename(src))
                if p is None:
                    unparsed += 1
                    continue
                off, grp = (p[1], p[2]), p[0]          # p[0] = id ortomosaik
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            for (cx, cy, bw, bh), c, k in zip(b.xywh.cpu().numpy(),
                                              b.conf.cpu().numpy(),
                                              b.cls.cpu().numpy().astype(int)):
                det.append((off[0] + cx, off[1] + cy, float(c), int(k), src, grp,
                            float(np.sqrt(bw * bh))))
    return det, unparsed


def merge(det, radius):
    """Penggabungan serakah menurut keyakinan - prosedur yang sama dengan
    `y12.predict_global()`, dipakai ulang supaya angkanya sebanding."""
    from scipy.spatial import cKDTree

    if not det:
        return []
    order = sorted(range(len(det)), key=lambda i: -det[i][2])
    xy = np.array([[det[i][0], det[i][1]] for i in order], float)
    kd = cKDTree(xy)
    keep = np.ones(len(order), bool)
    for i in range(len(order)):
        if not keep[i]:
            continue
        for j in kd.query_ball_point(xy[i], radius):
            if j > i:
                keep[j] = False
    return [det[order[i]] for i in range(len(order)) if keep[i]]


def spacing_of(xy):
    from scipy.spatial import cKDTree

    if len(xy) < 2:
        return float("nan")
    d, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(d[:, 1]))


# Radius penggabungan diambil dari UKURAN KOTAK TAJUK, bukan dari jarak tetangga
# terdekat deteksi mentah.
#
# MENGAPA. Pada ubin yang bertindih ~34x, tetangga terdekat sebuah deteksi adalah
# DUPLIKAT DIRINYA SENDIRI di ubin sebelah, berjarak <1 px. Median jarak tetangga
# terdekat karena itu mengukur jitter duplikat (0,4 px), bukan jarak tanam (105 px),
# sehingga radius gabung menjadi ~0,2 px dan TIDAK ADA yang tergabung - 1.849 pohon
# terbaca sebagai 51.459. `y12.predict_global()` lolos dari jebakan ini karena ia
# mengambil skala dari kotak kebenaran-dasar; citra sembarang tidak punya itu.
#
# Faktor 0,5 menyamai `y12.predict_global()`, yang menggabungkan pada 0,5 x jarak
# tanam dan terbukti menghasilkan presisi 0,95. Pada ds_B ukuran kotak (100 px) dan
# jarak tanam (101-106 px) memang berdekatan, jadi keduanya setara. Diuji: faktor
# 0,3 TERLALU KECIL - 44000_16000 terbaca 2.097 pohon, bukan ~1.400.
MERGE_FRAC = 0.5


def merge_two_pass(g, box):
    """Gabungkan duplikat, lalu perbaiki skalanya sekali dan gabungkan ulang.

    MENGAPA DUA LANGKAH. Skala awal diambil dari ukuran kotak tajuk, dan itu hanya
    setara jarak tanam bila tajuknya sudah saling bersentuhan. Pada sawit muda,
    tajuk jauh lebih kecil daripada jarak tanam, sehingga radius langkah pertama
    terlalu sempit dan sebagian duplikat lolos. Setelah langkah pertama, jarak
    tetangga terdekat sudah cukup bersih untuk dipakai sebagai skala sebenarnya.

    Radius HANYA boleh membesar di langkah kedua. Kalau estimasi jarak tanam keluar
    lebih kecil daripada ukuran tajuk, itu tanda duplikat masih ada - dan mengecilkan
    radius karena estimasi yang tercemar justru mengunci kesalahannya.
    """
    g1 = merge(g, MERGE_FRAC * box)
    sp1 = spacing_of(np.array([[d[0], d[1]] for d in g1], float))
    scale = max(box, sp1) if np.isfinite(sp1) else box
    if scale <= box * 1.05:
        return g1, box                       # langkah kedua tidak akan mengubah apa pun
    return merge(g, MERGE_FRAC * scale), scale


def degrees(xy, r):
    """-> (deg per titik, rata2 semua, rata2 pohon dalam, n_dalam)."""
    from scipy.spatial import cKDTree

    kd = cKDTree(xy)
    deg = np.array([len(kd.query_ball_point(p, r)) - 1 for p in xy], float)
    lo, hi = xy.min(0), xy.max(0)
    inner = ((xy[:, 0] > lo[0] + r) & (xy[:, 0] < hi[0] - r)
             & (xy[:, 1] > lo[1] + r) & (xy[:, 1] < hi[1] - r))
    return (deg, float(deg.mean()),
            float(deg[inner].mean()) if inner.any() else float("nan"),
            int(inner.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="berkas citra atau folder")
    ap.add_argument("-o", "--out", default="centres.csv")
    ap.add_argument("--fold", default="fold0", choices=["fold0", "fold1", "fold2"],
                    help="lipatan mana bobotnya dipakai (default fold0)")
    ap.add_argument("-w", "--weights", default=None, help="timpa jalur bobot")
    ap.add_argument("--conf", type=float, default=0.75,
                    help="default 0,75 - dipilih silang-lipatan, jangan diubah tanpa alasan")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--stitch", action="store_true",
                    help="nama berkas ubin ds_B (X_Y_tx_ty) -> satukan ke koordinat ortomosaik")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    w = a.weights or weights_for(a.fold)
    if w is None:
        raise SystemExit(
            "bobot %s tidak ada.\n  latih dulu:  FOLDS=%s python train_folds_gpu.py"
            % (a.fold, a.fold[-1]))

    paths = images_in(a.source)
    if not paths:
        raise SystemExit("tidak ada citra di %s" % a.source)

    import torch
    from ultralytics import YOLO

    dev = a.device if a.device is not None else (0 if torch.cuda.is_available() else "cpu")
    model = YOLO(w)

    print("bobot   : %s" % os.path.relpath(w, BASE))
    if not a.weights:
        print("          F1 pusat %.3f pada ortomosaik yang DITAHAN lipatan ini"
              % FOLD_F1[a.fold])
    print("citra   : %d  |  conf %.2f  |  imgsz %d  |  device %s"
          % (len(paths), a.conf, a.imgsz, dev))

    det, unparsed = detect(model, paths, a.conf, a.imgsz, a.batch, dev, a.stitch)
    if unparsed:
        print("PERHATIAN: %d berkas namanya tidak berpola ubin ds_B, dilewati "
              "(--stitch aktif)" % unparsed)
    if not det:
        raise SystemExit("TIDAK ADA deteksi di atas conf=%.2f. Turunkan --conf, atau "
                         "skala citranya di luar jangkauan model." % a.conf)

    groups = {}
    for d in det:
        groups.setdefault(d[5], []).append(d)
    print("deteksi : %d mentah dalam %d bidang koordinat (%s)"
          % (len(det), len(groups), "ortomosaik" if a.stitch else "citra"))

    rows, summary = [], []
    for gname in sorted(groups):
        g = groups[gname]
        n_raw = len(g)
        box = float(np.median([d[6] for d in g]))
        g, scale = merge_two_pass(g, box)
        xy = np.array([[d[0], d[1]] for d in g], float)
        spacing = spacing_of(xy)

        show = len(groups) <= 8            # jangan banjiri layar untuk ratusan citra
        if show:
            print("\n== %s: %d deteksi -> %d pohon (kotak tajuk %.0f px, gabung <%.0f px) =="
                  % (gname, n_raw, len(g), box, MERGE_FRAC * scale))
        # Jarak tanam WAJIB jauh lebih besar daripada radius gabung. Kalau tidak, yang
        # terukur masih jitter duplikat dan seluruh angka graf di bawah tak berarti.
        dup_left = np.isfinite(spacing) and spacing < MERGE_FRAC * scale * 1.5
        if dup_left:
            print("   PERINGATAN: jarak tanam terukur %.1f px terlalu dekat dengan radius "
                  "gabung %.1f px.\n   Duplikat kemungkinan belum tuntas - JANGAN pakai "
                  "angka derajat di bawah." % (spacing, MERGE_FRAC * scale))

        # ---- pemeriksaan skala. Ini yang mencegah kegagalan diam-diam. -------
        ok_scale, ok_n = False, len(g) >= MIN_TREES_FOR_SPACING and not dup_left
        if not ok_n:
            if show:
                print("   hanya %d pohon (< %d) - jarak tanam tak dapat diestimasi, "
                      "graf TIDAK dibangun" % (len(g), MIN_TREES_FOR_SPACING))
        else:
            ratio = spacing / np.mean(SPACING_REF)
            ok_scale = SCALE_WINDOW[0] <= ratio <= SCALE_WINDOW[1]
            if show:
                print("   jarak tanam %.1f px  ->  rasio skala %.2fx  (acuan ds_B %.0f-%.0f px)"
                      % (spacing, ratio, SPACING_REF[0], SPACING_REF[1]))
                print("   %s jendela %.2f-%.2fx%s"
                      % ("DI DALAM" if ok_scale else "DI LUAR", SCALE_WINDOW[0],
                         SCALE_WINDOW[1],
                         "" if ok_scale else " - resample dulu; angka di bawah TIDAK sebanding"))

        deg = np.full(len(g), np.nan)
        d_all = d_in = float("nan")
        n_in = 0
        if ok_n:
            deg, d_all, d_in, n_in = degrees(xy, R_GRAPH * spacing)
            if show:
                print("   derajat @ r=%.1f : %.2f semua  |  %.2f pohon dalam (%d)"
                      % (R_GRAPH, d_all, d_in, n_in))
        summary.append((gname, len(g), spacing, ok_scale, d_all, d_in, n_in))
        for i, (cx, cy, c, k, src, _, _) in enumerate(g):
            rows.append([gname, os.path.basename(src), i, round(cx, 1), round(cy, 1),
                         round(c, 4), y12.NAMES[k],
                         "" if not np.isfinite(deg[i]) else int(deg[i])])

    if len(groups) > 8:
        ok = [s for s in summary if s[3] and s[6]]
        print("\n%d bidang; %d lolos pemeriksaan skala dan punya pohon bagian dalam."
              % (len(groups), len(ok)))
        if ok:
            v = np.array([s[5] for s in ok])
            print("   derajat pohon dalam: %.2f +/- %.2f antar bidang" % (v.mean(), v.std()))

    if any(s[3] and s[6] for s in summary):
        print("\n   pembanding: 5,54 +/- 0,12 prediksi ds_B (3 ortomosaik)  |  5,74 Eg9PP")

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["group", "image", "tree_id", "cx", "cy", "conf", "class", "deg"])
        wr.writerows(rows)
    print("\nditulis: %s  (%d baris)" % (a.out, len(rows)))

    print("\n" + "-" * 74)
    print("BATAS YANG MELEKAT PADA ANGKA DI ATAS")
    print("  1. F1 0,960 +/- 0,024 diukur leave-one-ortho-out DI DALAM SATU KEBUN.")
    print("     Citra dari kebun/sensor/ketinggian lain adalah domain yang BELUM")
    print("     PERNAH diukur. Jangan kutip angka itu untuk keluaran ini.")
    print("  2. Kelas 'Unhealthy' = kesehatan tajuk generik tanpa verifikasi lapangan.")
    print("     BUKAN BSR, BUKAN Ganoderma.")
    print("  3. Keluaran ini adalah KOORDINAT + graf. Ia TIDAK dapat disuapkan ke")
    print("     layer2_real/stgnn_final.pt: checkpoint itu meminta 24 fitur, 18 di")
    print("     antaranya (4 waktu + 14 genotipe) tidak ada pada citra tunggal.")
    print("-" * 74)


if __name__ == "__main__":
    main()
