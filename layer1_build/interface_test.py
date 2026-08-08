"""Uji antarmuka "lebar sepur" — versi PUSAT TAJUK HASIL PREDIKSI DETEKTOR.

APA YANG BERUBAH DIBANDING VERSI LAMA.

Angka yang selama ini dilaporkan makalah — derajat tetangga rata-rata 5,74
(posisi tanam Eg9PP) lawan 5,62 (centroid tajuk ds_B), selisih 2% — dihitung dari
KOTAK KEBENARAN-DASAR ds_B. Karena itu `paper/section3.tex` menyebutnya BATAS
ATAS: ia menjawab "seandainya detektornya sempurna, apakah graf keluaran
Lapisan 1 sebangun dengan graf yang dimakan Lapisan 2?" — bukan "apakah graf
yang BENAR-BENAR dihasilkan pipeline sebangun".

Skrip ini menjawab pertanyaan kedua. Pusat tajuk diambil dari prediksi YOLOv12
pada ortomosaik yang DITAHAN (leave-one-ortho-out), digabungkan lintas-ubin
persis seperti `y12.predict_global()`, lalu derajatnya dihitung dengan
metodologi yang SAMA PERSIS dengan yang dipakai untuk Eg9PP dan untuk kotak
kebenaran-dasar. Ketiga angka karena itu sebanding.

PERINGATAN YANG MELEKAT (jangan dibuang saat mengutip angkanya).

Dugaan awal kami keliru dan dikoreksi di sini, bukan disembunyikan. Kami
menduga recall detektor akan membuat himpunan titik prediksi LEBIH JARANG
daripada kebenaran-dasar sehingga derajatnya bias KE BAWAH. Yang terukur justru
kebalikannya pada ambang bawaan pipeline (conf = 0,25): recall 0,96-0,999
tetapi presisi hanya 0,79-0,86, jadi himpunan prediksi 15-22% LEBIH PADAT
daripada kebenaran-dasar dan derajatnya bias KE ATAS.

Arah biasnya karena itu BERGANTUNG PADA AMBANG, dan keduanya nyata:
  * conf rendah  -> kelebihan deteksi. Jarak tanam ditaksir dari median jarak
    tetangga-terdekat himpunan itu sendiri, sehingga titik berlebih menarik
    taksiran itu ke bawah (terukur 0,85-0,99 x jarak tanam sebenarnya), seluruh
    kurva derajat bergeser, dan derajat naik.
  * conf tinggi  -> pohon hilang. Pada conf 0,70 recall fold2 jatuh ke 0,73 dan
    derajatnya bias ke bawah, persis seperti dugaan semula.

Karena itu satu angka pada satu ambang TIDAK cukup, dan `conf_sweep()` di bawah
wajib dilaporkan bersamanya. Jumlah pohon, recall, DAN presisi selalu ditulis
berdampingan dengan derajatnya. Lihat `INTERFACE_TEST.md`.

Jalankan:
    python interface_test.py                # semua; melatih ulang fold0 bila perlu
    python interface_test.py --no-train     # lewati fold0 bila bobotnya tak ada
    python interface_test.py --train-only   # hanya melatih ulang fold0
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

import y12

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(BASE), "data_clean")
OUT = os.path.join(BASE, "yolo12_results", "interface_test.json")

# Dataran kurva derajat. Di bawah 1,25 kurva masih menanjak, di atas 1,5 Eg9PP
# melompat ke cangkang tetangga kedua (kisi ideal) sementara ds_B melandai
# (posisi nyata berderau) — artefak presisi, bukan geometri yang berbeda.
FRACS = (1.25, 1.375, 1.5)

# Setelan detektor. Identik dengan lari ablasi `yolo12n_base` supaya bobot fold0
# yang dilatih ulang sebanding dengan bobot fold1/fold2 yang sudah ada.
MODEL = "yolo12n.pt"
SEED = 42
EPOCHS = 30
IMGSZ = 640
CONF = 0.25
FOLDS = ("fold0", "fold1", "fold2")


# --------------------------------------------------------------------------
# 1. Metodologi derajat — SATU definisi, dipakai untuk ketiga himpunan titik
# --------------------------------------------------------------------------
def interior_mask(xy, margin):
    """Pohon yang berjarak >= `margin` dari tepi convex hull himpunannya.

    MENGAPA POHON TEPI HARUS DIBUANG. Pohon di pinggir petak kehilangan tetangga
    hanya karena petaknya habis, bukan karena geometri tanamnya berbeda. Pada
    kisi segitiga sempurna pun derajat rata-rata seluruh pohon jatuh ke ~5,6
    sedangkan pohon bagian dalam tepat 6. Membandingkan dua himpunan yang rasio
    keliling-terhadap-luasnya berbeda tanpa membuang tepi = membandingkan bentuk
    petak, bukan jarak tanam.

    `margin` diset sama dengan radius tetangga r, sehingga setiap pohon yang
    lolos punya cakram radius r yang seluruhnya berada di dalam petak. Tetangga
    tetap dihitung terhadap SELURUH titik (termasuk titik tepi); yang dibatasi
    hanya himpunan yang dirata-ratakan.
    """
    h = ConvexHull(xy)
    # equations = [a, b, c] ternormalisasi, a*x + b*y + c <= 0 di dalam hull,
    # jadi jarak ke tepi = -max(a*x + b*y + c).
    dist = -(xy @ h.equations[:, :2].T + h.equations[:, 2]).max(1)
    return dist >= margin


def degree_stats(xy, frac, spacing=None, label=""):
    """Derajat tetangga rata-rata pada r = `frac` x jarak tanam.

    Jarak tanam = median jarak tetangga-terdekat HIMPUNAN ITU SENDIRI. Ini yang
    membuat ketiga himpunan sebanding walau satuannya beda (Eg9PP dalam satuan
    jarak tanam, ds_B dalam piksel 8,5-8,9 cm).
    """
    xy = np.asarray(xy, float)
    kd = cKDTree(xy)
    if spacing is None:
        spacing = float(np.median(kd.query(xy, k=2)[0][:, 1]))
    r = frac * spacing
    deg = np.array([len(kd.query_ball_point(p, r)) - 1 for p in xy], float)
    m = interior_mask(xy, r)
    return dict(label=label, frac=frac, spacing=spacing, radius=r,
                n_total=int(len(xy)), n_interior=int(m.sum()),
                deg_all=float(deg.mean()),
                deg_interior=float(deg[m].mean()) if m.any() else float("nan"),
                deg_interior_std=float(deg[m].std(ddof=1)) if m.sum() > 1 else float("nan"))


# --------------------------------------------------------------------------
# 2. Tiga himpunan titik
# --------------------------------------------------------------------------
def eg9pp_blocks():
    """Posisi tanam Eg9PP, DIPISAH PER PARCEL.

    Ini bukan detail kosmetik. Parcel 44A (y = 3,0-14,5) dan 44B (y = 22,0-33,5)
    terpisah 7,5 jarak tanam; `layer2_edges.csv` memang mencatat 0 sisi
    lintas-parcel. Satu convex hull atas gabungan keduanya membentang menyeberangi
    celah kosong itu, sehingga pohon di tepi-DALAM tiap parcel — yang justru
    kehilangan separuh tetangganya — dinyatakan "bagian dalam". Lihat catatan
    reproduksi angka 5,74 di INTERFACE_TEST.md.
    """
    import csv
    blocks = {}
    with open(os.path.join(DATA, "layer2_nodes.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            blocks.setdefault(r["parcel"], []).append((float(r["xm"]), float(r["ym"])))
    return {k: np.array(v, float) for k, v in sorted(blocks.items())}


def dsb_gt_blocks():
    """Centroid tajuk KEBENARAN-DASAR ds_B, per ortomosaik (5.077 pohon unik)."""
    return {o: y12.gt_trees(o)[0] for o in sorted(_orthos().values())}


def _orthos():
    """{nama_fold: prefiks_ortomosaik} untuk fold yang berkas val-nya ada."""
    out = {}
    for f in FOLDS:
        p = os.path.join(y12.ROOT, f"{f}_val.txt")
        if os.path.isfile(p):
            out[f] = y12._tile_offset(os.path.basename(open(p).readline().strip()))[0]
    return out


def weights_for(fold):
    return os.path.join(y12.RUNS, f"yolo12n_base_{fold}_s{SEED}", "weights", "best.pt")


def train_fold0(force=False):
    """Latih ulang fold0 dengan setelan IDENTIK dengan lari ablasi `yolo12n_base`.

    Bobot fold0 terhapus tak sengaja; metriknya selamat di
    `yolo12_results/yolo12n_base.json`. Fungsi ini memulihkan direktori lari yang
    hilang dan SENGAJA tidak menulis apa pun ke `yolo12_results/` — menulis ke
    sana dengan daftar fold yang berbeda akan menimpa hasil ablasi fold1/fold2.

    Argumennya disalin dari `y12.train_arm(arm="base", ...)`; lengan "base" tidak
    punya override augmentasi (`y12.ARMS["base"]` kosong), jadi tidak ada cfg
    tambahan yang perlu diteruskan.
    """
    import torch
    from ultralytics import YOLO

    w = weights_for("fold0")
    if os.path.isfile(w) and not force:
        print("fold0: bobot sudah ada, tidak dilatih ulang -> %s" % w)
        return w
    dev = 0 if torch.cuda.is_available() else "cpu"
    print("fold0: melatih ulang (epochs=%d imgsz=%d seed=%d device=%s)"
          % (EPOCHS, IMGSZ, SEED, dev), flush=True)
    m = YOLO(MODEL)
    m.train(data=os.path.join(y12.ROOT, "fold0.yaml"), epochs=EPOCHS, imgsz=IMGSZ,
            device=dev, batch=16, cache="ram", workers=0, seed=SEED,
            deterministic=True, project=y12.RUNS, name=f"yolo12n_base_fold0_s{SEED}",
            exist_ok=True, verbose=False, plots=False, **y12.arm_cfg("base"))
    return w


def dsb_pred_blocks(folds):
    """Pusat tajuk PREDIKSI per ortomosaik yang ditahan + recall-nya.

    Recall diukur dengan pencocokan satu-ke-satu pada 0,5 x jarak tanam
    kebenaran-dasar (`y12.centre_match`) — ambang yang sama dengan
    `y12.centre_eval`, yaitu "tidak mungkin tertukar dengan pohon tetangga".
    """
    out = {}
    for fold in folds:
        w = weights_for(fold)
        if not os.path.isfile(w):
            print("  %s: bobot TIDAK ADA (%s) -> dilewati" % (fold, w))
            continue
        ortho = _orthos()[fold]
        gt_xy, _ = y12.gt_trees(ortho)
        gt_sp = float(np.median(cKDTree(gt_xy).query(gt_xy, k=2)[0][:, 1]))
        p_xy, _, _ = y12.predict_global(w, fold, conf=CONF, imgsz=IMGSZ)
        mt = y12.centre_match(p_xy, gt_xy, 0.5 * gt_sp)
        out[fold] = dict(ortho=ortho, xy=p_xy, n_gt=len(gt_xy), n_pred=len(p_xy),
                         tp=len(mt),
                         recall=len(mt) / max(1, len(gt_xy)),
                         precision=len(mt) / max(1, len(p_xy)))
        print("  %s (%s): %d prediksi vs %d kebenaran-dasar | recall=%.3f presisi=%.3f"
              % (fold, ortho, len(p_xy), len(gt_xy), out[fold]["recall"],
                 out[fold]["precision"]))
    return out


def conf_sweep(folds, confs=(0.25, 0.4, 0.5, 0.6, 0.7)):
    """Kepekaan seluruh uji terhadap ambang keyakinan detektor.

    MENGAPA INI WAJIB ADA. Pada conf = 0,25 detektor MENGHASILKAN LEBIH BANYAK
    titik daripada kebenaran-dasar (presisi 0,79-0,86), sehingga median jarak
    tetangga-terdekat himpunan prediksi - yang dipakai sebagai penaksir jarak
    tanam - tertarik ke bawah. Akibatnya r = frac x jarak-tanam-taksiran menjadi
    radius absolut yang terlalu kecil dan seluruh kurva derajat bergeser ke kiri.
    Angka tunggal pada satu conf karena itu tidak dapat dipercaya sendirian.

    Sapuan ini dilaporkan APA ADANYA. Ia BUKAN alat memilih conf yang paling
    cocok dengan Eg9PP: memilih conf berdasarkan kecocokan dengan jawaban yang
    dituju adalah menyetel pada hasil, dan itu persis yang tidak boleh dilakukan.
    """
    rows = []
    for fold in folds:
        w = weights_for(fold)
        if not os.path.isfile(w):
            continue
        ortho = _orthos()[fold]
        gt_xy, _ = y12.gt_trees(ortho)
        gt_sp = float(np.median(cKDTree(gt_xy).query(gt_xy, k=2)[0][:, 1]))
        for c in confs:
            p_xy, _, _ = y12.predict_global(w, fold, conf=c, imgsz=IMGSZ, verbose=False)
            if len(p_xy) < 4:
                continue
            mt = y12.centre_match(p_xy, gt_xy, 0.5 * gt_sp)
            sp = float(np.median(cKDTree(p_xy).query(p_xy, k=2)[0][:, 1]))
            row = dict(fold=fold, ortho=ortho, conf=c, n_pred=int(len(p_xy)),
                       n_gt=int(len(gt_xy)), recall=len(mt) / len(gt_xy),
                       precision=len(mt) / len(p_xy),
                       spacing_pred=sp, spacing_gt=gt_sp,
                       spacing_ratio=sp / gt_sp)
            for frac in FRACS:
                row["deg_self_%.3f" % frac] = degree_stats(p_xy, frac)["deg_interior"]
                # DIAGNOSTIK: radius yang sama tetapi diskalakan dengan jarak tanam
                # KEBENARAN-DASAR. Memisahkan "bentuk awan titiknya berubah" dari
                # "penaksir jarak tanamnya yang bergeser". Angka ini memakai
                # ground truth, jadi ia BUKAN ukuran ujung-ke-ujung.
                row["deg_gtscale_%.3f" % frac] = degree_stats(
                    p_xy, frac, spacing=gt_sp)["deg_interior"]
            rows.append(row)
            print("  %s conf=%.2f n=%4d (GT %4d) R=%.3f P=%.3f sp=%.1f (%.2fx GT)  "
                  "deg@1,5 sendiri=%.3f / skala-GT=%.3f"
                  % (fold, c, row["n_pred"], row["n_gt"], row["recall"],
                     row["precision"], sp, row["spacing_ratio"],
                     row["deg_self_1.500"], row["deg_gtscale_1.500"]), flush=True)
    return rows


# --------------------------------------------------------------------------
# 3. Laporan
# --------------------------------------------------------------------------
def _agg(rows):
    v = np.array([r["deg_interior"] for r in rows], float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


def main(train=True, force_train=False, sweep=True):
    y12.build(extra_mode="ignore")
    print()

    folds = list(FOLDS)
    if train:
        try:
            train_fold0(force=force_train)
        except Exception as e:                       # noqa: BLE001
            print("fold0: pelatihan GAGAL (%s) -> dilaporkan pada fold yang ada" % e)
    folds = [f for f in folds if os.path.isfile(weights_for(f))]
    print("\nfold dengan bobot tersedia: %s" % (folds or "TIDAK ADA"))

    eg = eg9pp_blocks()
    gt = dsb_gt_blocks()
    print("\n=== prediksi detektor pada ortomosaik yang ditahan ===")
    pr = dsb_pred_blocks(folds)

    res = dict(fracs=list(FRACS), conf=CONF, imgsz=IMGSZ, seed=SEED, epochs=EPOCHS,
               folds_used=folds, eg9pp={}, dsb_gt={}, dsb_pred={},
               recall={f: {k: v for k, v in d.items() if k != "xy"}
                       for f, d in pr.items()})

    for frac in FRACS:
        res["eg9pp"][str(frac)] = [degree_stats(v, frac, label=k) for k, v in eg.items()]
        res["dsb_gt"][str(frac)] = [degree_stats(v, frac, label=k) for k, v in gt.items()]
        res["dsb_pred"][str(frac)] = [degree_stats(d["xy"], frac, label=d["ortho"])
                                      for d in pr.values()]

    print("\n" + "=" * 78)
    print("DERAJAT TETANGGA RATA-RATA, POHON BAGIAN DALAM")
    print("bagian dalam = jarak ke tepi convex hull >= r; blok = parcel (Eg9PP) "
          "/ ortomosaik (ds_B)")
    print("=" * 78)
    hdr = "%-28s %8s %8s %8s" % ("himpunan titik", *["r=%.3f" % f for f in FRACS])
    print(hdr)
    for name, key in (("Eg9PP (posisi tanam)", "eg9pp"),
                      ("ds_B kotak kebenaran-dasar", "dsb_gt"),
                      ("ds_B PREDIKSI YOLOv12", "dsb_pred")):
        if not res[key][str(FRACS[0])]:
            print("%-28s  (tidak ada data)" % name)
            continue
        cells = []
        for frac in FRACS:
            m, s = _agg(res[key][str(frac)])
            cells.append("%.3f" % m if np.isnan(s) else "%.3f" % m)
        print("%-28s %8s %8s %8s" % (name, *cells))
        for frac in FRACS:
            m, s = _agg(res[key][str(frac)])
            res.setdefault("summary", {}).setdefault(key, {})[str(frac)] = dict(
                mean=m, std=s, n_blocks=len(res[key][str(frac)]))
    print()
    for key, name in (("eg9pp", "Eg9PP"), ("dsb_gt", "ds_B GT"), ("dsb_pred", "ds_B PRED")):
        if not res[key][str(FRACS[0])]:
            continue
        for frac in FRACS:
            m, s = _agg(res[key][str(frac)])
            per = "  ".join("%s=%.3f (%d/%d)"
                            % (r["label"], r["deg_interior"], r["n_interior"], r["n_total"])
                            for r in res[key][str(frac)])
            print("  %-10s r=%.3f  %.3f +/- %.3f   | %s"
                  % (name, frac, m, s, per))

    # ---- putusan pada dataran, memakai aturan pita derau repo ----
    if res["dsb_pred"][str(FRACS[0])]:
        print("\n" + "=" * 78)
        print("PUTUSAN — aturan pita derau yang sama dengan run_experiment.py::paired")
        print("=" * 78)
        for frac in FRACS:
            e, _ = _agg(res["eg9pp"][str(frac)])
            g, gs = _agg(res["dsb_gt"][str(frac)])
            p, ps = _agg(res["dsb_pred"][str(frac)])
            band = max(ps, gs)
            d = p - e
            print("  r=%.3f  Eg9PP %.3f | GT %.3f+/-%.3f | PRED %.3f+/-%.3f | "
                  "PRED-Eg9PP = %+.3f (%.1f%%)  %s"
                  % (frac, e, g, gs, p, ps, d, 100 * abs(d) / e,
                     "DI DALAM pita derau" if abs(d) <= band else "DI LUAR pita derau"))

    # ---- kepekaan terhadap ambang keyakinan ----
    if folds and sweep:
        print("\n" + "=" * 78)
        print("KEPEKAAN TERHADAP AMBANG KEYAKINAN (dilaporkan, bukan disetel)")
        print("=" * 78)
        res["conf_sweep"] = conf_sweep(folds)
    elif os.path.isfile(OUT):
        # Lari --no-sweep TIDAK BOLEH diam-diam menghapus sapuan yang sudah ada:
        # INTERFACE_TEST.md mengutipnya. Ia dibawa serta dan ditandai basi.
        old = json.load(open(OUT)).get("conf_sweep")
        if old:
            res["conf_sweep"] = old
            res["conf_sweep_stale"] = True
            print("\ncatatan: sapuan conf dilewati; hasil sapuan LAMA dipertahankan "
                  "di berkas keluaran dan ditandai `conf_sweep_stale`.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print("\ndisimpan -> %s" % OUT)

    print("\nCATATAN WAJIB. Pada conf = 0,25 himpunan prediksi LEBIH PADAT daripada")
    print("kebenaran-dasar (presisi 0,79-0,86), bukan lebih jarang. Derajatnya karena")
    print("itu bias KE ATAS, dan penaksir jarak tanamnya tertarik KE BAWAH. Kecocokan")
    print("6,00 vs 6,00 pada r = 1,5 adalah kebetulan dua bias yang saling meniadakan,")
    print("BUKAN bukti kesesuaian yang lebih baik daripada angka kotak kebenaran-dasar.")
    print("Baca angka mana pun HANYA bersama sapuan conf di atas.")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-train", action="store_true",
                    help="jangan melatih ulang fold0; laporkan fold yang bobotnya ada")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--no-sweep", action="store_true",
                    help="lewati sapuan ambang keyakinan (lebih cepat)")
    a = ap.parse_args()
    if a.train_only:
        y12.build(extra_mode="ignore")
        train_fold0(force=a.force_train)
    else:
        main(train=not a.no_train, force_train=a.force_train, sweep=not a.no_sweep)
