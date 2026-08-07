"""Lapisan 1 - Tahap 1: deteksi tajuk sawit dengan YOLOv12 (leave-one-ortho-out).

Modul ini adalah SATU-SATUNYA sumber logika untuk notebook
`solution_layer1_yolov12.ipynb`. Notebook memanggil fungsi di sini, tidak
menyalin ulang isinya, supaya keduanya tidak mungkin berbeda.

Tiga hal yang dijaga modul ini, dan alasannya:

1. HARNESS DIKUNCI DULU. Evaluasi selalu leave-one-ortho-out. Hanya ada 3
   ortomosaik dan ubin Roboflow saling bertindih ~30x, jadi split acak bocor
   100% (lihat ../layer1_data_audit/AUDIT_REPORT.md). Tidak ada jalan untuk
   meminta split acak dari modul ini - fungsinya memang tidak disediakan.

2. AUGMENTASI ADALAH ABLASI, BUKAN SAKELAR. Setiap konfigurasi augmentasi
   dijalankan pada lipatan DAN seed yang sama dengan garis dasarnya, lalu
   dibandingkan berpasangan. Aturan putusnya sama persis dengan Lapisan 2
   (`run_experiment.py::paired`) dan `compare_models.py`: selisih yang berada
   di dalam satu simpangan baku dinyatakan TIDAK KONKLUSIF.

3. DATA TAMBAHAN TIDAK PERNAH MASUK LEWAT PINTU BELAKANG. Ubin dari sumber
   lain (mis. tangkapan layar citra satelit) diberi region-nya sendiri dan
   diperlakukan sebagai blok spasial terpisah - bukan diaduk acak ke dalam
   train. Lihat `build()` argumen `extra_mode`.

Peringatan yang melekat: n = 3 ortomosaik. Pada ukuran ini pita derau lebar
dan sebagian besar selisih akan jatuh TIDAK KONKLUSIF. Itu jawaban yang benar.
Melipatgandakan seed menambah pasangan, tetapi TIDAK menambah situs - ia
mempersempit derau optimisasi, bukan ketidakpastian antar-kebun.
"""
import collections
import glob
import json
import os
import re
import shutil

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

CLS = {"Healthy": 0, "Unhealthy": 1}
NAMES = {0: "Healthy", 1: "Unhealthy"}
SINGLE = False        # mode kelas-tunggal; diatur oleh set_mode()/build()

# Direktori kerja. set_mode() menukar seluruh blok ini sekaligus supaya dua mode
# TIDAK PERNAH berbagi direktori: hasil 2-kelas dan 1-kelas tidak sebanding, dan
# menaruhnya di folder yang sama adalah cara termudah membandingkan dua hal beda.
ROOT = os.path.join(BASE, "yolo12")
IMG = os.path.join(ROOT, "images")
LAB = os.path.join(ROOT, "labels")
RUNS = os.path.join(BASE, "yolo12_runs")
RESDIR = os.path.join(BASE, "yolo12_results")


def set_mode(single_class=False, verbose=True):
    """Pilih 2-kelas (Healthy/Unhealthy) atau 1-kelas (palm).

    MENGAPA MODE 1-KELAS ADA, dan mengapa ia bukan akal-akalan metrik.

    Pada mode 2-kelas, mAP adalah RATA-RATA atas kelas. Terukur pada lari 30
    epoch: Healthy AP50 = 0,978 dan 0,989, sedangkan Unhealthy AP50 = 0,426 dan
    0,474, sehingga mAP50 jatuh ke 0,702 dan 0,731. Angka "0,7" itu bukan
    kemampuan deteksi tajuk; ia rata-rata satu kelas yang nyaris sempurna dengan
    satu kelas langka yang bersandar pada 17-31 pohon unik per ortomosaik.

    Tugas Tahap 1 dalam pipeline ini adalah MENGHASILKAN KOORDINAT. Penilaian
    kesehatan adalah Tahap 2 (`exp_health.py`, LightGBM atas fitur tajuk), yang
    melaporkan PR-AUC dengan pita deraunya sendiri. Detektor 2-kelas mengerjakan
    ulang tugas Tahap 2 dengan buruk, lalu menghukum metrik Tahap 1 karenanya.

    Yang mode ini TIDAK lakukan: ia tidak membuat pendeteksian pohon sakit jadi
    lebih baik. Persoalan itu tidak hilang, ia pindah ke Tahap 2 tempat ia
    memang ditangani - dan angka Tahap 2 (PR-AUC 0,182 +/- 0,059) tetap harus
    dilaporkan berdampingan. Menyebut 0,98 tanpa menyebut ini adalah menyesatkan.
    """
    global SINGLE, ROOT, IMG, LAB, RUNS, RESDIR
    SINGLE = bool(single_class)
    sfx = "_1c" if SINGLE else ""
    ROOT = os.path.join(BASE, "yolo12" + sfx)
    IMG = os.path.join(ROOT, "images")
    LAB = os.path.join(ROOT, "labels")
    RUNS = os.path.join(BASE, "yolo12_runs" + sfx)
    RESDIR = os.path.join(BASE, "yolo12_results" + sfx)
    if verbose:
        print("mode: %s  ->  %s"
              % ("1 KELAS (palm)" if SINGLE else "2 kelas (Healthy/Unhealthy)",
                 os.path.basename(ROOT)))
    return SINGLE


def _names():
    return {0: "palm"} if SINGLE else dict(NAMES)


def _yaml_names():
    return ("names:\n  0: palm\n" if SINGLE
            else "names:\n  0: Healthy\n  1: Unhealthy\n")

# Ubin dari sumber luar ditaruh di sini oleh pengguna:
#   extra/images/*.jpg  +  extra/labels/*.txt   (format YOLO, kelas 0/1)
EXTRA = os.path.join(BASE, "extra")
EXTRA_REGION = "EXTRA"


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def region(fn):
    """Prefiks ortomosaik dari nama berkas Roboflow, mis. '44000_16000_...' .

    Inilah blok spasialnya. Berkas yang tidak cocok pola dianggap 'OTHER' dan
    dilaporkan terpisah, bukan diam-diam ikut ke salah satu lipatan.
    """
    m = re.match(r"(\d+)_(\d+)_", fn.split(".rf.")[0])
    return f"{m.group(1)}_{m.group(2)}" if m else "OTHER"


def _link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)          # hardlink: satu drive, tanpa menyalin 2 GB
    except Exception:
        shutil.copy(src, dst)


def _write_label(stem, lines):
    with open(os.path.join(LAB, stem + ".txt"), "w") as f:
        f.write("\n".join(lines))


def _ingest_coco(tag="B"):
    """COCO dataset Roboflow -> yolo12/images + yolo12/labels. Kembalikan peta region."""
    from grids import load_coco

    _, anns = load_coco(tag)
    by_img = collections.defaultdict(list)
    for a in anns:
        by_img[a["path"]].append(a)

    img_region, n_box = {}, 0
    for pth, alist in by_img.items():
        fn = os.path.basename(pth)
        stem = os.path.splitext(fn)[0]
        W, H = alist[0]["W"], alist[0]["H"]
        dst = os.path.join(IMG, fn)
        _link(pth, dst)
        lines = []
        for a in alist:
            if a["cat"] not in CLS:
                continue
            x, y, w, h = a["bbox"]
            k = 0 if SINGLE else CLS[a["cat"]]
            lines.append("%d %.6f %.6f %.6f %.6f"
                         % (k, (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
        _write_label(stem, lines)
        n_box += len(lines)
        img_region[dst] = region(fn)
    return img_region, n_box


def _ingest_extra():
    """Ubin dari sumber luar. Wajib sudah berlabel YOLO; tanpa label ia dilewati.

    Dikembalikan sebagai region terpisah (`EXTRA`) supaya pemanggil bisa
    memutuskan perlakuannya secara eksplisit, bukan otomatis ikut train.
    """
    ei = os.path.join(EXTRA, "images")
    el = os.path.join(EXTRA, "labels")
    if not os.path.isdir(ei):
        return {}, 0, []

    out, n_box, skipped = {}, 0, []
    for p in sorted(glob.glob(os.path.join(ei, "*.*"))):
        if os.path.splitext(p)[1].lower() not in (".jpg", ".jpeg", ".png"):
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        lp = os.path.join(el, stem + ".txt")
        if not os.path.isfile(lp):
            skipped.append(os.path.basename(p))   # tanpa label = bukan data latih
            continue
        dst = os.path.join(IMG, os.path.basename(p))
        _link(p, dst)
        lines = []
        for ln in open(lp):
            if not ln.strip():
                continue
            f = ln.split()
            c = int(float(f[0]))
            if c not in NAMES:
                raise ValueError(f"{lp}: kelas {c} di luar {{0,1}}")
            lines.append(" ".join(["0" if SINGLE else str(c)] + f[1:]))
        _write_label(stem, lines)
        n_box += len(lines)
        out[dst] = EXTRA_REGION
    return out, n_box, skipped


def build(extra_mode="holdout", single_class=None, verbose=True):
    """Siapkan yolo12/ dan berkas yaml per lipatan.

    extra_mode:
      'ignore'  - ubin extra/ tidak dipakai sama sekali.
      'holdout' - ubin extra/ menjadi LIPATAN TAMBAHAN yang hanya pernah diuji,
                  tidak pernah dilatih. Ini pemakaian yang sah untuk citra dari
                  sumber/sensor lain: ia mengukur generalisasi lintas-domain dan
                  tidak bisa mencemari angka 3-ortho.
      'train'   - ubin extra/ ditambahkan ke train SETIAP lipatan, tidak pernah
                  ke val. Menjawab "apakah data tambahan menaikkan performa pada
                  3 ortomosaik nyata". Tetap tidak melanggar block-CV karena ia
                  blok tersendiri dan tak pernah muncul di sisi evaluasi.

    Yang tidak tersedia: mengaduk extra/ secara acak ke train+val. Itu akan
    mengukur dirinya sendiri.
    """
    assert extra_mode in ("ignore", "holdout", "train"), extra_mode
    if single_class is not None:
        set_mode(single_class, verbose=verbose)
    os.makedirs(IMG, exist_ok=True)
    os.makedirs(LAB, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)

    img_region, n_box = _ingest_coco("B")
    ex, ex_box, skipped = ({}, 0, []) if extra_mode == "ignore" else _ingest_extra()

    regions = sorted({r for r in img_region.values() if r != "OTHER"})
    other = [p for p, r in img_region.items() if r == "OTHER"]

    folds = []
    for i, held in enumerate(regions):
        tr = [p for p, r in img_region.items() if r != held and r != "OTHER"]
        va = [p for p, r in img_region.items() if r == held]
        if extra_mode == "train":
            tr += list(ex)
        folds.append((f"fold{i}", held, tr, va))

    if extra_mode == "holdout" and ex:
        # Dilatih pada SELURUH tiga ortomosaik, diuji hanya pada ubin luar.
        tr = [p for p, r in img_region.items() if r != "OTHER"]
        folds.append(("foldX", EXTRA_REGION, tr, list(ex)))

    for name, held, tr, va in folds:
        open(os.path.join(ROOT, f"{name}_train.txt"), "w").write("\n".join(tr))
        open(os.path.join(ROOT, f"{name}_val.txt"), "w").write("\n".join(va))
        open(os.path.join(ROOT, f"{name}.yaml"), "w").write(
            "path: %s\ntrain: %s_train.txt\nval: %s_val.txt\n%s"
            % (ROOT, name, name, _yaml_names()))

    if verbose:
        print("citra   : %d  kotak: %d  (dataset B)" % (len(img_region), n_box))
        if extra_mode != "ignore":
            print("extra   : %d citra berlabel, %d kotak, mode=%s" % (len(ex), ex_box, extra_mode))
            if skipped:
                print("          %d citra extra DILEWATI (tak ada label): %s%s"
                      % (len(skipped), ", ".join(skipped[:3]),
                         " ..." if len(skipped) > 3 else ""))
        if other:
            print("PERHATIAN: %d citra tak cocok pola region -> tidak dipakai" % len(other))
        for name, held, tr, va in folds:
            print("  %-6s tahan %-12s train=%5d  val=%4d" % (name, held, len(tr), len(va)))
    return [f[0] for f in folds]


def class_balance(fold_names=None):
    """Hitung kotak per kelas per lipatan-val. Angka Unhealthy-lah yang menentukan
    seberapa jauh AP kelas itu boleh dibaca."""
    rows = []
    for name in (fold_names or [os.path.basename(p)[:-10]
                                for p in sorted(glob.glob(os.path.join(ROOT, "*_val.txt")))]):
        va = [ln.strip() for ln in open(os.path.join(ROOT, f"{name}_val.txt")) if ln.strip()]
        c = collections.Counter()
        for p in va:
            lp = os.path.join(LAB, os.path.splitext(os.path.basename(p))[0] + ".txt")
            for ln in open(lp):
                if ln.strip():
                    c[int(float(ln.split()[0]))] += 1
        rows.append(dict(fold=name, images=len(va),
                         healthy=c[0], unhealthy=c[1],
                         pct_unhealthy=100 * c[1] / max(1, c[0] + c[1])))
    return rows


# --------------------------------------------------------------------------
# 1b. Audit mutu label - mengapa mAP bukan metrik utama di sini
# --------------------------------------------------------------------------
def label_audit():
    """Bukti terukur bahwa kotak ds_B adalah CAP berukuran tetap, bukan kotak
    yang digambar per tajuk. Angka-angka inilah yang membatasi mAP50-95, dan
    tidak satu pun dapat diperbaiki dengan melatih lebih lama."""
    import csv as _csv

    from scipy.spatial import cKDTree

    p = os.path.join(os.path.dirname(BASE), "data_clean", "layer1_crowns.csv")
    by = collections.defaultdict(list)
    for r in _csv.DictReader(open(p, encoding="utf-8")):
        by[r["ortho"]].append(r)

    out = []
    for o in sorted(by):
        g = np.array([[float(r["gx"]), float(r["gy"])] for r in by[o]])
        d, _ = cKDTree(g).query(g, k=2)
        spacing = float(np.median(d[:, 1]))
        bw = np.array([float(r["bw"]) for r in by[o]])
        bh = np.array([float(r["bh"]) for r in by[o]])
        side = np.sqrt(bw * bh)
        sizes = collections.Counter(zip(bw.astype(int), bh.astype(int)))
        top5 = sum(n for _, n in sizes.most_common(5)) / len(bw)
        out.append(dict(ortho=o, trees=len(g), spacing_px=spacing,
                        box_med_px=float(np.median(side)),
                        box_cv=float(side.std() / side.mean()),
                        n_unique_sizes=len(sizes), top5_share=float(top5),
                        frac_box_gt_spacing=float((side > spacing).mean()),
                        frac_box_lt_half=float((side < 0.5 * spacing).mean()),
                        top_sizes=["%dx%d:%d" % (w, h, n)
                                   for (w, h), n in sizes.most_common(5)]))
    return out


def redundancy_audit():
    """Tutupan piksel unik vs jumlah piksel ubin. Ubin Roboflow diambil pada
    offset acak, jadi 'jumlah citra latih' jauh melebihi jumlah informasi."""
    T = 1024
    tiles = collections.defaultdict(list)
    for fn in os.listdir(IMG):
        m = re.match(r"(\d+)_(\d+)_(\d+)_(\d+)", fn.split(".rf.")[0])
        if m:
            tiles[f"{m.group(1)}_{m.group(2)}"].append((int(m.group(3)), int(m.group(4))))
    out = []
    for o in sorted(tiles):
        xs = [t[0] for t in tiles[o]]
        ys = [t[1] for t in tiles[o]]
        g = 8
        mask = np.zeros(((max(ys) + T) // g + 1, (max(xs) + T) // g + 1), bool)
        for x, y in tiles[o]:
            mask[y // g:(y + T) // g, x // g:(x + T) // g] = True
        union = float(mask.sum() * g * g)
        out.append(dict(ortho=o, tiles=len(tiles[o]), unique_mpx=union / 1e6,
                        summed_mpx=len(tiles[o]) * T * T / 1e6,
                        redundancy=len(tiles[o]) * T * T / union))
    return out


def leak_audit():
    """Berapa persen pohon uji pada split BAWAAN Roboflow yang juga ada di train.

    Ini pembenaran empiris untuk membuang split bawaan. Angka mendekati 100%
    menjelaskan mengapa papan skor Roboflow tampak nyaris sempurna: ia menilai
    model pada pohon yang sama dengan yang dipakai melatihnya.
    """
    from scipy.spatial import cKDTree

    from grids import load_coco

    _, anns = load_coco("B")
    rec = collections.defaultdict(list)
    for a in anns:
        m = re.match(r"(\d+)_(\d+)_(\d+)_(\d+)", os.path.basename(a["path"]).split(".rf.")[0])
        if not m:
            continue
        x, y, w, h = a["bbox"]
        rec[f"{m.group(1)}_{m.group(2)}"].append(
            (int(m.group(3)) + x + w / 2, int(m.group(4)) + y + h / 2,
             (w * h) ** 0.5, a["split"]))

    out = []
    for o in sorted(rec):
        arr = np.array([(r[0], r[1]) for r in rec[o]], float)
        side = float(np.median([r[2] for r in rec[o]]))
        spl = [r[3] for r in rec[o]]
        tree = -np.ones(len(arr), int)
        kd = cKDTree(arr)
        nxt = 0
        for i in range(len(arr)):
            if tree[i] >= 0:
                continue
            for j in kd.query_ball_point(arr[i], 0.5 * side):
                if tree[j] < 0:
                    tree[j] = nxt
            nxt += 1
        bysplit = collections.defaultdict(set)
        for t, s in zip(tree, spl):
            bysplit[s].add(t)
        tr = bysplit["train"]
        out.append(dict(ortho=o, trees=nxt,
                        valid_in_train=len(bysplit["valid"] & tr) / max(1, len(bysplit["valid"])),
                        test_in_train=len(bysplit["test"] & tr) / max(1, len(bysplit["test"]))))
    return out


# --------------------------------------------------------------------------
# 2. Lengan augmentasi
# --------------------------------------------------------------------------
# Setiap lengan HANYA berisi selisihnya terhadap default Ultralytics. Yang tidak
# disebut = default. Alasan fisiknya ditulis, karena augmentasi yang tidak punya
# alasan fisik hanyalah pencarian hiperparameter yang menyamar.
ARMS = {
    "base": dict(
        _why="Default Ultralytics apa adanya. Garis dasar; semua lengan lain diukur terhadap ini.",
    ),
    "nadir": dict(
        degrees=180.0, flipud=0.5,
        _why=("Citra nadir tidak punya arah 'atas' yang kanonik: kebun difoto dengan "
              "heading wahana yang sembarang. Rotasi penuh dan flip vertikal karena itu "
              "SAH secara fisik di sini, padahal terlarang pada citra permukaan tanah."),
    ),
    "nadir_geom": dict(
        degrees=180.0, flipud=0.5, scale=0.6, translate=0.2,
        _why=("Tambahan ragam skala dan geser: GSD bervariasi antar-terbang (8,5-8,9 cm/px "
              "terukur pada tiga ortomosaik) dan ubin Roboflow diambil pada offset acak."),
    ),
    "nadir_nohsv": dict(
        degrees=180.0, flipud=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        _why=("Uji tandingan yang penting. Sinyal kelas Unhealthy adalah WARNA - fitur "
              "teratas Tahap 2 adalah (G-R), exg_std, (G-B), yaitu greenness/klorosis. "
              "Jitter HSV berpotensi menghapus justru sinyal yang harus dipelajari. "
              "Lengan ini mematikannya sepenuhnya untuk melihat apakah itu benar."),
    ),
    "nadir_strong": dict(
        degrees=180.0, flipud=0.5, scale=0.6, translate=0.2,
        hsv_s=0.9, hsv_v=0.6, mixup=0.15, close_mosaic=10,
        _why=("Lengan sengaja agresif. Ada untuk menunjukkan batas atas: bila 'lebih banyak "
              "augmentasi' memang selalu lebih baik, lengan ini menang. Prediksi kami "
              "sebaliknya, karena mixup pada objek kecil-padat mengaburkan batas tajuk."),
    ),
}

# TIDAK dipakai, dan alasannya dicatat supaya tidak ada yang menambahkannya diam-diam:
#   copy_paste - butuh mask segmentasi. Dataset ini bounding box semua, tanpa mask.
#   erasing    - parameter tugas klasifikasi, bukan deteksi.
#   auto_augment - kebijakan untuk klasifikasi; tidak berlaku pada jalur deteksi.
NOT_USED = {
    "copy_paste": "butuh mask segmentasi; dataset ini bbox murni (lihat AUDIT_REPORT.md)",
    "erasing": "parameter klasifikasi, tidak berlaku pada tugas deteksi",
    "auto_augment": "kebijakan klasifikasi, tidak berlaku pada tugas deteksi",
}


def arm_cfg(name):
    return {k: v for k, v in ARMS[name].items() if not k.startswith("_")}


def preview_aug(arm, n=6, imgsz=640, fold="fold0"):
    """Citra latih SETELAH augmentasi lengan `arm` benar-benar diterapkan.

    Ada supaya augmentasi DILIHAT, bukan diasumsikan. Beberapa setelan yang
    terdengar benar merusak ubin nadir dengan cara yang baru kelihatan di mata:
    mosaic memotong tajuk di jahitan, mixup membuat dua tajuk saling tembus.
    """
    import os as _os

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.utils import DEFAULT_CFG

    ov = dict(mode="train", imgsz=imgsz)
    ov.update(arm_cfg(arm))
    args = get_cfg(DEFAULT_CFG, overrides=ov)
    data = dict(path=ROOT, train=f"{fold}_train.txt", val=f"{fold}_val.txt",
                names=NAMES, nc=2, channels=3)
    ds = build_yolo_dataset(args, _os.path.join(ROOT, f"{fold}_train.txt"),
                            n, data, mode="train", rect=False, stride=32)
    out = []
    for i in range(n):
        s = ds[i]
        im = s["img"].numpy().transpose(1, 2, 0)[:, :, ::-1]     # CHW BGR -> HWC RGB
        out.append((np.ascontiguousarray(im), s["bboxes"].numpy(),
                    s["cls"].numpy().ravel().astype(int)))
    return out


def scale_check(sample=250):
    """Diameter tajuk median (px) per region - uji kecocokan skala untuk ubin luar.

    Menggabungkan ubin dari sensor lain hanya masuk akal bila tajuknya berukuran
    piksel yang sebanding. Kalau angka EXTRA jauh dari angka ds_B, ubin itu harus
    di-resample dulu; kalau tidak, detektor diminta belajar dua skala sekaligus
    dari data yang jumlahnya tidak cukup untuk satu skala pun.
    """
    import random as _random

    from PIL import Image

    by = collections.defaultdict(list)
    paths = sorted(glob.glob(os.path.join(IMG, "*.*")))
    _random.Random(0).shuffle(paths)
    seen = collections.Counter()
    for p in paths:
        r = region(os.path.basename(p))
        if r == "OTHER" and os.path.isfile(os.path.join(
                EXTRA, "images", os.path.basename(p))):
            r = EXTRA_REGION
        if seen[r] >= sample:
            continue
        lp = os.path.join(LAB, os.path.splitext(os.path.basename(p))[0] + ".txt")
        if not os.path.isfile(lp):
            continue
        W, H = Image.open(p).size
        for ln in open(lp):
            if not ln.strip():
                continue
            _, _, _, w, h = ln.split()
            by[r].append(((float(w) * W) ** 2 + (float(h) * H) ** 2) ** 0.5)
        seen[r] += 1
    return [dict(region=r, tiles=seen[r], boxes=len(v),
                 diag_med_px=float(np.median(v)),
                 diag_p10=float(np.percentile(v, 10)),
                 diag_p90=float(np.percentile(v, 90)))
            for r, v in sorted(by.items())]


# --------------------------------------------------------------------------
# 3. Latih & nilai
# --------------------------------------------------------------------------
def _res_path(tag):
    return os.path.join(RESDIR, f"{tag}.json")


def train_arm(arm, model="yolo12n.pt", folds=None, seeds=(42,), epochs=30, imgsz=640,
              batch=None, workers=0, cache="ram", device=None, resume_ok=True,
              tag_suffix="", verbose=True):
    """Latih satu lengan augmentasi pada semua (lipatan x seed) dan simpan hasilnya.

    Berkas keluaran `yolo12_results/<tag>.json` memuat setelan lengkapnya; fungsi
    `paired()` menolak membandingkan dua berkas yang setelannya berbeda.

    `tag_suffix` memisahkan ruang nama. WAJIB dipakai (mis. "final") ketika lengan
    yang sama dilatih ulang dengan jumlah epoch berbeda: tanpa itu lari kedua
    MENIMPA hasil ablasi lengan tersebut, dan seluruh perbandingan ikut batal
    karena setelan antar-berkas tidak lagi seragam.

    `workers=0` adalah default DISENGAJA: di Windows dataloader YOLO menelurkan
    proses anak yang meng-import ulang modul pemanggil, dan di dalam notebook
    itu menggantung. Skrip .py boleh menaikkannya.
    """
    import torch
    from ultralytics import YOLO

    folds = folds or [os.path.splitext(os.path.basename(p))[0]
                      for p in sorted(glob.glob(os.path.join(ROOT, "fold*.yaml")))]
    dev = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
    batch = batch or (16 if imgsz <= 640 else 8)      # 8 GB VRAM tidak muat 16 @1024
    mtag = os.path.splitext(os.path.basename(model))[0]
    tag = f"{mtag}_{arm}" + (f"_{tag_suffix}" if tag_suffix else "")
    cfg = arm_cfg(arm)

    prev = {}
    if resume_ok and os.path.isfile(_res_path(tag)):
        old = json.load(open(_res_path(tag)))
        if old.get("setting") == _setting(model, epochs, imgsz, folds, seeds):
            prev = old["runs"]                        # lanjutkan, jangan ulang

    runs = dict(prev)
    for f in folds:
        for s in seeds:
            key = f"{f}|{s}"
            if key in runs:
                if verbose:
                    print("  lewati (sudah ada): %s %s" % (tag, key), flush=True)
                continue
            yaml = os.path.join(ROOT, f"{f}.yaml")
            m = YOLO(model)
            m.train(data=yaml, epochs=epochs, imgsz=imgsz, device=dev, batch=batch,
                    cache=cache, workers=workers, seed=s, deterministic=True,
                    project=RUNS, name=f"{tag}_{f}_s{s}", exist_ok=True,
                    verbose=False, plots=False, **cfg)
            # `workers` WAJIB diteruskan ke val juga, bukan hanya ke train:
            # m.val() default-nya workers=8, dan di Windows proses anaknya
            # meng-import ulang pemanggil sehingga notebook menggantung/mati.
            mt = m.val(data=yaml, device=dev, batch=batch, workers=workers,
                       verbose=False, plots=False)
            nm = _names()
            ap50 = {nm[int(c)]: float(v)
                    for c, v in zip(mt.box.ap_class_index, mt.box.ap50)}
            runs[key] = dict(map50=float(mt.box.map50), map=float(mt.box.map),
                             mp=float(mt.box.mp), mr=float(mt.box.mr), ap50=ap50)
            if verbose:
                print("  [%s %s seed%d] mAP50=%.3f mAP50-95=%.3f P=%.3f R=%.3f  %s"
                      % (tag, f, s, runs[key]["map50"], runs[key]["map"],
                         runs[key]["mp"], runs[key]["mr"],
                         " ".join("%s=%.3f" % kv for kv in sorted(ap50.items()))),
                      flush=True)
            json.dump(dict(tag=tag, arm=arm, model=model, cfg=cfg,
                           setting=_setting(model, epochs, imgsz, folds, seeds),
                           runs=runs), open(_res_path(tag), "w"), indent=2)

    out = dict(tag=tag, arm=arm, model=model, cfg=cfg,
               setting=_setting(model, epochs, imgsz, folds, seeds), runs=runs)
    json.dump(out, open(_res_path(tag), "w"), indent=2)
    return out


def _setting(model, epochs, imgsz, folds, seeds):
    return dict(epochs=epochs, imgsz=imgsz, folds=list(folds), seeds=list(seeds))


# --------------------------------------------------------------------------
# 4. Putusan berpasangan - aturan yang sama dengan Lapisan 2
# --------------------------------------------------------------------------
def load_results(only=None, exclude_suffix=("_final",)):
    """Muat berkas hasil. Lari bertanda `_final` dikecualikan secara default:
    ia dilatih dengan jumlah epoch berbeda, jadi ia bukan anggota ablasi."""
    runs = {}
    for p in sorted(glob.glob(os.path.join(RESDIR, "*.json"))):
        d = json.load(open(p))
        if only is not None:
            if d["tag"] not in only:
                continue
        elif any(d["tag"].endswith(s) for s in exclude_suffix):
            continue
        runs[d["tag"]] = d
    return runs


def paired(runs, base, key="map", exclude_folds=("foldX",)):
    """Selisih berpasangan per (lipatan, seed) terhadap `base`.

    ddof=1 disengaja: pada n kecil ddof=0 menyempitkan pita derau secara palsu.
    '<=' disengaja: selisih persis nol jatuh TIDAK KONKLUSIF, bukan NEGATIF.

    `foldX` (blok data luar) dikeluarkan secara default - ia mengukur pertanyaan
    yang berbeda (generalisasi lintas-domain) dan tidak boleh diaduk ke dalam
    rata-rata 3-ortho.
    """
    if base not in runs:
        raise KeyError(f"garis dasar '{base}' tidak ada. Tersedia: {sorted(runs)}")
    settings = {t: json.dumps(d["setting"], sort_keys=True) for t, d in runs.items()}
    if len(set(settings.values())) > 1:
        raise SystemExit("SETELAN BERBEDA, perbandingan dibatalkan:\n  "
                         + "\n  ".join("%-22s %s" % kv for kv in sorted(settings.items())))

    keys = [k for k in runs[base]["runs"] if k.split("|")[0] not in exclude_folds]
    rows = []
    for tag, d in sorted(runs.items()):
        common = [k for k in keys if k in d["runs"]]
        v = np.array([d["runs"][k][key] for k in common])
        row = dict(tag=tag, n=len(common), mean=float(v.mean()),
                   std=float(v.std(ddof=1)) if len(v) > 1 else float("nan"))
        if tag != base:
            b = np.array([runs[base]["runs"][k][key] for k in common])
            dd = v - b
            mean, std = float(dd.mean()), float(dd.std(ddof=1)) if len(dd) > 1 else float("nan")
            row.update(delta=mean, delta_std=std,
                       wins=int((dd > 0).sum()), pairs=len(dd),
                       verdict=("TIDAK KONKLUSIF" if not (abs(mean) > std)
                                else ("POSITIF" if mean > 0 else "NEGATIF")),
                       per_pair={k: float(x) for k, x in zip(common, dd)})
        rows.append(row)
    return rows


def report(key="map", base=None, only=None):
    runs = load_results(only)
    if len(runs) < 2:
        raise SystemExit("Perlu >= 2 lengan. Ditemukan: %s" % (sorted(runs) or "tidak ada"))
    base = base or next((t for t in runs if t.endswith("_base")), sorted(runs)[0])
    rows = paired(runs, base, key)
    label = dict(map="mAP50-95", map50="mAP50", mp="Precision", mr="Recall")[key]
    st = runs[base]["setting"]
    print("=" * 78)
    print("%s  |  epochs=%d imgsz=%d folds=%s seeds=%s"
          % (label, st["epochs"], st["imgsz"], st["folds"], st["seeds"]))
    print("garis dasar: %s\n" % base)
    for r in rows:
        line = "  %-24s %.4f +/- %.4f  (n=%d)" % (r["tag"], r["mean"], r["std"], r["n"])
        if "delta" in r:
            line += "   d=%+0.4f +/- %0.4f  tanda %d/%d  [%s]" % (
                r["delta"], r["delta_std"], r["wins"], r["pairs"], r["verdict"])
        print(line)
    print("\nCATATAN. Pasangan berasal dari %d ortomosaik x %d seed. Seed menambah"
          % (len(st["folds"]), len(st["seeds"])))
    print("pasangan tetapi TIDAK menambah situs: ia mempersempit derau optimisasi,")
    print("bukan ketidakpastian antar-kebun. Klaim generalisasi tetap dibatasi n=3.")
    return rows


# --------------------------------------------------------------------------
# 5. Evaluasi berbasis PUSAT TAJUK
# --------------------------------------------------------------------------
# Mengapa metrik ini ada, dan mengapa ia yang utama.
#
# Kotak kebenaran-dasar ds_B bukan kotak yang digambar per tajuk: hanya ada
# 23-30 UKURAN kotak berbeda untuk 1.379-1.849 tajuk per ortomosaik, lima ukuran
# teratas menutup 53-73% kotak, dan rasio aspeknya terkunci di 0,99. Kotak itu
# CAP berukuran tetap yang ditempelkan pada pusat tajuk. Akibatnya:
#
#   * mAP50-95 punya langit-langit yang tidak dapat dinaikkan oleh model apa pun.
#     Detektor yang menggambar batas tajuk SEBENARNYA justru dihukum, karena
#     kebenaran-dasarnya tidak mengikuti batas tajuk.
#   * 33-42% kotak lebih besar daripada jarak tanam, jadi kotak tetangga saling
#     bertindih secara bawaan dan NMS memotong recall.
#
# Yang TETAP dapat dipercaya dari label ini adalah PUSATnya, dan pusat itulah
# yang dikonsumsi Lapisan 2 - bukan kotak. Metrik di bawah karena itu mengukur
# hal yang benar-benar dipakai hilir, dan ia tidak dibatasi geometri cap.
#
# Bonus penting: evaluasinya dilakukan pada POHON UNIK setelah deteksi dari ubin
# bertindih digabungkan. mAP per-ubin merata-ratakan ~34 tampilan berkorelasi
# dari pohon yang sama; metrik ini tidak.
def _tile_offset(fn):
    m = re.match(r"(\d+)_(\d+)_(\d+)_(\d+)", fn.split(".rf.")[0])
    return (f"{m.group(1)}_{m.group(2)}", int(m.group(3)), int(m.group(4))) if m else None


def gt_trees(ortho):
    """Pohon unik + label dari layer1_crowns.csv (5.077 pohon, sudah dideduplikasi)."""
    import csv as _csv

    p = os.path.join(os.path.dirname(BASE), "data_clean", "layer1_crowns.csv")
    xy, lab = [], []
    for r in _csv.DictReader(open(p, encoding="utf-8")):
        if r["ortho"] == ortho:
            xy.append((float(r["gx"]), float(r["gy"])))
            lab.append(CLS[r["label"]])
    return np.array(xy, float), np.array(lab, int)


def predict_global(weights, fold, conf=0.25, imgsz=640, device=None, batch=32,
                   verbose=True):
    """Deteksi pada seluruh ubin val satu lipatan -> pusat tajuk global, digabungkan.

    Ubin bertindih ~34x, jadi satu pohon terdeteksi berkali-kali. Deteksi
    digabungkan pada radius 0,5 x jarak tanam, mempertahankan yang paling yakin -
    persis prosedur yang dipakai audit untuk membentuk 5.077 pohon unik.
    """
    import torch
    from scipy.spatial import cKDTree
    from ultralytics import YOLO

    dev = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
    paths = [l.strip() for l in open(os.path.join(ROOT, f"{fold}_val.txt")) if l.strip()]
    m = YOLO(weights)
    det = []
    n_unparsed = 0
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        res = m.predict(chunk, imgsz=imgsz, conf=conf, device=dev, verbose=False)
        # WAJIB zip terhadap `chunk`, JANGAN pakai r.path: ketika sumbernya berupa
        # DAFTAR path, Ultralytics 8.4 menamai ulang hasilnya 'image0.jpg',
        # 'image1.jpg', ... sehingga offset ubin tidak bisa diurai dan seluruh
        # deteksi terbuang diam-diam (gejalanya P = R = 0, bukan galat).
        for src, r in zip(chunk, res):
            off = _tile_offset(os.path.basename(src))
            if off is None:
                n_unparsed += 1
                continue
            _, tx, ty = off
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xy = b.xywh.cpu().numpy()
            for (cx, cy, _, _), c, k in zip(xy, b.conf.cpu().numpy(),
                                            b.cls.cpu().numpy().astype(int)):
                det.append((tx + cx, ty + cy, float(c), int(k)))
    if n_unparsed:
        print("  PERHATIAN: %d ubin tidak dapat diurai offset-nya, dilewati" % n_unparsed)
    if not det:
        print("  TIDAK ADA deteksi di atas conf=%.3f. Turunkan conf, atau modelnya "
              "memang belum belajar." % conf)
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, int)

    det = np.array(det, float)
    order = np.argsort(-det[:, 2])                     # paling yakin menang
    det = det[order]
    gt_xy, _ = gt_trees(_tile_offset(os.path.basename(paths[0]))[0])
    d, _ = cKDTree(gt_xy).query(gt_xy, k=2)
    r_merge = 0.5 * float(np.median(d[:, 1]))

    kd = cKDTree(det[:, :2])
    keep = np.ones(len(det), bool)
    for i in range(len(det)):
        if not keep[i]:
            continue
        for j in kd.query_ball_point(det[i, :2], r_merge):
            if j > i:
                keep[j] = False
    if verbose:
        print("  %s: %d deteksi mentah -> %d pohon setelah penggabungan (r=%.0f px)"
              % (fold, len(det), keep.sum(), r_merge))
    d = det[keep]
    return d[:, :2], d[:, 2], d[:, 3].astype(int)


def centre_match(pred_xy, gt_xy, radius):
    """Pasangan satu-ke-satu serakah menurut jarak menaik. -> (idx_pred, idx_gt)."""
    from scipy.spatial import cKDTree

    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return np.zeros((0, 2), int)
    kd = cKDTree(gt_xy)
    cand = []
    for i, p in enumerate(pred_xy):
        for j in kd.query_ball_point(p, radius):
            cand.append((float(np.hypot(*(p - gt_xy[j]))), i, j))
    cand.sort()
    used_p, used_g, out = set(), set(), []
    for _, i, j in cand:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); out.append((i, j))
    return np.array(out, int).reshape(-1, 2)


def centre_eval(weights, fold, radius_frac=0.5, conf=0.25, imgsz=640, verbose=True):
    """Presisi/recall/F1 pusat tajuk pada POHON UNIK di ortomosaik yang ditahan.

    `radius_frac` dinyatakan sebagai kelipatan JARAK TANAM, bukan piksel, supaya
    ambangnya berarti sama di ketiga ortomosaik (jarak tanam 101-106 px).
    0,5 x jarak tanam = "tidak mungkin tertukar dengan pohon tetangga".
    """
    from scipy.spatial import cKDTree

    ortho = open(os.path.join(ROOT, f"{fold}_val.txt")).readline().strip()
    ortho = _tile_offset(os.path.basename(ortho))[0]
    gt_xy, gt_lab = gt_trees(ortho)
    d, _ = cKDTree(gt_xy).query(gt_xy, k=2)
    spacing = float(np.median(d[:, 1]))

    p_xy, p_conf, p_cls = predict_global(weights, fold, conf=conf, imgsz=imgsz,
                                         verbose=verbose)
    mt = centre_match(p_xy, gt_xy, radius_frac * spacing)
    tp, fp, fn = len(mt), len(p_xy) - len(mt), len(gt_xy) - len(mt)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    rmse = (float(np.sqrt(((p_xy[mt[:, 0]] - gt_xy[mt[:, 1]]) ** 2).sum(1).mean()))
            if tp else float("nan"))

    # Kelas pada pohon yang cocok. Unhealthy sangat langka, jadi dilaporkan mentah.
    cls_ok = int((p_cls[mt[:, 0]] == gt_lab[mt[:, 1]]).sum()) if tp else 0
    unh_gt = int((gt_lab == 1).sum())
    unh_hit = int(((gt_lab[mt[:, 1]] == 1) & (p_cls[mt[:, 0]] == 1)).sum()) if tp else 0

    out = dict(fold=fold, ortho=ortho, spacing_px=spacing,
               radius_px=radius_frac * spacing, n_gt=len(gt_xy), n_pred=len(p_xy),
               tp=tp, fp=fp, fn=fn, precision=prec, recall=rec,
               f1=2 * prec * rec / max(1e-9, prec + rec), rmse_px=rmse,
               rmse_frac_spacing=rmse / spacing if tp else float("nan"),
               cls_acc_on_matched=cls_ok / max(1, tp),
               unhealthy_gt=unh_gt, unhealthy_found=unh_hit)
    if verbose:
        print("  %s (%s): P=%.3f R=%.3f F1=%.3f  RMSE=%.1f px (%.2f x jarak tanam)  "
              "TP=%d FP=%d FN=%d" % (fold, ortho, prec, rec, out["f1"], rmse,
                                     out["rmse_frac_spacing"], tp, fp, fn))
        print("     kelas pada pohon cocok: %.3f | Unhealthy ditemukan %d dari %d"
              % (out["cls_acc_on_matched"], unh_hit, unh_gt))
    return out


def radius_sweep(weights, fold, fracs=(0.25, 0.35, 0.5, 0.75, 1.0), conf=0.25, imgsz=640):
    """F1 sebagai fungsi radius pencocokan. Kurva yang datar = pusat benar-benar
    tepat; kurva yang menanjak tajam = 'benar' hanya karena ambangnya longgar."""
    p_xy, _, _ = predict_global(weights, fold, conf=conf, imgsz=imgsz, verbose=False)
    ortho = _tile_offset(os.path.basename(
        open(os.path.join(ROOT, f"{fold}_val.txt")).readline().strip()))[0]
    gt_xy, _ = gt_trees(ortho)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(gt_xy).query(gt_xy, k=2)
    spacing = float(np.median(d[:, 1]))
    rows = []
    for f in fracs:
        mt = centre_match(p_xy, gt_xy, f * spacing)
        tp = len(mt)
        pr = tp / max(1, len(p_xy)); rc = tp / max(1, len(gt_xy))
        rows.append(dict(frac=f, radius_px=f * spacing, precision=pr, recall=rc,
                         f1=2 * pr * rc / max(1e-9, pr + rc)))
    return rows


def domain_gap(tag):
    """Selisih ubin luar (foldX) terhadap rata-rata 3 ortomosaik, untuk satu lengan."""
    d = load_results([tag]).get(tag)
    if not d:
        return None
    ins = [v["map"] for k, v in d["runs"].items() if not k.startswith("foldX")]
    out = [v["map"] for k, v in d["runs"].items() if k.startswith("foldX")]
    if not out:
        return None
    return dict(tag=tag, in_domain=float(np.mean(ins)), extra=float(np.mean(out)),
                gap=float(np.mean(out) - np.mean(ins)), n_extra=len(out))
