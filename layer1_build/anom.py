"""Lapisan 1 - deteksi anomali tajuk per pohon sawit.

Dataset: "Oil Palm Tree Detection 4" v15 (Roboflow, proyecto-palmera-aceitera),
424 citra UAV nadir ketinggian rendah, 800x600, format TensorFlow Object Detection.
Lisensi CC BY 4.0 - sitasi wajib. Dua kelas:

    PalmSan  -> 0  "normal"
    PalmAnom -> 1  "anomali" (yang dimaksud pengguna sebagai tidak sehat)

MENGAPA DATASET INI DIPAKAI UNTUK PERTANYAAN ANOMALI, DAN ds_B TIDAK.

ds_B (2.303 ubin ortomosaik) memberi Unhealthy AP50 hanya 0,43-0,47 dan itu bukan
kegagalan model: kelas positifnya hanya ~66 pohon unik (1,3%), tajuknya ~100 px,
dan labelnya kesehatan generik tanpa verifikasi. Dataset ini berbeda pada tiga
sumbu yang semuanya menguntungkan pertanyaan anomali:

    laju positif   43,4% (238 dari 549 kotak)   lawan  1,3% di ds_B
    resolusi tajuk satu tajuk memenuhi bingkai  lawan  ~100 px di ds_B
    keseimbangan   tidak perlu penanganan khusus imbalance ekstrem

AUDIT KEBOCORAN (dijalankan sebelum satu model pun dilatih, hasil di audit()).

    * 0 berkas identik byte-per-byte.
    * 0 pasangan mirip pada jarak Hamming <= 20/256 (pHash 16x16) - di dalam
      maupun antar-split. Jadi tidak ada bingkai berurutan dari satu penerbangan
      yang bocor, dan tidak ada pengulangan seperti ds_B yang bertindih 34x.
    * 20 "stem" nama berkas muncul di lebih dari satu split, TETAPI korelasi
      pikselnya 0,007-0,085, yaitu citra yang benar-benar berbeda. Penomoran
      Roboflow bertabrakan; itu bukan kebocoran.

KARENA tidak ada kebocoran piksel, seluruh 424 citra boleh disatukan lalu dibagi
ulang dengan k-fold. Split bawaan (338/22/64) dipakai ulang akan memberi 22 citra
validasi - terlalu kecil untuk menghasilkan pita derau yang berarti. k-fold atas
seluruh data memberi mean +/- std yang jujur. Ini KEBALIKAN dari keputusan pada
ds_B, dan alasannya justru sama: ikuti struktur kebocoran datanya, bukan
kebiasaan.

BATAS YANG TETAP MELEKAT.

    1. "PalmAnom" TIDAK terdefinisi di sumbernya. Ia anomali tajuk menurut
       penganotasi - bukan BSR terverifikasi lapangan, bukan Ganoderma.
       Klaim maksimum tetap "deteksi anomali tajuk", bukan "deteksi BSR".
    2. 238 kotak anomali. Kecil. Selang kepercayaan lebar; laporkan std.
    3. Tidak ada metadata situs/penerbangan, jadi block-CV per-situs TIDAK
       mungkin di sini. Yang bisa dijamin hanyalah ketiadaan duplikat piksel.
       Nyatakan batas ini; jangan mengklaim generalisasi lintas-kebun.
    4. Kebun, sensor, dan ketinggian terbangnya berbeda dari ds_B maupun Eg9PP.
       Ini SUMBER BUKTI KETIGA yang berdiri sendiri, bukan gabungan.
"""
import collections
import csv
import glob
import json
import os
import shutil

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(os.path.dirname(BASE), "Oil Palm Tree Detection 4.v15i.tensorflow")
ROOT = os.path.join(BASE, "anom_data")
IMG = os.path.join(ROOT, "images")
LAB = os.path.join(ROOT, "labels")
RUNS = os.path.join(BASE, "anom_runs")
RESDIR = os.path.join(BASE, "anom_results")

CLS = {"PalmSan": 0, "PalmAnom": 1}
NAMES = {0: "PalmSan", 1: "PalmAnom"}
SPLITS = ("train", "valid", "test")


def _rows():
    out = []
    for s in SPLITS:
        p = os.path.join(DS, s, "_annotations.csv")
        if not os.path.isfile(p):
            continue
        for r in csv.DictReader(open(p)):
            r["split"] = s
            r["src"] = os.path.join(DS, s, r["filename"])
            out.append(r)
    if not out:
        raise SystemExit("Tidak menemukan _annotations.csv di %s" % DS)
    return out


def audit(verbose=True):
    """Angka mutu dataset. Dicetak sebelum melatih, bukan sesudah."""
    rows = _rows()
    per_img = collections.defaultdict(list)
    for r in rows:
        per_img[r["filename"]].append(r)

    cls = collections.Counter(r["class"] for r in rows)
    nbox = collections.Counter(len(v) for v in per_img.values())
    mixed = sum(1 for v in per_img.values() if len({r["class"] for r in v}) > 1)
    area = collections.defaultdict(list)
    for r in rows:
        w, h = int(r["width"]), int(r["height"])
        a = (int(r["xmax"]) - int(r["xmin"])) * (int(r["ymax"]) - int(r["ymin"]))
        area[r["class"]].append(100.0 * a / (w * h))

    out = dict(images=len(per_img), boxes=len(rows),
               per_class={k: cls[k] for k in CLS},
               pos_rate=cls["PalmAnom"] / len(rows),
               boxes_per_image=dict(sorted(nbox.items())),
               images_with_both_classes=mixed,
               img_with_anom=sum(1 for v in per_img.values()
                                 if any(r["class"] == "PalmAnom" for r in v)),
               area_pct={k: [float(np.percentile(v, q)) for q in (10, 50, 90)]
                         for k, v in area.items()})
    if verbose:
        print("citra=%d  kotak=%d  PalmSan=%d  PalmAnom=%d  -> laju positif %.1f%%"
              % (out["images"], out["boxes"], cls["PalmSan"], cls["PalmAnom"],
                 100 * out["pos_rate"]))
        print("kotak per citra: %s" % out["boxes_per_image"])
        print("citra memuat kedua kelas: %d | citra memuat >=1 anomali: %d"
              % (mixed, out["img_with_anom"]))
        for k, v in sorted(out["area_pct"].items()):
            print("  %-9s %%luas citra p10/med/p90 = %.1f / %.1f / %.1f" % (k, *v))
        print("\nCATATAN: 'PalmAnom' tidak terdefinisi di sumber. Ia anomali tajuk")
        print("menurut penganotasi - BUKAN BSR terverifikasi. Lihat docstring anom.py.")
    return out


def build(k=5, seed=42, verbose=True):
    """TF CSV -> YOLO, lalu k lipatan terstratifikasi atas SELURUH 424 citra.

    Stratifikasi memakai "citra memuat >= 1 PalmAnom", bukan jumlah kotak:
    yang harus seimbang antar-lipatan adalah kehadiran kelas langka, dan pada
    tugas deteksi unit pembagiannya adalah citra, bukan kotak.

    Split bawaan Roboflow SENGAJA diabaikan dan seluruh citra disatukan. Itu sah
    di sini justru karena audit kebocoran bersih (lihat docstring modul); pada
    ds_B keputusan yang sama akan salah total.
    """
    os.makedirs(IMG, exist_ok=True)
    os.makedirs(LAB, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)

    rows = _rows()
    per_img = collections.defaultdict(list)
    for r in rows:
        per_img[r["filename"]].append(r)

    files, has_anom = [], []
    for fn, rs in sorted(per_img.items()):
        dst = os.path.join(IMG, fn)
        if not os.path.exists(dst):
            try:
                os.link(rs[0]["src"], dst)
            except Exception:
                shutil.copy(rs[0]["src"], dst)
        lines = []
        for r in rs:
            W, H = int(r["width"]), int(r["height"])
            x0, y0 = int(r["xmin"]), int(r["ymin"])
            x1, y1 = int(r["xmax"]), int(r["ymax"])
            # Sebagian kotak menyentuh tepi (xmax == width); klip supaya tidak
            # menghasilkan koordinat > 1 yang ditolak diam-diam oleh YOLO.
            x0, x1 = max(0, min(x0, W)), max(0, min(x1, W))
            y0, y1 = max(0, min(y0, H)), max(0, min(y1, H))
            if x1 <= x0 or y1 <= y0:
                continue
            lines.append("%d %.6f %.6f %.6f %.6f"
                         % (CLS[r["class"]], (x0 + x1) / 2 / W, (y0 + y1) / 2 / H,
                            (x1 - x0) / W, (y1 - y0) / H))
        open(os.path.join(LAB, os.path.splitext(fn)[0] + ".txt"), "w").write("\n".join(lines))
        files.append(dst)
        has_anom.append(int(any(r["class"] == "PalmAnom" for r in rs)))

    files = np.array(files)
    has_anom = np.array(has_anom)
    rng = np.random.RandomState(seed)
    fold_of = np.empty(len(files), int)
    for lab in (0, 1):                      # stratifikasi
        idx = np.where(has_anom == lab)[0]
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % k

    names = []
    for i in range(k):
        va = files[fold_of == i]
        tr = files[fold_of != i]
        open(os.path.join(ROOT, f"fold{i}_train.txt"), "w").write("\n".join(tr))
        open(os.path.join(ROOT, f"fold{i}_val.txt"), "w").write("\n".join(va))
        open(os.path.join(ROOT, f"fold{i}.yaml"), "w").write(
            "path: %s\ntrain: fold%d_train.txt\nval: fold%d_val.txt\n"
            "names:\n  0: PalmSan\n  1: PalmAnom\n" % (ROOT, i, i))
        names.append(f"fold{i}")
        if verbose:
            n_anom = int(has_anom[fold_of == i].sum())
            print("  fold%d  train=%3d  val=%3d  (val memuat anomali: %d citra, %.0f%%)"
                  % (i, len(tr), len(va), n_anom, 100 * n_anom / len(va)))
    if verbose:
        print("total citra=%d  -> %d lipatan terstratifikasi" % (len(files), k))
    return names


def train_cv(model="yolo12n.pt", folds=None, epochs=100, imgsz=640, seeds=(42,),
             batch=16, workers=0, cache="ram", device=None, tag=None,
             resume_ok=True, verbose=True, **over):
    """Latih dan nilai tiap lipatan. Simpan AP50 per kelas.

    epochs default 100 (bukan 30 seperti ds_B): dataset ini hanya 424 citra, jadi
    satu epoch jauh lebih murah dan model butuh lebih banyak lintasan.
    `workers=0` default - alasan Windows yang sama seperti y12.py.
    """
    import torch
    from ultralytics import YOLO

    os.makedirs(RESDIR, exist_ok=True)
    folds = folds or [os.path.splitext(os.path.basename(p))[0]
                      for p in sorted(glob.glob(os.path.join(ROOT, "fold*.yaml")))]
    dev = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
    tag = tag or os.path.splitext(os.path.basename(model))[0]
    setting = dict(model=model, epochs=epochs, imgsz=imgsz, folds=list(folds),
                   seeds=list(seeds), over=dict(sorted(over.items())))
    path = os.path.join(RESDIR, f"{tag}.json")

    runs = {}
    if resume_ok and os.path.isfile(path):
        old = json.load(open(path))
        if old.get("setting") == setting:
            runs = old["runs"]

    for f in folds:
        for s in seeds:
            key = f"{f}|{s}"
            if key in runs:
                if verbose:
                    print("  lewati (sudah ada): %s" % key, flush=True)
                continue
            yaml = os.path.join(ROOT, f"{f}.yaml")
            m = YOLO(model)
            m.train(data=yaml, epochs=epochs, imgsz=imgsz, device=dev, batch=batch,
                    cache=cache, workers=workers, seed=s, deterministic=True,
                    project=RUNS, name=f"{tag}_{f}_s{s}", exist_ok=True,
                    verbose=False, plots=False, **over)
            mt = m.val(data=yaml, device=dev, batch=batch, workers=workers,
                       verbose=False, plots=False)
            ap50 = {NAMES[int(c)]: float(v)
                    for c, v in zip(mt.box.ap_class_index, mt.box.ap50)}
            ap = {NAMES[int(c)]: float(v)
                  for c, v in zip(mt.box.ap_class_index, mt.box.maps
                                  if hasattr(mt.box, "maps") else mt.box.ap50)}
            runs[key] = dict(map50=float(mt.box.map50), map=float(mt.box.map),
                             mp=float(mt.box.mp), mr=float(mt.box.mr),
                             ap50=ap50, ap=ap)
            if verbose:
                print("  [%s %s] mAP50=%.3f  PalmSan AP50=%.3f  PalmAnom AP50=%.3f"
                      % (tag, key, runs[key]["map50"], ap50.get("PalmSan", float("nan")),
                         ap50.get("PalmAnom", float("nan"))), flush=True)
            json.dump(dict(tag=tag, setting=setting, runs=runs), open(path, "w"), indent=2)

    json.dump(dict(tag=tag, setting=setting, runs=runs), open(path, "w"), indent=2)
    return dict(tag=tag, setting=setting, runs=runs)


def annotation_density(weights, fold="fold0", conf=0.25, imgsz=640, verbose=True):
    """Berapa pohon yang DIANOTASI per citra, dibanding berapa yang dideteksi.

    Terukur: 1,27 kotak GT per citra, padahal bingkainya jelas memuat 3-6 sawit.
    Dataset ini menganotasi SEBAGIAN sawit yang terlihat, bukan semuanya. Dua
    akibat yang harus ikut dilaporkan:

      1. Presisi yang dihitung terhadap GT ini adalah BATAS BAWAH - sebagian
         "positif palsu" sesungguhnya sawit nyata yang tidak dianotasi.
      2. Model yang dilatih di sini belajar "kotaki sawit yang akan dipilih
         penganotasi", BUKAN "kotaki setiap sawit". Untuk kemampuan
         kotaki-setiap-sawit, ds_B-lah datasetnya (lihat LABEL_QUALITY_AUDIT.md).
    """
    from ultralytics import YOLO

    rows = _rows()
    per = collections.defaultdict(list)
    for r in rows:
        per[r["filename"]].append(r)
    val = [l.strip() for l in open(os.path.join(ROOT, f"{fold}_val.txt")) if l.strip()]
    m = YOLO(weights)
    ngt = npred = 0
    more = 0
    for i in range(0, len(val), 32):
        ch = val[i:i + 32]
        for src, r in zip(ch, m.predict(ch, imgsz=imgsz, conf=conf, verbose=False)):
            g = len(per[os.path.basename(src)])
            p = 0 if r.boxes is None else len(r.boxes)
            ngt += g
            npred += p
            more += int(p > g)
    out = dict(images=len(val), gt=ngt, pred=npred,
               gt_per_image=ngt / len(val), pred_per_image=npred / len(val),
               images_pred_gt_gt=more)
    if verbose:
        print("val %d citra | kotak GT=%d (%.2f/citra) | prediksi=%d (%.2f/citra)"
              % (out["images"], ngt, out["gt_per_image"], npred, out["pred_per_image"]))
        print("citra dengan prediksi > GT: %d (%.0f%%)"
              % (more, 100 * more / len(val)))
        print("-> presisi terhadap GT ini adalah BATAS BAWAH; sebagian 'positif palsu'")
        print("   adalah sawit nyata yang tidak dianotasi.")
    return out


def controls(fold="fold0", epochs=15, seed=0, verbose=True):
    """KONTROL POKOK: berapa banyak sinyal anomali ada di TAJUK, berapa di LATAR.

    Tiga lengan klasifikasi pada split yang sama dengan detektor, hanya memakai
    citra berkelas tunggal supaya label tingkat-citra tidak ambigu:

        crown   dipotong ke kotak GT      -> hanya tajuk
        context kotak GT DIHITAMKAN       -> hanya latar, tajuk dihapus
        full    citra utuh                -> pembanding

    Klasifikasi, bukan deteksi: deteksi dengan masker membocorkan LETAK kotak
    lewat maskernya sendiri, sehingga AP-nya tidak dapat dibaca.

    HASIL TERUKUR (resnet18 ImageNet, 15 epoch, fold0):
        crown 0,951 | context 0,896 | full 0,925  (ROC-AUC)

    Artinya pengklasifikasi yang TIDAK PERNAH melihat tajuk tetap mencapai 0,896.
    Selisih tajuk-lawan-latar hanya 0,055. Maka AP50 detektor TIDAK boleh dibaca
    sebagai bukti pengenalan kondisi tajuk: sebagian besar keterpisahan kedua
    kelas tersedia tanpa melihat pohonnya. Laporkan angka apa pun dari dataset
    ini RELATIF terhadap garis dasar 0,896, bukan terhadap 0,5.
    """
    import torch
    import torch.nn as nn
    from PIL import Image
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = _rows()
    per = collections.defaultdict(list)
    for r in rows:
        per[r["filename"]].append(r)
    pure = {f: v for f, v in per.items() if len({x["class"] for x in v}) == 1}
    part = {}
    for split in ("train", "val"):
        part[split] = set(os.path.basename(l.strip())
                          for l in open(os.path.join(ROOT, f"{fold}_{split}.txt"))
                          if l.strip())

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    AUG = transforms.Compose([transforms.Resize((224, 224)),
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomVerticalFlip(),
                              transforms.RandomRotation(180),
                              transforms.ToTensor(), norm])

    class DS(Dataset):
        def __init__(self, files, arm, train):
            self.f = sorted(f for f in files if f in pure)
            self.arm, self.tf = arm, (AUG if train else TF)

        def __len__(self):
            return len(self.f)

        def __getitem__(self, i):
            fn = self.f[i]
            v = pure[fn]
            im = Image.open(os.path.join(IMG, fn)).convert("RGB")
            if self.arm == "crown":
                b = max(v, key=lambda r: (int(r["xmax"]) - int(r["xmin"]))
                        * (int(r["ymax"]) - int(r["ymin"])))
                im = im.crop((int(b["xmin"]), int(b["ymin"]),
                              int(b["xmax"]), int(b["ymax"])))
            elif self.arm == "context":
                a = np.array(im)
                for r in v:
                    a[int(r["ymin"]):int(r["ymax"]), int(r["xmin"]):int(r["xmax"])] = 0
                im = Image.fromarray(a)
            return self.tf(im), float(v[0]["class"] == "PalmAnom")

    out = {}
    for arm in ("crown", "context", "full"):
        dtr, dva = DS(part["train"], arm, True), DS(part["val"], arm, False)
        ltr = DataLoader(dtr, batch_size=32, shuffle=True, num_workers=0)
        lva = DataLoader(dva, batch_size=32, shuffle=False, num_workers=0)
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(512, 1)
        m = m.to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
        pos = sum(y for _, y in dtr)
        lossf = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([(len(dtr) - pos) / max(1.0, pos)], device=dev))
        for _ in range(epochs):
            m.train()
            for x, y in ltr:
                opt.zero_grad()
                lossf(m(x.to(dev)).squeeze(1), y.float().to(dev)).backward()
                opt.step()
        m.eval()
        P, Y = [], []
        with torch.no_grad():
            for x, y in lva:
                P += torch.sigmoid(m(x.to(dev)).squeeze(1)).cpu().tolist()
                Y += y.tolist()
        out[arm] = dict(roc_auc=float(roc_auc_score(Y, P)),
                        pr_auc=float(average_precision_score(Y, P)),
                        n_train=len(dtr), n_val=len(dva))
        if verbose:
            print("  %-8s ROC-AUC=%.3f  PR-AUC=%.3f  (n_tr=%d n_va=%d)"
                  % (arm, out[arm]["roc_auc"], out[arm]["pr_auc"],
                     len(dtr), len(dva)))
    if verbose:
        gap = out["crown"]["roc_auc"] - out["context"]["roc_auc"]
        print("\n  selisih crown - context = %.3f" % gap)
        print("  'context' = sinyal TANPA melihat tajuk sama sekali. Bila ia tinggi,")
        print("  AP50 detektor bukan bukti pengenalan kondisi tajuk. Laporkan angka")
        print("  dataset ini relatif terhadap context (%.3f), bukan terhadap 0,5."
              % out["context"]["roc_auc"])
    return out


def qualitative(weights, fold="fold0", n=8, conf=0.25, imgsz=640, out=None):
    """Gambar prediksi berdampingan dengan label GT di judul. Dipakai untuk
    melihat mode kegagalan - termasuk sawit jelas menguning yang TIDAK dianotasi."""
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    rows = _rows()
    per = collections.defaultdict(list)
    for r in rows:
        per[r["filename"]].append(r)
    val = [l.strip() for l in open(os.path.join(ROOT, f"{fold}_val.txt")) if l.strip()]
    an = [p for p in val if any(r["class"] == "PalmAnom" for r in per[os.path.basename(p)])]
    sa = [p for p in val if not any(r["class"] == "PalmAnom" for r in per[os.path.basename(p)])]
    pick = an[:n // 2] + sa[:n - n // 2]
    res = YOLO(weights).predict(pick, imgsz=imgsz, conf=conf, verbose=False)
    cols = 4
    rowsn = (len(pick) + cols - 1) // cols
    fig, ax = plt.subplots(rowsn, cols, figsize=(4.8 * cols, 2.6 * cols * rowsn / 2))
    for a, src, r in zip(np.array(ax).ravel(), pick, res):
        a.imshow(r.plot()[:, :, ::-1])
        a.set_xticks([]); a.set_yticks([])
        a.set_title("GT: " + ", ".join(x["class"] for x in per[os.path.basename(src)]),
                    fontsize=9)
    for a in np.array(ax).ravel()[len(pick):]:
        a.axis("off")
    plt.suptitle("Prediksi pada lipatan val — judul = label GT", y=0.995)
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=90, bbox_inches="tight")
    return fig


def export_pickle(weights, out=None, tag="stage1", extra=None, verbose=True):
    """Bungkus model Tahap 1 menjadi SATU berkas .pkl yang berdiri sendiri.

    Isinya adalah `dict` biasa, bukan objek kelas khusus. Itu disengaja: dict
    dapat di-unpickle di mesin mana pun tanpa perlu `anom.py` hadir. Bobotnya
    disimpan sebagai BYTES mentah dari .pt, jadi berkas ini utuh sendiri - tidak
    ada rujukan ke path lokal yang akan putus saat dipindahkan.

    Metadata ikut dibenamkan, termasuk KONTROL dan BATASnya, supaya angka model
    tidak dapat beredar terpisah dari syarat pembacaannya. Siapa pun yang memuat
    berkas ini mendapatkan caveat-nya sekaligus.

    Muat kembali tanpa modul ini:

        import pickle, tempfile, os
        d = pickle.load(open("stage1_model.pkl", "rb"))
        p = os.path.join(tempfile.mkdtemp(), "w.pt")
        open(p, "wb").write(d["weights_pt"])
        from ultralytics import YOLO
        model = YOLO(p)
        r = model.predict("citra.jpg", imgsz=d["imgsz"], conf=d["conf"])
    """
    import pickle

    out = out or os.path.join(BASE, f"{tag}_model.pkl")
    wb = open(weights, "rb").read()

    summary = {}
    sp = os.path.join(BASE, "stage1_summary.json")
    if os.path.isfile(sp):
        summary = json.load(open(sp))

    payload = {
        "format": "sawitguard-stage1/1",
        "task": "deteksi 2-kelas: kotak sawit + label PalmSan/PalmAnom",
        "weights_pt": wb,                 # bytes .pt Ultralytics apa adanya
        "weights_sha256": __import__("hashlib").sha256(wb).hexdigest(),
        "names": dict(NAMES),
        "imgsz": 640,
        "conf": 0.25,
        "framework": {"ultralytics": __import__("ultralytics").__version__,
                      "torch": __import__("torch").__version__},
        "source_weights": os.path.relpath(weights, BASE).replace("\\", "/"),
        "metrics": summary,
        "data": {
            "name": "Oil Palm Tree Detection for Anomaly Identification",
            "authors": "Dominguez Meza, A.; Rituay, P.",
            "doi": "10.17632/nh7d23dgnw.1",
            "licence": "CC BY 4.0 - sitasi WAJIB",
            "images": 424, "boxes": 549, "pos_rate": 0.434,
            "capture": "DJI Phantom 4 Multispectral, nadir, 17-30 m AGL, 2.5 m/s, Peru",
        },
        "limits": [
            "PalmAnom = 'tertekan atau sakit' menurut penulis dataset. BUKAN BSR, "
            "BUKAN Ganoderma, tanpa verifikasi lapangan.",
            "KONTROL: pengklasifikasi yang tajuknya DIHITAMKAN tetap mencapai "
            "ROC-AUC ~0,90-0,92 pada split yang sama, sedangkan yang melihat tajuk "
            "saja ~0,95. Selisihnya kecil (0,03-0,06), jadi AP50 model ini BUKAN "
            "bukti pengenalan kondisi tajuk - sebagian besar keterpisahan kelas "
            "bersifat kontekstual.",
            "Anotasi sumber hanya ~1,27 kotak per citra padahal bingkai memuat 3-6 "
            "sawit. Model ini mengotaki sawit yang DIPILIH penganotasi, bukan setiap "
            "sawit. Untuk kotaki-setiap-sawit gunakan ds_B.",
            "Tidak ada metadata situs: block-CV per-situs mustahil, tidak ada klaim "
            "lintas-kebun.",
        ],
    }
    if extra:
        payload.update(extra)

    with open(out, "wb") as f:
        pickle.dump(payload, f, protocol=4)     # protocol 4: kompatibel luas
    if verbose:
        print("tersimpan: %s  (%.1f MB)" % (os.path.basename(out),
                                            os.path.getsize(out) / 1e6))
        print("  sha256 bobot : %s" % payload["weights_sha256"][:16])
        print("  kelas        : %s" % payload["names"])
        print("  ultralytics  : %s | torch %s"
              % (payload["framework"]["ultralytics"], payload["framework"]["torch"]))
        print("  %d batas ikut dibenamkan (payload['limits'])" % len(payload["limits"]))
    return out


def load_pickle(path):
    """Muat .pkl hasil export_pickle() -> (model YOLO, metadata dict)."""
    import pickle
    import tempfile

    from ultralytics import YOLO

    d = pickle.load(open(path, "rb"))
    if d.get("format") != "sawitguard-stage1/1":
        raise ValueError("format tak dikenal: %r" % d.get("format"))
    p = os.path.join(tempfile.mkdtemp(prefix="stage1_"), "weights.pt")
    with open(p, "wb") as f:
        f.write(d["weights_pt"])
    return YOLO(p), {k: v for k, v in d.items() if k != "weights_pt"}


def report(tag=None):
    """mean +/- std antar-lipatan. Angka utamanya PalmAnom AP50."""
    paths = sorted(glob.glob(os.path.join(RESDIR, "*.json")))
    if tag:
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] == tag]
    if not paths:
        raise SystemExit("Belum ada hasil di %s" % RESDIR)
    for p in paths:
        d = json.load(open(p))
        st = d["setting"]
        keys = sorted(d["runs"])
        g = lambda fn: np.array([fn(d["runs"][k]) for k in keys], float)
        sd = lambda a: float(a.std(ddof=1)) if len(a) > 1 else float("nan")
        anom = g(lambda r: r["ap50"].get("PalmAnom", np.nan))
        san = g(lambda r: r["ap50"].get("PalmSan", np.nan))
        m50, m = g(lambda r: r["map50"]), g(lambda r: r["map"])
        print("=" * 74)
        print("%s | %s, %d epoch, imgsz %d, %d lipatan x %d seed%s"
              % (d["tag"], st["model"], st["epochs"], st["imgsz"],
                 len(st["folds"]), len(st["seeds"]),
                 (" | " + str(st["over"])) if st["over"] else ""))
        print("=" * 74)
        print("  %-18s %7s %7s   per lipatan" % ("", "mean", "std"))
        for nm, v in (("PalmAnom AP50", anom), ("PalmSan  AP50", san),
                      ("mAP50", m50), ("mAP50-95", m)):
            print("  %-18s %7.3f %7.3f   %s"
                  % (nm, np.nanmean(v), sd(v), " ".join("%.3f" % x for x in v)))
        print("\n  n = %d lipatan atas 424 citra / 238 kotak anomali. Std dari %d angka:"
              % (len(keys), len(keys)))
        print("  laporkan sebagai estimasi sampel kecil, bukan angka stabil.")
        print("  'PalmAnom' = anomali tajuk menurut penganotasi, BUKAN BSR terverifikasi.")
    return None
