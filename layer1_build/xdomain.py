"""Uji lintas-domain: latih pengklasifikasi tajuk di PERU, uji di ds_B (Medan).

    python xdomain.py

PERTANYAAN. Kontrol `anom.controls()` menunjukkan pengklasifikasi yang tajuknya
DIHITAMKAN tetap mencapai ROC-AUC ~0,90-0,92 di Peru - artinya sebagian besar
keterpisahan kelas di sana bersifat kontekstual (tanah, gulma, pencahayaan).
Konteks Peru TIDAK MUNGKIN ikut berpindah ke kebun di Medan. Maka:

    berpindah baik  -> model memang membaca TAJUK; confound konteks kurang parah
                       daripada yang disiratkan kontrol.
    berpindah buruk -> model belajar konteks khas Peru, persis seperti dugaan
                       kontrol.

Kedua hasil informatif. Yang kedua lebih mungkin, dan tetap wajib dilaporkan.

RANCANGAN, dan mengapa bukan transfer detektor.

Detektor TIDAK dipindahkan, hanya pengklasifikasi tajuk. Sebabnya kerapatan
anotasi: Peru menganotasi ~1,27 dari 3-6 sawit yang terlihat, sedangkan ds_B
menganotasi SETIAP pohon. Detektor Peru yang dijalankan di ds_B akan kurang
mendeteksi karena KEBIJAKAN ANOTASI, bukan karena pergeseran domain, dan
penurunan recall-nya akan salah dibaca sebagai "gagal generalisasi".

Karena itu: potong ke kotak KEBENARAN-DASAR di kedua sisi, lalu tanyakan satu
hal saja - "anomali atau tidak". Deteksi dan kerapatan anotasi hilang dari
persamaan; yang tersisa hanya penampakan tajuk.

Pemotongan dibuat KETAT pada kotak (tanpa padding) supaya konteks seminimal
mungkin ikut terbawa. Skala otomatis ternormalisasi karena kedua sisi diubah
ukuran ke 224x224 - tetapi RESOLUSI EFEKTIF-nya tidak sama (tajuk Peru ~465 px
diperkecil, tajuk ds_B ~100 px diperbesar), jadi lengan `peru_val_degraded`
menurunkan resolusi Peru ke tingkat ds_B untuk memisahkan efek resolusi dari
efek domain.

BATAS YANG TIDAK DAPAT DIHILANGKAN. Label kedua dataset didefinisikan penganotasi
berbeda di benua berbeda: `PalmAnom` = "tertekan atau sakit" (Peru), `Unhealthy`
= kesehatan tajuk generik tanpa verifikasi (ds_B). Uji ini karena itu mengukur
transfer penampakan DAN kesepakatan definisi label sekaligus; hasil negatif tidak
dapat memisahkan keduanya. Nyatakan, jangan sembunyikan.
"""
import collections
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import anom

BASE = os.path.dirname(os.path.abspath(__file__))
CROWNS = os.path.join(os.path.dirname(BASE), "data_clean", "layer1_crowns.csv")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SZ = 224
DSB_PX = 100          # sisi tajuk median ds_B, untuk lengan degradasi resolusi

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF = transforms.Compose([transforms.Resize((SZ, SZ)), transforms.ToTensor(), NORM])
AUG = transforms.Compose([transforms.Resize((SZ, SZ)),
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomVerticalFlip(),
                          transforms.RandomRotation(180),
                          transforms.ColorJitter(0.15, 0.15, 0.10, 0.02),
                          transforms.ToTensor(), NORM])


def degrade(im):
    """Turunkan resolusi ke tingkat ds_B lalu kembalikan ke SZ."""
    return im.resize((DSB_PX, DSB_PX), Image.BILINEAR).resize((SZ, SZ), Image.BILINEAR)


# ---------------------------------------------------------------- Peru
class PeruCrops(Dataset):
    def __init__(self, files, train=False, degraded=False):
        rows = anom._rows()
        per = collections.defaultdict(list)
        for r in rows:
            per[r["filename"]].append(r)
        self.items = [(f, r) for f in sorted(files) if f in per for r in per[f]]
        self.train, self.degraded = train, degraded

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, r = self.items[i]
        im = Image.open(os.path.join(anom.IMG, f)).convert("RGB")
        im = im.crop((int(r["xmin"]), int(r["ymin"]), int(r["xmax"]), int(r["ymax"])))
        if self.degraded:
            im = degrade(im)
        return (AUG if self.train else TF)(im), float(r["class"] == "PalmAnom")


# ---------------------------------------------------------------- ds_B
def dsb_index():
    """(tile_path -> [(bx,by,bw,bh,label)]) untuk 5.077 pohon unik ds_B."""
    root = os.path.dirname(BASE)
    by_tile = collections.defaultdict(list)
    n = collections.Counter()
    for r in csv.DictReader(open(CROWNS, encoding="utf-8")):
        p = os.path.join(root, r["tile_path"].replace("\\", os.sep))
        by_tile[p].append((int(r["bx"]), int(r["by"]), int(r["bw"]), int(r["bh"]),
                           1 if r["label"] == "Unhealthy" else 0))
        n[r["label"]] += 1
    return by_tile, n


class DsbCrops(Dataset):
    """Dibaca per-ubin agar tiap berkas hanya dibuka sekali."""

    def __init__(self):
        by_tile, self.counts = dsb_index()
        self.flat = [(p, b) for p, bs in sorted(by_tile.items()) for b in bs]

    def __len__(self):
        return len(self.flat)

    def __getitem__(self, i):
        p, (bx, by, bw, bh, y) = self.flat[i]
        im = Image.open(p).convert("RGB").crop((bx, by, bx + bw, by + bh))
        return TF(im), float(y)


# ---------------------------------------------------------------- model
def train_peru(epochs=20, seed=0, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = set(os.path.basename(l.strip())
             for l in open(os.path.join(anom.ROOT, "fold0_train.txt")) if l.strip())
    va = set(os.path.basename(l.strip())
             for l in open(os.path.join(anom.ROOT, "fold0_val.txt")) if l.strip())
    dtr, dva = PeruCrops(tr, train=True), PeruCrops(va)
    dvd = PeruCrops(va, degraded=True)
    if verbose:
        print("Peru crop: latih=%d kotak | val=%d kotak" % (len(dtr), len(dva)))

    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(512, 1)
    m = m.to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    pos = sum(1 for _, r in dtr.items if r["class"] == "PalmAnom")
    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(len(dtr) - pos) / max(1, pos)], device=DEV))
    ltr = DataLoader(dtr, batch_size=32, shuffle=True, num_workers=0)
    for _ in range(epochs):
        m.train()
        for x, y in ltr:
            opt.zero_grad()
            lossf(m(x.to(DEV)).squeeze(1), y.float().to(DEV)).backward()
            opt.step()
    return m, dva, dvd


@torch.no_grad()
def score(m, dset, bs=64, workers=0, note=""):
    m.eval()
    P, Y = [], []
    dl = DataLoader(dset, batch_size=bs, shuffle=False, num_workers=workers)
    for i, (x, y) in enumerate(dl):
        P += torch.sigmoid(m(x.to(DEV)).squeeze(1)).cpu().tolist()
        Y += y.tolist()
        if note and i % 20 == 0:
            print("   %s %d/%d" % (note, i * bs, len(dset)), end="\r", flush=True)
    Y = np.array(Y)
    return dict(n=len(Y), pos=int(Y.sum()), base=float(Y.mean()),
                roc_auc=float(roc_auc_score(Y, P)),
                pr_auc=float(average_precision_score(Y, P)))


def main():
    print("=" * 78)
    print("UJI LINTAS-DOMAIN  Peru (latih)  ->  ds_B / Medan (uji)")
    print("=" * 78)
    m, dva, dvd = train_peru()

    rows = []
    rows.append(("peru_val (dalam domain)", score(m, dva)))
    rows.append(("peru_val resolusi ds_B", score(m, dvd)))
    dsb = DsbCrops()
    print("ds_B crop: %d pohon unik (%s)" % (len(dsb), dict(dsb.counts)))
    rows.append(("ds_B (LINTAS DOMAIN)", score(m, dsb, note="ds_B")))

    print(" " * 60, end="\r")
    print()
    print("  %-26s %6s %6s %8s %9s %9s" %
          ("lengan", "n", "pos", "base", "ROC-AUC", "PR-AUC"))
    for name, r in rows:
        print("  %-26s %6d %6d %8.4f %9.3f %9.3f"
              % (name, r["n"], r["pos"], r["base"], r["roc_auc"], r["pr_auc"]))

    xd = rows[-1][1]
    print()
    print("  PR-AUC acak pada ds_B = laju dasar = %.4f" % xd["base"])
    print("  lift PR-AUC lintas-domain = %.2fx acak" % (xd["pr_auc"] / xd["base"]))
    print()
    print("BACAAN. ROC-AUC 0,5 dan PR-AUC = laju dasar berarti TIDAK ADA transfer.")
    print("Ingat uji ini mengukur transfer penampakan DAN kesepakatan definisi")
    print("label sekaligus; hasil negatif tidak memisahkan keduanya.")
    return rows


if __name__ == "__main__":
    main()
