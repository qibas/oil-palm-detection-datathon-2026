"""Lapisan 1 — pembekuan dataset bersih dari ds_B (COCO Roboflow).

Masalah yang dipecahkan: ubin Roboflow diambil pada offset ACAK, bukan grid, sehingga
ubin saling bertindih dan satu pohon fisik muncul di median 32 ubin. Melatih/menilai
pada 151.060 baris anotasi berarti melatih pada 5.077 pohon yang direplikasi ~30x.

Keluaran (satu-satunya pintu masuk semua skrip Lapisan 1 hilir):
  layer1_crowns.csv          satu baris per POHON UNIK + ubin "tampilan kanonik"-nya
  layer1_tiles_disjoint.csv  subset ubin yang benar-benar tidak bertindih (evaluasi deteksi jujur)

Jalankan dari mana pun: path di-anchor ke lokasi file ini.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(ROOT, "layer1_build"))
from grids import load_coco   # noqa: E402  (butuh sys.path di atas)

TAG = "B"
# Nama ubin: <ortho>_<offx>_<offy>_jpg.rf.<hash>.jpg  — (offx, offy) = sudut kiri-atas
# potongan pada bidang ortomosaik skala penuh.
TILE_RE = re.compile(r"^(\d+_\d+)_(\d+)_(\d+)_")
DROP_CATS = {"Status"}        # kategori-induk COCO Roboflow, bukan objek
ROUND = 1                     # desimal untuk kunci dedup koordinat global


def annotations():
    """Semua anotasi ds_B + koordinat global pada bidang ortomosaik."""
    _, anns = load_coco(TAG)
    rows = []
    for a in anns:
        if a["cat"] in DROP_CATS:
            continue
        fn = os.path.basename(a["path"])
        m = TILE_RE.match(fn)
        if m is None:
            continue
        ox, oy = int(m.group(2)), int(m.group(3))
        x, y, w, h = a["bbox"]
        lx, ly = x + w / 2, y + h / 2          # pusat kotak, koordinat lokal ubin
        rows.append((
            m.group(1), fn, os.path.relpath(a["path"], ROOT),
            ox, oy, ox + lx, oy + ly, a["cat"],
            x, y, w, h, a["W"], a["H"],
            min(lx, ly, a["W"] - lx, a["H"] - ly),   # margin pusat ke tepi ubin
        ))
    return pd.DataFrame(rows, columns=[
        "ortho", "tile", "tile_path", "ox", "oy", "gx", "gy", "label",
        "bx", "by", "bw", "bh", "tw", "th", "margin"])


def disjoint_tiles(df):
    """Pemilihan greedy ubin yang saling lepas, per ortomosaik.

    Bukti langsung besarnya redundansi: tiap ortomosaik ~5.000x5.000 px hanya muat
    ~25 ubin 1.024 px, tapi datasetnya berisi 737-799 ubin.
    """
    out = []
    for ortho, s in df.drop_duplicates("tile").groupby("ortho"):
        b = s[["ox", "oy", "tw", "th", "tile", "tile_path"]].to_numpy(object)
        order = np.lexsort((s.ox.to_numpy(), s.oy.to_numpy()))
        kept = []
        for i in order:
            ox, oy, tw, th = b[i][0], b[i][1], b[i][2], b[i][3]
            if all(not (abs(ox - k[0]) < tw and abs(oy - k[1]) < th) for k in kept):
                kept.append((ox, oy, b[i][4], b[i][5]))
        out += [(ortho, k[0], k[1], k[2], k[3]) for k in kept]
    return pd.DataFrame(out, columns=["ortho", "ox", "oy", "tile", "tile_path"])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df = annotations()
    n_ann = len(df)

    df["kx"] = df.gx.round(ROUND)
    df["ky"] = df.gy.round(ROUND)
    key = ["ortho", "kx", "ky"]
    g = df.groupby(key, sort=False)
    n_uni = g.ngroups

    # -- Gerbang: duplikat harus konsisten label. Roboflow menyalin kotak identik,
    #    bukan menganotasi ulang, jadi seharusnya nol konflik. Kalau suatu saat
    #    tidak nol, jangan diam-diam pilih salah satu -- laporkan.
    conflicts = int((g.label.nunique() > 1).sum())
    assert conflicts == 0, f"{conflicts} pohon punya label bertentangan antar-ubin"

    # -- Tampilan kanonik: ubin di mana pohon paling di tengah (paling kecil
    #    kemungkinan terpotong tepi ubin).
    df["n_views"] = g.margin.transform("size")
    best = df.loc[df.groupby(key, sort=False).margin.idxmax()].copy()
    assert len(best) == n_uni

    best = best.sort_values(["ortho", "ky", "kx"]).reset_index(drop=True)
    best.insert(0, "crown_id", [f"{r.ortho}_{i:05d}" for i, r in enumerate(best.itertuples())])
    best["split_fold"] = best.ortho

    crowns = best[["crown_id", "ortho", "gx", "gy", "label", "n_views", "margin",
                   "tile", "tile_path", "bx", "by", "bw", "bh", "split_fold"]].rename(
        columns={"margin": "best_margin_px"})
    crowns.to_csv(os.path.join(BASE, "layer1_crowns.csv"), index=False)

    tiles = disjoint_tiles(df)
    tiles.to_csv(os.path.join(BASE, "layer1_tiles_disjoint.csv"), index=False)

    # ---- ringkasan (angka ini yang masuk DATASET_CARD.md) --------------------
    pos = crowns.label.eq("Unhealthy")
    per = crowns.groupby("ortho").label.apply(lambda s: int(s.eq("Unhealthy").sum()))
    n_tiles = df.drop_duplicates("tile").groupby("ortho").size()
    n_dis = tiles.groupby("ortho").size()

    print(f"[L1] anotasi {n_ann} -> tajuk unik {n_uni} ({n_ann/n_uni:.1f}x) | "
          f"konflik label {conflicts}")
    print(f"[L1] Unhealthy unik {int(pos.sum())} ({100*pos.mean():.2f}%) | per-ortho "
          + " / ".join(str(per[o]) for o in sorted(per.index)))
    print(f"[L1] tampilan/tajuk: median {int(crowns.n_views.median())} "
          f"(min {int(crowns.n_views.min())}, maks {int(crowns.n_views.max())}) | "
          f"tampilan kanonik >=60px dari tepi: {int((crowns.best_margin_px>=60).sum())}/{n_uni}")
    print("[L1] ubin saling-lepas "
          + " / ".join(str(n_dis[o]) for o in sorted(n_dis.index))
          + "  (dari " + " / ".join(str(n_tiles[o]) for o in sorted(n_tiles.index)) + ")")
    for o in sorted(per.index):
        s = crowns[crowns.ortho == o]
        xy = s[["gx", "gy"]].to_numpy()
        d, _ = cKDTree(xy).query(xy, k=2)
        print(f"       {o}: n={len(s)} jarak_tanam={np.median(d[:,1]):.0f}px "
              f"pos={per[o]} ({100*per[o]/len(s):.2f}%)")
    print(f"[L1] ditulis: layer1_crowns.csv ({len(crowns)}), "
          f"layer1_tiles_disjoint.csv ({len(tiles)})")

    assert n_uni == 5077, n_uni
    assert int(pos.sum()) == 66, int(pos.sum())


if __name__ == "__main__":
    main()
