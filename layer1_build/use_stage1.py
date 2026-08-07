"""Pakai model Tahap 1 yang sudah disimpan (`stage1_model.pkl`) pada citra apa pun.

    python use_stage1.py <citra-atau-folder> [-o keluaran.csv] [--conf 0.25]

Keluaran: satu baris per kotak terdeteksi -

    image, det_id, class, conf, cx, cy, w, h, xmin, ymin, xmax, ymax

`cx`,`cy` adalah PUSAT TAJUK dalam piksel citra. Pusat inilah bentuk keluaran
Tahap 1 yang berguna hilir; kotaknya sendiri hanya perantara.

Berkas .pkl berdiri sendiri: ia memuat bobot, nama kelas, setelan inferensi, dan
BATAS pembacaannya. Skrip ini tidak butuh `anom.py`.
"""
import argparse
import csv
import os
import pickle
import sys
import tempfile

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load(pkl):
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    if d.get("format") != "sawitguard-stage1/1":
        raise SystemExit("format .pkl tak dikenal: %r" % d.get("format"))
    from ultralytics import YOLO
    p = os.path.join(tempfile.mkdtemp(prefix="stage1_"), "weights.pt")
    with open(p, "wb") as f:
        f.write(d["weights_pt"])
    return YOLO(p), d


def images_in(path):
    if os.path.isfile(path):
        return [path]
    out = []
    for root, _, files in os.walk(path):
        out += [os.path.join(root, f) for f in sorted(files)
                if f.lower().endswith(IMG_EXT)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="berkas citra atau folder")
    ap.add_argument("-m", "--model", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "stage1_model.pkl"))
    ap.add_argument("-o", "--out", default="stage1_detections.csv")
    ap.add_argument("--conf", type=float, default=None, help="default: dari .pkl")
    ap.add_argument("--imgsz", type=int, default=None, help="default: dari .pkl")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--quiet-limits", action="store_true")
    a = ap.parse_args()

    model, meta = load(a.model)
    conf = a.conf if a.conf is not None else meta["conf"]
    imgsz = a.imgsz if a.imgsz is not None else meta["imgsz"]
    names = {int(k): v for k, v in meta["names"].items()}

    files = images_in(a.source)
    if not files:
        raise SystemExit("tidak ada citra di %s" % a.source)
    print("model  : %s (sha %s)" % (os.path.basename(a.model),
                                    meta["weights_sha256"][:12]))
    print("kelas  : %s | conf=%.2f imgsz=%d" % (names, conf, imgsz))
    print("citra  : %d" % len(files), flush=True)

    rows = []
    for i in range(0, len(files), a.batch):
        chunk = files[i:i + a.batch]
        # WAJIB zip terhadap `chunk`: bila sumbernya DAFTAR path, Ultralytics
        # menamai ulang hasilnya 'image0.jpg', 'image1.jpg', ... sehingga
        # r.path tidak lagi menunjuk berkas aslinya.
        for src, r in zip(chunk, model.predict(chunk, imgsz=imgsz, conf=conf,
                                               verbose=False)):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xywh = b.xywh.cpu().numpy()
            xyxy = b.xyxy.cpu().numpy()
            for j, ((cx, cy, w, h), (x0, y0, x1, y1), c, k) in enumerate(
                    zip(xywh, xyxy, b.conf.cpu().numpy(), b.cls.cpu().numpy())):
                rows.append(dict(image=os.path.relpath(src), det_id=j,
                                 **{"class": names[int(k)]},
                                 conf=round(float(c), 4),
                                 cx=round(float(cx), 1), cy=round(float(cy), 1),
                                 w=round(float(w), 1), h=round(float(h), 1),
                                 xmin=round(float(x0), 1), ymin=round(float(y0), 1),
                                 xmax=round(float(x1), 1), ymax=round(float(y1), 1)))

    cols = ["image", "det_id", "class", "conf", "cx", "cy", "w", "h",
            "xmin", "ymin", "xmax", "ymax"]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    n_anom = sum(1 for r in rows if r["class"] == "PalmAnom")
    print("\nterdeteksi %d kotak pada %d citra -> %s"
          % (len(rows), len({r["image"] for r in rows}), a.out))
    print("  PalmAnom %d | PalmSan %d" % (n_anom, len(rows) - n_anom))

    if not a.quiet_limits:
        print("\nBATAS PEMBACAAN (dibenamkan di dalam .pkl):")
        for i, t in enumerate(meta["limits"], 1):
            print("  %d. %s" % (i, t))


if __name__ == "__main__":
    main()
