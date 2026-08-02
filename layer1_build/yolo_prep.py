"""Convert dataset B (COCO) to YOLO format with leave-one-ortho-out folds.
Flat yolo_B/images + yolo_B/labels; per-fold train/val image lists + data yaml.
Class map: Healthy=0, Unhealthy=1. We ignore B's leaky provided split and block by ortho."""
import warnings, os, re, shutil, collections
warnings.filterwarnings("ignore")
from grids import load_coco

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "yolo_B")
IMG = os.path.join(ROOT, "images"); LAB = os.path.join(ROOT, "labels")
os.makedirs(IMG, exist_ok=True); os.makedirs(LAB, exist_ok=True)
CLS = {"Healthy": 0, "Unhealthy": 1}

def region(fn):
    m = re.match(r"(\d+)_(\d+)_", fn.split(".rf.")[0])
    return f"{m.group(1)}_{m.group(2)}" if m else "OTHER"

cats, anns = load_coco("B")
by_img = collections.defaultdict(list)
for a in anns:
    by_img[a["path"]].append(a)

img_region = {}
n_lab = 0
for pth, alist in by_img.items():
    fn = os.path.basename(pth)
    stem = os.path.splitext(fn)[0]
    W, H = alist[0]["W"], alist[0]["H"]
    dst_img = os.path.join(IMG, fn)
    if not os.path.exists(dst_img):
        try: os.link(pth, dst_img)          # hardlink (same drive, no copy)
        except Exception: shutil.copy(pth, dst_img)
    lines = []
    for a in alist:
        if a["cat"] not in CLS: continue
        x, y, w, h = a["bbox"]
        cx, cy = (x + w/2)/W, (y + h/2)/H
        lines.append(f"{CLS[a['cat']]} {cx:.6f} {cy:.6f} {w/W:.6f} {h/H:.6f}")
    open(os.path.join(LAB, stem + ".txt"), "w").write("\n".join(lines))
    n_lab += len(lines)
    img_region[dst_img] = region(fn)

regions = sorted(set(img_region.values()))
print(f"images={len(img_region)} labels={n_lab} regions={regions}")
for r in regions:
    print(f"  {r}: {sum(v==r for v in img_region.values())} images")

# per-fold (leave-one-ortho-out) image lists + yaml
for i, held in enumerate(regions):
    tr = [p for p, rg in img_region.items() if rg != held]
    va = [p for p, rg in img_region.items() if rg == held]
    open(os.path.join(ROOT, f"fold{i}_train.txt"), "w").write("\n".join(tr))
    open(os.path.join(ROOT, f"fold{i}_val.txt"), "w").write("\n".join(va))
    yaml = (f"path: {ROOT}\n"
            f"train: fold{i}_train.txt\n"
            f"val: fold{i}_val.txt\n"
            f"names:\n  0: Healthy\n  1: Unhealthy\n")
    open(os.path.join(ROOT, f"fold{i}.yaml"), "w").write(yaml)
    print(f"fold{i} (hold {held}): train={len(tr)} val={len(va)} -> fold{i}.yaml")
print("YOLO_PREP DONE")
