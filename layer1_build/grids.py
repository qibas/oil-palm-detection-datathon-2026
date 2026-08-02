import warnings, os, glob, json, random, collections, statistics, sys
warnings.filterwarnings("ignore")
from PIL import Image, ImageDraw

random.seed(1234)
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)

def load_coco(tag):
    root = os.path.join(BASE, f"ds_{tag}")
    imgs_by_id = {}        # (split,id) -> path,w,h
    cats = {}              # id -> name
    anns = []              # dict: split, image_id, path, bbox, cat
    for split in ["train", "valid", "test"]:
        jp = os.path.join(root, split, "_annotations.coco.json")
        if not os.path.isfile(jp):
            continue
        d = json.load(open(jp))
        for c in d.get("categories", []):
            cats[c["id"]] = c["name"]
        idmap = {}
        for im in d["images"]:
            p = os.path.join(root, split, im["file_name"])
            idmap[im["id"]] = (p, im["width"], im["height"])
        for a in d["annotations"]:
            im = idmap.get(a["image_id"])
            if im is None: continue
            anns.append(dict(split=split, path=im[0], W=im[1], H=im[2],
                             bbox=a["bbox"], cat=cats.get(a["category_id"], str(a["category_id"]))))
    return cats, anns

def stats(tag):
    cats, anns = load_coco(tag)
    print(f"\n===== [{tag}] categories: {cats}")
    by = collections.defaultdict(list)
    for a in anns:
        by[a["cat"]].append(a)
    for cat, lst in sorted(by.items(), key=lambda kv: -len(kv[1])):
        ws = [a["bbox"][2] for a in lst]
        hs = [a["bbox"][3] for a in lst]
        diag = [ (w*w+h*h)**0.5 for w,h in zip(ws,hs)]
        # fraction of image the box occupies
        fr = [ (a["bbox"][2]*a["bbox"][3])/(a["W"]*a["H"]) for a in lst]
        print(f"  {cat:10s} n={len(lst):6d}  box_w med={statistics.median(ws):5.0f} "
              f"box_h med={statistics.median(hs):5.0f}  diag med={statistics.median(diag):5.0f}px "
              f"  box%img med={100*statistics.median(fr):.2f}%")
    return cats, anns, by

def grid(cells, cols, cell, path, labels=None):
    rows = (len(cells)+cols-1)//cols
    W, H = cols*cell, rows*cell
    canvas = Image.new("RGB", (W,H), (30,30,30))
    dr = ImageDraw.Draw(canvas)
    for i, im in enumerate(cells):
        r,c = divmod(i, cols)
        im = im.copy(); im.thumbnail((cell-4, cell-4))
        canvas.paste(im, (c*cell+2, r*cell+2))
        if labels: dr.text((c*cell+4, r*cell+4), labels[i], fill=(255,255,0))
    canvas.save(path)
    print("   saved", path)

def class_crops(tag, by, n=16, pad=0.4):
    for cat, lst in by.items():
        sample = random.sample(lst, min(n, len(lst)))
        cells = []
        for a in sample:
            try:
                im = Image.open(a["path"]).convert("RGB")
            except Exception:
                continue
            x,y,w,h = a["bbox"]
            px, py = w*pad, h*pad
            box = (max(0,x-px), max(0,y-py), min(a["W"],x+w+px), min(a["H"],y+h+py))
            crop = im.crop(box)
            cells.append(crop)
        if cells:
            grid(cells, 4, 240, os.path.join(OUT, f"{tag}_class_{cat}.jpg"))

def full_images(tag, anns, n=12):
    paths = list({a["path"] for a in anns})
    sample = random.sample(paths, min(n, len(paths)))
    cells = [Image.open(p).convert("RGB") for p in sample]
    grid(cells, 4, 320, os.path.join(OUT, f"{tag}_full_images.jpg"))

if __name__ == "__main__":
    tags = sys.argv[1:] or ["A","B","C"]
    for tag in tags:
        cats, anns, by = stats(tag)
        full_images(tag, anns)
        class_crops(tag, by)
