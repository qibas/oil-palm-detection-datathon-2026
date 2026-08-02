"""Layer-1 per-tree health classifier — feature extraction + leave-one-ortho-out block-CV.

UNIT ANALISIS = POHON UNIK, bukan baris anotasi. Ubin Roboflow bertindih ~30x, jadi
melatih pada 151.060 anotasi berarti melatih pada 5.077 pohon yang direplikasi. Skrip ini
membaca dataset beku `data_clean/layer1_crowns.csv` (satu baris per pohon + ubin
"tampilan kanonik"-nya) yang dihasilkan `data_clean/build_layer1.py`.

Honest metric: PR-AUC (threshold-free) pada kelas minoritas (Unhealthy), mean +/- std
antar-3-fold ortomosaik. Positif unik hanya 66 (17/31/18 per fold) -> hasilnya
UNDERPOWERED dan harus dilaporkan begitu; band derau lebar itu nyata, bukan kegagalan."""
import warnings, os, collections, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2
import lightgbm as lgb
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
CLEAN = os.path.join(ROOT, "data_clean", "layer1_crowns.csv")
t0 = time.time()

# ---- dataset bersih: 1 baris = 1 pohon unik, fold = ortomosaik ----------------
cr = pd.read_csv(CLEAN)
print(f"unique crowns: {len(cr)}  labels={cr.label.value_counts().to_dict()}  "
      f"folds={sorted(cr.ortho.unique())}")
print(f"views collapsed per crown: median {int(cr.n_views.median())} "
      f"(the {int(cr.n_views.sum())} raw annotations are only {len(cr)} trees)")

# buka tiap ubin kanonik sekali saja
by_img = collections.defaultdict(list)
for r in cr.itertuples():
    by_img[os.path.join(ROOT, r.tile_path)].append(r)

LABEL = {"Healthy": 0, "Unhealthy": 1}
rows = []  # (features dict, label, region)
for pth, alist in by_img.items():
    im = cv2.imread(pth)
    if im is None:
        continue
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)
    H, W = im.shape[:2]
    for a in alist:
        reg = a.ortho
        x, y, w, h = a.bx, a.by, a.bw, a.bh
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(W, int(x + w)), min(H, int(y + h))
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        c = im[y0:y1, x0:x1]
        R, G, Bch = c[..., 0], c[..., 1], c[..., 2]
        exg = 2 * G - R - Bch
        # crown mask via ExG>0
        mask = exg > 0
        cov = float(mask.mean())
        gray = c.mean(2)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        gmag = np.sqrt(gx * gx + gy * gy)
        denom = (R + G + Bch + 1e-6)
        feats = dict(
            box_w=w, box_h=h, box_area=w * h, aspect=w / (h + 1e-6),
            crown_cov=cov,
            R_mean=R.mean(), G_mean=G.mean(), B_mean=Bch.mean(),
            R_std=R.std(), G_std=G.std(), B_std=Bch.std(),
            exg_mean=exg.mean(), exg_std=exg.std(),
            GmR=(G - R).mean(), GmB=(G - Bch).mean(),
            green_frac=(G / denom).mean(),
            lap_var=float(lap.var()),
            gmag_mean=float(gmag.mean()), gmag_std=float(gmag.std()),
            bright=gray.mean(),
        )
        rows.append((feats, LABEL[a.label], reg))

feat_names = list(rows[0][0].keys())
X = np.array([[r[0][k] for k in feat_names] for r in rows], dtype=np.float32)
y = np.array([r[1] for r in rows])
regs = np.array([r[2] for r in rows])
uregs = sorted(set(regs))
print(f"crowns with features: {len(y)}  positives(Unhealthy)={int(y.sum())} "
      f"base_rate={100*y.mean():.2f}%  regions={uregs}  feat_extract={time.time()-t0:.0f}s")

# ---- leave-one-ortho-out block CV ----
def run(params, tag):
    praucs, f1s, aucs = [], [], []
    for held in uregs:
        tr = regs != held; te = regs == held
        dtr = lgb.Dataset(X[tr], label=y[tr])
        m = lgb.train(params, dtr, num_boost_round=300)
        p = m.predict(X[te])
        prauc = average_precision_score(y[te], p)
        auc = roc_auc_score(y[te], p)
        f1 = f1_score(y[te], (p > 0.5).astype(int), zero_division=0)
        praucs.append(prauc); f1s.append(f1); aucs.append(auc)
        print(f"  [{tag}] hold={held}: PR-AUC={prauc:.3f} ROC-AUC={auc:.3f} F1@.5={f1:.3f} "
              f"(n_te={te.sum()}, pos={int(y[te].sum())})")
    print(f"  [{tag}] PR-AUC mean+/-std = {np.mean(praucs):.3f} +/- {np.std(praucs):.3f}  "
          f"| ROC-AUC {np.mean(aucs):.3f} | F1 {np.mean(f1s):.3f}")
    return np.mean(praucs), np.std(praucs)

base_rate = y.mean()
print(f"\n== BASELINE reference: random PR-AUC ~= base rate = {base_rate:.4f} ==")
common = dict(objective="binary", verbose=-1, num_leaves=31, learning_rate=0.05,
             feature_fraction=0.8, min_data_in_leaf=50, seed=42)
print("\n== vanilla LightGBM (no class weighting) ==")
run({**common}, "vanilla")
print("\n== LightGBM is_unbalance=True ==")
run({**common, "is_unbalance": True}, "unbal")

# feature importance on a full-data model (context only)
m = lgb.train({**common}, lgb.Dataset(X, label=y), num_boost_round=300)
imp = sorted(zip(feat_names, m.feature_importance()), key=lambda t: -t[1])
print("\ntop features:", [f"{k}:{v}" for k, v in imp[:8]])
print(f"\nTOTAL {time.time()-t0:.0f}s")
