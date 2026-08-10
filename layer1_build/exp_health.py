"""Layer-1 per-tree health classifier — feature extraction + leave-one-ortho-out block-CV.

UNIT ANALISIS = POHON UNIK, bukan baris anotasi. Ubin Roboflow bertindih ~30x, jadi
melatih pada 151.060 anotasi berarti melatih pada 5.077 pohon yang direplikasi. Skrip ini
membaca dataset beku `data_clean/layer1_crowns.csv` (satu baris per pohon + ubin
"tampilan kanonik"-nya) yang dihasilkan `data_clean/build_layer1.py`.

Honest metric: PR-AUC (threshold-free) pada kelas minoritas (Unhealthy), mean +/- std
antar-3-fold ortomosaik. Positif unik hanya 66 (17/31/18 per fold) -> hasilnya
UNDERPOWERED dan harus dilaporkan begitu; band derau lebar itu nyata, bukan kegagalan.

CATATAN STRUKTUR. Ekstraksi fitur dan block-CV dipisah jadi `extract()` dan `run()`
supaya `exp_health_nb.py` (varian fitur tetangga) memanggil kode yang SAMA, bukan
menyalinnya. Perilaku dan angka skrip ini tidak berubah oleh pemisahan itu."""
import warnings, os, collections, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2
import lightgbm as lgb
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
CLEAN = os.path.join(ROOT, "data_clean", "layer1_crowns.csv")

LABEL = {"Healthy": 0, "Unhealthy": 1}
COMMON = dict(objective="binary", verbose=-1, num_leaves=31, learning_rate=0.05,
              feature_fraction=0.8, min_data_in_leaf=50, seed=42)


def crown_feats(c):
    """Fitur penampakan dari SATU potongan tajuk (RGB float32). Tidak menyentuh label."""
    R, G, Bch = c[..., 0], c[..., 1], c[..., 2]
    exg = 2 * G - R - Bch
    mask = exg > 0                      # crown mask via ExG>0
    cov = float(mask.mean())
    gray = c.mean(2)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    gmag = np.sqrt(gx * gx + gy * gy)
    denom = (R + G + Bch + 1e-6)
    return dict(
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


def extract(verbose=True):
    """-> dict(X, y, regs, feat_names, gx, gy). Satu baris = satu pohon unik."""
    t0 = time.time()
    cr = pd.read_csv(CLEAN)
    if verbose:
        print(f"unique crowns: {len(cr)}  labels={cr.label.value_counts().to_dict()}  "
              f"folds={sorted(cr.ortho.unique())}")
        print(f"views collapsed per crown: median {int(cr.n_views.median())} "
              f"(the {int(cr.n_views.sum())} raw annotations are only {len(cr)} trees)")

    by_img = collections.defaultdict(list)
    for r in cr.itertuples():
        by_img[os.path.join(ROOT, r.tile_path)].append(r)

    rows, coords = [], []
    for pth, alist in by_img.items():
        im = cv2.imread(pth)
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)
        H, W = im.shape[:2]
        for a in alist:
            x, y_, w, h = a.bx, a.by, a.bw, a.bh
            x0, y0 = max(0, int(x)), max(0, int(y_))
            x1, y1 = min(W, int(x + w)), min(H, int(y_ + h))
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            feats = dict(box_w=w, box_h=h, box_area=w * h, aspect=w / (h + 1e-6))
            feats.update(crown_feats(im[y0:y1, x0:x1]))
            rows.append((feats, LABEL[a.label], a.ortho))
            coords.append((float(a.gx), float(a.gy)))

    feat_names = list(rows[0][0].keys())
    X = np.array([[r[0][k] for k in feat_names] for r in rows], dtype=np.float32)
    y = np.array([r[1] for r in rows])
    regs = np.array([r[2] for r in rows])
    xy = np.array(coords, dtype=np.float64)
    if verbose:
        print(f"crowns with features: {len(y)}  positives(Unhealthy)={int(y.sum())} "
              f"base_rate={100*y.mean():.2f}%  regions={sorted(set(regs))}  "
              f"feat_extract={time.time()-t0:.0f}s")
    return dict(X=X, y=y, regs=regs, feat_names=feat_names, xy=xy)


def run(X, y, regs, params, tag, verbose=True):
    """Leave-one-ortho-out block CV. -> (mean PR-AUC, std PR-AUC, per-fold list).

    std memakai ddof=0 (np.std bawaan) demi kompatibilitas dengan angka beku
    0,182 +/- 0,059 yang sudah dikutip di 00_RINGKASAN.csv. Perhatikan bahwa
    `y12.paired()` justru memakai ddof=1 dengan alasan eksplisit ("pada n kecil
    ddof=0 menyempitkan pita derau secara palsu"); pada n=3 selisihnya besar
    (0,059 lawan 0,073). Lihat exp_health_nb.py yang mencetak keduanya."""
    praucs, f1s, aucs = [], [], []
    for held in sorted(set(regs)):
        tr = regs != held; te = regs == held
        m = lgb.train(params, lgb.Dataset(X[tr], label=y[tr]), num_boost_round=300)
        p = m.predict(X[te])
        prauc = average_precision_score(y[te], p)
        auc = roc_auc_score(y[te], p)
        f1 = f1_score(y[te], (p > 0.5).astype(int), zero_division=0)
        praucs.append(prauc); f1s.append(f1); aucs.append(auc)
        if verbose:
            print(f"  [{tag}] hold={held}: PR-AUC={prauc:.3f} ROC-AUC={auc:.3f} "
                  f"F1@.5={f1:.3f} (n_te={te.sum()}, pos={int(y[te].sum())})")
    if verbose:
        print(f"  [{tag}] PR-AUC mean+/-std = {np.mean(praucs):.3f} +/- {np.std(praucs):.3f}"
              f"  | ROC-AUC {np.mean(aucs):.3f} | F1 {np.mean(f1s):.3f}")
    return np.mean(praucs), np.std(praucs), praucs


def main():
    t0 = time.time()
    d = extract()
    X, y, regs, feat_names = d["X"], d["y"], d["regs"], d["feat_names"]

    print(f"\n== BASELINE reference: random PR-AUC ~= base rate = {y.mean():.4f} ==")
    print("\n== vanilla LightGBM (no class weighting) ==")
    run(X, y, regs, {**COMMON}, "vanilla")
    print("\n== LightGBM is_unbalance=True ==")
    run(X, y, regs, {**COMMON, "is_unbalance": True}, "unbal")

    m = lgb.train({**COMMON}, lgb.Dataset(X, label=y), num_boost_round=300)
    imp = sorted(zip(feat_names, m.feature_importance()), key=lambda t: -t[1])
    print("\ntop features:", [f"{k}:{v}" for k, v in imp[:8]])
    print(f"\nTOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
