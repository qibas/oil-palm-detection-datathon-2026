"""Uji-mandiri `dataset.py` — mencetak BUKTI tiap penjaga (guard) berlaku.

Jalankan: python layer2_real/test_dataset.py
Tidak ada GPU, tidak ada training. ~1 detik.

Empat penjaga, dan apa yang bisa/tidak bisa ditangkap masing-masing:

  G1 identitas-per-progeny  blok SELF+GENO harus sama untuk semua pohon sefamili
                            pada sensus yang sama  -> menangkap fitur per-pohon
                            apa pun yang berasal dari outcome (mis. y_t1s).
  G2 STATE nol di risk set  blok STATE harus persis 0 untuk setiap contoh
                            -> menangkap kebocoran status diri sendiri, termasuk
                            status masa depan.
  G3 bangun-ulang-terpotong bangun ulang dari panel yang dipotong di sensus k
                            harus identik bit-per-bit dengan k sensus pertama
                            versi penuh -> menangkap APA PUN yang menyentuh t+1
                            ke atas, termasuk agregat tingkat famili.
  G4 probe AUC satu-fitur   ROC-AUC tiap kolom sendirian terhadap label
                            -> bukan penjaga formal, melainkan alat bau: kolom
                            dengan AUC ~1,0 adalah bukti kebocoran telanjang.

Tiga varian bocor SENGAJA dibangun di bawah untuk membuktikan penjaga menyala.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as ds  # noqa: E402

OK, BAD = "OK ", "GAGAL "
_fails = []


def check(name, cond, detail=""):
    print(f"  [{OK if cond else BAD}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        _fails.append(name)
    return cond


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# ---------------------------------------------------------------------------
# PENJAGA
# ---------------------------------------------------------------------------
def g1_progeny_identity(X, n_selfgeno=None):
    """Blok SELF+GENO: pada tiap sensus, baris hanya boleh ditentukan oleh progeny.
    Mengembalikan jumlah baris-unik maksimum antar sensus (harus == 14); 1e9 bila
    ada pohon sefamili yang berbeda (= penjaga menyala)."""
    nodes, _, _ = ds.load()
    code = nodes["progeny"].map({g: i for i, g in enumerate(ds.progeny_levels())}).to_numpy()
    blk = X[:, :, :(ds.GENO_SLICE.stop if n_selfgeno is None else n_selfgeno)]
    worst = 0
    for t in range(blk.shape[0]):
        u = np.unique(blk[t], axis=0)
        worst = max(worst, len(u))
        for g in np.unique(code):
            rows = blk[t][code == g]
            if not np.allclose(rows, rows[0]):
                return 10 ** 9
    return worst


def g2_state_zero_on_risk(X, h=1, state_start=None):
    """Blok STATE harus persis 0 pada setiap contoh risk set. Kembalikan |maks|."""
    tr, tt, _ = ds.build_examples(h)
    v = X[tt, tr][:, (ds.STATE_SLICE.start if state_start is None else state_start):]
    return float(np.abs(v).max()) if v.size else np.nan


def g3_past_only(build_fn, ks=(5, 20, 40)):
    """Bangun ulang dari panel terpotong; harus identik bit-per-bit."""
    st, cen = ds.status_matrix(), ds.census()
    full = build_fn(st, cen)
    out = []
    for k in ks:
        tr = build_fn(st[:k], cen[:k])
        out.append((k, bool(np.array_equal(full[:k], tr))))
    return out


def g4_leak_probe(X, h=1, names=None):
    """ROC-AUC tiap kolom sendirian. Kembalikan (nama, auc) urut menurun."""
    tr, tt, y = ds.build_examples(h)
    names = names or ds.feature_names()
    Z = X[tt, tr]
    res = []
    for j, nm in enumerate(names):
        col = Z[:, j]
        if np.allclose(col, col[0]):
            res.append((nm, 0.5))
            continue
        a = roc_auc_score(y, col)
        res.append((nm, max(a, 1 - a)))
    return sorted(res, key=lambda kv: -kv[1])


# ---------------------------------------------------------------------------
# VARIAN BOCOR (sengaja salah — untuk membuktikan penjaga menyala)
# ---------------------------------------------------------------------------
def leak_static_outcome(status, cen):
    """L1 — sisipkan y_t1s (waktu gejala pertama) sebagai kolom statis.
    Kebocoran outcome paling telanjang: fitur per-pohon dari kolom survival."""
    nodes, _, _ = ds.load()
    X = ds.raw_features(status, cen)
    y1 = nodes["y_t1s"].to_numpy(float)
    lk = np.repeat(y1[None, :, None], X.shape[0], axis=0)
    # disisipkan DI DALAM blok SELF+GENO supaya G1 berkesempatan menangkapnya
    return np.concatenate([X[:, :, :ds.GENO_SLICE.stop], lk,
                           X[:, :, ds.STATE_SLICE]], axis=-1)


def leak_time_to_event(status, cen):
    """L1b — sisa waktu menuju gejala, y_t1s - t. Bentuk kebocoran outcome yang
    paling sering terjadi tanpa sengaja ('time until event' dihitung sebagai fitur).
    Sama-sama per-pohon, tapi berbeda dari L1: ia SELARAS dengan sensus, jadi
    probe AUC gabungan pun menyala."""
    nodes, _, _ = ds.load()
    X = ds.raw_features(status, cen)
    y1 = nodes["y_t1s"].to_numpy(float)
    lk = (y1[None, :] - np.asarray(cen)[:, None])[:, :, None]
    return np.concatenate([X[:, :, :ds.GENO_SLICE.stop], lk,
                           X[:, :, ds.STATE_SLICE]], axis=-1)


def leak_own_future_status(status, cen):
    """L2 — sisipkan status DIRI SENDIRI pada t+1 ke dalam blok STATE."""
    X = ds.raw_features(status, cen)
    sym = np.isin(status, ds.POS_STATUS).astype(np.float64)
    fut = np.concatenate([sym[1:], sym[-1:]], axis=0)          # geser satu ke depan
    return np.concatenate([X, fut[:, :, None]], axis=-1)


def leak_family_future_hazard(status, cen):
    """L3 — insidensi MASA DEPAN tingkat famili: untuk tiap (sensus t, progeny),
    fraksi pohon famili itu yang menjadi bergejala di (t, t+1].
    Konstan di dalam famili -> G1 lolos. Tidak menyentuh blok STATE -> G2 lolos.
    HANYA G3 yang bisa menangkapnya."""
    nodes, _, _ = ds.load()
    X = ds.raw_features(status, cen)
    code = nodes["progeny"].map({g: i for i, g in enumerate(ds.progeny_levels())}).to_numpy()
    T, N = status.shape
    sym = np.isin(status, ds.POS_STATUS)
    col = np.zeros((T, N))
    for t in range(T):
        nxt = sym[min(t + 1, T - 1)] & ~sym[t]                 # jadi bergejala di (t,t+1]
        for g in np.unique(code):
            m = code == g
            col[t, m] = nxt[m].mean()
    return np.concatenate([X[:, :, :ds.GENO_SLICE.stop], col[:, :, None],
                           X[:, :, ds.STATE_SLICE]], axis=-1)


# ---------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    nodes, panel, edges = ds.load()
    cen = ds.census()
    T, N = ds.shape()
    st = ds.status_matrix()

    hdr("0. MUATAN DASAR")
    print(f"  nodes {nodes.shape} | panel {panel.shape} | edges {edges.shape}")
    check("N = 1200", N == 1200, f"N={N}")
    check("T = 45 sensus", T == 45, f"{cen.min()}–{cen.max()} th, dt unik {sorted(set(np.round(np.diff(cen),3)))}")
    check("panel = N x T baris", len(panel) == N * T, f"{len(panel)}")
    check("at_risk == (status=='A')", bool(((panel.at_risk == 1) == (panel.status == "A")).all()))
    print(f"  cacah status: {panel.status.value_counts().to_dict()}")
    # monotonisitas: sekali keluar dari 'A', tidak pernah kembali
    left = (st != "A").cumsum(axis=0) > 0            # sudah pernah keluar dari 'A'
    back = int((left[:-1] & (st[1:] == "A")).sum())  # lalu kembali jadi 'A'
    check("status monoton (tidak ada kembali ke 'A')", back == 0, f"{back} pelanggaran")

    # -----------------------------------------------------------------------
    hdr("1. FITUR SIMPUL — bentuk, blok, dan scaler")
    train_t = np.arange(T)                      # lipatan = parcel, bukan waktu
    X = ds.node_features(train_t)
    names = ds.feature_names()
    print(f"  node_features(train_t) -> {X.shape} {X.dtype}   (T, N, d) d={X.shape[2]}")
    print(f"  blok  SELF  {ds.SELF_SLICE}  {list(names[ds.SELF_SLICE])}")
    print(f"  blok  GENO  {ds.GENO_SLICE}  {len(ds.progeny_levels())} level: {list(ds.progeny_levels())}")
    print(f"  blok  STATE {ds.STATE_SLICE} {list(names[ds.STATE_SLICE])}")
    check("shape (T,N,d) benar", X.shape == (T, N, len(names)))
    check("float32", X.dtype == np.float32)
    check("tidak ada NaN/Inf", bool(np.isfinite(X).all()))

    # scaler HANYA melihat sensus latih — dibuktikan dengan hitung ulang manual
    print("\n  Bukti scaler hanya melihat sensus latih:")
    raw = ds.raw_features()
    for frac, label in ((0.7, "train_t = 31 sensus pertama (temporal_split 0.7)"),
                        (1.0, "train_t = seluruh 45 sensus (lipatan parcel murni)")):
        tt = ds.temporal_split(frac)[0] if frac < 1 else np.arange(T)
        sc = ds.fit_scaler(tt)
        manual = raw[np.ix_(tt, np.arange(N))][:, :, ds.SELF_SLICE].reshape(-1, 4)
        ok = np.allclose(sc.mean_, manual.mean(0)) and np.allclose(sc.scale_, manual.std(0), atol=1e-9)
        allc = raw[:, :, ds.SELF_SLICE].reshape(-1, 4).mean(0)
        print(f"    {label}")
        print(f"      scaler.mean_ = {np.round(sc.mean_, 4)}")
        print(f"      mean sensus latih saja  = {np.round(manual.mean(0), 4)}  -> cocok: {ok}")
        print(f"      mean SELURUH sensus     = {np.round(allc, 4)}"
              f"  -> beda dari scaler: {not np.allclose(sc.mean_, allc)}")
        check(f"scaler[{label[:12]}...] = statistik sensus latih saja", bool(ok))
    sc70 = ds.fit_scaler(ds.temporal_split(0.7)[0])
    scall = ds.fit_scaler(np.arange(T))
    check("train_t benar-benar mengubah scaler (bukan no-op)",
          not np.allclose(sc70.mean_, scall.mean_),
          f"delta mean = {np.round(np.abs(sc70.mean_ - scall.mean_), 4)}")
    # kejujuran: train_nodes TIDAK mengubah apa pun dengan set fitur saat ini,
    # karena blok SELF identik antar simpul dan GENO/STATE tidak diskalakan.
    trn0 = ds.folds()[0][0]
    same_nodes = np.allclose(ds.fit_scaler(train_t, trn0).mean_, scall.mean_)
    print(f"    train_nodes (argumen tambahan): scaler identik dengan/atau tanpa "
          f"pembatasan simpul = {same_nodes}")
    print("      -> dengan set fitur ini argumennya no-op; disediakan agar klaim")
    print("         'scaler hanya melihat data latih' tetap benar bila kelak ada")
    print("         kolom yang bervariasi antar pohon.")

    # -----------------------------------------------------------------------
    hdr("2. GRAF — view, derajat, lintas-parcel")
    A_true = ds.adjacency("true")
    A_zero = ds.adjacency("zero")
    deg_true = ds.degrees(A_true)
    par = nodes["parcel"].to_numpy()
    xy = nodes[["xm", "ym"]].to_numpy()

    def cross(A):
        i, j = np.where(np.triu(A) > 0)
        return int((par[i] != par[j]).sum()), len(i)

    def meanlen(A):
        i, j = np.where(np.triu(A) > 0)
        return float(np.linalg.norm(xy[i] - xy[j], axis=1).mean())

    c_t, e_t = cross(A_true)
    print(f"  true   : {e_t} sisi | derajat rata-rata {deg_true.mean():.2f} "
          f"(min {deg_true.min()}, maks {deg_true.max()}) | panjang sisi rata-rata {meanlen(A_true):.3f}")
    check("simetris, diagonal 0", bool((A_true == A_true.T).all() and (np.diag(A_true) == 0).all()))
    check("jumlah sisi = 3354", e_t == 3354, f"{e_t}")
    check("0 sisi lintas-parcel (view true)", c_t == 0, f"{c_t}")
    check("derajat cocok dengan kolom `deg` di panel",
          bool((panel[panel.t == cen[0]].set_index("palm_id")
                .reindex(ds.palm_ids())["deg"].to_numpy() == deg_true).all()))
    check("view zero seluruhnya nol", bool((A_zero == 0).all()),
          f"nnz={int((A_zero!=0).sum())}, shape={A_zero.shape}")

    print("\n  view `random` (double-edge swap dalam parcel):")
    for s in (0, 1, 2):
        A_r = ds.adjacency("random", seed=s)
        c_r, e_r = cross(A_r)
        d_r = ds.degrees(A_r)
        same = int((np.triu(A_true) * np.triu(A_r)).sum())
        print(f"    seed {s}: sisi {e_r} | derajat identik {np.array_equal(d_r, deg_true)} | "
              f"lintas-parcel {c_r} | sisi bertahan {same}/{e_t} ({100*same/e_t:.1f}%) | "
              f"panjang sisi rata-rata {meanlen(A_r):.3f}")
        check(f"random(seed={s}) mempertahankan BARISAN derajat",
              np.array_equal(d_r, deg_true))
        check(f"random(seed={s}) tetap 0 sisi lintas-parcel", c_r == 0, f"{c_r}")
        check(f"random(seed={s}) menghancurkan struktur",
              meanlen(A_r) > 5 * meanlen(A_true))

    print("\n  view `perturb` (interpolasi true<->random):")
    for eps in (0.0, 0.25, 0.5, 0.75, 1.0):
        A_p = ds.adjacency("perturb", seed=0, eps=eps)
        i, j = np.where(np.triu(A_p) > 0)
        surv = int((np.triu(A_true) * np.triu(A_p)).sum())
        print(f"    eps={eps:<5} sisi bertahan {surv}/{e_t} ({100*surv/e_t:5.1f}%) | "
              f"derajat identik {np.array_equal(ds.degrees(A_p), deg_true)} | "
              f"panjang sisi rata-rata {meanlen(A_p):.3f}")
        check(f"perturb(eps={eps}) mempertahankan derajat",
              np.array_equal(ds.degrees(A_p), deg_true))
    check("perturb(eps=0) == true", bool((ds.adjacency("perturb", 0, 0.0) == A_true).all()))
    check("perturb(eps=1) == random (seed sama)",
          bool((ds.adjacency("perturb", 0, 1.0) == ds.adjacency("random", 0)).all()))
    try:
        ds.adjacency("perturb")
        raised = False
    except ValueError:
        raised = True
    check("perturb tanpa eps gagal keras (bukan diam-diam jadi acak penuh)", raised)
    try:
        ds.adjacency("wind")
        raised2 = False
    except ValueError:
        raised2 = True
    check("view tak dikenal ditolak (mis. 'wind' — data ini hanya punya 1 relasi)", raised2)

    # -----------------------------------------------------------------------
    hdr("3. DIFUSI — D = A @ F membawa sinyal epidemi tetangga, bukan F")
    Fr = ds.raw_features()
    Draw = ds.diffuse(Fr, A_true, scale=False)[:, :, 0, :]
    k_sym = names.index("is_sympt"); k_dead = names.index("is_dead"); k_cen = names.index("is_cens")
    p_sym = panel.pivot(index="t", columns="palm_id", values="n_nb_sympt").reindex(
        index=cen, columns=ds.palm_ids()).to_numpy()
    p_dead = panel.pivot(index="t", columns="palm_id", values="n_nb_dead").reindex(
        index=cen, columns=ds.palm_ids()).to_numpy()
    p_obs = panel.pivot(index="t", columns="palm_id", values="n_nb_obs").reindex(
        index=cen, columns=ds.palm_ids()).to_numpy()
    check("difusi is_sympt == kolom panel n_nb_sympt",
          bool(np.allclose(Draw[:, :, k_sym], p_sym)),
          f"maks selisih {np.abs(Draw[:,:,k_sym]-p_sym).max():.3g}")
    check("difusi is_dead  == kolom panel n_nb_dead",
          bool(np.allclose(Draw[:, :, k_dead], p_dead)),
          f"maks selisih {np.abs(Draw[:,:,k_dead]-p_dead).max():.3g}")
    check("difusi is_cens  == deg - n_nb_obs",
          bool(np.allclose(Draw[:, :, k_cen], deg_true[None, :] - p_obs)))
    print(f"  skala adjacency global 1/mean(rowsum) = {ds.adjacency_scale(A_true):.4f} "
          f"-> D_scaled kira-kira RATA-RATA tetangga")
    Dz = ds.diffuse(Fr, A_zero, scale=False)
    check("difusi pada view zero seluruhnya nol", bool((Dz == 0).all()))

    # -----------------------------------------------------------------------
    hdr("4. TUGAS PERAMALAN — cacah contoh, pos-rate, sensor")
    print("  Acuan DATASET_CARD.md (tanpa pemangkasan WINDOW): "
          "h=1 1,58% | h=2 3,03% | h=3 4,45% | h=4 5,65%")
    print(f"  Di sini t harus >= WINDOW-1 = {ds.WINDOW-1} DAN t+h < T, jadi cacahnya lebih kecil.\n")
    print(f"  {'h':>2} {'contoh':>8} {'positif':>8} {'pos-rate':>9} {'sensus sah':>11} "
          f"{'negatif tersensor':>19}")
    for h in ds.HORIZONS:
        tr, tt, y = ds.build_examples(h)
        nc, nneg = ds.censored_in_horizon(h)
        print(f"  {h:>2} {y.size:>8} {int(y.sum()):>8} {100*y.mean():>8.2f}% "
              f"{len(np.unique(tt)):>11} {nc:>10}/{nneg} ({100*nc/nneg:.2f}%)")
        check(f"h={h}: risk set bebas pohon tersensor",
              bool((st[tt, tr] != "C").all()), f"{int((st[tt,tr]=='C').sum())} pohon 'C'")
        check(f"h={h}: SEMUA contoh berstatus 'A'",
              bool((st[tt, tr] == "A").all()))
        check(f"h={h}: t >= WINDOW-1 dan t+h < T", bool(tt.min() >= ds.WINDOW - 1 and (tt + h).max() < T))
        # label dihitung ulang dari nol, secara independen
        y2 = np.array([1.0 if np.isin(st[a + 1:a + h + 1, i], ds.POS_STATUS).any() else 0.0
                       for i, a in zip(tr[:5000], tt[:5000])], dtype=np.float32)
        check(f"h={h}: label cocok dengan hitung-ulang independen (5000 pertama)",
              bool((y2 == y[:5000]).all()))
    # reproduksi angka kartu dataset apa adanya (tanpa pemangkasan WINDOW)
    print("\n  Reproduksi apa adanya angka DATASET_CARD.md (WINDOW dimatikan):")
    W = ds.WINDOW
    ds.WINDOW = 1
    try:
        for h in ds.HORIZONS:
            _, _, y = ds.build_examples(h)
            print(f"    h={h}: {y.size} contoh, {int(y.sum())} positif ({100*y.mean():.2f}%)")
    finally:
        ds.WINDOW = W
    check("WINDOW dipulihkan ke 3", ds.WINDOW == 3)

    # -----------------------------------------------------------------------
    hdr("5. LIPATAN — leave-one-parcel-out")
    fl = ds.folds()
    check("2 lipatan", len(fl) == 2)
    for k, (trn, tes) in enumerate(fl):
        p = sorted(set(par[tes]))
        gtr = set(nodes.progeny[trn]); gte = set(nodes.progeny[tes])
        print(f"  lipatan {k}: latih {trn.sum():>4} simpul | uji {tes.sum():>4} simpul "
              f"(parcel {p}) | famili latih {len(gtr)} uji {len(gte)} irisan {len(gtr & gte)}")
        check(f"lipatan {k}: latih/uji saling lepas dan menutupi N",
              bool((trn ^ tes).all() and not (trn & tes).any()))
        check(f"lipatan {k}: 14/14 famili ada di kedua sisi -> tidak terkonfound genotipe",
              len(gtr & gte) == 14, f"{len(gtr & gte)}")
        cut = int((np.triu(A_true)[np.ix_(np.where(trn)[0], np.where(tes)[0])] > 0).sum()
                  + (np.triu(A_true)[np.ix_(np.where(tes)[0], np.where(trn)[0])] > 0).sum())
        check(f"lipatan {k}: 0 sisi terputus antara latih dan uji", cut == 0, f"{cut}")
        for h in (1, 4):
            _, _, ytr = ds.build_examples(h, nodes=trn)
            _, _, yte = ds.build_examples(h, nodes=tes)
            print(f"      h={h}: latih {ytr.size} contoh / {100*ytr.mean():.2f}% pos | "
                  f"uji {yte.size} contoh / {100*yte.mean():.2f}% pos")

    # -----------------------------------------------------------------------
    hdr("6. BENTUK TENSOR MODEL (F_seq, D_seq)")
    Ds = ds.diffuse(X, A_true)
    tr, tt, y = ds.build_examples(1)
    Fq, Dq = ds.make_windows(tr[:1024], tt[:1024], X, Ds)
    print(f"  F_seq {Fq.shape}  (B, WINDOW={ds.WINDOW}, d={X.shape[2]})")
    print(f"  D_seq {Dq.shape}  (B, WINDOW, N_REL={ds.N_REL}, d)")
    check("F_seq [B,W,d]", Fq.shape == (1024, ds.WINDOW, X.shape[2]))
    check("D_seq [B,W,N_REL,d]", Dq.shape == (1024, ds.WINDOW, ds.N_REL, X.shape[2]))
    check("N_REL == 1 (hanya kontak akar)", ds.N_REL == 1)
    check("jendela hanya-masa-lalu (berakhir tepat di t)",
          bool(np.allclose(Fq[:, -1, :], X[tt[:1024], tr[:1024]])))

    # -----------------------------------------------------------------------
    hdr("7. PENJAGA KEBOCORAN — fitur jujur")
    u = g1_progeny_identity(X)
    check("G1 blok SELF+GENO hanya ditentukan oleh (sensus, progeny)",
          u == len(ds.progeny_levels()), f"baris unik per sensus = {u} (harus 14)")
    for h in ds.HORIZONS:
        m = g2_state_zero_on_risk(X, h)
        check(f"G2 blok STATE persis 0 di risk set (h={h})", m == 0.0, f"maks |nilai| = {m}")
    for k, ok in g3_past_only(ds.raw_features):
        check(f"G3 bangun-ulang dari panel terpotong di sensus {k} identik", ok)
    probe = g4_leak_probe(X, h=1)
    print("\n  G4 probe AUC satu-fitur (h=1), 6 teratas:")
    for nm, a in probe[:6]:
        print(f"    {nm:<16} {a:.3f}")
    check("G4 tidak ada kolom dengan AUC >= 0,90",
          probe[0][1] < 0.90, f"maks = {probe[0][0]} {probe[0][1]:.3f}")

    # lantai untuk Agen 3: seberapa jauh satu fitur waktu saja bisa membawa AUC-PR
    print("\n  Lantai AUC-PR fitur tunggal (untuk kalibrasi Agen 3):")
    for h in ds.HORIZONS:
        tr_, tt_, y_ = ds.build_examples(h)
        ap_t = average_precision_score(y_, X[tt_, tr_][:, names.index("t_years")])
        # oracle difusi: cacah tetangga bergejala pada t (hanya masa lalu)
        nb = ds.diffuse(ds.raw_features(), A_true, scale=False)[:, :, 0, k_sym]
        ap_n = average_precision_score(y_, nb[tt_, tr_])
        print(f"    h={h}: base-rate {100*y_.mean():.2f}% | t_years saja AP={ap_t:.4f} | "
              f"cacah tetangga bergejala saja AP={ap_n:.4f}")
    print("    -> keduanya jauh di atas base-rate, jadi baik MLP maupun jalur graf")
    print("       punya sinyal nyata; dekomposisi tidak akan mengukur derau belaka.")

    print("\n  PRATINJAU DEKOMPOSISI di tingkat FITUR (belum ada model sama sekali).")
    print("  AUC distratifikasi PER SENSUS: efek waktu dinetralkan, jadi yang tersisa")
    print("  hanyalah informasi lintas-pohon. Ini yang akan diperebutkan STGNN.")

    def strat_auc(score, tr_, tt_, y_):
        num = den = 0.0
        for t in np.unique(tt_):
            m = tt_ == t
            yy = y_[m]
            if yy.min() == yy.max():
                continue
            w = float(yy.sum() * (len(yy) - yy.sum()))
            num += w * roc_auc_score(yy, score[m])
            den += w
        return num / den if den else np.nan

    Fr_ = ds.raw_features()
    nb_true = ds.diffuse(Fr_, A_true, scale=False)[:, :, 0, :]
    nb_rand = ds.diffuse(Fr_, ds.adjacency("random", seed=0), scale=False)[:, :, 0, :]
    print(f"    {'h':>2} {'t_years':>9} {'deg':>9} {'nb_sympt(true)':>16} "
          f"{'nb_sympt(random)':>18} {'d_sympt(true)':>15}")
    for h in ds.HORIZONS:
        tr_, tt_, y_ = ds.build_examples(h)
        a_t = strat_auc(X[tt_, tr_][:, names.index("t_years")], tr_, tt_, y_)
        a_d = strat_auc(deg_true[tr_].astype(float), tr_, tt_, y_)
        a_nt = strat_auc(nb_true[tt_, tr_, k_sym], tr_, tt_, y_)
        a_nr = strat_auc(nb_rand[tt_, tr_, k_sym], tr_, tt_, y_)
        a_dl = strat_auc(nb_true[tt_, tr_, names.index("d_sympt")], tr_, tt_, y_)
        print(f"    {h:>2} {a_t:>9.3f} {a_d:>9.3f} {a_nt:>16.3f} {a_nr:>18.3f} {a_dl:>15.3f}")
    print("    t_years -> 0,500 persis: konstan di dalam sensus, benar secara konstruksi.")
    print("    Selisih nb_sympt(true) - nb_sympt(random) adalah pratinjau jujur")
    print("    komponen STRUKTUR. Kalau kecil di sini, jangan berharap model")
    print("    membesarkannya — laporkan apa adanya.")

    # -----------------------------------------------------------------------
    hdr("8. UJI BOCOR SENGAJA — apakah penjaga benar-benar menyala?")
    # (nama, fungsi, jumlah kolom SELF+GENO pada varian ini, harapan)
    # `n_sg` menentukan kolom mana yang diperiksa G1 dan mana yang diperiksa G2,
    # supaya jasa tiap penjaga tidak salah alamat.
    G = ds.GENO_SLICE.stop
    variants = (
        ("L1  y_t1s mentah (outcome statis)", leak_static_outcome, G + 1,
         "G1 menyala; G3 buta (kolom statis); G4 GAGAL menangkap — lihat catatan"),
        ("L1b y_t1s - t (sisa waktu ke kejadian)", leak_time_to_event, G + 1,
         "G1 dan G4 menyala; G3 tetap buta"),
        ("L2  status DIRI SENDIRI di t+1", leak_own_future_status, G,
         "G2 dan G3 menyala; G1 buta (bukan blok SELF/GENO)"),
        ("L3  insidensi masa depan tingkat FAMILI", leak_family_future_hazard, G + 1,
         "HANYA G3 yang menangkap; G1 dan G2 lolos, G4 di bawah ambang"),
    )
    for label, fn, n_sg, expect in variants:
        Xl = fn(ds.status_matrix(), ds.census()).astype(np.float32)
        nm = [f"c{i}" for i in range(Xl.shape[2])]
        g1_fire = g1_progeny_identity(Xl, n_selfgeno=n_sg) != len(ds.progeny_levels())
        g2_fire = g2_state_zero_on_risk(Xl, 1, state_start=n_sg) > 0.0
        g3_fire = not g3_past_only(fn, ks=(20,))[0][1]
        pr = g4_leak_probe(Xl, h=1, names=nm)
        g4_fire = pr[0][1] >= 0.90
        print(f"\n  {label}\n    harapan: {expect}")
        print(f"    G1 {'MENYALA' if g1_fire else 'diam   '} | "
              f"G2 {'MENYALA' if g2_fire else 'diam   '} | "
              f"G3 {'MENYALA' if g3_fire else 'diam   '} | "
              f"G4 {'MENYALA' if g4_fire else 'diam   '} (AUC maks {pr[0][1]:.3f} @ {pr[0][0]})")
        check(f"{label.split()[0]} ditangkap minimal satu penjaga FORMAL (G1/G2/G3)",
              g1_fire or g2_fire or g3_fire,
              f"G1/G2/G3 = {g1_fire}/{g2_fire}/{g3_fire}")

    print("\n  Batas kejujuran penjaga (dinyatakan, bukan disembunyikan):")
    print("    * G4 GAGAL pada L1. y_t1s mentah punya AUC gabungan hanya ~0,79 —")
    print("      lebih rendah dari umur — karena AUC dihitung menggabungkan semua")
    print("      sensus: pohon di sensus awal dengan y_t1s kecil adalah NEGATIF")
    print("      namun diperingkat tinggi. Kebocoran telanjang bisa terlihat jinak")
    print("      di probe gabungan. Itulah sebabnya G1 (struktural) yang memutus,")
    print("      bukan G4. Jangan pernah memakai probe AUC sebagai satu-satunya uji.")
    print("    * G1 hanya menjaga blok SELF+GENO; kolom per-pohon yang diselipkan")
    print("      ke blok STATE lolos G1 dan harus ditangkap G2.")
    print("    * G3 hanya menangkap kebocoran yang bergantung pada baris panel")
    print("      SETELAH sensus t. Kolom statis dari nodes.csv (y_t1s) tidak berubah")
    print("      saat panel dipotong -> G3 buta; itu wilayah G1.")
    print("    * TIDAK ADA penjaga yang menangkap kebocoran yang sekaligus statis")
    print("      DAN konstan di dalam famili. Pertahanan satu-satunya bersifat")
    print("      struktural: raw_features() hanya membaca census, status, progeny —")
    print("      tidak pernah menyentuh event_t1s/y_t1s/event_td/y_td.")

    # -----------------------------------------------------------------------
    hdr("RINGKASAN")
    if _fails:
        print(f"  {len(_fails)} pemeriksaan GAGAL:")
        for f in _fails:
            print(f"    - {f}")
        sys.exit(1)
    print("  Semua pemeriksaan lulus.")


if __name__ == "__main__":
    main()
