"""Logika demo, TANPA Streamlit — supaya bisa diuji tanpa menjalankan server.

`demo_app.py` hanya menggambar; seluruh perhitungan ada di sini. Pemisahan ini
disengaja: kalau angka di layar salah, yang diperiksa satu berkas ini, dan ia bisa
dijalankan langsung:

    python demo_core.py        # uji mandiri, mencetak semua angka layar
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
L1 = os.path.join(ROOT, "layer1_build")
L2 = os.path.join(ROOT, "layer2_real")
for p in (L1, L2):          # L2 dibutuhkan untuk models_real (checkpoint v3-foto)
    if p not in sys.path:
        sys.path.insert(0, p)

RISK_CSV = os.path.join(L2, "risk_ranked.csv")
CENTRE_JSON = os.path.join(L1, "yolo12_results", "centre_eval.json")

# --- Angka tetap yang boleh ditampilkan. Sumber tunggal, jangan ditulis ulang
#     di demo_app.py. Semua berasal dari centre_eval.json / 00_RINGKASAN.csv.
FACTS = {
    "f1": (0.960, 0.024),
    "precision": (0.950, 0.019),
    "recall": (0.971, 0.030),
    "rmse_frac": (0.071, 0.011),
    "deg_l1_pred": (5.54, 0.12),
    "deg_l1_gt": (5.62, 0.05),
    "deg_l2": 5.74,
    "gap_pct": 3.5,
    "n_censuses": 45,
    "n_years": 25,
    "n_palms": 1200,
}

# --- Angka Lapisan 2 yang boleh ditampilkan. Semua dari results_v3*.csv dan
#     00_RINGKASAN.csv; jangan ditulis ulang di tempat lain.
V3_FACTS = {
    "ap_full_pooled": (0.1818, 0.0077),
    "ap_full_within": (0.0973, 0.0107),
    "ap_photo_within": (0.1015, 0.0079),
    "ap_nograph_within": (0.0632, 0.0031),
    "ap_1col_within": (0.0916, 0.0081),
    "ap_1col_noisy": (0.0800, 0.0070),
    "struktur": (0.0296, 0.0057, "40/40"),
    "perm_progeny": (0.0268, 6.25, "0/200"),
    "perm_strict": (0.0244, 6.04, "0/200"),
    "kinship_pct": 36,
    "signal_kept_pct": 59,
    "lift_top5": 1.81,
    "per_case_model": 8.8,
    "per_case_random": 15.8,
}


# --- Palet: token SawitGuard Design System (claude.ai/design), diverifikasi ulang.
#
# Grafik duduk di atas KARTU PUTIH (--surface-card), bukan latar mint aplikasi,
# jadi semua validasi dijalankan terhadap #FFFFFF.
#
# YANG DIPAKAI APA ADANYA
#   ink, surface, border, dan hijau/merah merek.
#   Deteksi 2 seri #4EC75B + #E5484D: CVD dE 9,9 (deutan) · normal 34,9 -> LULUS.
#   (#4EC75B berkontras 2,18:1, di bawah 3:1 - aturan relief terpenuhi oleh
#    legenda berlabel dan tampilan tabel di layar yang sama.)
#
# YANG DIGANTI, DAN MENGAPA
#   Ramp --risk-1..6 bawaan design system GAGAL empat cek sekaligus: kecerahan
#   tidak monoton (0,74 - 0,80 - 0,83 - 0,73 - 0,63 - 0,51), langkah lime->amber
#   hanya berjarak 0,028, ujung terangnya 1,70:1, dan rentang rona 122 derajat
#   (rainbow tidak sah untuk skala berurutan). Yang paling berbahaya: lime lawan
#   amber berjarak dE 1,7 pada protanopia - dua tingkat risiko yang praktis
#   warna yang sama bagi ~8% pembaca pria, di peta yang seluruh pesannya warna.
#   Penggantinya satu-warna, memakai --risk-5 dan --risk-6 merek sebagai langkah
#   tengah, dan lolos keempat cek. Merah tetap berarti bahaya.
#
#   Tiga seri sekaligus TIDAK mungkin dari palet ini: lime jatuh di luar pita
#   kecerahan dan hijau-tua lawan merah gagal protan (dE 5,0). Karena itu grafik
#   antarmuka memakai satu warna merek untuk pengukuran hidup, dan menggambar
#   kedua acuannya sebagai chrome ink - dibedakan BENTUK dan label, bukan warna.
PALETTE = {
    "surface": "#FFFFFF",           # --surface-card
    "app_bg": "#EDF2E9",            # --mint-50
    "ink": "#1A1C17",               # --ink-900
    "ink_2": "#3D423A",             # --ink-700
    "muted": "#6A7065",             # --ink-500
    "grid": "#D4D9CF",              # --ink-200
    "series_1": "#4EC75B",          # --green-500  Healthy / pengukuran hidup
    "series_2": "#E5484D",          # --risk-5     Unhealthy
    "node": "#27803A",              # --green-700  simpul graf (seri tunggal)
    "quintile": ["#EE9A87", "#E5484D", "#B32B30", "#822024", "#4F1315"],
}

# Blok fitur checkpoint Lapisan 2 — rincian teknis, disimpan di balik expander.
FEATURE_BLOCKS = [
    ("SELF — waktu/umur", 4, False, "butuh sumbu waktu; satu foto hanya satu tanggal"),
    ("GENO — genotipe", 14, False, "tidak terlihat dari udara sama sekali"),
    ("STATE — status tetangga", 6, True, "sebagian bisa dari deteksi, tapi penyakitnya beda"),
]


def readiness(info):
    """Daftar periksa kesiapan, DIHITUNG dari citra yang barusan diproses.

    MENGAPA INI MENGGANTIKAN TABEL 24-KOLOM DI LAYAR.

    "Model meminta 24 kolom, foto mengisi 6" itu benar tetapi tidak menjawab apa
    pun yang baru saja dilakukan pengguna - ia ceramah, bukan tanggapan. Yang
    berguna: dari fotomu, bahan mana yang SUDAH ada dan mana yang kurang, dengan
    angka dari fotomu sendiri. Rinciannya tetap tersedia di balik expander untuk
    pembaca teknis.

    Angka 6-dari-24 juga terlalu murah hati. Dari SATU foto, blok STATE hanya
    terisi sebagian (gejala dan mati bisa ditebak; selisihnya butuh dua tanggal),
    dan itu pun sebagai kesehatan tajuk umum - bukan Ganoderma terverifikasi.
    Daftar bahan di bawah menghindari klaim itu.
    """
    n = info.get("n", 0) if info else 0
    ok_graph = bool(info and info.get("ok_n") and info.get("ok_scale"))
    n_sick = int(info.get("n_sympt", 0)) if info else 0
    return [
        {"bahan": "Daftar pohon + posisinya", "ada": n > 0,
         "punyamu": "%d pohon terdeteksi" % n if n else "belum ada citra diproses",
         "cara": "otomatis dari foto drone"},
        # CATATAN: grafnya SELALU dibangun begitu ada >= 20 pohon (`ok_n`), karena
        # radiusnya diturunkan dari jarak tanam yang TERUKUR di citra itu sendiri -
        # jadi ia menyesuaikan diri terhadap skala. Yang dijaga `ok_scale` adalah
        # apakah DETEKTOR masih berada di dalam distribusi latihnya, bukan apakah
        # grafnya benar. Pesan lama ("graf tidak dibangun") keliru di dua sisi
        # sekaligus: grafnya dibangun, dan yang berisiko bukan grafnya.
        {"bahan": "Peta kontak antar-pohon",
         "ada": bool(info and info.get("ok_n")),
         "punyamu": ("derajat %.2f, %d pohon bagian dalam%s"
                     % (info["deg_inner"], info["n_inner"],
                        "" if ok_graph else
                        " — TAPI skala citra di luar jangkauan latih detektor,"
                        " jadi daftar pohonnya yang belum tentu lengkap"))
                    if (info and info.get("ok_n"))
                    else "kurang dari 20 pohon terdeteksi, jarak tanam tak terukur",
         "cara": "otomatis dari foto drone"},
        {"bahan": "Tajuk bergejala sebagai sumber", "ada": n_sick > 0,
         "punyamu": ("%d tajuk terdeteksi tidak sehat" % n_sick if n_sick
                     else "tidak ada — tanpa sumber, seluruh skor identik"),
         "cara": "otomatis dari foto drone (kelas Unhealthy detektor)"},
        {"bahan": "Riwayat kunjungan + genotipe", "ada": False,
         "punyamu": "belum ada — TIDAK dibutuhkan untuk peringkat dalam satu bidikan",
         "cara": "hanya perlu kalau ingin lintasan 25 tahun seperti Eg9PP"},
    ]


def photo_checks(info):
    """Pemeriksaan syarat foto, DIHITUNG dari citra yang barusan diproses.

    MENGAPA INI BUKAN DAFTAR SYARAT DI LAYAR UNGGAH.

    Syarat yang ditulis di muka adalah ceramah: pengguna belum punya konteks untuk
    menilainya, dan empat kartu teks membuat layar pertama terasa seperti formulir.
    Yang berguna adalah pemeriksaan SESUDAH unggah, dengan angka dari fotonya
    sendiri, dan hanya muncul kalau ada yang tidak lolos.

    Tiap butir memuat `saran` yang bisa langsung dikerjakan - termasuk faktor
    perbesaran yang sudah dihitung, bukan "sesuaikan skala".

    `syarat_foto` memisahkan dua hal yang mudah tertukar. Jumlah sawit dan skala
    adalah SYARAT: fotonya memang belum bisa dibaca dengan benar. Ketiadaan tajuk
    bergejala BUKAN syarat - itu petak yang sehat, dan pada laju Unhealthy 1,3%
    sebuah ubin 1024 memang diharapkan hanya memuat ~0,85 pohon sakit. Memunculkan
    dialog peringatan untuk itu akan menuduh mayoritas foto normal sebagai cacat.

    -> list of dict. Dialog hanya ditampilkan bila ada butir `syarat_foto=True`
       yang `ok=False`; butir informatif ikut ditampilkan kalau dialognya terlanjur
       terbuka, tetapi tidak pernah memicunya sendiri.
    """
    import numpy as _np

    import detect_centres as dc

    n = int(info.get("n", 0)) if info else 0
    ok_n = bool(info and info.get("ok_n"))
    sp = info.get("spacing_px") if info else None
    ratio = info.get("scale_ratio") if info else None
    target = float(_np.mean(dc.SPACING_REF))
    lo, hi = dc.SCALE_WINDOW[0] * target, dc.SCALE_WINDOW[1] * target

    out = [{
        "judul": "Jumlah sawit",
        "syarat_foto": True,
        "ok": ok_n,
        "punyamu": "%d sawit terdeteksi" % n,
        "syarat": "minimal %d" % dc.MIN_TREES_FOR_SPACING,
        "saran": ("Pakai potongan foto yang lebih luas. Satu potongan sekitar "
                  "1.000 × 1.000 piksel biasanya memuat ~65 sawit."),
    }]

    if ok_n and sp and _np.isfinite(sp):
        ok_s = bool(info.get("ok_scale"))
        f = target / float(sp)
        saran = ("Ubah ukuran foto: %s sekitar %.1f×."
                 % (("perbesar", f) if f > 1 else ("perkecil", 1.0 / f)))
        out.append({
            "judul": "Skala foto",
            "syarat_foto": True,
            "ok": ok_s,
            "punyamu": "jarak antar sawit %.0f piksel (%.2f× acuan)" % (sp, ratio),
            "syarat": "%.0f–%.0f piksel" % (lo, hi),
            "saran": saran + (" Di luar rentang ini detektor bekerja di luar kondisi "
                              "latihnya, jadi sebagian sawit bisa terlewat."),
        })

    n_s = int(info.get("n_sympt", 0)) if info else 0
    out.append({
        "judul": "Sawit bergejala",
        "syarat_foto": False,          # informatif; tidak pernah memicu dialog
        "ok": n_s > 0,
        "punyamu": "%d tajuk tidak sehat" % n_s,
        "syarat": "minimal 1",
        # Ini BUKAN foto yang buruk - ini laju dasar. Ubin 1024 memuat ~65 sawit;
        # pada laju Unhealthy 1,3% ia memang diharapkan memuat ~0,85 pohon sakit.
        "saran": ("Bukan galat: petak yang sehat memang begitu. Tanpa sawit sakit "
                  "tidak ada sumber penularan, jadi seluruh sawit mendapat peringkat "
                  "yang sama dan daftar prioritas tidak ditampilkan."),
    })
    return out


def load_risk():
    """-> DataFrame 1.200 sawit. Kolom penting: xm, ym, in_risk_set, risk_decile."""
    if not os.path.isfile(RISK_CSV):
        raise SystemExit(
            "%s tidak ada.\n  jalankan dulu:  python layer2_real/export_risk.py" % RISK_CSV)
    df = pd.read_csv(RISK_CSV)
    # `plot` HARUS diakses lewat kurung siku. Pada Series, `.plot` menjadi accessor
    # plotting pandas, bukan kolom bernama 'plot'.
    df["plot"] = df["plot"].astype(str)
    return df


def risk_summary(df):
    """Angka funnel untuk layar 4: 1.200 -> 672, plus bukti 'model membaca tetangga'."""
    risk = df[df.in_risk_set == 1]
    out = df[df.in_risk_set == 0]
    top10 = risk.nsmallest(10, "rank")
    bot10 = risk.nlargest(10, "rank")
    return {
        "n_total": len(df),
        "n_risk": len(risk),
        "n_out": len(out),
        "status_out": out.status.value_counts().to_dict(),
        "sick_nb_top10": float(top10.n_sick_neighbours.mean()),
        "sick_nb_all": float(risk.n_sick_neighbours.mean()),
        "sick_nb_bot10": float(bot10.n_sick_neighbours.mean()),
        "logit_min": float(risk.logit.min()),
        "logit_max": float(risk.logit.max()),
        "top10": top10[["rank", "palm_id", "parcel", "logit",
                        "risk_percentile", "n_sick_neighbours", "n_neighbours"]],
    }


def detect_image(path, fold="fold0", conf=0.75, imgsz=640):
    """Jalankan detektor ds_B pada SATU citra. -> (df, info).

    Membungkus `detect_centres.py` supaya UI dan CLI tidak pernah berbeda hasil.
    """
    import detect_centres as dc

    w = dc.weights_for(fold)
    if w is None:
        raise SystemExit("bobot %s tidak ada — jalankan train_folds_gpu.py" % fold)

    import torch
    from ultralytics import YOLO

    dev = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(w)
    det, _ = dc.detect(model, [path], conf, imgsz, 1, dev, stitch=False)
    if not det:
        return pd.DataFrame(columns=["cx", "cy", "conf", "cls", "deg"]), {
            "n": 0, "ok_scale": False, "note": "tidak ada deteksi di atas conf %.2f" % conf}

    box = float(np.median([d[6] for d in det]))
    det, scale = dc.merge_two_pass(det, box)
    xy = np.array([[d[0], d[1]] for d in det], float)
    spacing = dc.spacing_of(xy)

    ok_n = len(det) >= dc.MIN_TREES_FOR_SPACING
    ratio = spacing / np.mean(dc.SPACING_REF) if np.isfinite(spacing) else float("nan")
    ok_scale = bool(ok_n and dc.SCALE_WINDOW[0] <= ratio <= dc.SCALE_WINDOW[1])

    deg = np.full(len(det), np.nan)
    d_all = d_in = float("nan")
    n_in = 0
    if ok_n:
        deg, d_all, d_in, n_in = dc.degrees(xy, dc.R_GRAPH * spacing)

    # --- skor gejala KONTINU, dari lintasan kedua pada conf rendah -------------
    #
    # Lokalisasi TETAP memakai conf 0,75 (yang tervalidasi, F1 pusat 0,960).
    # Lintasan kedua pada conf 0,10 hanya dipakai untuk memberi tiap pohon sebuah
    # skor Unhealthy lunak: keyakinan TERTINGGI di antara deteksi Unhealthy yang
    # jatuh pada pohon itu. Prosedur yang sama dengan
    # `layer1_build/unhealthy_threshold.py::per_tree_scores`.
    #
    # BELUM TERVALIDASI. Model dilatih pada `is_sympt` Eg9PP yang BINER (status S/D
    # terverifikasi lapangan). Memberinya 0,37 adalah masukan di luar distribusi
    # latih. Peringkatnya masuk akal karena modelnya monoton, tetapi tidak ada
    # ground truth kontinu di Eg9PP untuk menguji apakah gradasi itu LEBIH BENAR.
    # UI wajib menandainya; `score_photo(mode=...)` menyimpan keduanya.
    from scipy.spatial import cKDTree as _KD
    soft = np.zeros(len(det), float)
    try:
        det_lo, _ = dc.detect(model, [path], 0.10, imgsz, 1, dev, stitch=False)
        if det_lo:
            lo_xy = np.array([[d[0], d[1]] for d in det_lo], float)
            lo_unh = np.array([d[2] if d[3] == 1 else 0.0 for d in det_lo], float)
            _, idx = _KD(xy).query(lo_xy, k=1)
            for i, v in zip(idx, lo_unh):
                if v > soft[i]:
                    soft[i] = v
    except Exception:                      # lintasan lunak opsional; jangan jatuhkan demo
        pass

    df = pd.DataFrame({
        "cx": xy[:, 0], "cy": xy[:, 1],
        "conf": [d[2] for d in det],
        "cls": [dc.NAMES_OF(d[3]) if hasattr(dc, "NAMES_OF") else
                __import__("y12").NAMES[d[3]] for d in det],
        "unh": soft,
        "deg": deg,
    })
    info = {
        "n": len(det), "box_px": box, "spacing_px": spacing, "scale_ratio": ratio,
        "ok_scale": ok_scale, "ok_n": ok_n, "deg_all": d_all, "deg_inner": d_in,
        "n_inner": n_in, "r_graph_px": dc.R_GRAPH * spacing if ok_n else float("nan"),
        "weights": os.path.relpath(w, ROOT), "fold": fold, "conf": conf,
        "n_sympt": int((df.cls == "Unhealthy").sum()),
    }
    return df, info


def edges_within(xy, r):
    """Pasangan (i, j) berjarak <= r. Untuk menggambar graf; dibatasi agar UI ringan."""
    from scipy.spatial import cKDTree

    if len(xy) < 2 or not np.isfinite(r):
        return []
    kd = cKDTree(xy)
    return [(i, j) for i, j in kd.query_pairs(r)]


def sample_images(n=6):
    """Ubin contoh siap-klik. Yang MENGANDUNG tajuk tidak-sehat didahulukan.

    MENGAPA URUTANNYA PENTING. ds_B hanya punya 66 pohon `Unhealthy` unik dari
    5.077, jadi ubin yang diambil sembarang hampir pasti tidak memuat satu pun.
    Pada ubin seperti itu difusi graf nol di mana-mana, seluruh skor identik, dan
    peringkatnya tidak berarti apa-apa - `score_photo()` menandainya `degenerate`.
    Itu perilaku yang BENAR, tetapi contoh bawaan yang selalu degenerate membuat
    demo tampak rusak. Jadi ubin ber-gejala didahulukan, dan ubin bersih tetap
    disertakan supaya kasus "tidak ada gejala" bisa diperagakan juga.
    """
    import csv as _csv
    import glob

    crowns = os.path.join(ROOT, "data_clean", "layer1_crowns.csv")
    sick_tiles, clean = [], []
    if os.path.isfile(crowns):
        seen = set()
        for r in _csv.DictReader(open(crowns, encoding="utf-8")):
            if r["label"] == "Unhealthy" and r["tile"] not in seen:
                seen.add(r["tile"])
                p = os.path.join(L1, "ds_B", "train", r["tile"])
                if os.path.isfile(p):
                    sick_tiles.append(p)
    for ortho in ("44000_16000", "44000_4000", "52000_20000"):
        g = sorted(glob.glob(os.path.join(L1, "ds_B", "train", ortho + "_*.jpg")))
        clean += g[:1]
    out = sick_tiles[:max(1, n - 2)] + [p for p in clean if p not in sick_tiles]
    return out[:n] if out else clean[:n]


V3_CKPT = os.path.join(L2, "stgnn_v3_photo.pt")


def score_photo(df, info, mode="biner"):
    """Peringkat risiko UNTUK CITRA YANG DIUNGGAH, memakai checkpoint v3-foto.

    Inilah yang membuat demo ini benar-benar unggah -> proses -> hasil. Checkpoint
    penuh (`stgnn_final.pt`) TIDAK BISA dipakai di sini: ia meminta 24 kolom, 18 di
    antaranya mustahil dari satu foto, dan mengisinya nol dilarang. `stgnn_v3_photo.pt`
    dilatih pada SATU kolom - `is_sympt` - yaitu persis yang detektor keluarkan.

    Set berisiko = tajuk yang terdeteksi SEHAT. Tajuk `Unhealthy` tidak diberi skor;
    ia sudah bergejala, jadi ia SUMBER, bukan sasaran - cermin aturan Eg9PP bahwa
    hanya pohon berstatus 'A' yang masuk risk set.

    Skala adjacency dihitung ulang dari graf foto itu sendiri (1 / derajat rata-rata),
    sama seperti saat latih, sehingga difusinya berarti "fraksi tetangga yang sakit"
    di kedua sisi.
    """
    import torch
    from scipy.spatial import cKDTree

    import models_real as M

    if not os.path.isfile(V3_CKPT):
        raise SystemExit("%s tidak ada — jalankan layer2_real/train_final_v3.py" % V3_CKPT)
    if not info.get("ok_n") or df.empty:
        return None

    xy = df[["cx", "cy"]].values.astype(np.float64)
    n = len(xy)
    r = R_GRAPH_MULT * info["spacing_px"]
    A = np.zeros((n, n), np.float32)
    for i, j in cKDTree(xy).query_pairs(r):
        A[i, j] = A[j, i] = 1.0
    deg = A.sum(1)
    scale = 1.0 / (float(deg.mean()) + 1e-8)

    if mode == "kontinu" and "unh" in df:
        sympt = df.unh.values.astype(np.float32)
    else:
        sympt = (df.cls.values == "Unhealthy").astype(np.float32)
    hard = (df.cls.values == "Unhealthy")
    F = torch.as_tensor(sympt.reshape(1, n, 1))                 # (T=1, N, d=1)
    D = torch.einsum("ij,tjd->tid", torch.as_tensor(A * scale), F).unsqueeze(2)

    try:
        ck = torch.load(V3_CKPT, map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(V3_CKPT, map_location="cpu", weights_only=False)
    model = M.build(ck["model_class"], ck["arch"]["in_dim"], horizon=ck["task"]["horizon"])
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()

    risk_idx = np.flatnonzero(~hard)              # risk set SELALU dari kelas keras,
    #                                            supaya kedua mode menilai pohon yang sama
    if risk_idx.size == 0:
        return None
    ii = torch.as_tensor(risk_idx, dtype=torch.long)
    with torch.no_grad():
        logit = model(F[0, ii].unsqueeze(1), D[0, ii].unsqueeze(1)).numpy()

    out = pd.DataFrame({
        "idx": risk_idx,
        "cx": xy[risk_idx, 0], "cy": xy[risk_idx, 1],
        "skor": logit,
        "tetangga": deg[risk_idx].astype(int),
        # SELALU dari kelas keras, di kedua mode. Di mode kontinu `sympt` berisi
        # keyakinan pecahan, sehingga A @ sympt menghasilkan jumlah KEYAKINAN
        # (maks 0,91 pada ubin khas) - dan astype(int) memotongnya jadi 0, membuat
        # seluruh kolom nol padahal tetangga sakitnya ada. Label kolom ini
        # menjanjikan HITUNGAN POHON, jadi itu yang harus dihitung.
        "tetangga_sakit": (A[risk_idx] @ hard.astype(np.float32)).round().astype(int),
    }).sort_values("skor", ascending=False).reset_index(drop=True)
    out.insert(0, "peringkat", np.arange(1, len(out) + 1))
    # Pita diwarnai menurut SKOR YANG BENAR-BENAR BERBEDA, bukan posisi urutan.
    #
    # MENGAPA. Model foto menerima satu masukan efektif: jumlah tetangga sakit
    # dibagi derajat rata-rata. Skornya karena itu fungsi deterministik dari satu
    # bilangan bulat - pohon dengan hitungan tetangga sakit yang sama mendapat skor
    # IDENTIK. Pada ubin khas dengan 0-1 pohon sakit, seluruh 86 pohon hanya punya
    # DUA skor berbeda.
    #
    # Membelahnya jadi lima kuintil menurut posisi urutan akan menampilkan lima pita
    # warna padahal model membedakan dua kelompok, dan urutan di dalam kelompok yang
    # seri itu sembarang. Itu memalsukan presisi yang tidak ada. Jadi jumlah pita =
    # jumlah tingkat yang sungguh-sungguh dibedakan model, dan UI menyebut angkanya.
    lev = out.skor.rank(method="dense", ascending=True).astype(int)   # 1 = paling aman
    n_lev = int(lev.max())
    out["tingkat"] = lev
    out["n_tingkat"] = n_lev
    # Petakan ke ujung-ujung ramp supaya dua tingkat tetap terbedakan jelas.
    span = np.linspace(1, 5, n_lev) if n_lev > 1 else np.array([5.0])
    out["kuintil"] = [int(round(span[v - 1])) for v in lev]
    # Kalau TIDAK ADA tajuk bergejala di foto, difusinya nol di mana-mana dan
    # seluruh skor identik. Peringkat dalam keadaan itu tidak berarti apa pun, dan
    # UI wajib mengatakannya alih-alih menampilkan urutan yang sebetulnya acak.
    degenerate = bool(hard.sum() == 0) or float(np.ptp(logit)) < 1e-9
    return {"tabel": out, "n_sumber": int(hard.sum()), "n_risk": len(out),
            "mode": mode,
            "degenerate": degenerate,
            "sumber_xy": xy[np.flatnonzero(hard)],
            "n_tingkat": int(out.tingkat.max()),
            "scope_warning": ck["scope_warning"], "ckpt": os.path.basename(V3_CKPT)}


R_GRAPH_MULT = 1.5


def outbreak_foci(df, info):
    """Kelompokkan tajuk bergejala jadi PUSAT WABAH. Tanpa model, tanpa klaim baru.

    Dua tajuk bergejala masuk pusat yang sama bila terhubung lewat graf kontak -
    yaitu komponen terhubung dari subgraf yang hanya berisi pohon bergejala. Itu
    murni pernyataan geometris: "gejala-gejala ini bersambung, yang itu terpisah".

    KENAPA INI BERGUNA JUSTRU KARENA TIDAK MEMAKAI MODEL.
    Seluruh angka model di paket ini punya batas yang harus ikut disebut. Keluaran
    di sini tidak: ia tidak meramal apa pun, tidak memeringkat apa pun, dan tidak
    bisa salah kecuali detektornya salah. Untuk mandor ia menjawab pertanyaan
    pertama yang sebenarnya ditanyakan - "wabahnya ada berapa titik, di mana?"

    Yang dilaporkan per pusat: jumlah tajuk bergejala, jumlah tetangga sehat yang
    bersentuhan langsung dengannya (itu yang perlu diperiksa duluan), dan titik
    tengahnya.
    """
    from scipy.spatial import cKDTree

    if df is None or df.empty or not info.get("ok_n"):
        return None
    xy = df[["cx", "cy"]].values.astype(float)
    sick = (df.cls.values == "Unhealthy")
    if sick.sum() == 0:
        return {"n_fokus": 0, "fokus": [], "n_sakit": 0, "n_terpapar": 0}

    r = R_GRAPH_MULT * info["spacing_px"]
    si = np.flatnonzero(sick)
    kd = cKDTree(xy[si])
    # union-find sederhana atas pasangan bergejala yang saling menjangkau
    parent = list(range(len(si)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in kd.query_pairs(r):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups = {}
    for k in range(len(si)):
        groups.setdefault(find(k), []).append(k)

    kd_all = cKDTree(xy)
    fokus, terpapar = [], set()
    for g in sorted(groups.values(), key=len, reverse=True):
        idx = si[g]
        nb = set()
        for i in idx:
            nb.update(j for j in kd_all.query_ball_point(xy[i], r) if not sick[j])
        terpapar |= nb
        fokus.append({"n_sakit": len(idx), "n_terpapar": len(nb),
                      "cx": float(xy[idx, 0].mean()), "cy": float(xy[idx, 1].mean()),
                      "sakit_xy": xy[idx].tolist()})
    return {"n_fokus": len(fokus), "fokus": fokus,
            "n_sakit": int(sick.sum()), "n_terpapar": len(terpapar)}


def eg9pp_payload():
    """Data layar bukti: kisi Eg9PP + angka yang divalidasi di sana.

    Layar ini ada karena jalur foto MENYEMBUNYIKAN nilai grafnya. Pada satu
    bidikan model hanya menerima satu bilangan bulat, jadi peringkatnya identik
    dengan menghitung tetangga sakit (Spearman +1,000). Di Eg9PP graf membawa
    enam kolom melintasi tiga sensus, dan di situlah jaringan terlatih benar-benar
    menimbang sesuatu. Tanpa layar ini penonton akan menilai model dari kasus
    terlemahnya.
    """
    df = load_risk()
    s = risk_summary(df)
    x0, x1 = float(df.xm.min()), float(df.xm.max())
    y0, y1 = float(df.ym.min()), float(df.ym.max())
    sx = lambda v: (float(v) - x0) / max(1e-9, x1 - x0)
    sy = lambda v: (float(v) - y0) / max(1e-9, y1 - y0)

    pts = []
    for r in df.itertuples():
        if r.in_risk_set == 1:
            q = int(np.ceil(r.risk_decile / 2))
            pts.append({"x": sx(r.xm), "y": sy(r.ym), "q": max(1, min(5, q)),
                        "nb": int(r.n_sick_neighbours)})
        else:
            pts.append({"x": sx(r.xm), "y": sy(r.ym), "st": r.status})

    risk = df[df.in_risk_set == 1]
    vc = risk.n_sick_neighbours.value_counts().sort_index()
    return {
        "points": pts, "aspect": (y1 - y0) / max(1e-9, x1 - x0),
        "n_total": s["n_total"], "n_risk": s["n_risk"], "n_out": s["n_out"],
        "status_out": s["status_out"],
        "sick_rate": float(df.status.isin(["S", "D"]).mean()),
        "levels": [{"nb": int(k), "n": int(v)} for k, v in vc.items()],
        # `pct` = "berada di X% teratas". Dihitung dari peringkat, bukan dari logit:
        # skala logit dua model berbeda tidak sebanding, sedangkan persentil bermakna
        # sama di mana pun. Layar hanya menampilkan `pct`; `skor` tetap dikirim untuk
        # pembaca teknis di balik expander.
        "top10": [{"rank": int(r.rank), "id": r.palm_id, "parcel": r.parcel,
                   "skor": round(float(r.logit), 4),
                   "pct": round(100.0 * int(r.rank) / max(1, s["n_risk"]), 1),
                   "nb_sick": int(r.n_sick_neighbours),
                   "nb": int(r.n_neighbours)} for r in s["top10"].itertuples()],
        "nb_top10": s["sick_nb_top10"], "nb_all": s["sick_nb_all"],
        "nb_bot10": s["sick_nb_bot10"],
        "facts": V3_FACTS,
    }


# ---------------------------------------------------------------- gambar
def _num(x, d=2):
    """Koma sebagai pemisah desimal - seluruh paket ini berbahasa Indonesia."""
    return ("%.*f" % (d, x)).replace(".", ",")


def _fig(w=7.0, h=4.2, axes=True):
    """`axes=False` untuk peta/graf: tidak ada sumbu yang berarti di sana, jadi
    garis sumbunya cuma derau."""
    import matplotlib.pyplot as plt

    f, ax = plt.subplots(figsize=(w, h))
    f.patch.set_facecolor(PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        if axes:
            ax.spines[s].set_color(PALETTE["grid"])
        else:
            ax.spines[s].set_visible(False)
    ax.tick_params(colors=PALETTE["muted"], labelsize=9)
    return f, ax


def fig_detection(img, df):
    """Citra + pusat tajuk. Dua seri -> legenda wajib ada."""
    h = 7.0 * img.shape[0] / img.shape[1]
    f, ax = _fig(7.0, h, axes=False)
    ax.imshow(img)
    for cls, col in (("Healthy", PALETTE["series_1"]),
                     ("Unhealthy", PALETTE["series_2"])):
        m = df.cls == cls
        if m.any():
            # Cincin permukaan: penanda ini duduk di atas FOTO, bukan permukaan
            # bervalidasi. Kontras terhadap dedaunan tidak bisa dijamin oleh warna.
            ax.scatter(df.cx[m], df.cy[m], s=46, c=col, edgecolors="white",
                       linewidths=1.4, zorder=3,
                       label="%s (%d)" % (cls, int(m.sum())))
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", frameon=True, facecolor=PALETTE["surface"],
              edgecolor=PALETTE["grid"], fontsize=9, labelcolor=PALETTE["ink_2"])
    return f


def fig_graph(xy, edges):
    """Simpul + sisi. SATU seri -> tanpa legenda; judul di UI yang menamainya."""
    span_x = max(1.0, float(xy[:, 0].max() - xy[:, 0].min()))
    span_y = max(1.0, float(xy[:, 1].max() - xy[:, 1].min()))
    f, ax = _fig(7.0, max(2.6, 7.0 * span_y / span_x), axes=False)
    for i, j in edges:
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color=PALETTE["grid"],
                lw=2, zorder=1, solid_capstyle="round")
    ax.scatter(xy[:, 0], xy[:, 1], s=52, c=PALETTE["node"],
               edgecolors=PALETTE["surface"], linewidths=1.5, zorder=2)
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    return f


def fig_interface(user_deg=None, user_n=0):
    """Dot plot dengan pita ketidakpastian - BUKAN batang.

    Yang harus terbaca justru dua hal yang disembunyikan batang: lebar pita ±0,12,
    dan fakta bahwa 5,74 jatuh DI LUAR pita itu. Dua batang setinggi 5,54 dan 5,74
    akan terlihat sama tinggi dan menyiratkan "keduanya sama", yang tidak benar.

    `user_deg` menempatkan CITRA YANG BARU DIUNGGAH pada sumbu yang sama. Tanpa ini
    layar terasa lepas dari masukan pengguna - ia melihat foto kebunnya lalu disodori
    dua angka yang tidak ada hubungannya dengan foto itu. Titik ketiga inilah yang
    menjadikan halaman ini pengukuran atas fotonya sendiri, bukan poster.
    """
    l1m, l1s = FACTS["deg_l1_pred"]
    l2 = FACTS["deg_l2"]
    has_u = user_deg is not None and np.isfinite(user_deg)
    rows = 3 if has_u else 2
    f, ax = _fig(7.0, 2.9 if not has_u else 3.5)

    y_u, y_1, y_2 = (2, 1, 0) if has_u else (None, 1, 0)
    ax.errorbar(l1m, y_1, xerr=l1s, fmt="o", ms=12, color=PALETTE["ink_2"],
                ecolor=PALETTE["ink_2"], elinewidth=2.5, capsize=6, zorder=3)
    ax.plot(l2, y_2, "s", ms=11, color=PALETTE["ink_2"], zorder=3)
    # Label langsung untuk SEMUA titik: identitas tidak boleh bergantung warna saja,
    # dan slot aqua di bawah ini berkontras 2,74:1 - aturan relief mewajibkan label.
    ax.text(l1m, y_1 + 0.30, "Acuan Lapisan 1 · %s ± %s (3 ortomosaik)"
            % (_num(l1m), _num(l1s, 2)), ha="center", fontsize=10, color=PALETTE["ink"])
    ax.text(l2, y_2 + 0.32, "Kebun Eg9PP · %s" % _num(l2), ha="center", fontsize=10,
            color=PALETTE["ink"])
    xs = [l1m - l1s, l1m + l1s, l2]
    if has_u:
        ax.plot(user_deg, y_u, "D", ms=13, color=PALETTE["series_1"],
                markeredgecolor=PALETTE["ink"], markeredgewidth=1.0, zorder=3)
        ax.text(user_deg, y_u + 0.30, "Citra Anda · %s  (%d pohon dalam)"
                % (_num(user_deg), user_n), ha="center", fontsize=10,
                color=PALETTE["ink"])
        xs.append(user_deg)

    lo, hi = min(xs) - 0.35, max(xs) + 0.35
    ax.set_yticks([])
    ax.set_ylim(-0.55, rows - 0.25 + 0.45)
    ax.set_xlim(min(5.2, lo), max(6.0, hi))
    ax.set_xlabel("tetangga per sawit   (r = 1,5 × jarak tanam)",
                  color=PALETTE["ink_2"], fontsize=10)
    ax.grid(axis="x", color=PALETTE["grid"], lw=1)
    ax.set_axisbelow(True)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _num(v, 1)))
    return f


def fig_photo_risk(img, res):
    """Peta risiko DI ATAS FOTO PENGGUNA. Kuintil sama dengan peta Eg9PP."""
    t = res["tabel"]
    h = 7.0 * img.shape[0] / img.shape[1]
    f, ax = _fig(7.0, h, axes=False)
    ax.imshow(img)
    # Sumber lebih dulu, di bawah: pohon yang SUDAH bergejala tidak diberi skor.
    src = res.get("sumber_xy")
    if src is not None and len(src):
        ax.scatter(src[:, 0], src[:, 1], s=70, marker="X", c=PALETTE["ink"],
                   edgecolors="white", linewidths=1.4, zorder=3,
                   label="sudah bergejala (%d) — sumber" % len(src))
    for q in range(1, 6):
        m = t.kuintil == q
        if m.any():
            ax.scatter(t.cx[m], t.cy[m], s=52, c=PALETTE["quintile"][q - 1],
                       edgecolors="white", linewidths=1.2, zorder=4,
                       label="kuintil %d%s" % (q, " · prioritas" if q == 5 else ""))
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3,
              frameon=False, fontsize=8, labelcolor=PALETTE["ink_2"])
    return f


def fig_risk_map(df):
    """Kisi 1.200 sawit. Kuintil (bukan desil - sepuluh langkah gagal uji jarak
    kecerahan). Yang keluar dari penilaian dibedakan BENTUK, bukan warna saja."""
    risk = df[df.in_risk_set == 1].copy()
    risk["q"] = np.ceil(risk.risk_decile / 2).astype(int).clip(1, 5)
    out = df[df.in_risk_set == 0]
    span_x = float(df.xm.max() - df.xm.min())
    span_y = float(df.ym.max() - df.ym.min())
    f, ax = _fig(8.0, max(3.4, 8.0 * span_y / span_x + 1.1), axes=False)
    for s, mk, lab in (("S", "^", "bergejala"), ("D", "x", "mati"),
                       ("C", "s", "disensor")):
        m = out.status == s
        if m.any():
            ax.scatter(out.xm[m], out.ym[m], s=16, marker=mk, c="#A9AFA3",
                       linewidths=1.1, zorder=1,
                       label="%s (%d)" % (lab, int(m.sum())))
    for q in range(1, 6):
        m = risk.q == q
        if m.any():
            ax.scatter(risk.xm[m], risk.ym[m], s=30, c=PALETTE["quintile"][q - 1],
                       edgecolors=PALETTE["surface"], linewidths=0.5, zorder=2,
                       label="kuintil %d%s" % (q, " · paling berisiko" if q == 5 else ""))
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4,
              frameon=False, fontsize=8, labelcolor=PALETTE["ink_2"])
    return f


if __name__ == "__main__":
    print("== layar 4: Eg9PP ==")
    df = load_risk()
    s = risk_summary(df)
    print("  %d sawit -> %d dinilai, %d keluar %s"
          % (s["n_total"], s["n_risk"], s["n_out"], s["status_out"]))
    print("  skor %.4f .. %.4f (peringkat, BUKAN peluang)" % (s["logit_min"], s["logit_max"]))
    print("  tetangga sakit: 10 teratas %.1f | semua %.1f | 10 teraman %.1f"
          % (s["sick_nb_top10"], s["sick_nb_all"], s["sick_nb_bot10"]))
    print("  10 teratas:", ", ".join(s["top10"].palm_id.tolist()[:5]), "...")

    print("\n== layar 1+2: deteksi ==")
    smp = sample_images(3)
    print("  contoh tersedia: %d" % len(smp))
    if smp:
        d, info = detect_image(smp[0])
        print("  %s" % os.path.basename(smp[0]))
        print("  %d pohon | jarak tanam %.1f px | skala %.2fx | lolos=%s"
              % (info["n"], info["spacing_px"], info["scale_ratio"], info["ok_scale"]))
        print("  derajat %.2f semua | %.2f dalam (%d)"
              % (info["deg_all"], info["deg_inner"], info["n_inner"]))
        print("  sisi tergambar: %d" % len(edges_within(d[["cx", "cy"]].values,
                                                        info["r_graph_px"])))

    print("\n== layar 3: antarmuka ==")
    print("  L1 %.2f +/- %.2f (prediksi) | L2 %.2f | selisih %.1f%%"
          % (*FACTS["deg_l1_pred"], FACTS["deg_l2"], FACTS["gap_pct"]))
    n_missing = sum(n for _, n, ok, _ in FEATURE_BLOCKS if not ok)
    print("  kolom checkpoint tak terisi dari citra: %d dari 24" % n_missing)

    # Render keempat gambar ke PNG supaya bisa DILIHAT tanpa menjalankan server.
    import matplotlib
    matplotlib.use("Agg")
    from PIL import Image

    outdir = os.path.join(ROOT, "figures", "demo_preview")
    os.makedirs(outdir, exist_ok=True)
    figs = {"3_antarmuka": fig_interface(), "4_peta_risiko": fig_risk_map(df)}
    if smp:
        figs["1_deteksi"] = fig_detection(np.array(Image.open(smp[0]).convert("RGB")), d)
        if info["ok_n"]:
            figs["2_graf"] = fig_graph(d[["cx", "cy"]].values,
                                       edges_within(d[["cx", "cy"]].values,
                                                    info["r_graph_px"]))
    print("\n== pratinjau gambar ==")
    for nm, fg in sorted(figs.items()):
        p = os.path.join(outdir, nm + ".png")
        fg.savefig(p, dpi=110, bbox_inches="tight", facecolor=PALETTE["surface"])
        print("  %s" % os.path.relpath(p, ROOT))
