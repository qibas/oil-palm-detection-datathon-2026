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
for p in (L1,):
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
    return [
        {"bahan": "Daftar pohon + posisinya", "ada": n > 0,
         "punyamu": "%d pohon terdeteksi" % n if n else "belum ada citra diproses",
         "cara": "otomatis dari foto drone"},
        {"bahan": "Peta kontak antar-pohon", "ada": ok_graph,
         "punyamu": ("derajat %.2f, %d pohon bagian dalam"
                     % (info["deg_inner"], info["n_inner"])) if ok_graph
                    else "skala citra di luar jangkauan, graf tidak dibangun",
         "cara": "otomatis dari foto drone"},
        {"bahan": "Riwayat minimal 3 kunjungan", "ada": False,
         "punyamu": "1 dari 3 kunjungan" if n else "0 dari 3 kunjungan",
         "cara": "terbang ulang, atau sensus lapangan tiap 6 bulan"},
        {"bahan": "Catatan genotipe (progeni)", "ada": False,
         "punyamu": "belum ada",
         "cara": "arsip tanam kebun — tidak bisa dilihat dari udara"},
    ]


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

    df = pd.DataFrame({
        "cx": xy[:, 0], "cy": xy[:, 1],
        "conf": [d[2] for d in det],
        "cls": [dc.NAMES_OF(d[3]) if hasattr(dc, "NAMES_OF") else
                __import__("y12").NAMES[d[3]] for d in det],
        "deg": deg,
    })
    info = {
        "n": len(det), "box_px": box, "spacing_px": spacing, "scale_ratio": ratio,
        "ok_scale": ok_scale, "ok_n": ok_n, "deg_all": d_all, "deg_inner": d_in,
        "n_inner": n_in, "r_graph_px": dc.R_GRAPH * spacing if ok_n else float("nan"),
        "weights": os.path.relpath(w, ROOT), "fold": fold, "conf": conf,
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
    """Beberapa ubin ds_B sebagai contoh siap-klik, supaya demo tidak butuh unggahan."""
    import glob
    out = []
    for ortho in ("44000_16000", "44000_4000", "52000_20000"):
        g = sorted(glob.glob(os.path.join(L1, "ds_B", "train", ortho + "_*.jpg")))
        out += g[:max(1, n // 3)]
    return out[:n]


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
