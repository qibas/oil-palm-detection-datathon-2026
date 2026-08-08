"""Peta risiko Tahap 2 — per pohon dan tingkat blok, dari risk_ranked.csv.

    python risk_heatmap.py

Membaca `risk_ranked.csv` (keluaran `export_risk.py`) dan menggambar dua hal:

  1. peta per pohon  — 1.200 sawit pada koordinat tanam sebenarnya
  2. peta tingkat blok — petak geometris, nilai = rata-rata PERSENTIL risiko

TIGA BATAS YANG DIPAKSAKAN OLEH METADATA MODEL ITU SENDIRI
(`risk_ranked.meta.json`, medan `scope_warning`) — bukan tambahan kami:

  a. "Trained on ALL 1,200 palms with no held-out set. This is an INFERENCE
     artifact, not an evaluation artifact." Gambar ini memperagakan BENTUK
     keluaran. Tidak satu pun angka kinerja boleh dikutip darinya.

  b. "Do not convert `logit` to a probability. It is uncalibrated." Karena itu
     pewarnaan memakai PERSENTIL, bukan peluang, dan skala warnanya tidak pernah
     diberi label persen. Melihat kolom `logit`? Jangan sigmoid-kan.

  c. 528 dari 1.200 sawit berada DI LUAR risk set (status D/S/C). Mereka digambar
     dengan BENTUK penanda berbeda dan warna netral — bukan sebagai "risiko
     rendah". Sawit yang sudah mati bukan sawit yang aman. Ini juga sebabnya
     agregasi blok hanya merata-ratakan pohon di dalam risk set.

MENGAPA PETAKNYA GEOMETRIS, BUKAN PER `plot`. Kolom `plot` menggoda karena sudah
ada, tetapi tiap plot berisi tepat 15 pohon dari SATU progeny (terukur: 80 plot,
1 progeny per plot). Peta beragregasi `plot` karena itu adalah peta kerentanan
GENOTIPE yang menyamar sebagai peta spasial — persis confound yang null permutasi
dalam-famili dibangun untuk mengendalikan. Petak geometris memotong lintas famili.

Warna mengikuti aturan: besaran = SATU rona terang->gelap (bukan pelangi), dan
status dikodekan lewat BENTUK, tidak pernah lewat warna saja.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "risk_ranked.csv")
OUT = os.path.join(HERE, "fig_risk_heatmap.png")

TILE = 6.0            # sisi petak dalam kelipatan jarak tanam
CMAP = "Reds"         # sekuensial satu rona, terang -> gelap
INK = "#222222"
MUTED = "#6b6b6b"

# Status di luar risk set: bentuk + warna netral, tidak pernah warna saja.
STATUS = {"D": ("x", "mati"), "S": ("s", "simptomatik"), "C": ("^", "tersensor")}


def load():
    d = pd.read_csv(CSV)
    need = {"palm_id", "parcel", "xm", "ym", "in_risk_set", "status", "risk_percentile"}
    missing = need - set(d.columns)
    if missing:
        raise SystemExit("kolom hilang di risk_ranked.csv: %s" % sorted(missing))
    return d


def blocks(g, tile=TILE):
    """Agregasi petak geometris. Nilai = rata-rata persentil pohon DI DALAM risk set."""
    g = g.copy()
    g["bx"] = np.floor((g.xm - g.xm.min()) / tile).astype(int)
    g["by"] = np.floor((g.ym - g.ym.min()) / tile).astype(int)
    out = []
    for (bx, by), t in g.groupby(["bx", "by"]):
        risk = t[t.in_risk_set == 1]
        out.append(dict(bx=bx, by=by, n=len(t), n_risk=len(risk),
                        x0=g.xm.min() + bx * tile, y0=g.ym.min() + by * tile,
                        mean_pct=risk.risk_percentile.mean() if len(risk) else np.nan))
    return pd.DataFrame(out)


def main():
    d = load()
    parcels = sorted(d.parcel.unique())
    # Sumbu x DISAMAKAN untuk seluruh panel supaya kedua parcel dapat
    # dibandingkan langsung; kalau tidak, 44B (x 18-52) tergambar lebih sempit
    # daripada 44A (x 8-59) dan perbedaan lebar terbaca seolah perbedaan data.
    XLO, XHI = d.xm.min() - 2, d.xm.max() + 2
    fig, axes = plt.subplots(2 * len(parcels), 1, figsize=(15, 13.5),
                             gridspec_kw=dict(hspace=0.55))
    fig.subplots_adjust(top=0.90, bottom=0.09, left=0.06, right=0.88)
    sm = None

    for i, pc in enumerate(parcels):
        g = d[d.parcel == pc]
        risk = g[g.in_risk_set == 1]
        rest = g[g.in_risk_set == 0]

        # ---- baris 1: per pohon -------------------------------------------
        ax = axes[2 * i]
        sm = ax.scatter(risk.xm, risk.ym, c=risk.risk_percentile, cmap=CMAP,
                        vmin=0, vmax=100, s=46, edgecolor="white", linewidth=0.6,
                        zorder=3)
        for s, (mk, _) in STATUS.items():
            q = rest[rest.status == s]
            if len(q):
                ax.scatter(q.xm, q.ym, marker=mk, s=30, c=MUTED, linewidth=1.1,
                           zorder=2)
        ax.set_title("Parcel %s — per pohon  (n=%d; %d dalam risk set, %d di luar)"
                     % (pc, len(g), len(risk), len(rest)),
                     fontsize=11, color=INK, loc="left")

        # ---- baris 2: tingkat blok ----------------------------------------
        axb = axes[2 * i + 1]
        b = blocks(g)
        norm = plt.Normalize(0, 100)
        cm = plt.get_cmap(CMAP)
        for _, r in b.iterrows():
            filled = not np.isnan(r.mean_pct)
            axb.add_patch(plt.Rectangle(
                (r.x0, r.y0), TILE, TILE,
                facecolor=cm(norm(r.mean_pct)) if filled else "#f0f0f0",
                edgecolor="white", linewidth=1.6,
                hatch=None if filled else "///", zorder=1))
            if filled:
                axb.text(r.x0 + TILE / 2, r.y0 + TILE / 2,
                         "%.0f\nn=%d" % (r.mean_pct, r.n_risk),
                         ha="center", va="center", fontsize=7.5, color=INK, zorder=4)
        axb.set_title("Parcel %s — tingkat blok, petak %.0f x %.0f jarak tanam "
                      "(nilai = rata-rata persentil risiko; arsir = tak ada pohon "
                      "dalam risk set)" % (pc, TILE, TILE),
                      fontsize=11, color=INK, loc="left")

        for a in (ax, axb):
            a.set_aspect("equal")
            a.set_xlim(XLO, XHI)
            a.set_ylim(g.ym.min() - 1.5, g.ym.max() + 1.5)
            a.set_ylabel("y", fontsize=8, color=MUTED)
            a.tick_params(labelsize=7, colors=MUTED)
            for sp in a.spines.values():
                sp.set_color("#dddddd")
        # label x hanya pada panel terbawah -> tidak menabrak judul di bawahnya
        for a in (ax, axb):
            a.set_xlabel("")
    axes[-1].set_xlabel("x (kelipatan jarak tanam)", fontsize=9, color=MUTED)

    cax = fig.add_axes([0.90, 0.30, 0.014, 0.40])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("persentil risiko\n(bukan peluang — skor tidak terkalibrasi)",
                 fontsize=9, color=INK)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    cb.outline.set_edgecolor("#dddddd")

    handles = [Line2D([], [], marker="o", linestyle="", color="#cb4335",
                      markeredgecolor="white", markersize=8,
                      label="dalam risk set (A) — diwarnai persentil")]
    handles += [Line2D([], [], marker=mk, linestyle="", color=MUTED, markersize=8,
                       label="di luar risk set: %s" % lab)
                for mk, lab in STATUS.values()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.47, 0.015),
               ncol=4, fontsize=9, frameon=False, labelcolor=INK)

    fig.text(0.47, 0.965,
             "Tahap 2 — peta risiko Eg9PP, sensus t = 24 th, horizon 3 sensus",
             ha="center", fontsize=13.5, color=INK)
    fig.text(0.47, 0.935,
             "ARTEFAK INFERENSI: model dilatih pada seluruh 1.200 sawit tanpa held-out. "
             "Tidak ada angka kinerja yang boleh dikutip dari gambar ini.",
             ha="center", fontsize=10, color="#b03a2e")
    fig.savefig(OUT, dpi=150, facecolor="white")
    print("tersimpan:", OUT)

    for pc in parcels:
        b = blocks(d[d.parcel == pc]).dropna(subset=["mean_pct"])
        top = b.nlargest(3, "mean_pct")
        print("\nparcel %s — %d petak berisi pohon risk-set" % (pc, len(b)))
        for _, r in top.iterrows():
            print("   petak (x %.0f-%.0f, y %.0f-%.0f)  persentil rata-rata %.1f  n_risk=%d"
                  % (r.x0, r.x0 + TILE, r.y0, r.y0 + TILE, r.mean_pct, r.n_risk))
    return OUT


if __name__ == "__main__":
    main()
