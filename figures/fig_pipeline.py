"""SawitGuard-GNN — Figure 1: arsitektur pipeline dua lapis (untuk Bab 3 paper).

Diagram skematik, bukan figur hasil: tidak membaca results.csv. Dipisah dari
make_figures.py supaya file itu tetap murni digerakkan hasil eksperimen.

Poin yang HARUS terbaca dari gambar ini: Lapisan 1 dan Lapisan 2 adalah dua
pipeline independen yang tidak berbagi data. Jembatan di antara keduanya digambar
putus-putus karena memang belum ada — bukan hiasan.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = r"C:\Users\Brian\Desktop\Project\AI CONCEPT PAPER\fig_pipeline.png"

# Palet kategorikal tervalidasi (dataviz slot 2 = hijau, slot 1 = biru).
L1_EDGE, L1_FILL = "#008300", "#e7f2e7"   # lapisan 1 — data nyata
L2_EDGE, L2_FILL = "#2a78d6", "#e4edfa"   # lapisan 2 — data sintetis
GREY_EDGE, GREY_FILL = "#52514e", "#f1f1ef"
INK, MUTED = "#0b0b0b", "#52514e"

TITLE_FS, BODY_FS, LANE_FS, STRIP_FS = 7.6, 6.4, 8.2, 6.4


def box(ax, x, y, w, h, title, body, edge, fill):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
        linewidth=1.1, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h - 3.6, title, ha="center", va="center",
            fontsize=TITLE_FS, fontweight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, y + h / 2 - 3.0, body, ha="center", va="center",
            fontsize=BODY_FS, color=MUTED, linespacing=1.45, zorder=3)


def arrow(ax, x0, y0, x1, y1, color, dashed=False, lw=1.1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
        linewidth=lw, color=color, zorder=4,
        linestyle=(0, (2.4, 2.0)) if dashed else "solid",
        shrinkA=0, shrinkB=0))


def strip(ax, y, h, label, text, edge):
    ax.add_patch(FancyBboxPatch(
        (0, y), 100, h, boxstyle="round,pad=0,rounding_size=1.2",
        linewidth=0.9, edgecolor=edge, facecolor=GREY_FILL, zorder=2))
    ax.text(2.2, y + h / 2, label, ha="left", va="center",
            fontsize=STRIP_FS, fontweight="bold", color=INK, zorder=3)
    ax.text(2.2 + 15.5, y + h / 2, text, ha="left", va="center",
            fontsize=STRIP_FS, color=MUTED, linespacing=1.45, zorder=3)


def main():
    fig, ax = plt.subplots(figsize=(7.1, 4.35))
    ax.set_xlim(-1, 101)
    ax.set_ylim(15.5, 100.5)
    ax.axis("off")

    # ---------------- LAPISAN 1 — demonstrator UAV (citra nyata) -------------
    ax.text(0, 98.0, "LAPISAN 1  ·  Demonstrator UAV — citra nyata",
            ha="left", va="center", fontsize=LANE_FS, fontweight="bold", color=L1_EDGE)
    ax.text(100, 98.0, "label kesehatan tajuk generik — BUKAN BSR",
            ha="right", va="center", fontsize=BODY_FS, color=MUTED, style="italic")

    w1, y1, h1 = 22.0, 80.0, 14.5
    xs1 = [0.0, 26.0, 52.0, 78.0]
    l1 = [("Citra UAV RGB", "3 ortomosaik\n2.745 tile nadir"),
          ("Deteksi tajuk", "YOLO11\n→ kotak + ID pohon"),
          ("Penilaian kesehatan", "LightGBM\n→ skor per-tajuk"),
          ("Luas tajuk", "ExG + Otsu\n→ proksi kanopi")]
    for x, (t, b) in zip(xs1, l1):
        box(ax, x, y1, w1, h1, t, b, L1_EDGE, L1_FILL)
    for x in xs1[:-1]:
        arrow(ax, x + w1, y1 + h1 / 2, x + w1 + 4.0, y1 + h1 / 2, L1_EDGE)

    strip(ax, 70.0, 7.6, "Evaluasi",
          "block-CV leave-one-ortho-out (3 ortomosaik; split acak bocor 100%)  ·  "
          "mAP untuk deteksi, PR-AUC / ROC-AUC\nuntuk kesehatan (imbalance ≈69:1) — bukan akurasi",
          GREY_EDGE)

    # ---------------- jembatan yang belum ada -------------------------------
    arrow(ax, 26, 68.5, 26, 52.0, "#8a8a85", dashed=True, lw=1.3)
    ax.text(29.5, 63.0,
            "belum tersambung — kedua lapisan tidak berbagi data maupun kode",
            ha="left", va="center", fontsize=BODY_FS, color=MUTED, fontweight="bold")
    ax.text(29.5, 58.4,
            "penyambungan menunggu label BSR per-pohon berlabel waktu (rencana: Sembawa)",
            ha="left", va="center", fontsize=BODY_FS, color=MUTED, style="italic")

    # ---------------- LAPISAN 2 — simulator + ablasi GNN ---------------------
    ax.text(0, 47.5, "LAPISAN 2  ·  Simulator penyebaran + ablasi GNN",
            ha="left", va="center", fontsize=LANE_FS, fontweight="bold", color=L2_EDGE)
    ax.text(100, 47.5, "epidemi sintetis — tidak ada data BSR nyata",
            ha="right", va="center", fontsize=BODY_FS, color=MUTED, style="italic")

    w2, y2, h2 = 18.0, 28.5, 14.5
    xs2 = [0.0, 20.5, 41.0, 61.5, 82.0]
    l2 = [("config.py", "parameter epidemi\n(PLACEHOLDER)"),
          ("Simulator SEIR", "1.600 pohon\n20 siklus survei"),
          ("Graf multi-relasi", "akar · angin\nmanajemen"),
          ("Fitur node", "spektra → PCA-16\n+ delta (32-d)"),
          ("Peramalan", "MLP · STGNN\nSTGNN-SEIR")]
    for x, (t, b) in zip(xs2, l2):
        box(ax, x, y2, w2, h2, t, b, L2_EDGE, L2_FILL)
    for x in xs2[:-1]:
        arrow(ax, x + w2, y2 + h2 / 2, x + w2 + 2.5, y2 + h2 / 2, L2_EDGE)

    strip(ax, 17.5, 8.6, "Ablasi &\nEvaluasi",
          "penukaran graph view: asli / acak / tanpa-graf / perturbasi ε  →  dekomposisi "
          "keunggulan menjadi\ntemporal | prevalensi | struktur  ·  20 seed, AUC-PR berpasangan "
          "dengan sign-count dan band noise",
          GREY_EDGE)
    # view ditukar masuk ke graf — panah balik dari strip ablasi
    arrow(ax, 50.0, 26.1, 50.0, 28.5, L2_EDGE, dashed=True)

    fig.tight_layout(pad=0.25)
    fig.savefig(OUT, dpi=400, facecolor="white")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
