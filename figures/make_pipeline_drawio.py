"""Gambar 1 (pipeline enam tahap) -> fig_pipeline.drawio + fig_pipeline.png

SATU tabel tata letak (LAYOUT + EDGES) menghasilkan DUA keluaran yang identik
geometrinya:

  fig_pipeline.drawio  XML mxfile, dibuka di app.diagrams.net atau draw.io desktop
  fig_pipeline.png     raster + XML yang sama ditanam pada chunk tEXt "mxfile",
                       sehingga PNG-nya sendiri bisa di-drag ke draw.io dan
                       langsung bisa diedit (bukan sekadar gambar mati)

Karena keduanya dibangun dari daftar yang sama, hasil raster dan hasil editnya
tidak bisa berbeda. Kalau kotaknya mau digeser, geser di draw.io (berkas .drawio
menjadi sumber kebenaran sejak saat itu) atau ubah LAYOUT di sini lalu bangun
ulang keduanya.

Isi diagram mengikuti naskah `paper/section_methodology.docx`:
  Lapisan 1 = Tahap 1..3 (citra UAV -> inventaris tajuk)
  Lapisan 2 = Tahap 4..6 (geometri kebun -> peringkat risiko)
Kotak "Uji antarmuka" digambar putus-putus karena kedua kebun memang TIDAK
digabungkan; yang diukur hanya kecocokan derajat grafnya (5,62 vs 5,74).

Aturan teks sama dengan naskah: tanpa em dash.
"""
import binascii
import os
import struct
from urllib.parse import quote
from xml.sax.saxutils import quoteattr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
PNG = os.path.join(HERE, "fig_pipeline.png")
DRAWIO = os.path.join(HERE, "fig_pipeline.drawio")

W, H = 1160, 716                    # satuan draw.io (1 unit = 1 px di draw.io)
PT = 72.0 / 100.0                   # 1 unit -> pt di matplotlib (figsize = W/100 in)

# Palet kategorikal tervalidasi, sama dengan fig_pipeline.py lama.
L1_EDGE, L1_FILL = "#008300", "#e7f2e7"     # Lapisan 1
L2_EDGE, L2_FILL = "#2a78d6", "#e4edfa"     # Lapisan 2
GREY_EDGE, GREY_FILL = "#52514e", "#f1f1ef"  # strip evaluasi
INK, MUTED, WHITE = "#0b0b0b", "#52514e", "#ffffff"

TITLE_PX, BODY_PX, LANE_PX, NOTE_PX = 12, 10, 13, 10

# --------------------------------------------------------------------------
# TATA LETAK
# kind: lane | box | source | note | strip
# --------------------------------------------------------------------------
LAYOUT = [
    dict(id="lane1", kind="lane", x=20, y=14, w=760, h=22, edge=L1_EDGE,
         title="LAPISAN 1  ·  Dari citra UAV menjadi inventaris tajuk"),
    dict(id="lane1n", kind="lane", x=380, y=14, w=740, h=22, edge=MUTED, align="right",
         title="data nyata; label kesehatan tajuk generik, bukan BSR", italic=True),

    dict(id="a0", kind="source", x=20, y=50, w=260, h=88, edge=L1_EDGE, fill=WHITE,
         title="Citra UAV RGB nadir",
         body="Roboflow ds_B\n3 ortomosaik · 2.303 ubin 1.024 px\nGSD ≈ 8,7 cm/px"),
    dict(id="a1", kind="box", x=300, y=50, w=260, h=88, edge=L1_EDGE, fill=L1_FILL,
         title="Tahap 1 · Deteksi tajuk",
         body="YOLO11\nkeluaran: kotak pembatas dan\nkoordinat pusat tajuk"),
    dict(id="a2", kind="box", x=580, y=50, w=260, h=88, edge=L1_EDGE, fill=L1_FILL,
         title="Tahap 2 · Estimasi luas tajuk",
         body="Excess Green + ambang Otsu\nproksi ukuran kanopi,\nkualitatif, tanpa klaim IoU"),
    dict(id="a3", kind="box", x=860, y=50, w=260, h=88, edge=L1_EDGE, fill=L1_FILL,
         title="Tahap 3 · Penilaian kesehatan",
         body="LightGBM atas statistik RGB\nkeluaran: skor kesehatan\nper tajuk"),

    dict(id="s1", kind="strip", x=20, y=158, w=900, h=44, edge=GREY_EDGE, fill=GREY_FILL,
         title="Evaluasi Lapisan 1",
         body="block-CV leave-one-ortho-out, 3 lipatan; pembagian acak dipastikan bocor\n"
              "metrik mAP untuk deteksi, PR-AUC dan ROC-AUC untuk kesehatan; "
              "akurasi tidak dilaporkan"),
    dict(id="brg", kind="lane", x=936, y=158, w=170, h=44, edge=MUTED, align="left",
         title="inventaris tajuk\n(posisi, ukuran, skor)", italic=True),

    # Catatan kanan sengaja hanya ada di Lapisan 1: di Lapisan 2 ia bertabrakan
    # dengan penghubung vertikal, dan isinya sudah dibawa kotak sumber Eg9PP.
    dict(id="lane2", kind="lane", x=20, y=222, w=760, h=22, edge=L2_EDGE,
         title="LAPISAN 2  ·  Dari inventaris tajuk menjadi peringkat risiko"),

    dict(id="b4", kind="box", x=20, y=252, w=1100, h=62, edge=L2_EDGE, fill=L2_FILL,
         title="Tahap 4 · Rekonstruksi geometri kebun",
         body="offset pada nama berkas ubin + pusat kotak pembatas → koordinat global  ·  "
              "151.060 kotak anotasi = 5.077 pohon unik  ·  "
              "skala dari jarak tetangga 101–106 px terhadap jarak tanam 9 m"),

    # "Uji antarmuka" duduk DI ANTARA Tahap 4 dan sumber Lapisan 2, disambung
    # panah putus-putus berkepala dua ke atas dan ke bawah. Kepala dua =
    # perbandingan, bukan aliran data; kedua kebun memang tidak digabungkan.
    dict(id="uji", kind="note", x=20, y=336, w=260, h=62, edge=MUTED, fill=WHITE,
         title="Uji antarmuka",
         body="dibandingkan, bukan dialirkan\nderajat 5,62 lawan 5,74, selisih 2%"),
    dict(id="ujin", kind="lane", x=300, y=336, w=620, h=62, edge=MUTED, align="left",
         italic=True,
         title="kebun, zaman, dan sistem koordinat kedua sumber berbeda, sehingga keduanya tidak "
               "digabungkan\n"
               "5,62 memakai kotak kebenaran-dasar, bukan prediksi YOLO, jadi ia batas atas\n"
               "5,74 dihitung pada pohon bagian dalam saja, karena itu berbeda dari derajat 5,59 "
               "graf penuh Tahap 5"),

    dict(id="b0", kind="source", x=20, y=420, w=340, h=92, edge=L2_EDGE, fill=WHITE,
         title="Panel lapangan Eg9PP",
         body="1.200 sawit · 14 famili · 2 parcel\n45 sensus sepanjang 25 tahun\n"
              "Ganoderma terverifikasi; tanpa citra"),
    dict(id="b5", kind="box", x=400, y=420, w=340, h=92, edge=L2_EDGE, fill=L2_FILL,
         title="Tahap 5 · Graf kedekatan dan peramalan",
         body="r = 1,5 × jarak tanam\n3.354 sisi · derajat rata-rata 5,59\n"
              "MLP · STGNN · STGNN+SI(D)"),
    dict(id="b6", kind="box", x=780, y=420, w=340, h=92, edge=L2_EDGE, fill=L2_FILL,
         title="Tahap 6 · Peringkat risiko",
         body="AUC-PR pada horizon h = 1…4\nkalibrasi skor ke persentase\nbelum dikerjakan"),

    # Keluaran operasional. Digambar putus-putus, sama seperti "Uji antarmuka",
    # karena keduanya BELUM dihasilkan: keluaran model saat ini masih peringkat,
    # dan kalibrasi skor menjadi persentase belum dikerjakan.
    dict(id="outn", kind="lane", x=20, y=546, w=510, h=84, edge=MUTED, align="left",
         italic=True,
         title="Keluaran yang dituju bagi tim kebun.\n"
               "Keduanya belum dihasilkan: keluaran model saat ini masih berupa peringkat,\n"
               "dan kalibrasi skor menjadi persentase belum dikerjakan."),
    dict(id="out1", kind="note", x=560, y=546, w=270, h=84, edge=MUTED, fill=WHITE,
         title="Daftar intervensi berperingkat",
         body="ID pohon · koordinat · skor risiko\ntop-k pohon pada anggaran tetap"),
    dict(id="out2", kind="note", x=850, y=546, w=270, h=84, edge=MUTED, fill=WHITE,
         title="Peta risiko tingkat blok",
         body="risiko ditumpangkan pada kisi tanam\nmengarahkan sensus lapangan berikutnya"),

    dict(id="s2", kind="strip", x=20, y=648, w=1100, h=50, edge=GREY_EDGE, fill=GREY_FILL,
         body_dx=250, title="Ablasi dan evaluasi Lapisan 2",
         body="tukar tampilan graf: asli / acak / acak-lokal / tanpa-graf  →  "
              "dekomposisi temporal | prevalensi | struktur\n"
              "leave-one-parcel-out, 2 lipatan  ·  20 seed × 2 lipatan = 40 pasangan  ·  "
              "selisih di dalam 1 simpangan baku dinyatakan tidak konklusif"),
]

# Sisi: titik jangkar absolut; pecahan exit/entry untuk draw.io dihitung darinya.
# double=True -> kepala panah di kedua ujung, dipakai KHUSUS untuk perbandingan
# (uji antarmuka), supaya tidak terbaca sebagai aliran data.
EDGES = [
    dict(src="a0", dst="a1", p0=(280, 94), p1=(300, 94), color=L1_EDGE),
    dict(src="a1", dst="a2", p0=(560, 94), p1=(580, 94), color=L1_EDGE),
    dict(src="a2", dst="a3", p0=(840, 94), p1=(860, 94), color=L1_EDGE),
    dict(src="a3", dst="b4", p0=(1080, 138), p1=(1080, 252), color=L1_EDGE),
    dict(src="b4", dst="uji", p0=(150, 314), p1=(150, 336), color=MUTED,
         dashed=True, double=True),
    dict(src="uji", dst="b0", p0=(150, 398), p1=(150, 420), color=MUTED,
         dashed=True, double=True),
    dict(src="b0", dst="b5", p0=(360, 466), p1=(400, 466), color=L2_EDGE),
    dict(src="b5", dst="b6", p0=(740, 466), p1=(780, 466), color=L2_EDGE),
    dict(src="b6", dst="out2", p0=(985, 512), p1=(985, 546), color=L2_EDGE, dashed=True),
    dict(src="b6", dst="out1", p0=(830, 512), p1=(695, 546), color=L2_EDGE, dashed=True,
         via=[(830, 530), (695, 530)]),
]


# --------------------------------------------------------------------------
# KELUARAN 1 — XML draw.io
# --------------------------------------------------------------------------
def html_value(node):
    """Nilai sel draw.io: judul tebal + badan kecil abu-abu."""
    title = node.get("title", "").replace("\n", "<br>")
    body = node.get("body")
    style = "font-size: %dpx; color: %s;" % (BODY_PX, MUTED)
    if node["kind"] == "lane":
        it = "font-style: italic; " if node.get("italic") else "font-weight: bold; "
        return '<span style="%sfont-size: %dpx; color: %s;">%s</span>' % (
            it, NOTE_PX if node.get("italic") else LANE_PX, node["edge"], title)
    if not body:
        return "<b>%s</b>" % title
    return '<b>%s</b><br><span style="%s">%s</span>' % (
        title, style, body.replace("\n", "<br>"))


def node_style(node):
    if node["kind"] == "lane":
        return ("text;html=1;strokeColor=none;fillColor=none;verticalAlign=middle;"
                "align=%s;spacingLeft=0;spacingRight=0;" % node.get("align", "left"))
    dash = "dashed=1;dashPattern=6 4;" if node["kind"] == "note" else ""
    rounding = "rounded=1;arcSize=6;" if node["kind"] != "strip" else "rounded=1;arcSize=4;"
    return (rounding + "whiteSpace=wrap;html=1;" + dash +
            "fillColor=%s;strokeColor=%s;strokeWidth=1.4;" % (node["fill"], node["edge"]) +
            "fontSize=%d;fontColor=%s;verticalAlign=middle;align=%s;spacing=6;"
            % (TITLE_PX, INK, "left" if node["kind"] == "strip" else "center") +
            ("spacingLeft=10;" if node["kind"] == "strip" else ""))


def frac(v, lo, span):
    return round((v - lo) / float(span), 4)


def build_xml():
    by_id = {n["id"]: n for n in LAYOUT}
    cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']

    def vertex(cid, value, style, x, y, w, h):
        cells.append(
            '<mxCell id=%s value=%s style=%s vertex="1" parent="1">'
            '<mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry" /></mxCell>'
            % (quoteattr(cid), quoteattr(value), quoteattr(style), x, y, w, h))

    txt = ("text;html=1;strokeColor=none;fillColor=none;align=left;"
           "verticalAlign=middle;spacingLeft=0;spacingRight=0;")

    for n in LAYOUT:
        # Strip dipecah jadi kotak + dua label supaya susunan judul-di-kiri /
        # badan-di-kanan pada draw.io persis sama dengan yang dirender ke PNG.
        if n["kind"] == "strip":
            dx = n.get("body_dx", 175)
            vertex(n["id"], "", node_style(n), n["x"], n["y"], n["w"], n["h"])
            vertex(n["id"] + "_t",
                   '<b style="font-size: %dpx;">%s</b>' % (TITLE_PX, n["title"]),
                   txt, n["x"] + 12, n["y"], dx - 18, n["h"])
            vertex(n["id"] + "_b",
                   '<span style="font-size: %dpx; color: %s;">%s</span>'
                   % (BODY_PX, MUTED, n["body"].replace("\n", "<br>")),
                   txt, n["x"] + dx, n["y"], n["w"] - dx - 12, n["h"])
            continue
        vertex(n["id"], html_value(n), node_style(n), n["x"], n["y"], n["w"], n["h"])

    for i, e in enumerate(EDGES):
        s, d = by_id[e["src"]], by_id[e["dst"]]
        style = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=blockThin;"
                 "endFill=1;strokeColor=%s;strokeWidth=1.4;" % e["color"] +
                 ("startArrow=blockThin;startFill=1;" if e.get("double") else "") +
                 ("dashed=1;dashPattern=6 4;" if e.get("dashed") else "") +
                 "exitX=%s;exitY=%s;exitDx=0;exitDy=0;entryX=%s;entryY=%s;entryDx=0;entryDy=0;"
                 % (frac(e["p0"][0], s["x"], s["w"]), frac(e["p0"][1], s["y"], s["h"]),
                    frac(e["p1"][0], d["x"], d["w"]), frac(e["p1"][1], d["y"], d["h"])))
        pts = "".join('<mxPoint x="%d" y="%d" />' % (x, y) for x, y in e.get("via", []))
        geom = ('<mxGeometry relative="1" as="geometry">'
                '<Array as="points">%s</Array></mxGeometry>' % pts) if pts else \
               '<mxGeometry relative="1" as="geometry" />'
        cells.append(
            '<mxCell id="e%d" style=%s edge="1" parent="1" source=%s target=%s>%s</mxCell>'
            % (i, quoteattr(style), quoteattr(e["src"]), quoteattr(e["dst"]), geom))

    return (
        '<mxfile host="app.diagrams.net" agent="sawitguard-make_pipeline_drawio" type="device">'
        '<diagram id="sawitguard-pipeline" name="Pipeline">'
        '<mxGraphModel dx="%d" dy="%d" grid="1" gridSize="10" guides="1" tooltips="1" '
        'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="%d" pageHeight="%d" '
        'math="0" shadow="0"><root>%s</root></mxGraphModel></diagram></mxfile>'
        % (W, H, W + 40, H + 40, "".join(cells)))


# --------------------------------------------------------------------------
# KELUARAN 2 — PNG dengan XML tertanam
# --------------------------------------------------------------------------
def draw_box(ax, n):
    ax.add_patch(FancyBboxPatch(
        (n["x"], n["y"]), n["w"], n["h"],
        boxstyle="round,pad=0,rounding_size=6",
        linewidth=1.4, edgecolor=n["edge"], facecolor=n["fill"], zorder=2,
        linestyle=(0, (5, 3)) if n["kind"] == "note" else "solid"))

    if n["kind"] == "strip":
        ax.text(n["x"] + 12, n["y"] + n["h"] / 2, n["title"], ha="left", va="center",
                fontsize=TITLE_PX * PT, fontweight="bold", color=INK, zorder=3)
        ax.text(n["x"] + n.get("body_dx", 175), n["y"] + n["h"] / 2, n["body"],
                ha="left", va="center",
                fontsize=BODY_PX * PT, color=MUTED, linespacing=1.5, zorder=3)
        return

    cx = n["x"] + n["w"] / 2
    ax.text(cx, n["y"] + 17, n["title"], ha="center", va="center",
            fontsize=TITLE_PX * PT, fontweight="bold", color=INK, zorder=3)
    ax.text(cx, n["y"] + (n["h"] + 24) / 2, n["body"], ha="center", va="center",
            fontsize=BODY_PX * PT, color=MUTED, linespacing=1.5, zorder=3)


def draw_lane(ax, n):
    ha = n.get("align", "left")
    x = n["x"] if ha == "left" else n["x"] + n["w"]
    italic = n.get("italic", False)
    ax.text(x, n["y"] + n["h"] / 2, n["title"], ha=ha, va="center",
            fontsize=(NOTE_PX if italic else LANE_PX) * PT,
            fontweight="normal" if italic else "bold",
            style="italic" if italic else "normal",
            color=n["edge"], linespacing=1.5, zorder=3)


def render_png():
    fig, ax = plt.subplots(figsize=(W / 100.0, H / 100.0))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)                  # y ke bawah, sama dengan draw.io
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for n in LAYOUT:
        (draw_lane if n["kind"] == "lane" else draw_box)(ax, n)

    for e in EDGES:
        # Rute ortogonal: titik antara digambar sebagai ruas garis biasa, dan
        # hanya ruas terakhir yang memakai kepala panah. Titik yang sama dikirim
        # ke draw.io sebagai waypoint, jadi kedua keluaran berimpit.
        style = dict(linewidth=1.4, color=e["color"], zorder=4,
                     linestyle=(0, (5, 3)) if e.get("dashed") else "solid")
        chain = [e["p0"]] + list(e.get("via", [])) + [e["p1"]]
        for a, b in zip(chain, chain[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], solid_capstyle="butt", **style)
        ax.add_patch(FancyArrowPatch(
            chain[-2], chain[-1], mutation_scale=9,
            arrowstyle="<|-|>" if e.get("double") else "-|>",
            shrinkA=0, shrinkB=0, **style))

    fig.savefig(PNG, dpi=300, facecolor="white")
    plt.close(fig)


def embed_xml(png_path, xml):
    """Sisipkan chunk tEXt "mxfile" sebelum IEND supaya draw.io bisa membukanya."""
    raw = open(png_path, "rb").read()
    idx = raw.rfind(b"\x00\x00\x00\x00IEND\xaeB`\x82")
    if idx < 0:
        raise SystemExit("FATAL: chunk IEND tidak ditemukan pada " + png_path)

    # payload = keyword + \0 + nilai; panjang chunk dihitung dari payload itu,
    # CRC dihitung atas nama chunk + payload.
    payload = b"mxfile\x00" + quote(xml, safe="!~*'()").encode("ascii")
    chunk = (struct.pack(">I", len(payload)) + b"tEXt" + payload +
             struct.pack(">I", binascii.crc32(b"tEXt" + payload) & 0xFFFFFFFF))
    open(png_path, "wb").write(raw[:idx] + chunk + raw[idx:])
    return len(payload)


def main():
    assert "—" not in repr(LAYOUT), "em dash tidak boleh muncul di diagram"

    xml = build_xml()
    open(DRAWIO, "w", encoding="utf-8").write(xml)
    render_png()
    n = embed_xml(PNG, xml)
    print("wrote", DRAWIO, "(%d karakter XML)" % len(xml))
    print("wrote", PNG, "(chunk mxfile %d bita, PNG dapat dibuka di draw.io)" % n)


if __name__ == "__main__":
    main()
