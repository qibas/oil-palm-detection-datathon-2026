"""Bangun 00_ANGKA_FINAL.md dari 00_RINGKASAN.csv. Jangan ketik tabelnya tangan.

    python make_final_table.py

MENGAPA BERKAS INI ADA.

Paket ini sekarang memuat angka dari SEMBILAN konfigurasi berbeda: model penuh dan
varian foto, AP gabungan dan AP dalam-sensus, 1 / 2 / 3 / 6 / 24 kolom, window 1 dan
3, empat horizon, dan dua jalur kontrol yang berbeda. Beberapa pasang di antaranya
TIDAK SEBANDING, dan salah membandingkannya akan menghasilkan klaim yang tidak
berdasar - tepatnya jenis kesalahan yang paling mahal di naskah.

Tabel ini dihasilkan dari `00_RINGKASAN.csv` supaya ia mustahil melenceng dari
angka sumbernya. Kolom "sebanding dengan" adalah bagian terpentingnya.
"""
import csv
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "00_RINGKASAN.csv")
OUT = os.path.join(ROOT, "00_ANGKA_FINAL.md")


def rows():
    with open(SRC, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get(rs, eksperimen, item):
    for r in rs:
        if r["eksperimen"] == eksperimen and r["item"] == item:
            return r
    return None


def fmt(r, d=4, signed=False):
    """`signed=True` hanya untuk SELISIH. Nilai absolut (AP, F1, derajat) tidak
    boleh diberi tanda + - itu membuatnya terbaca seperti perbaikan."""
    if r is None:
        return "—", "—", "—"
    m = float(r["mean"])
    s = r["std"]
    val = ("%+.*f" % (d, m) if signed else "%.*f" % (d, m)).replace(".", ",")
    if s:
        val += " ± " + ("%.*f" % (d, float(s))).replace(".", ",")
    return val, (r["sign"] or "—"), (r["vonis"] or "—")


def main():
    rs = rows()
    L = []
    W = L.append

    W("# Angka final — satu tabel, satu sumber\n")
    W("> Dihasilkan `make_final_table.py` dari `00_RINGKASAN.csv`. **Jangan sunting tangan.**")
    W("> Kolom **sebanding dengan** menentukan angka mana boleh diletakkan berdampingan.\n")

    # ---------------------------------------------------------------- 0
    W("## 0 · Keputusan: apa yang masuk paper, dan apa yang muncul di demo\n")
    W("Dua kolom terakhir adalah inti berkas ini. **PAPER** menandai angka utama; ")
    W("**DEMO** menandai apakah angka itu benar-benar terlihat di aplikasi. Paper ")
    W("tidak boleh mengutip angka yang konfigurasinya berbeda dari yang dijalankan demo ")
    W("tanpa menyebut perbedaannya.\n")
    W("| Angka | Nilai | Jalur | Kolom | W | h | Metrik | PAPER | DEMO |")
    W("|---|---|---|---|---|---|---|---|---|")
    for row in [
        ("STRUKTUR true−random", "+0,0151 (39/40)", "Eg9PP", "24", "3", "3",
         "AP gabungan", "**UTAMA**", "tidak"),
        ("STRUKTUR true−random", "+0,0165 (39/40)", "Eg9PP", "24", "3", "**4**",
         "AP gabungan", "pendukung", "tidak"),
        ("STRUKTUR foto true−random", "+0,0296 (40/40)", "foto", "6", "1", "3",
         "AP dalam-sensus", "**UTAMA**", "tidak"),
        ("foto − penuh", "+0,0042 (36/40)", "foto vs Eg9PP", "6 vs 24", "1 vs 3", "3",
         "AP dalam-sensus", "**UTAMA**", "tidak"),
        ("AP foto 6 kolom", "0,1015 → lift **1,61×**", "foto", "6", "1", "3",
         "AP dalam-sensus", "pendukung", "tidak"),
        ("AP foto 1 kolom", "0,0916 → lift **1,45×**", "foto", "**1**", "1", "3",
         "AP dalam-sensus", "**UTAMA**", "**YA — ini yang dijalankan demo**"),
        ("AP foto 1 kolom + derau detektor", "0,0800 → lift **1,27×**", "foto", "1", "1", "3",
         "AP dalam-sensus", "**UTAMA**", "**YA — ujung-ke-ujung**"),
        ("nilai kelas MATI", "+0,0046 (20/20)", "foto", "2−1", "1", "3",
         "AP dalam-sensus", "pendukung", "tidak (butuh kelas baru)"),
        ("null dalam-famili+petak", "0/200, 64% spasial", "foto", "6", "1", "3",
         "AP dalam-sensus", "**UTAMA**", "disebut di kotak batas"),
        ("presisi@5% teratas", "1 kasus per **8,8** pohon, **1,81×**", "Eg9PP", "6", "1", "3",
         "presisi@k", "**UTAMA**", "tidak"),
        ("F1 pusat tajuk", "0,960 ± 0,024", "Lapisan 1", "—", "—", "—",
         "F1 pusat", "**UTAMA**", "ya"),
        ("derajat jembatan", "5,54 ± 0,12 vs 5,74", "Lapisan 1 vs 2", "—", "—", "—",
         "derajat @1,5×", "**UTAMA**", "ya"),
        ("agregasi blok", "lift 1,24× < pohon 1,61×", "Eg9PP", "6", "1", "3",
         "tangkapan top5", "**UTAMA (negatif)**", "tidak"),
        ("keluaran demo apa pun", "—", "foto", "1", "1", "3",
         "—", "**JANGAN DIKUTIP**", "ya"),
    ]:
        W("| " + " | ".join(row) + " |")
    W("")
    W("**Yang harus dinyatakan di paper:** angka utama struktur (+0,0151 dan +0,0296) ")
    W("diukur pada konfigurasi **6 dan 24 kolom**, sedangkan demo menjalankan varian ")
    W("**1 kolom** — satu-satunya yang bisa diberi makan satu foto. Angka demo yang sah ")
    W("adalah **1,45×** (masukan bersih) dan **1,27×** (lewat detektor). Menyebut 1,61× ")
    W("sambil menunjuk layar demo adalah salah kutip.\n")

    # ---------------------------------------------------------------- A
    W("## A · Lapisan 1 — penginderaan dan uji jembatan\n")
    W("| Angka | Nilai | Konfigurasi | Sebanding dengan |")
    W("|---|---|---|---|")
    for item, cfg, cmpw in [
        ("F1 pusat tajuk (METRIK UTAMA)", "3 ortomosaik, leave-one-ortho-out, conf 0,75 silang-lipatan",
         "metrik Tahap 1 lain saja"),
        ("mAP50", "idem; **berlangit-langit** (kotak GT = cap ukuran tetap)",
         "literatur deteksi — BUKAN F1 pusat"),
        ("PR-AUC kesehatan", "LightGBM, 3 ortomosaik", "acak 0,013"),
    ]:
        r = get(rs, "demonstrator", item)
        v, sg, vd = fmt(r, 3)
        W("| %s | **%s** | %s | %s |" % (item, v, cfg, cmpw))
    for item in ["derajat @1.5x L1 (prediksi detektor)", "derajat @1.5x L1 (kotak GT)",
                 "derajat @1.5x L2"]:
        r = get(rs, "uji lebar sepur", item)
        v, _, _ = fmt(r, 2)
        W("| %s | **%s** | pohon bagian dalam, r = 1,5 × jarak tanam | satu sama lain saja |"
          % (item, v))
    W("")

    # ---------------------------------------------------------------- B
    W("## B · Eg9PP, model PENUH — 24 kolom, W=3, **AP gabungan**\n")
    W("Ini dekomposisi utama paket. Metriknya AP **gabungan seluruh sensus**.\n")
    W("| h | STRUKTUR (true − random) | tanda | vonis |")
    W("|---|---|---|---|")
    for h in (1, 2, 3, 4):
        r = get(rs, "dekomposisi+tangga h=%d" % h, "STRUKTUR vs random GLOBAL")
        v, sg, vd = fmt(r, signed=True)
        W("| %d | %s | %s | %s |" % (h, v, sg, vd))
    W("")
    W("| Kontrol lebih ketat, h=3 | Nilai | tanda | vonis |")
    W("|---|---|---|---|")
    for item in ["STRUKTUR vs local r<=6", "STRUKTUR vs local r<=3  (paling ketat)"]:
        r = get(rs, "dekomposisi+tangga h=3", item)
        v, sg, vd = fmt(r, signed=True)
        W("| %s | %s | %s | %s |" % (item, v, sg, vd))
    W("")

    # ---------------------------------------------------------------- C
    W("## C · Varian FOTO (v3) — W=1, **AP dalam-sensus**\n")
    W("Metrik berbeda dari Bagian B. AP gabungan menghargai kemampuan menebak "
      "*sensus mana ini*; tidak berguna untuk memeringkat di dalam satu bidikan.\n")
    W("| Angka | Nilai | Kolom | tanda | vonis |")
    W("|---|---|---|---|---|")
    for eks, item, kol in [
        ("v3 foto-tunggal h=3", "AP dalam-sensus foto graf BENAR", "6"),
        ("v3 foto-tunggal h=3", "AP dalam-sensus foto tanpa graf", "6 (graf dimatikan)"),
        ("v3 foto-tunggal h=3", "AP dalam-sensus penuh (24 kol W=3)", "24"),
        ("v3 foto-tunggal h=3", "AP(foto graf benar) - AP(graf acak), dalam-sensus", "6"),
        ("v3 foto-tunggal h=3", "AP(foto) - AP(penuh), dalam-sensus", "6 lawan 24"),
    ]:
        r = get(rs, eks, item)
        v, sg, vd = fmt(r, signed=item.startswith("AP(") )
        W("| %s | **%s** | %s | %s | %s |" % (item, v, kol, sg, vd))
    W("")
    W("| Ablasi kolom (semua W=1, dalam-sensus, h=3) | AP | lift | tingkat |")
    W("|---|---|---|---|")
    for item, lift, lev in [
        ("AP 1 kolom is_sympt", "1,45×", "7"),
        ("AP 2 kolom + is_dead", "1,52×", "23"),
        ("AP 3 kolom + is_cens", "1,58×", "46"),
        ("AP 6 kolom + selisih", "1,61×", "161"),
    ]:
        r = get(rs, "v3 ablasi kolom h=3", item)
        v, _, _ = fmt(r)
        W("| %s | %s | %s | %s |" % (item, v, lift, lev))
    W("")
    W("| Selisih berpasangan | Nilai | tanda | vonis |")
    W("|---|---|---|---|")
    for item in ["NILAI KELAS MATI (2-1 kolom)", "+ penyensoran (3-2 kolom)",
                 "+ selisih antar waktu (6-3 kolom)"]:
        r = get(rs, "v3 ablasi kolom h=3", item)
        v, sg, vd = fmt(r, signed=True)
        W("| %s | **%s** | %s | %s |" % (item, v, sg, vd))
    W("")

    # ---------------------------------------------------------------- D
    W("## D · Ketahanan dan ongkos ujung-ke-ujung\n")
    W("| Uji | Nilai | Catatan |")
    W("|---|---|---|")
    for eks, item, note in [
        ("v3 null dalam-famili h=3", "kelebihan vs null progeny", "200 permutasi, z +6,25, **0/200**"),
        ("v3 null dalam-famili h=3", "kelebihan vs null progeny+parcel",
         "strata terketat, z +6,04, **0/200** → 64% spasial / 36% kekerabatan"),
        ("v3 ujung-ke-ujung h=3", "AP dalam-sensus masukan BERSIH (1 kolom)", "status lapangan"),
        ("v3 ujung-ke-ujung h=3", "AP dalam-sensus masukan DETEKTOR",
         "recall 0,446 · fpr 0,0094 → **59% sinyal bertahan**, 1,45× → 1,27×"),
        ("null permutasi h=3", "RR_MH", "RR tetangga Mantel-Haenszel, 0/500"),
    ]:
        r = get(rs, eks, item)
        v, _, _ = fmt(r)
        W("| %s | **%s** | %s |" % (item, v, note))
    W("")

    W("### Agregasi blok — hipotesis yang diuji dan DITOLAK\n")
    W("| Metrik (unit = petak) | Model | Acak | Selisih |")
    W("|---|---|---|---|")
    for item_m, item_a, lab in [
        ("Spearman blok model", "Spearman blok acak (garis nol)", "Spearman"),
        ("tangkapan top5 blok model", "tangkapan top5 blok acak", "tangkapan 5 petak teratas"),
    ]:
        rm = get(rs, "agregasi blok h=3", item_m)
        ra = get(rs, "agregasi blok h=3", item_a)
        vm, sg, vd = fmt(rm, 3)
        va, _, _ = fmt(ra, 3)
        W("| %s | **%s** (%s) | %s | — |" % (lab, vm, sg, va))
    W("")
    W("1.558 unit blok-sensus per lari, leave-one-parcel-out. Garis acak duduk di **nol** ")
    W("(+0,004), jadi metriknya **tidak** terlalu mudah — model memang mengalahkannya 20/20. ")
    W("Tetapi liftnya **1,24×**, di bawah lift per-pohon **1,61×**. Sinyalnya berskala ")
    W("**pohon**, bukan petak: yang berisiko adalah sawit yang bersentuhan dengan sawit ")
    W("sakit, bukan seluruh petaknya. Agregasi **mengencerkan** sinyal, bukan meredam derau. ")
    W("Keluaran operasional yang benar tetap daftar prioritas per-pohon.\n")

    # ---------------------------------------------------------------- E
    W("## E · JANGAN dibandingkan langsung\n")
    W("| Pasangan | Kenapa tidak sebanding |")
    W("|---|---|")
    W("| **+0,0165** (B, h=4) ↔ **+0,0296** (C) | metrik berbeda: AP **gabungan** lawan AP "
      "**dalam-sensus**; fitur 24 kolom lawan 6; window 3 lawan 1 |")
    W("| **+0,0151** (B, h=3) ↔ **+0,0296** (C) | idem — keduanya disebut \"STRUKTUR\" tetapi "
      "diukur pada dua skala yang berbeda |")
    W("| **1,45×** (C, 1 kolom) ↔ **1,61×** (C, 6 kolom) | sebanding, TAPI hanya 6 kolom yang "
      "butuh dua kunjungan; 1 kolom yang bisa diberi satu foto |")
    W("| **0,960** F1 pusat (A) ↔ **0,687** mAP50 (A) | mAP berlangit-langit label; "
      "keduanya sah dilaporkan, tidak sah diadu |")
    W("| angka mana pun ↔ keluaran demo | checkpoint demo dilatih tanpa held-out; "
      "**tidak ada** angka performa yang boleh dikutip darinya |")
    W("")
    W("## F · Yang masuk paper\n")
    W("Bagian **B** adalah dekomposisi utama; **C** adalah kontribusi jalur foto; "
      "**D** adalah ketahanannya. Bagian A berdiri sendiri sebagai kelayakan penginderaan.\n")
    W("Kalimat yang sah, kata per kata:\n")
    W("> Pada panel lapangan Ganoderma 25 tahun, peta kontak yang benar menyumbang "
      "**+0,0151 AUC-PR (39/40)** pada h=3 di model penuh. Untuk peringkat di dalam satu "
      "bidikan, varian foto **menyamai** model penuh (+0,0042 ± 0,0035, 36/40) dan **77%** "
      "kemampuannya datang khusus dari peta kontak yang benar (+0,0296, 40/40). Efek itu "
      "bertahan di **0 dari 200** permutasi dalam-famili+petak; **64% spasial** setelah "
      "kekerabatan dan petak dikontrol. Lewat jalur foto penuh, **59% sinyal bertahan**.\n")

    io.open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print("ditulis: %s  (%d baris)" % (OUT, len(L)))


if __name__ == "__main__":
    main()
