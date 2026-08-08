# Angka final — satu tabel, satu sumber

> Dihasilkan `make_final_table.py` dari `00_RINGKASAN.csv`. **Jangan sunting tangan.**
> Kolom **sebanding dengan** menentukan angka mana boleh diletakkan berdampingan.

## 0 · Keputusan: apa yang masuk paper, dan apa yang muncul di demo

Dua kolom terakhir adalah inti berkas ini. **PAPER** menandai angka utama; 
**DEMO** menandai apakah angka itu benar-benar terlihat di aplikasi. Paper 
tidak boleh mengutip angka yang konfigurasinya berbeda dari yang dijalankan demo 
tanpa menyebut perbedaannya.

| Angka | Nilai | Jalur | Kolom | W | h | Metrik | PAPER | DEMO |
|---|---|---|---|---|---|---|---|---|
| STRUKTUR true−random | +0,0151 (39/40) | Eg9PP | 24 | 3 | 3 | AP gabungan | **UTAMA** | tidak |
| STRUKTUR true−random | +0,0165 (39/40) | Eg9PP | 24 | 3 | **4** | AP gabungan | pendukung | tidak |
| STRUKTUR foto true−random | +0,0296 (40/40) | foto | 6 | 1 | 3 | AP dalam-sensus | **UTAMA** | tidak |
| foto − penuh | +0,0042 (36/40) | foto vs Eg9PP | 6 vs 24 | 1 vs 3 | 3 | AP dalam-sensus | **UTAMA** | tidak |
| AP foto 6 kolom | 0,1015 → lift **1,61×** | foto | 6 | 1 | 3 | AP dalam-sensus | pendukung | tidak |
| AP foto 1 kolom | 0,0916 → lift **1,45×** | foto | **1** | 1 | 3 | AP dalam-sensus | **UTAMA** | **YA — ini yang dijalankan demo** |
| AP foto 1 kolom + derau detektor | 0,0800 → lift **1,27×** | foto | 1 | 1 | 3 | AP dalam-sensus | **UTAMA** | **YA — ujung-ke-ujung** |
| nilai kelas MATI | +0,0046 (20/20) | foto | 2−1 | 1 | 3 | AP dalam-sensus | pendukung | tidak (butuh kelas baru) |
| null dalam-famili+petak | 0/200, 64% spasial | foto | 6 | 1 | 3 | AP dalam-sensus | **UTAMA** | disebut di kotak batas |
| presisi@5% teratas | 1 kasus per **8,8** pohon, **1,81×** | Eg9PP | 6 | 1 | 3 | presisi@k | **UTAMA** | tidak |
| F1 pusat tajuk | 0,960 ± 0,024 | Lapisan 1 | — | — | — | F1 pusat | **UTAMA** | ya |
| derajat jembatan | 5,54 ± 0,12 vs 5,74 | Lapisan 1 vs 2 | — | — | — | derajat @1,5× | **UTAMA** | ya |
| agregasi blok | lift 1,24× < pohon 1,61× | Eg9PP | 6 | 1 | 3 | tangkapan top5 | **UTAMA (negatif)** | tidak |
| keluaran demo apa pun | — | foto | 1 | 1 | 3 | — | **JANGAN DIKUTIP** | ya |

**Yang harus dinyatakan di paper:** angka utama struktur (+0,0151 dan +0,0296) 
diukur pada konfigurasi **6 dan 24 kolom**, sedangkan demo menjalankan varian 
**1 kolom** — satu-satunya yang bisa diberi makan satu foto. Angka demo yang sah 
adalah **1,45×** (masukan bersih) dan **1,27×** (lewat detektor). Menyebut 1,61× 
sambil menunjuk layar demo adalah salah kutip.

## A · Lapisan 1 — penginderaan dan uji jembatan

| Angka | Nilai | Konfigurasi | Sebanding dengan |
|---|---|---|---|
| F1 pusat tajuk (METRIK UTAMA) | **0,960 ± 0,024** | 3 ortomosaik, leave-one-ortho-out, conf 0,75 silang-lipatan | metrik Tahap 1 lain saja |
| mAP50 | **0,687 ± 0,071** | idem; **berlangit-langit** (kotak GT = cap ukuran tetap) | literatur deteksi — BUKAN F1 pusat |
| PR-AUC kesehatan | **0,182 ± 0,059** | LightGBM, 3 ortomosaik | acak 0,013 |
| derajat @1.5x L1 (prediksi detektor) | **5,54 ± 0,12** | pohon bagian dalam, r = 1,5 × jarak tanam | satu sama lain saja |
| derajat @1.5x L1 (kotak GT) | **5,62 ± 0,05** | pohon bagian dalam, r = 1,5 × jarak tanam | satu sama lain saja |
| derajat @1.5x L2 | **5,74** | pohon bagian dalam, r = 1,5 × jarak tanam | satu sama lain saja |

## B · Eg9PP, model PENUH — 24 kolom, W=3, **AP gabungan**

Ini dekomposisi utama paket. Metriknya AP **gabungan seluruh sensus**.

| h | STRUKTUR (true − random) | tanda | vonis |
|---|---|---|---|
| 1 | +0,0044 ± 0,0046 | 30/40 | INCONCLUSIVE |
| 2 | +0,0098 ± 0,0060 | 37/40 | POS |
| 3 | +0,0151 ± 0,0081 | 39/40 | POS |
| 4 | +0,0165 ± 0,0097 | 39/40 | POS |

| Kontrol lebih ketat, h=3 | Nilai | tanda | vonis |
|---|---|---|---|
| STRUKTUR vs local r<=6 | +0,0146 ± 0,0074 | 40/40 | POS |
| STRUKTUR vs local r<=3  (paling ketat) | +0,0128 ± 0,0084 | 36/40 | POS |

## C · Varian FOTO (v3) — W=1, **AP dalam-sensus**

Metrik berbeda dari Bagian B. AP gabungan menghargai kemampuan menebak *sensus mana ini*; tidak berguna untuk memeringkat di dalam satu bidikan.

| Angka | Nilai | Kolom | tanda | vonis |
|---|---|---|---|---|
| AP dalam-sensus foto graf BENAR | **0,1015 ± 0,0079** | 6 | — | — |
| AP dalam-sensus foto tanpa graf | **0,0632 ± 0,0031** | 6 (graf dimatikan) | — | — |
| AP dalam-sensus penuh (24 kol W=3) | **0,0973 ± 0,0107** | 24 | — | — |
| AP(foto graf benar) - AP(graf acak), dalam-sensus | **+0,0296 ± 0,0057** | 6 | 40/40 | POS |
| AP(foto) - AP(penuh), dalam-sensus | **+0,0042 ± 0,0035** | 6 lawan 24 | 36/40 | POS |

| Ablasi kolom (semua W=1, dalam-sensus, h=3) | AP | lift | tingkat |
|---|---|---|---|
| AP 1 kolom is_sympt | 0,0916 ± 0,0081 | 1,45× | 7 |
| AP 2 kolom + is_dead | 0,0962 ± 0,0081 | 1,52× | 23 |
| AP 3 kolom + is_cens | 0,0999 ± 0,0077 | 1,58× | 46 |
| AP 6 kolom + selisih | 0,1015 ± 0,0081 | 1,61× | 161 |

| Selisih berpasangan | Nilai | tanda | vonis |
|---|---|---|---|
| NILAI KELAS MATI (2-1 kolom) | **+0,0046 ± 0,0004** | 20/20 | POS |
| + penyensoran (3-2 kolom) | **+0,0037 ± 0,0008** | 20/20 | POS |
| + selisih antar waktu (6-3 kolom) | **+0,0016 ± 0,0007** | 20/20 | POS |

## D · Ketahanan dan ongkos ujung-ke-ujung

| Uji | Nilai | Catatan |
|---|---|---|
| kelebihan vs null progeny | **0,0268 ± 0,0043** | 200 permutasi, z +6,25, **0/200** |
| kelebihan vs null progeny+parcel | **0,0244 ± 0,0040** | strata terketat, z +6,04, **0/200** → 64% spasial / 36% kekerabatan |
| AP dalam-sensus masukan BERSIH (1 kolom) | **0,0916 ± 0,0081** | status lapangan |
| AP dalam-sensus masukan DETEKTOR | **0,0800 ± 0,0070** | recall 0,446 · fpr 0,0094 → **59% sinyal bertahan**, 1,45× → 1,27× |
| RR_MH | **1,6462** | RR tetangga Mantel-Haenszel, 0/500 |

### Agregasi blok — hipotesis yang diuji dan DITOLAK

| Metrik (unit = petak) | Model | Acak | Selisih |
|---|---|---|---|
| Spearman | **0,179 ± 0,003** (20/20) | 0,004 ± 0,020 | — |
| tangkapan 5 petak teratas | **0,185** (19/20) | 0,149 | — |

1.558 unit blok-sensus per lari, leave-one-parcel-out. Garis acak duduk di **nol** 
(+0,004), jadi metriknya **tidak** terlalu mudah — model memang mengalahkannya 20/20. 
Tetapi liftnya **1,24×**, di bawah lift per-pohon **1,61×**. Sinyalnya berskala 
**pohon**, bukan petak: yang berisiko adalah sawit yang bersentuhan dengan sawit 
sakit, bukan seluruh petaknya. Agregasi **mengencerkan** sinyal, bukan meredam derau. 
Keluaran operasional yang benar tetap daftar prioritas per-pohon.

## E · JANGAN dibandingkan langsung

| Pasangan | Kenapa tidak sebanding |
|---|---|
| **+0,0165** (B, h=4) ↔ **+0,0296** (C) | metrik berbeda: AP **gabungan** lawan AP **dalam-sensus**; fitur 24 kolom lawan 6; window 3 lawan 1 |
| **+0,0151** (B, h=3) ↔ **+0,0296** (C) | idem — keduanya disebut "STRUKTUR" tetapi diukur pada dua skala yang berbeda |
| **1,45×** (C, 1 kolom) ↔ **1,61×** (C, 6 kolom) | sebanding, TAPI hanya 6 kolom yang butuh dua kunjungan; 1 kolom yang bisa diberi satu foto |
| **0,960** F1 pusat (A) ↔ **0,687** mAP50 (A) | mAP berlangit-langit label; keduanya sah dilaporkan, tidak sah diadu |
| angka mana pun ↔ keluaran demo | checkpoint demo dilatih tanpa held-out; **tidak ada** angka performa yang boleh dikutip darinya |

## F · Yang masuk paper

Bagian **B** adalah dekomposisi utama; **C** adalah kontribusi jalur foto; **D** adalah ketahanannya. Bagian A berdiri sendiri sebagai kelayakan penginderaan.

Kalimat yang sah, kata per kata:

> Pada panel lapangan Ganoderma 25 tahun, peta kontak yang benar menyumbang **+0,0151 AUC-PR (39/40)** pada h=3 di model penuh. Untuk peringkat di dalam satu bidikan, varian foto **menyamai** model penuh (+0,0042 ± 0,0035, 36/40) dan **77%** kemampuannya datang khusus dari peta kontak yang benar (+0,0296, 40/40). Efek itu bertahan di **0 dari 200** permutasi dalam-famili+petak; **64% spasial** setelah kekerabatan dan petak dikontrol. Lewat jalur foto penuh, **59% sinyal bertahan**.
