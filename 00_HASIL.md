# SawitGuard-GNN — Pipeline dan Hasil

Satu berkas. Apa yang dibangun, apa yang dijalankan, apa hasilnya.
**Cakupan: data nyata saja.** Simulator SEIR sintetis dikeluarkan dari paper ini.

---

# BAGIAN 1 — PIPELINE

## Pertanyaan yang dijawab

> Diberi kebun sawit yang disurvei berkala, **pohon mana yang akan bergejala BSR dalam *h* sensus ke depan**, sehingga penyensusan berikutnya bisa diarahkan ke blok paling berisiko?

Targetnya **bukan** mendiagnosis pohon yang sudah sakit. Targetnya pohon yang **masih tampak sehat sekarang** tapi akan bergejala nanti. Setiap pohon yang sudah bergejala atau mati **dikeluarkan dari risk set**.

## Alur

```
  ┌──────────────────── LAPISAN 1 — Penginderaan ────────────────────┐
  │  Citra UAV RGB nadir · 3 ortomosaik · GSD ≈ 8,7 cm/px            │
  │                              │                                   │
  │   [1] Deteksi tajuk          │  YOLO11n        → mAP50 0,758     │
  │                              ▼                                   │
  │   [2] Kesehatan tajuk        │  LightGBM       → PR-AUC 0,182    │
  │                              ▼                                   │
  │   [3] Geometri kebun         │  5.077 pohon + posisi global      │
  └──────────────────────────────┼───────────────────────────────────┘
                                 │
                    ✂  TERPUTUS. Tidak ada dataset yang punya
                       citra UAV DAN penyebaran per-pohon sekaligus.
                       Yang bisa dilakukan: MENGUKUR apakah
                       antarmukanya cocok.

                       Derajat @ r = 1,5 × jarak tanam
                         Lapisan 1  5,62  ┐  selisih 2%
                         Lapisan 2  5,74  ┘  keduanya kisi segitiga
                                 │
  ┌──────────────────────────────┼───────────────────────────────────┐
  │                              ▼      LAPISAN 2 — Peramalan        │
  │  Eg9PP: 1.200 sawit · 45 sensus · 25 tahun · Ganoderma NYATA     │
  │                              │                                   │
  │   [4] Graf kontak akar       │  r = 1,5 × jarak tanam            │
  │                              ▼  3.354 sisi · derajat 5,59        │
  │   [5] Peramalan              │  MLP / STGNN / STGNN+SI(D)        │
  │                              ▼                                   │
  │   [6] Peringkat risiko       │  AUC-PR pada horizon h            │
  └──────────────────────────────────────────────────────────────────┘
```

## Dua sumber bukti — sengaja TIDAK digabung

| | **Lapisan 1** — Roboflow ds_B | **Lapisan 2** — Eg9PP |
|---|---|---|
| Citra UAV | ✅ nyata | ❌ tidak ada |
| Penyakit | ⚠ kesehatan tajuk generik, **BUKAN BSR** | ✅ **Ganoderma terverifikasi lapangan** |
| Rentang waktu | 1 tanggal | **45 sensus / 25 tahun** |
| Unit | 5.077 pohon unik | 1.200 sawit |
| Blok spasial | 3 ortomosaik | 2 parcel |
| Pertanyaannya | apakah tahap penginderaan layak? | apakah metodenya bekerja di lapangan nyata? |

Kebun berbeda, zaman berbeda (petak Eg9PP dibongkar 2012, citra DJI pasca-2013), tanpa kunci join dan tanpa georeferensi. **Keterbatasan ini diukur, bukan disembunyikan.**

## Model yang dibandingkan

| Model | Graf | Parameter | Menguji |
|---|---|---|---|
| `MLPBaseline` | ❌ | 3.713 | baseline — cukupkah waktu + genotipe saja? |
| `STGNN` (`n_rel=1`) | ✅ | 8.875 | apakah graf menambah nilai? |
| `STGNN_SID` | ✅ + kepala mekanistik | 8.878 | apakah lapisan epidemiologi terlatih menambah nilai? |

> **Koreksi 2026-07-24.** Tabel ini sebelumnya memuat 4.225 / 9.422 / 9.425. Angka-angka
> itu berasal dari konfigurasi **sintetis** (`models.py`, `in_dim=32`, `n_rel=3`), bukan dari
> run Eg9PP. Nilai yang benar dicetak oleh `run_real.log` baris 7–11 pada `in_dim=24` dengan
> `models_real.STGNN` yang sudah membuang parameter `attn`. Yang 9.425 bahkan tidak berasal
> dari jalur kode mana pun: ia STGNN akar `n_rel=3` ditambah kepala SI(D) nyata. Kepala SI(D)
> menyumbang **3 parameter** (`rates` 2 + `res_scale` 1), bukan 112 seperti kepala SEIR sintetis.

## Tiga *graph view* — inti eksperimennya

| View | Definisi | Menjawab |
|---|---|---|
| `true` | graf kontak sebenarnya | — |
| `random` | **derajat tiap pohon dipertahankan, struktur dihancurkan** (rewire dalam-parcel) | butuh peta yang *benar*, atau cukup punya graf apa pun? |
| `random_local` | derajat **dan lokalitas** dipertahankan — pasangan diacak tapi tiap sisi baru wajib ≤ 3 jarak tanam | butuh peta yang benar, atau cukup **berada di dekatnya**? |
| `zero` | tanpa graf | apakah graf penting sama sekali? |

**Kenapa `random_local` perlu ada.** Semua 3.354 sisi asli panjangnya **persis 1,000** — pada kisi
segitiga, r = 1,5 hanya menjangkau cangkang pertama. View `random` global menghubungkan pohon
berjarak median **13,2** jarak tanam, jadi `true − random` membandingkan *"6 tetangga langsung"*
lawan *"6 pohon di seberang petak"* — ia bisa saja mengukur **lokalitas**, bukan **peta yang benar**.
Rewire jaga-jarak mustahil di sini (satu-satunya pasangan berjarak 1,0 adalah tetangga yang benar,
jadi ia akan mereproduksi graf aslinya), sehingga kontrolnya dibatasi radius sebagai gantinya.

Dari sini datang **dekomposisi** yang memecah selisih STGNN−MLP jadi tiga bagian yang saling menjumlah:

```
temporal    = nograph − MLP        nilai memodelkan waktu + genotipe
prevalensi  = random  − nograph    nilai punya graf APA PUN
STRUKTUR    = true    − random     nilai peta kontak yang BENAR
```

## Protokol evaluasi

| | Lapisan 1 | Lapisan 2 |
|---|---|---|
| Split | **leave-one-ortho-out**, 3 lipatan | **leave-one-parcel-out**, 2 lipatan |
| Kenapa | hanya 3 ortomosaik; split acak **bocor 100%** | 0 dari 3.354 sisi lintas-parcel ⇒ memisahkan lipatan **tak memutus graf** |
| Kontrol | — | ke-14 famili ada di **kedua** parcel ⇒ lipatan tak terkonfound genotipe |
| Seed | — | 20 seed × 2 lipatan = **40 pasangan** |
| Metrik | PR-AUC / ROC-AUC / mAP | **AUC-PR** (pos-rate 1,7–6,0%) |
| Vonis | selisih di dalam 1 std ⇒ `INCONCLUSIVE`, dilaporkan apa adanya | idem |

---

# BAGIAN 2 — HASIL

## 2.1 Lapisan 1 — apakah penginderaan UAV layak?

| Tahap | Metrik | Hasil | Catatan |
|---|---|---|---|
| Deteksi tajuk | mAP50 | **0,758** | YOLO11n, **1 lipatan saja**, 15 epoch |
| | mAP50-95 | 0,524 | Presisi 0,862 · Recall 0,683 |
| Kesehatan tajuk | **PR-AUC** | **0,182 ± 0,059** | acak = 0,013 → **14× di atas acak** |
| | ROC-AUC | 0,861 | |
| | per lipatan | 0,264 / 0,155 / 0,126 | |
| `is_unbalance=True` | PR-AUC | 0,181 ± 0,091 | Δ0,001 ≪ 1 std → **DITOLAK** |

**Bacaannya:** deteksi tajuk sawit dari UAV berhasil; penilaian kesehatan jauh di atas acak. Tapi seluruhnya bersandar pada **66 pohon sakit unik**.

## 2.2 Lapisan 2 — Eg9PP, lapangan nyata 25 tahun

### Dekomposisi, seluruh horizon

Semua **20 seed × 2 lipatan = 40 pasangan**, kecuali kolom SI(D) (n=8).

| h | temporal | prevalensi | **STRUKTUR** | total STGNN−MLP | kepala SI(D) *(n=8)* |
|---|---|---|---|---|---|
| 1 | +0,0044 (34/40) INCONCL | +0,0026 (32/40) INCONCL | +0,0044 (30/40) **INCONCL** | +0,0114 (38/40) POS | −0,0115 **NEG** |
| 2 | +0,0034 (27/40) INCONCL | +0,0029 (33/40) INCONCL | **+0,0098 (37/40) POS** | +0,0161 (37/40) POS | −0,0230 **NEG** |
| **3** | +0,0021 (21/40) INCONCL | +0,0021 (25/40) INCONCL | **+0,0151 (39/40) POS** | +0,0193 (36/40) POS | −0,0293 **NEG** (4/40) |
| 4 | +0,0011 (19/40) INCONCL | +0,0018 (26/40) INCONCL | **+0,0165 (39/40) POS** | +0,0195 (35/40) POS | −0,0426 **NEG** |

**Struktur menguat seiring horizon** (+0,004 → +0,017 dari h=1 ke h=4), sementara temporal justru
**meluruh** (+0,0044 → +0,0011). Makin jauh ke depan, makin peta kontak yang menentukan.

**Pemeriksaan silang.** Baris h=3 dihasilkan dua kali oleh dua implementasi berbeda pada perangkat
berbeda — `run_real.py` (CPU) dan `run_v2.py` (CUDA) — dan keempat komponennya **identik sampai
empat desimal beserta sign-count-nya** (+0,0021 21/40 · +0,0021 25/40 · +0,0151 39/40 · +0,0193 36/40).

### Uji permutasi terkontrol-genotipe (500 permutasi, diacak **dalam famili**)

| h | RR teramati | null | kelebihan | z | mencapai |
|---|---|---|---|---|---|
| 1 | 5,17× | 4,13 ± 0,24 | **1,25×** | +4,4 | **0/500** |
| 2 | 4,74× | 3,73 ± 0,21 | **1,27×** | +4,8 | **0/500** |
| 3 | 4,47× | 3,46 ± 0,19 | **1,29×** | +5,3 | **0/500** |
| 4 | 4,06× | 3,14 ± 0,17 | **1,29×** | +5,5 | **0/500** |

Null diacak **dalam famili**, bukan acak bebas — kalau tidak, kekerabatan yang ditanam berdekatan akan dihitung sebagai efek penularan. Null naif memberi 1,36×; yang benar 1,29×.

## 2.3 Tangga lokalitas — apakah STRUKTUR bertahan terhadap kontrol yang lebih ketat?

Kontrol dinaikkan bertingkat, derajat tiap pohon **selalu** dipertahankan. Makin ke kanan, graf
pembandingnya makin mirip graf asli — makin sulit dimenangkan.

| kontrol | panjang sisi median | sisa sisi asli |
|---|---|---|
| `random` (global) | 13,2 | 1,0% |
| `random_local` r ≤ 6 | 4,0 | 6,7% |
| `random_local` r ≤ 3 | 2,0 | 22,0% |

**STRUKTUR = true − kontrol**, 20 seed × 2 lipatan = 40 pasangan:

| h | vs `random` global | vs local r ≤ 6 | vs local r ≤ 3 (terketat) | efek yang bertahan |
|---|---|---|---|---|
| 1 | +0,0044 (30/40) INCONCL | +0,0045 (36/40) **POS** | +0,0035 (35/40) **POS** | 79% |
| 2 | +0,0098 (37/40) POS | +0,0097 (38/40) **POS** | +0,0076 (38/40) **POS** | 78% |
| 3 | +0,0151 (39/40) POS | +0,0146 (40/40) **POS** | +0,0128 (36/40) **POS** | 85% |
| 4 | +0,0165 (39/40) POS | +0,0154 (40/40) **POS** | +0,0139 (39/40) **POS** | 84% |

**Efeknya tidak meluruh.** Bahkan ketika graf pembanding menghubungkan tiap pohon ke 6 pohon yang
*dekat tapi salah*, model tetap butuh peta yang benar — **78–85%** efeknya bertahan. Jadi yang
diukur bukan sekadar lokalitas. Di h=1, kontrol lokal bahkan **lebih tajam** daripada kontrol global
(POS 36/40 vs INCONCLUSIVE 30/40) karena rewire global jauh lebih berderau.

## 2.4 Empat upaya menaikkan mutu model — semuanya GAGAL

Dilaporkan karena hasil negatif adalah hasil, dan karena satu di antaranya nyaris lolos sebagai
"perbaikan".

| Upaya | Alasan biologisnya | Hasil (h=3, 40 pasangan) | Vonis |
|---|---|---|---|
| **Umur inokulum + paparan kumulatif** (4 kolom: lama bergejala, lama mati, tahun-bergejala, tahun-mati) | tunggul yang mati 5 tahun lalu inokulumnya lebih besar daripada yang baru mati | AP 0,1737 vs 0,1818 → **−0,0080** (6/40) | tidak menolong |
| **Difusi 2-hop** (relasi kedua: terjangkau 2 langkah, bukan tetangga langsung) | Ganoderma bisa menjangkau melampaui tetangga langsung | AP 0,1731 → **−0,0086** (4/40) | **NEG** |
| **Radius graf lebih besar** (r = 1,8 / 2,05 / 3,05 → derajat 10,9 / 16,1 / 30,8) | kontak akar mungkin lebih jauh dari cangkang pertama | −0,0021 / −0,0025 / **−0,0069** | tidak menolong; r=3,05 **NEG** |
| **Jendela riwayat lebih panjang** (5 dan 8 sensus, dari 3) | BSR berkembang lambat; penumpukan tekanan butuh waktu | +0,0023 (7/40) dan −0,0000 (6/40) | **INCONCLUSIVE** |

⚠ **Jendela panjang nyaris lolos sebagai perbaikan palsu.** Diukur apa adanya, WINDOW=5 memberi
AP 0,1867 vs 0,1824 — kelihatan seperti kenaikan. Tapi jendela lebih panjang **membuang sensus-sensus
awal** yang justru paling sulit diprediksi, jadi himpunan ujinya berbeda. Setelah semua varian
dinilai pada **himpunan contoh identik**, kenaikannya hilang: +0,0023 INCONCLUSIVE untuk w=5, dan
**persis −0,0000** untuk w=8. Seluruh "perbaikan" itu adalah efek seleksi.

**Bacaannya:** enam tetangga langsung membawa seluruh sinyal yang bisa ditangkap. Menambah
jangkauan, menambah riwayat, atau menambah penanda umur inokulum tidak menambahkan apa pun —
dan beberapa di antaranya justru mengencerkannya.

---

# BAGIAN 3 — APA ARTINYA

**① Peta kontak yang benar memang bekerja — dan bertahan terhadap tiga kontrol berbeda.**
Struktur menyumbang **+0,0151 (39 dari 40 pasangan)** pada h=3, POS juga di h=2 dan h=4. Ia lolos tiga uji yang masing-masing mematikan satu penjelasan tandingan:
**(a)** view `random` mempertahankan derajat tiap pohon → bukan artefak jumlah tetangga;
**(b)** null permutasi diacak **dalam famili** → bukan artefak kekerabatan yang ditanam berdekatan (kelebihan 1,25–1,29×, **0 dari 500** permutasi mencapainya);
**(c)** tangga lokalitas → **bukan** sekadar artefak "dekat itu penting". Ketika graf pembanding dipaksa tetap lokal (tiap sisi ≤ 3 jarak tanam), **78–85% efeknya tetap bertahan** di keempat horizon.

**② Menambahkan lapisan epidemiologi terlatih justru MERUGIKAN.**
Kepala SI(D) **NEG di keempat horizon**, memburuk seiring horizon (−0,012 → −0,043), dan hanya menang 0–4 dari 8–40 pasangan. Dugaan mudahnya — inisialisasi laju yang tidak netral — sudah diuji: ia memperbaiki h=1 saja, dan **tidak** memperbaiki h≥2. Struktur mekanistik yang ditempelkan ke model tidak membuatnya lebih baik.

**③ Sebagian besar "efek tetangga" ternyata confounding waktu kalender.**
RR gabungan 4,47× **runtuh jadi 1,65×** setelah stratifikasi Mantel-Haenszel per sensus. Kalau angka 4,47× dilaporkan tanpa stratifikasi, klaimnya menggelembung hampir tiga kali lipat. Yang bertahan setelah dikoreksi tetap nyata (z +5,6, 0/500) — tapi jauh lebih kecil.

**④ Enam tetangga langsung sudah memuat seluruh sinyal yang bisa ditangkap.**
Empat upaya menaikkan mutu model — umur inokulum, difusi 2-hop, radius lebih besar, jendela lebih panjang — **semuanya gagal**, dua di antaranya NEG. Satu di antaranya (jendela panjang) sempat tampak seperti kenaikan +0,004 sampai himpunan contohnya disamakan, lalu jatuh ke nol. Model ini sudah menyentuh langit-langit informasi yang tersedia di data tanpa citra.

**Dan satu yang harus dinyatakan, bukan disembunyikan:** temporal dan prevalensi **INCONCLUSIVE di hampir semua horizon**. Itu bukan berarti waktu tidak penting — itu berarti **Eg9PP tak punya citra**, sehingga riwayat pohon itu sendiri selagi asimptomatik terbukti **persis nol**. Lengan temporal memang tak punya apa pun untuk dibawa. Dekomposisi ini menanyakan tiga pertanyaan, tapi hanya satu lengan yang benar-benar terisi.

---

# BAGIAN 4 — BATAS

| Batas | Angka | Akibatnya |
|---|---|---|
| Positif Lapisan 1 | **66 pohon unik** (17/31/18 per lipatan) | hasil kesehatan **underpowered**; std dari 3 angka |
| Label Lapisan 1 | kesehatan tajuk generik | **BUKAN BSR**, tanpa verifikasi lapangan |
| Deteksi YOLO | **1 lipatan, 15 epoch** | mAP50 0,758 bukan mean ± std; 3-fold penuh belum dijalankan |
| Blok spasial Eg9PP | **2 parcel** | ±0,008 itu derau optimisasi, bukan ketidakpastian data. Efek beda **2,6×** antar blok (44A +0,0084 vs 44B +0,0219) |
| ~~Kontrol `random` terlalu lemah~~ | **DITUTUP** | Dulu mata rantai terlemah: `random` global menghubungkan pohon berjarak median 13,2 sehingga bisa mengukur lokalitas, bukan peta. Kini diuji dengan tangga `random_local` (≤6 dan ≤3 jarak tanam) dan **78–85% efeknya bertahan** di keempat horizon. Rewire jaga-jarak murni tetap mustahil di r=1,5 — semua sisi asli panjangnya 1,0 — jadi kontrol berbatas-radius ini adalah yang terketat yang bisa dibangun |
| Horizon h≠3 untuk SI(D) | **4 seed** (n=8) | dekomposisi utama kini 20 seed di keempat horizon; hanya kepala SI(D) yang belum |
| Sensor Eg9PP | 498–621 negatif/horizon | nasib sesungguhnya tak diketahui; bias kecil ke atas pada spesifisitas |
| Kepala SI(D) | 112 → **3 parameter** | kompartemen laten E **tak teramati**; tak bisa dibandingkan dengan varian SEIR mana pun |
| Uji jembatan | kotak **ground-truth**, bukan prediksi YOLO | angka 5,62 adalah **batas atas** |

## Satu klaim yang kami tarik sendiri

Laporan sebelumnya menyebut satu ortomosaik "kolaps" ke PR-AUC 0,030 sebagai bukti variasi antar-situs besar. Setelah deduplikasi ke pohon unik, ortomosaik itu mencapai **0,126** dan ketiga lipatan berada di 0,13–0,26. **Kolapsnya artefak pseudo-replikasi 29,8×, bukan efek situs.** Klaim ditarik.

---

## Peta berkas

| Ingin lihat | Buka |
|---|---|
| ringkasan semua angka | `00_RINGKASAN.csv` |
| tangga lokalitas + ablasi fitur | `layer2_real/results_v2.csv` · kode: `run_v2.py`, `dataset_v2.py`, `sweep_v2.py` |
| angka mentah Eg9PP | `layer2_real/results_real.csv` · log: `run_real.log` |
| angka mentah Lapisan 1 | `layer1_build/RESULTS_LAYER1.md` |
| fakta data yang boleh diklaim | `data_clean/DATASET_CARD.md` |
| bukti pseudo-replikasi 29,8× | `layer1_data_audit/AUDIT_REPORT.md` |
| penjaga kebocoran (~90 asersi) | `layer2_real/test_dataset.py` |
| status naskah & sisa pekerjaan | `README.md` |
