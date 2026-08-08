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
  │   [1] Deteksi tajuk          │  YOLOv12n       → F1 pusat 0,960  │
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

                       Derajat @ r = 1,5 × jarak tanam (pohon dalam)
                         Lapisan 1  5,54 ± 0,12  ┐  dari prediksi YOLOv12
                         Lapisan 2  5,74         ┘  selisih 3,5%
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
| Deteksi tajuk | **F1 pusat** | **0,960 ± 0,024** | **metrik utama** — pusat tajuk pada pohon unik; YOLOv12n, **3 lipatan × 30 epoch**, satu lingkungan |
| | Presisi · Recall | 0,950 ± 0,019 · 0,971 ± 0,030 | ambang conf dipilih **silang-lipatan**; ketiganya sepakat di 0,75 |
| | RMSE pusat | 0,071 ± 0,011 × jarak tanam | F1 **datar** terhadap radius pencocokan ⇒ pusatnya memang tepat |
| | mAP50 | 0,687 ± 0,071 | metrik **sekunder, berlangit-langit** — kotak GT adalah cap ukuran tetap |
| | mAP50-95 | 0,425 ± 0,078 | idem; mekanismenya di `layer1_build/LABEL_QUALITY_AUDIT.md` |
| Kesehatan tajuk | **PR-AUC** | **0,182 ± 0,059** | acak = 0,013 → **14× di atas acak** |
| | ROC-AUC | 0,861 | |
| | per lipatan | 0,264 / 0,155 / 0,126 | |
| `is_unbalance=True` | PR-AUC | 0,181 ± 0,091 | Δ0,001 ≪ 1 std → **DITOLAK** |

**Bacaannya:** deteksi **pusat** tajuk sawit dari UAV berhasil meyakinkan (F1 0,960); penilaian kesehatan jauh di atas acak. Jarak antara F1 pusat 0,960 dan mAP50 0,687 bukan cacat model — kotak kebenaran-dasarnya cap berukuran tetap, sehingga mAP punya langit-langit yang tidak bergantung model. Yang dikonsumsi Lapisan 2 adalah koordinat, bukan kotak. Tapi sisi kesehatan seluruhnya tetap bersandar pada **66 pohon sakit unik**.

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

## 2.5 Varian foto-tunggal (v3) — riwayat waktu ternyata tidak dibutuhkan

Pertanyaannya praktis: model utama meminta 24 kolom, tetapi satu foto drone hanya bisa
memberi posisi, kondisi tajuk, dan graf tetangga. Bisakah Lapisan 2 dilatih ulang **hanya**
dengan itu? Kode: `layer2_real/dataset_v3.py` · `run_v3.py` · `run_v3_perm.py`.

**Metriknya harus berubah dulu, dan itu temuan tersendiri.** AP gabungan memberi nilai pada
kemampuan menebak *sensus mana ini* — berguna untuk angka gabungan, nol guna untuk
memeringkat pohon di dalam satu bidikan. Karena itu v3 dinilai dengan **AP dalam-sensus**:
AP dihitung per sensus lalu dirata-rata. 20 seed × 2 lipatan = 40 pasangan, h=3.

| | AP gabungan | **AP dalam-sensus** |
|---|---|---|
| penuh (24 kolom, W=3) | 0,1818 ± 0,0077 | 0,0973 ± 0,0107 |
| foto — tanpa graf | 0,0468 ± 0,0003 | 0,0632 ± 0,0031 |
| foto — graf acak | 0,0974 ± 0,0072 | 0,0719 ± 0,0064 |
| **foto — graf benar** | 0,1259 ± 0,0009 | **0,1015 ± 0,0079** |

**Model penuh kehilangan 47% nilainya** begitu dinilai dalam-sensus (0,1818 → 0,0973).
Separuh angka gabungan itu adalah model menebak tanggal, bukan menilai pohon. Model foto
nyaris tidak kehilangan apa pun (0,1259 → 0,1015), karena ia tidak punya waktu untuk
disandari.

| putusan berpasangan (dalam-sensus) | selisih | tanda | vonis |
|---|---|---|---|
| graf apa pun (random − nograph) | +0,0087 ± 0,0047 | 39/40 | POS |
| **PETA BENAR (true − random)** | **+0,0296 ± 0,0057** | **40/40** | **POS** |
| harga kepraktisan (foto − penuh) | +0,0042 ± 0,0035 | 36/40 | POS |

`nograph` mendarat di 0,0468 melawan laju dasar 0,0468 — **persis**. Blok STATE terbukti
konstan 0 di risk set, jadi tanpa graf v3 benar-benar buta; seluruh kemampuannya datang
dari graf, dan **77% di antaranya khusus dari peta kontak yang BENAR**.

**Harga kepraktisannya nol.** Pada tugas yang sebenarnya, model foto menyamai model penuh.
Marginnya +0,0042 dengan std 0,0035 — lolos ambang, tetapi hanya 1,2 std, jadi klaim yang
sah adalah "tidak lebih lemah", **bukan** "lebih unggul".

### Uji permutasi dalam-famili untuk v3 (200 permutasi per strata)

v3 membuang genotipe, yang **melanggar larangan #5**. Kontaminasi famili karena itu wajib
diukur, bukan diasumsikan. Lintasan per-pohon ditukar hanya antar pohon sefamili; kisi tidak
pernah bergerak, sehingga komposisi famili tiap ketetanggaan dipertahankan dan hanya susunan
spasial halus yang dihancurkan.

| strata | teramati | null | kelebihan | z | perm ≥ teramati |
|---|---|---|---|---|---|
| `progeny` | 0,1016 | 0,0748 ± 0,0043 | +0,0268 | +6,25 | **0/200** |
| `progeny+parcel` (terketat) | 0,1016 | 0,0772 ± 0,0040 | +0,0244 | +6,04 | **0/200** |

Terhadap garis tanpa-graf 0,0632, pada strata terketat:

```
kekerabatan + petak   +0,0140    36%
susunan spasial       +0,0244    64%
```

**Bacaannya:** kontaminasi famili itu **nyata dan sebesar 36%** — bukan nol, bukan
segalanya. Yang tersisa setelah famili dan petak dipegang tetap adalah efek spasial, dan
tak satu pun dari 200 permutasi menyentuhnya. Perhatikan arah bacanya: di uji ini
**kemampuan yang bertahan di bawah null adalah kabar buruk**, kebalikan dari kebanyakan uji
permutasi.

Rasio graf-benar terhadap tanpa-skill dalam-sensus adalah **1,61×**, sementara RR tetangga
Mantel-Haenszel di §2.2 adalah **1,65×**. Dua metode yang sama sekali berbeda — GNN dan
epidemiologi klasik — mendarat di besaran yang sama.

Pencabutan larangan #5 **hanya berlaku untuk v3**, karena hanya v3 yang punya null yang
menguantifikasi kontaminasinya. Rinciannya di `layer2_real/INTERFACE.md`.

### Ongkos ujung-ke-ujung: apa yang hilang saat masukan datang dari detektor

Sampai di sini v3 masih dinilai dengan status Eg9PP yang **terverifikasi lapangan**. Di jalur
foto, kolom itu diisi kelas `Unhealthy` detektor. Ongkos substitusinya diukur di
`layer2_real/run_v3_noisy.py`: dilatih pada status bersih, **diuji pada status berderau** —
meniru penyerahan sesungguhnya, di mana tidak ada label lapangan di kebun tujuan untuk
melatih ulang.

Laju detektor diukur dari ds_B pada conf 0,75: **recall 0,446 · fpr 0,0094**
(`layer1_build/unhealthy_threshold.py`).

| masukan | AP dalam-sensus | lift atas tanpa-skill |
|---|---|---|
| status lapangan (bersih) | 0,0916 ± 0,0081 | 1,45× |
| **keluaran detektor (berderau)** | **0,0800 ± 0,0070** | **1,27×** |
| garis tanpa-skill | 0,0632 | 1× |

**59% sinyal bertahan.** Substitusi detektor memakan 41% — nyata, tetapi tidak menghapus
efeknya. Perhatikan model di sini memakai **satu kolom** (`is_sympt`), satu-satunya yang dapat
diisi satu foto; itu sebabnya garis bersihnya 0,0916, bukan 0,1015 versi enam kolom.

### Berapa nilainya kalau detektor bisa membedakan MATI dari BERGEJALA

Jalur foto memakai satu kolom biner, dan pada ubin khas itu hanya menghasilkan **dua
tingkat** skor. Menambah kelas ke detektor adalah pekerjaan berminggu-minggu, jadi
pertanyaan yang benar bukan "bisakah" melainkan **"berapa nilainya"** — dan itu
terjawab tanpa menyentuh detektor, karena Eg9PP menyimpan S / D / C terpisah.
Kode: `layer2_real/run_v3_cols.py`. Semua W=1, graf benar, 10 seed × 2 lipatan.

| kolom | AP dalam-sensus | lift | tingkat yang mungkin |
|---|---|---|---|
| 1 — `is_sympt` (yang detektor beri **sekarang**) | 0,0916 ± 0,0081 | 1,45× | 7 |
| 2 — **+ `is_dead`** (kalau ada kelas mati) | **0,0962 ± 0,0081** | **1,52×** | **23** |
| 3 — + `is_cens` (mustahil dari foto) | 0,0999 ± 0,0077 | 1,58× | 46 |
| 6 — + selisih antar sensus (butuh 2 kunjungan) | 0,1015 ± 0,0081 | 1,61× | 161 |

| selisih berpasangan | | tanda | vonis |
|---|---|---|---|
| **nilai kelas MATI** (2 − 1 kolom) | **+0,0046 ± 0,0004** | **20/20** | **POS** |
| + penyensoran (3 − 2 kolom) | +0,0037 ± 0,0008 | 20/20 | POS |
| + selisih antar waktu (6 − 3 kolom) | +0,0016 ± 0,0007 | 20/20 | POS |

**Bacaannya:** menambah kelas "mati" pada detektor adalah **perolehan tunggal
terbesar yang masih tersedia** — +0,0046 dengan std 0,0004, yaitu 11,5 std, dan
20/20 tanda searah. Ia merebut **47%** dari seluruh jarak antara 1 kolom dan 6 kolom,
dan melipattigakan granularitas (7 → 23 tingkat). Dua sisanya di luar jangkauan foto
tunggal: penyensoran tidak pernah terlihat dari udara, dan selisih antar waktu butuh
kunjungan kedua.

**Yang TIDAK dijanjikan angka ini:** lebih banyak pita di ubin yang sehat. "Tingkat
yang mungkin" dihitung pada Eg9PP yang laju gejalanya **40,6%**; ubin drone khas
1,1%. Ubin dengan satu pohon sakit tetap memberi dua pita berapa pun kolomnya —
karena dua pita memang kebenarannya di sana. Yang naik adalah **mutu peringkat**,
bukan resolusi tampilan.

### Kalibrasi: seberapa jauh skor dari peluang sungguhan

Larangan "jangan sajikan `sigmoid(logit)` sebagai persentase" selama ini berupa
aturan. Sekarang ia punya angka. Diukur leave-one-parcel-out pada 40.828 contoh uji,
varian v3 6-kolom:

| `sigmoid(skor)` | kalau dibaca % | sesungguhnya sakit | meleset |
|---|---|---|---|
| 0,50 – 0,60 (n=509) | 55% | **23,6%** | −31 poin |
| 0,40 – 0,50 (n=3.467) | 45% | **13,6%** | −31 poin |
| 0,30 – 0,40 (n=12.069) | 35% | **6,9%** | −28 poin |
| 0,20 – 0,30 (n=24.746) | 25% | **2,0%** | −23 poin |

Laju dasar sesungguhnya 4,7%; rentang sigmoid model 0,273–0,654.

**Sebabnya bukan cacat:** model dilatih dengan focal loss (γ 2,0, α 0,75) yang
sengaja membobot kelas langka supaya ia belajar **membedakan**, bukan supaya
angkanya benar. Model semacam ini **memeringkat baik dan menaksir buruk** — dua
tugas berbeda, dan yang kedua tidak pernah dilatih.

Konsekuensinya operasional dan tajam: memutuskan "tebang yang di atas 50%" akan
menebang pohon yang **tiga dari empat** di antaranya sebenarnya sehat. Itulah yang
dijaga larangan tersebut, dan sekarang besarnya tercatat.

### Agregasi blok: hipotesis kedua yang kami uji dan TOLAK

Dugaannya operasional: mandor tidak memeriksa satu pohon, ia mengirim regu ke satu
petak. Kalau unit prediksinya petak, merata-ratakan puluhan pohon semestinya
memangkas derau dan menghasilkan angka yang lebih kuat sekaligus lebih relevan.
Diuji di `layer2_real/run_v3_blocks.py`, leave-one-parcel-out, **1.558 unit
blok-sensus** per lari (39,4 petak × 39,5 sensus).

| metrik, unit = petak | model | acak | selisih |
|---|---|---|---|
| Spearman peringkat petak | **+0,179 ± 0,003** | +0,004 ± 0,020 | +0,175, **20/20**, POS |
| tangkapan 5 petak teratas | **18,5%** | 14,9% | +3,6 pp, 19/20 |

Garis acak dihitung dengan mengacak skor pohon **di dalam sensus** lalu
mengagregasi dengan cara yang sama persis. Ia duduk di **nol** (+0,004), jadi
metriknya **tidak** terlalu mudah — model memang mengalahkannya 20 dari 20.

**Tetapi liftnya 1,24×, di bawah lift per-pohon 1,61×.** Hipotesisnya ditolak.

**Bacaannya, dan ini memperkuat klaim inti paket:** sinyalnya berskala **pohon**,
bukan petak. Yang berisiko adalah sawit yang bersentuhan dengan sawit sakit, bukan
seluruh petaknya; merata-ratakan 8–17 pohon mencampur 1–2 yang benar-benar terancam
dengan belasan yang tidak. Agregasi **mengencerkan** sinyal, bukan meredam derau.
Graf kontak bekerja justru **karena ia lokal** — begitu unitnya melewati jangkauan
kontak akar, keunggulannya luruh. Keluaran operasional yang benar tetap **daftar
prioritas per-pohon**.

### Pusat wabah — keluaran tanpa model

Komponen terhubung di antara tajuk bergejala pada graf kontak. Tidak meramal apa
pun, tidak memeringkat apa pun, dan tidak membawa batas baru: ia pernyataan
geometris murni ("gejala-gejala ini bersambung, yang itu terpisah"). Dilaporkan per
pusat: jumlah tajuk bergejala, jumlah sawit sehat yang bersentuhan langsung, dan
titik tengahnya. Implementasi: `demo_core.outbreak_foci()`; tampil di layar Hasil.

### Ambang kelas Unhealthy: hipotesis yang kami uji dan TOLAK

Dugaan awalnya: detektor jarang memanggil `Unhealthy` karena ambang 0,75 dipilih untuk
memaksimalkan F1 **pusat tajuk**, yang didominasi kelas Healthy — jadi ambang terpisah untuk
kelas penyakit akan menolong. Diuji, dan **tidak**:

| | F1 kelas Unhealthy |
|---|---|
| ambang dipilih silang-lipatan | 0,370 ± 0,052 |
| ambang 0,75 apa adanya | **0,406 ± 0,058** |

Menyetel ambang justru memperburuk. Optimum tiap lipatan melompat — 0,85 · 0,55 · 0,85 —
karena dengan 17–31 positif per ortomosaik letaknya adalah derau, sehingga tidak berpindah
antar lipatan. **Ambang tetap 0,75.**

Ikut terkoreksi satu tafsir kami sendiri: "detektor hanya menemukan 0–1 gejala per ubin"
bukan kegagalan. Ubin 1024² memuat ~65 sawit; pada laju Unhealthy 1,3% ia **diharapkan**
memuat ~0,85 pohon sakit. Yang terlihat adalah laju dasar, bukan detektor yang lumpuh.

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
| ~~Deteksi YOLO 1 lipatan, 15 epoch~~ | **DITUTUP** | Kini YOLOv12n, **3 lipatan × 30 epoch** dalam satu lingkungan, dengan mean ± std. Yang tersisa dan tetap harus dinyatakan: mAP berlangit-langit label, dan **`52000_20000` menyeret seluruh std** — turun di semua metrik (mAP50 0,605 lawan 0,726/0,731) karena anotasinya tidak konsisten (CV kotak 0,327 lawan 0,126/0,170), **bukan** karena efek situs. Pada metrik pusat jaraknya menyusut 3× (0,933 lawan 0,969/0,977), persis karena pusat adalah bagian label yang dapat dipercaya |
| Blok spasial Eg9PP | **2 parcel** | ±0,008 itu derau optimisasi, bukan ketidakpastian data. Efek beda **2,6×** antar blok (44A +0,0084 vs 44B +0,0219) |
| ~~Kontrol `random` terlalu lemah~~ | **DITUTUP** | Dulu mata rantai terlemah: `random` global menghubungkan pohon berjarak median 13,2 sehingga bisa mengukur lokalitas, bukan peta. Kini diuji dengan tangga `random_local` (≤6 dan ≤3 jarak tanam) dan **78–85% efeknya bertahan** di keempat horizon. Rewire jaga-jarak murni tetap mustahil di r=1,5 — semua sisi asli panjangnya 1,0 — jadi kontrol berbatas-radius ini adalah yang terketat yang bisa dibangun |
| Horizon h≠3 untuk SI(D) | **4 seed** (n=8) | dekomposisi utama kini 20 seed di keempat horizon; hanya kepala SI(D) yang belum |
| Sensor Eg9PP | 498–621 negatif/horizon | nasib sesungguhnya tak diketahui; bias kecil ke atas pada spesifisitas |
| Kepala SI(D) | 112 → **3 parameter** | kompartemen laten E **tak teramati**; tak bisa dibandingkan dengan varian SEIR mana pun |
| Varian v3 tanpa genotipe | **36% kekerabatan** | efek graf v3 mengandung 36% kontaminasi famili (null dalam-famili+petak, 200 permutasi). Angka mentahnya **tidak boleh** dikutip seolah seluruhnya penularan; 64% yang tersisa adalah spasial |
| ~~Masukan v3 dari citra belum diuji~~ | **DIUKUR: −41%** | Disimulasikan pada laju detektor ds_B (recall 0,446 · fpr 0,0094): AP dalam-sensus 0,0916 → **0,0800**, lift 1,45× → **1,27×**, **59% sinyal bertahan**. Yang tersisa sebagai batas: laju itu diukur di ds_B, kebun yang **berbeda** dari Eg9PP, jadi ini simulasi ongkos — bukan pengukuran lapangan di kebun tujuan |
| Positif Unhealthy terlalu sedikit untuk menyetel ambang | **66 pohon unik** | Ambang kelas Unhealthy dipilih silang-lipatan justru **lebih buruk** (F1 0,370) daripada memakai 0,75 apa adanya (0,406); optimum per lipatan melompat 0,85/0,55/0,85 karena letaknya derau. Ambang tetap 0,75 |
| ~~Uji jembatan pakai kotak ground-truth~~ | **DITUTUP** | Kini diukur dari **prediksi detektor**, bukan kotak GT: derajat pohon-dalam **5,54 ± 0,12** lawan 5,62 ± 0,05 pada kotak GT (−1,4%). Terhadap 5,74 Eg9PP selisihnya **3,5%** — dan 5,74 berada **di luar** pita ±0,12, jadi kalimatnya "berselisih 3,5%", **bukan** "keduanya sama" |

## Satu klaim yang kami tarik sendiri

Laporan sebelumnya menyebut satu ortomosaik "kolaps" ke PR-AUC 0,030 sebagai bukti variasi antar-situs besar. Setelah deduplikasi ke pohon unik, ortomosaik itu mencapai **0,126** dan ketiga lipatan berada di 0,13–0,26. **Kolapsnya artefak pseudo-replikasi 29,8×, bukan efek situs.** Klaim ditarik.

---

## Peta berkas

| Ingin lihat | Buka |
|---|---|
| ringkasan semua angka | `00_RINGKASAN.csv` |
| tangga lokalitas + ablasi fitur | `layer2_real/results_v2.csv` · kode: `run_v2.py`, `dataset_v2.py`, `sweep_v2.py` |
| varian foto-tunggal (v3) | `layer2_real/results_v3.csv` · kode: `dataset_v3.py`, `run_v3.py` |
| null dalam-famili untuk v3 | `layer2_real/results_v3_perm_progeny.csv` · `results_v3_perm_progeny_parcel.csv` · kode: `run_v3_perm.py` |
| penyimpangan kontrak v3 | `layer2_real/INTERFACE.md` bagian akhir |
| angka mentah Eg9PP | `layer2_real/results_real.csv` · log: `run_real.log` |
| angka mentah Lapisan 1 | `layer1_build/RESULTS_LAYER1.md` |
| fakta data yang boleh diklaim | `data_clean/DATASET_CARD.md` |
| bukti pseudo-replikasi 29,8× | `layer1_data_audit/AUDIT_REPORT.md` |
| penjaga kebocoran (~90 asersi) | `layer2_real/test_dataset.py` |
| status naskah & sisa pekerjaan | `README.md` |
