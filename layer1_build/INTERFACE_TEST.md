# Uji antarmuka "lebar sepur" dengan pusat tajuk hasil PREDIKSI detektor

Skrip: `interface_test.py` · Keluaran mentah: `yolo12_results/interface_test.json`
Tanggal: 2026-08-08 · Detektor: YOLOv12n, 30 epoch, imgsz 640, seed 42, leave-one-ortho-out

## 1. Apa yang dijawab, dan apa yang berubah

Makalah melaporkan derajat tetangga rata-rata **5,74** (posisi tanam Eg9PP) lawan **5,62**
(centroid tajuk ds_B), berselisih **2%**, sebagai bukti bahwa graf keluaran Lapisan 1
sebangun dengan graf yang dikonsumsi Lapisan 2. Angka 5,62 dihitung dari **kotak
kebenaran-dasar**, sehingga `00_HASIL.md` dan `DATASET_CARD.md` menyebutnya **batas atas**:
ia menjawab "seandainya detektornya sempurna", bukan "apa yang benar-benar keluar dari
pipeline".

Uji ini mengulang pengukuran itu dari **pusat tajuk hasil prediksi YOLOv12** pada
ortomosaik yang ditahan. Yang berubah bukan hanya sumber titiknya. Dengan memaksakan
**satu definisi yang sama** kepada ketiga himpunan titik, dua hal ikut terbongkar, dan
keduanya dilaporkan di bawah:

1. Angka **5,74 dan 5,62 tidak dihitung dengan aturan yang sama**. Selisih "2%" itu
   sebagian adalah artefak ketidakseragaman metode, bukan seluruhnya geometri.
2. Arah bias prediksi **berlawanan dengan dugaan**. Pada ambang bawaan pipeline detektor
   MENGHASILKAN TERLALU BANYAK titik, bukan terlalu sedikit.

## 2. Metodologi — satu aturan untuk ketiga himpunan

**Radius tetangga.** `r = frac x jarak tanam`, dengan `frac` = 1,25 / 1,375 / 1,5 —
dataran kurva derajat yang dipakai perbandingan lama. **Jarak tanam ditaksir dari
himpunan titik itu sendiri** sebagai median jarak tetangga-terdekat, supaya satuan
Eg9PP (jarak tanam) dan ds_B (piksel 8,5–8,9 cm) sebanding.

**Pohon bagian dalam.** Sebuah pohon disebut bagian dalam bila **jaraknya ke tepi convex
hull himpunannya >= r**, sehingga cakram radius r di sekelilingnya seluruhnya berada di
dalam petak. Tetangga tetap dihitung terhadap **seluruh** titik; yang dibatasi hanya
himpunan yang dirata-ratakan. Tanpa pembuangan tepi, yang dibandingkan adalah rasio
keliling-terhadap-luas kedua petak, bukan jarak tanamnya.

**Blok.** Hull dihitung **per parcel** untuk Eg9PP dan **per ortomosaik** untuk ds_B.
Untuk Eg9PP ini menentukan: parcel 44A (y = 3,0–14,5) dan 44B (y = 22,0–33,5) terpisah
7,5 jarak tanam dan `layer2_edges.csv` memang mencatat 0 sisi lintas-parcel. Satu hull
atas gabungan keduanya membentang menyeberangi celah kosong itu, sehingga pohon di
tepi-**dalam** tiap parcel — yang justru kehilangan separuh tetangganya — salah
dinyatakan "bagian dalam".

**Mean ± std** diambil antar-blok: 2 parcel (Eg9PP), 3 ortomosaik (ds_B). Aturan putusnya
sama dengan `run_experiment.py::paired`: selisih di dalam satu simpangan baku dinyatakan
**tidak konklusif**.

### Validasi metodologi terhadap angka yang sudah terbit

Sebelum dipakai, aturan di atas diuji apakah ia mereproduksi angka yang sudah ada:

| Angka terbit | Sumber | Hasil hitung ulang | Cocok? |
|---|---|---|---|
| Derajat penuh Eg9PP **5,59** | `make_pipeline_drawio.py` | **5,590** (r = 1,5, seluruh pohon) | ya |
| Derajat ds_B **5,37–5,52** pada radius 13 m | `section3.tex` Tabel 3 | **5,370 / 5,429 / 5,521** (r = 1,444, seluruh pohon) | ya |
| Derajat ds_B **5,62** | `DATASET_CARD.md` | **5,628** (r = 1,5, bagian dalam, margin = r) | ya |
| Derajat Eg9PP **5,74** | `DATASET_CARD.md` | **6,000** dengan aturan yang sama | **TIDAK** |

Tiga dari empat tereproduksi. Angka **5,74 tidak dapat direproduksi** dengan aturan yang
menghasilkan 5,62.

### Dari mana 5,74 sebenarnya berasal

Derajat Eg9PP pada r = 1,5, menurut cara tepi dibuang:

| Aturan bagian dalam | margin 0 | 0,25 | 0,5 | 1,0 | 1,5 (= r) |
|---|---|---|---|---|---|
| hull **per parcel** (benar) | 5,587 | 5,909 | 5,909 | **6,000** | **6,000** |
| hull **gabungan** (cara lama) | 5,590 | **5,745** | 5,748 | 5,784 | 5,779 |
| ds_B kotak kebenaran-dasar | 5,464 | 5,564 | 5,582 | 5,625 | **5,628** |

**5,74 = 5,745** muncul hanya dari hull **gabungan** dua parcel dengan margin longgar.
Jadi angka "2%" yang terbit memasangkan sisi Eg9PP yang dihitung dengan aturan longgar
melawan sisi ds_B yang dihitung dengan aturan ketat. Dengan aturan yang seragam, Eg9PP
adalah kisi segitiga sempurna: **derajat bagian dalamnya tepat 6,000** pada ketiga radius,
dan selisih terhadap kotak kebenaran-dasar bukan 2% melainkan **6,2%**.

## 3. Hasil utama — derajat bagian dalam, mean ± std antar-blok

conf = 0,25 (ambang bawaan pipeline, sama dengan `y12.centre_eval`)

| Himpunan titik | blok | r = 1,25 | r = 1,375 | r = 1,5 |
|---|---|---|---|---|
| **Eg9PP** (posisi tanam) | 2 parcel | 6,000 ± 0,000 | 6,000 ± 0,000 | **6,000 ± 0,000** |
| **ds_B kotak kebenaran-dasar** | 3 ortho | 5,321 ± 0,311 | 5,564 ± 0,112 | **5,628 ± 0,058** |
| **ds_B PREDIKSI YOLOv12** | 3 ortho | 4,105 ± 1,589 | 5,488 ± 0,434 | **6,001 ± 0,140** |

Per blok pada r = 1,5:

| Blok | Eg9PP | ds_B GT | ds_B PRED |
|---|---|---|---|
| 44A / 44000_16000 | 6,000 (504/720) | 5,694 (1.217/1.379) | 5,959 (1.406/1.595) |
| 44B / 44000_4000 | 6,000 (324/480) | 5,601 (1.595/1.849) | 6,157 (2.028/2.262) |
| — / 52000_20000 | — | 5,588 (1.611/1.849) | 5,887 (1.981/2.259) |

Angka dalam kurung = pohon bagian dalam / seluruh pohon blok itu.

## 4. Jumlah pohon dan recall — WAJIB dibaca bersama derajatnya

| Fold | Ortomosaik | Kebenaran-dasar | Prediksi | Recall | Presisi |
|---|---|---|---|---|---|
| fold0 | 44000_16000 | 1.379 | **1.595** (+15,7%) | 0,999 | 0,864 |
| fold1 | 44000_4000 | 1.849 | **2.262** (+22,3%) | 0,999 | 0,817 |
| fold2 | 52000_20000 | 1.849 | **2.259** (+22,2%) | 0,961 | 0,786 |
| total | | 5.077 | **6.116** | 0,986 | 0,822 |

Recall/presisi diukur dengan pencocokan satu-ke-satu pada 0,5 × jarak tanam
kebenaran-dasar (`y12.centre_match`), ambang yang sama dengan `y12.centre_eval`.

**Dugaan awal kami keliru, dan koreksinya penting.** Kami menduga recall yang tidak
sempurna akan membuat himpunan prediksi lebih **jarang** sehingga derajatnya bias **ke
bawah**. Pada conf = 0,25 yang terjadi kebalikannya: recall hampir sempurna tetapi presisi
hanya 0,79–0,86, sehingga himpunan prediksi **15–22% lebih padat**. Dua akibatnya:

1. Derajat bias **ke atas** — titik berlebih menambah tetangga.
2. Penaksir jarak tanam bias **ke bawah** — median jarak tetangga-terdekat himpunan
   prediksi turun ke **0,85–0,99 ×** nilai sebenarnya (86,3 / 92,4 / 104,4 px lawan
   101,1 / 103,0 / 105,8 px). Karena `r = frac × jarak tanam taksiran`, seluruh kurva
   derajat **bergeser ke kiri**.

Pergeseran itulah yang menjelaskan kolom r = 1,25 yang berantakan (4,105 ± 1,589, dengan
52000_20000 jatuh ke 2,416): pada ortomosaik itu 1,25 × 86,3 px = 108 px, yaitu hanya
1,07 × jarak tanam yang sebenarnya — masih di kaki kurva yang menanjak, bukan di dataran.

## 5. Kepekaan terhadap ambang keyakinan

Dilaporkan apa adanya. Ia **bukan** alat memilih conf yang paling cocok dengan Eg9PP:
memilih ambang berdasarkan kecocokan dengan jawaban yang dituju adalah menyetel pada
hasil.

| conf | n prediksi | recall | presisi | jarak tanam / GT | derajat r = 1,5 |
|---|---|---|---|---|---|
| 0,25 | 6.116 | 0,986 | 0,822 | 0,85–0,99 | 6,001 ± 0,140 |
| 0,40 | 5.511 | 0,969 | 0,892 | 0,96–1,00 | 5,802 ± 0,148 |
| 0,50 | 5.246 | 0,953 | 0,919 | 0,98–1,01 | **5,660 ± 0,139** |
| 0,60 | 4.991 | 0,932 | 0,943 | 0,99–1,03 | 5,539 ± 0,221 |
| 0,70 | 4.757 | 0,905 | 0,957 | 0,99–1,04 | 5,521 ± 0,167 |

Kedua arah bias terlihat di tabel ini. Pada conf rendah kelebihan deteksi mengangkat
derajat ke 6,00. Pada conf 0,70 recall fold2 jatuh ke 0,731 dan derajatnya tertekan ke
bawah — persis bias yang semula kami duga. Di antara keduanya, pada conf 0,40–0,50, jarak
tanam taksiran menyatu ke nilai sebenarnya (0,96–1,01 ×) dan derajatnya mendarat di
**5,66–5,80**, yang **mengurung angka kotak kebenaran-dasar 5,628**.

## 6. Putusan

Pada r = 1,5, dengan aturan pita derau `run_experiment.py::paired`:

| Perbandingan | Selisih | Pita derau | Putusan |
|---|---|---|---|
| PRED (conf 0,25) − Eg9PP | +0,001 (0,02%) | 0,140 | di dalam pita |
| PRED (conf 0,25) − ds_B GT | +0,373 (6,6%) | 0,140 | **di luar pita** |
| ds_B GT − Eg9PP | −0,372 (6,2%) | 0,058 | **di luar pita** |
| PRED (conf 0,50) − Eg9PP | −0,340 (5,7%) | 0,139 | **di luar pita** |
| PRED (conf 0,50) − ds_B GT | +0,032 (0,6%) | 0,139 | di dalam pita |

**Kecocokan 6,001 lawan 6,000 pada conf 0,25 adalah kebetulan, bukan bukti.** Ia lahir
dari dua bias yang saling meniadakan, dan pada baris berikutnya himpunan prediksi yang
sama ternyata berbeda secara signifikan dari kotak kebenaran-dasar yang seharusnya ia
tiru. Angka yang benar untuk dikutip adalah conf 0,50, yang menyetujui kotak
kebenaran-dasar (0,6%, di dalam pita) dan berselisih **5,7%** dari Eg9PP.

**Apakah kesepakatan 2% bertahan? Tidak sebagai angka.** Dengan metodologi yang seragam,
selisihnya **5,7–6,6%**, bukan 2%. Sebagian besar kenaikan itu **bukan** karena beralih ke
kotak prediksi: kotak kebenaran-dasar pun sudah berselisih 6,2% begitu sisi Eg9PP-nya
dihitung dengan aturan yang sama. "2%" yang terbit adalah artefak pemasangan dua aturan
tepi yang berbeda.

**Kesimpulan kualitatifnya bertahan.** Di seluruh sapuan conf 0,25–0,70 derajat bagian
dalam himpunan prediksi berada di **5,52–6,00**, yaitu di dalam ~8% dari 6. Keduanya tetap
**kisi segitiga berderajat 6**, dan pembenaran untuk membandingkan (bukan menggabungkan)
kedua dataset tidak berubah. Yang harus berubah adalah angka yang dikutip dan klaim
presisinya.

## 7. Status "batas atas"

Framing lama — "5,62 memakai kotak kebenaran-dasar, jadi ia batas atas" — **tidak lagi
tepat sebagai dikotomi**. Angka kebenaran-dasar bukan batas atas kesesuaian: himpunan
prediksi pada conf 0,25 justru mendarat **lebih dekat** ke Eg9PP daripada kebenaran-dasar,
karena kelebihan deteksinya menaikkan derajat melewati nilai sebenarnya. Yang benar:

- Kotak kebenaran-dasar memberi **geometri tanam yang sebenarnya**: 5,628 ± 0,058.
- Kotak prediksi memberi **geometri yang benar-benar dikonsumsi hilir**, dan geometri itu
  **bergantung pada ambang**: 5,52–6,00 di sepanjang conf 0,25–0,70.
- Tidak ada satu pun yang membatasi yang lain dari satu arah. Yang ada adalah **rentang
  ketergantungan-ambang**, dan rentang itu yang harus dikutip.

## 8. Batas yang tetap melekat

- **n = 3 ortomosaik dan 2 parcel.** Std di sini adalah keragaman antar-blok pada sampel
  yang sangat kecil; ia tidak menyempit dengan menambah seed.
- Kedua kebun tetap **tidak digabungkan**: kebun berbeda, zaman berbeda (petak Eg9PP
  dibongkar 2012; citra DJI pasca-2013), tanpa kunci join, tanpa georeferensi.
- Uji ini mengukur **geometri saja**. Ia tidak mengklaim kesetaraan laju epidemi
  antar-era, dan label ds_B tetap **kesehatan tajuk generik, BUKAN BSR**.
- Bobot fold0 sebelumnya terhapus dan **dilatih ulang** dengan konfigurasi identik. Hasil
  validasinya mereproduksi arsip di `yolo12_results/yolo12n_base.json` (mAP50 0,701 lawan
  0,7018; mAP50-95 0,483 lawan 0,4837), jadi bobot penggantinya sah dipakai.

## 9. Cara menjalankan ulang

```bash
cd layer1_build
python interface_test.py                # semua; melatih ulang fold0 bila bobotnya hilang
python interface_test.py --no-sweep     # lewati sapuan conf (jauh lebih cepat)
python interface_test.py --no-train     # laporkan hanya fold yang bobotnya ada
```

Skrip memanggil `y12.build(extra_mode="ignore")` untuk membangun ulang `yolo12/` bila
perlu, dan **sengaja tidak menulis apa pun ke `yolo12_results/*.json` milik ablasi** —
menulis ke sana dengan daftar fold yang berbeda akan menimpa hasil fold1/fold2. Keluaran
uji ini hanya `yolo12_results/interface_test.json`.

## 10. Berkas yang perlu disunting bila angka ini diadopsi

Angka 5,74 / 5,62 / "2%" muncul di:

- `data_clean/DATASET_CARD.md` (tabel derajat)
- `00_HASIL.md` (diagram ringkas, dan baris "batas atas" pada tabel batas)
- `paper/METHODOLOGY_PLAN.md` (Subbab 3.6, tabel angka kunci)
- `figures/make_pipeline_drawio.py` (kotak "Uji antarmuka" dan catatannya)
- `paper/make_theory_docx.py`, `paper/make_methodology_docx.py`
