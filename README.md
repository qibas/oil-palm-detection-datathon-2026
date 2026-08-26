# SawitGuard-GNN

Sistem peringatan dini untuk Busuk Pangkal Batang (BSR, disebabkan oleh *Ganoderma boninense*)
pada perkebunan sawit. Dikembangkan untuk Datathon 2026, Ristek Fasilkom UI.

---

## Mulai di sini

Dokumentasi lengkap pipeline dan seluruh hasil eksperimen ada di [`00_HASIL.md`](00_HASIL.md).
Angka mentahnya tersimpan di [`00_RINGKASAN.csv`](00_RINGKASAN.csv) (98 baris).

Sebelum mengutip angka apa pun ke naskah atau presentasi, periksa
[`00_ANGKA_FINAL.md`](00_ANGKA_FINAL.md) terlebih dahulu. Paket ini memuat angka dari sembilan
konfigurasi eksperimen berbeda, dan sebagian pasangan angka tidak sebanding satu sama lain.
Berkas tersebut menandai mana yang sah dikutip sebagai angka utama.

### Temuan utama

**Peta kontak akar yang benar berpengaruh nyata terhadap prediksi.** Pada data lapangan
Ganoderma 25 tahun, struktur graf kontak menyumbang +0,0151 AUC-PR (39 dari 40 pasangan
pengujian) dan bertahan terhadap uji permutasi terkontrol genotipe (kelebihan 1,25 sampai
1,29 kali, dicapai 0 dari 500 permutasi acak).

**Riwayat waktu ternyata tidak dibutuhkan.** Dinilai dalam-sensus, satu-satunya metrik yang adil
untuk memeringkat pohon di dalam satu bidikan foto, varian model yang hanya memakai kondisi
tetangga (persis yang bisa diberikan satu foto drone) menyamai model penuh (+0,0042 ± 0,0035,
36 dari 40 pasangan). Sebanyak 77% dari kemampuan itu berasal khusus dari peta kontak yang
benar (+0,0296, 40 dari 40 pasangan). Kontaminasi kekerabatan yang menyertainya terukur 36%,
dan 59% sinyal tetap bertahan saat masukan datang dari detektor otomatis, bukan status lapangan
yang terverifikasi.

**Lapisan epidemiologi terlatih justru merugikan hasil.** Kepala SI(D) negatif di keempat
horizon prediksi dan memburuk seiring horizon bertambah jauh. Dugaan penyebab paling sederhana,
inisialisasi laju yang tidak netral, sudah diuji dan ditolak untuk horizon dua ke atas.

**Sebagian besar "efek tetangga" adalah perancu waktu kalender.** Risiko relatif mentah 4,47 kali
turun menjadi 1,65 kali setelah stratifikasi per sensus. Efek yang tersisa tetap signifikan
secara statistik, tetapi jauh lebih kecil daripada klaim tanpa koreksi.

---

## Cakupan

Paket ini hanya memakai data nyata: citra UAV dari Roboflow (Lapisan 1) dan data lapangan Eg9PP
(Lapisan 2). Repositori kerja juga memuat simulator SEIR sintetis beserta rangkaian ablasinya;
komponen itu sengaja dikeluarkan dari cakupan paket ini agar setiap angka yang dilaporkan
berasal dari kebun sungguhan.

Konsekuensinya, `paper/section3.tex` dan `paper/METHODOLOGY_PLAN.md` masih menulis tentang
simulator tersebut. Keduanya naskah yang belum diperbarui; lihat bagian status naskah di bawah.

## Status naskah

`section3.tex` belum ditulis ulang. Empat masalah berikut terverifikasi masih ada di dalamnya:

| # | Masalah |
|---|---|
| 1 | Masih membahas simulator SEIR, yang sudah di luar cakupan paket ini |
| 2 | Menyatakan Tahap 5 menjalankan dinamika di atas geometri kebun yang nyata, padahal kode simulator tidak pernah membaca awan titik Lapisan 1 |
| 3 | Bagian 3.6 sampai 3.7 menjanjikan kalibrasi, skor Brier, kurva keandalan, tingkat risiko, dan presisi@k. Tidak ada kode yang menghasilkan angka-angka itu |
| 4 | Ketimpangan kelas ditulis 69:1 (tingkat anotasi), padahal pada satuan pohon unik angkanya sekitar 76:1 (5.011 banding 66) |

`section3.docx` lebih basi lagi. `METHODOLOGY_PLAN.md` berisi rencana penulisan ulangnya. Belum
ada hasil Eg9PP yang masuk ke naskah mana pun.

---

## Struktur folder

| Folder | Isi |
|---|---|
| `data_clean/` | Pintu masuk semua eksperimen: CSV beku, dua skrip pembangunnya, `DATASET_CARD.md`, lisensi Eg9PP |
| `layer1_data_audit/` | Audit data dan 8 citra bukti bahwa tiga dataset Roboflow adalah satu dataset yang di-fork tiga kali, dengan pseudo-replikasi 29,8 kali |
| `layer1_build/` | Pipeline UAV: deteksi tajuk, kesehatan, segmentasi. Termasuk `ds_B`, 2.303 ubin citra sumber |
| `layer2_real/` | Data panel Eg9PP: dataset, model, driver eksperimen, null permutasi, dan sekitar 90 asersi penjaga kebocoran |
| `paper/` | Naskah dan rencana penulisan ulang |
| `figures/` | `fig_pipeline.png` dan skripnya |
| `models.py` | `MLPBaseline`, dipakai bersama oleh `layer2_real/models_real.py` |
| `env_context.py` | Konteks lingkungan (angin, hujan, tanah). Bukan fitur model, lihat [`ENV_CONTEXT.md`](ENV_CONTEXT.md) |
| `demo_api.py`, `web/` | Aplikasi demo (lihat bagian "Demo web" di bawah) |

### Urutan baca

1. [`00_HASIL.md`](00_HASIL.md), pipeline lengkap dan seluruh hasil
2. `data_clean/DATASET_CARD.md`, fakta data yang boleh diklaim, termasuk batas yang dipaksakan per dataset
3. `layer1_data_audit/AUDIT_REPORT.md`, audit data dan pseudo-replikasi 29,8 kali
4. `layer1_build/RESULTS_LAYER1.md`, hasil Lapisan 1 termasuk klaim yang ditarik kembali
5. `paper/METHODOLOGY_PLAN.md`, rencana merangkainya menjadi Bab 3

Prinsip kerja yang mengikat seluruh isi repositori ini adalah leakage-first: kebocoran data, hasil
nol, dan keterbatasan dilaporkan sebelum hasil apa pun disajikan, dan tidak ada perbaikan yang
diklaim selama ia masih berada di dalam pita derau statistik. Beberapa hasil di sini negatif, dan
satu klaim sebelumnya ditarik kembali oleh tim sendiri. Itu disengaja, bukan kelalaian.

---

## Reproduksi eksperimen

Tata letak paket ini mencerminkan repositori kerja, sehingga seluruh skrip berjalan di tempat.

```bash
pip install -r requirements.txt
```

Untuk melatih ulang Lapisan 1 dibutuhkan GPU. Pasang varian CUDA dari torch terlebih dahulu
(harus cu126, bukan cu130, karena arsitektur Pascal dibuang mulai CUDA 13):

```bash
pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.13.0+cu126" "torchvision==0.28.0+cu126"
```

Demo tidak membutuhkan GPU.

```bash
# 1. Bekukan dataset bersih (opsional, CSV yang sudah jadi ikut tersimpan di repo)
cd data_clean
python build_layer1.py         # -> 5.077 pohon unik, 66 positif
python build_layer2_real.py    # -> 1.200 node, 45 sensus, 3.354 sisi

# 2. Lapisan 1 (kesehatan tajuk di CPU sekitar 20 detik; deteksi butuh GPU, sekitar 45 menit per lipatan)
cd ../layer1_build
python exp_health.py                  # kesehatan tajuk, LightGBM, leave-one-ortho-out
FOLDS=all python train_folds_gpu.py   # YOLOv12n, 3 lipatan x 30 epoch
python centre_eval_folds.py           # metrik utama Tahap 1 dan angka uji jembatan
python unhealthy_threshold.py         # ambang kelas Unhealthy (hasilnya tetap 0,75)
python detect_centres.py <citra>      # citra apa pun -> pusat tajuk dan graf kontak

# 3. Lapisan 2, data panel Eg9PP (seluruhnya di CPU, total sekitar 24 menit)
cd ../layer2_real
python test_dataset.py         # sekitar 90 asersi dan 4 penjaga kebocoran, ~10 detik
python run_real.py             # dekomposisi utama -> results_real.csv, ~22 menit
python perm_null.py            # null permutasi terkontrol genotipe, ~45 detik

# Varian foto-tunggal (v3): membuang riwayat waktu dan genotipe, dinilai dalam-sensus
python run_v3.py 20                                  # -> results_v3.csv, ~12 menit
python run_v3_perm.py 200 2                          # null dalam-famili, ~28 menit
STRATA=progeny_parcel python run_v3_perm.py 200 2    # strata terketat, ~28 menit
RECALL=0.446 FPR=0.0094 python run_v3_noisy.py 10 20  # ongkos substitusi detektor, ~6 menit
python train_final_v3.py       # checkpoint 1-kolom untuk demo -> stgnn_v3_photo.pt
```

Varian v3 melanggar dua ketentuan kontrak Lapisan 2 dengan sengaja (window 1 sensus, genotipe
dibuang). Alasan dan ongkos terukurnya dijelaskan di bagian akhir `layer2_real/INTERFACE.md`.

---

## Demo web

```bash
python demo_api.py
```

Buka `http://localhost:8000`. React dan Babel di-vendor secara lokal di `web/vendor/`, sehingga
demo berjalan tanpa internet dan tanpa `npm install`. Satu-satunya pemanggilan jaringan asli
adalah fitur konteks lingkungan (angin, hujan, tanah), yang punya jaring pengaman offline
sendiri (lihat [`ENV_CONTEXT.md`](ENV_CONTEXT.md)).

Seluruh perhitungan berada di `demo_core.py`. `demo_api.py` hanya lapisan HTTP tipis di atasnya
dan tidak boleh menghitung apa pun sendiri, sehingga angka yang tampil di layar tidak pernah
menyimpang dari yang dihasilkan `python demo_core.py` langsung dari baris perintah.

```bash
python demo_core.py           # cetak semua angka layar dan render gambar, tanpa server
node web/check_jsx.js         # wajib dijalankan setelah menyunting web/app.jsx
```

`check_jsx.js` memeriksa dua hal yang lolos dari pencocokan string biasa: sintaks JSX (berkas
yang rusak tetap dikirim server dengan status 200) dan escape unicode yang bocor menjadi teks
literal di layar.

### Fitur

- **Deteksi dan peta kontak.** Unggah satu foto drone, sistem menemukan setiap tajuk sawit dan
  membangun graf kontak akarnya. Arahkan kursor (atau ketuk pada perangkat sentuh) ke satu pohon
  di layar graf untuk menyorot tetangga langsungnya.
- **Peringkat risiko.** Sawit sehat diperingkat berdasarkan kondisi tetangganya, memakai
  checkpoint v3-foto. Lihat bagian metodologi di bawah untuk penjelasan lengkapnya.
- **Survei banyak foto sekaligus.** Jatuhkan atau pilih lebih dari satu berkas di layar unggah
  untuk memproses satu blok kebun sekaligus. Foto diproses berurutan, bukan paralel, karena
  detektor berjalan di CPU dan pemrosesan paralel hanya akan berebut inti prosesor yang sama.
  Hasilnya digabung menjadi satu daftar prioritas lintas foto, diurutkan berdasarkan jumlah
  tetangga bergejala, bukan skor model. Skor v3-foto dinormalisasi memakai derajat rata-rata
  graf pada foto itu sendiri dan belum pernah diukur apakah skalanya sama antar foto yang
  berbeda, sementara jumlah tetangga bergejala adalah bilangan bulat yang berarti sama di foto
  mana pun.
- **Ekspor CSV.** Lihat bagian "Format data yang diekspor" di bawah.
- **Cetak laporan lapangan.** Tombol pada layar hasil memakai `window.print()` dengan lembar
  gaya cetak khusus (`@media print` di `web/styles.css`), tanpa dependensi PDF tambahan. Kontrol
  interaktif disembunyikan saat mencetak; peta risiko dan daftar prioritas tetap tampil.
- **Konteks lingkungan.** Menampilkan angin, hujan, dan tekstur tanah dari data publik asli
  (Open-Meteo, ISRIC SoilGrids) untuk satu koordinat kebun. Ini murni informasi pendukung
  keputusan, bukan masukan model; penjelasan lengkap dan batasannya ada di
  [`ENV_CONTEXT.md`](ENV_CONTEXT.md). Koordinat yang diisi tersimpan di penyimpanan lokal
  peramban dan otomatis muncul kembali sebagai kartu ringkas di layar hasil.
- **Animasi 25 tahun.** Pada layar Bukti dan Validasi, memutar status lapangan Eg9PP sensus demi
  sensus untuk memperlihatkan penyebaran gejala antar tetangga. Ini pemutaran ulang catatan
  lapangan apa adanya, bukan hasil menjalankan model berkali-kali.

### Sumber data lingkungan

| Sumber | Data yang diambil |
|---|---|
| [Open-Meteo](https://open-meteo.com) | Kecepatan dan arah angin saat ini, curah hujan 30 hari terakhir |
| [ISRIC SoilGrids v2.0](https://rest.isric.org/soilgrids/v2.0/docs) | Persen liat, pasir, dan bulk density tanah pada kedalaman 0 sampai 5 cm |

Keduanya sumber publik gratis, tanpa API key. Drainase dibaca dari tekstur tanah memakai segitiga
tekstur USDA (di atas 40% liat digolongkan buruk, 20 sampai 40% sedang, di bawah 20% baik), bukan
pengukuran drainase langsung; topografi dan muka air tanah tidak ikut terukur.

Literatur patologi sawit mencatat tanah tergenang atau berdrainase buruk, serta curah hujan
tinggi, sebagai kondisi yang mendukung perkembangan Ganoderma boninense: Rees et al. (2009),
Naher et al. (2013), dan Susanto et al. (2005). Ini bukan hubungan sebab akibat yang dibuktikan
di paket ini, dan detail bibliografis ketiga rujukan tersebut belum diverifikasi silang.

Fitur ini tidak pernah divalidasi terhadap kejadian BSR nyata: Eg9PP, satu-satunya dataset dengan
status Ganoderma yang terverifikasi lapangan, tidak memiliki koordinat lintang dan bujur untuk
diperiksa silang terhadap drainase atau curah hujan pada satu titik. Panel ini murni informasi
pendukung keputusan, bukan skor risiko tambahan.

### Format data yang diekspor

**`prioritas.csv`** (tombol "Ekspor daftar prioritas" pada satu foto), sepuluh baris teratas:

| Kolom | Arti |
|---|---|
| `peringkat` | Urutan pada foto ini, 1 adalah prioritas tertinggi |
| `tingkat` | Tingkat ordinal yang benar-benar dibedakan model, 1 adalah paling aman |
| `dari_n_tingkat` | Jumlah total tingkat yang dibedakan model pada foto ini |
| `persentil_teratas` | Posisi dalam persen, dihitung dari skor, bukan dari peringkat |
| `tetangga` | Jumlah tetangga dalam radius kontak akar |
| `tetangga_sakit` | Jumlah tetangga yang terdeteksi bergejala |
| `ada_sakit` | "ya" atau "tidak", apakah `tetangga_sakit` lebih dari nol |
| `skor_mentah` | Logit mentah model. Hanya berarti sebagai urutan pada foto ini, tidak sebanding antar foto atau antar checkpoint |
| `pita` | Kuintil warna pada peta risiko, 1 sampai 5 |

**`prioritas_survei.csv`** (tombol "Ekspor CSV" pada survei banyak foto), dua puluh baris
teratas gabungan:

| Kolom | Arti |
|---|---|
| `peringkat` | Urutan pada daftar gabungan, diurutkan dari `tetangga_sakit` |
| `foto` | Nama berkas asal |
| `peringkat_di_foto` | Peringkat pohon ini di dalam foto asalnya sendiri |
| `tetangga` | Jumlah tetangga dalam radius kontak akar |
| `tetangga_sakit` | Jumlah tetangga yang terdeteksi bergejala |

Daftar gabungan sengaja tidak memuat `skor_mentah`: skor dari foto yang berbeda tidak pernah
diukur sebanding satu sama lain, sehingga menampilkannya berdampingan berisiko dibaca sebagai
perbandingan yang sah padahal bukan.

### Metodologi dan batasan model

Ringkasan berikut sebelumnya tampil sebagai expander di layar hasil. Dipindahkan ke sini agar
layar produk tetap ringkas untuk pengguna operasional, tanpa menghilangkan detail yang
dibutuhkan siapa pun yang meninjau metodologinya.

**Checkpoint yang dipakai.** Demo memakai checkpoint v3-foto, dilatih ulang hanya dengan kolom
`is_sympt`, satu-satunya informasi yang bisa diisi dari satu foto. Pada tugas memeringkat pohon
di dalam satu bidikan, checkpoint ini menyamai model 24-kolom (AP dalam-sensus 0,1015 berbanding
0,0973). Checkpoint penuh `stgnn_final.pt` tidak dipakai di jalur foto karena ia meminta 18 kolom
yang mustahil diisi dari satu foto.

**Label citra.** Kelas dari detektor adalah kesehatan tajuk secara umum, bukan status Ganoderma
yang terverifikasi lapangan. Tidak ada diagnosis penyakit yang terjadi di layar mana pun pada
demo ini.

**Asal model.** Model dilatih pada kebun percobaan pemuliaan Eg9PP (2 parcel), bukan kebun
produksi. Efeknya sendiri berbeda 2,6 kali antar kedua parcel tersebut.

**Kontaminasi kekerabatan.** Efek graf mengandung 36% kontaminasi kekerabatan karena keluarga
sekandung ditanam berdampingan (diukur lewat null dalam-famili dan petak, 200 permutasi, 0 dari
200 mencapai nilai teramati).

**Masukan gejala.** Model dilatih pada status yang terverifikasi lapangan; di demo, kolom yang
sama diisi kelas dari detektor otomatis. Ongkos substitusi ini terukur pada pengujian
leave-one-parcel-out: 59% sinyal tetap bertahan, dan lift turun dari 1,45 kali menjadi 1,27 kali.

**Kalibrasi.** Model memeringkat dengan baik, tetapi menaksir probabilitas dengan buruk, dan itu
terukur. Pada pengujian leave-one-parcel-out, sawit dengan sigmoid(skor) antara 0,50 dan 0,60
sesungguhnya sakit pada 23,6% kasus, bukan 55%, meleset 31 poin persentase. Penyebabnya adalah
focal loss (alfa 0,75) yang sengaja membobot kelas langka agar model belajar membedakan, bukan
agar keluarannya berupa probabilitas yang benar. Karena itu, keluaran model adalah peringkat, dan
nilai sigmoidnya tidak boleh disajikan sebagai persentase.

**Checkpoint pada demo ini.** Dilatih pada seluruh 1.200 sawit tanpa kumpulan uji terpisah. Ini
adalah artefak inferensi, bukan artefak evaluasi: tidak ada angka performa yang boleh dikutip
dari checkpoint ini. Angka performa yang sah datang dari `results_real.csv`, hasil pengujian
leave-one-parcel-out yang terdokumentasi di [`00_HASIL.md`](00_HASIL.md).

### Berkas wajib ada

Berkas berikut dibutuhkan agar demo berjalan; pastikan ikut ter-commit:

| Berkas | Isi |
|---|---|
| `layer2_real/stgnn_v3_photo.pt` | Checkpoint Lapisan 2 varian foto (54 KB) |
| `layer2_real/risk_ranked.csv` | Peringkat Eg9PP untuk layar Bukti dan Validasi |
| `layer1_build/yolo12_runs/yolo12n_base_fold*/weights/best.pt` | Bobot detektor, lari 3 lipatan, angka naskah |
| `layer1_build/yolo12_runs/yolo12n_base_1fold_fold0_s42/weights/best.pt` | Bobot yang dipakai demo lebih dahulu; tanpa berkas ini demo jatuh ke bobot 3 lipatan dan tetap berjalan |
| `web/vendor/*.js` | React, ReactDOM, Babel (3,2 MB) |
| `data_clean/*.csv` | Dataset beku |
| `env_context_cache.json` | Snapshot angin, hujan, dan tanah asli untuk jaring pengaman offline |

`layer1_build/yolo12/` dan `layer1_build/anom_data/` sengaja tidak dilacak oleh git. Keduanya
direktori turunan yang dibangun ulang oleh `y12.build()` dan `anom.build()` dalam hitungan detik,
dan berkas lipatannya memuat path absolut milik mesin yang membangunnya.

Terverifikasi: `python layer2_real/test_dataset.py` dijalankan dari dalam paket ini dan lulus
seluruh pemeriksaan (exit 0).

---

## Enam larangan yang berlaku di seluruh paket

Ada dua daftar enam larangan di paket ini. Daftar di bawah berlaku untuk seluruh paket, dan
`layer2_real/INTERFACE.md` memiliki daftarnya sendiri khusus Lapisan 2 dengan isi dan penomoran
yang berbeda. Saat mengutip, sebutkan sumbernya secara eksplisit: "larangan #5 `INTERFACE.md`"
(genotipe wajib menjadi kovariat) bukan hal yang sama dengan "#5" di bawah ini (pohon tersensor).

1. Jangan memakai split acak pada Lapisan 1. Hanya ada 3 ortomosaik, dan split acak bocor 100%.
2. Jangan mengutip 151.060 sebagai ukuran sampel. Unit yang benar adalah 5.077 pohon unik.
3. Jangan menyebut label Roboflow sebagai BSR. Itu kesehatan tajuk generik tanpa verifikasi lapangan.
4. Jangan melaporkan akurasi. Pos-rate berkisar 1,3 sampai 6,0%; klasifikator yang selalu
   menjawab "sehat" sudah lebih dari 98% akurat tanpa guna apa pun.
5. Jangan menganggap pohon tersensor sebagai sehat. Ia keluar dari risk set.
6. Jangan membandingkan kepala SI(D) dengan varian SEIR mana pun. Kompartemen laten E tidak
   teramati di data lapangan, dan kepala ini turun dari 112 parameter menjadi hanya 3.

---

## Data pihak ketiga

**Eg9PP.** Tisné S., Pomiès V., Riou V., Syahputra I., Cochard B., Denis M. (2017).
*Identification of Ganoderma disease resistance loci using natural field infection of an oil
palm multi-parent population.* G3, 7(6):1683-1692. doi:10.1534/g3.117.041764. Hak cipta PalmElit
dan CIRAD, lisensi CC BY-SA 4.0. Detail lengkap di `data_clean/Eg9PP_LICENSE.md`.

**Roboflow ds_B.** `health-detection/oil-palm-health-detection`. Lihat berkas README Roboflow di
dalam `layer1_build/ds_B/`.

**Anomali tajuk sawit Peru (PalmAnom/PalmSan).** Mendeley Data, doi:10.17632/nh7d23dgnw.1,
lisensi CC BY 4.0. Dipakai sebagai jalur bukti ketiga yang berdiri sendiri (lihat `00_HASIL.md`
bagian 2.6) dan tidak pernah digabung dengan ds_B.

---

## Catatan teknis

Citra di `layer1_build/ds_B/` adalah hardlink ke repositori kerja: nol byte tambahan di disk,
tetapi berperilaku seperti berkas biasa untuk semua perangkat termasuk kompresi zip. Kode dan
dokumen adalah salinan sungguhan, sehingga menyunting paket ini tidak mengubah repositori aslinya.

Ukuran total sekitar 21 MB data nyata, ditambah 377 MB yang ter-hardlink.
