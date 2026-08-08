# Kartu Dataset — SawitGuard-GNN

> Isi berkas ini adalah **karakteristik dataset bersih**, bukan prosedur prapemrosesan.
> Prosedurnya ada di `build_layer1.py` dan `build_layer2_real.py`; keduanya berhenti dengan
> `AssertionError` bila salah satu angka di bawah bergeser.
>
> Setiap dataset punya satu baris **"Batas yang dipaksakan"** — inilah yang menentukan klaim
> maksimum yang boleh ditulis di paper.

Semua eksperimen hilir membaca dari sini. Tidak ada skrip yang boleh membaca COCO mentah lagi.

---

## Dataset 1 — Inventaris tajuk UAV (Lapisan 1)

| | |
|---|---|
| Berkas | `layer1_crowns.csv` · `layer1_tiles_disjoint.csv` |
| Sumber | Roboflow `health-detection/oil-palm-health-detection` (ds_B) |
| Modalitas | UAV RGB nadir, ortomosaik, GSD **≈ 8,7 cm/px** (jarak tetangga terdekat 101–106 px pada jarak tanam acuan 9 m) |
| Label | **Healthy / Unhealthy — kesehatan tajuk generik, BUKAN BSR**, tanpa verifikasi lapangan |
| **Unit analisis** | **5.077 pohon unik** (dari 151.060 kotak anotasi) |
| Kelas positif | **66 pohon Unhealthy (1,30%)** — 17 / 31 / 18 per ortomosaik |
| Unit spasial | **3 ortomosaik** — 1.379 / 1.849 / 1.849 pohon |
| Fold | `split_fold = ortho` → **leave-one-ortho-out**, 3 lipatan |
| Metrik | mAP (deteksi) · PR-AUC & ROC-AUC (kesehatan). **Akurasi tidak dilaporkan** |
| **Batas yang dipaksakan** | **n positif unik = 66** ⇒ selang kepercayaan lebar, `is_unbalance` tidak dapat dibedakan dari derau; klaim maksimum = **demonstrator satu-situs kesehatan tajuk generik**, bukan detektor BSR |

**Mengapa unitnya pohon, bukan baris.** Ubin Roboflow diambil pada offset acak, bukan grid,
sehingga ubin saling bertindih dan satu pohon fisik muncul di **median 32 ubin** (rentang 1–77).
Tiap ortomosaik hanya seluas ~5.000 × 5.000 px — muat ~25 ubin 1.024 px — namun datasetnya
berisi 737–799 ubin; hanya **13 / 14 / 16** ubin yang benar-benar tidak bertindih
(`layer1_tiles_disjoint.csv`). Melatih pada 151.060 baris = melatih pada 5.077 pohon yang
direplikasi 29,8×.

**Mutu deduplikasi (terverifikasi, bukan asumsi).**

| Pemeriksaan | Hasil |
|---|---|
| Konflik label antar-duplikat | **0** — Roboflow menyalin kotak identik, bukan menganotasi ulang ⇒ deduplikasi tidak ambigu |
| Tetangga-hantu (< 0,5 × jarak tanam) | **0** di ketiga ortomosaik ⇒ 5.077 bukan over-count kotak terpotong |
| Tampilan kanonik ≥ 60 px dari tepi ubin | **5.048 / 5.077** ⇒ hampir tiap pohon punya satu tampilan tak-terpotong |

**Tampilan kanonik.** Tiap pohon diwakili oleh **satu** ubin: yang membuat pusat tajuk paling
jauh dari tepi ubin (paling kecil kemungkinan terpotong). 1.836 ubin terpakai, dibuka sekali.

---

## Dataset 2 — Panel epidemi lapangan 25 tahun (Lapisan 2, nyata)

| | |
|---|---|
| Berkas | `layer2_nodes.csv` · `layer2_panel.csv` · `layer2_edges.csv` |
| Sumber | Tisné et al. 2017, G3 7(6):1683–1692, doi:10.1534/g3.117.041764 — kebun SOCFINDO, Medan |
| Lisensi | **CC BY-SA 4.0**, hak cipta PalmElit & CIRAD — sitasi wajib (lihat `Eg9PP_LICENSE.md`) |
| Label | **Ganoderma / BSR sungguhan**, gejala terverifikasi lapangan per pohon |
| Unit analisis | **1.200 sawit**, 14 famili, 80 plot, 2 parcel |
| Rentang waktu | **45 tanggal sensus**, tahun 0,5 – 25,5 (panel 54.000 baris) |
| Kejadian | simptomatik **702 (58,5%)** · mati **366 (30,5%)** |
| Sensor | **498 pohon** tak pernah bergejala selama diamati; sensor terdini **t = 6,0 th** |
| Graf | kontak akar pada r = 1,5 × jarak tanam → **3.354 sisi, derajat rata-rata 5,59** |
| Fold | `fold = parcel` → **leave-one-parcel-out**, 2 lipatan |
| Metrik | PR-AUC (pos-rate 1,6–5,7%) |
| **Batas yang dipaksakan** | **Tidak ada citra sama sekali** dan kompartemen laten (E) **tidak teramati** — hanya waktu gejala pertama dan waktu mati. Kepala SEIR harus disederhanakan jadi SI(D); hasilnya **tidak dapat dibandingkan langsung** dengan varian SEIR simulator. Hanya **1 relasi** (akar): tak ada data angin maupun jalur panen ⇒ `n_rel = 1` |

**Dua sifat yang membuat block-CV di sini sah (terverifikasi).**

| Sifat | Nilai | Mengapa penting |
|---|---|---|
| Sisi lintas-parcel pada r = 1,5 | **0 dari 3.354** | 44A menempati Y 3,0–14,5 dan 44B Y 22,0–33,5 — terpisah spasial, jadi memisahkan fold **tidak memutus satu sisi pun**; tak ada informasi tetangga yang bocor lintas-fold |
| Famili yang hadir di kedua parcel | **14 dari 14** | fold parcel **tidak terkonfound genotipe** — perbedaan antar-fold bukan perbedaan susceptibilitas famili |

**Koreksi geometri.** Sumbu `X_POSITION` dan `Y_POSITION` pada data asli **tidak sekala**.
Setelah `xm = X × cos 30°`, keenam tetangga terdekat jatuh tepat di jarak **1,000** di kedua
parcel — tanam segitiga sama sisi sempurna. Tanpa koreksi ini grafnya salah.

**Grid sensus.** Sensus **tidak reguler** (bukan langkah 0,5 tahun rapi). Grid mentah berisi 46
tanggal berbeda; satu nilai di luar grid (12,2 pada 1 pohon) dirapatkan ke tanggal sensus
terdekat (12,0), menyisakan **45 tanggal**. Grid yang dipakai adalah tanggal yang **benar-benar
teramati**, bukan grid sintetis.

**Penanganan sensor (empat status, bukan dua).** Pohon yang keluar pengamatan diberi status
`C` dan **dikeluarkan dari risk set sejak saat itu** — tidak boleh dianggap sehat sampai akhir.

| Status | Arti | Masuk risk set |
|---|---|---|
| `A` | asimptomatik, masih diamati | ✅ |
| `S` | simptomatik, masih hidup | ❌ |
| `D` | mati | ❌ |
| `C` | keluar pengamatan (tersensor) | ❌ |

**Pos-rate tugas peramalan** (target: pohon `A` pada sensus *t* menjadi `S`/`D` dalam *h* sensus
berikutnya):

| h | contoh | positif | pos-rate |
|---|---|---|---|
| 1 | 44.311 | 701 | **1,58%** |
| 2 | 43.782 | 1.327 | **3,03%** |
| 3 | 43.225 | 1.924 | **4,45%** |
| 4 | 42.553 | 2.406 | **5,65%** |

---

## Dataset 3 — Spektra emisi simulator (Lapisan 2, sintetis)

| | |
|---|---|
| Berkas | `../data/pwd.csv` (di luar folder ini — jangan dipindah) |
| Isi | 1.226 spektrum, 240 band, λ 395,0–1004,3 nm; 1.029 sakit / 197 sehat |
| Label | **Penyakit layu pinus, bukan BSR, bukan sawit** |
| Peran | sumber distribusi fitur node simulator SEIR (S/I dari CSV; E interpolasi α = 0,1; R = 10% NIR terendah kolam sakit) |
| **Batas yang dipaksakan** | **tidak dapat direproduksi ulang** bila hilang — seluruh Lapisan 2 sintetis berhenti. Sifat spektralnya milik spesies dan patogen lain; ia menyediakan *bentuk* distribusi, bukan bukti spektral BSR |

---

## Yang tidak bisa disambungkan, dan mengapa itu dinyatakan

Dataset 1 dan Dataset 2 **tidak digabungkan**. Kebunnya berbeda, zamannya berbeda (petak Eg9PP
dibongkar 2012; citra DJI pasca-2013), tak ada kunci join, dan tak satu pun bersistem koordinat
bergeoreferensi. Yang diuji hanyalah **kompatibilitas antarmuka**: apakah graf keluaran Lapisan 1
sebangun dengan graf yang dikonsumsi Lapisan 2.

| Derajat rata-rata pada r = 1,5 × jarak tanam, **pohon bagian dalam** | |
|---|---|
| Eg9PP (posisi tanam) | **5,74** |
| Roboflow, **prediksi YOLOv12n** | **5,54 ± 0,12** (3 ortomosaik) |
| Roboflow, kotak kebenaran-dasar | 5,62 ± 0,05 |
| selisih prediksi ↔ Eg9PP | **3,5%** — keduanya kisi segitiga berderajat 6 |
| selisih prediksi ↔ kotak GT | 1,4% — biaya memakai detektor, bukan label |

Dibandingkan **hanya pada dataran r = 1,25–1,5**. Di luar itu Eg9PP melompat (kisi ideal)
sementara Roboflow melandai (posisi nyata berderau) — artefak presisi, bukan geometri berbeda.
Uji ini **tidak lagi** memakai kotak ground-truth. Angka 5,54 dihitung dari pusat tajuk yang
benar-benar dideteksi YOLOv12n pada ortomosaik yang **ditahan** di tiap lipatan, digabungkan ke
pohon unik, dengan ambang keyakinan dipilih **silang-lipatan** supaya ortomosaik uji tidak ikut
memilih apa pun tentang dirinya sendiri. Angka 5,62 pada kotak GT tetap dicantumkan sebagai
pembanding — selisih 1,4% di antara keduanya adalah biaya memakai detektor.

Perlu dinyatakan: 5,74 berada **di luar** pita ±0,12, jadi klaim yang sah adalah "berselisih
3,5%", bukan "keduanya sama". Reproduksi: `python layer1_build/centre_eval_folds.py`.
