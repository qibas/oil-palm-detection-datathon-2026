# SawitGuard-GNN — Paket Reproduksi

Konsep sistem peringatan dini **Busuk Pangkal Batang (BSR / *Ganoderma boninense*)** pada sawit.
**Datathon 2026, Ristek CSUI.**

---

# ➜ MULAI DI SINI: [`00_HASIL.md`](00_HASIL.md)

Satu berkas berisi **pipeline lengkap** dan **seluruh hasil eksperimen**. Kalau cuma sempat baca
satu berkas, baca itu. Angka mentahnya di [`00_RINGKASAN.csv`](00_RINGKASAN.csv) (98 baris).

⚠ **Sebelum mengutip angka ke naskah:** buka [`00_ANGKA_FINAL.md`](00_ANGKA_FINAL.md). Paket
ini memuat angka dari sembilan konfigurasi berbeda dan beberapa pasang di antaranya **tidak
sebanding**; berkas itu menandai mana yang masuk paper sebagai angka utama.

**Empat temuan utama:**

1. **Peta kontak yang benar bekerja** — pada data lapangan Ganoderma 25 tahun, struktur graf
   menyumbang **+0,0151 AUC-PR (39 dari 40 pasangan)**, bertahan terhadap null permutasi
   terkontrol-genotipe (kelebihan 1,25–1,29×, **0 dari 500** permutasi mencapainya).
2. **Riwayat waktu ternyata tidak dibutuhkan** — dinilai **dalam-sensus** (satu-satunya
   metrik yang adil untuk memeringkat di dalam satu bidikan), varian yang hanya memakai
   kondisi tetangga — persis yang bisa diberikan satu foto drone — **menyamai** model penuh
   (+0,0042 ± 0,0035, 36/40), dan **77%** kemampuannya datang khusus dari peta kontak yang
   benar (+0,0296, **40/40**). Ongkosnya diukur: kontaminasi kekerabatan **36%**, dan
   **59% sinyal bertahan** saat masukan datang dari detektor.
3. **Lapisan epidemiologi terlatih justru merugikan** — kepala SI(D) **NEG di keempat horizon**,
   memburuk seiring horizon. Dugaan mudahnya (inisialisasi) sudah diuji dan ditolak untuk h≥2.
4. **Sebagian besar "efek tetangga" adalah confounding waktu** — RR 4,47× runtuh jadi **1,65×**
   setelah stratifikasi per sensus. Yang bertahan tetap nyata, tapi jauh lebih kecil.

---

## Cakupan

Paket ini memakai **data nyata saja**: citra UAV Roboflow (Lapisan 1) dan data lapangan Eg9PP
(Lapisan 2). Repo kerja juga memuat **simulator SEIR sintetis** dengan rangkaian ablasinya; itu
**sengaja dikeluarkan** dari paper ini agar cakupannya tidak melebar dan agar setiap angka yang
dilaporkan berasal dari kebun sungguhan.

⚠ Konsekuensi: `paper/section3.tex` dan `paper/METHODOLOGY_PLAN.md` **masih menulis tentang
simulator**. Keduanya naskah yang belum diperbarui — lihat blok status di bawah.

---

## ⚠ Status naskah — baca sebelum mengutip apa pun dari `paper/`

`section3.tex` belum ditulis ulang. Empat masalah terverifikasi masih ada di dalamnya:

| # | Masalah |
|---|---|
| 1 | Masih membahas simulator SEIR, yang sudah di luar cakupan paket ini |
| 2 | Menyatakan Tahap 5 menjalankan dinamika **"di atas geometri kebun yang nyata"** — kode simulator tak pernah membaca awan titik Lapisan 1 |
| 3 | §3.6–3.7 menjanjikan kalibrasi, skor Brier, kurva keandalan, tingkat risiko, presisi@k. **Tak ada kode yang menghasilkannya** |
| 4 | Ketimpangan kelas ditulis **69:1** (tingkat anotasi); pada **pohon unik** angkanya **~76:1** (5.011 : 66) |

`section3.docx` lebih basi lagi. `METHODOLOGY_PLAN.md` berisi rencana penulisan ulangnya.
Belum ada hasil Eg9PP yang masuk ke naskah mana pun.

---

## Isi

| Folder | Isi |
|---|---|
| `data_clean/` | **Pintu masuk semua eksperimen.** CSV beku + dua skrip pembangunnya + `DATASET_CARD.md` + lisensi Eg9PP |
| `layer1_data_audit/` | Audit data + 8 citra bukti: tiga dataset Roboflow adalah **satu dataset di-fork 3×**, dan pseudo-replikasi 29,8× |
| `layer1_build/` | Pipeline UAV: deteksi tajuk, kesehatan, segmentasi. Termasuk `ds_B` (2.303 ubin citra sumber) |
| `layer2_real/` | Eg9PP: dataset, model, driver, null permutasi, dan **~90 asersi penjaga kebocoran** |
| `paper/` | Naskah + rencana penulisan ulang |
| `figures/` | `fig_pipeline.png` + skripnya |
| `models.py` | `MLPBaseline` — dipakai bersama oleh `layer2_real/models_real.py` |

## Urutan baca

1. **[`00_HASIL.md`](00_HASIL.md)** — pipeline + seluruh hasil
2. `data_clean/DATASET_CARD.md` — fakta data yang boleh diklaim, plus **"batas yang dipaksakan"** per dataset
3. `layer1_data_audit/AUDIT_REPORT.md` — audit + pseudo-replikasi 29,8×
4. `layer1_build/RESULTS_LAYER1.md` — hasil Lapisan 1, termasuk klaim yang kami tarik
5. `paper/METHODOLOGY_PLAN.md` — rencana merangkainya jadi Bab 3

Nilai kerja yang mengikat seluruh isi folder ini: **leakage-first** — kebocoran, nol, dan
keterbatasan dilaporkan **sebelum** hasil apa pun, dan tidak ada perbaikan yang diklaim bila ia
berada di dalam band derau. Beberapa hasil di sini **negatif** dan **satu klaim kami tarik
sendiri**. Itu disengaja.

---

## Reproduksi

Tata letak paket ini **mencerminkan repo kerja**, jadi semua skrip jalan di tempat.

```bash
pip install -r requirements.txt        # versi yang dipakai saat angka dihasilkan
# GPU untuk MELATIH: pasang torch cu126 lebih dulu (BUKAN cu130 - Pascal dibuang di CUDA 13)
#   pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.13.0+cu126" "torchvision==0.28.0+cu126"
# DEMO tidak butuh GPU.

# 1. bekukan dataset bersih (asersi berhenti kalau ada angka meleset) — opsional, CSV sudah ada
cd data_clean
python build_layer1.py         # -> 5.077 pohon unik, 66 positif
python build_layer2_real.py    # -> 1.200 node, 45 sensus, 3.354 sisi

# 2. Lapisan 1  (kesehatan CPU ~20 dtk; deteksi butuh GPU, ~45 mnt/lipatan)
cd ../layer1_build
python exp_health.py                  # kesehatan tajuk, LightGBM, leave-one-ortho-out
FOLDS=all python train_folds_gpu.py   # YOLOv12n, 3 lipatan x 30 epoch (y12.build otomatis)
python centre_eval_folds.py           # METRIK UTAMA Tahap 1 + angka uji jembatan
python unhealthy_threshold.py         # ambang kelas Unhealthy (hasil: TETAP 0,75)
python detect_centres.py <citra>      # citra apa pun -> pusat tajuk + graf
# jalur YOLO11 lama (yolo_prep.py / yolo_train.py) masih ada tapi DIGANTIKAN oleh y12.py

# Lari VERIFIKASI satu lipatan (~20 mnt, 1 GPU) - buka run_1fold_yolov12n.ipynb.
# Setelan identik (yolo12n, 30 epoch, imgsz 640, seed 42), ruang nama terpisah
# (`_1fold`), jadi ia tidak dapat menimpa hasil atau bobot 3 lipatan.
# 1 lipatan = TANPA std: angkanya BUKAN angka naskah dan tidak boleh diadu
# dengan F1 pusat 0,960 +/- 0,024. Lihat kotak batas di sel pertama notebook.

# 3. Lapisan 2 — Eg9PP  (CPU; total ~24 mnt)
cd ../layer2_real
python test_dataset.py         # ~90 asersi + 4 penjaga kebocoran  (~10 dtk)
python run_real.py             # dekomposisi -> results_real.csv   (~22 mnt)
python perm_null.py            # null permutasi                     (~45 dtk)

# varian foto-tunggal (v3): buang waktu + genotipe, nilai DALAM-SENSUS
python run_v3.py 20            # -> results_v3.csv                  (~12 mnt)
python run_v3_perm.py 200 2    # null dalam-famili -> results_v3_perm_progeny.csv        (~28 mnt)
STRATA=progeny_parcel python run_v3_perm.py 200 2   # strata terketat  (~28 mnt)
RECALL=0.446 FPR=0.0094 python run_v3_noisy.py 10 20   # ongkos substitusi detektor (~6 mnt)
python train_final_v3.py       # checkpoint 1-kolom untuk demo -> stgnn_v3_photo.pt
# v3 MELANGGAR kontrak dengan sengaja (WINDOW=1, genotipe dibuang).
# Alasan dan ongkos terukurnya: layer2_real/INTERFACE.md bagian akhir.
```


## Demo web

```bash
python demo_api.py            # http://localhost:8000
```

React + Babel **di-vendor lokal** di `web/vendor/`, jadi demo berjalan tanpa internet
dan tanpa `npm install`. Seluruh perhitungan ada di `demo_core.py`; `demo_api.py` cuma
lapisan HTTP, dan `demo_app.py` (Streamlit) adalah cadangan yang memanggil core yang sama —
keduanya tidak mungkin memberi angka berbeda.

```bash
python demo_core.py           # cetak semua angka layar + render gambar, TANPA server
node web/check_jsx.js         # WAJIB setelah menyunting web/app.jsx
```

`check_jsx.js` memeriksa dua hal yang lolos dari pencocokan string: **sintaks JSX**
(berkas rusak tetap dikirim server dengan status 200) dan **escape unicode yang bocor**
(`\u2014` yang terlanjur jadi teks; pernah terjadi 23 kali sekaligus).

Berkas yang WAJIB ada agar demo jalan — pastikan ikut ter-commit:

| berkas | isi |
|---|---|
| `layer2_real/stgnn_v3_photo.pt` | checkpoint Lapisan 2 varian foto (54 KB) |
| `layer2_real/risk_ranked.csv` | peringkat Eg9PP untuk layar Bukti |
| `layer1_build/yolo12_runs/yolo12n_base_fold*/weights/best.pt` | bobot detektor (lari 3 lipatan; angka naskah) |
| `layer1_build/yolo12_runs/yolo12n_base_1fold_fold0_s42/weights/best.pt` | bobot yang DIPAKAI demo lebih dahulu; tanpa berkas ini demo jatuh ke bobot 3 lipatan dan tetap jalan |
| `web/vendor/*.js` | React, ReactDOM, Babel (3,2 MB) |
| `data_clean/*.csv` | dataset beku |

`layer1_build/yolo12/` dan `layer1_build/anom_data/` **sengaja tidak dilacak** — keduanya
direktori turunan yang dibangun ulang `y12.build()` dan `anom.build()` dalam hitungan detik,
dan berkas lipatannya memuat path absolut mesin pembangunnya.

Terverifikasi: `python layer2_real/test_dataset.py` dijalankan **dari dalam paket ini** dan lulus
seluruh pemeriksaan (exit 0).

---

## Enam larangan yang berlaku di seluruh paket

⚠ **Penomoran.** Ada **dua** daftar enam larangan di paket ini: yang di bawah berlaku untuk
seluruh paket, dan `layer2_real/INTERFACE.md` punya daftarnya sendiri khusus Lapisan 2 dengan
isi **dan nomor yang berbeda**. Saat mengutip, sebut daftarnya — "larangan #5 `INTERFACE.md`"
(genotipe wajib) bukan hal yang sama dengan "#5" di bawah (pohon tersensor).

1. **Jangan** memakai split acak pada Lapisan 1 — hanya 3 ortomosaik, split acak bocor 100%.
2. **Jangan** mengutip 151.060 sebagai ukuran sampel. Unitnya **5.077 pohon unik**.
3. **Jangan** menyebut label Roboflow sebagai BSR. Itu kesehatan tajuk generik tanpa verifikasi lapangan.
4. **Jangan** melaporkan akurasi. Pos-rate 1,3–6,0%; klasifikator yang selalu menjawab "sehat" sudah >98% akurat tanpa guna apa pun.
5. **Jangan** menganggap pohon tersensor sebagai sehat. Ia keluar dari risk set.
6. **Jangan** membandingkan kepala SI(D) dengan varian SEIR mana pun. Kompartemen laten E **tak teramati** di data lapangan; kepalanya turun dari 112 parameter jadi 3.

---

## Data pihak ketiga — atribusi wajib

**Eg9PP** — Tisné S., Pomiès V., Riou V., Syahputra I., Cochard B., Denis M. (2017),
*Identification of Ganoderma disease resistance loci using natural field infection of an oil palm
multi-parent population*, **G3** 7(6):1683–1692, doi:`10.1534/g3.117.041764`.
Hak cipta **PalmElit & CIRAD**, lisensi **CC BY-SA 4.0**. Detail: `data_clean/Eg9PP_LICENSE.md`.

**Roboflow ds_B** — `health-detection/oil-palm-health-detection`. Lihat berkas README Roboflow di
dalam `layer1_build/ds_B/`.

**Anomali tajuk sawit Peru (`PalmAnom` / `PalmSan`)** — Mendeley Data,
doi:`10.17632/nh7d23dgnw.1`, lisensi **CC BY 4.0**. Dipakai sebagai jalur bukti ketiga yang
berdiri sendiri (`00_HASIL.md` §2.6); **tidak pernah digabung** dengan ds_B.

---

## Catatan teknis

Citra di `layer1_build/ds_B/` adalah **hardlink** ke repo kerja — nol byte tambahan di disk, tapi
berperilaku seperti berkas biasa untuk semua alat termasuk zip. Kode dan dokumen adalah salinan
sungguhan, jadi menyunting paket ini **tidak** mengubah repo aslinya.

Ukuran: **~21 MB nyata** + 377 MB ter-hardlink.
