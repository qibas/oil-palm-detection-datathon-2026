# SawitGuard-GNN — Paket Reproduksi

Konsep sistem peringatan dini **Busuk Pangkal Batang (BSR / *Ganoderma boninense*)** pada sawit.
**Datathon 2026, Ristek CSUI.**

---

# ➜ MULAI DI SINI: [`00_HASIL.md`](00_HASIL.md)

Satu berkas berisi **pipeline lengkap** dan **seluruh hasil eksperimen**. Kalau cuma sempat baca
satu berkas, baca itu. Angka mentahnya di [`00_RINGKASAN.csv`](00_RINGKASAN.csv) (38 baris).

**Tiga temuan utama:**

1. **Peta kontak yang benar bekerja** — pada data lapangan Ganoderma 25 tahun, struktur graf
   menyumbang **+0,0151 AUC-PR (39 dari 40 pasangan)**, bertahan terhadap null permutasi
   terkontrol-genotipe (kelebihan 1,25–1,29×, **0 dari 500** permutasi mencapainya).
2. **Lapisan epidemiologi terlatih justru merugikan** — kepala SI(D) **NEG di keempat horizon**,
   memburuk seiring horizon. Dugaan mudahnya (inisialisasi) sudah diuji dan ditolak untuk h≥2.
3. **Sebagian besar "efek tetangga" adalah confounding waktu** — RR 4,47× runtuh jadi **1,65×**
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
pip install numpy pandas scipy scikit-learn torch lightgbm opencv-python ultralytics roboflow

# 1. bekukan dataset bersih (asersi berhenti kalau ada angka meleset) — opsional, CSV sudah ada
cd data_clean
python build_layer1.py         # -> 5.077 pohon unik, 66 positif
python build_layer2_real.py    # -> 1.200 node, 45 sensus, 3.354 sisi

# 2. Lapisan 1  (CPU ~20 dtk; YOLO butuh GPU)
cd ../layer1_build
python exp_health.py
python yolo_prep.py && FOLDS=0,1,2 EPOCHS=50 IMGSZ=640 python yolo_train.py

# 3. Lapisan 2 — Eg9PP  (CPU; total ~24 mnt)
cd ../layer2_real
python test_dataset.py         # ~90 asersi + 4 penjaga kebocoran  (~10 dtk)
python run_real.py             # dekomposisi -> results_real.csv   (~22 mnt)
python perm_null.py            # null permutasi                     (~45 dtk)
```

Terverifikasi: `python layer2_real/test_dataset.py` dijalankan **dari dalam paket ini** dan lulus
seluruh pemeriksaan (exit 0).

---

## Enam larangan yang berlaku di seluruh paket

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

---

## Catatan teknis

Citra di `layer1_build/ds_B/` adalah **hardlink** ke repo kerja — nol byte tambahan di disk, tapi
berperilaku seperti berkas biasa untuk semua alat termasuk zip. Kode dan dokumen adalah salinan
sungguhan, jadi menyunting paket ini **tidak** mengubah repo aslinya.

Ukuran: **~21 MB nyata** + 377 MB ter-hardlink.
