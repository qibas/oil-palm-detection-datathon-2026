# Lapisan 1 — Hasil Demonstrator (data yang ada, Jalur A)

> Demonstrator jujur tahap Lapisan 1 pada **dataset B** (2.303 tile UAV nadir, 1024², biner
> Healthy/Unhealthy). **Bukan BSR** — label kesehatan-tajuk generik (lihat `../layer1_data_audit/AUDIT_REPORT.md`).
> Semua evaluasi memakai **block-CV leave-one-ortho-out** (3 ortomosaik), bukan split acak (bocor 100%).

## Ringkasan hasil

### Tahap 2 — Klasifikasi kesehatan per-pohon (LightGBM, fitur tajuk) — **3-fold block-CV penuh**

> **Basis analisis diperbaiki 2026-07-23.** Unit analisis kini **pohon unik**, bukan baris anotasi.
> Ubin Roboflow bertindih ~30×, jadi angka lama (151.060 tajuk) melatih dan menguji pada 5.077 pohon
> yang direplikasi. Sumber data: `../data_clean/layer1_crowns.csv` (lihat `../data_clean/DATASET_CARD.md`).

- **5.077 pohon unik**, base rate Unhealthy = **1,30%** (**66 positif unik**; 17 / 31 / 18 per ortomosaik). PR-AUC acak = 0,0130.
- **Vanilla LightGBM: PR-AUC = 0,182 ± 0,059 · ROC-AUC = 0,861** (mean ± std antar-3-ortho).
- `is_unbalance=True`: PR-AUC 0,181 ± 0,091 — **di dalam noise band** (Δ0,001 ≪ 1 std) → **tetap ditolak**.
- Fitur teratas: (G−R), R_std, exg_std, (G−B), B_mean → **sinyal warna/greenness** (konsisten dengan klorosis).
- **UNDERPOWERED — wajib dinyatakan.** Seluruh hasil bersandar pada 66 pohon sakit; lipatan terkecil hanya 17 positif. Std ±0,059 dihitung dari **3 angka**. Angka apa pun di sini harus dibaca sebagai berbasis sampel kecil, bukan estimasi stabil.
- Metrik: PR-AUC / ROC-AUC (threshold-free); F1@0.5 tak informatif pada imbalance seekstrem ini.

#### Dua koreksi terhadap laporan sebelumnya
| | Basis anotasi (BATAL) | Basis pohon unik (berlaku) |
|---|---|---|
| n / positif | 151.060 / 2.179 | **5.077 / 66** |
| Vanilla PR-AUC | 0,126 ± 0,068 | **0,182 ± 0,059** |
| PR-AUC per fold | 0,030 / 0,17 / 0,26 | **0,264 / 0,155 / 0,126** |

1. **PR-AUC naik, bukan turun.** Pseudo-replikasi ternyata *menekan* metrik, bukan menggelembungkannya — bobot efektif tiap pohon sebanding jumlah ubin yang memuatnya, dan pohon sakit kebetulan muncul di lebih sedikit ubin.
2. **"Kolaps satu ortomosaik" adalah artefak duplikasi.** Laporan lama menyebut `52000_20000` kolaps ke PR-AUC 0,030 sebagai bukti variasi antar-situs. Pada pohon unik ortomosaik itu mencapai **0,126** dan ketiga lipatan berada di rentang 0,13–0,26. Variasi antar-situs **tetap ada tetapi jauh lebih kecil** dari yang pernah kami laporkan; klaim kolaps ditarik.

### Tahap 1 — Deteksi tajuk (YOLO11n, leave-one-ortho-out)
- **Preliminary (in-session): fold0 (hold 44000_16000), 15 epoch, imgsz 512, yolo11n** →
  **mAP50 = 0,758 · mAP50-95 = 0,524 · Precision = 0,862 · Recall = 0,683** (pada ortomosaik held-out).
- Deteksi tajuk sawit dari UAV **berhasil** (presisi tinggi, recall ~2/3 pada model nano/15 epoch) → tahap deteksi Lapisan 1 layak.
- **CATATAN: ini SATU fold** (bukan mean ± std). Untuk paper, jalankan full 3-fold + epoch lebih panjang sendiri:
  `FOLDS=0,1,2 EPOCHS=50 IMGSZ=640 python yolo_train.py` (~15–20 mnt/fold di RTX 5060) → laporkan mAP mean ± std antar-3-ortho.

### Tahap 3 — Segmentasi luas tajuk (kualitatif)
- ExG+Otsu memisahkan tajuk dari tanah (lihat `../layer1_data_audit/SEG_B_crownarea.jpg`).
- **Tanpa klaim IoU** — demonstrator tak punya mask GT (IoU menunggu Sembawa, future work).

## Klaim maksimum yang jujur (untuk paper)
> "Kami mendemonstrasikan tahap deteksi & penilaian-kesehatan tajuk sawit per-pohon dari citra
> UAV RGB: deteksi tajuk mAP50 = 0,76 pada ortomosaik held-out, dan klasifikasi kesehatan
> ROC-AUC = 0,86 / PR-AUC = 0,18 ± 0,06 (block-CV antar-ortomosaik, **5.077 pohon unik**).
> Hasil ini **underpowered**: hanya **66 pohon Unhealthy unik** di seluruh tiga ortomosaik, jadi
> selang kepercayaannya lebar dan `is_unbalance` tak dapat dibedakan dari derau. Label bersifat
> **kesehatan-tajuk generik, bukan BSR**; validasi BSR per-pohon (data Sembawa) adalah pekerjaan
> lanjutan."

## Cara menjalankan (reproduksi)
```bash
# 0. env: pip install roboflow ultralytics lightgbm scikit-image opencv-python imagehash
#    (GPU: torch CUDA — JANGAN biarkan pip ultralytics menimpa torch CUDA-mu!)
setx ROBOFLOW_API_KEY "kunci_kamu"          # atau $env:ROBOFLOW_API_KEY="..."
python download.py                          # unduh ds_A/B/C (COCO)
python ../data_clean/build_layer1.py        # WAJIB DULU: bekukan 5.077 pohon unik
python exp_health.py                        # Tahap 2: LightGBM health block-CV (CPU, ~20 dtk)
python yolo_prep.py                         # siapkan yolo_B + 3 fold
FOLDS=0,1,2 EPOCHS=50 IMGSZ=640 python yolo_train.py   # Tahap 1: deteksi 3-fold (GPU)
python seg.py                               # Tahap 3: figur segmentasi tajuk
```

## Catatan disiplin (/ml-competition)
Harness tetap (block-CV per-ortho) · baseline dulu (LightGBM/yolo11n vanilla) · noise band dilaporkan ·
`is_unbalance` ditolak karena di dalam noise · metrik = PR-AUC/mAP (bukan akurasi) · imbalance & kolaps-situs dilaporkan apa adanya.
