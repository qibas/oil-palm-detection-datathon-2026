# Lapisan 1 — Data & Metodologi (Draf, Jalur A: concept paper jujur)

> Rancangan untuk bagian *Data & Metodologi* Lapisan 1 SawitGuard-GNN. Ditulis dengan disiplin
> datathon: harness evaluasi tetap, baseline sebelum kompleksitas, audit dicatat sebagai angka,
> dan tak ada klaim di luar bukti. `[isi]` = bagian yang perlu kamu lengkapi.

---

## 1. Data

### 1.1 Kesenjangan data (motivasi jujur)
Pemeriksaan menyeluruh (Indonesia, Malaysia, MPOB, Mendeley, Zenodo, IEEE DataPort, Nature)
menunjukkan **tidak ada dataset UAV ber-label BSR per-pohon yang tersedia publik**. Ini bukan
kelemahan metode kami melainkan **kondisi lapangan yang memang kami hadapi** — dan menjadi salah
satu kontribusi paper: memetakan sumber yang ada dan protokol yang jujur untuk data selangka ini.

### 1.2 Dataset rujukan / target — Sembawa (Universitas Indonesia)
Dataset paling relevan adalah milik Manessa dkk. (Frontiers in Remote Sensing, 2026;
doi:10.3389/frsen.2026.1788857), sebuah studi kasus di perkebunan **Sembawa, Sumatera Selatan**:

- UAV DJI Matrice 300 RTK; produk **RGB + multispektral** (Blok F 7,8 ha) dan hyperspektral 100-band (tidak kami pakai — di luar modalitas kami).
- **720 mask segmentasi tajuk individu** (anotasi manual, IoU antar-anotator 0,91).
- **Label BSR per-pohon terverifikasi lapangan** (biner simptomatik vs sehat; kriteria: klorosis tajuk >30%, keruntuhan pelepah dini, kerapatan tajuk turun).

Status: **diminta secara resmi** (surel on-request; pemilik menyatakan data tersedia "without
undue reservation"). **Karena tenggat concept paper tinggal 1 minggu, validasi Sembawa dijadikan
FUTURE WORK** — surel tetap dikirim, tetapi klaim empiris paper **tidak bergantung** pada kedatangannya.
Dalam paper, Sembawa disajikan sebagai **sumber validasi BSR yang dituju** untuk pekerjaan lanjutan.

### 1.3 Dataset demonstrator — citra tajuk UAV (tersedia sekarang)
Untuk mendemonstrasikan **tahap deteksi tajuk** Lapisan 1 secara konkret, kami memakai kumpulan
tile UAV nadir publik (Roboflow). **Audit kami mencatat apa adanya (CLOSED):**

| Properti | Nilai terukur |
|---|---|
| Gambar unik | **2.745 tile** (tiga dataset ternyata satu sumber yang di-fork 3×; inti 2.303 identik) |
| Modalitas / anotasi | UAV nadir RGB · **bounding box** (bukan mask) |
| Kelas | Healthy / Yellow / Small / Dead / Grass |
| **Makna kelas sebenarnya** | **Kesehatan tajuk umum + umur (Small = pohon muda) + tutupan lahan (Grass)** — **BUKAN BSR** |
| Ukuran tajuk | 46–100 px (median), tergantung resolusi tile |
| Kebocoran split bawaan | **100%** (hanya 3 ortomosaik, ketiganya tersebar di train/val/test) |
| Ketidakseimbangan | ~68:1 (sehat : tidak sehat) pada varian biner |

> **Deklarasi kejujuran (wajib di paper):** label demonstrator ini **kesehatan tajuk generik,
> bukan BSR**. Ia dipakai **hanya** untuk membuktikan tahap *deteksi & segmentasi tajuk* dapat
> berjalan pada citra UAV — **bukan** untuk mengklaim deteksi BSR. Semua klaim BSR bersandar pada
> dataset rujukan Sembawa (§1.2), bukan demonstrator ini.

> ⚠️ **Cek aturan Datathon 2026:** disiplin kompetisi melarang **melatih pada data eksternal**
> bila panitia menyediakan dataset resmi. Karena ini concept paper mandiri, konfirmasi dulu aturan
> sumber data Datathon; pastikan lisensi Roboflow (CC BY 4.0) dan status data Sembawa sesuai.

---

## 2. Metodologi Lapisan 1

Tujuan Lapisan 1: dari citra UAV → **deteksi tiap tajuk sawit → nilai kesehatan per-pohon** →
keluaran menjadi simpul untuk graf epidemi Lapisan 2.

### 2.1 Pipeline tiga tahap
1. **Deteksi tajuk** (RGB). Baseline off-the-shelf (mis. YOLO) — *baseline sebelum kompleksitas*.
2. **Ekstraksi fitur kesehatan per-pohon** (luas/geometri tajuk, greenness, indeks vegetasi).
3. **Klasifikasi kesehatan per-pohon** → keluaran menjadi simpul Lapisan 2.

**Jumlah & sudut pelepah — DICORET dari komponen aktif.** Audit membuktikan tak terekstrak pada
resolusi UAV (tajuk 40–100 px; pelepah menyatu jadi gumpalan). Disebut sebagai keterbatasan &
future-work (butuh GSD ≪ 5 cm), bukan fitur yang diklaim.

### 2.2 Apa yang bisa dibuktikan SEKARANG vs yang menunggu Sembawa (kejujuran ruang lingkup)
Ini pembeda kredibilitas paper: data-di-tangan (RGB, **kotak**, label kesehatan-umum) hanya bisa
mengevaluasi sebagian pipeline. Jangan kaburkan keduanya.

| Komponen | Bisa dibuktikan SEKARANG (demonstrator RGB, kotak) — **lingkup paper** | FUTURE WORK / pasca-tenggat (mask + multispektral + label BSR Sembawa) |
|---|---|---|
| Deteksi tajuk | ✅ **mAP** (kotak GT ada) | — |
| Segmentasi tajuk ber-**IoU** | ❌ **tidak** — demonstrator tak punya mask GT; hanya **luas tajuk kualitatif** via ExG/SAM di dalam kotak (lih. gambar `SEG_B_crownarea.jpg`) | ✅ IoU vs 720 mask Sembawa |
| Fitur greenness/warna (RGB) | ✅ deskriptif | ✅ |
| **NDVI/NDRE** (stres dini) | ❌ **tidak** — demonstrator RGB-only | ✅ dari multispektral Sembawa |
| Klasifikasi kesehatan per-pohon | ✅ tapi pada **label kesehatan-umum**, **bukan BSR** | ✅ **BSR** per-pohon terverifikasi |

> Dua dari tiga fitur khas Lapisan 1 (IoU segmentasi, NDVI/NDRE) **bergantung pada Sembawa**, bukan
> "sudah dibuktikan sekarang." Menyajikan ini apa adanya justru memperkuat Jalur A — menunjukkan
> kamu tahu persis batas datamu.

### 2.3 Protokol evaluasi (bagian disiplin — ini yang bikin kredibel)
- **Harness tetap: block-split per ortomosaik (leave-one-ortho-out CV).** JANGAN split acak
  per-tile — akan bocor 100% (audit) dan melaporkan angka palsu. Ini "satu harness CV tetap"
  versi visi-komputer.
- **Noise band dilaporkan:** rata-rata ± std antar-fold/seed. **Perbaikan lebih kecil dari ~1 std = noise** sampai terbukti multi-seed.
- **Metrik jujur untuk kelas timpang:** **PR-AUC / F1 kelas minoritas**, bukan akurasi (akurasi menyesatkan pada 68:1).
- **Klaim maksimum per dataset:**
  - Demonstrator (sekarang) → *"deteksi tajuk sawit dari UAV berhasil (mAP), luas tajuk terekstrak kualitatif"* — **tanpa** klaim IoU, tanpa klaim BSR.
  - Sembawa (bila tiba) → *"skrining simptomatik BSR per-pohon di satu situs"* dengan block-CV + IoU; tetap PoC, bukan generalisasi.

### 2.4 Ledger hasil-negatif (cantumkan; ini kekuatan, bukan kelemahan)
| Yang diuji | Hasil | Status |
|---|---|---|
| 3 dataset Roboflow sebagai data BSR | Ternyata satu fork, label kesehatan umum, bukan BSR | Ditolak |
| Jumlah/sudut pelepah sebagai fitur BSR dini | Tak terekstrak di resolusi UAV | Ditolak |
| Split acak per-tile | Bocor 100% (3 orto) | Diganti block-split |
| *(Rujukan silang — temuan Lapisan 2, bukan Lapisan 1)* struktur graf sebagai sumber sinyal | Hanya ~+0,01; graf acak ≈ graf benar | Dicatat; sinyal utama = temporal + prevalensi |

---

## 3. Keterbatasan (nyatakan terang-terangan)
- Data rujukan Sembawa: **satu kebun, ~194 pohon, label biner** → **proof-of-concept satu-situs**, bukan bukti detektor tergeneralisasi.
- Demonstrator: **bukan BSR**; hanya memvalidasi tahap deteksi tajuk.
- Modalitas: multispektral menangkap stres dini; **RGB saja hanya melihat gejala lanjut**.
- Perlu konfirmasi ke pemilik data: berapa pohon berlabel di cakupan RGB/multispektral, dan komposisi band multispektral.

---

### Catatan pemetaan ke rambu /ml-competition
Harness tetap (§2.2) · baseline sebelum kompleksitas (§2.1) · audit sebagai angka CLOSED (§1.3) ·
ledger negatif (§2.3) · tak ada klaim dalam noise band (§2.2) · larangan data-eksternal untuk latih (§1.3 ⚠️).
