# Lapisan 1, Tahap 0 — Audit Data (SawitGuard-GNN)

**Tanggal:** 2026-07-16 · **Status:** audit selesai, tidak ada model dilatih.
**Kandidat:** A = `gunadarma/oil-palm-health-vglxy` · B = `health-detection/oil-palm-health-detection` · C = `oil-palm-health-detection/oil-palm-tree-health-detection`

Semua angka di bawah berasal dari **berkas yang diunduh** (COCO), bukan klaim halaman. Gambar contoh ada di folder ini.

---

## Ringkasan 6 jawaban yang diminta

1. **Berapa gambar unik?** → **2.745 tile unik**, bukan 7.500. Inti 2.303 tile identik di A, B, C. C menambah 442 tile. A menggelembung ke 3.915 karena augmentasi 1,70×.
2. **Udara/darat? Kotak/mask?** → **Udara (UAV nadir), kotak (bounding box).** Ketiganya `object-detection` — **tidak ada mask segmentasi** di mana pun.
3. **Kelas sebenarnya tentang apa?** → **Kesehatan tajuk umum + tutupan lahan + umur pohon.** Bukan tentang BSR.
4. **BSR, kesehatan umum, atau bukan?** → **Kesehatan sawit umum.** Tidak ada label BSR, tidak ada basidiokarp yang terlihat, tidak ada "rok" pelepah runtuh yang bisa dipastikan.
5. **Jumlah/sudut pelepah bisa diekstrak?** → **TIDAK.** Tajuk 40–100 px; pelepah menyatu jadi gumpalan. Luas tajuk: bisa. Jumlah/sudut pelepah: tidak.
6. **Rekomendasi:** → **Jangan sajikan sebagai data BSR.** Jika perlu demonstrator deteksi-tajuk Lapisan 1, pakai **B saja** (bersih, 1024², biner) dan **buang komponen fitur struktural pelepah**. Untuk klaim BSR sejati, tetap kejar dataset on-request.

---

## Gerbang 1 — Inventaris & duplikasi

**Ini SATU dataset yang di-fork tiga kali. Kecurigaanmu benar.**

Bukti paling kuat = nama berkas asli Roboflow (bagian sebelum `.rf.`), yang bertahan meski resolusi berbeda:

| | File di disk | Tile unik | Augmentasi | Dimensi | Anotasi |
|---|---|---|---|---|---|
| A | 3.915 | **2.303** | 1,70× | 416×416 | kotak |
| B | 2.303 | **2.303** | 1,00× | 1024×1024 | kotak |
| C | 2.946 | **2.745** | 1,07× | 640×640 | kotak |

Tumpang tindih (berdasarkan nama tile asli):
- **A ∩ B = 2.303** → 100% A, 100% B (tile identik).
- **A ∩ C = 2.303** → 100% A, 83,9% C.
- **B ∩ C = 2.303** → 100% B, 83,9% C.
- **Ketiganya berbagi 2.303 tile inti yang sama. Gabungan unik = 2.745.**

Catatan: phash silang-dataset hanya menemukan 17–30% karena resolusi berbeda (416/640/1024) + re-encode JPEG mendorong jarak Hamming melewati ambang — itu **batas bawah**, bukan kebenaran. Nama berkas membuktikan 100% inti sama. Dimensi seragam per dataset + nama tile berkoordinat (`44000_16000_3758_3651`) = tanda pasti **tiling dari satu ortomosaik**. Beberapa tile C bernama `DJI_0104` → **rekaman drone DJI** (konfirmasi UAV).

### ⚠️ Kebocoran split spasial — TEMUAN KRITIS
Seluruh "dataset" hanya berasal dari **3 ortomosaik** (region prefix: `52000_20000` = 799 tile, `44000_4000` = 767, `44000_16000` = 737). **Ketiga region muncul di train, valid, DAN test sekaligus** → **100% tile (2.303/2.303) bocor antar-split.** Split bawaan Roboflow acak per-tile, jadi tile bersebelahan dari penerbangan yang sama (pencahayaan sama, pohon yang sama terpotong di batas tile) tersebar ke train dan test. **Metrik apa pun pada split bawaan digelembungkan oleh autokorelasi spasial** — persis jebakan yang ingin kita hindari.

Implikasi lebih dalam: hanya ada **3 unit spasial independen**. Split blok yang jujur (tahan seluruh region) berarti latih ~2 orto, uji 1 orto — dasar yang sangat tipis untuk klaim generalisasi apa pun.

### ⚠️ Pseudo-replikasi 29,8× — TEMUAN KRITIS KEDUA (ditambahkan 2026-07-23)

Tabel "Tile unik" di atas menghitung **ubin**, dan itu menyesatkan soal berapa banyak **pohon**
yang sesungguhnya ada. Nama ubin menyimpan offset potongannya (`<ortho>_<offx>_<offy>`), jadi
setiap kotak anotasi bisa dikembalikan ke koordinat global ortomosaik. Setelah dikembalikan:

> **151.060 kotak anotasi ds_B = hanya 5.077 pohon unik (29,8×).**

Penyebabnya: **potongan ubin diambil pada offset ACAK, bukan grid.** Ubin karena itu saling
bertindih rapat — satu pohon fisik muncul di **median 32 ubin** (rentang 1–77). Tiap ortomosaik
hanya seluas ~5.000 × 5.000 px (muat ~25 ubin 1.024 px) tetapi datasetnya berisi 737–799 ubin;
pemilihan greedy hanya menemukan **13 / 14 / 16** ubin yang benar-benar tidak bertindih.

| Ortomosaik | Kotak anotasi | Pohon unik | Unhealthy unik | Ubin | Ubin saling-lepas |
|---|---|---|---|---|---|
| 44000_16000 | — | **1.379** | **17** (1,23%) | 737 | 13 |
| 44000_4000 | — | **1.849** | **31** (1,68%) | 767 | 14 |
| 52000_20000 | — | **1.849** | **18** (0,97%) | 799 | 16 |
| **total** | **151.060** | **5.077** | **66 (1,30%)** | 2.303 | 43 |

Deduplikasi ini **tidak ambigu**, dan itu diverifikasi, bukan diasumsikan:
- **0 konflik label** antar-duplikat → Roboflow menyalin kotak identik, bukan menganotasi ulang.
- **0 tetangga-hantu** (< 0,5 × jarak tanam) di ketiga ortomosaik → 5.077 bukan over-count akibat kotak terpotong tepi ubin.
- **5.048 / 5.077** pohon punya setidaknya satu tampilan dengan pusat ≥ 60 px dari tepi ubin → tersedia "tampilan kanonik" tak-terpotong untuk hampir semua pohon.

**Konsekuensi yang harus dinyatakan di paper:** kelas positif sesungguhnya berjumlah **66 pohon**,
bukan 2.179 kotak. Setiap selang kepercayaan wajib dihitung pada tingkat pohon unik. Duplikasi ini
**tidak bocor antar-fold** (semua salinan satu pohon berada di ortomosaik yang sama, dan block-CV
menahan ortomosaik utuh), tetapi ia **membobot ulang** data: pengaruh tiap pohon sebanding jumlah
ubin yang memuatnya. Dataset beku bertingkat-pohon ada di `../data_clean/layer1_crowns.csv`.

## Gerbang 2 — Audit semantik

Lihat `A_full_images.jpg`, `B_full_images.jpg`: citra nadir kebun sawit, tajuk bintang tersusun baris, jalan tanah, petak tanam-ulang gundul. **Definitif udara, satu gambar = banyak pohon** (~29–64 tajuk/gambar).

Arti kelas (dari piksel, bukan nama):

| Label | Sebenarnya | Bukti |
|---|---|---|
| Healthy | Tajuk hijau padat dewasa | `ZOOM_B_Healthy.jpg` |
| **Small (A) / immature (C)** | **Pohon MUDA** — tajuk kecil, jarak lebar, banyak rumput di antara | `ZOOM_A_Small.jpg`, `ZOOM_C_immature.jpg` |
| Yellow (A) / stressed (C) | Tajuk pucat/klorosis atau tipis-terbuka; **penyebab tak terbedakan** | `ZOOM_A_Yellow.jpg`, `ZOOM_C_stressed.jpg` |
| Dead (A) / unhealthy (C) | Pohon mati/jarang; **banyak kotak jatuh di tanah nyaris kosong** (label berisik) | `ZOOM_A_Dead.jpg` |
| **Grass** | **Rumput/tutupan lahan — bukan kelas pohon** | grid |
| Unhealthy (B) | Gabungan stressed+yellow+dead jadi satu bin "tak sehat" | `ZOOM_B_Unhealthy.jpg` |

**Vonis satu kalimat:**
> Kelas-kelas ini paling tepat dideskripsikan sebagai **tingkat kesehatan tajuk sawit yang kasar dan tercampur dengan umur pohon (Small/immature) dan tutupan lahan (Grass)**, dan klaim terkuat yang bisa dipertahankan adalah **"mendeteksi dan menilai-kesehatan-kasar tajuk sawit individu dari citra UAV nadir" — BUKAN "mendeteksi BSR."**

Tidak ada tanda BSR yang bisa dipastikan: basidiokarp (2–65 cm di pangkal batang) tak mungkin terlihat dari nadir pada resolusi ini; "rok" pelepah runtuh tak teramati; kelas "Unhealthy/stressed/Yellow/Dead" adalah stres generik (bisa hara, air, umur, atau penyakit apa pun).

## Gerbang 3 — Kelayakan fitur struktural

**Ukuran tajuk (diameter kotak median):**

| Kelas | A (416²) | C (640²) | B (1024², terbaik) |
|---|---|---|---|
| Healthy | 46 px | 62 px | **100 px** |
| Yellow/stressed | 37 px | 60 px | (Unhealthy) 88 px |
| Small/immature | 25 px | 31 px | — |
| Dead | 36 px | (unhealthy) 44 px | — |

1. **Resolusi cukup untuk hitung pelepah? TIDAK.** Tajuk terbaik ~100 px (B). Pada 100 px hanya arah pelepah utama yang samar terlihat; ~40+ pelepah saling tumpang-tindih jadi gumpalan. Kelas penyakit justru lebih kecil (36–88 px) — di situ pelepah tak terpisahkan sama sekali. Sawit muda 25–31 px = titik hijau. **Rencana "jumlah pelepah sebagai detektor BSR dini T1" tidak didukung data ini.** (Ditambah: literatur bilang gejala BSR dini ada di pelepah bawah yang terhalang dari nadir — jadi dua kali terhalang.)
2. **Segmentasi tajuk di dalam kotak?** Lihat `SEG_B_crownarea.jpg` (ExG+Otsu, 9 tajuk): **luas tajuk bisa** dipisahkan dari tanah (dengan pembersihan), tapi mask keluar sebagai **gumpalan padat, bukan pelepah terpisah**. Jadi luas/geometri kasar: ya. Jumlah/sudut pelepah: tidak.

---

## Petunjuk `non-bsr`
Tidak ditemukan dalam pencarian singkat (time-boxed). Satu-satunya dataset Roboflow ber-label Ganoderma yang muncul adalah **tingkat-daun** (`palm-oil-leaf-ganoderma`) — hampir pasti citra darat/dekat, modalitas berbeda dari pipeline UAV. Tidak relevan.

## Rekomendasi rinci
- **Untuk klaim BSR:** jangan pakai A/B/C. Tak satu pun ber-label BSR. Kejar data on-request (mis. UAV hiperspektral Sembawa di Frontiers, MPOB) untuk label BSR per-pohon sejati.
- **Untuk demonstrator deteksi tajuk Lapisan 1 (jika paper hanya butuh menunjukkan tahap deteksi):** pakai **B saja** — paling bersih (2.303 tile, tanpa augmentasi, 1024² resolusi tertinggi, label biner). Nyatakan jujur di paper bahwa label ini **kesehatan/tutupan-lahan umum, bukan BSR**, dan **buang komponen fitur struktural pelepah** (jumlah/sudut) karena resolusi tak mendukung.
- **WAJIB re-split per region (block split), JANGAN pakai split bawaan.** Tahan seluruh ortomosaik sebagai test (mis. latih `44000_4000`+`44000_16000`, uji `52000_20000`). Split bawaan bocor 100% dan akan melaporkan angka palsu. Dengan hanya 3 orto, ini juga berarti n=3 unit spasial — laporkan keterbatasan ini terang-terangan; validasi silang leave-one-ortho-out (3 lipatan) adalah maksimum yang jujur.
- **Ketidakseimbangan kelas ekstrem — batasan nyata.** B ≈ 68:1 (148.881 Healthy : 2.179 Unhealthy); kotak "penyakit" langka di semua (A: Dead 295, Yellow ~1.900). Demonstrator kesehatan-tajuk generik pun **terbatas oleh ketidakseimbangan**, dan justru kelas sakit — yang paling penting — adalah tempat data paling tipis.
- **Jangan** gabung A+B+C sebagai "7.500 gambar" — itu 2.745 unik, dan menggabung akan membocorkan tile identik antar train/val.
