# Pencarian Dataset BSR Sawit — Sapuan Repositori 2026

**Tanggal:** 2026-08-08 · **Lingkup:** pencarian dataset baru untuk menutup *blocking gap* Lapisan 1
**Dokumen terkait:** `AUDIT_REPORT.md` (audit ds_A/B/C), `BSR_DATA_ACQUISITION.md` (jalur on-request, 2026-07-16)

> **Aturan yang dipakai di dokumen ini:** sebuah dataset hanya disebut "ADA" kalau ada **tautan unduh yang benar-benar berfungsi**. Makalah yang mendeskripsikan data tanpa merilisnya dicatat sebagai **TIDAK ADA**. Temuan negatif yang tegas lebih berguna daripada daftar tautan mati yang optimistis.

---

## 1. Bottom line

**Blocking gap TIDAK tertutup.** Per Agustus 2026, **tidak ada satu pun dataset publik** yang punya *citra* **DAN** *status penyakit BSR per-pohon terverifikasi lapangan* **DAN** *georeferensi* **DAN** *survei berulang* di situs yang sama.

Empat pernyataan negatif eksplisit:

| Prioritas | Target | Hasil |
|---|---|---|
| **A** | Citra sawit + label Ganoderma/BSR terverifikasi lapangan per-pohon | **NIHIL.** Nol dataset ter-deposit publik. Semua yang ada = *on request* ke penulis, atau makalah tanpa Data Availability Statement sama sekali. |
| **B** | Citra sawit georeferensi **DAN** multi-temporal, tingkat tajuk | **NIHIL.** Yang georeferensi (Sembawa, Ahmadi) = *single-date* dan tidak dirilis. Yang multi-temporal (peta Zenodo 10 m) = skala kebun, bukan pohon. Yang multi-temporal + per-pohon (TLS Husin 2021) **bukan citra**. |
| **C** | Dataset tajuk/kesehatan sawit dari Indonesia atau Malaysia | **SATU DITEMUKAN & TERUNDUH: MOPAD** (2 situs di Indonesia, 363.877 pohon). Label generik, **tanpa lisensi**. |
| **D** | Hyperspectral/multispectral BSR | Riset **ada dan berkualitas** (satu dengan konfirmasi lab GSM), tapi **nol data dirilis publik**. Makalahnya ada, datanya tidak. |

**Satu temuan baru yang paling penting untuk paper** (lihat §3.1): kandidat on-request terbaik kita — Sembawa/UI — ternyata hanya punya **15 pohon simptomatik dari 194**, dan **seluruh positif berada di satu blok** (Blok F terinfeksi, Blok I referensi sehat dengan **0 positif**). Artinya identitas blok memprediksi label secara sempurna. Bahkan kalau datanya datang, **block-CV yang jujur untuk kelas positif tidak mungkin dilakukan**.

---

## 2. Peringkat kegunaan

| # | Dataset | Bisa diunduh? | Label = BSR? | Asal | Verdict |
|---|---|---|---|---|---|
| 1 | **MOPAD** (Zheng et al. 2021) | ✅ Ya (Google Drive/Baidu) | ❌ Status pertumbuhan | 🇮🇩 Indonesia | **PAKAI, dengan syarat lisensi** |
| 2 | Sembawa / UI (Frontiers 2026) | ⚠️ On request | ✅ Simptomatik lapangan | 🇮🇩 Indonesia | Minta — tapi n-positif=15 & confound blok total |
| 3 | Ahmadi et al. 2022 (RS 14:1239) | ❌ On request | ✅ **+ uji lab GSM** | 🇲🇾 Malaysia | Label terkuat yang ada; data tertutup |
| 4 | Kurihara et al. 2022 (RS 14:799) | ❌ Tanpa DAS | ✅ Survei lapangan | 🇲🇾 Malaysia | Data tidak pernah dirilis |
| 5 | Yong et al. 2023 (Agriculture 13:69) | ❌ Dibatasi eksplisit | ✅ | 🇲🇾 Malaysia | Tertutup |
| 6 | TLS multi-temporal (Husin et al. 2021) | ❌ Tidak dinyatakan | ✅ | 🇲🇾 Malaysia | Satu-satunya BSR multi-temporal — tapi bukan citra |
| 7 | PALMS/PRISM (arXiv 2502.13023) | ⚠️ Tidak dikonfirmasi | ❌ Tidak ada label sehat | 🇪🇨 Ekuador | Salah spesies, salah benua |
| 8 | GeoAI Oil Palm Benchmark (arXiv 2509.08303) | ✅ CC-BY | ❌ Tutupan lahan | 🇮🇩 Indonesia | Salah skala (poligon kebun) |
| 9 | Peta sawit 10 m Zenodo | ✅ CC-BY 4.0 | ❌ | 🇮🇩🇲🇾 | Salah skala total |
| 10 | Polibatam JAIC (7.348 citra) | ❌ "not yet available" | ❓ Tidak diungkap | ❓ | Tolak |

---

## 3. Rincian per kandidat

### 3.1 Sembawa / Universitas Indonesia — **REVISI PENTING atas penilaian sebelumnya**

| | |
|---|---|
| Makalah | Frontiers in Remote Sensing, 2026 · doi:10.3389/frsen.2026.1788857 (open access) |
| Situs | Sembawa Rubber Research Institute, Sumatera Selatan, Indonesia — **2 blok** |
| Sensor | DJI Matrice 300 RTK + HAIP BlackBird V2 hyperspectral push-broom; **GSD 5 cm**; juga DJI Phantom 4 Multispectral RTK |
| Georeferensi | ✅ **Ya** — RTK + GCP, alur SfM |
| Multi-temporal | ❌ **Tidak** — akuisisi **satu tanggal** (penerbangan 10:00–14:00, langit cerah) |
| Mask tajuk | 720 mask anotasi manual (625 latih / 95 uji) |
| Jumlah label | **194 pohon** dianalisis |
| **Keseimbangan kelas** | **15 simptomatik BSR : 179 sehat (≈ 8%)** |
| **Sebaran blok** | **Blok F = blok terinfeksi (seluruh 15 positif). Blok I = referensi sehat, 0 positif.** |
| Verifikasi label | 3 indikator visual bersamaan oleh biolog kebun: klorosis tajuk >30%, keruntuhan pelepah/kematian daun tombak dini, kerapatan tajuk turun. **Tanpa konfirmasi molekuler/laboratorium.** |
| Akses | On request (tidak ada DOI repositori) |

**Verdict — turun peringkat dari penilaian 2026-07-16.** Dokumen `BSR_DATA_ACQUISITION.md` mencatat "194 pohon, biner sehat/sakit" tanpa keseimbangan kelas. Angka sebenarnya **15 positif** membuat ini lebih kecil dari yang kita kira, dan yang lebih serius:

- **Confound blok bersifat total.** Semua positif di Blok F, semua Blok I sehat. Sebuah klasifikator yang hanya belajar "ini Blok F atau Blok I" akan mendapat skor tinggi tanpa melihat penyakit sama sekali. Ini **persis pola yang sudah kita ekspos dua kali**: confound konteks pada data Peru (tajuk dihitamkan masih ROC-AUC ~0,92) dan kebocoran split acak pada ds_B.
- **Leave-one-block-out mustahil untuk kelas positif.** Hanya ada 2 blok dan satu di antaranya kosong dari positif. Tidak ada cara jujur melakukan block-CV.
- Tetap layak diminta — mask tajuk 720 poligon punya nilai sendiri, dan georeferensi RTK adalah aset. Tapi **klaim maksimum jujur turun** menjadi: demonstrasi segmentasi tajuk, bukan bukti deteksi BSR.

### 3.2 MOPAD — kandidat terunduh terbaik (Prioritas C)

| | |
|---|---|
| Makalah | Zheng et al., *ISPRS J. Photogrammetry & Remote Sensing*, 2021 — "Growing status observation for individual oil palm trees using UAV images" |
| Repo | https://github.com/rs-dl/MOPAD (terakhir di-push 2022-10-08) |
| Unduh | ✅ **Berfungsi.** Situs 2: Google Drive + Baidu Wangpan (kode `qpaw`). Situs 1: Baidu Wangpan (kode `fgfv`) |
| Format | COCO — `train2017/`, `val2017/`, `annotations/*.json` |
| Situs | **2 situs di Indonesia**, 3 citra UAV besar, total ~28,85 km² |
| Skala | **363.877 pohon sawit** (bandingkan: ds_B hanya 5.077) |
| Kelas | 5 — *healthy*, **dead**, *mismanaged*, *smallish*, **yellowish** |
| **Lisensi** | ❌ **TIDAK ADA.** GitHub API mengembalikan `license: None`, README tidak menyebut lisensi. Default hukum = **hak cipta penuh penulis**; redistribusi dan karya turunan **tidak diizinkan** tanpa izin tertulis. |
| Georeferensi | ❌ Tidak dalam rilis (tile JPG COCO, koordinat dibuang) |
| Multi-temporal | ❌ Tidak |
| Verifikasi label | ❌ **Interpretasi visual citra**, bukan survei lapangan |

**Verdict: PAKAI, dengan dua syarat keras.**

- **Nilai:** menjawab dua kelemahan ds_B sekaligus — **provenance Indonesia** (relevan setelah transfer Peru→Indonesia jatuh ke kebetulan, ROC-AUC 0,437) dan **skala** (~72× lebih banyak pohon). Kelas `dead` dan `yellowish` adalah **proxy terdekat ke gejala BSR lanjut yang tersedia publik** — pohon mati dan tajuk menguning adalah dua tanda kanonik BSR stadium akhir.
- **Syarat 1 — lisensi.** Tanpa lisensi eksplisit, kita tidak boleh mendistribusikan ulang atau menerbitkan turunan. Harus kirim email ke penulis meminta izin tertulis sebelum dipakai di deliverable publik. Ini masalah yang sama kelasnya dengan larangan tile Google Maps.
- **Syarat 2 — semantik label tetap harus jujur.** `yellowish` ≠ BSR. Menguning bisa berasal dari defisiensi hara, stres air, atau penyakit lain. Kalimat "generic health, NOT BSR" **tetap berlaku penuh**, hanya sekarang dengan proxy yang sedikit lebih dekat dan n yang jauh lebih besar. **Jangan** mengklaim MOPAD sebagai data BSR.
- **Catatan metodologis:** karena hanya ada 2 situs dan 3 citra, disiplin *leave-one-image-out* / *leave-one-site-out* dari `region()` harus dipertahankan. Split acak akan bocor persis seperti di ds_B.

### 3.3 Ahmadi et al. 2022 — label terkuat yang ditemukan, data tertutup

| | |
|---|---|
| Makalah | *Remote Sensing* 14(5):1239 · doi:10.3390/rs14051239 |
| Situs | Machap, Melaka, **Malaysia** (2,402°N, 102,327°E), estate United Malacca Berhad; sawit umur 12 tahun, 2 ha, 374 pohon sensus |
| Sensor | Hexacopter Tarot 680PRO + Canon PowerShot SX260 HS **modifikasi NIR** — 3 band: **Green, Red, NIR** |
| GSD | **0,026 m** (2,6 cm) |
| Georeferensi | ✅ 10 GCP dengan RTK-DGPS, RMSE 0,22 m |
| Multi-temporal | ❌ Satu tanggal: 31 Oktober 2014 |
| Jumlah | **451 pohon disurvei** — T1 sehat **233**, T2 dini **38**, T3 sedang **16**, T4 parah **12** |
| **Verifikasi label** | ⭐ **Terkuat dari semua kandidat**: gejala visual kanopi **+ keberadaan basidiokarp di pangkal batang + uji laboratorium GSM (*Ganoderma Selective Medium*)** pada sampel batang untuk pohon tanpa tubuh buah. Ini konfirmasi patogen sungguhan, bukan proxy visual. |
| **DAS (verbatim)** | *"The datasets used in this study are available from the corresponding author on reasonable request."* |

**Verdict: inilah label yang seharusnya kita punya, dan tidak bisa kita dapat.** Satu-satunya kandidat dengan konfirmasi *mikrobiologis*. Tapi datanya on-request, dan positif dini hanya **38 pohon**. Layak dijadikan target permintaan kedua setelah Sembawa, khususnya karena modalitas **G/R/NIR** cocok dengan keputusan modalitas RGB+multispektral kita.

### 3.4 Kurihara et al. 2022 — tanpa Data Availability Statement sama sekali

| | |
|---|---|
| Makalah | *Remote Sensing* 14(3):799 · doi:10.3390/rs14030799 |
| Situs | Segamat, Johor, **Malaysia** (2°49'24"N, 102°42'51"E), kebun FGV R&D Sdn. Bhd.; 8 ha, **1.113 pohon**, tanam 13 tahun, 139 pohon/ha |
| Sensor | Hyperspectral tunable **460–780 nm** (interval min. 1 nm, lebar band 6–23 nm), 656×494 px, FOV 90°, di atas drone DJI Agriculture Series |
| Altitude / GSD | 50–60 m → 0,12–0,14 m; 90–100 m → 0,22–0,24 m |
| Jumlah | **Hanya 96 pohon berlabel** dari 1.113 di situs: Healthy **10**, Early-stage **10**, Late-stage **8**, Dead **68**. → 303 "sampel" hasil **penghitungan ganda pohon yang sama di 73 scene berbeda** |
| Verifikasi label | Inspeksi visual lapangan oleh 8 staf berpengalaman selama 3 jam (7 Nov 2018), koordinat geografis dicatat per pohon terinfeksi |
| **DAS** | ❌ **TIDAK ADA.** Makalah hanya memuat Author Contributions, Funding, IRB Statement, Informed Consent, Acknowledgments, Conflicts of Interest. Tidak ada seksi Data Availability. |

**Verdict: TOLAK sebagai sumber data.** Dua catatan kejujuran yang relevan untuk paper kita:

1. **Judul menjanjikan "Early Detection"; basisnya 10 pohon early-stage.** Ini contoh sempurna dari klaim yang jauh melampaui daya dukung n-nya — persis jenis over-claim yang paper kita hindari.
2. **303 sampel dari 96 pohon adalah pengambilan sampel yang menggandakan pohon yang sama di scene berbeda.** Kalau split dilakukan per-sampel dan bukan per-pohon, itu **kebocoran langsung** — pohon yang sama muncul di train dan test. Makalahnya menyatakan hal ini secara terbuka ("including duplicate counting of the same tree in different scenes"), jadi ini catatan metodologis, bukan tuduhan.

### 3.5 Yong et al. 2023 — dibatasi secara eksplisit

- *Agriculture* 13(1):69 · doi:10.3390/agriculture13010069 · Khairunniza-Bejo dkk., Malaysia; hyperspectral, VGG16 + Mask R-CNN, 938 nm dilaporkan optimal.
- **DAS (verbatim):** *"The data presented in this study are available on request from the corresponding author. **The data are not publicly available due to restrictions.**"*
- **Verdict: TOLAK.** Pembatasan dinyatakan eksplisit; kecil kemungkinan permintaan dikabulkan untuk penggunaan datathon.

### 3.6 Husin et al. 2021 — satu-satunya data BSR multi-temporal, tapi bukan citra

- *Precision Agriculture* · doi:10.1007/s11119-021-09829-4 — "Multi-temporal analysis of terrestrial laser scanning data to detect basal stem rot in oil palm trees", UPM, Malaysia.
- **Multi-temporal sungguhan:** pemindaian pada bulan ke-0, ke-2, dan ke-4. Parameter terbaik: luas tajuk dan strata kanopi pada 850 cm dari puncak.
- **Verdict: relevan secara konseptual, tidak bisa dipakai.** Modalitas = **terrestrial laser scanning** (point cloud dari bawah tajuk), bukan citra UAV nadir. Tidak bisa memberi makan Lapisan 1. Ketersediaan data tidak dinyatakan. **Nilainya sebagai sitasi:** ini bukti terbit bahwa *perubahan struktur tajuk antar-waktu* membawa sinyal BSR — mendukung premis Lapisan 2 bahwa dinamika temporal itu informatif, sekaligus menjelaskan mengapa kita harus mensimulasikannya (datanya tidak ada dalam bentuk citra).

### 3.7 Kandidat yang ditolak karena salah skala atau salah domain

- **PALMS / PRISM** (arXiv 2502.13023) — 21 situs di **Ekuador**, 8.830 bbox + 5.026 titik pusat palem, **georeferensi ya**, lisensi makalah CC BY 4.0 tapi **rilis dataset tidak dikonfirmasi**. Tidak ada label kesehatan/penyakit, dan spesies palem tidak dispesifikkan. **Tolak** — pelajaran transfer Peru→Indonesia (ROC-AUC 0,437) berlaku sama untuk Ekuador.
- **Open Benchmark GeoAI Oil Palm Mapping Indonesia** (arXiv 2509.08303) — Indonesia, CC-BY, multi-temporal 2020–2024. Tapi isinya **poligon tutupan lahan tingkat kebun** dari citra satelit, bukan tajuk individual, dan **tidak ada label kesehatan/penyakit**. **Tolak — salah skala.**
- **Peta sawit 10 m Zenodo** (doi:10.5281/zenodo.17768444; zenodo.org/records/15618532) — CC-BY 4.0, Sentinel 10 m, Malaysia+Indonesia 2020–2024. Piksel 10 m ≈ satu tajuk sawit dewasa. **Tolak — salah skala total**, tidak ada informasi per-pohon.
- **Polibatam JAIC (7.348 citra, DenseNet161 91,75%)** — halaman jurnal menyatakan *"Download data is not yet available"*. Sumber gambar, definisi "unhealthy", lokasi situs, dan protokol verifikasi BSR **semuanya tidak diungkap**. **Tolak** — dan tetap curigai satu garis keturunan dengan data Roboflow yang sudah kita audit.
- **Kinetik 11(2) 2026, CNN/SCNN BSR Indonesia** — akurasi 96,48% dilaporkan, tapi halaman jurnal tidak mengungkap sumber data, jumlah citra, protokol verifikasi, maupun ketersediaan. **Tidak terverifikasi** — perlu baca PDF penuh sebelum dinilai.
- **Roboflow "Palm Oil Leaf Ganoderma"** — halaman mem-block pembacaan otomatis (403), **tidak terverifikasi**. Dari judulnya, ini kemungkinan besar **foto daun jarak dekat**, bukan citra UAV nadir — ruang input berbeda, tidak akan transfer ke pipeline tajuk kita.

---

## 4. Implikasi untuk paper

1. **Pernyataan pembatas tetap berdiri tanpa perubahan.** "Label = kesehatan tajuk generik, BUKAN BSR terverifikasi" tetap benar untuk setiap dataset yang bisa kita unduh. Sapuan ini **memperkuat**, bukan melemahkan, framing itu — sekarang kita bisa menyatakan bahwa gap tersebut sudah dicari secara sistematis di Mendeley, Zenodo, Figshare, Kaggle, IEEE DataPort, Roboflow, GitHub, dan data suplemen makalah, dan **tetap kosong**.

2. **Argumen keberadaan simulator Lapisan 2 menguat.** Alasan kita membangun simulator SEIR sintetis bukan preferensi metodologis — ini **konsekuensi langsung dari ketiadaan data**. Tidak ada citra sawit publik yang multi-temporal dan berlabel penyakit per-pohon; Husin et al. 2021 menunjukkan sinyal temporal itu nyata, tetapi hanya dalam modalitas TLS yang tidak bisa memberi makan Lapisan 1.

3. **Temuan Sembawa harus dilaporkan.** Bahkan target akuisisi terbaik kita punya 15 positif dan confound blok total. Ini pola ketiga dari jenis yang sama (Peru: confound konteks; ds_B: kebocoran split acak; Sembawa: confound blok). Layak dinyatakan sebagai pola berulang di bidang ini, bukan kecelakaan satu dataset.

4. **MOPAD adalah satu-satunya peningkatan konkret yang tersedia.** Kalau ada waktu dan izin lisensi didapat: ganti/lengkapi ds_B dengan MOPAD untuk mendapat provenance Indonesia dan skala ~364k pohon. Kalau tidak, catat sebagai *future work* dengan alasan lisensi yang dinyatakan terbuka.

## 5. Langkah lanjut yang disarankan

| Prioritas | Aksi | Alasan |
|---|---|---|
| 1 | Email penulis MOPAD (rs-dl) minta **izin lisensi tertulis** | Satu-satunya data terunduh yang relevan; tanpa lisensi tidak boleh dipakai di deliverable publik |
| 2 | Kirim permintaan Sembawa (draf ada di `draft_email_manessa_ID.md`) — **tambahkan pertanyaan sebaran positif per blok** | Konfirmasi apakah ada positif di luar Blok F; kalau tidak, nilai datanya turun drastis |
| 3 | Permintaan kedua ke Ahmadi dkk. (RS 14:1239) | Satu-satunya label dengan konfirmasi lab GSM; modalitas G/R/NIR cocok dengan modalitas kita |
| 4 | Baca PDF penuh Kinetik 11(2) 2026 | Satu-satunya kandidat Indonesia yang belum terverifikasi |
| — | **Jangan** kejar Yong et al. 2023 | Pembatasan dinyatakan eksplisit |

---

## Sumber

- [MOPAD — GitHub rs-dl/MOPAD](https://github.com/rs-dl/MOPAD) · [Zheng et al. 2021, ISPRS J.](https://www.sciencedirect.com/science/article/abs/pii/S0924271621000083)
- [Sembawa — Frontiers in Remote Sensing 2026, doi:10.3389/frsen.2026.1788857](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2026.1788857/full)
- [Ahmadi et al. 2022, Remote Sensing 14(5):1239](https://www.mdpi.com/2072-4292/14/5/1239)
- [Kurihara et al. 2022, Remote Sensing 14(3):799](https://www.mdpi.com/2072-4292/14/3/799)
- [Yong et al. 2023, Agriculture 13(1):69](https://www.mdpi.com/2077-0472/13/1/69)
- [Husin et al. 2021, Precision Agriculture](https://link.springer.com/article/10.1007/s11119-021-09829-4)
- [PALMS/PRISM, arXiv 2502.13023](https://arxiv.org/html/2502.13023) · [GeoAI Oil Palm Benchmark, arXiv 2509.08303](https://arxiv.org/html/2509.08303v1)
- [Mendeley — Oil Palm Tree Detection for Anomaly Identification](https://data.mendeley.com/datasets/nh7d23dgnw/1)
- [Polibatam JAIC](https://jurnal.polibatam.ac.id/index.php/JAIC/article/view/9437) · [Kinetik 11(2) 2026](https://kinetik.umm.ac.id/index.php/kinetik/article/view/2546)
- [Izzuddin et al., Journal of Oil Palm Research (MPOB)](https://jopr.mpob.gov.my/analysis-of-multispectral-imagery-from-unmanned-aerial-vehicle-uav-using-object-based-image-analysis-for-detection-of-ganoderma-disease-in-oil-palm/)
