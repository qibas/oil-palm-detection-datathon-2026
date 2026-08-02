# Jalur Akuisisi Data BSR On-Request — Lapisan 1

**Tanggal:** 2026-07-16 · **Tujuan:** menemukan data BSR sawit sungguhan (idealnya UAV, label per-pohon) yang bisa diminta, karena audit membuktikan tiga dataset Roboflow **bukan BSR** (lihat `AUDIT_REPORT.md`).

## Bottom line
- **Tidak ada dataset BSR-UAV per-pohon yang ter-deposit publik** (Zenodo/Mendeley/IEEE DataPort/Kaggle) — pencarian mengkonfirmasi temuanmu selama seminggu. Semua jalur = **permintaan ke pemilik**.
- **Kandidat terbaik untukmu: Sembawa / Universitas Indonesia** — Indonesia, label BSR per-pohon, **DAN mask segmentasi tajuk** (persis yang tak dimiliki data Roboflow), status akses **on-request terverifikasi**.
- **Peringatan kejujuran (sama seperti audit Roboflow):** kandidat terbaik pun **satu kebun, ~194 pohon, biner sehat/sakit**. Semua caveat yang kita kenakan ke data Roboflow — situs tunggal, n-kecil, autokorelasi spasial, ketidakseimbangan kelas — **berlaku juga di sini**. Data ini bisa **memvalidasi prototipe tahap seg-tajuk + per-pohon sebagai proof-of-concept satu-situs**, BUKAN bukti detektor BSR yang tergeneralisasi.

---

## Modalitas Lapisan 1: **RGB + multispektral** (DIPUTUSKAN 2026-07-16)
Implikasi:
- Dari Sembawa, yang relevan = **ortomosaik RGB + multispektral (Blok F 7,8 ha) + mask tajuk + label per-pohon**. **Kubus hyperspektral 100-band TIDAK diminta** (tak akan transfer ke inferensi RGB/multispektral — ruang input beda).
- **Peringatan jujur yang harus dikonfirmasi ke pemilik:** 720 mask tajuk & 194 pohon berlabel di makalah dianalisis pada bidang **hyperspektral**. Belum pasti berapa pohon berlabel-lapangan yang jatuh di dalam cakupan **multispektral** 7,8 ha, dan **band multispektral apa** yang direkam (mis. 5-band tipe MicaSense: B/G/R/RedEdge/NIR). Ini WAJIB ditanyakan sebelum bergantung padanya.
- Grup **UPM naik relevansinya**: kerja RGB+termal+fitur struktural mereka pas dengan modalitas ini dan dengan rencana fitur strukturalmu.

---

## Shortlist terperingkat

### ① Sembawa — Universitas Indonesia  ★ FIT TERBAIK
| | |
|---|---|
| Makalah | Frontiers in Remote Sensing, 2026 — *"Spectral detection of BSR ... Sembawa plantation"* · doi:10.3389/frsen.2026.1788857 (open access) |
| Kontak | **Assoc. Prof. Dr.Eng. Masita Dwi Mandini Manessa**, Dept. Geografi, Universitas Indonesia · **manessa@ui.ac.id** |
| Sensor | DJI Matrice 300 RTK + HAIP BlackBird V2 **hyperspektral** (100 band, 500–1000 nm, **GSD 5 cm**); Blok F: 1 ha hyperspektral + **7,8 ha multispektral**; Blok I: 0,9 ha |
| Label | **Per-pohon**, biner (simptomatik BSR vs sehat), **terverifikasi lapangan** (klorosis tajuk >30%, keruntuhan pelepah dini, kerapatan tajuk turun) |
| Segmentasi tajuk | **720 mask tajuk individu, anotasi manual, IoU antar-anotator 0,91** (625 latih / 95 uji) — ini yang tak dimiliki data Roboflow |
| Jumlah | 194 pohon dalam analisis hyperspektral (95 Blok F + 99 Blok I) |
| Lokasi | Sembawa, Banyuasin, Sumatera Selatan, Indonesia |
| **Akses** | **✅ TERVERIFIKASI** — DAS verbatim: *"The raw data supporting the conclusions of this article will be made available by the authors, without undue reservation."* = on request |
| Caveat | Satu situs; ~194–720 pohon; label biner (bukan tingkat keparahan); modalitas utama hyperspektral (mahal); tetap perlu block-split per-blok |

### ② UPM — grup Khairunniza-Bejo / Nur Azuan Husin  ★ ANGLE FITUR STRUKTURAL
| | |
|---|---|
| Makalah relevan | (a) ScienceDirect S266615432600030X, 2026 — *"UAV-based integration of RGB, thermal, and structural features ... multi-class BSR severity"* → **mengekstrak fitur STRUKTURAL tajuk tampak-atas + tingkat keparahan multi-kelas** (paling dekat dengan rencana fitur strukturalmu); (b) MDPI Remote Sensing 14/3/799, 2022 — UAV hyperspektral deteksi dini; (c) SSRN 5304086 — RGB+termal+indeks spektral |
| Kontak | **Prof. Ts. Dr. Siti Khairunniza-Bejo** & **Dr. Nur Azuan Husin**, Dept. Biological & Agricultural Engineering, Fakulti Kejuruteraan, UPM · email domain **@upm.edu.my** (alamat persis: konfirmasi dari halaman makalah — jangan tebak) |
| Sensor | UAV RGB + termal + multispektral/hyperspektral (bervariasi per makalah) |
| Label | **Per-pohon, tingkat keparahan multi-kelas** (sehat → parah), terverifikasi lapangan; 390 & 1278 pohon (per makalah) |
| **Akses** | **⚠️ BELUM TERVERIFIKASI** — halaman Elsevier/MDPI/SSRN mem-block pembacaan (403). Cek DAS langsung di halaman makalah sebelum mengirim permintaan |
| Fit | Multi-kelas keparahan + **fitur struktural tajuk** (relevan langsung), RGB+termal (modalitas lebih mudah di-deploy). Caveat: Malaysia, kemungkinan on-request, estate tunggal |

### ③ Repositori publik — TIDAK ADA yang cocok
- Tak ada dataset BSR-UAV per-pohon terverifikasi yang ter-deposit publik.
- Set biner 7.348-gambar (Polibatam JAIC, CNN ensemble) dan set Roboflow = "healthy/unhealthy" tanpa verifikasi BSR lapangan — kemungkinan **satu garis keturunan dengan data yang sudah kita audit**. **Tidak direkomendasikan** untuk klaim BSR.

---

## Yang diminta (urutan prioritas — modalitas RGB + multispektral)
1. **Anotasi segmentasi tajuk (720 poligon/mask) + label BSR per-pohon + metadata verifikasi lapangan** (klorosis / keruntuhan pelepah / kerapatan tajuk). ← paling membuka pipeline-mu; tak tergantung sensor.
2. **Ortomosaik RGB + multispektral** (Blok F 7,8 ha, Blok I) — plus **konfirmasi band multispektral** (mis. B/G/R/RedEdge/NIR) dan **GSD**.
3. **Konfirmasi:** berapa pohon berlabel-lapangan yang berada di cakupan RGB/multispektral (bukan hanya 194 pohon hyperspektral).
4. **Kubus hyperspektral: TIDAK diminta** (di luar modalitas).
- Sertakan: tujuan akademik/non-komersial, tawaran sitasi/ko-authorship, kesediaan menandatangani perjanjian penggunaan data (DUA).

## Setelah dapat data — disiplin yang sama
- **Block-split per blok/ortomosaik** (leave-one-block-out), bukan split acak per-pohon. Sembawa punya ≥2 blok (F, I) → itu batas jumlah unit spasial independen.
- Laporkan n pohon, keseimbangan kelas, dan bahwa ini **PoC satu-situs** — jangan over-claim generalisasi.
- Karena label biner: klaim maksimum jujur = "deteksi tajuk + skrining simptomatik BSR per-pohon di satu kebun", bukan tingkat keparahan atau deteksi dini pra-gejala.
