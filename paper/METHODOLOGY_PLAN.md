# Rencana Metodologi — SawitGuard-GNN (Bab 3 dan turunannya)

> **Status:** rancangan naratif, bukan naskah jadi. Dokumen ini menentukan *urutan* pembaca
> bertemu setiap gagasan, *kalimat* untuk setiap pengakuan yang sulit, dan *batas* klaim.
> Naskah Bab 3 yang berlaku sekarang ada di `section3.tex`; dokumen ini adalah instruksi
> revisinya.
>
> **Aturan angka (mengikat).** Setiap angka dalam naskah harus dapat ditelusuri ke berkas di
> repositori ini — daftarnya ada di §7 (Bank Angka). Angka yang belum ada ditulis
> `[MENUNGGU: sumber]` dan **tidak boleh** ditebak. Satu angka karangan menghancurkan seluruh
> premis paper ini, karena satu-satunya modal kami adalah bahwa angka kami boleh dipercaya.
>
> **Register.** Terukur, teknis, Bahasa Indonesia; koma desimal, titik ribuan; hindari
> superlatif. Voice acuan: `section3.tex`.

---

## 0. Ringkasan untuk pembaca tergesa

Paper ini mengusulkan sistem estimasi risiko penularan *Basal Stem Rot* per pohon dari citra
UAV. Kendalanya jujur dan menentukan seluruh rancangan: **tidak ada satu pun dataset publik
yang memuat citra UAV sekaligus riwayat penularan per pohon.** Karena itu sistem dibangun
sebagai dua ujung yang masing-masing berpijak pada data nyata, ditambah satu terowongan angin
sintetis untuk ablasi yang tidak mungkin dilakukan di lapangan:

| Sumber bukti | Yang nyata di dalamnya | Yang tidak ada |
|---|---|---|
| **Lapisan 1** — citra UAV Roboflow | tajuk, posisi, geometri kebun | label BSR, dimensi waktu |
| **Lapisan 2A** — panel lapangan Eg9PP | BSR terverifikasi, 45 sensus, 25 tahun, posisi tanam | citra, kompartemen laten *E* |
| **Lapisan 2B** — simulator SEIR | spektra (penyakit lain), geometri | dinamika (placeholder, dikalibrasi ke literatur) |

Ketiganya tidak digabungkan. Yang diuji adalah **kompatibilitas antarmukanya**, dan itu diukur,
bukan diasumsikan. Di atas kerangka itu dijalankan satu dekomposisi yang memisahkan sumbangan
*waktu*, *prevalensi*, dan *struktur graf* — dan struktur graf, yang merupakan premis kami
sendiri, adalah komponen terkecil. Itu dilaporkan, bukan disembunyikan.

---

## 1. Tulang punggung naratif

### 1.1 Lima ketukan (urutan yang tidak boleh ditukar)

1. **Keputusan, bukan diagnosis.** BSR membunuh pohon yang masih tampak sehat dari udara;
   ketika gejala terlihat, penularan ke tetangga sudah berlangsung. Pertanyaan operasional
   seorang manajer kebun bukan "pohon mana yang sakit" — itu pekerjaan pengamatan — melainkan
   **"blok mana yang harus disensus lebih dulu siklus depan"**. Keluaran yang berguna karena
   itu adalah *daftar pohon berperingkat risiko*, bukan peta gejala.
2. **Kendala yang dinyatakan, bukan disiasati.** Tidak ada data publik yang memuat citra dan
   penularan sekaligus. Cara lazim menutupi kekosongan ini adalah split acak (yang membocorkan
   autokorelasi spasial) atau data sintetis tanpa label lapangan. Kami tidak menutupinya; kami
   **mengukurnya** dan membangun rancangan di sekelilingnya.
3. **Dua ujung nyata, satu terowongan angin.** Mata sistem (citra → inventaris tajuk →
   geometri kebun) dibangun di atas citra nyata. Jam sistem (dinamika penularan) dibangun di
   atas panel lapangan 25 tahun dengan Ganoderma terverifikasi. Simulator hanya dipakai untuk
   ablasi terkendali yang mustahil di lapangan — dan **tidak pernah** menjadi dasar klaim
   lapangan.
4. **Yang tidak bisa disambung tetap bisa diukur.** Uji "lebar sepur": derajat rata-rata graf
   kontak keluaran Lapisan 1 dan graf masukan Lapisan 2 dibandingkan pada dataran radius yang
   sama — **5,62 versus 5,74, selisih 2 persen**. Kami tidak dapat menggabungkan kedua dataset;
   kami dapat menunjukkan dengan angka bahwa sambungannya tidak akan meleset.
5. **Sistem yang menguji premisnya sendiri.** Dekomposisi memecah keunggulan model graf menjadi
   temporal, prevalensi, dan struktur. Struktur — satu-satunya komponen yang membenarkan
   pendekatan berbasis graf — hanya menyumbang **+0,012 ± 0,012** pada simulator. Dilaporkan
   apa adanya. Justru karena komponen itu dilaporkan kecil, angka-angka lain dalam paper ini
   layak dipercaya.

### 1.2 Kalimat yang harus diulang juri

Versi utama (untuk ditulis nyaris apa adanya di penutup Bab 3):

> **"SawitGuard-GNN dirancang untuk menjatuhkan premisnya sendiri: setiap tahapnya memiliki
> satu uji yang, bila gagal, membatalkan tahap itu — dan satu di antaranya memang hampir
> gagal, lalu dilaporkan."**

Versi pendek yang realistis diulang juri ke juri (dua kalimat, penuh angka konkret):

> *"Tim itu membuktikan 151.060 anotasi mereka sebenarnya cuma 5.077 pohon, lalu menarik salah
> satu temuan mereka sendiri karenanya. Dan mereka tetap melaporkan bahwa peta kontak — inti
> metode mereka — hanya menambah 0,01."*

Ujilah setiap paragraf Bab 3 terhadap kalimat itu: bila sebuah paragraf tidak memperkuat salah
satu dari dua kalimat di atas, ia terlalu panjang.

### 1.3 Mekanika persuasi yang dipinjam dari paper acuan — dan yang sengaja tidak

Paper acuan (`Laporan_blekping`, sistem AI peternakan ayam) menang pada mekanika, bukan pada
kedalaman metode. Yang **dipinjam**:

| Mekanika acuan | Adaptasi kami |
|---|---|
| Angka kerugian rupiah di paragraf pembuka | Angka kerugian BSR nasional di Bab 1 — `[MENUNGGU: sitasi kerugian ekonomi BSR, mis. Ditjenbun/MPOB]`. **Jangan tulis sebelum sitasi ada.** |
| Tema lomba dikutip di Abstrak **dan** Kesimpulan | Kutip tema resmi di kedua tempat — `[MENUNGGU: tema resmi Datathon 2026 Ristek CSUI]` |
| "Di sisi bisnis, ini berarti…" setelah tiap hasil teknis | Pola sama, tetapi **terikat bukti** (§4.6). Bila dampaknya tidak diukur, kalimatnya menyebutkan bahwa ia tidak diukur. |
| Saran bernomor sebagai peta jalan | Saran bernomor 1–5, tiap butir menutup satu batas yang dinyatakan di Bab 3 (§4.7) |
| Diagram arsitektur di awal Bab 3 | Gambar 1, direvisi (§3, G1) |

Yang **sengaja tidak dipinjam**, dan mengapa — ini juga menjadi pembeda diam-diam kami:

- **mAP50 0,976 dari 24 gambar latih.** Acuan melaporkannya sebagai keberhasilan dan menyebut
  keterbatasan ukuran dataset baru di Bab 5. Angka setinggi itu pada 21 gambar latih dan 3
  gambar validasi adalah tanda kebocoran atau *overfit*, bukan tanda mutu — dan juri yang tajam
  akan melihatnya. Kami melakukan kebalikannya: **ukuran sampel efektif dinyatakan sebelum
  metrik**, di subbab evaluasi, bukan sesudah.
- **"akurasi hingga 100%"** dari literatur dikutip tanpa syarat. Kami tidak mengutip angka
  performa pihak lain sebagai bukti kelayakan rancangan kami.
- **Klaim real-time / Edge AI tanpa pengukuran latensi pada perangkat sasaran.** Kami tidak
  mengklaim penempatan (*deployment*) apa pun yang tidak diuji.

---

## 2. Kerangka bab

Bab 3 memuat **rancangan saja**: tidak ada satu pun metrik performa model (konvensi yang sudah
dipakai `section3.tex` dan paper acuan). Angka yang muncul di Bab 3 seluruhnya adalah
**deskripsi data** — jumlah, jarak, derajat, ukuran sampel, tingkat kejadian.

Target panjang total: **4–5 halaman** (paper acuan: Bab 3 ≈ 4 halaman).

| # | Subbab | Harus menegakkan | Panjang | Gambar/Tabel |
|---|---|---|---|---|
| — | *Pembuka Bab 3* | Rumusan pertanyaan (ke mana penyakit bergerak, bukan pohon mana yang sakit) dan bentuk keluaran (daftar berperingkat). Pertahankan dua paragraf pembuka `section3.tex` yang ada — voice-nya sudah tepat. | 2 par. | — |
| 3.1 | **Kerangka Bukti: Tiga Sumber, Satu Pertanyaan** | **Subbab baru, paling menentukan.** Nyatakan di muka bahwa tidak ada dataset tunggal yang memuat citra + penularan; perkenalkan tiga sumber dan apa yang nyata di masing-masing; nyatakan bahwa ketiganya **tidak digabungkan**. Ini memindahkan pengakuan terbesar ke posisi paling awal, tempat ia terbaca sebagai rancangan, bukan permintaan maaf. | 3–4 par. | **Gambar 1** |
| 3.2 | **Lapisan 1 · Tahap 1–3 — Dari Citra UAV ke Inventaris Tajuk** | Deteksi tajuk (YOLO11) → luas tajuk (ExG+Otsu, tanpa klaim IoU) → penilaian kesehatan (LightGBM, fitur warna beralasan agronomis). Pertahankan hampir seluruh teks `section3.tex` §Tahap 1–3; tambahkan satu kalimat bahwa label di sini **kesehatan tajuk generik, bukan BSR**. | 4 par. | — |
| 3.3 | **Lapisan 1 · Tahap 4 — Rekonstruksi Geometri dan Audit Duplikasi** | Tanda tangan metodologis paper ini. Nama berkas ubin menyimpan offset absolut → koordinat global per pohon → awan titik kebun. Lalu **temuan forensiknya**: 151.060 kotak = 5.077 pohon (29,8×), positif nyata 66. Sertakan bukti bahwa deduplikasi tidak ambigu (0 konflik label, 0 tetangga-hantu, 5.048/5.077 tampilan kanonik) — tanpa itu, temuan ini hanya klaim. Tutup dengan kalibrasi skala 8,5–8,9 cm/px yang konsisten pada tiga ortomosaik. | 5 par. | **Gambar 2** |
| 3.4 | **Lapisan 2A — Panel Epidemi Lapangan 25 Tahun** | Eg9PP (Tisné dkk. 2017; SOCFINDO, Medan; CC BY-SA 4.0, sitasi wajib). 1.200 sawit, 45 sensus, 0,5–25,5 tahun; **BSR terverifikasi lapangan**. Empat disiplin yang membuatnya sah: koreksi geometri cos 30°, grid sensus yang benar-benar teramati (bukan sintetis), penanganan sensor empat status (`C` keluar dari risk set), dan fold = parcel dengan 0 sisi lintas-parcel. Nyatakan dua batas: **tidak ada citra**, dan **kompartemen E tidak teramati** ⇒ kepala SEIR turun menjadi SI(D) dan tidak dapat dibandingkan langsung dengan varian simulator. | 5 par. | **Gambar 5** (bila muat) |
| 3.5 | **Lapisan 2B — Simulator sebagai Terowongan Angin Terkendali** | Untuk apa simulator masih ada setelah ada data nyata: hanya di simulator peta kontak dapat **diganti** (asli/acak/nihil/perturbasi) sambil menahan segalanya tetap — eksperimen yang mustahil di lapangan. Nyatakan bahwa laju epidemi adalah *placeholder* yang wajib dikalibrasi, dan fitur node berasal dari spektra penyakit layu pinus (nyata, tetapi patogen dan spesies lain), dengan *E* sebagai interpolasi α = 0,1. Satu kalimat kunci: **simulator tidak pernah menjadi dasar klaim lapangan.** | 4 par. | — |
| 3.6 | **Uji Antarmuka "Lebar Sepur"** | Mengapa penggabungan ditolak (kebun berbeda, zaman berbeda, tanpa kunci join, tanpa georeferensi) dan apa yang menggantikannya: perbandingan kurva derajat pada dataran r = 1,25–1,5 × jarak tanam → 5,74 vs 5,62 (2%). Nyatakan bahwa angka itu **batas atas** karena memakai kotak kebenaran-dasar, bukan prediksi detektor. | 3 par. | **Gambar 3** |
| 3.7 | **Model Peramalan dan Kalibrasi Risiko** | Rumusan tugas peringatan dini (pohon asimptomatik pada *t* → bergejala dalam *h*; pohon yang sudah bergejala dikeluarkan). Tiga model diadu pada tugas identik: MLP (garis dasar kuat, disetel dengan baik), STGNN, STGNN-SEIR. Pengaman: tidak ada model yang melihat status sebenarnya sebagai masukan; genotipe wajib menjadi kovariat di **semua** model pada data Eg9PP. Kalibrasi + precision@k **hanya ditulis bila Bab 4 memuat hasilnya** — lihat §9. | 4 par. | — |
| 3.8 | **Protokol Evaluasi Anti-Kebocoran** | Ditetapkan sebelum hasil dilihat. Lapisan 1: hanya 3 ortomosaik, satu pohon muncul di median 32 ubin ⇒ split acak **pasti** bocor ⇒ *leave-one-ortho-out*. Lapisan 2A: fold = parcel, terverifikasi 0 sisi terputus. Lapisan 2B: 20 seed, berpasangan per-seed, sign-count. Aturan keputusan di muka: **\|rerata\| < simpangan baku ⇒ tidak konklusif**. Metrik: mAP, PR-AUC, ROC-AUC — akurasi tidak dilaporkan, dengan alasannya. Di sini pula ukuran sampel efektif (66 positif) dinyatakan. | 5 par. | — |
| 3.9 | **Rancangan Ablasi — Apa yang Sebenarnya Menentukan Risiko** | Selisih STGNN−MLP tidak dapat ditafsirkan karena mencampur tiga efek. Definisikan temporal / prevalensi / struktur sebagai selisih aditif hasil penukaran *tampilan graf*. Nyatakan bahwa dekomposisi yang sama dijalankan pada **kedua** sumber Lapisan 2 (simulator dan Eg9PP), sehingga temuan simulator dapat dikonfirmasi atau dibantah oleh data lapangan. Tegaskan: komponen struktur adalah yang mempertaruhkan premis paper ini, dan hasilnya dilaporkan ke arah mana pun ia jatuh. | 4 par. | (payoff: **Gambar 4** di Bab 4) |
| 3.10 | **Ringkasan Data dan Batas yang Dipaksakan** | Tabel sumber data: berkas, jumlah, label, status nyata/sintetis, dan **satu kolom "klaim maksimum"** per baris. Paragraf penutup: segala yang menyangkut *ruang* bersumber dari data nyata; yang menyangkut *waktu* nyata pada Eg9PP dan disimulasikan pada simulator. | 1 tabel + 2 par. | **Tabel 1** |

**Hook ke bab lain** (di luar cakupan dokumen ini tetapi harus direncanakan sekarang):

- **Bab 1** — angka kerugian BSR `[MENUNGGU]`, tema lomba `[MENUNGGU]`, dan satu paragraf
  "mengapa deteksi saja tidak cukup" yang menyiapkan rumusan peramalan.
- **Bab 4** — memuat Gambar 4 dan seluruh metrik. Setiap janji Bab 3 harus punya pasangannya
  di sini (§9).
- **Bab 5** — pola "Di sisi operasional, ini berarti…" (§4.6).
- **Bab 6** — Saran bernomor (§4.7) + kutipan tema lomba.

---

## 3. Spesifikasi gambar

Lima gambar. Empat wajib (G1–G4), satu opsional (G5). Palet dan aturan visual mengikuti
`fig_pipeline.py` yang sudah ada: hijau `#008300` = data nyata, biru `#2a78d6` = sintetis,
abu `#52514e` = protokol/evaluasi. Konsistensi warna ini sendiri adalah alat retoris — juri
dapat menghitung berapa banyak gambar yang hijau.

---

### G1 — Arsitektur bukti tiga-sumber *(revisi `fig_pipeline.py`; Bab 3, §3.1)*

**Berkas:** `fig_pipeline.py` → `fig_pipeline.png` **sudah ada, tetapi kedaluwarsa.** Versi
sekarang menggambar dua lajur (Lapisan 1 dan simulator) dengan jembatan putus-putus berlabel
*"belum tersambung"*. Itu tidak lagi benar: Lapisan 2 kini memiliki sumber lapangan nyata, dan
jembatannya bukan "belum ada" melainkan **"diukur dan tidak digabung"**.

**Yang harus terbaca:** tiga lajur, bukan dua.

- Lajur A (hijau) — **Lapisan 1**: citra UAV → deteksi tajuk → kesehatan → luas tajuk →
  **Tahap 4: rekonstruksi geometri**. Strip evaluasi: *leave-one-ortho-out*.
- Lajur B (hijau) — **Lapisan 2A · Eg9PP**: 1.200 sawit, 45 sensus, 25 tahun, BSR
  terverifikasi → graf kontak akar → tugas peramalan. Strip evaluasi: *leave-one-parcel-out*.
- Lajur C (biru) — **Lapisan 2B · simulator**: config (placeholder) → SEIR berjaring → fitur
  spektral → penukaran tampilan graf. Strip: 20 seed, dekomposisi.
- **Konektor A→B**: garis **utuh tipis berlabel angka**, bukan putus-putus —
  "uji antarmuka: derajat 5,62 vs 5,74 (Δ 2%)". Konektor C→B: panah dua arah berlabel
  "dekomposisi yang sama dijalankan di kedua sisi".
- Pita bawah selebar gambar: **"tidak ada satu dataset pun yang memuat citra dan penularan
  sekaligus — ketiganya tidak digabungkan"**.

**Mengapa meyakinkan:** juri memperoleh seluruh arsitektur kejujuran dalam lima detik, dan
melihat bahwa dua dari tiga lajur berdiri di atas data nyata. Konektor berangka mengubah
kelemahan ("tidak tersambung") menjadi pengukuran.

**Data:** skematik, tidak membaca hasil. Angka pada label berasal dari `DATASET_CARD.md`.

**Tata letak:** lanskap ≈ 7,1 × 5,2 inci, tiga baris lajur + dua strip protokol, 400 dpi.

---

### G2 — "151.060 → 5.077": forensik duplikasi *(Bab 3, §3.3)*

**Dua panel bersebelahan, satu ortomosaik (`44000_4000`, 1.849 pohon, 31 Unhealthy).**

- **Panel kiri — mengapa duplikasi terjadi.** Kanvas ortomosaik (~5.000 × 5.000 px) dengan
  jejak 767 ubin 1.024 px digambar sebagai kotak transparan bertumpuk; 14 ubin yang benar-benar
  saling lepas ditandai garis penuh. Anotasi: *"767 ubin di atas kanvas yang hanya memuat ~25;
  hanya 14 yang tidak bertindih"*.
- **Panel kanan — akibatnya.** Awan titik 1.849 pohon unik hasil rekonstruksi; 31 pohon
  Unhealthy ditandai. Anotasi: *"satu pohon fisik muncul di median 32 ubin (rentang 1–77)"*.

**Mengapa meyakinkan:** ini satu-satunya gambar dalam paper yang menunjukkan tim menemukan
sesuatu yang tidak dicari orang lain, dan ia sekaligus menampilkan kelangkaan kelas positif
tanpa satu kalimat pun pembelaan. Panel kanan juga secara diam-diam memperlihatkan bahwa
geometri kebun yang direkonstruksi memang tampak seperti kebun — pembuktian visual untuk
Tahap 4.

**Data (semua ada):** `layer1_build/out/crowns_B_44000_4000.npy` (awan titik),
`data_clean/layer1_crowns.csv` (pohon unik + label), `data_clean/layer1_tiles_disjoint.csv`
(ubin saling-lepas), offset ubin dari nama berkas COCO `ds_B`.

**Tata letak:** dua panel bujur sangkar berdampingan, ≈ 7,1 × 3,6 inci; sumbu dalam meter
memakai skala 8,7 cm/px.

---

### G3 — Uji lebar sepur: kurva derajat *(Bab 3, §3.6)*

Satu panel. Sumbu-x: radius dalam kelipatan jarak tanam (0,8 → 2,2). Sumbu-y: derajat
rata-rata. Dua kurva: **Eg9PP (posisi tanam)** dan **Roboflow (centroid tajuk hasil Tahap 4)**.
Pita vertikal menyorot dataran pembanding **r = 1,25–1,5**, dengan dua penanda titik berlabel
**5,74** dan **5,62** serta anotasi **"Δ 2% — keduanya kisi segitiga berderajat 6"**.

Dua anotasi kecil yang wajib ada, sebab tanpanya gambar ini menipu:

- di luar dataran, kurva Eg9PP **melompat bertangga** (kisi ideal) sementara Roboflow
  **melandai** (posisi nyata berderau) → *"artefak presisi, bukan geometri berbeda"*;
- catatan kaki: *"dihitung dari kotak kebenaran-dasar, bukan prediksi detektor ⇒ batas atas."*

**Mengapa meyakinkan:** mengubah kalimat "kami tidak bisa menggabungkan data" menjadi hasil
terukur dengan pita ketidakpastian yang jujur. Tidak ada tim lain yang akan melaporkan
kegagalan integrasi sebagai eksperimen.

**Data (ada):** `data_clean/layer2_nodes.csv` (posisi Eg9PP setelah koreksi cos 30°),
`data_clean/layer1_crowns.csv` (centroid tajuk unik). **Skrip pembangkitnya belum ada** —
`[MENUNGGU: skrip figur, mis. data_clean/fig_interface.py]`.

**Tata letak:** satu panel ≈ 5,0 × 3,4 inci.

---

### G4 — Dekomposisi keunggulan: simulator vs lapangan *(payoff di Bab 4, dijanjikan di §3.9)*

Dua kelompok batang bertumpuk berdampingan.

- **Kiri — simulator (20 seed, h = 3):** temporal **+0,033** (19/20) · prevalensi **+0,077**
  (20/20) · **struktur +0,012 ± 0,012** (20/20) · total STGNN−MLP **+0,123**. Galat = simpangan
  baku berpasangan; sign-count dicetak di dalam/di atas tiap batang.
- **Kanan — Eg9PP lapangan (h = `[MENUNGGU]`):** tiga komponen yang sama —
  `[MENUNGGU: layer2_real/results_real.csv, dihasilkan Agen 1 + Agen 3]`.
- **Pita "band derau" horizontal** membentang di kedua kelompok pada tinggi ± 1 simpangan
  baku berpasangan; batang yang puncaknya berada di dalam pita diberi label **TIDAK KONKLUSIF**.
  Batang struktur harus terlihat hampir menyentuh pita — itulah isi ilmiah gambar ini.

**Mengapa meyakinkan:** gambar ini adalah paper dalam satu bidang gambar. Ia menunjukkan bahwa
klaim besar (model graf menang) dan klaim kecil (peta kontak yang benar hampir tidak berarti)
dilaporkan dengan skala yang sama, dan bahwa temuan simulator diuji ulang pada data lapangan
nyata alih-alih dibiarkan sebagai artefak simulasi.

**Data:** `results.csv` baris `DECOMP` (sudah ada) + hasil `layer2_real` `[MENUNGGU]`.
Bila hasil lapangan belum siap saat tenggat, **gambar tetap dibuat dengan satu kelompok saja**
dan kelompok kanan diganti kotak berlabel *"replikasi lapangan — pekerjaan berjalan"*;
**jangan** mengisi batangnya dengan tebakan.

**Tata letak:** satu panel ≈ 6,0 × 3,6 inci, sumbu-y ΔAUC-PR, garis nol tebal.

---

### G5 *(opsional, bila halaman cukup)* — Disiplin sensor dan pembagian fold Eg9PP *(Bab 3, §3.4)*

Dua panel.

- **Kiri:** peta 1.200 pohon pada koordinat terkoreksi, diwarnai per parcel (44A: Y 3,0–14,5;
  44B: Y 22,0–33,5), dengan sisi graf kontak digambar tipis. Anotasi: **"0 dari 3.354 sisi
  melintasi parcel ⇒ memisahkan fold tidak memutus satu sisi pun"** dan **"14 dari 14 famili
  hadir di kedua parcel ⇒ fold tidak terkonfound genotipe"**.
- **Kanan:** diagram pita status sepanjang 45 sensus — proporsi pohon berstatus `A` / `S` /
  `D` / `C` per tanggal sensus. Anotasi: **"498 pohon tersensor, tersedini t = 6,0 tahun —
  dikeluarkan dari risk set, tidak dianggap sehat"**.

**Mengapa meyakinkan:** memperlihatkan disiplin yang hampir tak pernah divisualkan tim lain —
bahwa data yang hilang diperlakukan sebagai hilang. Panel kiri sekaligus membuktikan klaim
anti-kebocoran secara visual alih-alih dengan janji.

**Data (ada):** `data_clean/layer2_nodes.csv`, `layer2_panel.csv`, `layer2_edges.csv`.

---

## 4. Playbook kejujuran-sebagai-kekuatan

### 4.0 Pola tiga langkah (berlaku untuk semua pengakuan)

Setiap pengakuan ditulis dalam tiga langkah, dalam urutan ini. Menukar urutannya mengubah
rigor menjadi permintaan maaf.

1. **Pengukuran yang menghasilkannya** — apa yang dilakukan, bukan apa yang dirasakan.
2. **Aturan yang kami paksakan pada diri sendiri sebagai akibatnya** — pengakuan menjadi
   *keputusan protokol*, bukan keluhan.
3. **Apa yang diperoleh pembaca darinya** — mengapa ia sekarang bisa membaca angka kami.

Dan satu aturan penempatan: **pengakuan selalu mendahului metrik yang dipengaruhinya.** Ukuran
sampel efektif ditulis di subbab evaluasi (§3.8), sebelum satu pun PR-AUC muncul di Bab 4.

### 4.1 Enam puluh enam positif

> Angka pertama yang perlu diketahui pembaca bukanlah hasil model, melainkan ukuran sampel yang
> menghasilkannya. Rekonstruksi Tahap~4 menunjukkan bahwa 151.060 kotak anotasi pada data ini
> sesungguhnya hanya 5.077 pohon fisik — satu pohon yang sama tercatat pada median 32 ubin yang
> berbeda — dan di antara kelima ribu pohon itu hanya **66 pohon** yang berlabel tidak sehat:
> 17, 31, dan 18 pohon pada masing-masing ortomosaik. Konsekuensinya kami tetapkan sebagai
> aturan, bukan sebagai catatan kaki: setiap selang kepercayaan dihitung pada tingkat pohon
> unik dan bukan pada tingkat anotasi, dan setiap perbedaan yang lebih kecil daripada simpangan
> bakunya dinyatakan tidak konklusif alih-alih dibulatkan menjadi kemenangan. Kami menyatakan
> ini di muka karena angka 151.060 akan terbaca meyakinkan padahal ia menghitung salinan, bukan
> pohon — dan pembaca berhak mengetahui bahwa seluruh Lapisan~1 bersandar pada 66 pengamatan.

*Penguat opsional satu kalimat, bila diperlukan:* "Dengan basis itu, ketimpangan kelasnya
adalah sekitar 76 banding 1 (5.011 sehat : 66 tidak sehat), bukan 69 banding 1 seperti yang
tampak pada tingkat anotasi."

### 4.2 Label bukan BSR

> Label pada citra UAV yang tersedia adalah **kesehatan tajuk generik** — gabungan stres hara,
> kekurangan air, umur tanaman, dan penyakit apa pun — bukan *Ganoderma* yang terverifikasi di
> lapangan. Audit tingkat piksel yang kami lakukan tidak menemukan satu pun basidiokarp maupun
> "rok" pelepah runtuh yang dapat dipastikan, dan itu memang yang diperkirakan: basidiokarp
> tumbuh di pangkal batang dan tidak mungkin terlihat dari sudut nadir pada resolusi ini,
> sementara gejala BSR paling awal muncul pada pelepah bawah yang terhalang tajuk sendiri.
> Karena itu Lapisan~1 kami sebut demonstrator **tahap deteksi dan penilaian kesehatan tajuk**,
> bukan detektor BSR, dan setiap klaim yang menyangkut *Ganoderma* dalam paper ini bersandar
> pada panel lapangan Lapisan~2, bukan pada citra. Yang dibuktikan Lapisan~1 adalah bahwa
> **antarmukanya** dapat dibangun dari citra drone biasa — dan itulah tepatnya yang dibutuhkan
> ketika label BSR per pohon kelak tersedia.

### 4.3 Klaim yang kami tarik

> Satu temuan dalam versi awal pekerjaan ini kami tarik, dan kami mencantumkannya di sini alih-alih
> menghapusnya diam-diam. Kami sempat melaporkan bahwa salah satu ortomosaik "kolaps" ke PR-AUC
> 0,030 dan menafsirkannya sebagai bukti variasi antar-situs yang besar. Setelah basis analisis
> diperbaiki dari anotasi ke pohon unik, ortomosaik yang sama mencapai 0,126 dan ketiga lipatan
> berada pada rentang 0,13–0,26: yang kami kira efek situs ternyata efek pembobotan akibat
> duplikasi ubin. Koreksi yang sama **menaikkan** PR-AUC utama kami dari 0,126 menjadi 0,182
> sekaligus **membatalkan** temuan yang sebelumnya kami anggap menarik — bergerak ke dua arah
> berlawanan sekaligus, yang merupakan tanda bahwa yang berubah adalah prosedurnya dan bukan
> pilihan angkanya. Variasi antar-situs tetap ada, tetapi jauh lebih kecil daripada yang pernah
> kami laporkan, dan klaim kolaps itu tidak lagi berlaku.

### 4.4 Struktur graf hanya menambah +0,01

> Premis setiap pendekatan berbasis graf adalah bahwa **peta kontak yang benar itu penting**.
> Premis itu tidak kami asumsikan, melainkan kami uji, dengan cara mengganti peta kontak yang
> benar dengan graf acak berderajat sama sambil menahan arsitektur model tetap. Hasilnya kami
> laporkan apa adanya: pada simulator, graf acak sudah memulihkan sebagian besar keunggulan,
> dan mengetahui **siapa persis** tetangga sebuah pohon hanya menambah **+0,012 ± 0,012**
> AUC-PR — konsisten positif pada 20 dari 20 *seed*, tetapi dengan rerata sebesar simpangan
> bakunya sendiri. Kami tidak membulatkannya menjadi kemenangan. Keunggulan model graf pada
> eksperimen ini terutama berasal dari pemodelan waktu dan dari mengetahui **seberapa banyak**
> penyakit ada di sekitar, bukan dari mengetahui di mana. Dekomposisi ini kami rancang justru
> agar kesimpulan semacam itu dapat muncul; melaporkannya bukan kelemahan paper ini melainkan
> alasan dekomposisi itu dibuat.

*Kalimat penutup yang mengubahnya menjadi rekomendasi, bukan kekalahan:*

> Temuan ini memiliki konsekuensi anggaran yang langsung: bila peta kontak presisi tinggi hanya
> bernilai +0,01, maka rupiah berikutnya lebih baik dibelanjakan untuk **menambah frekuensi
> survei dan cakupan label**, bukan untuk meningkatkan ketelitian pemetaan posisi pohon.

### 4.5 Kedua dataset tidak dapat digabungkan

> Kedua sumber data ini **tidak** kami gabungkan, dan itu keputusan yang disengaja: kebunnya
> berbeda, zamannya berbeda — petak Eg9PP dibongkar pada 2012 sementara citra DJI berasal dari
> pasca-2013 — tidak ada kunci join, dan tak satu pun bersistem koordinat bergeoreferensi.
> Menggabungkannya akan menghasilkan tabel yang tampak lengkap dan tidak berarti apa pun.
> Sebagaimana dua ruas jalur kereta yang dibangun terpisah hanya dapat disambung apabila lebar
> sepurnya sama, yang dapat diuji di sini bukanlah penggabungan melainkan **kompatibilitas
> antarmuka**: apakah graf yang dihasilkan Lapisan~1 sebangun dengan graf yang dikonsumsi
> Lapisan~2. Pada dataran radius 1,25–1,5 kali jarak tanam, derajat rata-rata keduanya adalah
> **5,74** dan **5,62** — berselisih 2 persen, keduanya kisi segitiga berderajat enam. Angka itu
> adalah **batas atas**, sebab dihitung dari kotak kebenaran-dasar dan bukan dari prediksi
> detektor. Kami tidak dapat menyatukan kedua dataset; kami dapat menunjukkan, dengan angka,
> bahwa sambungannya tidak akan meleset.

### 4.6 Pola "Di sisi operasional, ini berarti…" (Bab 5)

Pola acuan dipakai, tetapi dengan satu syarat tambahan: **bila dampaknya tidak diukur, kalimat
itu sendiri yang menyatakannya.** Empat contoh siap pakai:

- **Setelah hasil deteksi:** *"Di sisi operasional, keluaran tahap ini adalah daftar koordinat
  per pohon untuk satu blok utuh — persis bentuk data yang kini dikumpulkan tim sensus secara
  manual. Kami tidak mengukur penghematan biayanya dan karena itu tidak mengklaimnya; yang kami
  tunjukkan adalah bahwa bentuk keluarannya sudah cocok dengan bentuk masukan yang dipakai
  keputusan lapangan."*
- **Setelah PR-AUC kesehatan:** *"Base rate kelas tidak sehat pada data ini adalah 1,30 persen,
  sehingga pemeringkatan acak menghasilkan PR-AUC 0,0130. Peringkat yang dihasilkan model
  mencapai 0,182, yaitu sekitar empat belas kali nilai acak tersebut.
  Di sisi operasional itu berarti kapasitas inspeksi yang terbatas dapat diarahkan; namun
  angka itu bersandar pada 66 pohon positif, sehingga ia menunjukkan kelayakan pendekatan,
  bukan tingkat layanan yang dapat dijanjikan."*
- **Setelah dekomposisi:** *"Di sisi operasional, temuan ini adalah rekomendasi belanja: peta
  kontak yang lebih presisi bernilai +0,01, sedangkan sekadar mengetahui seberapa banyak
  penyakit ada di sekitar bernilai +0,077. Prioritas investasi karena itu adalah kerapatan dan
  frekuensi pengamatan, bukan ketelitian pemetaan."*
- **Setelah lead-time:** *"Pada simulator, peringatan muncul sekitar 2,6–2,9 siklus survei —
  kira-kira delapan bulan — sebelum gejala tampak, tetapi hanya untuk sekitar
  sepertiga pohon yang kelak tertular, dan model graf tidak lebih baik daripada garis dasar
  per-pohon pada kedua ukuran itu. Angka ini berasal dari dinamika sintetis dan tidak boleh
  dibaca sebagai janji lapangan."*

### 4.7 Saran bernomor (Bab 6) — tiap butir menutup satu batas Bab 3

1. **Label BSR per pohon berlabel waktu.** Menutup batas §4.2. Sasaran: kerja sama akuisisi
   citra UAV berpasangan dengan sensus Ganoderma darat pada kebun yang sama —
   `[MENUNGGU: mitra/lokasi yang benar-benar dihubungi; jangan sebut nama tanpa konfirmasi]`.
2. **Menaikkan daya statistik dari 66 positif.** Menutup batas §4.1. Sasaran konkret: jumlah
   pohon positif berlabel yang membuat selisih 0,05 PR-AUC dapat dibedakan dari derau
   `[MENUNGGU: perhitungan daya]`.
3. **Kalibrasi laju epidemi ke literatur *Ganoderma*.** Menutup batas simulator (§3.5):
   mengganti blok `PLACEHOLDER` dengan laju terbitan berdampingan dengan laju yang diestimasi
   dari panel Eg9PP.
4. **Replikasi dekomposisi lintas-kebun.** Menutup batas satu-situs: Eg9PP hanya dua parcel di
   satu kebun; kesimpulan struktur baru dapat digeneralisasi bila bertahan pada kebun kedua.
5. **Validasi antarmuka ujung-ke-ujung.** Menutup batas §4.5: mengulang uji lebar sepur memakai
   **prediksi detektor**, bukan kotak kebenaran-dasar, sehingga angka 2 persen berubah dari
   batas atas menjadi estimasi operasional.

---

## 5. Q&A anti-rapuh

Delapan pertanyaan tersulit yang dapat diajukan juri yang bermusuhan, jawaban jujurnya, dan
tempat jawaban itu **sudah didahulukan** di dalam naskah. Bila sebuah jawaban belum
didahulukan di mana pun, itu adalah lubang yang harus ditambal sebelum tenggat.

---

**Q1. "Data citra kalian bukan BSR. Jadi apa sebenarnya yang kalian buktikan?"**

**Jawab:** Bahwa antarmukanya dapat dibangun. Lapisan 1 membuktikan tiga hal yang tidak
memerlukan label BSR: tajuk sawit individual dapat dideteksi dari citra UAV RGB biasa; posisi
global setiap pohon dapat direkonstruksi sehingga geometri kebun terukur, bukan diasumsikan;
dan gradasi kesehatan tajuk memang membawa sinyal warna yang dapat dipelajari. Klaim yang
menyangkut *Ganoderma* seluruhnya bersandar pada panel lapangan Eg9PP, tempat gejala BSR
terverifikasi per pohon selama 25 tahun. Kami memisahkan keduanya secara eksplisit alih-alih
menggabungkannya menjadi klaim tunggal yang tidak didukung.
**Didahulukan di:** §3.1 (kerangka bukti), §3.2 kalimat penutup, §4.2, Tabel 1 kolom "klaim
maksimum", Gambar 1 pita bawah.

---

**Q2. "Kalau peta kontak hanya menambah 0,01, bukankah paper kalian menggugurkan dirinya
sendiri?"**

**Jawab:** Yang gugur adalah versi kuat dari premis, bukan pekerjaannya — dan cara ia gugur
adalah hasil. Tanpa dekomposisi, selisih STGNN−MLP sebesar +0,123 akan dilaporkan sebagai bukti
bahwa "peta kontak penting", dan itu keliru: +0,077 di antaranya diperoleh graf **acak**. Paper
yang menjalankan ablasi yang sama pada data lain akan menghadapi konfound yang sama, sehingga
temuan ini berlaku melampaui kasus kami. Yang kami usulkan tetap berdiri pada komponen yang
terbukti besar — pemodelan waktu dan tekanan wabah agregat — dan komponen struktur dilaporkan
kecil, positif konsisten (20/20 *seed*), dan pada batas band derau.
**Didahulukan di:** §3.9 paragraf terakhir ("dilaporkan ke arah mana pun ia jatuh"), §4.4,
Gambar 4 dengan pita band derau.

---

**Q3. "Hanya 66 pohon positif. Bukankah semua angka Lapisan 1 kalian tidak berarti?"**

**Jawab:** Angkanya berarti persis sejauh 66 pengamatan mengizinkan, dan kami menyatakan
batas itu sebelum melaporkan satu pun metrik. Karena itu pula `is_unbalance=True` — yang
menaikkan PR-AUC sebesar 0,001 dengan simpangan baku 0,091 — kami **tolak**, bukan kami klaim.
Yang membuat 66 dapat dipertahankan bukan besarnya sampel melainkan protokolnya: unit analisis
adalah pohon, fold adalah ortomosaik utuh, dan aturan tidak-konklusif ditetapkan sebelum hasil
dilihat. Perlu ditambahkan bahwa 66 adalah angka yang **kami temukan sendiri**; data yang sama
tampil sebagai 2.179 kotak positif bagi siapa pun yang tidak melakukan rekonstruksi.
**Didahulukan di:** §3.3 (temuan duplikasi), §3.8 (ukuran sampel efektif dinyatakan sebelum
metrik), §4.1, Gambar 2 panel kanan.

---

**Q4. "Kalian menarik satu klaim kalian sendiri. Apa jaminan sisanya tidak akan ditarik juga?"**

**Jawab:** Tidak ada jaminan — yang ada adalah prosedur yang membuat penarikan mungkin terjadi
dan tercatat. Penarikan itu terjadi karena kami mengubah unit analisis dan menjalankan ulang
seluruhnya, bukan karena kami memilih angka yang lebih baik: koreksi yang sama menaikkan PR-AUC
utama dari 0,126 ke 0,182 **dan** membunuh temuan "kolaps situs" yang sebelumnya kami anggap
menarik. Bergerak ke dua arah sekaligus adalah tanda prosedur, bukan seleksi. Prosedur yang
sama masih terpasang: aturan tidak-konklusif, block-CV, dan berkas beku `data_clean/` yang
berhenti dengan galat bila salah satu angka bergeser.
**Didahulukan di:** §3.8 (aturan keputusan di muka), §4.3, dan catatan reproduksibilitas pada
Tabel 1.

---

**Q5. "Kedua dataset tidak bisa digabung — berarti sistem kalian tidak pernah utuh. Ini konsep
di atas kertas?"**

**Jawab:** Ini memang paper konsep, dan itu dinyatakan sejak judul bab. Namun "belum utuh"
tidak sama dengan "tidak diuji": kedua ujungnya dibangun di atas data nyata dan diuji secara
terpisah dengan protokol anti-kebocoran masing-masing, dan sambungan di antaranya diukur
alih-alih diasumsikan — derajat 5,62 versus 5,74, berselisih 2 persen. Yang hilang persis satu
hal, dan kami dapat menyebutnya dengan tepat: satu dataset yang memuat citra UAV **dan** sensus
BSR berlabel waktu pada kebun yang sama. Itu bukan kelemahan yang kabur melainkan satu butir
akuisisi data yang terdefinisi, dan ia menjadi Saran nomor 1.
**Didahulukan di:** §3.1, §3.6, §4.5, Saran 1 dan 5, konektor berangka pada Gambar 1.

---

**Q6. "PR-AUC 0,18 itu rendah. Tim lain melaporkan mAP di atas 0,97."**

**Jawab:** Kedua angka itu mengukur hal yang berbeda pada tingkat kesulitan yang berbeda, dan
perbandingannya tidak sah. PR-AUC 0,182 diperoleh pada kelas positif dengan base rate 1,30
persen — pemeringkatan acak menghasilkan 0,0130, sehingga angka kami sekitar empat belas kali
di atas kebetulan — dan diperoleh pada ortomosaik yang **tidak pernah dilihat model**. Tahap
deteksi kami, yang setara dengan angka mAP yang dibandingkan, mencapai mAP50 0,758 pada
ortomosaik yang ditahan penuh. Nilai mAP yang mendekati sempurna pada validasi yang berbagi
lokasi atau penerbangan dengan data latih mengukur autokorelasi spasial, bukan generalisasi;
pada data ini split bawaan membocorkan 100 persen ubin, dan itulah sebabnya kami membuangnya.
**Didahulukan di:** §3.8 (mengapa split acak pasti bocor; mengapa akurasi tidak dilaporkan),
dan kalimat base-rate pada §4.6.

---

**Q7. "Simulatornya kalian buat sendiri. Bukankah hasilnya otomatis mendukung metode kalian?"**

**Jawab:** Justru sebaliknya, dan itu dapat diperiksa: hasil yang paling merugikan posisi kami
lahir **di dalam** simulator kami sendiri. Simulator itu membangkitkan penularan yang benar-benar
mengalir melalui graf kontak, sehingga ia adalah kondisi paling ramah yang mungkin bagi model
graf — dan bahkan di sana, peta kontak yang benar hanya menambah +0,012 dibanding graf acak,
sementara kepala SEIR tidak menambah apa pun (5 dari 20 *seed*, tidak konklusif). Simulator
dipakai untuk satu hal yang tidak dapat dilakukan di lapangan: mengganti peta kontak sambil
menahan segala hal lain tetap. Klaim lapangan tidak pernah bersandar padanya, dan dekomposisi
yang sama dijalankan ulang pada panel Eg9PP nyata.
**Didahulukan di:** §3.5 ("simulator tidak pernah menjadi dasar klaim lapangan"), §3.9
(dekomposisi dijalankan di kedua sumber), Gambar 4 dua kelompok.

---

**Q8. "Model graf tidak memberi keuntungan waktu peringatan dibanding MLP. Lalu apa nilai
bisnisnya?"**

**Jawab:** Nilainya ada pada apa yang dipelajari tentang belanja, bukan pada kemenangan
arsitektur. Pada simulator, STGNN unggul pada AUC-PR namun **tidak** unggul secara operasional:
*lead time* -nya 2,58 siklus dibanding 2,90 siklus milik MLP (berpasangan −0,32, unggul hanya
pada 1 dari 20 *seed*) dan cakupannya tidak lebih baik. Kami melaporkannya karena sebuah sistem
yang dibeli berdasarkan AUC-PR lalu gagal memberi waktu tambahan di lapangan adalah kegagalan
pengadaan yang mahal. Rekomendasi yang mengikutinya konkret: dahulukan frekuensi survei dan
cakupan label (bernilai +0,077) di atas presisi pemetaan kontak (bernilai +0,012), dan jadikan
*lead time* beserta cakupannya — bukan AUC-PR — sebagai kriteria penerimaan.
**Didahulukan di:** §3.7 (lead time dan precision@k dilaporkan berdampingan dengan peringkat),
§4.4 kalimat anggaran, §4.6 butir keempat.

---

**Dua pertanyaan cadangan** (kurang mungkin, tetapi murah untuk disiapkan):

- **"Eg9PP berasal dari kebun yang dibongkar 2012 dan populasi *multi-parent* khusus — masih
  relevan?"** → Populasi itu justru dirancang untuk memvariasikan ketahanan, sehingga genotipe
  wajib menjadi kovariat pada semua model — tanpa itu keunggulan graf tercemar oleh fakta bahwa
  famili sekerabat ditanam berdekatan. Kesamaan geometrinya dengan kebun modern diukur, bukan
  diasumsikan (5,74 vs 5,62). Kesetaraan laju epidemi antar-era **tidak** kami klaim.
- **"Lisensi datanya?"** → CC BY-SA 4.0, hak cipta PalmElit & CIRAD, sitasi Tisné dkk. 2017
  wajib; ketentuan dan sisa risikonya dicatat di `data_clean/Eg9PP_LICENSE.md`.

---

## 6. Yang TIDAK boleh diklaim

Daftar periksa untuk setiap draf berikutnya. Setiap butir pernah menjadi klaim yang menggoda.

**Tentang data dan label**

1. **Jangan** menyebut Lapisan 1 sebagai deteksi, klasifikasi, atau peringatan dini **BSR /
   Ganoderma**. Batasnya: "deteksi tajuk dan penilaian kesehatan tajuk generik".
2. **Jangan** mengutip **151.060** (atau 2.179, atau 2.745, atau 7.500) sebagai ukuran sampel.
   Unitnya adalah **5.077 pohon unik** dan **66 positif**.
3. **Jangan** menyebut A + B + C sebagai tiga dataset independen. Ketiganya berbagi inti 2.303
   ubin identik; hanya `ds_B` yang dipakai.
4. **Jangan** menyatakan Eg9PP memiliki citra, kompartemen laten *E*, relasi angin, atau relasi
   jalur panen. Yang ada: waktu gejala pertama, waktu mati, posisi, genotipe — satu relasi.
5. **Jangan** menyajikan spektra `pwd.csv` sebagai bukti spektral BSR. Itu penyakit layu pinus
   pada spesies lain; ia menyediakan *bentuk* distribusi fitur, bukan bukti.

**Tentang metode dan angka**

6. **Jangan** melaporkan **akurasi**, F1@0,5, atau metrik berambang mana pun pada tugas
   kesehatan.
7. **Jangan** memakai atau melaporkan hasil **split acak** di sisi mana pun.
8. **Jangan** menyatakan `is_unbalance=True` membantu (Δ0,001 dengan std 0,091 — ditolak).
9. **Jangan** menghidupkan kembali klaim **"kolaps situs 52000_20000"**; ia ditarik.
10. **Jangan** menyatakan struktur graf terbukti penting, atau membulatkan +0,012 ± 0,012
    menjadi kemenangan tanpa menyebut band deraunya.
11. **Jangan** menyatakan kepala SEIR meningkatkan performa (tidak konklusif pada semua
    horizon), dan **jangan** membandingkan langsung kepala SI(D) Eg9PP dengan kepala SEIR
    simulator.
12. **Jangan** mengklaim keunggulan *lead time* atau cakupan bagi STGNN atas MLP.
13. **Jangan** melaporkan mAP 0,758 sebagai hasil 3-lipatan. Ia **satu lipatan, 15 epoch,
    imgsz 512, pendahuluan** sampai lari penuh selesai.
14. **Jangan** mengklaim IoU atau mutu segmentasi apa pun untuk ExG+Otsu — tidak ada *mask*
    kebenaran-dasar.
15. **Jangan** melaporkan peluang terkalibrasi, skor Brier, kurva keandalan, tingkat risiko,
    atau precision@k **kecuali** Bab 4 benar-benar memuat hasilnya (lihat §9).

**Tentang integrasi dan generalisasi**

16. **Jangan** menyatakan kedua dataset digabungkan, disejajarkan, atau saling melengkapi pada
    tingkat pohon.
17. **Jangan** membaca selisih 2 persen sebagai bukti bahwa keduanya dapat dipertukarkan; ia
    **batas atas**, dihitung dari kotak kebenaran-dasar.
18. **Jangan** menggeneralisasi melampaui 3 ortomosaik (Lapisan 1) dan 2 parcel satu kebun
    (Eg9PP). Tidak ada klaim nasional, provinsial, atau lintas-kultivar.
19. **Jangan** melekatkan angka rupiah pada keluaran model kami. Angka kerugian BSR boleh
    dikutip sebagai **konteks masalah** dengan sitasi, tidak pernah sebagai penghematan yang
    dijanjikan sistem ini.
20. **Jangan** mengklaim penempatan: *real-time*, Edge AI, latensi, biaya perangkat, atau
    integrasi dasbor. Tidak satu pun diukur.

---

## 7. Bank angka — setiap angka dan berkas sumbernya

Semua angka yang boleh muncul di naskah. Bila sebuah angka tidak ada di sini dan tidak ditandai
`[MENUNGGU]`, ia tidak boleh ditulis.

**Lapisan 1 — data** (`layer1_data_audit/AUDIT_REPORT.md`, `data_clean/DATASET_CARD.md`)

| Angka | Nilai |
|---|---|
| Ortomosaik | 3 (`44000_16000`, `44000_4000`, `52000_20000`) |
| Ubin ds_B / gabungan unik A∪B∪C | 2.303 / 2.745 |
| Ubin per ortomosaik | 737 / 767 / 799 |
| Ubin benar-benar saling lepas | 13 / 14 / 16 (total 43) |
| Kotak anotasi → pohon unik | 151.060 → **5.077** (29,8×) |
| Kemunculan satu pohon | median **32** ubin (rentang 1–77) |
| Pohon per ortomosaik | 1.379 / 1.849 / 1.849 |
| Unhealthy unik | **66** (1,30%); 17 / 31 / 18 |
| Ketimpangan pada tingkat pohon | ≈ 76 : 1 (5.011 : 66) — *turunan aritmetik dari dua angka di atas* |
| Konflik label antar-duplikat | 0 |
| Tetangga-hantu (< 0,5 × jarak tanam) | 0 |
| Tampilan kanonik ≥ 60 px dari tepi | 5.048 / 5.077 |
| Jarak tetangga terdekat | 101–106 px |
| GSD | 8,5–8,9 cm/px (≈ 8,7) pada asumsi jarak tanam 9 m |
| Luas blok | 16,9 / 17,6 / 18,8 ha |
| Derajat akar @ 13 m | 5,52 / 5,43 / 5,37 |

**Lapisan 1 — hasil** (`layer1_build/RESULTS_LAYER1.md`) — *hanya untuk Bab 4/5*

| Angka | Nilai |
|---|---|
| PR-AUC acak (base rate) | 0,0130 |
| LightGBM vanilla | PR-AUC **0,182 ± 0,059** · ROC-AUC 0,861 |
| Per lipatan | 0,264 / 0,155 / 0,126 |
| `is_unbalance=True` | 0,181 ± 0,091 → **ditolak** (di dalam band derau) |
| Fitur teratas | (G−R), R_std, exg_std, (G−B), B_mean |
| YOLO11n **pendahuluan, 1 lipatan** | mAP50 0,758 · mAP50-95 0,524 · P 0,862 · R 0,683 (15 epoch, imgsz 512) |
| Basis anotasi yang **DIBATALKAN** | 0,126 ± 0,068; per lipatan 0,030 / 0,17 / 0,26 — hanya boleh dikutip di dalam narasi penarikan |

**Lapisan 2A — Eg9PP** (`data_clean/DATASET_CARD.md`, `Eg9PP_LICENSE.md`)

| Angka | Nilai |
|---|---|
| Sitasi | Tisné dkk. 2017, G3 7(6):1683–1692, doi:10.1534/g3.117.041764; SOCFINDO, Medan |
| Lisensi | CC BY-SA 4.0; hak cipta PalmElit & CIRAD |
| Pohon / famili / plot / parcel | 1.200 / 14 / 80 / 2 |
| Sensus | **45** tanggal, tahun 0,5–25,5; panel 54.000 baris |
| Simptomatik / mati | 702 (58,5%) / 366 (30,5%) |
| Tersensor | 498 pohon; tersedini t = 6,0 tahun |
| Graf r = 1,5 × jarak tanam | 3.354 sisi, derajat rata-rata 5,59 |
| Sisi lintas-parcel | **0 dari 3.354** |
| Famili di kedua parcel | 14 dari 14 |
| Rentang Y parcel | 44A 3,0–14,5 · 44B 22,0–33,5 |
| Koreksi geometri | `X × cos 30°` → enam tetangga terdekat tepat di 1,000 |
| Pos-rate h = 1/2/3/4 | 1,58% / 3,03% / 4,45% / 5,65% |
| n contoh h = 1/2/3/4 | 44.311 / 43.782 / 43.225 / 42.553 |

**Uji antarmuka** (`data_clean/DATASET_CARD.md`)

| Angka | Nilai |
|---|---|
| Derajat Eg9PP (pohon bagian dalam) | **5,74** |
| Derajat Roboflow (centroid tajuk) | **5,62** |
| Selisih | **2%**; dataran pembanding r = 1,25–1,5; **batas atas** (kotak GT) |

**Lapisan 2B — simulator** (`results.csv`, `RECAP.md`, `data_clean/DATASET_CARD.md`)

| Angka | Nilai |
|---|---|
| Spektra `pwd.csv` | 1.226 spektrum, 240 band, λ 395,0–1004,3 nm; 1.029 sakit / 197 sehat |
| Keterpisahan | AUC S-vs-I 0,995 · S-vs-E @ α = 0,1 **0,710** (α = 0,2 → 0,86, terlalu mudah) |
| Parameter model | MLP 4.225 · STGNN 9.422 · STGNN-SEIR 9.534 |
| Dekomposisi (mismatch, h = 3, 20 seed) | temporal **+0,033** (19/20) · prevalensi **+0,077** (20/20) · **struktur +0,012 ± 0,012** (20/20) · total graf +0,090 · total STGNN−MLP **+0,123** |
| Kepala SEIR | −0,006 (5/20) → **tidak konklusif** |
| C1 perturbasi ε = 0 → 1,0 | +0,090 → +0,084 (datar) |
| C2 derajat 4,1 → 12,4 | +0,087 → +0,096 (datar); radius setara 9,4 / 13,2 / 13,3 / 18,5 m |
| C3 (mismatch, FPR 5%) | MLP lead 2,90 · cakupan 37% \| STGNN lead 2,58 · cakupan 29% \| berpasangan −0,32 (1/20) |
| Skala waktu | 1 siklus survei ≈ 3 bulan; 20 siklus |

---

## 8. Yang masih `[MENUNGGU]`

| Butir | Dibutuhkan untuk | Sumber yang diharapkan |
|---|---|---|
| Dekomposisi pada Eg9PP nyata (temporal/prevalensi/struktur, mean ± std, sign-count) | §3.9, Gambar 4 kelompok kanan, seluruh Bab 4 sisi lapangan | `layer2_real/` (Agen 1 + Agen 3) |
| Hasil `premise_test.py` (join count & Moran's I pada 66 Unhealthy nyata) | Pembenaran empiris premis "pohon sakit mengelompok secara spasial" — bila signifikan, ini menguatkan §3.9; bila tidak, **wajib dilaporkan** | jalankan `layer1_build/premise_test.py`; belum tercatat di berkas mana pun |
| mAP 3-lipatan penuh (mean ± std) | Menggantikan angka pendahuluan satu-lipatan | `FOLDS=0,1,2 EPOCHS=50 IMGSZ=640 python yolo_train.py` |
| Kalibrasi: kurva keandalan + skor Brier | §3.7 dan setiap kalimat "persentase risiko" | belum ada implementasi di repositori |
| precision@k dan tingkat risiko | §3.7 dan pola operasional §4.6 | belum ada implementasi di repositori |
| Angka kerugian ekonomi BSR + sitasi | Paragraf pembuka Bab 1 | belum ada; butuh sitasi terbitan |
| Tema resmi Datathon 2026 (Ristek CSUI) | Abstrak dan Kesimpulan | belum ada |
| Rasio risiko struktural Eg9PP (RR teramati vs *null* permutasi dalam-famili) | Penguat §3.9 | angka prototipe tercatat di memori proyek, **belum dibekukan ke berkas repositori** — wajib dihitung ulang dan disimpan sebelum dikutip |
| Skrip pembangkit Gambar 2, 3, 5 | Ketiga gambar itu | belum ada |

---

## 9. Risiko terbesar, dan tindakannya

**Risiko utama: Bab 3 versi sekarang mendeskripsikan pipeline yang tidak dibangun oleh kode yang
ada.** Ini persis mode kegagalan yang paper ini dirancang untuk mengecam, dan seorang juri yang
membandingkan Bab 3 dengan Bab 4 akan menemukannya. Dua wujudnya:

1. **Klaim "satu pipeline berkelanjutan enam tahap".** `section3.tex` (revisi 2026-07-23)
   menyatakan Tahap 5 menjalankan dinamika penularan **di atas geometri kebun nyata hasil
   Tahap 4**, sehingga kerapatan graf "terukur dari data". Namun `layer2_real/INTERFACE.md`
   menetapkan `adjacency()` dibangun dari `layer2_edges.csv` — geometri **Eg9PP**, bukan
   geometri Roboflow — dan `DATASET_CARD.md` menyatakan kedua dataset tidak digabungkan.
   Simulator di akar repositori pun berjalan di atas kisinya sendiri (1.600 pohon), bukan di
   atas awan titik Lapisan 1. Naskah karena itu menjanjikan sambungan yang tidak dieksekusi.
   **Koreksi 2026-08-08:** kisi itu **persegi**, bukan heksagonal — `config.GEOMETRY = "square"`,
   derajat akar rata-rata **7,70**. Seluruh hasil Lapisan 2 yang terbit dihasilkan di atas kisi
   persegi tersebut. Kebun nyata justru **segitiga**: menyuntikkan 5.077 koordinat tajuk nyata
   ke simulator memberi derajat **5,39–5,52** (lawan 5,80 kisi segitiga sintetis dan 7,70
   persegi), yaitu kisi persegi **melebihkan konektivitas ~42%**. Ini **menguatkan** temuan
   negatif Lapisan 2, bukan melemahkannya: struktur graf gagal menolong justru pada regime yang
   paling menguntungkannya. Lihat `real_geometry.py` dan `REAL_GEOMETRY.md`.
   **Tindakan:** ganti bingkai "satu pipeline enam tahap" dengan bingkai **tiga sumber bukti +
   uji antarmuka** (§3.1 dan §3.6). Ini tidak melemahkan paper — ia menukar satu klaim yang
   tidak dapat dipertahankan dengan satu hasil terukur (5,74 vs 5,62), dan ia menyelamatkan
   seluruh nilai Tahap 4, yang tetap nyata dan tetap merupakan temuan terkuat paper ini.
   `fig_pipeline.py` harus direvisi bersamaan; versi sekarang justru menggambarkan keadaan
   **sebelum** Eg9PP ditemukan.

2. **Janji yang belum ada implementasinya.** `section3.tex` menjanjikan kalibrasi peluang,
   skor Brier, kurva keandalan, tingkat risiko, dan precision@k. Tidak satu pun diproduksi oleh
   kode di repositori ini, dan tidak satu pun tercantum dalam kontrak `INTERFACE.md`.
   **Tindakan:** pilih satu dari dua, sebelum menulis draf berikutnya — **(a)** implementasikan
   kalibrasi dan precision@k pada Eg9PP dan laporkan di Bab 4; atau **(b)** turunkan bahasanya
   di §3.7 dari "dilaporkan" menjadi **rancangan yang dinyatakan belum dieksekusi**, dan
   pindahkan ke Saran. Yang tidak boleh: membiarkannya berbunyi seperti hasil.

**Risiko kedua (kecil tetapi memalukan bila lolos): ketidakkonsistenan angka ketimpangan
kelas.** `CLAUDE.md` dan `section3.tex` menyebut "≈69:1", yang benar pada **tingkat anotasi**
(148.881 : 2.179). Pada tingkat pohon unik — satu-satunya unit yang boleh dipakai paper ini —
angkanya ≈ **76:1** (5.011 : 66). Menyebut 69:1 berdampingan dengan 5.077 pohon adalah persis
jenis inkonsistensi yang dicari juri yang tajam pada paper yang mengklaim ketelitian.
**Tindakan:** ganti seluruh kemunculan 69:1 di naskah, atau labeli eksplisit sebagai tingkat
anotasi.
