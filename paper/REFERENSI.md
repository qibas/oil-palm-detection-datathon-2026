# Referensi — SawitGuard-GNN

Daftar sumber yang benar-benar dipakai paket ini. Aturannya sama dengan aturan angka di
`METHODOLOGY_PLAN.md`: **tidak ada entri tebakan**. Setiap entri menyebut dari mana ia bisa
ditelusuri ulang, dan yang belum punya sumber ditulis `[BELUM TERCATAT]`, bukan dikarang.

Kolom **Verifikasi** menyatakan cara entri itu dicek:

| Kode | Artinya |
|---|---|
| `repo` | tertulis di berkas lisensi/README di dalam paket ini |
| `web-2026-07-23` | dicek lewat pencarian web pada 23 Juli 2026 |
| `[BELUM TERCATAT]` | dipakai atau dibutuhkan, tetapi sumbernya belum ada |

---

## 1. Sumber data

### 1.1 Eg9PP — panel lapangan Ganoderma (Lapisan 2)

| | |
|---|---|
| Repositori | https://github.com/DenisMarie/Eg9PP_Ganoderma (berkas `Eg9PP_Phenotypes.csv`) |
| Makalah | Tisné S., Pomiès V., Riou V., Syahputra I., Cochard B., Denis M. (2017). *Identification of Ganoderma disease resistance loci using natural field infection of an oil palm multi-parent population*. **G3: Genes\|Genomes\|Genetics** 7(6):1683–1692 |
| DOI | `10.1534/g3.117.041764` |
| Lisensi | **CC BY-SA 4.0** — http://creativecommons.org/licenses/by-sa/4.0/ (kode R terpisah di bawah GNU AGPL v3, https://www.gnu.org/licenses/agpl.html) |
| Hak cipta | **PalmElit** (http://www.palmelit.com/en/) dan **CIRAD** |
| Lokasi lapangan | Kebun SOCFINDO, Medan, Sumatera Utara |
| Verifikasi | `repo` — `data_clean/Eg9PP_LICENSE.md`, `data_clean/Eg9PP_upstream_README.md` |

**Kewajiban yang mengikat setiap deliverable:** sitasi Tisné dkk. 2017, sebut PalmElit dan CIRAD
sebagai pemilik hak cipta, dan berlakukan *share-alike* pada turunan yang mendistribusikan ulang
data.

### 1.2 Roboflow ds_B — citra UAV (Lapisan 1)

| | |
|---|---|
| Dataset | https://universe.roboflow.com/health-detection/oil-palm-health-detection |
| Versi | *Oil Palm Health Detection*, v2 2024-04-21; diekspor 20 April 2024; 2.303 gambar, format COCO |
| Lisensi | **CC BY 4.0** — https://creativecommons.org/licenses/by/4.0/ |
| Penyedia | "Provided by a Roboflow user" (tidak ada nama penulis pada README hulu) |
| Kunci API | https://app.roboflow.com/settings/api (dibaca `layer1_build/download.py`) |
| Verifikasi | `repo` — `layer1_build/ds_B/README.dataset.txt`, `README.roboflow.txt` |

**Peringatan yang harus ikut setiap kali dataset ini dikutip:** labelnya adalah kesehatan tajuk
generik, **bukan BSR**, dan tanpa verifikasi lapangan. Lihat `layer1_data_audit/AUDIT_REPORT.md`.

### 1.3 `data/pwd.csv` — spektra penyakit layu pinus

Dipakai **hanya** oleh simulator SEIR sintetis, yang sengaja dikeluarkan dari paket ini. Sumber
hulunya tidak tercatat di repositori, sehingga berkas itu tidak dapat dibuat ulang bila hilang.

| | |
|---|---|
| Sumber hulu | `[BELUM TERCATAT]` |
| Verifikasi | `repo` — `data_clean/DATASET_CARD.md` (Dataset 3), tanpa URL |

---

## 2. Metode dan model

| Dipakai untuk | Referensi | Tautan | Verifikasi |
|---|---|---|---|
| Deteksi tajuk (Tahap 1) | Jocher G., Qiu J. (2024). *Ultralytics YOLO11*, versi 11.0.0, lisensi AGPL-3.0. Belum memiliki DOI dan belum ada makalah resmi | https://github.com/ultralytics/ultralytics · https://docs.ultralytics.com/models/yolo11 | `web-2026-07-23` |
| Klasifikasi kesehatan (Tahap 3) | Ke G., Meng Q., Finley T., Wang T., Chen W., Ma W., Ye Q., Liu T.-Y. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. **Advances in Neural Information Processing Systems 30**, 3146–3154 | https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf · https://github.com/microsoft/LightGBM | `web-2026-07-23` |
| Ambang segmentasi tajuk (Tahap 2) | Otsu N. (1979). *A Threshold Selection Method from Gray-Level Histograms*. **IEEE Transactions on Systems, Man, and Cybernetics** 9(1):62–66. DOI `10.1109/TSMC.1979.4310076` | https://ieeexplore.ieee.org/document/4310076 | `web-2026-07-23` |
| Indeks *Excess Green* (Tahap 2) | Woebbecke D.M., Meyer G.E., Von Bargen K., Mortensen D.A. (1995). *Color Indices for Weed Identification Under Various Soil, Residue, and Lighting Conditions*. **Transactions of the ASAE** 38(1):259–269. DOI `10.13031/2013.27838` | https://elibrary.asabe.org/abstract.asp?aid=27838 | `web-2026-07-23` |
| Peringkas riwayat pada STGNN (Tahap 5) | Cho K., van Merriënboer B., Gulcehre C., Bahdanau D., Bougares F., Schwenk H., Bengio Y. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*. arXiv:1406.1078 | https://arxiv.org/abs/1406.1078 | `web-2026-07-23` |
| Stratifikasi RR per sensus (Bab 4) | Mantel N., Haenszel W. (1959). *Statistical Aspects of the Analysis of Data From Retrospective Studies of Disease*. **Journal of the National Cancer Institute** 22(4):719–748. DOI `10.1093/jnci/22.4.719` | https://academic.oup.com/jnci/article-abstract/22/4/719/900746 | `web-2026-07-23` |

---

## 2b. Pendukung klaim domain (jarak tanam, penyebaran BSR, penggabungan dua dataset)

Ditambahkan 23 Juli 2026 untuk menopang uji antarmuka "derajat 5,62 lawan 5,74, selisih 2%" dan
pilihan graf kontak pada Tahap 5. **Baca juga peringatan di bawah tabel**, karena salah satu
kelompok referensi ini justru melawan premis kita sendiri.

| Menopang | Referensi | Tautan | Verifikasi |
|---|---|---|---|
| **Jarak tanam segitiga sama sisi**, yang membuat derajat 6 pada cangkang pertama menjadi konsekuensi rancangan tanam, bukan kebetulan | Bonneau X., Impens R., Buabeng M. (2018). *Optimum oil palm planting density in West Africa*. **OCL** 25(2):A201. DOI `10.1051/ocl/2017060`. Menyatakan sawit ditanam dalam *equilateral triangle design*, dengan optimum 143–160 pohon/ha, yaitu jarak 8,5–9 m | https://doi.org/10.1051/ocl/2017060 | `web-2026-07-23` |
| Infeksi lewat akar sebagai jalur nyata | Rees R.W., Flood J., Hasan Y., Potter U., Cooper R.M. (2009). *Basal stem rot of oil palm (Elaeis guineensis); mode of root infection and lower stem invasion by Ganoderma boninense*. **Plant Pathology** 58(5):982–989. DOI `10.1111/j.1365-3059.2009.02100.x` | https://doi.org/10.1111/j.1365-3059.2009.02100.x | `web-2026-07-23` |
| **Bantahan** terhadap dominasi kontak akar: basidiospora sebagai sumber inokulum, dan pohon BSR bertetangga sering membawa isolat yang berbeda secara genetik | Pilotti C.A., Gorea E.A., Bonneau L. (2018). *Basidiospores as sources of inoculum in the spread of Ganoderma boninense in oil palm plantations in Papua New Guinea*. **Plant Pathology** 67(9):1841–1849. DOI `10.1111/ppa.12915` | https://doi.org/10.1111/ppa.12915 | `web-2026-07-23` |
| Peran basidiospora di perkebunan | Rees R.W. dkk. (2012). *Ganoderma boninense basidiospores in oil palm plantations: evaluation of their possible role in stem rots of Elaeis guineensis*. **Plant Pathology** 61(3). DOI `10.1111/j.1365-3059.2011.02533.x` | https://doi.org/10.1111/j.1365-3059.2011.02533.x | `web-2026-07-23` |
| Tinjauan faktor perkembangan BSR | *A Review of Factors Affecting Ganoderma Basal Stem Rot Disease Progress in Oil Palm*. **Plants** (2022) 11(19):2462. DOI `10.3390/plants11192462` | https://doi.org/10.3390/plants11192462 | `web-2026-07-23` (halaman penuh 403, entri dari hasil pencarian) |
| Tinjauan patogen, insidensi, dan pengendalian | *Basal Stem Rot of Oil Palm: The Pathogen, Disease Incidence, and Control Methods*. **Plant Disease**. DOI `10.1094/PDIS-02-22-0358-FE` | https://doi.org/10.1094/PDIS-02-22-0358-FE | `web-2026-07-23` |
| Kerangka formal untuk "dua sumber data tidak digabungkan, hanya antarmukanya diukur" | Bareinboim E., Pearl J. (2016). *Causal inference and the data-fusion problem*. **PNAS** 113(27):7345–7352. DOI `10.1073/pnas.1510507113` | https://doi.org/10.1073/pnas.1510507113 | `web-2026-07-23` |

### Peringatan yang harus ikut ketika referensi ini dipakai

**(a) Bonneau dkk. 2018 menguatkan sahihnya uji antarmuka, tetapi menurunkan bobot retorisnya.**
Pada kisi segitiga sama sisi berjarak *d*, cangkang tetangga pertama berisi tepat 6 pohon pada
jarak *d*, dan cangkang kedua baru muncul di *d*√3 ≈ 1,73*d*. Radius *r* = 1,5*d* karena itu
menangkap cangkang pertama saja, sehingga derajat ≈ 6 adalah **konsekuensi rancangan tanam
industri**, bukan temuan. Selisih 2% wajib dibaca sebagai **pemeriksaan kewarasan**, bukan bukti
bahwa model dapat dipindahkan antarkebun.

**(b) Ada unsur sirkular yang harus diakui.** Skala meter Tahap 4 diperoleh dengan memadankan
jarak tetangga terdekat ke jarak tanam 9 m, lalu radius grafnya ditetapkan relatif terhadap jarak
tanam itu juga. Yang masih benar-benar diuji hanyalah: (i) kedua kebun memakai **tipe kisi yang
sama** (kisi persegi akan memberi derajat 8 pada radius yang sama, bukan 6), dan (ii) deteksi dan
deduplikasi Tahap 4 tidak merusak kerapatan lokal (duplikat sisa akan menggelembungkan derajat,
pohon terlewat akan mengempiskannya).

**(c) Pilotti dkk. 2018 melawan penamaan mekanistik graf kita.** Literatur melaporkan bahwa pohon
BSR sering **tidak** membentuk klaster yang meluas dari satu sumber, dan pohon bertetangga yang
sama-sama sakit kerap membawa isolat *Ganoderma* yang berbeda secara genetik. Itu melemahkan
klaim bahwa penularan didominasi kontak akar, sekaligus melemahkan justifikasi `n_rel = 1`.
Temuan empiris kita tidak batal karenanya, sebab komponen struktur bertahan terhadap kontrol
derajat, kontrol lokalitas, dan null permutasi dalam famili. Yang harus berubah adalah namanya:
sebut ia **graf kedekatan (proksi kontak)**, bukan "graf kontak akar", dan nyatakan bahwa
kedekatan dapat mengandung tanah bersama, bahan tanam bersama, mikroiklim bersama, atau gradien
hujan basidiospora, dan rancangan kita tidak dapat memisahkannya.

**(d) Bareinboim & Pearl 2016 dipakai sebagai istilah, bukan sebagai klaim.** Kesamaan derajat
adalah **satu syarat perlu** bagi transportabilitas, jauh dari cukup. Jangan menuliskannya
seolah-olah transportabilitas sudah ditegakkan.

---

## 3. Perangkat lunak

| Paket | Peran | Tautan |
|---|---|---|
| PyTorch | pelatihan MLP, STGNN, STGNN+SI(D) | https://pytorch.org |
| scikit-learn | AUC-PR / *average precision*, penyekala, PCA | https://scikit-learn.org |
| NumPy, pandas, SciPy | pengolahan panel dan graf | https://numpy.org · https://pandas.pydata.org · https://scipy.org |
| OpenCV | pembacaan ubin dan indeks warna | https://opencv.org |
| matplotlib | figur | https://matplotlib.org |
| python-docx | pembangkit naskah `.docx` | https://python-docx.readthedocs.io |
| draw.io / diagrams.net | penyuntingan Gambar 1 | https://app.diagrams.net · https://github.com/jgraph/drawio |

---

## 4. Yang masih kosong dan wajib diisi sebelum naskah final

| Butuh | Dipakai di | Status |
|---|---|---|
| Tema resmi Datathon 2026 Ristek CSUI | Abstrak dan Kesimpulan | `[BELUM TERCATAT]` |
| Sitasi kerugian ekonomi BSR nasional (mis. Ditjenbun atau MPOB) | Bab 1, paragraf pembuka | `[BELUM TERCATAT]` |
| ~~Sitasi biologi penyebaran *Ganoderma boninense* lewat kontak akar~~ | Bab 2 dan justifikasi graf Tahap 5 | **TERISI** di §2b, tetapi hasilnya **tidak mendukung penamaan "graf kontak akar"**. Lihat peringatan (c). Yang harus dikerjakan sekarang bukan mencari sitasi lagi, melainkan mengganti nama relasinya di naskah dan kode |
| Sumber hulu `data/pwd.csv` | hanya relevan bila simulator sintetis dimasukkan kembali | `[BELUM TERCATAT]` |

Empat baris di atas **tidak boleh** ditulis di naskah sebelum sumbernya ada. Satu sitasi karangan
merusak seluruh premis paket ini, karena modal utamanya adalah bahwa angka dan rujukannya boleh
dipercaya.

---

## Padanan penanda sitasi

Penanda nama sudah **diganti nomor** pada 2026-07-24, mengikuti daftar pustaka naskah Brian.
Nomor [1] sampai [14] adalah milik naskah yang sudah ada; [15] sampai [20] ditambahkan di
belakang agar penomoran lama tidak perlu digeser.

| Nomor | Entri | Muncul di |
|---|---|---|
| [2] | Tisné dkk. 2017, dataset Eg9PP (§1.1) | 3.2, sumber data lapangan |
| [11] | Ultralytics YOLO11 (§2) | 3.1, deteksi tajuk |
| [12] | Otsu 1979, ambang (§2) | 3.1, estimasi luas tajuk |
| [13] | Ke dkk. 2017, LightGBM (§2) | 3.1, penilaian kesehatan |
| [14] | Woebbecke dkk. 1995, indeks *Excess Green* (§2) | 3.1, estimasi luas tajuk |
| **[15]** | Rees dkk. 2009, infeksi lewat akar (§2b) | 2.1 dan 3.2, penamaan relasi graf |
| **[16]** | Pilotti dkk. 2018, basidiospora (§2b) | 2.1 dan 3.2, penamaan relasi graf |
| **[17]** | Bonneau dkk. 2018, jarak tanam segitiga (§2b) | 2.3 dan 3.2, pembacaan selisih 2% |
| **[18]** | Hurlbert 1984, pseudo-replikasi | 2.2, penggabungan deteksi ganda |
| **[19]** | Mantel & Haenszel 1959 (§2) | 2.4 dan 3.3, stratifikasi risiko relatif |
| **[20]** | Cho dkk. 2014, GRU (§2) | 2.3, arsitektur STGNN |

Bareinboim & Pearl 2016 **tidak jadi disitasi** karena ruang naskah terbatas; kalimat "kedua
sumber tidak digabungkan dan yang diukur hanyalah kesesuaian antarmukanya" kini berdiri tanpa
sitasi. Entri lengkapnya tetap disimpan di §2b bila nanti diperlukan.

Perlu ditambahkan ke `REFERENSI.md` §2b jika belum: Hurlbert, S.H. (1984). *Pseudoreplication and
the Design of Ecological Field Experiments*. **Ecological Monographs** 54(2):187–211.
DOI `10.2307/1942661`.

## Catatan

- Register penulisan Bab 3 mengacu pada laporan *"Laporan_Timnya Olip - Lisa Olivia"* (berkas PDF
  lokal, bukan terbitan daring). Itu **acuan gaya penulisan, bukan sitasi**, dan tidak boleh masuk
  daftar pustaka naskah.
- Entri pada §2 dicek lewat pencarian web pada 23 Juli 2026. Entri pada §1 berasal dari berkas
  lisensi dan README yang ikut dibekukan di dalam paket ini, sehingga dapat diperiksa ulang secara
  luring.
