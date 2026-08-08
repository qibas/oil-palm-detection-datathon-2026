# Kontrak antarmuka `layer2_real/` — DIKUNCI, jangan diubah sepihak

Tiga agen bekerja paralel di atas kontrak ini. Siapa pun yang perlu mengubahnya harus
mengatakannya di laporan akhir, bukan mengubah diam-diam.

- **Agen 1 (data)** mengimplementasikan `dataset.py`.
- **Agen 3 (model)** mengimplementasikan `models_real.py` + `run_real.py` **terhadap kontrak ini**,
  memakai stub sementara sampai `dataset.py` milik Agen 1 mendarat.
- **Agen 2 (narasi)** tidak menulis kode; ia membaca kontrak ini agar metodologinya tidak
  mengarang langkah yang tidak ada.

Sumber data (sudah beku, jangan buat ulang): `../data_clean/layer2_nodes.csv`,
`layer2_panel.csv`, `layer2_edges.csv`. Spesifikasinya di `../data_clean/DATASET_CARD.md`.

## Konstanta

```python
WINDOW   = 3            # jumlah sensus riwayat yang masuk GRU
HORIZONS = (1, 2, 3, 4) # dalam satuan SENSUS, bukan tahun (sensus tidak reguler)
N_REL    = 1            # HANYA kedekatan. Tidak ada angin, tidak ada jalur panen.
```

**Perubahan istilah 2026-07-24 (dinyatakan, bukan diam-diam).** Relasi tunggal ini dulu disebut
"kontak akar"; sekarang disebut **kedekatan (proksi kontak)**. Yang berubah hanya namanya:
`layer2_edges.csv`, radius r = 1,5 x jarak tanam, jumlah sisi 3.354, dan seluruh perilaku kode
tetap sama, jadi kontrak ini **tidak** berubah. Alasannya, infeksi lewat akar memang
terdokumentasi (Rees dkk. 2009), tetapi basidiospora juga berperan dan pohon sakit bertetangga
kerap membawa isolat yang berbeda secara genetik (Pilotti dkk. 2018). Kedekatan karena itu dapat
pula mengandung tanah, bahan tanam, dan mikroiklim bersama, dan rancangan ini tidak dapat
memisahkannya. Rincian sitasi ada di `../paper/REFERENSI.md` §2b.

## `dataset.py` — wajib mengekspor

```python
def load() -> (nodes, panel, edges)          # DataFrame apa adanya dari data_clean/
def census() -> np.ndarray                   # (45,) tanggal sensus, urut naik

def node_features(train_t) -> np.ndarray     # (T, N, d) float32
    """Fitur per pohon per sensus. WAJIB hanya-masa-lalu: fitur pada sensus t tidak
    boleh menyentuh apa pun pada t+1 ke atas. Scaler/encoder dipasang HANYA pada
    indeks sensus di `train_t`."""

def adjacency(view, seed=0) -> np.ndarray    # (N, N) float32, simetris, diagonal 0
    """view in {"true", "zero", "random", "perturb"}.
       true    = graf kedekatan dari layer2_edges.csv (r = 1,5 x jarak tanam)
       zero    = matriks nol (tanpa graf)
       random  = DERAJAT TIAP SIMPUL DIPERTAHANKAN, struktur dihancurkan
       perturb = interpolasi true<->random dengan parameter eps"""

def build_examples(h, cycles) -> (tree_idx, t_idx, y)
    """Pohon pada risk set (status 'A', at_risk==1) di sensus t; y=1 bila pohon itu
    menjadi 'S' atau 'D' dalam (t, t+h] sensus. t yang sah: t >= WINDOW-1 dan t+h < T."""

def folds() -> list[(train_nodes, test_nodes)]
    """Leave-one-parcel-out: 2 lipatan, mask boolean panjang N. Terverifikasi 0 sisi
    lintas-parcel, jadi memisahkan lipatan tidak memutus graf."""
```

## Bentuk tensor yang dimakan model (sama persis dengan Lapisan 2 sintetis)

```
F_seq : [B, WINDOW, d]
D_seq : [B, WINDOW, N_REL, d]      # d = hasil difusi tetangga, N_REL = 1
```

`models.py` di akar repo sudah memakai bentuk ini; `train.py::diffuse()` dan
`_gather_window()` bisa dipakai ulang setelah `rel` dikurangi jadi satu relasi.

## Larangan keras (kalau dilanggar, hasilnya batal)

1. **Jangan pernah** memberi status ground-truth pohon itu sendiri pada sensus t sebagai
   fitur untuk memprediksi t+h — hanya riwayat sampai t.
2. **Jangan** memasang scaler/PCA pada seluruh data; hanya pada sensus latih.
3. **Jangan** menganggap pohon berstatus `C` (tersensor) sebagai sehat. Ia keluar dari
   risk set, tidak menjadi negatif.
4. **Jangan** memakai split acak. Fold = parcel.
5. **Genotipe (`progeny`) wajib** menjadi kovariat di SEMUA model termasuk MLP — tanpa itu
   keunggulan graf tercemar oleh fakta bahwa famili sekerabat ditanam berdekatan.
6. Kompartemen laten **E tidak teramati** di data ini. Kepala SEIR harus turun jadi SI(D),
   dan hasilnya **tidak boleh** dibandingkan langsung dengan varian SEIR simulator.

## Eksperimen inti yang harus bisa dijalankan

Dekomposisi selisih STGNN−MLP, identik dengan `run_experiment.py::decomposition()`:

```
temporal    = nograph  - MLP        nilai memodelkan waktu + genotipe
prevalensi  = random   - nograph    nilai punya graf apa pun
STRUKTUR    = true     - random     nilai peta kontak yang benar
```

Pelaporan: mean ± std antar-seed, **sign-count** (mis. 19/20), dan vonis `INCONCLUSIVE`
bila selisih berada di dalam 1 std. Metrik = **AUC-PR** (pos-rate 1,6–5,7%).

---

## Penyimpangan kontrak yang DINYATAKAN: varian v3 (foto tunggal)

`dataset_v3.py` + `run_v3.py` **sengaja melanggar dua butir di atas**. Ditulis di sini
karena CLAUDE.md mewajibkan perubahan kontrak dinyatakan eksplisit, bukan dilakukan
diam-diam. Garis dasar beku tidak tersentuh: v3 adalah berkas baru, `results_real.csv`
dan `results_v2.csv` tidak berubah.

| butir | kontrak | v3 | alasan |
|---|---|---|---|
| `WINDOW` | dikunci 3 | **1** | satu foto = satu sensus; tidak ada riwayat untuk dibaca |
| Larangan **#5** | genotipe wajib | **dibuang** | genotipe tak terlihat dari udara — 14 dari 24 kolom |

### Mengapa `WINDOW=1` bukan pelanggaran yang merusak

Di dalam satu sensus, `t_years`, `census_idx`, `dt_prev`, dan `log1p_t` **identik untuk
semua pohon**. Blok SELF karena itu tidak membedakan apa pun antar-pohon di dalam satu
bidikan; ia hanya berguna untuk membandingkan antar-sensus, tugas yang bukan tugas
varian ini. Membuangnya tidak menghilangkan informasi yang relevan bagi peringkat
dalam-bidikan.

Konsekuensinya juga menuntut metrik yang berbeda, dan itu yang membuat perbandingannya
sah: AP gabungan memberi nilai pada kemampuan menebak *sensus mana ini* — berguna untuk
angka gabungan, nol guna untuk memeringkat pohon di dalam satu foto. `run_v3.py` karena
itu melaporkan **AP dalam-sensus** sebagai angka yang menentukan. Model penuh kehilangan
47% nilainya di bawah metrik itu (0,1818 → 0,0973); v3 nyaris tidak kehilangan apa pun
(0,1259 → 0,1015).

### Mengapa larangan #5 boleh dicabut DI SINI, dan hanya di sini

Larangan #5 ada karena kontaminasi famili **tidak terukur**. Untuk v3 ia sudah diukur,
dengan null permutasi dalam-famili (`run_v3_perm.py`, 200 permutasi, lintasan per-pohon
ditukar hanya antar pohon sefamili, kisi tidak pernah bergerak):

| strata | teramati | null | kelebihan | z | perm ≥ teramati |
|---|---|---|---|---|---|
| `progeny` | 0,1016 | 0,0748 ± 0,0043 | **+0,0268** | +6,25 | **0/200** |
| `progeny+parcel` | 0,1016 | 0,0772 ± 0,0040 | **+0,0244** | +6,04 | **0/200** |

Terhadap garis tanpa-graf 0,0632, pada strata terketat:

```
kekerabatan + petak   +0,0140    36%
susunan spasial       +0,0244    64%
```

Jadi kontaminasi famili **nyata dan sebesar 36%** — bukan nol, bukan segalanya. Yang
tersisa setelah famili DAN petak dipegang tetap adalah efek spasial, dan tak satu pun
dari 200 permutasi menyentuhnya.

Pencabutan ini **tidak berlaku** untuk `run_real.py` maupun `run_v2.py`. Di sana
genotipe tetap wajib: angka-angka itu tidak punya null yang menguantifikasi kontaminasi
famili, jadi larangan #5 tetap mengikat sepenuhnya.

### Batas yang tetap melekat pada v3

1. **Efek graf v3 mengandung 36% kekerabatan.** Angka mentahnya tidak boleh dikutip
   seolah seluruhnya penularan.
2. **"Kondisi tetangga" dari foto bukan Ganoderma.** v3 dilatih pada status Eg9PP yang
   terverifikasi lapangan; dipakai pada citra, masukannya kesehatan tajuk generik hasil
   detektor. Ongkos substitusi itu kini **terukur**: pada laju detektor ds_B (recall 0,446 · fpr 0,0094) AP dalam-sensus turun 0,0916 -> 0,0800, lift 1,45x -> 1,27x, **59% sinyal bertahan**. Sisa batasnya: laju itu diukur di ds_B, kebun yang berbeda dari Eg9PP.
3. Tetap satu kebun percobaan, dua parcel. Tidak ada yang berubah soal itu.
