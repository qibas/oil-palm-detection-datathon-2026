# Brief UI demonstrasi — SawitGuard-GNN

Dokumen ini untuk siapa pun yang membangun tampilan demo. Bahasa sengaja dibuat sederhana.
Semua angka di sini sudah terverifikasi; jangan menambah angka yang tidak ada di sini.

---

## 1. Ceritanya dalam tiga kalimat

**Lapisan 1 adalah MATA.** Diberi foto drone, ia menemukan setiap pohon sawit dan di mana persis
letaknya.

**Lapisan 2 adalah INGATAN.** Diberi satu kebun yang dipantau 25 tahun, ia menebak pohon sehat mana
yang paling mungkin sakit berikutnya.

**Mata dan ingatan ini berasal dari dua kebun berbeda, dan kami tidak menyambung datanya.** Kebun
beda, zaman beda, tidak ada kunci penghubung. Yang kami lakukan: **mengukur apakah keduanya akan
cocok kalau disambung.**

> ### ⚠ Diperbarui — varian foto-tunggal (v3) mengubah klaim alur produk
>
> Dokumen ini semula menulis bahwa peringkat risiko **tidak bisa** dihasilkan dari satu foto,
> karena model meminta 24 kolom dan foto hanya mengisi sebagian. Itu benar untuk checkpoint
> `stgnn_final.pt`, dan **masih benar** untuknya.
>
> Tetapi varian `v3` yang dilatih ulang tanpa waktu dan tanpa genotipe **menyamai model penuh**
> pada tugas yang sebenarnya — memeringkat pohon di dalam satu bidikan (AP dalam-sensus
> 0,1015 lawan 0,0973; selisih +0,0042 ± 0,0035, 36/40). Alasannya: di dalam satu sensus, umur
> dan tanggal identik untuk semua pohon, jadi kolom waktu tidak membedakan apa pun.
>
> **Untuk UI ini artinya:** alur tiga langkah di §2 tetap sah sebagai cerita produk *lengkap*,
> tetapi Langkah 3 **tidak lagi harus** menunggu riwayat 3 kunjungan kalau yang diminta hanya
> peringkat dalam satu bidikan. Kalau UI diperbarui, pakai angka v3 di §4, bukan angka model
> penuh — dan bawa serta kedua batasnya: efek graf v3 mengandung **36% kekerabatan**, dan
> masukan "kondisi tetangga" dari foto adalah kesehatan tajuk generik, **belum pernah diuji**
> sebagai pengganti status terverifikasi lapangan.
>
> Rincian: `00_HASIL.md` §2.5 · kontrak: `layer2_real/INTERFACE.md` bagian akhir.

### Perumpamaan untuk juri: lebar rel kereta

Dua perusahaan membangun rel di dua pulau berbeda. Relnya tidak bisa disambung hari ini. Tapi kamu
**bisa mengukur lebar relnya** — kalau sama, suatu hari nanti kereta yang sama bisa lewat keduanya.

Itulah yang kami ukur. "Lebar rel" di sini adalah **berapa banyak tetangga yang dimiliki tiap pohon**
dalam jarak kontak akar.

```
Lapisan 1 (dari foto drone)  : 5,54 ± 0,12 tetangga
Lapisan 2 (kebun Eg9PP)      : 5,74        tetangga
                               selisih 3,5%
```

Ini **pengukuran kecocokan**, bukan kereta yang lewat. Layar 3 ada khusus untuk menjelaskan ini.

---

## 2. Alur demo — empat layar

```
 [1] FOTO DRONE          [2] GRAF KONTAK         [3] GUNTING           [4] PETA RISIKO
     upload gambar   ->      titik jadi           bandingkan       |       1.200 sawit
     tandai tiap          jaring tetangga         5,54 vs 5,74     |       diwarnai desil
     pusat tajuk                                  jelaskan jurang  |       10 paling berisiko
        |                       |                       |          |             |
     Lapisan 1 -------------- Lapisan 1 ------------- BATAS -------+-------- Lapisan 2
```

Layar 1 dan 2 memakai foto. Layar 4 memakai data lapangan 25 tahun. **Layar 3 adalah tempat kamu
menjelaskan bahwa 1→2 tidak mengalir ke 4.** Jangan lewati layar 3; di situlah letak kontribusi
ilmiahnya.

---

## 3. Apa yang dibutuhkan tiap layar

| Layar | Masukan | Perintah / berkas | Keluaran yang ditampilkan | Waktu |
|---|---|---|---|---|
| 1 Deteksi | satu gambar `.jpg` | `python layer1_build/detect_centres.py <gambar> -o out.csv` | titik pusat tajuk di atas gambar, jumlah pohon | < 1 dtk (CPU) |
| 2 Graf | `out.csv` dari layar 1 | kolom `cx, cy, deg` | titik + garis ke tetangga, derajat rata-rata | instan |
| 3 Gunting | — | angka tetap (lihat §4) | dua batang: 5,54 lawan 5,74 | instan |
| 4 Risiko | — | `layer2_real/risk_ranked.csv` | peta kisi + tabel 10 teratas | instan |

**Seluruh demo jalan di CPU.** GPU hanya dibutuhkan untuk melatih, bukan untuk memperagakan.
Tidak perlu khawatir soal mesin di lokasi lomba.

### Kolom `detect_centres.py` (layar 1 & 2)

```
group, image, tree_id, cx, cy, conf, class, deg
```

`cx`,`cy` = pusat tajuk dalam piksel · `class` = Healthy / Unhealthy · `deg` = jumlah tetangga.

Skrip juga mencetak **pemeriksaan skala**. Kalau ia bilang "DI LUAR jendela", tampilkan peringatan
di layar dan **jangan tampilkan angka derajatnya** — artinya foto itu skalanya terlalu jauh dari
data latih dan hasilnya tidak berarti.

### Kolom `risk_ranked.csv` (layar 4)

```
rank, palm_id, parcel, plot, progeny, xm, ym,
in_risk_set, status, logit, risk_percentile, risk_decile,
n_neighbours, n_sick_neighbours
```

Cara menggambar petanya:

- sumbu = `xm`, `ym` (1.200 titik, kisi segitiga)
- **hanya** titik dengan `in_risk_set == 1` (672 sawit) yang diwarnai menurut `risk_decile` 1–10
- sisanya (528) digambar abu-abu, dibedakan menurut `status`: `S` bergejala, `D` mati, `C` disensor
- tabel 10 teratas: urutkan `rank`, tampilkan `palm_id`, `risk_percentile`, `n_sick_neighbours`

Kolom `n_sick_neighbours` adalah **penjelas terbaik** di seluruh demo. Sepuluh sawit teratas
rata-rata punya 4,0 tetangga sakit; seluruh risk set 2,0; sepuluh teraman 0,4. Itu memperlihatkan
model membaca tetangga, bukan pohon itu sendiri.

---

## 4. Angka yang boleh ditampilkan

| Apa | Angka | Dari |
|---|---|---|
| Deteksi pusat tajuk, F1 | **0,960 ± 0,024** | 3 ortomosaik, leave-one-ortho-out |
| Presisi · Recall | 0,950 ± 0,019 · 0,971 ± 0,030 | idem |
| Ketepatan posisi | meleset 7,1% jarak antar-pohon | idem |
| Derajat graf, Lapisan 1 | **5,54 ± 0,12** | dari prediksi detektor |
| Derajat graf, Lapisan 2 | **5,74** | Eg9PP |
| Selisih antarmuka | **3,5%** | |
| Jumlah sawit dinilai | **672** dari 1.200 | sisanya sudah sakit/mati/disensor |
| Panjang pemantauan | **45 sensus, 25 tahun** | Eg9PP |
| **v3 — peringkat dari satu bidikan** | **AP dalam-sensus 0,1015 ± 0,0079** | 1,61× garis tanpa-skill 0,0632 |
| v3 — model penuh, metrik yang sama | 0,0973 ± 0,0107 | v3 **menyamai**, +0,0042 ± 0,0035 (36/40) |
| v3 — sumbangan peta kontak benar | +0,0296 ± 0,0057 (40/40) | 77% sinyal v3 |
| v3 — kontaminasi kekerabatan | **36%** | null dalam-famili+petak, 0/200 permutasi |

**Jangan** tampilkan mAP sebagai prestasi utama. Angkanya 0,687 dan itu **bukan** kelemahan model —
kotak acuan pada datanya adalah cap berukuran tetap, jadi mAP punya langit-langit yang tidak bisa
dilewati model mana pun. Kalau tetap ingin menyebutnya, sebut sekaligus alasannya.

---

## 5. Kalimat yang wajib ada di layar

Ini bukan hiasan. Juri akan memeriksanya, dan inilah yang membedakan paket ini.

**Di layar 4, dekat skor:**
> Angka ini **peringkat**, bukan persentase. "Skor 0,51" berarti "lebih berisiko daripada yang
> berskor 0,30", **bukan** "51% kemungkinan sakit".

**Di layar 3:**
> Kedua lapisan **tidak digabung**. Kebun berbeda, zaman berbeda, tanpa kunci penghubung. Yang
> diukur adalah kecocokan antarmuka: 5,54 lawan 5,74, selisih 3,5%.

**Di layar 1:**
> Label "Unhealthy" adalah kesehatan tajuk secara umum, **bukan** BSR/Ganoderma yang diverifikasi
> di lapangan.

**Di mana saja yang terlihat:**
> Model dilatih di kebun percobaan pemuliaan, bukan kebun produksi.

---

## 6. Teknologi yang disarankan

**Streamlit** paling cocok: tim ini sudah Python, tidak perlu belajar web, dan satu berkas cukup.

```
pip install streamlit
streamlit run demo_app.py
```

Kalau khawatir jaringan atau instalasi di lokasi, alternatifnya **halaman HTML statis** dengan
gambar dan CSV yang sudah dihitung sebelumnya — kehilangan fitur unggah gambar, tapi tidak bisa
gagal. Gambar siap pakai sudah ada di `figures/`:

- `fig_layer2_risk_map.png` — peta risiko
- `fig_layer2_score_dist.png` — sebaran skor
- `fig_pipeline.png` — diagram alur

---

## 7. Yang tidak boleh dilakukan

1. **Jangan** menjalankan `stgnn_final.pt` di atas pohon hasil deteksi Lapisan 1. Secara teknis bisa
   dipaksa dengan mengisi nol pada kolom yang hilang, angkanya akan keluar dan tampak masuk akal,
   dan **tidak berarti apa-apa**. Checkpoint itu meminta 24 fitur; 18 di antaranya (4 waktu +
   14 genotipe) tidak ada pada foto. Ini satu-satunya cara demo bisa merusak seluruh posisi
   kejujuran paket ini.
2. **Jangan** menampilkan `sigmoid(skor)` sebagai persentase.
3. **Jangan** menyebut akurasi. Model yang menebak "sehat" untuk semua pohon akurasinya di atas 98%
   dan tidak berguna sama sekali.
4. **Jangan** menghitung pohon yang disensor (`C`) sebagai sehat.
5. **Jangan** memakai `stage1_model.pkl` (model Peru) untuk membangun graf — ia hanya mengotaki
   sawit pilihan penganotasi, jadi grafnya akan bolong tanpa memunculkan galat.

---

## 8. Kalau ada waktu lebih

Urutan yang paling menambah nilai:

1. **Animasi 25 tahun** pada layar 4 — putar sensus 1→45, penyakit menyebar terlihat merambat antar
   tetangga. Datanya sudah ada di `data_clean/layer2_panel.csv` (kolom `t`, `status`).
2. **Sisi tersorot** pada layar 2 — saat kursor menunjuk satu pohon, terangi tetangganya. Membuat
   gagasan "graf" langsung terpahami tanpa penjelasan.
3. **Unggah gambar sendiri** pada layar 1 — sudah didukung `detect_centres.py`; tinggal disambungkan.
