# Audit mutu label ds_B — mengapa mAP bukan metrik utama Lapisan 1

> Semua angka di bawah dihasilkan oleh `y12.py` (`label_audit()`, `redundancy_audit()`,
> `leak_audit()`) dan dicetak ulang oleh bagian 2 `solution_layer1_yolov12.ipynb`. Tidak ada
> angka di berkas ini yang ditulis tangan.
>
> Melengkapi `../layer1_data_audit/AUDIT_REPORT.md`, yang menetapkan bahwa ketiga dataset
> Roboflow adalah satu dataset yang di-*fork* 3× dan bahwa labelnya kesehatan tajuk generik,
> **bukan BSR**. Berkas ini menjawab pertanyaan berikutnya: *seberapa jauh label yang ada
> dapat dipercaya, dan metrik apa yang karena itu sah dilaporkan.*

Latar: model deteksi berhenti di sekitar mAP50 ≈ 0,76 sementara papan skor Roboflow untuk
dataset yang sama tampak nyaris sempurna. Tiga pengukuran di bawah menjelaskan keduanya
sekaligus, dan keduanya bukan cacat model.

---

## (a) Kotak kebenaran-dasar adalah **cap berukuran tetap**, bukan kotak yang digambar

| ortomosaik | pohon | jarak tanam | kotak med | CV kotak | ukuran unik | 5 teratas menutup | kotak > jarak tanam |
|---|---|---|---|---|---|---|---|
| `44000_16000` | 1.379 | 105,8 px | 100,5 px | 0,170 | **23** | 64,9% | 40,3% |
| `44000_4000` | 1.849 | 103,0 px | 100,5 px | 0,126 | **23** | 73,1% | 32,7% |
| `52000_20000` | 1.849 | 101,1 px | 100,5 px | **0,327** | **30** | 52,5% | 42,0% |

Ukuran yang paling sering muncul adalah angka bulat berulang — `100x101`, `113x113`,
`112x113`, `87x88`, `125x126`, `50x50` — dan rasio aspeknya terkunci di **p5 = 0,98 · median
0,99 · p95 = 1,00**. Dua puluh tiga ukuran berbeda untuk 1.379 tajuk bukan hasil menggambar
batas tajuk satu per satu; itu hasil menempelkan cap dari palet ukuran pada pusat tajuk.

**Konsekuensi yang tidak dapat dihindari.**

1. **mAP50-95 punya langit-langit yang tidak bergantung pada model.** Kebenaran-dasarnya tidak
   mengikuti batas tajuk, jadi detektor yang menggambar batas tajuk *sebenarnya* justru
   dihukum, sementara detektor yang meniru palet cap diberi nilai lebih tinggi. Selisih
   mAP50 ≈ 0,76 terhadap mAP50-95 ≈ 0,52 pada lari pendahuluan adalah tanda tangan persis
   dari keadaan ini. **Epoch tambahan, backbone lebih besar, dan augmentasi lebih berat tidak
   dapat menaikkannya** — yang dibatasi bukan model, melainkan label.
2. **Kotak bertindih secara bawaan.** 33–42% kotak lebih besar daripada jarak tanam, sehingga
   kotak tajuk bertetangga saling memotong sejak di kebenaran-dasar. NMS kemudian memangkas
   recall. Recall 0,683 pada lari pendahuluan konsisten dengan ini.
3. **`52000_20000` dianotasi tidak konsisten, dan itu BUKAN perbedaan skala.** Jarak tanamnya
   101,1 px melawan 103,0 dan 105,8 pada dua ortomosaik lain — GSD praktis sama. Yang berbeda
   adalah sebaran ukuran kotaknya: CV **0,327** melawan 0,126 dan 0,170, dengan **20,2%** kotak
   di bawah setengah jarak tanam dan `50x50` masuk lima besar. Skala tanah sama, perilaku
   penganotasi berbeda. Ini kandidat kuat penyebab yang sama untuk PR-AUC kesehatan ortomosaik
   itu yang menyimpang (0,126) pada Tahap 2.

---

## (b) Redundansi piksel: 34× lebih banyak citra daripada informasi

| ortomosaik | ubin | piksel unik | piksel dijumlah | redundansi |
|---|---|---|---|---|
| `44000_16000` | 737 | 22,4 Mpx | 772,8 Mpx | 34,5× |
| `44000_4000` | 767 | 24,2 Mpx | 804,3 Mpx | 33,3× |
| `52000_20000` | 799 | 24,3 Mpx | 837,8 Mpx | 34,5× |
| **total** | 2.303 | **70,9 Mpx** | 2.414,9 Mpx | **34,1×** |

Ubin Roboflow diambil pada offset acak, bukan pada kisi, sehingga saling bertindih. Melatih
pada "1.566 citra" per lipatan berarti melatih pada sekitar **46 ubin 1024² senilai tanah
unik**. Epoch tambahan tidak dapat menambah informasi yang memang tidak ada di dalam 70,9 Mpx
itu.

Ini juga berarti **mAP per-ubin merata-ratakan ~34 tampilan berkorelasi dari pohon yang sama**,
sehingga simpangan bakunya terlalu sempit dan ketidakpastiannya tampak lebih kecil daripada
yang sebenarnya.

---

## (c) Split bawaan Roboflow bocor ~100%

| ortomosaik | pohon unik | pohon `valid` yang juga ada di `train` | pohon `test` yang juga ada di `train` |
|---|---|---|---|
| `44000_16000` | 1.379 | 98,9% | 99,8% |
| `44000_4000` | 1.849 | 99,3% | 99,3% |
| `52000_20000` | 1.849 | 99,3% | 99,9% |
| **rata-rata antar-ortomosaik** | | **99,1%** | **99,7%** |

Diukur pada tingkat piksel, **99,8–100,0%** area ubin `valid`/`test` juga tertutup ubin `train`.

Karena satu pohon fisik muncul di median 32 ubin, split acak atas ubin praktis menjamin tiap
pohon uji sudah pernah dilihat saat latih. **Skor tinggi pada split bawaan karena itu mengukur
hafalan, bukan generalisasi.** Inilah penjelasan angka papan skor Roboflow yang nyaris
sempurna, dan angka itu **tidak sebanding** dengan angka block-CV mana pun di repositori ini.

Sebelumnya repositori menyatakan "split acak bocor 100%" sebagai penalaran. Sekarang ia angka
terukur, dan itulah pembenaran empiris untuk `region()` + *leave-one-ortho-out*.

---

## Yang justru TIDAK bermasalah

Dua hipotesis yang diuji dan **ditolak** — jangan habiskan usaha di sini:

| dugaan | hasil |
|---|---|
| Tajuk terpotong di tepi ubin merusak target | hanya **0,9%** kotak menyentuh tepi ubin (1.345 dari 151.060) |
| Sebagian pohon hanya punya tampilan terpotong | **0 pohon**; hanya 29 dari 5.077 (0,6%) bermargin < 60 px pada tampilan kanoniknya |

Konflik label antar-duplikat juga **0** (`AUDIT_REPORT.md`) — Roboflow menyalin kotak identik,
tidak menganotasi ulang.

---

## Apa yang berubah karena audit ini

**Metrik utama Lapisan 1 Tahap 1 menjadi presisi/recall/F1 PUSAT TAJUK pada pohon unik**,
ditambah RMSE pusat dinyatakan sebagai kelipatan jarak tanam, dengan radius pencocokan 0,5 ×
jarak tanam. Implementasinya `y12.centre_eval()`.

Alasannya, dan bukan sekadar karena angkanya lebih enak dibaca:

1. **Pusat adalah bagian label yang memang dapat dipercaya.** Cap ditempelkan *pada* pusat
   tajuk; yang disintesis adalah ukurannya, bukan posisinya. Jarak tanam terukur konsisten
   101–106 px di ketiga ortomosaik dan tidak ada tetangga-hantu.
2. **Pusat adalah yang dikonsumsi Lapisan 2.** Yang dibutuhkan hilir adalah koordinat tajuk per
   pohon untuk membangun graf kontak akar — bukan kotak yang ketat. Kotak tidak pernah menjadi
   keluaran; koordinat yang menjadi keluaran.
3. **Evaluasinya pada pohon unik.** Deteksi dari ubin bertindih digabungkan lebih dulu pada
   radius 0,5 × jarak tanam, jadi tiap pohon dihitung sekali — memperbaiki masalah (b).
4. **Ia tidak dibatasi geometri cap.** Berbeda dengan mAP50-95, metrik ini dapat naik bila
   modelnya benar-benar membaik.

mAP50 dan mAP50-95 **tetap dilaporkan** sebagai metrik sekunder, selalu disertai pernyataan
langit-langitnya. Menghapusnya akan menyulitkan pembanding dengan literatur; melaporkannya
tanpa langit-langitnya akan menyesatkan.

Sapuan radius (`y12.radius_sweep()`) adalah pemeriksaan kejujuran metrik barunya sendiri: kurva
F1 yang **datar** terhadap radius berarti pusatnya memang tepat, sedangkan kurva yang
**menanjak tajam** berarti "benar" hanya diperoleh dengan melonggarkan ambang.

---

## Angkanya, setelah metrik ini dijalankan

`centre_eval_folds.py`, YOLOv12n, 3 lipatan × 30 epoch, satu lingkungan. Ambang keyakinan
dipilih **silang-lipatan** — ambang untuk lipatan *f* diambil dari kurva lipatan lain saja,
sehingga ortomosaik yang ditahan tidak pernah ikut memilih apa pun tentang dirinya sendiri.
Ketiganya sepakat di conf 0,75, dan sapuan sampai 0,90 menunjukkan itu optimum **interior**,
bukan ujung grid.

| lipatan | ortomosaik | P | R | F1 | RMSE pusat |
|---|---|---|---|---|---|
| fold0 | `44000_16000` | 0,949 | 0,991 | 0,969 | 0,064 × jarak tanam |
| fold1 | `44000_4000` | 0,969 | 0,985 | 0,977 | 0,083 × jarak tanam |
| fold2 | `52000_20000` | 0,931 | 0,936 | **0,933** | 0,066 × jarak tanam |
| **rata-rata** | | **0,950 ± 0,019** | **0,971 ± 0,030** | **0,960 ± 0,024** | **0,071 ± 0,011** |

**Prediksi audit ini terbukti.** Ia meramalkan dua hal sebelum satu lipatan pun dilatih penuh,
dan keduanya terjadi:

1. **Langit-langit mAP itu nyata.** mAP50 berhenti di 0,687 ± 0,071 dan mAP50-95 di 0,425 ± 0,078,
   sementara F1 pusat mencapai 0,960 pada model yang **sama persis**. Detektornya nyaris sempurna
   menemukan pusat tajuk dan sedang-sedang saja meniru palet cap — persis yang diperkirakan bila
   yang dibatasi label, bukan model.
2. **`52000_20000` memang penganotasinya, bukan situsnya.** Ortomosaik itu turun di semua metrik,
   tetapi jaraknya terhadap dua lainnya **menyusut 3×** ketika diukur dengan pusat: −0,124 pada
   mAP50 (0,605 lawan 0,726/0,731) menjadi −0,040 pada F1 pusat. Kalau penyebabnya efek situs,
   kedua metrik akan jatuh sebanding. Yang terjadi justru metrik yang tidak bergantung ukuran
   kotak jauh lebih tahan — bukti langsung bahwa yang rusak adalah geometri kotaknya.

RMSE 0,071 × jarak tanam berarti pusatnya meleset sekitar 7% jarak antar-pohon. Itulah sebabnya
graf kontak yang dibangun dari prediksi mereproduksi derajat graf kotak-GT dalam 1,4%
(5,54 ± 0,12 lawan 5,62 ± 0,05) — jitter lokalisasi terlalu kecil untuk membalik sisi pada
radius 1,5 × jarak tanam.

---

## Batas yang harus tetap dinyatakan di naskah

- Audit ini menyangkut **mutu geometri label**, bukan kebenaran kelasnya. `Unhealthy` tetap
  kesehatan tajuk generik tanpa verifikasi lapangan, dan tetap **bukan BSR**.
- Ia tidak memperbaiki jumlah positif: masih ~66 pohon `Unhealthy` unik di seluruh dataset.
- Ia tidak menambah situs: masih 3 ortomosaik, satu kebun.
- Angka papan skor Roboflow tidak boleh dikutip berdampingan dengan angka repositori ini
  seolah keduanya sebanding.
