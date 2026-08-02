# Eg9PP — provenans, lisensi, dan kewajiban sitasi

**Berkas sumber:** `Eg9PP_Phenotypes.csv` (1.200 baris, tidak diedit — semua transformasi
terjadi di `build_layer2_real.py`).

## Asal

| | |
|---|---|
| Repositori | `https://github.com/DenisMarie/Eg9PP_Ganoderma` (berkas `Eg9PP_Phenotypes.csv`) |
| Makalah | Tisné S., Pomiès V., Riou V., Syahputra I., Cochard B., Denis M. (2017), *Identification of Ganoderma disease resistance loci using natural field infection of an oil palm multi-parent population*, **G3: Genes\|Genomes\|Genetics** 7(6):1683–1692 |
| DOI | `10.1534/g3.117.041764` |
| Lokasi lapangan | Kebun SOCFINDO, Medan, Sumatera Utara, Indonesia |
| Pemilik hak cipta | **PalmElit** dan **CIRAD** |

## Lisensi

`README.md` hulu (disalin apa adanya ke `Eg9PP_upstream_README.md`) menyatakan:

> *"The project hence is currently funded, and the copyright owned, by PalmElit and CIRAD.
> Document is under the [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)
> license, and all R code are under the [GNU AGPL v3](https://www.gnu.org/licenses/agpl.html)."*

Artinya: **penggunaan diizinkan dengan atribusi + share-alike (CC BY-SA 4.0)** — bukan izin
per-permintaan. Yang wajib dipenuhi:

1. **Sitasi Tisné et al. 2017** di setiap deliverable yang memakai data ini.
2. **Sebut pemilik hak cipta** (PalmElit & CIRAD) pada bagian data/ucapan terima kasih.
3. **Share-alike**: turunan yang mendistribusikan ulang data harus memakai lisensi setara.

## Risiko sisa (rendah, tapi dicatat)

Kata *"Document"* pada pernyataan lisensi tidak secara eksplisit menyebut berkas data. Bacaan
paling wajar adalah bahwa seluruh isi repositori non-kode tercakup CC BY-SA 4.0, dan itulah
dasar yang dipakai di sini. Jika panitia Datathon menuntut kejelasan lisensi tertulis,
kirim surel konfirmasi ke penulis korespondensi makalah sebelum publikasi final. Ini
**tidak memblokir** penggunaan untuk analisis dan penulisan sekarang.

## Yang TIDAK diambil

`Eg9PP_Phenotypes_Mapping.csv` (604 pohon tergenotipe penuh), `KIN_Eg9PP_10.Rdata`, dan
`Eg9PP_Pedigree` adalah aset untuk pemetaan QTL. SawitGuard-GNN tidak melakukan pemetaan QTL,
jadi ketiganya di luar cakupan. `PROGENY` pada `Eg9PP_Phenotypes.csv` sudah cukup sebagai
kovariat genotipe.
