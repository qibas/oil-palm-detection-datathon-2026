# Konteks lingkungan (angin · hujan · tanah) — bukan fitur model

Ekstensi baru, dokumen ini untuk siapa pun yang meninjau atau menulis tentangnya di naskah/pitch
deck. Kode: [`env_context.py`](env_context.py) · UI: tombol "Konteks lingkungan" di navigasi atas
`web/app.jsx` · API: `GET /api/env_context?lat=..&lon=..` di `demo_api.py`.

## Apa ini, dan apa BUKANnya

**Apa ini:** panel yang mengambil data **asli** dari dua sumber publik gratis — angin & hujan
dari Open-Meteo, tekstur tanah dari ISRIC SoilGrids — untuk satu koordinat kebun yang dimasukkan
pengguna, dan menampilkannya sebagai **konteks pendukung keputusan** di samping peringkat risiko.

**Apa BUKANnya:**
- **Bukan fitur model.** Tidak pernah masuk ke GNN mana pun (`STGNN`, `STGNN_SID`, varian v3).
  Ganoderma di paket ini menyebar lewat graf kontak akar, dan itu **sudah divalidasi** (`00_HASIL.md`
  §2.1–2.3). Menambah angin/tanah ke model butuh koordinat per-pohon yang **tidak ada** — lihat
  di bawah.
- **Bukan pengganti georeferensi.** `data_clean/DATASET_CARD.md` sudah mencatat "tak ada data
  angin ⇒ `n_rel = 1`" sebagai batas yang dipaksakan pada Eg9PP; `layer1_data_audit/` juga tidak
  mencatat GPS/EXIF pada `ds_B`. Kedua dataset **tidak bergeoreferensi**, jadi tidak ada kunci
  sungguhan untuk menyambung lintang/bujur ke satu pohon pun di keduanya.
- **Bukan hasil tervalidasi.** Tidak ada cara mengukur apakah drainase/hujan di suatu titik
  berhubungan dengan kejadian BSR **nyata** di paket ini, karena Eg9PP (satu-satunya dataset
  dengan status Ganoderma terverifikasi lapangan) tidak punya koordinat untuk diperiksa silang.

## Kenapa dibuat begini, bukan disimulasikan per-pohon

Ide awalnya: tambahkan angin dan tanah sebagai kolom baru ke checkpoint Lapisan 2, seperti
`SELF`/`GENO`/`STATE`. Itu **ditolak** setelah diperiksa: satu-satunya cara mengisi kolom
per-pohon itu adalah mengarang nilainya (mis. menyalin satu angka regional ke seluruh 1.200
sawit, atau memperkirakan variasi mikro-lokasi yang tidak ada datanya) — persis jenis "data
dummy" yang ditolak paket ini di setiap tempat lain (lihat enam larangan `README.md` dan
`INTERFACE.md`). Menambahkannya diam-diam akan membuat model tampak lebih kaya tanpa menambah
satu bit informasi nyata pun, dan berisiko digugat kalau ditinjau ulang.

Bentuk yang dipilih sebagai gantinya: **data hidup, asli, untuk SATU titik**, ditampilkan sebagai
konteks — bukan angka per-pohon yang direka lalu diselundupkan ke dalam graf.

## Sumber data

| Sumber | Apa yang diambil | Butuh API key? |
|---|---|---|
| [Open-Meteo](https://open-meteo.com) | angin saat ini (kecepatan + arah), hujan 30 hari terakhir | tidak |
| [ISRIC SoilGrids v2.0](https://rest.isric.org/soilgrids/v2.0/docs) | liat/pasir/bulk density, 0–5cm | tidak |

Keduanya dipanggil lewat `urllib` bawaan Python — **tidak menambah dependensi baru**
(`requirements.txt` tidak berubah), mengikuti alasan yang sama dengan pemilihan Starlette di
`demo_api.py`.

## Heuristik drainase & hujan

`classify_drainage()` memakai **segitiga tekstur tanah USDA standar**, bukan model terlatih:
≥40% liat → "buruk", 20–40% → "sedang", <20% → "baik". Ini keterangan **tekstur**, bukan
pengukuran drainase langsung — topografi dan muka air tanah tidak ikut terukur.

`classify_rain()` membandingkan hujan 30 hari terhadap acuan kasar 200mm/bulan (rata-rata umum
tropis basah) — **bukan** normal klimatologis situs, karena situs mana pun di sini belum pernah
diukur cukup lama untuk itu.

## Rujukan literatur (`env_context.CITATIONS`)

Tiga rujukan dicatat dari pengetahuan umum patologi sawit tentang drainase buruk/tanah tergenang
sebagai faktor yang **mendukung** perkembangan Ganoderma boninense (Rees et al. 2009; Naher et
al. 2013; Susanto et al. 2005). **Detail bibliografis (volume/halaman) belum diverifikasi
silang** — periksa ulang dari sumber asli sebelum dikutip di naskah/pitch deck akhir. Ini
ditandai eksplisit di kode dan di UI (expander "Sumber data & rujukan literatur"), mengikuti
kebiasaan paket ini menandai `00_ANGKA_FINAL.md`-style: lebih baik menandai tidak-pasti daripada
menyembunyikannya.

## Keandalan saat demo tanpa internet

`DEMO_BRIEF.md` §6 sudah mewanti-wanti: jaringan di lokasi lomba tidak boleh diandalkan.
`env_context.py` menangani ini dengan jaring pengaman dua lapis:

1. **Coba data hidup dulu**, timeout 6 detik per panggilan. Angin/hujan (Open-Meteo)
   dan tanah (ISRIC SoilGrids) diambil BERSAMAAN lewat `ThreadPoolExecutor`, bukan
   berurutan — ISRIC terukur butuh ~3,3 detik sendirian, jadi berurutan berarti
   kasus terburuk dua kali lipat. Seluruh `get_context()` lalu dipanggil lewat
   `asyncio.to_thread` di backend, jadi menunggunya tidak membekukan server.
2. **Kalau gagal** (jaringan mati, API down, timeout), jatuh ke `env_context_cache.json` — satu
   snapshot data **asli** yang direkam sebelumnya lewat `REFRESH_CACHE=1 python env_context.py`.
   UI menandai dengan jelas kapan ia menampilkan cache, bukan data hidup.

**Sebelum berangkat ke lokasi lomba**, jalankan sekali dengan internet untuk memperbarui cache:

```bash
REFRESH_CACHE=1 python env_context.py
```

Cache **tidak pernah** dihasilkan otomatis oleh UI — supaya tidak diam-diam menjadi usang tanpa
disadari.

## Kalimat yang wajib ada di layar/naskah kalau ekstensi ini disebut

> Panel ini menampilkan data lingkungan **asli** (Open-Meteo, ISRIC SoilGrids) untuk satu titik
> koordinat sebagai **konteks**, bukan masukan model. Ganoderma di sistem ini diperingkat lewat
> graf kontak akar yang tervalidasi; angin dan tanah **tidak pernah** dilatih atau diuji terhadap
> kejadian BSR nyata karena tidak ada georeferensi yang menghubungkan koordinat ke pohon di
> dataset mana pun pada paket ini.

**Jangan** menyebut panel ini sebagai "fitur AI baru" atau menyiratkan ia meningkatkan akurasi
model — ia tidak menyentuh model sama sekali.
