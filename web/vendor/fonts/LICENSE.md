# Font pihak ketiga — atribusi wajib

Ketiga keluarga font di direktori ini berlisensi **SIL Open Font License 1.1**,
yang secara eksplisit mengizinkan penyertaan ulang (self-hosting) selama lisensinya
ikut disertakan dan font tidak dijual sendirian. Itulah yang dilakukan berkas ini.

| Keluarga | Perancang | Lisensi |
|---|---|---|
| **Space Grotesk** | Florian Karsten | SIL OFL 1.1 |
| **Instrument Sans** | Rodrigo Fuenzalida, Jordan Egstad | SIL OFL 1.1 |
| **JetBrains Mono** | JetBrains, Philipp Nurullin, Konstantin Bulenkov | SIL OFL 1.1 |

Teks lisensi lengkap: <https://openfontlicense.org/open-font-license-official-text/>

## Kenapa di-vendor, bukan dimuat dari Google

Demo ini harus jalan **tanpa internet** di lokasi lomba. Sebelumnya `web/index.html`
memuat `fonts.googleapis.com`; tanpa jaringan permintaan itu gagal **diam-diam** —
halaman tetap tampil, tetapi tipografinya jatuh ke font sistem dan tampilannya
berbeda dari yang dirancang. Kegagalan senyap seperti itu paling buruk saat demo
langsung, jadi fontnya disalin ke sini.

Hanya subset **latin** dan **latin-ext** yang diambil (14 dari 24 blok `@font-face`);
cyrillic, greek dan vietnamese dibuang karena tidak dipakai teks Indonesia.
Totalnya 285 KB.

Dibangun ulang dengan `python tools/vendor_fonts.py` (butuh internet **sekali**,
saat membangun — bukan saat demo).
