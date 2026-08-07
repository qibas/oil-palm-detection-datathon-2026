# `extra/` — ubin dari sumber luar

Slot untuk ubin citra di luar tiga ortomosaik Roboflow. Dibaca oleh `y12.build()` bila
`EXTRA_MODE != "ignore"` di `solution_layer1_yolov12.ipynb`.

```
extra/
  images/<apa_saja>.jpg      # .jpg | .jpeg | .png
  labels/<apa_saja>.txt      # format YOLO: "<kelas> <cx> <cy> <w> <h>", ternormalisasi
```

`<kelas>`: `0` = Healthy, `1` = Unhealthy. Nama batang berkas gambar dan label **harus sama**.
Gambar tanpa berkas label **dilewati** dan dilaporkan jumlahnya — gambar tak beranotasi bukan
data latih, dan tidak akan diam-diam dihitung sebagai ubin kosong.

## Tiga hal yang harus diputuskan sadar sebelum mengisi folder ini

**1. Lisensi.** Ketentuan layanan Google Maps/Earth melarang pengambilan ubin massal dan
pembuatan dataset turunan dari citranya. Ubin dari sana tidak dapat didistribusikan bersama
paper dan sebaiknya tidak masuk ke angka yang dilaporkan. Bila tetap dipakai untuk eksplorasi
pribadi, pakai `EXTRA_MODE="holdout"` dan sebut statusnya apa adanya. Citra udara berlisensi
terbuka adalah masukan yang bersih untuk slot yang sama.

**2. Skala.** Acuan ds_B: **diameter tajuk median ≈ 124–142 px** pada ubin 1024², yaitu GSD
8,5–8,9 cm/px pada jarak tanam acuan 9 m. Citra satelit konsumen umumnya 0,15–0,5 m/px, jadi
tajuknya jatuh jauh lebih kecil. Sel `y12.scale_check()` di notebook mencetak angka ini untuk
`EXTRA` berdampingan dengan acuannya; bila rasionya di luar 0,8–1,25×, ubin harus di-resample
dulu atau dipakai `holdout` saja. Melatih dua skala sekaligus dari data yang jumlahnya tidak
cukup untuk satu skala pun adalah cara cepat memperburuk model.

**3. Label.** `Unhealthy` pada dataset ini pun sudah label kesehatan tajuk generik tanpa
verifikasi lapangan. Melabeli sehat/sakit dengan mata dari citra satelit menambah data sekaligus
menambah **sumber derau label baru yang tidak terukur**. Bila folder ini diisi, catat siapa yang
melabeli, kapan, dan dengan kriteria apa — di berkas ini.

## Mode

| `EXTRA_MODE` | perlakuan | pertanyaan yang dijawab |
|---|---|---|
| `ignore` | tidak dipakai | — |
| `holdout` | lipatan tambahan `foldX`: **hanya diuji, tak pernah dilatih** | apakah model 3-ortho bertahan di domain lain? |
| `train` | ditambah ke train **setiap** lipatan, **tak pernah** ke val | apakah data tambahan menaikkan performa pada 3 ortomosaik nyata? |

Tidak ada mode yang mengaduk ubin ini secara acak ke train+val. Itu akan membuat model
mengukur dirinya sendiri, persis kebocoran yang block-CV per-ortomosaik dibangun untuk mencegah.
`foldX` juga dikeluarkan dari rata-rata 3-ortho di `y12.paired()` — ia menjawab pertanyaan lain.

## Catatan provenans

> Isi tabel ini bila folder diisi. Kosong = folder belum pernah dipakai.

| tanggal | sumber | lisensi | jumlah ubin | GSD / rasio skala | pelabel & kriteria |
|---|---|---|---|---|---|
| | | | | | |
