"""Varian "foto tunggal": hanya fitur yang bisa diperoleh dari SATU citra.

MEMPERLUAS `dataset.py`, TIDAK MENGUBAHNYA. Pola yang sama dengan `dataset_v2.py`,
supaya garis dasar yang sudah beku tetap bit-reproducible.

APA YANG DIBUANG, DAN MENGAPA BOLEH DIBUANG.

    SELF  (4 kolom: t_years, census_idx, dt_prev, log1p_t)   DIBUANG
    GENO  (14 kolom one-hot progeni)                          DIBUANG
    STATE (6 kolom status)                                    DIPERTAHANKAN

Satu foto adalah SATU sensus. Di dalam satu sensus, umur dan tanggal identik untuk
semua pohon, jadi blok SELF tidak membedakan apa pun antar-pohon di foto itu. Ia
hanya berguna untuk membandingkan antar-sensus - tugas yang memang bukan tugas
produk ini. Membuangnya bukan mengorbankan informasi yang relevan bagi peringkat
dalam satu bidikan.

MENGAPA `nograph` PADA VARIAN INI SEHARUSNYA SETARA TEBAK ACAK.

Blok STATE terbukti KONSTAN 0 pada risk set - tiap pohon yang dinilai berstatus
sehat, kalau tidak ia sudah keluar dari daftar. Jadi tanpa graf, model v3 tidak
punya masukan sama sekali dan AP-nya harus jatuh ke laju dasar. Itu bukan cacat:
itu uji sanity yang menyatakan seluruh kemampuan v3 datang dari graf, dan
`run_v3.py` memeriksanya secara eksplisit.

DUA BATAS YANG MELEKAT PADA VARIAN INI, DAN TIDAK BOLEH DIHILANGKAN.

1. GENOTIPE DIBUANG, DAN ITU MELEPAS SATU PENGAMAN. Larangan #7 repositori ini
   menjadikan progeni kovariat WAJIB justru karena keluarga berkerabat ditanam
   berdampingan; tanpanya, keunggulan graf dapat tercemar tata letak famili dan
   bukan penularan. Efek graf pada v3 karena itu berpotensi MENGGELEMBUNG.
   Nyatakan; jangan nikmati.
2. "KONDISI TETANGGA" DARI FOTO BUKAN GANODERMA. v3 dilatih pada status Eg9PP
   yang terverifikasi lapangan. Dipakai pada citra, masukannya kesehatan tajuk
   generik hasil detektor. Ada pergeseran definisi label di titik penyerahan.
"""
import numpy as np

import dataset as ds

# Diteruskan apa adanya supaya pemanggil cukup mengimpor satu modul.
census = ds.census
folds = ds.folds
adjacency = ds.adjacency
build_examples = ds.build_examples
load = ds.load
diffuse = ds.diffuse

N_STATE = 6


def feature_names_v3():
    return list(ds.feature_names())[ds.STATE_SLICE]


def node_features_v3(train_t, train_nodes=None):
    """-> (T, N, 6). Blok STATE saja.

    STATE tidak diskalakan di `dataset.py` (hanya SELF yang diskalakan), jadi
    mengiris slice-nya aman dan tetap fold-invariant - tidak ada statistik latih
    yang ikut terbawa.
    """
    try:
        X = ds.node_features(train_t, train_nodes=train_nodes)
    except TypeError:                       # tanda tangan kontrak: node_features(train_t)
        X = ds.node_features(train_t)
    X = np.asarray(X, np.float32)[:, :, ds.STATE_SLICE]
    assert X.shape[2] == N_STATE, (X.shape, N_STATE)
    return X


def assert_state_is_dead_without_graph(h):
    """Buktikan klaim di docstring: STATE konstan 0 pada risk set.

    Dipanggil `run_v3.py` sebelum satu model pun dilatih. Kalau ini gagal, seluruh
    penafsiran varian v3 batal - artinya ada informasi per-pohon yang bocor ke
    simpul kueri lewat jalur yang bukan graf.
    """
    T = len(census())
    X = node_features_v3(np.arange(T))
    tree, t_idx, _ = build_examples(h, np.arange(T))
    q = X[t_idx, tree]                      # (B, 6) fitur pada simpul yang dinilai
    mx = float(np.abs(q).max()) if q.size else 0.0
    assert mx == 0.0, (
        "STATE TIDAK nol pada risk set (maks |x| = %g). Varian v3 mengandaikan "
        "simpul kueri tidak membawa informasi sendiri; asumsi itu gugur." % mx)
    return q.shape
