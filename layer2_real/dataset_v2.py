"""Eg9PP v2 — kontrol yang lebih kuat + fitur yang lebih bermuatan biologi.

Memperluas `dataset.py`, TIDAK mengubahnya: baseline beku tetap bisa direproduksi
bit-per-bit, dan `test_dataset.py` tetap berlaku apa adanya.

Tiga tambahan, masing-masing menjawab satu kelemahan yang terukur.

1. KONTROL `random_local`
   Semua 3.354 sisi asli panjangnya PERSIS 1,000 (kisi segitiga, r = 1,5 x jarak
   tanam hanya menjangkau cangkang pertama). View `random` yang ada menghubungkan
   pohon berjarak median 13,2 jarak tanam. Jadi `true - random` membandingkan
   "6 tetangga langsung" lawan "6 pohon di seberang petak" -- ia bisa jadi mengukur
   LOKAL vs GLOBAL, bukan PETA BENAR vs PETA SALAH.

   Rewire jaga-jarak MUSTAHIL di sini: satu-satunya pasangan berjarak 1,0 adalah
   keenam tetangga yang benar, jadi ia akan mereproduksi graf aslinya. Yang bisa
   dilakukan adalah membatasi radius: `random_local` mengacak pasangan tetapi
   memaksa setiap sisi baru tetap di dalam MAX_LOCAL jarak tanam. Derajat tiap
   pohon, parcel, dan LOKALITAS dipertahankan; hanya peta persisnya yang hancur.

   Kalau STRUKTUR bertahan terhadap kontrol ini, temuannya kokoh. Kalau runtuh,
   yang kita ukur selama ini adalah lokalitas -- dan itu tetap dilaporkan.

2. FITUR umur inokulum + paparan kumulatif
   Panel hanya punya penanda biner is_sympt / is_dead. Ganoderma menumpuk: tunggul
   yang mati lima tahun lalu adalah sumber inokulum yang jauh lebih besar daripada
   yang baru mati. Empat kolom muatan baru, semuanya NOL di simpul kueri (yang
   selalu berstatus 'A'), jadi larangan #1 tetap dipatuhi -- persis seperti blok
   STATE. Nilainya baru muncul lewat difusi D = A @ F.

3. DIFUSI 2-HOP
   `N_REL=1` membuat softmax perhatian-relasi mati (identik 1,0, gradien nol).
   Menambahkan cangkang 2-hop sebagai relasi kedua menghidupkannya kembali dan
   memberi model jangkauan melampaui tetangga langsung.
"""
import os
import re
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as ds   # noqa: E402

# Radius maksimum sisi hasil rewire (kelipatan jarak tanam) + kedalaman pengadukan.
# Disapu, bukan ditebak — sisa sisi ASLI setelah rewire (makin kecil makin kuat):
#   max_local  sweeps=60  sweeps=300        panjang sisi median
#      2,5       84,7%      51,2%                 1,00
#      3,0       54,4%      22,0%                 2,00   <- dipakai (ketat & lokal)
#      4,0       33,1%      13,2%                 2,65
#      6,0       11,7%       6,7%                 4,00   <- anak tangga menengah
# Pembanding: view `random` global punya panjang sisi median 13,2 dan sisa 1,0%.
MAX_LOCAL = 3.0
SWEEPS = 300
_V2_NAMES = ("age_sympt", "age_dead", "cum_sympt", "cum_dead")
_cache = {}


# ---------------------------------------------------------------------------
# 1. KONTROL — rewire yang mempertahankan derajat, parcel, DAN lokalitas
# ---------------------------------------------------------------------------
def _coords():
    nodes, _, _ = ds.load()
    return nodes[["xm", "ym"]].to_numpy(float), nodes["parcel"].to_numpy()


def adjacency_local(seed=0, max_local=MAX_LOCAL, sweeps=SWEEPS):
    """Seperti view `random`, tapi setiap sisi baru wajib <= max_local jarak tanam.

    Double-edge swap: ambil sisi (a,b) dan (c,d), usulkan (a,d) dan (c,b). Diterima
    hanya bila tidak ada gelang/sisi ganda, kedua sisi baru sepetak, DAN kedua sisi
    baru berada di dalam radius. Derajat tiap simpul kekal secara konstruksi.
    """
    key = ("local", seed, max_local, sweeps)
    if key in _cache:
        return _cache[key]
    xy, par = _coords()
    N = len(xy)
    A = ds.adjacency("true")
    ei = np.argwhere(np.triu(A) > 0)
    rng = np.random.default_rng(20_000 + seed)

    near = cKDTree(xy).sparse_distance_matrix(cKDTree(xy), max_local).keys()
    ok = np.zeros((N, N), dtype=bool)
    for i, j in near:
        if i != j and par[i] == par[j]:
            ok[i, j] = True

    E = [tuple(e) for e in ei]
    have = set(E) | {(b, a) for a, b in E}
    n_sw = 0
    for _ in range(sweeps):
        order = rng.permutation(len(E))
        for k in range(0, len(order) - 1, 2):
            i1, i2 = order[k], order[k + 1]
            a, b = E[i1]
            c, d = E[i2]
            if rng.random() < 0.5:
                c, d = d, c
            if len({a, b, c, d}) < 4:
                continue
            if not (ok[a, d] and ok[c, b]):
                continue
            if (a, d) in have or (c, b) in have:
                continue
            have.discard((a, b)); have.discard((b, a))
            have.discard((c, d)); have.discard((d, c))
            E[i1] = (a, d); E[i2] = (c, b)
            have.add((a, d)); have.add((d, a))
            have.add((c, b)); have.add((b, c))
            n_sw += 1

    out = np.zeros((N, N), dtype=np.float32)
    for a, b in E:
        out[a, b] = out[b, a] = 1.0
    _cache[key] = out
    return out


def view(name, seed=0):
    """`true` / `zero` / `random` diteruskan ke dataset.py; `random_local` baru."""
    if name.startswith("random_local"):
        m = re.match(r"random_local(\d+(?:\.\d+)?)$", name)
        return adjacency_local(seed, max_local=float(m.group(1)) if m else MAX_LOCAL)
    return ds.adjacency(name, seed=seed)


def edge_stats(A):
    """Diagnostik kontrol: panjang sisi + berapa persen sisi asli yang tersisa."""
    xy, par = _coords()
    ei = np.argwhere(np.triu(A) > 0)
    L = np.linalg.norm(xy[ei[:, 0]] - xy[ei[:, 1]], axis=1)
    T = ds.adjacency("true")
    kept = int((T[ei[:, 0], ei[:, 1]] > 0).sum())
    cross = int((par[ei[:, 0]] != par[ei[:, 1]]).sum())
    return dict(n=len(ei), len_med=float(np.median(L)), len_max=float(L.max()),
                kept_frac=kept / len(ei), cross=cross)


# ---------------------------------------------------------------------------
# 2. FITUR — umur inokulum + paparan kumulatif (muatan difusi, NOL di simpul kueri)
# ---------------------------------------------------------------------------
def raw_features_v2(status=None, cen=None):
    """(T, N, d+4). Empat kolom muatan tambahan, semuanya hanya-masa-lalu.

    age_sympt : tahun sejak pohon ini mulai bergejala   (0 bila belum)
    age_dead  : tahun sejak pohon ini mati              (0 bila belum)
    cum_sympt : integral is_sympt x dt sepanjang riwayat (tahun-bergejala)
    cum_dead  : integral is_dead  x dt sepanjang riwayat (tahun-mati)

    Semuanya 0 untuk pohon berstatus 'A', jadi simpul kueri tetap melihat konstanta
    tentang dirinya sendiri. Nilainya hanya muncul setelah difusi: "berapa lama
    tetangga saya sudah jadi sumber inokulum".
    """
    if status is None:
        status = ds.status_matrix()
    if cen is None:
        cen = ds.census()
    base = ds.raw_features(status=status, cen=cen)
    T, N, _ = base.shape

    is_s = np.isin(status, ds.POS_STATUS).astype(np.float64)
    is_d = (status == "D").astype(np.float64)
    dt = np.zeros(T); dt[1:] = np.diff(cen)

    age_s = np.zeros((T, N)); age_d = np.zeros((T, N))
    cum_s = np.zeros((T, N)); cum_d = np.zeros((T, N))
    for t in range(1, T):
        age_s[t] = np.where(is_s[t] > 0, age_s[t - 1] + dt[t], 0.0)
        age_d[t] = np.where(is_d[t] > 0, age_d[t - 1] + dt[t], 0.0)
        cum_s[t] = cum_s[t - 1] + is_s[t] * dt[t]
        cum_d[t] = cum_d[t - 1] + is_d[t] * dt[t]
    extra = np.stack([age_s, age_d, cum_s, cum_d], axis=-1)
    return np.concatenate([base, extra], axis=-1)


def node_features_v2(train_t, train_nodes=None):
    """Sama disiplin dengan dataset.node_features: scaler HANYA pada sensus latih,
    dan hanya blok SELF kontinu yang diskalakan. Kolom muatan dibiarkan mentah
    supaya difusi terbaca sebagai jumlah/akumulasi tetangga."""
    X = raw_features_v2()
    s_self, _, _ = ds._blocks()
    tr = np.asarray(train_t, dtype=int)
    sub = X[tr][:, train_nodes] if train_nodes is not None else X[tr]
    blk = sub[..., s_self].reshape(-1, s_self.stop - s_self.start)
    mu, sd = blk.mean(0), blk.std(0)
    sd[sd < 1e-8] = 1.0
    X[..., s_self] = (X[..., s_self] - mu) / sd
    return X.astype(np.float32)


def feature_names_v2():
    return tuple(ds.feature_names()) + _V2_NAMES


# ---------------------------------------------------------------------------
# 3. DIFUSI 2-HOP
# ---------------------------------------------------------------------------
def hop2(A):
    """Cangkang tepat-2-langkah: terjangkau dalam 2 hop, TAPI bukan tetangga langsung."""
    B = (A > 0).astype(np.float32)
    R = ((B @ B) > 0).astype(np.float32)
    R -= R * B
    np.fill_diagonal(R, 0.0)
    return R


def relations(A, two_hop=True):
    """Daftar matriks relasi untuk D. n_rel = 2 bila two_hop, kalau tidak 1."""
    return [A, hop2(A)] if two_hop else [A]


def adjacency_radius(r):
    """Graf kontak pada radius sembarang (kelipatan jarak tanam).

    Baseline memakai r = 1,5 yang hanya menjangkau cangkang pertama (6 tetangga,
    semua berjarak 1,0). Cangkang berikutnya ada di 1,732 lalu 2,0 — jadi r yang
    lebih besar menambah tetangga BERJARAK BERBEDA, bukan sekadar lebih banyak.
    Dipakai untuk menguji apakah kontak akar Ganoderma menjangkau lebih jauh
    daripada tetangga langsung.
    """
    key = ("rad", float(r))
    if key in _cache:
        return _cache[key]
    xy, par = _coords()
    N = len(xy)
    A = np.zeros((N, N), dtype=np.float32)
    for i, j in cKDTree(xy).query_pairs(float(r)):
        if par[i] == par[j]:
            A[i, j] = A[j, i] = 1.0
    _cache[key] = A
    return A
