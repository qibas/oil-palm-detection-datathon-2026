"""Lapisan 2 (data NYATA) — loader + konstruksi tugas untuk panel epidemi Eg9PP.

Implementasi kontrak `INTERFACE.md`. Sumbernya beku di `../data_clean/`
(`layer2_nodes.csv`, `layer2_panel.csv`, `layer2_edges.csv`); berkas ini TIDAK
membuat ulang apa pun, ia hanya mengubah panel jadi tensor yang bisa dimakan model.

Semua path ditambatkan ke lokasi berkas ini, bukan cwd.

---------------------------------------------------------------------------
Keputusan desain fitur (baca ini sebelum menambah kolom apa pun)
---------------------------------------------------------------------------
Eg9PP TIDAK punya citra, TIDAK punya spektrum. Satu-satunya sinyal per pohon per
sensus adalah statusnya sendiri (A/S/D/C) — dan pada risk set status itu SELALU
'A' (transisi bersifat monoton: A -> S -> D, atau A/S -> C; terverifikasi 0
pelanggaran). Jadi riwayat status milik pohon itu sendiri **degenerate**: nol
informasi. Konsekuensinya matriks fitur dibelah tiga blok:

  SELF  (4 kolom)  waktu/umur: t_years, census_idx, dt_prev, log1p_t
                   -> hanya-masa-lalu, sama untuk semua pohon pada sensus yang sama
  GENO  (14 kolom) one-hot `progeny` — kovariat WAJIB di semua model (larangan #5)
                   -> statis, tidak mungkin membocorkan outcome
  STATE (6 kolom)  status pohon itu sendiri: is_sympt, is_dead, is_cens + beda
                   maju-satu-langkah-ke-belakang (d_*). Pada risk set nilainya
                   **terbukti konstan 0** (diasersi di test_dataset.py).

Blok STATE ada BUKAN untuk simpul kueri — ia nol di sana. Ia ada semata sebagai
*muatan difusi*: D = A @ F, sehingga tetangga yang bergejala/mati muncul di D_seq.
Tanpa blok ini graf tidak membawa sinyal epidemi sama sekali dan seluruh ablasi
runtuh secara sepele. Dengan blok ini, larangan #1 tetap dipatuhi: nilai yang
dilihat simpul kueri tentang dirinya sendiri adalah konstanta.

Yang SENGAJA TIDAK dimasukkan ke F (dan alasannya):
  * `prev_global` (prevalensi seluruh kebun pada t) — itu agregat lintas-pohon.
    Kalau dijejalkan ke F, MLP dapat "prevalensi" gratis dan komponen
    `prevalensi = random - nograph` runtuh jadi ~0 secara konstruksi.
  * `deg`, `n_nb_*` — agregat tetangga. Itu jalur DIFUSI, bukan fitur simpul.
    Kalau dibakar ke F, MLP diam-diam dapat graf dan dekomposisi tidak berarti.
  * `xm`, `ym` — posisi spasial. MLP bisa menghafal hotspot lokal = graf terselubung.
  * `parcel`, `plot` — parcel ADALAH fold; konstan di dalam lipatan, jadi hanya
    bergeser bias. plot tidak pernah muncul di kedua lipatan.
  * `event_t1s`, `y_t1s`, `event_td`, `y_td` — itu OUTCOME-nya. Dilarang keras.
"""
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- konstanta kontrak (INTERFACE.md) --------------------------------------
WINDOW = 3             # jumlah sensus riwayat yang masuk GRU
HORIZONS = (1, 2, 3, 4)  # dalam satuan SENSUS, bukan tahun
N_REL = 1              # HANYA kontak akar

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(BASE, os.pardir, "data_clean"))

# status yang MASUK risk set. C = tersensor, bukan sehat (larangan #3).
RISK_STATUS = "A"
POS_STATUS = ("S", "D")

_SELF_NAMES = ("t_years", "census_idx", "dt_prev", "log1p_t")
_STATE_NAMES = ("is_sympt", "is_dead", "is_cens", "d_sympt", "d_dead", "d_cens")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load():
    """(nodes, panel, edges) apa adanya dari data_clean/. Urutan simpul = urutan
    baris layer2_nodes.csv; semua indeks simpul di modul ini merujuk ke situ."""
    nodes = pd.read_csv(os.path.join(DATA, "layer2_nodes.csv"))
    panel = pd.read_csv(os.path.join(DATA, "layer2_panel.csv"))
    edges = pd.read_csv(os.path.join(DATA, "layer2_edges.csv"))
    return nodes, panel, edges


@lru_cache(maxsize=1)
def census():
    """(45,) tanggal sensus, urut naik. Grid tidak reguler — dt 0,5 dan 1,0 th."""
    _, panel, _ = load()
    c = np.array(sorted(panel["t"].unique()), dtype=np.float64)
    assert len(c) == 45, f"grid sensus bergeser: {len(c)} != 45"
    assert (np.diff(c) > 0).all()
    return c


@lru_cache(maxsize=1)
def palm_ids():
    nodes, _, _ = load()
    return nodes["palm_id"].to_numpy()


@lru_cache(maxsize=1)
def status_matrix():
    """(T, N) '<U1' berisi {'A','S','D','C'}; baris = sensus, kolom = simpul."""
    nodes, panel, _ = load()
    piv = panel.pivot(index="t", columns="palm_id", values="status")
    piv = piv.reindex(index=census(), columns=palm_ids())
    A = piv.to_numpy().astype("<U1")
    assert not (A == "").any(), "ada sel panel kosong setelah pivot"
    return A


@lru_cache(maxsize=1)
def progeny_levels():
    nodes, _, _ = load()
    return tuple(sorted(nodes["progeny"].unique()))


def shape():
    """(T, N)."""
    return len(census()), len(palm_ids())


# ---------------------------------------------------------------------------
# FITUR SIMPUL
# ---------------------------------------------------------------------------
def feature_names():
    """Nama kolom, urut sesuai sumbu terakhir node_features()."""
    return (list(_SELF_NAMES)
            + [f"geno={g}" for g in progeny_levels()]
            + list(_STATE_NAMES))


def _blocks():
    """Slice tiga blok fitur pada sumbu terakhir."""
    n_self, n_geno = len(_SELF_NAMES), len(progeny_levels())
    return (slice(0, n_self),                                  # SELF
            slice(n_self, n_self + n_geno),                    # GENO
            slice(n_self + n_geno, n_self + n_geno + len(_STATE_NAMES)))  # STATE


SELF_SLICE, GENO_SLICE, STATE_SLICE = None, None, None  # diisi saat impor selesai


def raw_features(status=None, cen=None):
    """Tensor fitur MENTAH (belum diskalakan), (T, N, d) float64.

    Dipisahkan dari node_features() supaya bisa diuji hanya-masa-lalu: bangun ulang
    dari panel yang dipotong di sensus k dan bandingkan bit-per-bit (lihat
    test_dataset.py::g3_past_only). Fungsi ini HANYA membaca:
      * `cen`     (tanggal sensus, 0..k)
      * `status`  (status teramati, sensus 0..k)
      * `progeny` (statis)
    Tidak ada satu pun kolom outcome (event_t1s/y_t1s/event_td/y_td) yang disentuh.
    """
    nodes, _, _ = load()
    if status is None:
        status = status_matrix()
    if cen is None:
        cen = census()
    T, N = status.shape
    assert len(cen) == T

    d = len(feature_names())
    X = np.zeros((T, N, d), dtype=np.float64)
    s_self, s_geno, s_state = _blocks()

    # --- SELF: waktu/umur. Sensus t hanya memakai t dan t-1 (beda ke belakang). --
    dt_prev = np.zeros(T)
    dt_prev[1:] = np.diff(cen)
    X[:, :, s_self.start + 0] = cen[:, None]
    X[:, :, s_self.start + 1] = np.arange(T, dtype=np.float64)[:, None]
    X[:, :, s_self.start + 2] = dt_prev[:, None]
    X[:, :, s_self.start + 3] = np.log1p(cen)[:, None]

    # --- GENO: one-hot progeny (statis, 14 level) -----------------------------
    lev = progeny_levels()
    code = nodes["progeny"].map({g: i for i, g in enumerate(lev)}).to_numpy()
    oh = np.zeros((N, len(lev)), dtype=np.float64)
    oh[np.arange(N), code] = 1.0
    X[:, :, s_geno] = oh[None, :, :]

    # --- STATE: status pohon itu sendiri. KONSTAN 0 di risk set. --------------
    # Bukan fitur simpul kueri; ini muatan difusi supaya D = A @ F membawa jumlah
    # tetangga bergejala/mati (identik dengan n_nb_sympt / n_nb_dead di panel).
    is_sympt = np.isin(status, POS_STATUS).astype(np.float64)
    is_dead = (status == "D").astype(np.float64)
    is_cens = (status == "C").astype(np.float64)
    lvl = np.stack([is_sympt, is_dead, is_cens], axis=-1)      # (T,N,3)
    dlt = np.zeros_like(lvl)
    dlt[1:] = lvl[1:] - lvl[:-1]                                # beda hanya-masa-lalu
    X[:, :, s_state] = np.concatenate([lvl, dlt], axis=-1)
    return X


def node_features(train_t, train_nodes=None):
    """(T, N, d) float32. Fitur per pohon per sensus, WAJIB hanya-masa-lalu.

    `train_t`     : indeks sensus latih — StandardScaler dipasang HANYA di sini.
    `train_nodes` : opsional (tambahan di luar kontrak, default None = semua simpul)
                    mask/indeks simpul latih, agar scaler juga tidak melihat simpul
                    lipatan uji. Blok SELF identik antar simpul, jadi praktis tidak
                    mengubah angka; disediakan supaya klaim "scaler hanya lihat
                    data latih" benar secara harfiah.

    Hanya blok SELF (4 kolom kontinu) yang diskalakan. GENO dan STATE dibiarkan
    0/1 mentah supaya difusi A @ F langsung terbaca sebagai CACAHAN tetangga
    (dan supaya konstanta risk-set-nya persis 0, bukan 0 yang tergeser).
    """
    X = raw_features().copy()
    T, N, _ = X.shape
    tt = np.asarray(sorted(set(int(i) for i in np.asarray(train_t).ravel())), dtype=int)
    assert tt.size and tt.min() >= 0 and tt.max() < T, "train_t di luar rentang sensus"
    if train_nodes is None:
        nn = np.arange(N)
    else:
        nn = np.asarray(train_nodes)
        nn = np.where(nn)[0] if nn.dtype == bool else nn.astype(int)

    s_self, _, _ = _blocks()
    fit_rows = X[np.ix_(tt, nn)][:, :, s_self].reshape(-1, s_self.stop - s_self.start)
    scaler = StandardScaler().fit(fit_rows)
    X[:, :, s_self] = scaler.transform(
        X[:, :, s_self].reshape(-1, fit_rows.shape[1])).reshape(T, N, -1)
    return X.astype(np.float32)


def fit_scaler(train_t, train_nodes=None):
    """Scaler yang DIPAKAI node_features(), diekspos untuk pemeriksaan (test)."""
    X = raw_features()
    T, N, _ = X.shape
    tt = np.asarray(sorted(set(int(i) for i in np.asarray(train_t).ravel())), dtype=int)
    nn = np.arange(N) if train_nodes is None else (
        np.where(train_nodes)[0] if np.asarray(train_nodes).dtype == bool
        else np.asarray(train_nodes, dtype=int))
    s_self, _, _ = _blocks()
    rows = X[np.ix_(tt, nn)][:, :, s_self].reshape(-1, s_self.stop - s_self.start)
    return StandardScaler().fit(rows)


# ---------------------------------------------------------------------------
# GRAF
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _edge_index():
    """(E,2) int — sisi sebagai indeks simpul (a<b), plus parcel tiap sisi."""
    nodes, _, edges = load()
    pos = {p: i for i, p in enumerate(palm_ids())}
    a = edges["a"].map(pos).to_numpy()
    b = edges["b"].map(pos).to_numpy()
    assert not (pd.isna(a).any() or pd.isna(b).any()), "ada palm_id sisi tak dikenal"
    ei = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1).astype(int)
    par = nodes["parcel"].to_numpy()
    # Sifat yang membuat leave-one-parcel-out sah — diasersi, bukan diasumsikan.
    cross = int((par[ei[:, 0]] != par[ei[:, 1]]).sum())
    assert cross == 0, f"{cross} sisi lintas-parcel — block-CV tidak lagi sah"
    return ei, par


def degrees(A=None):
    """Derajat (jumlah sisi tak-nol per baris)."""
    if A is None:
        A = adjacency("true")
    return (A > 0).sum(1).astype(int)


def _to_dense(ei, N):
    A = np.zeros((N, N), dtype=np.float32)
    A[ei[:, 0], ei[:, 1]] = 1.0
    A[ei[:, 1], ei[:, 0]] = 1.0
    np.fill_diagonal(A, 0.0)
    return A


def _rewire(ei, par, rng, frac=1.0, sweeps=20):
    """Double-edge swap MENJAGA DERAJAT, dilakukan DI DALAM parcel.

    (a,b),(c,d) -> (a,d),(c,b). Derajat tiap simpul tidak berubah; lokalitas
    spasial hancur. Swap dibatasi per parcel supaya sifat "0 sisi lintas-parcel"
    tetap berlaku pada view `random`/`perturb` juga — kalau tidak, view acak akan
    membocorkan informasi lintas-lipatan padahal view `true` tidak, dan
    perbandingan STRUKTUR = true - random jadi tidak apple-to-apple.

    `frac` = target fraksi sisi yang BERUBAH dari graf asli (diukur, bukan
    ditebak). frac>=1 -> randomisasi penuh (`sweeps` x |E| percobaan swap).
    Mengembalikan (edge_index_baru, fraksi_berubah_terukur).
    """
    ei = ei.copy()
    orig = {(int(u), int(v)) for u, v in ei}
    cur = {(int(u), int(v)) for u, v in ei}
    m = len(ei)
    if frac <= 0:
        return ei, 0.0

    groups = [np.where(par[ei[:, 0]] == p)[0] for p in np.unique(par)]
    n_same = m  # jumlah sisi yang masih identik dengan graf asli
    target = m if frac >= 1.0 else int(np.ceil(frac * m))
    budget = sweeps * m

    while budget > 0 and (m - n_same) < target:
        g = groups[int(rng.integers(len(groups)))]
        if len(g) < 2:
            continue
        i, j = g[rng.integers(len(g))], g[rng.integers(len(g))]
        budget -= 1
        if i == j:
            continue
        a, b = ei[i]
        c, d = ei[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({int(a), int(b), int(c), int(d)}) < 4:
            continue                                   # hindari self-loop
        e1 = (min(int(a), int(d)), max(int(a), int(d)))
        e2 = (min(int(c), int(b)), max(int(c), int(b)))
        if e1 in cur or e2 in cur or e1 == e2:
            continue                                   # hindari multi-edge
        o1 = (int(min(a, b)), int(max(a, b)))
        o2 = (int(min(c, d)), int(max(c, d)))
        cur.discard(o1); cur.discard(o2)
        cur.add(e1); cur.add(e2)
        n_same -= (o1 in orig) + (o2 in orig)
        n_same += (e1 in orig) + (e2 in orig)
        ei[i] = e1
        ei[j] = e2
    return ei, (m - n_same) / m


_ADJ_CACHE = {}


def adjacency(view, seed=0, eps=None):
    """(N, N) float32, simetris, diagonal 0, bobot biner {0,1}.

    view:
      true    graf kontak akar dari layer2_edges.csv (r = 1,5 x jarak tanam)
      zero    matriks nol (tanpa graf)
      random  DERAJAT TIAP SIMPUL DIPERTAHANKAN, struktur dihancurkan
      perturb interpolasi true<->random; `eps` = target fraksi sisi yang berubah
              (eps=0 -> true, eps=1 -> identik dengan `random` pada seed yang sama)

    Catatan kontrak: `eps` adalah argumen kata-kunci TAMBAHAN. Tanda tangan
    posisional `adjacency(view, seed)` di INTERFACE.md tetap berlaku apa adanya;
    kontrak menyebut parameter eps untuk `perturb` tapi tidak mencantumkannya di
    tanda tangan. `eps` SENGAJA tidak diberi nilai default: `adjacency("perturb")`
    tanpa eps akan gagal keras, bukan diam-diam mengembalikan graf acak penuh.
    """
    if view == "perturb" and eps is None:
        raise ValueError('view="perturb" butuh eps eksplisit, mis. '
                         'adjacency("perturb", seed=0, eps=0.5)')
    key = (view, int(seed), float(-1.0 if eps is None else eps))
    if key in _ADJ_CACHE:
        return _ADJ_CACHE[key]
    ei, par = _edge_index()
    N = len(palm_ids())
    if view == "true":
        A = _to_dense(ei, N)
    elif view == "zero":
        A = np.zeros((N, N), dtype=np.float32)
    elif view in ("random", "perturb"):
        f = 1.0 if view == "random" else float(eps)
        rng = np.random.default_rng(90_000 + int(seed))
        ei2, _ = _rewire(ei, par, rng, frac=f)
        A = _to_dense(ei2, N)
    else:
        raise ValueError(f"view tidak dikenal: {view!r}")
    _ADJ_CACHE[key] = A
    return A


def rewired_fraction(seed=0, eps=1.0):  # noqa: D401 — diagnostik, bukan kontrak
    """Fraksi sisi yang benar-benar berubah untuk view perturb/random (diagnostik)."""
    ei, par = _edge_index()
    rng = np.random.default_rng(90_000 + int(seed))
    _, f = _rewire(ei, par, rng, frac=float(eps))
    return f


def adjacency_scale(A):
    """Skala global tunggal, cermin train.py::adjacency_scale: 1 / rata-rata
    jumlah baris. Pada graf biner ini = 1/derajat rata-rata, jadi D = A_scaled @ F
    kira-kira RATA-RATA tetangga (bukan jumlah)."""
    tot = A.sum(1)
    return float(1.0 / (tot.mean() + 1e-8))


# ---------------------------------------------------------------------------
# LIPATAN (leave-one-parcel-out)
# ---------------------------------------------------------------------------
def folds():
    """[(train_nodes, test_nodes)] — 2 lipatan, mask boolean panjang N.
    Terverifikasi 0 sisi lintas-parcel (diasersi di _edge_index), jadi memisahkan
    lipatan tidak memutus satu sisi pun."""
    nodes, _, _ = load()
    f = nodes["fold"].to_numpy()
    out = []
    for p in sorted(pd.unique(f)):
        te = (f == p)
        out.append((~te, te))
    return out


# ---------------------------------------------------------------------------
# TUGAS PERAMALAN
# ---------------------------------------------------------------------------
def build_examples(h, cycles=None, nodes=None):
    """(tree_idx, t_idx, y) untuk horizon h.

    Risk set = pohon berstatus 'A' (at_risk==1) pada sensus t. Pohon 'C'
    (tersensor) TIDAK PERNAH masuk — ia keluar risk set, bukan negatif.
    y = 1 bila pohon menjadi 'S' atau 'D' pada salah satu sensus di (t, t+h].
    t yang sah: t >= WINDOW-1 dan t+h < T.

    `nodes` opsional (di luar kontrak): mask/indeks simpul, untuk menyaring per
    lipatan tanpa memfilter ulang di sisi pemanggil.

    CAVEAT JUJUR: pohon yang menjadi 'C' di dalam (t, t+h] diberi y=0 walau
    nasibnya sebenarnya TIDAK DIKETAHUI. Ini definisi yang sama dengan tabel
    pos-rate di DATASET_CARD.md, jadi angkanya sebanding; test_dataset.py
    melaporkan berapa banyak negatif yang sebetulnya tersensor.
    """
    st = status_matrix()
    T, N = st.shape
    if cycles is None:
        cycles = np.arange(T)
    if nodes is None:
        keep = np.ones(N, dtype=bool)
    else:
        keep = np.asarray(nodes)
        if keep.dtype != bool:
            k = np.zeros(N, dtype=bool); k[keep.astype(int)] = True; keep = k

    trees, ts, ys = [], [], []
    for t in np.asarray(cycles, dtype=int):
        if t < WINDOW - 1 or t + h >= T:
            continue
        at_risk = (st[t] == RISK_STATUS) & keep
        fut = np.isin(st[t + 1:t + h + 1], POS_STATUS).any(0)
        idx = np.where(at_risk)[0]
        trees.append(idx)
        ts.append(np.full(idx.size, t, dtype=int))
        ys.append(fut[idx].astype(np.float32))
    if not trees:
        return (np.array([], int), np.array([], int), np.array([], np.float32))
    return np.concatenate(trees), np.concatenate(ts), np.concatenate(ys)


def censored_in_horizon(h, cycles=None, nodes=None):
    """Berapa contoh berlabel 0 yang sebenarnya TERSENSOR di dalam (t, t+h].
    Bukan bagian kontrak; dipakai untuk melaporkan batas kejujuran label."""
    st = status_matrix()
    T, N = st.shape
    if cycles is None:
        cycles = np.arange(T)
    keep = np.ones(N, dtype=bool) if nodes is None else np.asarray(nodes, dtype=bool)
    n_neg = n_c = 0
    for t in np.asarray(cycles, dtype=int):
        if t < WINDOW - 1 or t + h >= T:
            continue
        m = (st[t] == RISK_STATUS) & keep
        w = st[t + 1:t + h + 1]
        pos = np.isin(w, POS_STATUS).any(0)
        cen_ = (w == "C").any(0)
        n_neg += int((m & ~pos).sum())
        n_c += int((m & ~pos & cen_).sum())
    return n_c, n_neg


def temporal_split(frac=0.7):
    """OPSIONAL. Lipatan resmi = parcel (larangan #4). Helper ini hanya untuk
    memilih `train_t` bila seseorang ingin scaler benar-benar buta terhadap sensus
    akhir. Dengan lipatan parcel murni, train_t = seluruh 45 indeks sensus."""
    T = len(census())
    k = int(round(T * frac))
    return np.arange(k), np.arange(k, T)


# ---------------------------------------------------------------------------
# PERAKITAN TENSOR MODEL (opsional, di luar kontrak — supaya bentuknya seragam)
# ---------------------------------------------------------------------------
def diffuse(F, A, scale=True):
    """F (T,N,d) -> D (T,N,N_REL,d). Satu relasi saja (akar)."""
    s = adjacency_scale(A) if (scale and A.any()) else 1.0
    D = np.einsum("ij,tjd->tid", (A * s).astype(np.float32), F.astype(np.float32))
    return D[:, :, None, :]


def make_windows(tree_idx, t_idx, F, D):
    """-> (F_seq [B,WINDOW,d], D_seq [B,WINDOW,N_REL,d]) float32.
    Jendela berakhir di t: sensus t-(WINDOW-1) .. t. Semua hanya-masa-lalu."""
    ti = np.asarray(t_idx, dtype=int)
    ni = np.asarray(tree_idx, dtype=int)
    Fs = np.stack([F[ti - (WINDOW - 1) + w, ni] for w in range(WINDOW)], axis=1)
    Ds = np.stack([D[ti - (WINDOW - 1) + w, ni] for w in range(WINDOW)], axis=1)
    return Fs.astype(np.float32), Ds.astype(np.float32)


# slice blok diisi setelah semua definisi tersedia
SELF_SLICE, GENO_SLICE, STATE_SLICE = _blocks()
