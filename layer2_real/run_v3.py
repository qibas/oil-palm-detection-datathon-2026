"""Varian foto-tunggal lawan model penuh, dinilai DALAM-SENSUS.

    python run_v3.py [seeds]        # default 10 seed x 2 lipatan

Menulis `results_v3.csv`. Tidak menyentuh `results_real.csv` maupun `results_v2.csv`.

MENGAPA "DALAM-SENSUS" ADALAH SATU-SATUNYA PERBANDINGAN YANG ADIL.

AP yang dilaporkan `run_real.py` menggabungkan seluruh sensus. Sebagian kemampuan
yang terukur di sana adalah model menebak "ini sensus akhir, semuanya lebih
berisiko" - berguna untuk metrik gabungan, NOL guna untuk memeringkat pohon di
dalam satu foto. Repositori ini sudah punya buktinya: RR tetangga 4,47x runtuh
menjadi 1,65x setelah stratifikasi Mantel-Haenszel per sensus; selisih itu persis
efek waktu yang menyamar sebagai efek tetangga.

Model v3 tidak punya kolom waktu, jadi menilainya dengan AP gabungan akan
menghukumnya untuk tugas yang memang bukan tugasnya. Karena itu skrip ini
melaporkan KEDUANYA:

    AP gabungan     seluruh contoh uji sekaligus  (sebanding dengan run_real.py)
    AP dalam-sensus AP dihitung per sensus lalu dirata-rata  <- angka yang menentukan

Sensus dengan positif terlalu sedikit dikeluarkan dari rata-rata dalam-sensus dan
JUMLAHNYA DICETAK - AP pada 1-2 positif adalah derau, bukan hasil.

ATURAN PUTUS sama persis dengan seluruh repositori: `run_real.py::paired()`, mean
+/- std atas pasangan (lipatan, seed), plus hitungan tanda; |mean| < 1 std =
TIDAK KONKLUSIF.
"""
import os
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import dataset as ds            # noqa: E402
import dataset_v3 as v3         # noqa: E402
import models_real as M         # noqa: E402
import run_real as R            # noqa: E402

OUT = os.path.join(HERE, "results_v3.csv")
H = int(os.environ.get("H", R.PRIMARY_H))
EPOCHS = int(os.environ.get("EPOCHS", R.EPOCHS))
MIN_POS = int(os.environ.get("MIN_POS", 5))     # positif minimum agar sensus dihitung
DEVICE = R.DEVICE
ROWS = []


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
class Arm:
    """Satu konfigurasi fitur: matriks fitur + panjang jendela + difusinya."""

    def __init__(self, name, Fnp, window):
        self.name, self.window = name, window
        self.Ft = torch.as_tensor(np.asarray(Fnp, np.float32), device=DEVICE)
        self.T, self.N, self.d = self.Ft.shape
        A = np.asarray(ds.adjacency("true"), np.float32)
        self.scale = R.adjacency_scale(A)
        self.D = {"true": R.diffuse(self.Ft, torch.as_tensor(A * self.scale, device=DEVICE)),
                  "zero": torch.zeros(self.T, self.N, R.N_REL, self.d, device=DEVICE)}
        self._rand = {}
        self.A_true = A

    def view(self, name, seed):
        if name != "random":
            return self.D[name]
        if seed not in self._rand:
            Ar = np.asarray(ds.adjacency("random", seed=seed), np.float32)
            # Skala adjacency SELALU diambil dari peta BENAR supaya semua view
            # berbagi satu normalisasi - itu yang membuat `random` kontrol yang adil.
            self._rand[seed] = R.diffuse(self.Ft, torch.as_tensor(Ar * self.scale,
                                                                  device=DEVICE))
        return self._rand[seed]

    def gather(self, idx_tree, idx_t):
        tt = torch.as_tensor(np.asarray(idx_t), device=DEVICE, dtype=torch.long)
        ii = torch.as_tensor(np.asarray(idx_tree), device=DEVICE, dtype=torch.long)
        Fs, Ds = [], []
        D = self._D_active
        for w in range(self.window):
            cyc = tt - (self.window - 1) + w
            Fs.append(self.Ft[cyc, ii])
            Ds.append(D[cyc, ii])
        return torch.stack(Fs, 1), torch.stack(Ds, 1)


def within_census_ap(t_idx, y, p):
    """-> (mean AP dalam-sensus, n sensus dipakai, n sensus dibuang)."""
    aps, dropped = [], 0
    for t in np.unique(t_idx):
        m = t_idx == t
        if int(y[m].sum()) < MIN_POS or y[m].sum() == m.sum():
            dropped += 1
            continue
        aps.append(average_precision_score(y[m], p[m]))
    return (float(np.mean(aps)) if aps else float("nan")), len(aps), dropped


def train_eval(arm, fold, seed, model_name, view):
    tree, t_idx, y = ds.build_examples(H, np.arange(arm.T))
    trm, tem = (np.asarray(m, bool) for m in ds.folds()[fold])
    a, b = trm[tree], tem[tree]
    if y[a].size == 0 or y[b].size == 0 or y[b].sum() == 0:
        return {}

    arm._D_active = arm.view(view, seed)
    torch.manual_seed(1234 + seed)
    model = M.build(model_name, arm.d, horizon=H).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=R.LR, weight_decay=R.WD)

    Ftr, Dtr = arm.gather(tree[a], t_idx[a])
    ytr = torch.as_tensor(y[a], device=DEVICE)
    Fte, Dte = arm.gather(tree[b], t_idx[b])

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        R._focal(model(Ftr, Dtr), ytr).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Fte, Dte)).cpu().numpy()

    wc, nk, nd = within_census_ap(t_idx[b], y[b], p)
    return {"pooled": float(average_precision_score(y[b], p)),
            "within": wc, "n_census": nk, "n_dropped": nd,
            "posrate": float(y[b].mean())}


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SEEDS", 10))
    log("=" * 74)
    log("VARIAN FOTO-TUNGGAL (v3)  |  h=%d  |  %d seed x %d lipatan  |  epoch %d"
        % (H, seeds, len(ds.folds()), EPOCHS))
    log("=" * 74)

    shape = v3.assert_state_is_dead_without_graph(H)
    log("penjaga: STATE terbukti KONSTAN 0 pada risk set %s -> tanpa graf, v3 buta" % (shape,))

    T = len(ds.census())
    full = Arm("penuh", R.Bundle().Fnp, R.WINDOW)
    snap = Arm("foto", v3.node_features_v3(np.arange(T)), 1)
    log("lengan penuh: d=%d, window=%d   |   lengan foto: d=%d, window=1"
        % (full.d, full.window, snap.d))
    log("fitur v3: %s\n" % ", ".join(v3.feature_names_v3()))

    cfg = [("penuh:true", full, "STGNN", "true"),
           ("foto:nograph", snap, "STGNN", "zero"),
           ("foto:random", snap, "STGNN", "random"),
           ("foto:true", snap, "STGNN", "true")]

    res = {k: {"pooled": [], "within": []} for k, *_ in cfg}
    posrate, ncen = [], []
    for fold in range(len(ds.folds())):
        for seed in range(seeds):
            for name, arm, mdl, view in cfg:
                r = train_eval(arm, fold, seed, mdl, view)
                if not r:
                    continue
                res[name]["pooled"].append(r["pooled"])
                res[name]["within"].append(r["within"])
                if name == "foto:true":
                    posrate.append(r["posrate"]); ncen.append((r["n_census"], r["n_dropped"]))
        log("  lipatan %d selesai" % fold)

    log("\n%-16s %-22s %-22s" % ("", "AP gabungan", "AP dalam-sensus"))
    for name, *_ in cfg:
        pv, wv = np.array(res[name]["pooled"]), np.array(res[name]["within"])
        log("%-16s %7.4f +/- %-12.4f %7.4f +/- %.4f"
            % (name, pv.mean(), pv.std(ddof=1), np.nanmean(wv), np.nanstd(wv, ddof=1)))
        ROWS.append(dict(blok="v3", varian=name, h=H, n=len(pv),
                         ap_pooled_mean=pv.mean(), ap_pooled_std=pv.std(ddof=1),
                         ap_within_mean=float(np.nanmean(wv)),
                         ap_within_std=float(np.nanstd(wv, ddof=1))))
    log("\nlaju dasar uji: %.4f   |   sensus dipakai/dibuang per lari: %s"
        % (np.mean(posrate), ncen[0] if ncen else "-"))

    log("\n--- putusan berpasangan (aturan run_real.py::paired) ---")
    for lbl, a, b in [("graf apa pun  (foto: random - nograph)", "foto:random", "foto:nograph"),
                      ("PETA BENAR    (foto: true - random)", "foto:true", "foto:random"),
                      ("harga kepraktisan (foto:true - penuh:true)", "foto:true", "penuh:true")]:
        for key in ("pooled", "within"):
            d = np.array(res[a][key]) - np.array(res[b][key])
            d = d[np.isfinite(d)]
            m, s = d.mean(), d.std(ddof=1)
            sign = int((d > 0).sum())
            verdict = "TIDAK KONKLUSIF" if abs(m) < s else ("POS" if m > 0 else "NEG")
            log("  %-42s %-7s %+.4f +/- %.4f  %2d/%d  %s"
                % (lbl, key, m, s, sign, len(d), verdict))
            ROWS.append(dict(blok="v3-paired", varian=lbl, h=H, metrik=key, n=len(d),
                             mean=m, std=s, sign=sign, vonis=verdict))

    import csv
    keys = sorted({k for r in ROWS for k in r})
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(ROWS)
    log("\nditulis: %s" % OUT)
    log("\nBATAS: genotipe dibuang, jadi efek graf v3 BISA menggelembung karena "
        "famili berkerabat ditanam berdampingan (larangan #7). Dan 'kondisi tetangga' "
        "dari foto adalah kesehatan tajuk generik, bukan Ganoderma terverifikasi.")


if __name__ == "__main__":
    main()
