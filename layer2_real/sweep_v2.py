"""Eg9PP — sapuan dua lever terakhir: RADIUS graf dan PANJANG JENDELA.

Fitur (umur inokulum, paparan kumulatif) dan difusi 2-hop sudah diuji di run_v2.py
dan keduanya MEMPERBURUK. Dua lever yang tersisa dan masih masuk akal secara biologi:

  RADIUS  baseline r = 1,5 hanya menjangkau cangkang pertama (6 tetangga, semuanya
          berjarak 1,0). r = 1,8 menambah cangkang 1,73; r = 2,05 menambah 2,0.
          Kalau kontak akar Ganoderma menjangkau lebih jauh, r besar harus menang.

  JENDELA baseline WINDOW = 3 sensus (~1,5 tahun riwayat). BSR berkembang lambat;
          jendela lebih panjang mungkin menangkap penumpukan tekanan.

Semua dinilai AUC-PR, leave-one-parcel-out x seed, dipasangkan terhadap baseline.
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataset as ds        # noqa: E402
import dataset_v2 as v2     # noqa: E402
import run_v2 as R          # noqa: E402

SEEDS = int(os.environ.get("SEEDS", 5))
H = R.HORIZON


class FeatCfg(R.Feat):
    """Varian dengan adjacency radius bebas dan WINDOW bebas."""

    def __init__(self, radius, window):
        self.radius, self.window = radius, window
        R.WINDOW = window                       # dipakai R._windows
        super().__init__(f"r{radius}_w{window}", use_v2=False, two_hop=False)
        tree, t, y = self.ex[H]
        keep = (t >= window - 1)                # jendela penuh harus muat
        self.ex[H] = (tree[keep], t[keep], y[keep])

    def D(self, view, seed):
        key = (view, seed if view.startswith("random") else 0)
        if key not in self._D:
            if view == "zero":
                self._D[key] = torch.zeros(self.T, self.N, 1, self.d, device=R.DEVICE)
            else:
                A = (v2.adjacency_radius(self.radius) if view == "true"
                     else np.asarray(v2.view(view, seed=seed), np.float32))
                self._D[key] = R.diffuse(self.Ft, [A], self.scale)
        return self._D[key]


def run(radius, window, seeds):
    R.WINDOW = window
    ft = FeatCfg(radius, window)
    ap = {}
    for fold in range(len(ft.folds)):
        for s in seeds:
            R.WINDOW = window
            ap[(fold, s)] = R.train_eval(ft, fold, s, "STGNN", "true")
    a = np.asarray([ap[k] for k in sorted(ap)], float)
    return ap, a


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    seeds = list(range(SEEDS))
    t0 = time.time()
    print(f"seeds={SEEDS} h={H} | baseline: r=1.5 window=3\n")

    base_ap, base = run(1.5, 3, seeds)
    print(f"{'konfigurasi':>22} | {'derajat':>7} | {'contoh':>7} | {'AP':>16} | "
          f"{'delta vs baseline':>19} | sign  vonis")
    print(f"{'r=1.5 w=3 (baseline)':>22} | {5.59:7.2f} | {'':>7} | "
          f"{base.mean():.4f} +/- {base.std():.4f} |")

    for radius in (1.8, 2.05, 3.05):
        ap, a = run(radius, 3, seeds)
        d = np.asarray([ap[k] - base_ap[k] for k in sorted(ap)], float)
        m, s, pos, n, v = R.paired(d)
        deg = v2.adjacency_radius(radius).sum(1).mean()
        print(f"{'r=%.2f w=3' % radius:>22} | {deg:7.2f} | {'':>7} | "
              f"{a.mean():.4f} +/- {a.std():.4f} | {m:+.4f} +/- {s:.4f} | {pos}/{n}  {v}")

    for window in (5, 8):
        ap, a = run(1.5, window, seeds)
        # jendela berbeda -> himpunan contoh berbeda, jadi TIDAK dipasangkan
        print(f"{'r=1.5 w=%d' % window:>22} | {5.59:7.2f} | {'':>7} | "
              f"{a.mean():.4f} +/- {a.std():.4f} | {'(tak dipasangkan)':>19} |   "
              f"himpunan contoh beda")
    R.WINDOW = 3
    print(f"\n{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
