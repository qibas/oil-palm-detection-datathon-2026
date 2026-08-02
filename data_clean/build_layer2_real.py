"""Lapisan 2 (data NYATA) — pembekuan panel epidemi Eg9PP.

Sumber: Tisne et al. 2017, G3 7(6):1683-1692, doi:10.1534/g3.117.041764.
1.200 sawit di kebun SOCFINDO Medan, disensus selama 25 tahun. Format aslinya
adalah data survival (waktu-kejadian + indikator sensor), BUKAN panel; skrip ini
mengubahnya jadi panel pohon x sensus yang bisa dimakan pipeline forecasting.

Keluaran:
  layer2_nodes.csv   1.200 baris — pohon, genotipe, koordinat terkoreksi, fold
  layer2_panel.csv   pohon x sensus (long) — status, risk set, ringkasan tetangga
  layer2_edges.csv   daftar sisi graf kontak akar pada r = 1,5 x jarak tanam

Label horizon SENGAJA tidak dihitung di sini: HORIZONS tetap satu sumber di hilir.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "Eg9PP_Phenotypes.csv")

# Sumbu X dan Y TIDAK sekala pada data asli. Sawit ditanam segitiga sama sisi;
# mengalikan X dengan cos30 membuat keenam tetangga terdekat jatuh tepat di 1,000.
COS30 = np.cos(np.pi / 6)
R_CONTACT = 1.5          # kelipatan jarak tanam — dataran kurva derajat (lihat uji jembatan)
OFFGRID = {12.2: 12.0}   # satu tanggal sensus di luar grid; dirapatkan ke tanggal terdekat


def load():
    """Pemisah file ini non-standar — sep=None + engine python WAJIB."""
    e = pd.read_csv(SRC, sep=None, engine="python")
    e = e.rename(columns=str.upper)
    n_snap = int(e[["Y_T1S", "Y_TD"]].isin(list(OFFGRID)).any(axis=1).sum())
    for c in ("Y_T1S", "Y_TD"):
        e[c] = e[c].replace(OFFGRID)
    e["xm"] = e.X_POSITION.astype(float) * COS30
    e["ym"] = e.Y_POSITION.astype(float)
    return e, n_snap


def status_of(t, ev1, y1, evd, yd):
    """Status satu pohon pada tanggal sensus t.

    A  asimptomatik, masih diamati        -> masuk risk set
    S  simptomatik, masih hidup
    D  mati
    C  keluar pengamatan (tersensor)      -> TIDAK boleh dianggap sehat
    """
    if ev1 == 0:                       # tak pernah bergejala selama diamati
        return "A" if t < y1 else "C"  # y1 = waktu SENSOR, bukan waktu kejadian
    if t < y1:
        return "A"
    if evd == 1:
        return "D" if t >= yd else "S"
    return "S" if t <= yd else "C"     # simptomatik lalu tersensor sebelum mati


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    e, snapped = load()

    # ---- grid sensus = tanggal yang BENAR-BENAR teramati, bukan langkah 0,5 th sintetis
    census = np.array(sorted(set(e.Y_T1S) | set(e.Y_TD)))
    assert len(census) == 45, len(census)

    # ---- geometri: verifikasi kisi segitiga per parcel -----------------------
    for p, s in e.groupby("PARCEL"):
        q = s[["xm", "ym"]].to_numpy()
        d, _ = cKDTree(q).query(q, k=7)
        nn6 = np.median(d[:, 1:7], axis=0)
        assert np.allclose(nn6, 1.0, atol=0.01), (p, nn6)

    xy = e[["xm", "ym"]].to_numpy()
    par = e.PARCEL.to_numpy()
    pairs = cKDTree(xy).query_pairs(R_CONTACT, output_type="ndarray")
    cross = int((par[pairs[:, 0]] != par[pairs[:, 1]]).sum())
    # Sifat inilah yang membuat leave-one-parcel-out sah: memisahkan fold tidak
    # memutus satu sisi pun, jadi tak ada informasi tetangga yang bocor lintas-fold.
    assert cross == 0, cross

    nodes = e.rename(columns={
        "ID": "palm_id", "PROGENY": "progeny", "GA_PARENT": "ga_parent",
        "GB_PARENT": "gb_parent", "PARCEL": "parcel", "PLOT": "plot",
        "EVENT_T1S": "event_t1s", "Y_T1S": "y_t1s",
        "EVENT_TD": "event_td", "Y_TD": "y_td"})
    nodes["fold"] = nodes.parcel
    nodes = nodes[["palm_id", "progeny", "ga_parent", "gb_parent", "parcel", "plot",
                   "xm", "ym", "event_t1s", "y_t1s", "event_td", "y_td", "fold"]]
    nodes.to_csv(os.path.join(BASE, "layer2_nodes.csv"), index=False)

    ids = nodes.palm_id.to_numpy()
    edges = pd.DataFrame({"a": ids[pairs[:, 0]], "b": ids[pairs[:, 1]]})
    edges["parcel"] = par[pairs[:, 0]]
    edges.to_csv(os.path.join(BASE, "layer2_edges.csv"), index=False)

    # ---- panel --------------------------------------------------------------
    N = len(nodes)
    nb = [[] for _ in range(N)]
    for a, b in pairs:
        nb[a].append(b)
        nb[b].append(a)
    nb = [np.array(v, dtype=int) for v in nb]
    deg = np.array([len(v) for v in nb])

    ev1 = nodes.event_t1s.to_numpy()
    y1 = nodes.y_t1s.to_numpy(float)
    evd = nodes.event_td.to_numpy()
    yd = nodes.y_td.to_numpy(float)

    rows = []
    for t in census:
        st = np.array([status_of(t, ev1[i], y1[i], evd[i], yd[i]) for i in range(N)])
        sympt = (st == "S") | (st == "D")      # pernah bergejala & masih tercatat
        dead = st == "D"
        obs = st != "C"                        # status diketahui pada t
        prev = float(sympt[obs].mean()) if obs.any() else np.nan
        rows.append(pd.DataFrame({
            "palm_id": ids,
            "t": t,
            "status": st,
            "at_risk": (st == "A").astype(int),
            "deg": deg,
            "n_nb_obs": [int(obs[v].sum()) for v in nb],
            "n_nb_sympt": [int(sympt[v].sum()) for v in nb],
            "n_nb_dead": [int(dead[v].sum()) for v in nb],
            "prev_global": prev,
        }))
    panel = pd.concat(rows, ignore_index=True)
    panel.to_csv(os.path.join(BASE, "layer2_panel.csv"), index=False)

    # ---- ringkasan ----------------------------------------------------------
    print(f"[L2] sensus {len(census)} tanggal {census.min()}-{census.max()} th "
          f"(12.2 -> 12.0; grid mentah 46 -> 45) | node {N} | panel {len(panel)} baris")
    print(f"[L2] kisi: 6 NN @ 1.000 di kedua parcel | sisi {len(pairs)} | "
          f"lintas-parcel {cross} | derajat rata-rata {deg.mean():.2f}")
    print(f"[L2] risk set: {int((ev1==0).sum())} pohon tersensor T1S, "
          f"sensor terdini t={y1[ev1==0].min()}")
    print(f"[L2] kejadian: simptomatik {int((ev1==1).sum())} "
          f"({100*(ev1==1).mean():.1f}%) | mati {int((evd==1).sum())} "
          f"({100*(evd==1).mean():.1f}%)")
    for p, s in nodes.groupby("parcel"):
        print(f"       {p}: n={len(s)} plot={s['plot'].nunique()} famili={s.progeny.nunique()} "
              f"simptomatik={100*s.event_t1s.mean():.1f}%")
    print(f"       famili di KEDUA parcel: "
          f"{len(set(nodes[nodes.parcel=='44A'].progeny) & set(nodes[nodes.parcel=='44B'].progeny))}"
          f"/{nodes.progeny.nunique()} -> fold parcel tidak terkonfound genotipe")

    # ---- sanity check: pos-rate per horizon (sekadar dilaporkan, bukan diasersi) --
    piv = panel.pivot(index="palm_id", columns="t", values="status").reindex(ids)
    A = piv.to_numpy()
    print("[L2] pos-rate tugas peramalan (pada risk set asimptomatik):")
    for h in (1, 2, 3, 4):
        n_ex = n_pos = 0
        for j in range(A.shape[1] - h):
            m = A[:, j] == "A"
            fut = (A[:, j + 1:j + 1 + h] == "S") | (A[:, j + 1:j + 1 + h] == "D")
            n_ex += int(m.sum())
            n_pos += int(fut.any(1)[m].sum())
        print(f"       h={h} ({h*0.5:.1f}-ish th): {n_ex} contoh, {n_pos} positif "
              f"({100*n_pos/n_ex:.2f}%)")
    print("[L2] ditulis: layer2_nodes.csv, layer2_panel.csv, layer2_edges.csv"
          f"  (snap off-grid menyentuh {snapped} pohon)")


if __name__ == "__main__":
    main()
