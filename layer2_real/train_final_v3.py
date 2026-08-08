"""Latih SATU model v3-foto final di atas SELURUH 1.200 sawit -> stgnn_v3_photo.pt

    python train_final_v3.py

Cermin `train_final.py`, tetapi untuk varian yang benar-benar bisa diberi makan
oleh satu foto drone.

MENGAPA HANYA SATU KOLOM MASUKAN.

`run_v3.py` memakai enam kolom STATE. Tiga di antaranya (`d_sympt`, `d_dead`,
`d_cens`) adalah SELISIH terhadap sensus sebelumnya - dari satu foto ketiganya
nol. Dan detektor ds_B hanya mengeluarkan Healthy/Unhealthy: ia tidak membedakan
mati dari bergejala, dan tidak tahu apa-apa soal penyensoran.

Yang benar-benar dapat diisi satu foto karena itu hanya `is_sympt`. Diukur
(5 seed x 2 lipatan, AP dalam-sensus, graf benar):

    6 kolom penuh   0,1015    butuh dua sensus
    3 kolom level   0,1001    butuh satu sensus
    1 kolom is_sympt 0,0916   yang bisa diberi foto        <- yang dilatih di sini
    tanpa graf      0,0632    garis tanpa-skill

Model satu-kolom menahan 74% sinyal graf. Alternatifnya - melatih enam kolom lalu
MENGISI NOL lima di antaranya saat inferensi - adalah persis zero-filling yang
dilarang paket ini. Model ini menerima saat inferensi persis bentuk yang ia
terima saat latih.

BATAS YANG DISIMPAN DI DALAM CHECKPOINT.

Model dilatih pada `is_sympt` Eg9PP, yaitu status S atau D yang TERVERIFIKASI
LAPANGAN. Dipakai pada foto, kolom itu diisi kelas `Unhealthy` detektor, yaitu
kesehatan tajuk generik tanpa verifikasi. Ongkos substitusi itu SUDAH DIUKUR
(`run_v3_noisy.py`): 59% sinyal bertahan, lift 1,45x -> 1,27x. Angka itu tertulis di
`scope_warning` supaya ikut ke mana pun berkas ini pergi.
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import dataset as ds            # noqa: E402
import dataset_v3 as v3         # noqa: E402
import models_real as M         # noqa: E402
import run_real as R            # noqa: E402

OUT = os.path.join(HERE, "stgnn_v3_photo.pt")
H = int(os.environ.get("H", R.PRIMARY_H))
SEED = int(os.environ.get("SEED", 0))
EPOCHS = int(os.environ.get("EPOCHS", R.EPOCHS))
COL = 0                      # is_sympt — satu-satunya kolom yang bisa diberi foto
DEVICE = R.DEVICE


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    print("melatih v3-foto pada SELURUH 1.200 sawit (tanpa held-out) ...")
    T = len(ds.census())
    X = v3.node_features_v3(np.arange(T))[:, :, [COL]]          # (T, N, 1)
    names = [v3.feature_names_v3()[COL]]
    Ft = torch.as_tensor(np.ascontiguousarray(X), device=DEVICE)

    A = np.asarray(ds.adjacency("true"), np.float32)
    scale = R.adjacency_scale(A)
    A_scaled = torch.as_tensor(A * scale, device=DEVICE)
    D = R.diffuse(Ft, A_scaled)

    tree, t_idx, y = ds.build_examples(H, np.arange(T))          # SEMUA sawit
    ii = torch.as_tensor(tree, device=DEVICE, dtype=torch.long)
    tt = torch.as_tensor(t_idx, device=DEVICE, dtype=torch.long)
    F_seq = Ft[tt, ii].unsqueeze(1)                              # window = 1
    D_seq = D[tt, ii].unsqueeze(1)
    yt = torch.as_tensor(y, device=DEVICE)
    print("  contoh: %d  |  positif: %d (%.2f%%)  |  fitur: %s"
          % (len(y), int(y.sum()), 100 * y.mean(), names))

    torch.manual_seed(1234 + SEED)
    model = M.build("STGNN", 1, horizon=H).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=R.LR, weight_decay=R.WD)
    model.train()
    for ep in range(EPOCHS):
        opt.zero_grad()
        loss = R._focal(model(F_seq, D_seq), yt)
        loss.backward()
        opt.step()
        if (ep + 1) % 20 == 0:
            print("  epoch %2d  loss %.5f" % (ep + 1, float(loss.detach())))

    model.eval()
    with torch.no_grad():
        probe_logits = model(F_seq[:64], D_seq[:64]).cpu().numpy()

    nodes, _, _ = ds.load()
    ck = {
        "format_version": "sawitguard-l2-v3-photo/1",
        "scope_warning": (
            "Dilatih pada SELURUH 1.200 sawit tanpa kumpulan uji tersendiri. Ini "
            "artefak INFERENSI, bukan artefak evaluasi: TIDAK ADA angka performa yang "
            "boleh dikutip darinya. Angka performa hanya dari run_v3.py "
            "(leave-one-parcel-out, AP dalam-sensus). Keluarannya SKOR RELATIF untuk "
            "memeringkat, bukan peluang terkalibrasi -- jangan pernah menyajikan "
            "sigmoid(logit) sebagai '% kemungkinan sakit'. "
            "SUBSTITUSI YANG BELUM DIUKUR: kolom is_sympt di sini dilatih pada status "
            "S/D Eg9PP yang TERVERIFIKASI LAPANGAN. Dipakai pada citra, kolom itu diisi "
            "kelas 'Unhealthy' detektor, yaitu kesehatan tajuk generik tanpa verifikasi. "
            "Ongkos substitusi itu SUDAH DIUKUR (run_v3_noisy.py): pada laju detektor ds_B "
            "(recall 0,446 fpr 0,0094) AP dalam-sensus turun 0,0916 -> 0,0800, lift 1,45x -> "
            "1,27x, 59% sinyal bertahan. Laju itu diukur di ds_B, kebun BERBEDA dari Eg9PP. "
            "Efek graf varian v3 juga mengandung 36% kontaminasi kekerabatan "
            "(null dalam-famili+petak, 200 permutasi) -- lihat INTERFACE.md."),
        "model_class": "STGNN",
        "arch": {"in_dim": 1, "hidden": int(getattr(model, "hidden", 34)), "n_rel": 1},
        "state_dict": model.state_dict(),
        "task": {"horizon": H, "window": 1, "risk_status": "A", "pos_status": ["S", "D"],
                 "features": names,
                 "output": "skor relatif untuk peringkat; BUKAN peluang terkalibrasi"},
        "train": {"epochs": EPOCHS, "lr": R.LR, "weight_decay": R.WD, "seed": SEED,
                  "n_examples": int(len(y)), "n_pos": int(y.sum())},
        "data": {"palm_ids": nodes.palm_id.tolist(),
                 "n_palms": int(len(nodes)),
                 "adjacency_scale": float(scale),
                 "feature_sha256": sha(X),
                 "adjacency_sha256": sha(A)},
        "probe": {"n": 64, "logits": probe_logits.tolist()},
    }
    torch.save(ck, OUT)
    print("\nditulis: %s  (%.1f KB)" % (OUT, os.path.getsize(OUT) / 1024))
    print("arch: in_dim=1 hidden=%d n_rel=1  |  parameter: %d"
          % (ck["arch"]["hidden"], sum(p.numel() for p in model.parameters())))
    print("\n" + ck["scope_warning"][:120] + " ...")


if __name__ == "__main__":
    main()
