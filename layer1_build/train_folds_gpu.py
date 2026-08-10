"""Latih lipatan yang bobotnya belum ada, di GPU, dengan setelan IDENTIK fold1.

    python train_folds_gpu.py            # default: fold0 dan fold2
    FOLDS=all python train_folds_gpu.py  # ketiganya (lihat catatan homogenitas)

Skrip ini TIDAK berisi logika latih sendiri. Ia memanggil `y12.train_arm()` apa
adanya, supaya angka yang keluar sebanding dengan fold1 yang sudah ada.

TIGA JEBAKAN YANG DITANGANI DI SINI, dan alasannya.

1. `train_arm()` MELANJUTKAN hanya bila `setting` di berkas hasil sama persis.
   `_setting()` memasukkan DAFTAR LIPATAN. Jadi memanggilnya dengan
   folds=['fold0','fold2'] menghasilkan setting yang berbeda dari yang tersimpan
   (['fold0','fold1','fold2']), `prev` jadi kosong, dan `json.dump` di akhir
   MENIMPA berkas - catatan fold1 hilang.
   Karena itu skrip selalu meneruskan KETIGA lipatan, lalu menghapus entri
   lipatan yang memang ingin dilatih ulang. Itu satu-satunya cara memakai
   mekanisme resume `train_arm` tanpa merusak apa pun.

2. Bobot fold0 TIDAK ADA di disk meski hasilnya tercatat di JSON (dilatih di
   mesin lain). Tanpa penghapusan entri di (1), `train_arm` akan melewatinya
   dengan pesan "sudah ada" dan kita tetap tidak punya bobotnya.

3. GTX 1060 adalah Pascal (sm_61). CUDA 13.0 MEMBUANG Pascal, jadi torch
   +cu130 akan gagal dengan "no kernel image is available for execution on the
   device" - bukan saat impor, melainkan saat kernel pertama dijalankan.
   `preflight()` menjalankan satu matmul di GPU justru untuk memicu galat itu
   lebih awal, bukan 20 menit di tengah pelatihan.

CATATAN HOMOGENITAS. fold1 dilatih di mesin lain dengan ultralytics 8.4.96.
Melatih fold0/fold2 di sini (8.4.116, GPU berbeda) membuat rata-rata 3-ortho
menjadi campuran dua lingkungan. Untuk angka naskah, `FOLDS=all` lebih bersih:
ketiganya satu lingkungan. Biayanya satu lari tambahan.
"""
import glob
import json
import os
import shutil
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import y12  # noqa: E402

ARM = "base"
MODEL = "yolo12n.pt"
EPOCHS = int(os.environ.get("EPOCHS", 30))
IMGSZ = int(os.environ.get("IMGSZ", 640))
SEED = int(os.environ.get("SEED", 42))
BATCH = int(os.environ["BATCH"]) if os.environ.get("BATCH") else None
CACHE = os.environ.get("CACHE", "ram")
# workers=0 DISENGAJA: fold1 dilatih dengan 0. Menaikkannya mengubah aliran RNG
# augmentasi, jadi angkanya tidak lagi sebanding lipatan-per-lipatan.
WORKERS = int(os.environ.get("WORKERS", 0))
ALL_FOLDS = ["fold0", "fold1", "fold2"]
TAG = "yolo12n_" + ARM


def want_folds():
    v = os.environ.get("FOLDS", "0,2").strip().lower()
    if v == "all":
        return list(ALL_FOLDS)
    return [f"fold{int(x)}" for x in v.split(",") if x.strip() != ""]


def preflight():
    """Gagal SEKARANG kalau lingkungannya salah, bukan 20 menit lagi."""
    import torch

    print("== preflight ==")
    print("  torch %s | cuda build %s" % (torch.__version__, torch.version.cuda))
    # device_count() ikut diperiksa: dengan CUDA_VISIBLE_DEVICES="" is_available()
    # tetap True sementara jumlah perangkatnya nol, dan preflight ini akan lolos
    # hanya untuk gagal 20 menit kemudian - persis yang ingin dicegahnya.
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise SystemExit(
            "torch tidak melihat GPU (is_available=%s, device_count=%d).\n"
            "  Kalau device_count 0 tapi is_available True, cek CUDA_VISIBLE_DEVICES.\n"
            "  Kalau keduanya kosong, kemungkinan terpasang build CPU-only:\n"
            "  GTX 1060 = Pascal (sm_61); CUDA 13 sudah membuang Pascal, jadi pakai cu126:\n"
            '    pip install --index-url https://download.pytorch.org/whl/cu126 \\\n'
            '        "torch==2.13.0+cu126" "torchvision==0.28.0+cu126"'
            % (torch.cuda.is_available(), torch.cuda.device_count()))
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    cc = torch.cuda.get_device_capability(0)
    print("  gpu    : %s  (sm_%d%d, %.1f GB)" % (name, cc[0], cc[1], vram))
    print("  arch   : %s" % ", ".join(torch.cuda.get_arch_list()))
    try:
        x = torch.randn(512, 512, device="cuda")
        float((x @ x).sum())          # memicu kernel sungguhan
        torch.cuda.synchronize()
    except Exception as e:
        raise SystemExit(
            "GPU terlihat tetapi kernel gagal jalan:\n  %s\n"
            "Ini gejala build torch tanpa sm_%d%d. Pasang varian cu126." % (e, cc[0], cc[1]))
    print("  kernel : OK")
    if vram < 5.5 and BATCH is None:
        print("  CATATAN: VRAM < 5,5 GB. Kalau OOM, ulangi dengan BATCH=8.")

    if not glob.glob(os.path.join(y12.ROOT, "fold*.yaml")):
        print("  yolo12/ belum ada -> build(extra_mode='ignore')")
        y12.build(extra_mode="ignore", verbose=True)
    else:
        print("  data   : yolo12/ siap")


def prepare_results(folds_to_train):
    """Cadangkan JSON hasil, lalu hapus entri lipatan yang akan dilatih ulang.

    Mengembalikan nilai lama entri yang dihapus, supaya bisa dibandingkan setelah
    pelatihan - angka dari mesin lain tidak boleh diam-diam tergantikan.
    """
    p = y12._res_path(TAG)
    if not os.path.isfile(p):
        print("  %s belum ada - semua lipatan akan dilatih dari nol" % os.path.basename(p))
        return {}

    old = json.load(open(p))
    expect = y12._setting(MODEL, EPOCHS, IMGSZ, ALL_FOLDS, [SEED])
    if old.get("setting") != expect:
        raise SystemExit(
            "SETTING TIDAK COCOK - dihentikan supaya tidak menimpa hasil lama.\n"
            "  tersimpan: %s\n  diminta  : %s\n"
            "Kalau memang ingin setelan berbeda, pakai tag_suffix (lihat train_arm)."
            % (old["setting"], expect))

    bak = p.replace(".json", ".bak-%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    shutil.copy(p, bak)
    print("  cadangan: %s" % os.path.basename(bak))

    dropped = {}
    for f in folds_to_train:
        k = "%s|%d" % (f, SEED)
        if k in old["runs"]:
            dropped[k] = old["runs"].pop(k)
            print("  entri %s dihapus supaya dilatih ulang (nilai lama disimpan)" % k)
    json.dump(old, open(p, "w"), indent=2)
    return dropped


def main():
    folds = want_folds()
    print("akan dilatih : %s" % ", ".join(folds))
    print("setelan      : %s, %d epoch, imgsz %d, seed %d, cache=%s, workers=%d"
          % (MODEL, EPOCHS, IMGSZ, SEED, CACHE, WORKERS))
    preflight()

    dropped = prepare_results(folds)

    t0 = time.time()
    # SELALU ketiga lipatan: menjaga `setting` cocok sehingga resume bekerja dan
    # lipatan yang tidak diminta hanya dilewati, bukan dihapus.
    out = y12.train_arm(ARM, model=MODEL, folds=ALL_FOLDS, seeds=(SEED,),
                        epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, workers=WORKERS,
                        cache=CACHE, resume_ok=True, verbose=True)
    print("\nlatih selesai dalam %.1f menit" % ((time.time() - t0) / 60))

    print("\n== hasil deteksi (metrik sekunder - ingat langit-langit label) ==")
    for f in ALL_FOLDS:
        k = "%s|%d" % (f, SEED)
        r = out["runs"].get(k)
        if not r:
            print("  %-6s belum ada" % f)
            continue
        line = ("  %-6s mAP50=%.3f mAP50-95=%.3f  Healthy=%.3f Unhealthy=%.3f"
                % (f, r["map50"], r["map"], r["ap50"].get("Healthy", float("nan")),
                   r["ap50"].get("Unhealthy", float("nan"))))
        if k in dropped:
            line += "   (sebelumnya mAP50=%.3f di mesin lain)" % dropped[k]["map50"]
        print(line)

    print("\nBerikutnya: python centre_eval_folds.py   <- metrik UTAMA Tahap 1")


if __name__ == "__main__":
    main()
