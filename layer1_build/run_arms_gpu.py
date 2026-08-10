"""Latih dua lengan kandidat, nilai dengan harness yang sama, bandingkan berpasangan.

    python run_arms_gpu.py                 # keduanya, 3 lipatan  (~3,5 jam)
    ARMS=i1024 python run_arms_gpu.py      # satu saja
    EPOCHS=1 FOLDS=0 ARMS=smoke python run_arms_gpu.py    # uji pipa, ~2 menit

DUA HIPOTESIS YANG DIUJI, dan kenapa hanya dua.

`i1024` - imgsz 640 -> 1024.
    Ubinnya 1024x1024 tetapi latih DAN inferensi berjalan di 640, jadi tiap ubin
    diperkecil 0,625x dan tajuk ~105 px menyusut ke ~66 px. Pembeda kelas Unhealthy
    adalah warna dan tekstur pelepah, dan itu yang paling dulu hilang saat gambar
    diperkecil. Bukti pendukungnya konkret: pada ubin 44000_16000_2242_2574 sebuah
    sawit yang dilabeli sakit mendapat Healthy 0,55 lawan Unhealthy 0,43 - keputusan
    setipis itu bisa berbalik kalau detailnya tidak dibuang lebih dulu.

`nadir_nohsv` - matikan jitter HSV.
    Alasannya sudah tertulis di `y12.ARMS` sejak awal: fitur teratas Tahap 2 adalah
    (G-R), exg_std, (G-B) - yaitu greenness. Augmentasi HSV bawaan mengacak rona dan
    saturasi tiap epoch, sehingga model justru diajari MENGABAIKAN pembeda kelasnya.

Lengan lain (`nadir`, `nadir_geom`, `nadir_strong`) sengaja tidak diikutkan: tak satu
pun punya hipotesis yang mengarah ke kelemahan yang sedang kita lihat, dan tiap lengan
memakan ~1 jam GPU.

TIGA HAL YANG DIJAGA SKRIP INI.

1. RUANG NAMA TERPISAH. `base` @1024 memakai `tag_suffix="i1024"`, sehingga hasil
   garis dasar @640 tidak tertimpa. Tanpa itu `train_arm` menganggap keduanya lari
   yang sama dan menimpa satu sama lain - lihat docstring `train_arm`.

2. BASELINE TIDAK DISENTUH. Penilaian memanggil `centre_eval_folds.py` sebagai
   subproses dengan TAG env, dan berkas itu kini menulis `centre_eval_<TAG>.json`
   untuk lengan non-baseline. `centre_eval.json` - yang dibaca demo - tetap milik
   garis dasar.

3. PUTUSANNYA ATURAN YANG SAMA. Selisih dilaporkan mean +/- std atas 3 lipatan
   berpasangan plus sign-count, dan |mean| < 1 std dinyatakan INCONCLUSIVE. n=3,
   jadi pita deraunya lebar dan sebagian besar selisih MEMANG akan jatuh tidak
   konklusif. Itu jawaban yang benar, bukan kegagalan skrip.
"""
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import y12  # noqa: E402

SEED = int(os.environ.get("SEED", 42))
EPOCHS = int(os.environ.get("EPOCHS", 30))
MODEL = os.environ.get("MODEL", "yolo12n.pt")
CACHE = os.environ.get("CACHE", "ram")
WORKERS = int(os.environ.get("WORKERS", 0))
BATCH = int(os.environ["BATCH"]) if os.environ.get("BATCH") else None
FOLDS = ["fold%d" % int(x) for x in os.environ.get("FOLDS", "0,1,2").split(",")]

BASE_TAG = "yolo12n_base"

EXP = {
    "i1024": dict(arm="base", imgsz=1024, suffix="i1024",
                  why="imgsz 640 -> 1024; detail tajuk tidak dibuang sebelum dilihat"),
    "nohsv": dict(arm="nadir_nohsv", imgsz=640, suffix="",
                  why="matikan jitter HSV; warna adalah pembeda kelas Unhealthy"),
    # hanya untuk menguji pipa; JANGAN dipakai sebagai hasil
    "smoke": dict(arm="base", imgsz=640, suffix="smoke",
                  why="uji pipa saja"),
}
WANT = [k.strip() for k in os.environ.get("ARMS", "i1024,nohsv").split(",") if k.strip()]


def tag_of(e):
    mtag = os.path.splitext(os.path.basename(MODEL))[0]
    return "%s_%s%s" % (mtag, e["arm"], ("_" + e["suffix"]) if e["suffix"] else "")


def preflight():
    """Dipinjam dari train_folds_gpu: gagal SEKARANG, bukan satu jam lagi."""
    import torch

    print("== preflight ==")
    print("  torch %s | cuda build %s" % (torch.__version__, torch.version.cuda))
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise SystemExit("torch tidak melihat GPU (is_available=%s, count=%d)."
                         % (torch.cuda.is_available(), torch.cuda.device_count()))
    cc = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print("  gpu    : %s (sm_%d%d, %.1f GB)"
          % (torch.cuda.get_device_name(0), cc[0], cc[1], vram))
    try:
        x = torch.randn(512, 512, device="cuda")
        float((x @ x).sum())
        torch.cuda.synchronize()
    except Exception as e:
        raise SystemExit("GPU terlihat tapi kernel gagal: %s\nPasang torch cu126." % e)
    print("  kernel : OK")
    if vram < 5.5 and any(EXP[k]["imgsz"] > 640 for k in WANT) and BATCH is None:
        print("  CATATAN: VRAM < 5,5 GB dan ada lengan @1024. Kalau OOM, ulangi BATCH=4.")
    import glob
    if not glob.glob(os.path.join(y12.ROOT, "fold*.yaml")):
        print("  yolo12/ belum ada -> build(extra_mode='ignore')")
        y12.build(extra_mode="ignore", verbose=True)
    else:
        print("  data   : yolo12/ siap")


def centre_json(tag):
    return os.path.join(y12.RESDIR, "centre_eval.json" if tag == BASE_TAG
                        else "centre_eval_%s.json" % tag)


def evaluate(tag, imgsz):
    """Panggil centre_eval_folds.py apa adanya. Jangan pernah menyalin logikanya:
    pemilihan ambang silang-lipatan ada di sana, dan menduplikasinya adalah cara
    termudah membuat dua angka yang tidak sebanding."""
    env = dict(os.environ, TAG=tag, IMGSZ=str(imgsz), SEED=str(SEED),
               FOLDS=",".join(f[-1] for f in FOLDS))
    print("\n-- centre_eval_folds.py  TAG=%s IMGSZ=%d --" % (tag, imgsz), flush=True)
    r = subprocess.run([sys.executable, os.path.join(BASE, "centre_eval_folds.py")],
                       env=env, cwd=BASE)
    if r.returncode != 0:
        print("  PERINGATAN: centre_eval gagal (exit %d) untuk %s" % (r.returncode, tag))
        return None
    p = centre_json(tag)
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None


def f1_per_fold(js):
    """-> {fold: f1} dari keluaran centre_eval_folds."""
    if not js:
        return {}
    out = {}
    for k, v in (js.get("folds") or {}).items():
        f1 = v.get("f1") if isinstance(v, dict) else None
        if f1 is not None:
            out[k] = float(f1)
    return out


def paired(a, b, label):
    """Aturan putusan yang sama dengan seluruh paket: mean +/- std + sign-count."""
    import numpy as np

    keys = sorted(set(a) & set(b))
    if not keys:
        print("  %-34s (tidak ada lipatan yang berpasangan)" % label)
        return
    d = np.array([a[k] - b[k] for k in keys], float)
    m, s = float(d.mean()), float(d.std(ddof=0))
    pos = int((d > 0).sum())
    vonis = "INCONCLUSIVE" if abs(m) < s or len(d) < 2 else ("POS" if m > 0 else "NEG")
    print("  %-34s %+.4f +/- %.4f  %d/%d  %s"
          % (label, m, s, pos, len(d), vonis))
    for k in keys:
        print("        %-7s %.4f vs %.4f  (%+.4f)" % (k, a[k], b[k], a[k] - b[k]))


def main():
    print("lengan  : %s" % ", ".join(WANT))
    print("lipatan : %s | %d epoch | seed %d" % (", ".join(FOLDS), EPOCHS, SEED))
    for k in WANT:
        if k not in EXP:
            raise SystemExit("lengan tidak dikenal: %s (pilihan: %s)"
                             % (k, ", ".join(EXP)))
        e = EXP[k]
        print("   %-7s arm=%-12s imgsz=%-5d tag=%-24s %s"
              % (k, e["arm"], e["imgsz"], tag_of(e), e["why"]))
    est = sum(len(FOLDS) * (18 if EXP[k]["imgsz"] <= 640 else 45) for k in WANT)
    print("perkiraan waktu latih: ~%.1f jam (kasar, %d epoch)\n" % (est / 60.0, EPOCHS))

    preflight()

    results = {}
    for k in WANT:
        e = EXP[k]
        tag = tag_of(e)
        t0 = time.time()
        print("\n" + "=" * 72)
        print("== LENGAN %s -> %s ==" % (k, tag))
        print("=" * 72, flush=True)
        y12.train_arm(e["arm"], model=MODEL, folds=FOLDS, seeds=(SEED,),
                      epochs=EPOCHS, imgsz=e["imgsz"], batch=BATCH, workers=WORKERS,
                      cache=CACHE, resume_ok=True, tag_suffix=e["suffix"],
                      verbose=True)
        print("  latih %s selesai dalam %.1f menit" % (k, (time.time() - t0) / 60))
        results[k] = f1_per_fold(evaluate(tag, e["imgsz"]))

    print("\n" + "=" * 72)
    print("== PUTUSAN: F1 pusat tajuk (METRIK UTAMA), berpasangan per lipatan ==")
    print("=" * 72)
    base = f1_per_fold(json.load(open(centre_json(BASE_TAG), encoding="utf-8"))
                       if os.path.isfile(centre_json(BASE_TAG)) else None)
    if not base:
        print("  centre_eval.json garis dasar tidak ada - jalankan dulu:")
        print("     python centre_eval_folds.py")
        return
    print("  garis dasar (%s): %s\n"
          % (BASE_TAG, ", ".join("%s=%.4f" % kv for kv in sorted(base.items()))))
    for k in WANT:
        paired(results.get(k, {}), base, "%s - base" % k)
    print("\n  |mean| < 1 std => INCONCLUSIVE, dan dengan n=%d lipatan itu sering"
          " terjadi.\n  Jangan mengklaim perbaikan yang jatuh di dalam pita derau."
          % len(FOLDS))


if __name__ == "__main__":
    main()
