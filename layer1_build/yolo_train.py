"""Leave-one-ortho-out YOLO training for crown detection (dataset B).
Reports mAP50 / mAP50-95 per fold + mean+/-std (the noise band).
Windows-safe: all work under __main__ guard (YOLO dataloader spawns workers)."""
import warnings, os, json
warnings.filterwarnings("ignore")

def main():
    import numpy as np, torch
    from ultralytics import YOLO
    BASE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.join(BASE, "yolo_B")
    DEV = 0 if torch.cuda.is_available() else "cpu"
    EPOCHS = int(os.environ.get("EPOCHS", 25))
    IMGSZ = int(os.environ.get("IMGSZ", 640))
    MODEL = os.environ.get("MODEL", "yolo11n.pt")
    CACHE = os.environ.get("CACHE", "ram")
    WORKERS = int(os.environ.get("WORKERS", 8))
    FOLDS = [int(x) for x in os.environ.get("FOLDS", "0,1,2").split(",")]
    # Tag dari nama bobot, mis. "yolo11n.pt" -> "yolo11n". Dipakai untuk memberi
    # ruang nama pada direktori run DAN pada berkas hasil, supaya menjalankan
    # model kedua tidak menimpa hasil model pertama (exist_ok=True akan menimpa).
    TAG = os.path.splitext(os.path.basename(MODEL))[0]
    print(f"device={DEV} cuda={torch.cuda.is_available()} model={MODEL} tag={TAG} "
          f"epochs={EPOCHS} imgsz={IMGSZ} cache={CACHE} workers={WORKERS} folds={FOLDS}",
          flush=True)

    res = {}
    for i in FOLDS:
        yaml = os.path.join(ROOT, f"fold{i}.yaml")
        m = YOLO(MODEL)
        m.train(data=yaml, epochs=EPOCHS, imgsz=IMGSZ, device=DEV, batch=16,
                cache=CACHE, workers=WORKERS,
                project=os.path.join(BASE, "yolo_runs"), name=f"{TAG}_fold{i}",
                exist_ok=True,
                verbose=False, plots=False, seed=42)
        mt = m.val(data=yaml, device=DEV, verbose=False, plots=False)
        res[i] = dict(map50=float(mt.box.map50), map=float(mt.box.map),
                      mp=float(mt.box.mp), mr=float(mt.box.mr))
        print(f"[fold{i}] mAP50={res[i]['map50']:.3f} mAP50-95={res[i]['map']:.3f} "
              f"P={res[i]['mp']:.3f} R={res[i]['mr']:.3f}", flush=True)

    m50 = [res[i]["map50"] for i in FOLDS]; m95 = [res[i]["map"] for i in FOLDS]
    print(f"\nmAP50   mean+/-std = {np.mean(m50):.3f} +/- {np.std(m50):.3f}")
    print(f"mAP50-95 mean+/-std = {np.mean(m95):.3f} +/- {np.std(m95):.3f}")

    # Setelan ikut disimpan: perbandingan antar model hanya sah kalau epochs,
    # imgsz dan daftar lipatannya identik. compare_models.py menolak kalau beda.
    out = dict(model=MODEL, tag=TAG, epochs=EPOCHS, imgsz=IMGSZ,
               folds=FOLDS, seed=42, folds_result={str(k): v for k, v in res.items()})
    json.dump(out, open(os.path.join(BASE, f"yolo_results_{TAG}.json"), "w"), indent=2)
    # Berkas lama tetap ditulis (bentuk lama, tanpa pembungkus) demi kebiasaan lama.
    json.dump(res, open(os.path.join(BASE, "yolo_results.json"), "w"), indent=2)
    print(f"tersimpan: yolo_results_{TAG}.json", flush=True)
    print("YOLO_TRAIN DONE", flush=True)

if __name__ == "__main__":
    main()
