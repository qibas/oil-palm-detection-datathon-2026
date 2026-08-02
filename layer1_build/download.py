import warnings, os, time
warnings.filterwarnings("ignore")
from roboflow import Roboflow

API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
assert API_KEY, "Set ROBOFLOW_API_KEY env var (get it from https://app.roboflow.com/settings/api)"
rf = Roboflow(api_key=API_KEY)

BASE = os.path.dirname(os.path.abspath(__file__))
TARGETS = {
    "A": ("gunadarma", "oil-palm-health-vglxy", 1),
    "B": ("health-detection", "oil-palm-health-detection", 2),
    "C": ("oil-palm-health-detection", "oil-palm-tree-health-detection", 1),
}

for tag, (ws, proj, ver) in TARGETS.items():
    loc = os.path.join(BASE, f"ds_{tag}")
    if os.path.isdir(loc) and os.listdir(loc):
        print(f"[{tag}] already present at {loc}, skipping")
        continue
    print(f"[{tag}] downloading {ws}/{proj} v{ver} -> {loc}")
    t0 = time.time()
    try:
        p = rf.workspace(ws).project(proj)
        p.version(ver).download("coco", location=loc, overwrite=True)
        print(f"[{tag}] done in {time.time()-t0:.0f}s")
    except Exception as e:
        print(f"[{tag}] DOWNLOAD ERR: {e}")
print("ALL DONE")
