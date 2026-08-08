"""Backend demo SawitGuard: Starlette + uvicorn, tanpa dependensi baru.

    python demo_api.py            # http://localhost:8000

Kenapa Starlette dan bukan FastAPI: `starlette`, `uvicorn`, dan `python-multipart`
sudah ikut terpasang bersama Streamlit, jadi frontend ini berjalan tanpa menambah
satu paket pun. React dan Babel di-vendor lokal di `web/vendor/`, sehingga demo
tidak membutuhkan internet sama sekali saat dijalankan di lokasi lomba.

Seluruh perhitungan tetap di `demo_core.py`. Berkas ini hanya lapisan HTTP; ia tidak
boleh menghitung apa pun sendiri, supaya versi web dan versi Streamlit tidak mungkin
memberi angka berbeda.
"""
import base64
import io
import json
import os
import tempfile

import numpy as np
from starlette.applications import Starlette
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles

import demo_core as core

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")
_SAMPLES = None


def samples():
    global _SAMPLES
    if _SAMPLES is None:
        _SAMPLES = core.sample_images(8)
    return _SAMPLES


def _thumb(path, px=220):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((px, px))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def _img_data_url(path, px=900):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((px, px))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=88)
    return ("data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode(),
            im.size)


_SICK_TILES = None


def sick_tiles():
    """Nama ubin yang GT-nya memuat tajuk Unhealthy — untuk melabeli contoh."""
    global _SICK_TILES
    if _SICK_TILES is None:
        import csv as _csv
        p = os.path.join(ROOT, "data_clean", "layer1_crowns.csv")
        s = set()
        if os.path.isfile(p):
            for r in _csv.DictReader(open(p, encoding="utf-8")):
                if r["label"] == "Unhealthy":
                    s.add(r["tile"])
        _SICK_TILES = s
    return _SICK_TILES


async def api_samples(request):
    out = []
    for i, p in enumerate(samples()):
        b = os.path.basename(p)
        ortho = b.split("_")[0] + "_" + b.split("_")[1]
        out.append({"id": i, "name": b[:18] + "…", "thumb": _thumb(p),
                    "label": ("Ortho %s · gejala" % ortho[:5]) if b in sick_tiles()
                             else ("Ortho %s" % ortho[:5])})
    return JSONResponse({"samples": out, "facts": core.FACTS})


async def api_analyze(request):
    """Satu foto -> deteksi + graf + peringkat risiko v3. Semua dari demo_core."""
    form = await request.form()
    if "file" in form and hasattr(form["file"], "read"):
        up = form["file"]
        path = os.path.join(tempfile.mkdtemp(prefix="sg_"), up.filename or "up.jpg")
        with open(path, "wb") as fh:
            fh.write(await up.read())
    else:
        path = samples()[int(form.get("sample", 0))]

    df, info = core.detect_image(path)
    res = core.score_photo(df, info, mode="biner") if info.get("ok_n") else None
    res_soft = core.score_photo(df, info, mode="kontinu") if info.get("ok_n") else None
    url, (W, Hh) = _img_data_url(path)

    # Koordinat dinormalisasi ke [0,1] supaya frontend bebas menskalakan.
    from PIL import Image
    ow, oh = Image.open(path).size
    def nx(v): return float(v) / ow
    def ny(v): return float(v) / oh

    crowns = [{"x": nx(r.cx), "y": ny(r.cy), "cls": r.cls, "conf": float(r.conf),
               "deg": None if not np.isfinite(r.deg) else int(r.deg)}
              for r in df.itertuples()]

    edges = []
    if info.get("ok_n"):
        xy = df[["cx", "cy"]].values
        for i, j in core.edges_within(xy, info["r_graph_px"]):
            edges.append([nx(xy[i, 0]), ny(xy[i, 1]), nx(xy[j, 0]), ny(xy[j, 1])])

    def pack(res):
        if res is None:
            return None
        t = res["tabel"]
        _lo = float(t.skor.min())
        _rng = max(1e-9, float(t.skor.max()) - _lo)
        return {
            "degenerate": bool(res["degenerate"]),
            "n_risk": int(res["n_risk"]), "n_sumber": int(res["n_sumber"]),
            "scope_warning": res["scope_warning"], "ckpt": res["ckpt"],
            # `v` = skor dinormalisasi ke [0,1]. Mode kontinu memakainya untuk
            # INTERPOLASI warna; mode biner tetap memakai pita diskret `q`.
            # Aturan "beda kecerahan >= 0,06" hanya berlaku untuk kategori
            # berurutan yang dibandingkan satu sama lain - untuk besaran kontinu
            # yang dibaca lewat colourbar, rentang penuh justru yang benar.
            "points": [{"x": nx(r.cx), "y": ny(r.cy), "q": int(r.kuintil),
                        "v": float((r.skor - _lo) / _rng),
                        "skor": float(r.skor), "nb": int(r.tetangga),
                        "nb_sick": int(r.tetangga_sakit), "rank": int(r.peringkat)}
                       for r in t.itertuples()],
            "sources": [{"x": nx(p[0]), "y": ny(p[1])} for p in res.get("sumber_xy", [])],
            "top10": [{"rank": int(r.peringkat), "skor": round(float(r.skor), 4),
                       "nb": int(r.tetangga), "nb_sick": int(r.tetangga_sakit),
                       "q": int(r.kuintil)} for r in t.head(10).itertuples()],
            "nb_sick_top5": float(t.head(5).tetangga_sakit.mean()),
            "nb_sick_all": float(t.tetangga_sakit.mean()),
            "n_tingkat": int(res["n_tingkat"]),
            "mode": res.get("mode", "biner"),
            "bands": [{"q": int(q), "n": int(g.shape[0]),
                       "skor": round(float(g.skor.iloc[0]), 4),
                       "nb_sick": int(g.tetangga_sakit.iloc[0])}
                      for q, g in t.groupby("kuintil")],
        }

    risk = pack(res)
    risk_soft = pack(res_soft)

    return JSONResponse({
        "image": url, "w": W, "h": Hh, "name": os.path.basename(path),
        "detect": {"n": int(info.get("n", 0)),
                   "n_sympt": int(info.get("n_sympt", 0)),
                   "spacing_px": None if not info.get("ok_n") else round(info["spacing_px"], 1),
                   "scale_ratio": None if not info.get("ok_n") else round(info["scale_ratio"], 2),
                   "ok_scale": bool(info.get("ok_scale")),
                   "ok_n": bool(info.get("ok_n")),
                   "deg_all": None if not info.get("ok_n") else round(info["deg_all"], 2),
                   "deg_inner": None if not info.get("ok_n") else round(info["deg_inner"], 2),
                   "n_inner": int(info.get("n_inner", 0)),
                   "weights": info.get("weights", ""), "conf": info.get("conf")},
        "edges": edges, "crowns": crowns, "risk": risk, "risk_soft": risk_soft,
        "foci": (lambda f: None if f is None else dict(
            f, fokus=[dict(k, cx=nx(k["cx"]), cy=ny(k["cy"]),
                           sakit_xy=[[nx(a), ny(b)] for a, b in k["sakit_xy"]])
                      for k in f["fokus"]]))(core.outbreak_foci(df, info)),
        "readiness": core.readiness(info),
    })


_EG = None


async def api_eg9pp(request):
    """Layar bukti: kisi Eg9PP + angka yang divalidasi di sana. Di-cache."""
    global _EG
    if _EG is None:
        _EG = core.eg9pp_payload()
    return JSONResponse(_EG)


class NoCacheStatic(StaticFiles):
    """Larang browser menyimpan app.jsx/styles.css.

    Berkas ini disunting berkali-kali dalam satu sesi. Tanpa header ini, browser
    menyajikan salinan lama dan pengguna melaporkan bug yang SUDAH diperbaiki -
    persis yang terjadi dengan teks "skor 0,51". Untuk demo lokal, tidak ada
    ongkosnya; berkasnya cuma puluhan KB dan berasal dari disk.
    """

    def file_response(self, *a, **k):
        r = super().file_response(*a, **k)
        r.headers["Cache-Control"] = "no-store, must-revalidate"
        return r


async def index(request):
    return FileResponse(os.path.join(WEB, "index.html"),
                        headers={"Cache-Control": "no-store, must-revalidate"})


app = Starlette(routes=[
    Route("/", index),
    Route("/api/samples", api_samples),
    Route("/api/analyze", api_analyze, methods=["POST"]),
    Route("/api/eg9pp", api_eg9pp),
    Mount("/web", NoCacheStatic(directory=WEB), name="web"),
])


if __name__ == "__main__":
    import uvicorn
    print("SawitGuard  ->  http://localhost:8000")
    print("  React + Babel di-vendor lokal; demo ini tidak butuh internet.")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
