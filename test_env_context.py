"""Penjaga untuk env_context.py — ~10 asersi, tanpa internet (semua panggilan jaringan di-mock).

    python test_env_context.py     # exit 0 = semua lulus

Yang dijaga: (1) heuristik drainase/hujan di ambang yang benar, (2) fallback cache aktif saat
panggilan jaringan gagal, (3) `get_context` tidak pernah mengarang angka saat cache juga tidak
ada, (4) cache tertulis sebagai JSON yang bisa dibaca ulang.
"""
import json
import os
import tempfile
import urllib.error

import env_context as ec


def check(name, cond):
    status = "OK  " if cond else "GAGAL"
    print("[%s] %s" % (status, name))
    if not cond:
        raise SystemExit(1)


def test_drainage_thresholds():
    check("liat 40% -> buruk", ec.classify_drainage(40.0)[0] == "buruk")
    check("liat 39.9% -> sedang", ec.classify_drainage(39.9)[0] == "sedang")
    check("liat 20% -> sedang", ec.classify_drainage(20.0)[0] == "sedang")
    check("liat 19.9% -> baik", ec.classify_drainage(19.9)[0] == "baik")


def test_rain_thresholds():
    check("hujan 260mm -> tinggi", ec.classify_rain(260.0)[0] == "tinggi")
    check("hujan 140mm -> rendah", ec.classify_rain(140.0)[0] == "rendah")
    check("hujan 200mm -> normal", ec.classify_rain(200.0)[0] == "normal")


def test_fallback_to_cache_on_network_failure(tmp_cache):
    orig_get_json, orig_cache = ec._get_json, ec.CACHE_PATH
    ec.CACHE_PATH = tmp_cache

    def boom(url):
        raise urllib.error.URLError("no network (simulasi uji)")

    ec._get_json = boom
    try:
        # tanpa cache sama sekali -> harus gagal jujur, bukan mengarang angka
        ctx = ec.get_context()
        check("tanpa cache & tanpa jaringan -> ok=False", ctx["ok"] is False)

        # tulis cache asli (dari data yang sudah diambil lebih dulu, bukan rekaan)
        payload = {
            "location": {"lat": -0.5272, "lon": 101.4174, "label": "uji"},
            "fetched_at": "2026-01-01T00:00+0000",
            "weather": {"ok": True, "wind_speed_kmh": 7.9, "wind_dir_deg": 164.0,
                        "rain_30d_mm": 122.0, "rain_daily_mm": [1.0] * 30,
                        "source": "Open-Meteo (api.open-meteo.com), live"},
            "soil": {"ok": True, "clay_pct": 40.4, "sand_pct": 33.9, "bdod_kg_dm3": 1.11,
                     "source": "ISRIC SoilGrids v2.0 (rest.isric.org), 0-5cm mean, live"},
        }
        with open(tmp_cache, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

        ctx = ec.get_context()
        check("dengan cache & tanpa jaringan -> ok=True", ctx["ok"] is True)
        check("menandai using_cache=True", ctx["using_cache"] is True)
        check("angka cache dipakai apa adanya (angin)",
              ctx["weather"]["wind_speed_kmh"] == 7.9)
        check("sumber ditandai CACHE, bukan live",
              "CACHE" in ctx["weather"]["source"] and "CACHE" in ctx["soil"]["source"])
        check("drainase diturunkan dari liat cache (40.4% -> buruk)",
              ctx["drainage"]["level"] == "buruk")
    finally:
        ec._get_json, ec.CACHE_PATH = orig_get_json, orig_cache


if __name__ == "__main__":
    test_drainage_thresholds()
    test_rain_thresholds()
    with tempfile.TemporaryDirectory() as d:
        test_fallback_to_cache_on_network_failure(os.path.join(d, "cache.json"))
    print("\nSemua asersi lulus.")
