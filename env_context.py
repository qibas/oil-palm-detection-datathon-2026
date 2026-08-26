"""Konteks lingkungan (angin · hujan · tekstur tanah) — TANPA menyentuh model.

    python env_context.py [lat] [lon]      # uji mandiri, cetak semua angka

KENAPA BERKAS INI TERPISAH DARI `demo_core.py`.

Ini BUKAN fitur model. Ganoderma menyebar lewat kontak akar (graf yang sudah
divalidasi di Lapisan 2); angin dan tekstur tanah TIDAK pernah diberikan ke
checkpoint mana pun di paket ini, dan tidak akan pernah — sebabnya struktural,
bukan malas: baik `ds_B` (Lapisan 1) maupun Eg9PP (Lapisan 2) TIDAK bergeoreferensi
(lihat `data_clean/DATASET_CARD.md` baris "batas yang dipaksakan": "tak ada data
angin ⇒ n_rel = 1"), jadi tidak ada kunci sungguhan untuk menyambung koordinat
lintang/bujur ke satu pohon pun di kedua dataset itu. Menyimulasikan nilai
per-pohon supaya terlihat terintegrasi akan persis jenis data ISENG yang paket
ini menolak di semua tempat lain.

Yang berkas ini lakukan sebagai gantinya: mengambil data ASLI dari dua sumber
publik gratis (BUKAN direka), untuk SATU titik koordinat kebun yang dimasukkan
pengguna, dan menampilkannya sebagai KONTEKS di samping peringkat risiko —
bukan sebagai masukan yang dilatih atau divalidasi terhadap Eg9PP. Panel ini
TIDAK BOLEH disebut sebagai "fitur model" di naskah atau pitch deck.

SUMBER (keduanya tanpa API key):
  - Open-Meteo, angin + hujan 30 hari terakhir  (api.open-meteo.com)
  - ISRIC SoilGrids v2.0, tekstur tanah 0-5cm    (rest.isric.org)

BATAS YANG HARUS DINYATAKAN, SELALU:
  1. Ini KONTEKS ESTATE/REGIONAL, bukan per-pohon — satu nilai untuk satu
     koordinat, sama untuk seluruh 1.200/5.077 pohon di petak itu.
  2. TIDAK PERNAH divalidasi terhadap kejadian BSR nyata di paket ini — tidak
     ada georeferensi Eg9PP atau ds_B untuk mengujinya, persis alasan Lapisan 1
     dan Lapisan 2 tidak bisa disambung (lihat modul ini, docstring atas).
  3. Ambang drainase/hujan di bawah adalah HEURISTIK tekstur-tanah standar
     (segitiga tekstur USDA), BUKAN model terlatih dan BUKAN spesifik-Ganoderma.
  4. Rujukan literatur di `CITATIONS` dicatat dari pengetahuan umum patologi
     sawit — detail bibliografis (volume/halaman) BELUM diverifikasi silang;
     periksa ulang sebelum dikutip di naskah akhir.
"""
import json
import os
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(ROOT, "env_context_cache.json")

TIMEOUT_S = 6

# Lokasi contoh generik — SABUK SAWIT Riau, Sumatra. BUKAN koordinat asli
# kebun Eg9PP atau ds_B (keduanya tidak bergeoreferensi di paket ini). Dipilih
# hanya supaya panel punya default yang masuk akal secara agronomis.
DEFAULT_LOCATION = {"lat": -0.5272, "lon": 101.4174,
                    "label": "Riau, Sumatra (contoh generik, BUKAN koordinat kebun asli)"}

CITATIONS = [
    ("Rees et al. 2009",
     "Distribusi Ganoderma boninense dan patogen busuk batang lain di kebun "
     "sawit Malaysia — drainase buruk & sisa tunggul tanaman sebelumnya "
     "sebagai faktor risiko. (detail volume/halaman belum diverifikasi ulang)"),
    ("Naher et al. 2013",
     "Status ekologis Ganoderma dan penyakit busuk pangkal batang sawit — "
     "tanah tergenang/drainase buruk sebagai kondisi yang mendukung infeksi. "
     "(detail volume/halaman belum diverifikasi ulang)"),
    ("Susanto et al. 2005",
     "Pengendalian hayati busuk pangkal batang (Ganoderma boninense) di "
     "kebun sawit — faktor lingkungan kebun sebagai penunjang penyebaran. "
     "(detail volume/halaman belum diverifikasi ulang)"),
]


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "sawitguard-demo/1"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_weather(lat, lon):
    """Angin saat ini + jumlah hujan 30 hari terakhir. -> dict, `ok=False` bila gagal."""
    url = ("https://api.open-meteo.com/v1/forecast"
           "?latitude=%s&longitude=%s"
           "&current=wind_speed_10m,wind_direction_10m"
           "&daily=precipitation_sum&past_days=30&forecast_days=1&timezone=auto"
           % (lat, lon))
    try:
        j = _get_json(url)
        daily = j["daily"]["precipitation_sum"][:-1]   # buang hari ini (belum penuh)
        return {
            "ok": True, "live": True,
            "wind_speed_kmh": float(j["current"]["wind_speed_10m"]),
            "wind_dir_deg": float(j["current"]["wind_direction_10m"]),
            "rain_30d_mm": round(float(sum(daily)), 1),
            "rain_daily_mm": daily,
            "source": "Open-Meteo (api.open-meteo.com), live",
            "fetched_at": j["current"]["time"],
        }
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, OSError) as e:
        return {"ok": False, "error": str(e)}


def fetch_soil(lat, lon):
    """Tekstur tanah 0-5cm (clay/sand %, bulk density). -> dict, `ok=False` bila gagal."""
    url = ("https://rest.isric.org/soilgrids/v2.0/properties/query"
           "?lon=%s&lat=%s&property=clay&property=sand&property=bdod&depth=0-5cm&value=mean"
           % (lon, lat))
    try:
        j = _get_json(url)
        layers = {L["name"]: L["depths"][0]["values"]["mean"] for L in j["properties"]["layers"]}
        return {
            "ok": True, "live": True,
            "clay_pct": round(layers["clay"] / 10.0, 1),
            "sand_pct": round(layers["sand"] / 10.0, 1),
            "bdod_kg_dm3": round(layers["bdod"] / 100.0, 2),
            "source": "ISRIC SoilGrids v2.0 (rest.isric.org), 0-5cm mean, live",
        }
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, OSError) as e:
        return {"ok": False, "error": str(e)}


def classify_drainage(clay_pct):
    """Heuristik tekstur-tanah STANDAR (segitiga USDA), bukan model terlatih.

    >= 40% liat  -> drainase umumnya BURUK (menahan air)
    20–40% liat  -> SEDANG
    < 20% liat   -> umumnya BAIK (tanah berpasir, cepat meloloskan air)

    Ini KETERANGAN TEKSTUR, bukan pengukuran drainase langsung (drainase
    sungguhan juga bergantung topografi & muka air tanah, yang TIDAK diukur
    di sini).
    """
    if clay_pct >= 40:
        return "buruk", ("liat %.1f%% — tanah bertekstur berat, cenderung menahan "
                          "air lebih lama" % clay_pct)
    if clay_pct >= 20:
        return "sedang", "liat %.1f%% — tekstur menengah" % clay_pct
    return "baik", ("liat %.1f%% — tanah bertekstur ringan, umumnya cepat "
                     "meloloskan air" % clay_pct)


def classify_rain(rain_30d_mm):
    """Bandingkan hujan 30 hari terhadap acuan KASAR 200mm/bulan (rata-rata umum
    daerah tropis basah). BUKAN normal klimatologis situs — situs ini tidak
    pernah diukur cukup lama untuk itu di paket ini."""
    ref = 200.0
    if rain_30d_mm >= 1.3 * ref:
        return "tinggi", "%.0fmm dalam 30 hari — di atas acuan kasar %.0fmm" % (rain_30d_mm, ref)
    if rain_30d_mm <= 0.7 * ref:
        return "rendah", "%.0fmm dalam 30 hari — di bawah acuan kasar %.0fmm" % (rain_30d_mm, ref)
    return "normal", "%.0fmm dalam 30 hari — sekitar acuan kasar %.0fmm" % (rain_30d_mm, ref)


def _load_cache():
    if os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def get_context(lat=None, lon=None):
    """Titik masuk tunggal dipakai UI. Coba data HIDUP dulu; kalau jaringan mati
    (mis. lokasi lomba tanpa internet — lihat DEMO_BRIEF.md §6), jatuh ke cache
    nyata yang sudah direkam sebelumnya. TIDAK PERNAH mengarang angka sendiri.

    -> dict: {"location", "weather", "soil", "drainage", "rain", "citations",
              "using_cache"}
    """
    loc = DEFAULT_LOCATION if lat is None or lon is None else {
        "lat": float(lat), "lon": float(lon), "label": "kebun kamu"}

    # Dijalankan BERSAMAAN, bukan berurutan: keduanya independen (angin/hujan
    # dari Open-Meteo, tanah dari SoilGrids), dan ISRIC SoilGrids terukur
    # butuh ~3,3 detik sendirian — kalau dijalankan berurutan, kasus terburuk
    # jadi dua kali TIMEOUT_S alih-alih satu.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as ex:
        fw = ex.submit(fetch_weather, loc["lat"], loc["lon"])
        fs = ex.submit(fetch_soil, loc["lat"], loc["lon"])
        w, s = fw.result(), fs.result()
    using_cache = not (w.get("ok") and s.get("ok"))

    if using_cache:
        cache = _load_cache()
        if cache is None:
            return {"ok": False,
                    "error": "Tidak ada koneksi dan belum ada cache lokal. "
                             "Jalankan sekali dengan internet untuk membuat "
                             "%s." % os.path.basename(CACHE_PATH)}
        loc = cache["location"]
        w = dict(cache["weather"], live=False,
                 source=cache["weather"]["source"].replace(", live", ", CACHE %s"
                                                             % cache["fetched_at"]))
        s = dict(cache["soil"], live=False,
                 source=cache["soil"]["source"].replace(", live", ", CACHE %s"
                                                          % cache["fetched_at"]))

    drain_level, drain_note = classify_drainage(s["clay_pct"])
    rain_level, rain_note = classify_rain(w["rain_30d_mm"])

    return {
        "ok": True, "using_cache": using_cache, "location": loc,
        "weather": w, "soil": s,
        "drainage": {"level": drain_level, "note": drain_note},
        "rain": {"level": rain_level, "note": rain_note},
        "citations": CITATIONS,
    }


def refresh_cache(lat=None, lon=None):
    """Rekam SATU snapshot data hidup ke `env_context_cache.json`, untuk dipakai
    sebagai jaring pengaman offline saat demo. Jalankan ini SEKALI, dengan
    internet, sebelum berangkat ke lokasi lomba — bukan dijalankan otomatis
    oleh UI, supaya cache tidak diam-diam menjadi lama tanpa disadari."""
    loc = DEFAULT_LOCATION if lat is None or lon is None else {
        "lat": float(lat), "lon": float(lon), "label": "kebun kamu"}
    w = fetch_weather(loc["lat"], loc["lon"])
    s = fetch_soil(loc["lat"], loc["lon"])
    if not (w.get("ok") and s.get("ok")):
        raise SystemExit("Gagal mengambil data hidup — cek koneksi. %s / %s"
                          % (w.get("error"), s.get("error")))
    payload = {"location": loc, "fetched_at": time.strftime("%Y-%m-%dT%H:%M%z"),
               "weather": w, "soil": s}
    with open(CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print("cache ditulis -> %s" % CACHE_PATH)
    return payload


if __name__ == "__main__":
    import sys

    a = sys.argv[1:]
    lat, lon = (float(a[0]), float(a[1])) if len(a) >= 2 else (None, None)

    if os.environ.get("REFRESH_CACHE"):
        refresh_cache(lat, lon)

    ctx = get_context(lat, lon)
    if not ctx["ok"]:
        print("GAGAL:", ctx["error"])
        raise SystemExit(1)

    print("== konteks lingkungan (%s) ==" % ("CACHE" if ctx["using_cache"] else "HIDUP"))
    print("  lokasi   : %s (%.4f, %.4f)"
          % (ctx["location"]["label"], ctx["location"]["lat"], ctx["location"]["lon"]))
    print("  angin    : %.1f km/h, arah %.0f°" % (ctx["weather"]["wind_speed_kmh"],
                                                   ctx["weather"]["wind_dir_deg"]))
    print("  hujan    : %s — %s" % (ctx["rain"]["level"], ctx["rain"]["note"]))
    print("  tanah    : liat %.1f%% · pasir %.1f%% · bulk density %.2f kg/dm³"
          % (ctx["soil"]["clay_pct"], ctx["soil"]["sand_pct"], ctx["soil"]["bdod_kg_dm3"]))
    print("  drainase : %s — %s" % (ctx["drainage"]["level"], ctx["drainage"]["note"]))
    print("\n  BUKAN masukan model. BUKAN divalidasi terhadap kejadian BSR nyata")
    print("  di paket ini — tidak ada georeferensi Eg9PP/ds_B untuk mengujinya.")
