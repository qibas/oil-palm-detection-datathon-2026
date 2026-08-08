"""Unduh font Google ke web/vendor/fonts/ supaya demo tidak butuh internet.

Pola yang sama dengan React/Babel: aset pihak ketiga di-vendor lokal. Ketiga
keluarga berlisensi SIL OFL 1.1, yang secara eksplisit mengizinkan self-hosting.
"""
import io
import os
import re
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "web", "vendor", "fonts")
os.makedirs(OUT, exist_ok=True)

URL = ("https://fonts.googleapis.com/css2"
       "?family=Space+Grotesk:wght@500;700"
       "&family=Instrument+Sans:wght@400;500;600"
       "&family=JetBrains+Mono:wght@400;500&display=swap")

# UA Chrome -> Google mengirim woff2 + subset unicode-range (paling kecil).
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

css = urllib.request.urlopen(
    urllib.request.Request(URL, headers={"User-Agent": UA}), timeout=30
).read().decode("utf-8")

# Hanya subset latin + latin-ext; sisanya (cyrillic, greek, vietnamese) dibuang
# supaya vendor tidak membengkak untuk teks Indonesia.
blocks = re.findall(r"/\*\s*([\w-]+)\s*\*/\s*(@font-face\s*\{[^}]*\})", css)
keep = [(s, b) for s, b in blocks if s in ("latin", "latin-ext")]
print("blok @font-face: %d total, %d dipakai (latin/latin-ext)"
      % (len(blocks), len(keep)))

seen = {}
out = [
    "/* Font di-vendor lokal -- lihat web/vendor/fonts/LICENSE.md.",
    " * Digenerasi oleh tools/vendor_fonts.py; jangan disunting tangan.",
    " * Demo HARUS jalan tanpa internet, jadi index.html tidak boleh",
    " * memuat fonts.googleapis.com lagi. */",
]
for subset, block in keep:
    m = re.search(r"url\((https://[^)]+\.woff2)\)", block)
    fam = re.search(r"font-family:\s*'([^']+)'", block).group(1)
    wt = re.search(r"font-weight:\s*(\d+)", block).group(1)
    if not m:
        continue
    name = "%s-%s-%s.woff2" % (fam.replace(" ", ""), wt, subset)
    dest = os.path.join(OUT, name)
    if name not in seen:
        data = urllib.request.urlopen(
            urllib.request.Request(m.group(1), headers={"User-Agent": UA}), timeout=30
        ).read()
        with open(dest, "wb") as fh:
            fh.write(data)
        seen[name] = len(data)
        print("  %-42s %6.1f KB" % (name, len(data) / 1024))
    out.append(block.replace(m.group(1), "/web/vendor/fonts/" + name))

io.open(os.path.join(ROOT, "web", "fonts.css"), "w", encoding="utf-8").write(
    "\n".join(out) + "\n")
print("\nweb/fonts.css ditulis; total %.0f KB dalam %d berkas"
      % (sum(seen.values()) / 1024, len(seen)))
