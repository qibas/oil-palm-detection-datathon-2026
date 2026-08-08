/* Pemeriksaan app.jsx sebelum menyajikannya.  node web/check_jsx.js
 *
 * DUA CEK, karena keduanya pernah lolos dari pencocokan string:
 *   1. SINTAKS - dengan Babel yang SAMA dengan yang dipakai browser. Berkas JSX
 *      yang rusak tetap dikirim server dengan status 200, jadi pemeriksaan
 *      "apakah teks ini ada" tidak akan pernah menangkapnya.
 *   2. ESCAPE UNICODE YANG BOCOR - "—" yang lolos dari skrip patch akan
 *      dirender apa adanya sebagai teks di layar, dan sintaksnya tetap sah.
 *      Pernah terjadi 23 kali sekaligus.
 */
const fs = require("fs");
const path = require("path");
const babel = require(path.join(__dirname, "vendor", "babel.min.js"));

const file = path.join(__dirname, "app.jsx");
const src = fs.readFileSync(file, "utf8");
let bad = 0;

// --- 1. escape unicode literal -------------------------------------------
const leaked = src.match(/\\u[0-9a-fA-F]{4}/g);
if (leaked) {
  const uniq = [...new Set(leaked)];
  console.error("GAGAL: %d escape unicode bocor jadi teks -> %s",
    leaked.length, uniq.join(" "));
  src.split("\n").forEach((ln, i) => {
    if (/\\u[0-9a-fA-F]{4}/.test(ln))
      console.error("   baris %d: %s", i + 1, ln.trim().slice(0, 90));
  });
  bad = 1;
} else {
  console.log("escape unicode : bersih");
}

// --- 2. sintaks ------------------------------------------------------------
try {
  babel.transform(src, { presets: ["react"], filename: "app.jsx" });
  console.log("sintaks JSX    : OK");
} catch (e) {
  console.error("GAGAL PARSE\n" + e.message);
  bad = 1;
}

process.exit(bad);
