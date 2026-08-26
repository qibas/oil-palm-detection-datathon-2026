/* SawitGuard: frontend React.
 *
 * Tata letak mengikuti "SawitGuard Mockups.dc.html" (proyek claude.ai/design
 * f00d4173): appbar dengan stepper di kanan, layar Unggah -> Proses -> Hasil.
 *
 * TIGA TEMPAT MOCKUP SENGAJA TIDAK DIIKUTI, dan alasannya:
 *
 * 1. PETA. Mockup menggambar kisi sintetis bergaya Eg9PP. Sistem kita memberi skor
 *    pada pohon di FOTO YANG DIUNGGAH, jadi petanya overlay di atas citra itu.
 *    Menampilkan kisi lain akan menyiratkan hasilnya berasal dari kebun lain.
 * 2. RAMP RISIKO. Mockup memakai --risk-1..6 (rainbow); gagal empat cek
 *    aksesibilitas - lihat komentar di styles.css. Dipakai ramp merah satu-warna.
 * 3. ANGKA. Mockup memuat placeholder ("1.187 sawit", "660 dinilai", "3,1 ha").
 *    Semua angka di sini datang dari /api/analyze; tidak ada yang ditulis tangan.
 *
 * Tidak ada perhitungan di berkas ini. Semuanya dari demo_core.py lewat API,
 * supaya versi web dan versi Streamlit tidak mungkin memberi angka berbeda.
 */
const { useState, useEffect, useRef, useCallback, useMemo } = React;

const Q = ["var(--q1)", "var(--q2)", "var(--q3)", "var(--q4)", "var(--q5)"];
const QHEX = ["#EE9A87", "#E5484D", "#B32B30", "#822024", "#4F1315"];
const GREEN = "#4EC75B", DANGER = "#E5484D", INK = "#1A1C17";
const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v))
  ? "—" : Number(v).toFixed(d).replace(".", ",");

// Backend mengembalikan {error: "..."} berbahasa Indonesia untuk kegagalan yang
// terduga (mis. berkas bukan gambar). Tanpa ini, pengguna hanya melihat
// "Error: HTTP 400/500" - benar tapi tidak memberi tahu apa yang harus dilakukan.
async function apiErrorMessage(r) {
  try {
    const j = await r.json();
    if (j && j.error) return j.error;
  } catch (e) { /* body bukan JSON - pakai fallback di bawah */ }
  return "Server gagal memproses (HTTP " + r.status + "). Coba lagi.";
}

/* Persentil menggantikan skor mentah di SELURUH tabel peringkat.
   Alasannya tiga, dan ketiganya sudah pernah menggigit:
     1. skor jalur foto adalah logit mentah yang semuanya NEGATIF (-0,93 .. -0,65);
        tanda minus tidak berarti apa pun bagi pembaca dan mengundang salah baca;
     2. skala logit model foto (1 kolom) dan model Eg9PP (24 kolom) TIDAK sebanding,
        padahal keduanya tampil sebagai kolom "Skor" yang terlihat setara;
     3. banyak skor kembar, sehingga tiga desimal menyiratkan presisi yang tidak ada.
   Persentil bermakna sama di model mana pun: "berada di X% teratas".

   PENTING soal kelompok terbawah. Jalur foto sering hanya membedakan dua tingkat,
   sehingga semua sawit di luar kelompok teratas berpersentil TEPAT 100 - benar secara
   aritmetika ("berada di 100% teratas") tetapi terbaca manusia sebagai prioritas
   MAKSIMUM, yaitu kebalikan artinya. Kelompok itu karena itu ditulis "sisanya".

   Tabel FOTO tidak memakai persentil sama sekali (lihat `Tingkat` di bawah);
   persentil hanya dipakai di tabel Eg9PP, yang punya 672 sawit dengan skor
   nyaris tanpa seri sehingga urutannya monoton dan tidak membingungkan. */
const Prioritas = ({ pct }) => {
  if (pct === null || pct === undefined || Number.isNaN(pct))
    return <span className="tt">—</span>;
  if (pct >= 99.95) return <span className="tt">sisanya</span>;
  return <span><b>{pct < 1 ? Number(pct).toFixed(1).replace(".", ",")
    : String(Math.round(pct))}%</b> <i className="tt">teratas</i></span>;
};

/* Label tingkat untuk tabel FOTO. Persentil terbukti membingungkan di sini:
   angkanya MENGECIL saat prioritas naik ("6% teratas" di atas "11% teratas"),
   dan pada jalur foto hampir semua sawit seri sehingga angkanya berulang.

   Kosakatanya menyesuaikan JUMLAH TINGKAT yang sungguh dibedakan model - biasanya
   dua. Memaksakan lima nama untuk dua tingkat akan mengarang perbedaan yang tidak
   diukur; itu persis kesalahan yang pernah terjadi dengan lima pita kuintil. */
const LADDER = {
  1: ["Tinggi"],
  2: ["Tinggi", "Rendah"],
  3: ["Tinggi", "Sedang", "Rendah"],
  4: ["Sangat tinggi", "Tinggi", "Rendah", "Sangat rendah"],
  5: ["Sangat tinggi", "Tinggi", "Sedang", "Rendah", "Sangat rendah"],
};
const Tingkat = ({ lvl, n }) => {
  if (!lvl || !n) return <span className="tt">—</span>;
  const kata = (LADDER[n] || LADDER[5])[Math.min(lvl, (LADDER[n] || LADDER[5]).length) - 1];
  // Warna diturunkan dari tingkat yang sama dengan katanya, sehingga titik di tabel
  // selalu cocok dengan warna sawit itu di peta. Dulu keduanya berasal dari dua
  // sumber berbeda dan bisa berselisih.
  const c = QHEX[n < 2 ? 4 : Math.round((n - lvl) / (n - 1) * 4)];
  return <span className="lvl"><i style={{ background: c }} />{kata}</span>;
};

/* "Ada sawit sakit yang bersentuhan?" -> Ya / Tidak, bukan 0 / 1. */
const YaTidak = ({ ada, n }) => <span className={"yn " + (ada ? "yn-ya" : "yn-no")}>
  {ada ? "Ya" : "Tidak"}{ada && n > 1 ? <i> · {n}</i> : null}</span>;

/* Interpolasi warna sepanjang ramp merah, untuk mode KONTINU.
   Mode biner tetap memakai lima pita diskret yang lolos gate ordinal; gate itu
   menuntut beda kecerahan >= 0,06 antar langkah, sehingga maksimum lima. Untuk
   besaran kontinu yang dibaca lewat colourbar gate itu TIDAK berlaku, dan
   memaksakan lima pita justru MENABRAKKAN tingkat yang sungguh dibedakan model
   (6 tingkat -> pita [1,2,3,3,4,5], dua tingkat jadi satu warna). */
const RGB = QHEX.map(h => [1, 3, 5].map(i => parseInt(h.substr(i, 2), 16)));
function rampColor(v) {
  const t = Math.max(0, Math.min(1, v)) * (RGB.length - 1);
  const i = Math.min(RGB.length - 2, Math.floor(t)), f = t - i;
  const c = RGB[i].map((a, k) => Math.round(a + (RGB[i + 1][k] - a) * f));
  return "rgb(" + c.join(",") + ")";
}
const RAMP_CSS = "linear-gradient(90deg," + QHEX.join(",") + ")";

const Icon = ({ d, size = 22, stroke = "var(--ink-900)", cls = "" }) =>
  <svg className={cls} viewBox="0 0 24 24" width={size} height={size} fill="none"
    stroke={stroke} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"
    dangerouslySetInnerHTML={{ __html: d }} />;

const I_UP = '<path d="M12 16V6m0 0-4 4m4-4 4 4"/><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>';
const I_GEAR = '<circle cx="12" cy="12" r="3"/><path d="M12 2v3m0 14v3M2 12h3m14 0h3m-3.9-7.1-2.1 2.1M7 17l-2.1 2.1M4.9 4.9 7 7m10 10 2.1 2.1"/>';

/* ------------------------------------------------------------------ appbar */
const SIDE_VIEWS = [["bukti", "Bukti & validasi"], ["env", "Konteks lingkungan"]];

function AppBar({ step, view, onView }) {
  const S = ["Unggah", "Proses", "Hasil"];
  return <div className="appbar">
    <div className="mark">Sawit<i>Guard</i></div>
    <div className="appbar-actions">
      {view === "app" && <div className="stepper" style={{ marginRight: 4 }}>
        {S.map((label, i) => <React.Fragment key={label}>
          {i > 0 && <span className="sep" />}
          <div className={"s " + (step > i ? "done" : step === i ? "active" : "")}>
            <span className="n">{step > i ? "✓" : i + 1}</span>
            <span className="lab">{label}</span>
          </div>
        </React.Fragment>)}
      </div>}
      {view === "app"
        ? SIDE_VIEWS.map(([k, l]) =>
            <button key={k} className="navlink" onClick={() => onView(k)}>{l}</button>)
        : <button className="navlink on" onClick={() => onView("app")}>
            ← Kembali ke aplikasi
          </button>}
    </div>
  </div>;
}

/* --------------------------------------------------------------- layar 1 */
function Upload({ samples, onRun, onRunBatch }) {
  const [sel, setSel] = useState(0);
  const [over, setOver] = useState(false);
  const fileRef = useRef();

  // Satu foto -> alur tunggal yang sudah ada. Lebih dari satu -> alur survei
  // (BatchResults), karena kebun sungguhan disurvei ubin demi ubin, bukan satu
  // foto sekali jalan.
  const handleFiles = fs => {
    if (fs.length === 1) onRun({ file: fs[0] });
    else if (fs.length > 1) onRunBatch(fs);
  };

  return <div className="page fade">
    <h1 style={{ font: "700 44px/1.08 var(--font-display)", letterSpacing: "-.02em" }}>
      Di mana tim kamu<br />harus melihat duluan?
    </h1>
    <p className="sec" style={{ margin: "14px 0 32px", maxWidth: 520 }}>
      Unggah foto drone kebunmu. Sistem menemukan setiap sawit, membangun
      graf kontak akarnya, lalu memeringkat sawit sehat mana yang paling berisiko
      tertular berikutnya.
    </p>

    <div className={"drop" + (over ? " over" : "")}
      onDragOver={e => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={e => { e.preventDefault(); setOver(false);
        handleFiles(Array.from(e.dataTransfer.files)); }}
      onClick={() => fileRef.current.click()}>
      <div className="ico"><Icon d={I_UP} size={22} /></div>
      <div className="big">Jatuhkan citra drone di sini</div>
      {/* Syaratnya sengaja TIDAK dijabarkan di sini. Empat kartu teks membuat layar
          pertama terasa seperti formulir, dan pengguna belum punya konteks untuk
          menilainya. Pemeriksaan lengkap dijalankan SESUDAH unggah dan hanya muncul
          sebagai dialog kalau ada yang tidak lolos - lihat `SyaratDialog`. */}
      <div className="muted" style={{ marginTop: 6, fontSize: 13 }}>
        atau klik untuk memilih berkas &nbsp;·&nbsp; JPG / PNG &nbsp;·&nbsp;
        tegak dari atas, minimal 20 sawit &nbsp;·&nbsp; pilih beberapa sekaligus
        untuk survei satu blok
      </div>
      <input ref={fileRef} type="file" accept="image/*" multiple hidden
        onChange={e => handleFiles(Array.from(e.target.files))} />
    </div>

    <div className="eyebrow" style={{ margin: "28px 0 12px" }}>Atau coba contoh</div>
    <div className="grid g3">
      {samples.slice(0, 6).map(s =>
        <div key={s.id} className={"samp" + (sel === s.id ? " sel" : "")}
          onClick={() => setSel(s.id)} onDoubleClick={() => onRun({ sample: s.id })}>
          <img src={s.thumb} alt="" />
          <span className="cap">{s.label}</span>
        </div>)}
    </div>

    <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 24 }}>
      <button className="btn btn-primary" onClick={() => onRun({ sample: sel })}>
        Analisis foto
      </button>
    </div>

    <p className="fine" style={{ marginTop: 32 }}>
      Model dilatih di kebun percobaan pemuliaan, bukan kebun produksi. Label citra
      adalah kesehatan tajuk umum, bukan BSR yang terverifikasi di lapangan.
    </p>
  </div>;
}

/* --------------------------------------------------------------- layar 2 */
function Processing({ phase, name }) {
  const P = [["Mendeteksi pusat tajuk", "detektor YOLOv12n"],
             ["Memeriksa skala kebun", "jendela 0,80–1,25×"],
             ["Membangun graf kontak", "r = 1,5 × jarak tanam"],
             ["Memeringkat risiko", "checkpoint v3-foto"]];
  return <div className="page fade" style={{ maxWidth: 560, paddingTop: 88 }}>
    <div className="card" style={{ padding: 40, borderRadius: "var(--radius-xl)" }}>
      <div style={{ width: 56, height: 56, borderRadius: "50%",
        background: "var(--lime-500)", display: "grid", placeItems: "center",
        marginBottom: 20 }}>
        <Icon d={I_GEAR} size={26} cls="spin" />
      </div>
      <div style={{ font: "600 26px/1.2 var(--font-display)", letterSpacing: "-.01em" }}>
        Memproses {name || "citra"}
      </div>
      <div className="muted" style={{ fontSize: 13, margin: "6px 0 26px" }}>
        Berjalan di CPU · tanpa koneksi internet
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        {P.map(([lab, det], i) =>
          <div key={lab} className="pb" style={{ opacity: i > phase ? .35 : 1 }}>
            <div className="row"><span className="lab">{lab}</span>
              <span className="det">{i < phase ? "selesai" : i === phase ? det : ""}</span></div>
            <div className="bar"><i style={{ width: (i < phase ? 100 : i === phase ? 62 : 0) + "%" }} /></div>
          </div>)}
      </div>
    </div>
  </div>;
}

/* --------------------------------------------------------- layar 2, survei */
function BatchProcessing({ progress }) {
  const pct = progress.total ? Math.round(100 * progress.done / progress.total) : 0;
  return <div className="page fade" style={{ maxWidth: 560, paddingTop: 88 }}>
    <div className="card" style={{ padding: 40, borderRadius: "var(--radius-xl)" }}>
      <div style={{ width: 56, height: 56, borderRadius: "50%",
        background: "var(--lime-500)", display: "grid", placeItems: "center",
        marginBottom: 20 }}>
        <Icon d={I_GEAR} size={26} cls="spin" />
      </div>
      <div style={{ font: "600 26px/1.2 var(--font-display)", letterSpacing: "-.01em" }}>
        Memproses survei, {progress.done} dari {progress.total} foto
      </div>
      <div className="muted" style={{ fontSize: 13, margin: "6px 0 26px" }}>
        {progress.name ? "Sekarang: " + progress.name : "Menyusun hasil…"}
      </div>
      <div className="bar"><i style={{ width: pct + "%" }} /></div>
    </div>
  </div>;
}

/* --------------------------------------------------------- layar 3, survei
   Menggabungkan beberapa foto BUKAN berarti skornya boleh disatukan. `skor`
   v3-foto adalah fungsi difusi yang dinormalisasi memakai DERAJAT RATA-RATA
   graf foto itu sendiri (lihat `score_photo` di demo_core.py) - belum pernah
   diukur apakah skala itu sama antar foto yang berbeda. Yang AMAN dibandingkan
   lintas-foto adalah bilangan bulat murni: jumlah tetangga bergejala. Itu
   sebabnya peringkat gabungan di sini diurutkan dari situ, BUKAN dari `skor` -
   dan itu dinyatakan eksplisit di layar, bukan didiamkan. */
function combinedPriorities(batchData, capN) {
  const rows = [];
  batchData.forEach((j, pi) => {
    if (j.error || !j.risk || j.risk.degenerate) return;
    j.risk.points.forEach(p => rows.push({
      photoIdx: pi, srcName: j.srcName, nb_sick: p.nb_sick, nb: p.nb, rankInPhoto: p.rank,
    }));
  });
  rows.sort((a, b) => b.nb_sick - a.nb_sick || a.rankInPhoto - b.rankInPhoto);
  return rows.slice(0, capN);
}

function BatchResults({ batchData, onReset, onOpenPhoto }) {
  const ok = batchData.filter(j => !j.error);
  const failed = batchData.filter(j => j.error);
  const scored = ok.filter(j => j.risk && !j.risk.degenerate);
  const totalTrees = ok.reduce((s, j) => s + (j.detect.n || 0), 0);
  const totalRisk = ok.reduce((s, j) => s + (j.risk ? j.risk.n_risk : 0), 0);
  const totalSick = ok.reduce((s, j) => s + (j.detect.n_sympt || 0), 0);
  const top = combinedPriorities(batchData, 20);

  return <div className="page-wide fade">
    <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between",
      gap: 16, flexWrap: "wrap" }}>
      <div>
        <h1 className="h1">Prioritas gabungan survei</h1>
        <div style={{ marginTop: 8, display: "flex", gap: 8, flexWrap: "wrap" }}>
          <span className="badge inverse">{batchData.length} foto</span>
          <span className="badge lime">{totalTrees} sawit</span>
          {failed.length > 0 && <span className="badge neutral">{failed.length} gagal</span>}
        </div>
      </div>
      <button className="btn btn-ghost btn-sm" onClick={onReset}>Survei baru</button>
    </div>

    <div className="grid g4" style={{ margin: "20px 0" }}>
      {[["Foto diproses", batchData.length], ["Sawit terdeteksi", totalTrees],
        ["Dinilai (sehat)", totalRisk], ["Sumber (bergejala)", totalSick]].map(kv =>
        <div className="card" key={kv[0]} style={{ padding: 16 }}>
          <div className="muted" style={{ fontSize: 12 }}>{kv[0]}</div>
          <div className="mono" style={{ fontSize: 21, fontWeight: 500 }}>{kv[1]}</div>
        </div>)}
    </div>

    <div className="note note-soft" style={{ marginBottom: 20 }}>
      <span className="claim">Diurutkan dari jumlah tetangga bergejala, bukan skor.</span>
      <span className="detail">Skor v3-foto dinormalisasi memakai derajat rata-rata graf
        tiap foto sendiri, dan belum pernah diukur apakah skalanya sama antar foto berbeda.
        Jumlah tetangga bergejala adalah bilangan bulat yang berarti sama di foto mana
        pun, jadi itu yang dipakai untuk membandingkan lintas-foto.</span>
    </div>

    {top.length > 0 ? <div className="card" style={{ padding: 0, overflow: "hidden" }}>
      <div style={{ padding: "16px 18px 10px", display: "flex",
        justifyContent: "space-between", alignItems: "baseline" }}>
        <div className="h2">20 prioritas teratas, lintas foto</div>
        <button className="btn btn-ghost btn-sm no-print" onClick={() => {
          const rows = [["peringkat", "foto", "peringkat_di_foto", "tetangga", "tetangga_sakit"]]
            .concat(top.map((r, i) => [i + 1, r.srcName, r.rankInPhoto, r.nb, r.nb_sick]));
          const csv = rows.map(r => r.join(",")).join("\n");
          const a = document.createElement("a");
          a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
          a.download = "prioritas_survei.csv"; a.click();
        }}>Ekspor CSV</button>
      </div>
      <table>
        <thead><tr><th>#</th><th>Foto</th><th>Tetangga</th><th>Tetangga sakit</th></tr></thead>
        <tbody>{top.map((r, i) => <tr key={i} style={{ cursor: "pointer" }}
          onClick={() => onOpenPhoto(r.photoIdx)}>
          <td><span className="rank-pill">{i + 1}</span></td>
          <td>{r.srcName}<span className="fine"> · #{r.rankInPhoto} di foto ini</span></td>
          <td className="num">{r.nb}</td>
          <td className="num">{r.nb_sick}</td>
        </tr>)}</tbody>
      </table>
    </div> : <div className="card">
      <div className="note note-warn">Tidak ada tajuk bergejala di foto mana pun pada
        survei ini, jadi tidak ada peringkat lintas-foto yang berarti.</div>
    </div>}

    <div className="eyebrow" style={{ margin: "24px 0 12px" }}>Per foto, klik untuk detail</div>
    <div className="grid g3">
      {batchData.map((j, i) => <div key={i} className="card"
        style={{ padding: 12, cursor: j.error ? "default" : "pointer",
          opacity: j.error ? .55 : 1 }}
        onClick={() => !j.error && onOpenPhoto(i)}>
        {j.image && <img src={j.image} alt="" style={{ width: "100%", height: 110,
          objectFit: "cover", borderRadius: "var(--radius-md)", marginBottom: 8 }} />}
        <div style={{ font: "600 13px var(--font-body)" }}>{(j.srcName || "").slice(0, 22)}</div>
        <div className="fine">
          {j.error ? "Gagal: " + j.error
            : (j.detect.n + " sawit · " + (j.risk && !j.risk.degenerate
                ? j.risk.n_risk + " dinilai" : "tanpa gejala"))}
        </div>
      </div>)}
    </div>

    {scored.length < ok.length && ok.length > 0 && <p className="fine" style={{ marginTop: 16 }}>
      {ok.length - scored.length} dari {ok.length} foto tidak memuat tajuk bergejala,
      jadi tidak ikut peringkat gabungan. Ini bukan galat, itu petak yang sehat.
    </p>}
  </div>;
}

/* ---------------------------------------------------------- kanvas overlay */
const HOVER_HIT_PX = 15;    // radius klik/hover, dalam px CSS layar
const ACCENT = "#C3EC3C";   // --lime-500, sengaja BEDA dari GREEN/DANGER pohon

function Overlay({ src, w, h, mode, data, showGreys }) {
  const cv = useRef(), box = useRef();
  // Sorot-hover HANYA berarti di mode "graph". Lihat DEMO_BRIEF.md §8 butir 2.
  // `hover` mengikuti kursor; `pinned` dikunci lewat klik/tap supaya perangkat
  // sentuh (tanpa hover) tetap bisa menjelajah ketetanggaan satu-per-satu.
  const [hover, setHover] = useState(-1);
  const [pinned, setPinned] = useState(-1);
  const active = pinned >= 0 ? pinned : hover;

  // Peta ketetanggaan dari INDEKS (edge_idx), bukan pencocokan koordinat float:
  // O(1) dan tidak mungkin meleset karena pembulatan independen di kedua sisi.
  const adj = useMemo(() => {
    const m = new Map();
    (data.edge_idx || []).forEach(([i, j]) => {
      if (!m.has(i)) m.set(i, new Set());
      if (!m.has(j)) m.set(j, new Set());
      m.get(i).add(j); m.get(j).add(i);
    });
    return m;
  }, [data.edge_idx]);

  useEffect(() => { setHover(-1); setPinned(-1); }, [mode, data]);

  const draw = useCallback(() => {
    const c = cv.current, wrap = box.current;
    if (!c || !wrap) return;
    const W = wrap.clientWidth, H = W * h / w;
    c.width = W * devicePixelRatio; c.height = H * devicePixelRatio;
    c.style.height = H + "px";
    const g = c.getContext("2d");
    g.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    g.clearRect(0, 0, W, H);

    const hoverOn = mode === "graph" && active >= 0;
    const nbrSet = hoverOn ? (adj.get(active) || new Set()) : null;

    if (mode === "graph") {
      (data.edge_idx || []).forEach(([i, j], k) => {
        const [x1, y1, x2, y2] = data.edges[k];
        const touches = hoverOn && (i === active || j === active);
        g.strokeStyle = touches ? ACCENT : "rgba(255,255,255,.45)";
        g.lineWidth = touches ? 2.6 : 1.3;
        g.globalAlpha = hoverOn && !touches ? 0.3 : 1;
        g.beginPath(); g.moveTo(x1 * W, y1 * H); g.lineTo(x2 * W, y2 * H); g.stroke();
      });
      g.globalAlpha = 1;
    }
    const dot = (x, y, fill, r = 5.5, alpha = 1, ring = null) => {
      g.globalAlpha = alpha;
      g.beginPath(); g.arc(x * W, y * H, r, 0, 7);
      g.fillStyle = fill; g.fill();
      g.lineWidth = ring ? 2.2 : 1.5; g.strokeStyle = ring || "#fff"; g.stroke();
      g.globalAlpha = 1;
    };
    const risk = data.risk;
    if (mode === "risk" && risk && !risk.degenerate) {
      if (showGreys) (risk.sources || []).forEach(s => {
        const x = s.x * W, y = s.y * H;
        g.lineWidth = 3.2; g.strokeStyle = INK;
        g.beginPath(); g.moveTo(x - 6, y - 6); g.lineTo(x + 6, y + 6);
        g.moveTo(x + 6, y - 6); g.lineTo(x - 6, y + 6); g.stroke();
        g.lineWidth = 1.1; g.strokeStyle = "#fff"; g.stroke();
      });
      const cont = risk.mode === "kontinu";
      risk.points.forEach(p => dot(p.x, p.y,
        cont ? rampColor(p.v) : QHEX[p.q - 1],
        (cont ? p.v > 0.8 : p.q === 5) ? 7 : 5.5));

      // Nomor untuk SETIAP sawit yang diperingkat, bukan hanya 10 teratas.
      // Menomori sebagian justru membingungkan: pembaca bertanya kenapa titik ini
      // punya nomor dan tetangganya tidak. Pada ubin khas ada ~90 sawit dengan
      // jarak layar ~70 px, jadi lencana 15-17 px masih longgar.
      //
      // Digambar SESUDAH semua titik agar tidak tertimpa tetangganya. Lencananya
      // netral, bukan mewarisi warna pita: pita terendah (#EE9A87) terlalu terang
      // untuk teks putih dan pita tertinggi (#4F1315) terlalu gelap untuk teks
      // gelap - satu gaya netral terbaca di atas keduanya sekaligus di atas citra
      // apa pun. Sepuluh teratas dibalik warnanya supaya tetap bisa dicari cepat
      // saat mata berpindah dari tabel ke peta.
      g.textAlign = "center";
      g.textBaseline = "middle";
      risk.points.forEach(p => {
        const top = p.rank <= 10;
        const R = top ? 8.5 : 7.5;
        const bx = Math.min(W - R - 1, Math.max(R + 1, p.x * W + (top ? 10 : 9)));
        const by = Math.min(H - R - 1, Math.max(R + 1, p.y * H - (top ? 10 : 9)));
        g.beginPath(); g.arc(bx, by, R, 0, 7);
        g.fillStyle = top ? INK : "rgba(255,255,255,.92)"; g.fill();
        g.lineWidth = top ? 1.6 : 1.2;
        g.strokeStyle = top ? "#fff" : INK; g.stroke();
        g.fillStyle = top ? "#fff" : INK;
        g.font = (top ? '700 11px ' : '600 10px ')
          + '"Instrument Sans", system-ui, sans-serif';
        g.fillText(String(p.rank), bx, by + .5);
      });
    } else {
      (data.crowns || []).forEach((c2, i) => {
        const fill = c2.cls === "Unhealthy" ? DANGER : GREEN;
        if (!hoverOn) { dot(c2.x, c2.y, fill); return; }
        if (i === active) dot(c2.x, c2.y, fill, 8, 1, ACCENT);
        else if (nbrSet.has(i)) dot(c2.x, c2.y, fill, 6.5, 1, ACCENT);
        else dot(c2.x, c2.y, fill, 5.5, 0.28);
      });

      // Lencana jumlah tetangga di atas pohon yang disorot, pola yang sama
      // dengan lencana peringkat di mode "risk" di atas, supaya bahasa visualnya
      // konsisten di seluruh layar Hasil.
      if (hoverOn) {
        const p = data.crowns[active];
        const nCount = nbrSet.size;
        const bx = p.x * W, by = p.y * H;
        g.font = '700 11px "Instrument Sans", system-ui, sans-serif';
        const label = nCount + (nCount === 1 ? " tetangga" : " tetangga");
        const tw = g.measureText(label).width, padX = 7, boxH = 20;
        const lx = Math.min(W - tw - padX * 2 - 2, Math.max(2, bx - tw / 2 - padX));
        const ly = Math.max(2, by - 14 - boxH);
        g.fillStyle = INK;
        g.fillRect(lx, ly, tw + padX * 2, boxH);
        g.fillStyle = "#fff";
        g.textAlign = "left"; g.textBaseline = "middle";
        g.fillText(label, lx + padX, ly + boxH / 2 + .5);
      }
    }
  }, [mode, data, w, h, showGreys, active, adj]);

  useEffect(() => { draw(); }, [draw]);
  useEffect(() => {
    const r = () => draw(); window.addEventListener("resize", r);
    return () => window.removeEventListener("resize", r);
  }, [draw]);

  // Cari pohon terdekat dari kursor, dalam radius HOVER_HIT_PX (px layar).
  // -1 kalau tidak ada yang cukup dekat.
  const nearest = useCallback((e) => {
    const wrap = box.current;
    if (!wrap || !data.crowns) return -1;
    const rect = wrap.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let best = -1, bestD = HOVER_HIT_PX * HOVER_HIT_PX;
    data.crowns.forEach((p, i) => {
      const dx = p.x * rect.width - mx, dy = p.y * rect.height - my;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD) { bestD = d2; best = i; }
    });
    return best;
  }, [data.crowns]);

  const isGraph = mode === "graph";
  return <div className="viz" ref={box}
    style={isGraph ? { cursor: "pointer" } : undefined}
    onMouseMove={isGraph ? (e => { if (pinned < 0) setHover(nearest(e)); }) : undefined}
    onMouseLeave={isGraph ? (() => setHover(-1)) : undefined}
    onClick={isGraph ? (e => {
      const i = nearest(e);
      setPinned(p => (i >= 0 && i === p) ? -1 : i);
    }) : undefined}>
    <img src={src} alt="" onLoad={draw} />
    <canvas ref={cv} />
  </div>;
}

/* --------------------------------------------------------------- layar 3 */
/* Kartu ringkas konteks lingkungan untuk layar Hasil - BUKAN duplikasi layar
   "Konteks lingkungan", cuma jendela satu baris ke sana. Memakai koordinat
   tersimpan (localStorage) kalau pengguna sudah pernah mengisinya di layar
   itu; kalau belum, tetap tampil dengan lokasi contoh supaya kartu ini tidak
   pernah kosong - dan badge-nya bilang jujur yang mana yang sedang dipakai. */
function EnvMini({ onOpen }) {
  const [d, setD] = useState(null);
  useEffect(() => {
    const saved = loadSavedCoords();
    const la = saved ? saved.lat : ENV_DEFAULT_LAT, lo = saved ? saved.lon : ENV_DEFAULT_LON;
    fetch("/api/env_context?lat=" + la + "&lon=" + lo)
      .then(r => r.json()).then(j => setD(j.ok ? j : null)).catch(() => setD(null));
  }, []);
  if (!d) return null;
  const isSaved = !!loadSavedCoords();
  return <div className="card" style={{ cursor: "pointer" }} onClick={onOpen}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
      <div className="h2">Konteks lingkungan</div>
      <span className="badge neutral">{isSaved ? "kebunmu" : "contoh"}</span>
    </div>
    <p className="fine" style={{ margin: "6px 0 2px" }}>
      Angin {num(d.weather.wind_speed_kmh, 1)} km/h · hujan{" "}
      {Math.round(d.weather.rain_30d_mm)} mm/30hr · drainase {d.drainage.level}.
      Bukan masukan model. Lihat detail →
    </p>
  </div>;
}

function Results({ d, onReset, onOpenEnv, backLabel }) {
  const [mode, setMode] = useState("risk");
  const [greys, setGreys] = useState(true);
  /* Mode kontinu MATI secara bawaan, dan itu disengaja.
     Ia memakai keyakinan Unhealthy DI BAWAH ambang 0,75 - masukan di luar
     distribusi latih, karena model dilatih pada status biner yang terverifikasi
     lapangan. Dua akibatnya terlihat langsung di layar: tingkatnya bertambah
     tanpa dasar terukur, dan kolom "ada sawit sakit di sekitarnya" jadi berbunyi
     "Tidak" pada sawit yang tetap diperingkat tinggi - karena penggeraknya
     tetangga berkeyakinan 0,3 yang tidak dihitung sakit. Tombolnya tetap ada
     untuk pembaca teknis, tapi bukan yang pertama dilihat orang. */
  const [soft, setSoft] = useState(false);
  const det = d.detect;
  const risk = (soft && d.risk_soft) ? d.risk_soft : d.risk;
  const ok = risk && !risk.degenerate;
  const dd = Object.assign({}, d, { risk: risk });

  return <div className="page-wide fade">
    {/* Muncul HANYA saat dicetak - appbar (nama produk) ikut disembunyikan lewat
        .no-print, jadi laporan cetak butuh identitasnya sendiri: siapa, foto
        mana, kapan. */}
    <div className="print-only">
      <div style={{ font: "700 17px var(--font-display)" }}>SawitGuard-GNN: laporan prioritas pemeriksaan</div>
      <div className="fine" style={{ marginBottom: 12 }}>
        {d.name} · dicetak {new Date().toLocaleDateString("id-ID",
          { day: "numeric", month: "long", year: "numeric" })}
      </div>
    </div>
    <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between",
      gap: 16, flexWrap: "wrap" }}>
      <div>
        <h1 className="h1">Peta risiko</h1>
        <div style={{ marginTop: 8, display: "flex", gap: 8, flexWrap: "wrap" }}>
          <span className="badge inverse">{det.n} sawit</span>
          {ok && <span className="badge lime">{risk.n_risk} dinilai</span>}
          <span className="badge neutral">{d.name.slice(0, 22)}…</span>
        </div>
      </div>
      <div className="no-print" style={{ display: "flex", gap: 12 }}>
        <button className="btn btn-ghost btn-sm" onClick={onReset}>{backLabel || "Foto baru"}</button>
        <button className="btn btn-ghost btn-sm" disabled={!ok}
          onClick={() => window.print()}>Cetak laporan lapangan</button>
        <button className="btn btn-primary btn-sm" disabled={!ok}
          onClick={() => {
            // Unduhan memuat `skor` mentah DI SAMPING persentil: layar sengaja
            // menyembunyikannya, tetapi berkas kerja tidak boleh kehilangan angka
            // aslinya. Urutan kolomnya persentil dulu, supaya yang terbaca lebih
            // dahulu adalah yang bermakna.
            const rows = [["peringkat", "tingkat", "dari_n_tingkat", "persentil_teratas",
                           "tetangga", "tetangga_sakit", "ada_sakit", "skor_mentah", "pita"]]
              .concat(risk.top10.map(r => [r.rank, r.lvl, risk.n_lev, r.pct, r.nb,
                                           r.nb_sick, r.nb_sick > 0 ? "ya" : "tidak",
                                           r.skor, r.q]));
            const csv = rows.map(r => r.join(",")).join("\n");
            const a = document.createElement("a");
            a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
            a.download = "prioritas.csv"; a.click();
          }}>Ekspor daftar prioritas</button>
      </div>
    </div>

    <div className="no-print" style={{ display: "flex", gap: 8, margin: "20px 0 16px",
      alignItems: "center", flexWrap: "wrap" }}>
      {[["risk", "Peringkat risiko"], ["graph", "Graf kontak"], ["crowns", "Deteksi"]]
        .map(([k, l]) =>
          <button key={k} className={"chip" + (mode === k ? " on" : "")}
            onClick={() => setMode(k)}>{l}</button>)}
      <div style={{ marginLeft: "auto", display: "flex", gap: 20,
        alignItems: "center", flexWrap: "wrap" }}>
        <div className={"sw" + (soft ? " on" : "")} onClick={() => setSoft(!soft)}>
          <span className="track"><i /></span>Skor gejala kontinu
        </div>
        <div className={"sw" + (greys ? " on" : "")} onClick={() => setGreys(!greys)}>
          <span className="track"><i /></span>Tampilkan yang sudah bergejala
        </div>
      </div>
    </div>

    {soft && d.risk_soft && <div className="note note-warn" style={{ marginBottom: 16 }}>
      <span className="claim">Skor kontinu ini belum tervalidasi.</span>
      <span className="detail">Kolom gejala di sini terisi keyakinan detektor (0–1),
        bukan status biner. Petanya lebih halus ({d.risk_soft.n_tingkat} tingkat,
        bukan {d.risk.n_tingkat}), tapi Eg9PP tidak punya data untuk membuktikan
        peringkatnya lebih akurat. Yang bertambah baru resolusi tampilan.
        Matikan sakelar untuk kembali ke mode biner yang terukur.</span>
    </div>}

    <div className="grid g-results">
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div className="card" style={{ padding: 20 }}>
        <Overlay src={d.image} w={d.w} h={d.h} mode={mode} data={dd} showGreys={greys} />
        <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 16 }}>
          {mode === "risk" && ok ? <React.Fragment>
            <div>
              {risk.mode === "kontinu"
                ? <div className="ramp" style={{ background: RAMP_CSS, height: 10 }} />
                : <div className="ramp">
                    {(risk.bands || []).map(b =>
                      <i key={b.q} style={{ background: QHEX[b.q - 1], flexGrow: b.n }} />)}
                  </div>}
              <div className="ramp-lab">
                <span>Risiko lebih rendah</span><span>Risiko lebih tinggi</span>
              </div>
            </div>
            <div className="legend">
              <span>✕ sudah bergejala, jadi sumber penularan dan tidak ikut diperingkat</span>
            </div>
          </React.Fragment> : <div className="legend">
            <span className="sw2"><i className="dot" style={{ background: GREEN }} />sehat</span>
            <span className="sw2"><i className="dot" style={{ background: DANGER }} />tidak sehat</span>
            {mode === "graph" && <React.Fragment>
              <span>garis = akar berpotensi bersentuhan</span>
              <span className="muted">· arahkan kursor (atau ketuk) ke satu pohon
                untuk menyorot tetangganya</span>
            </React.Fragment>}
          </div>}
        </div>
      </div>
    {/* Satu note saja di layar utama - inti yang harus dibaca SETIAP orang
        sebelum bertindak. Metodologi (kenapa v3-foto, kontaminasi kekerabatan,
        kalibrasi, dll) pindah ke expander di bawah: sama isinya, cuma tidak
        lagi memblokir jalan ke peta risiko. Skala citra HANYA muncul di sini
        kalau bermasalah - kalau baik-baik saja, itu bukan hal yang perlu
        diperingatkan, cukup baris kecil di expander. */}
    <div className="note note-soft">
      <span className="claim">Ini panduan urutan periksa, bukan diagnosis Ganoderma.</span>
      <span className="detail">Sistem menandai sawit yang kondisinya terlihat buruk dari
        udara. Cek lapangan tetap wajib sebelum bertindak.</span>
    </div>
    {det.ok_n && !det.ok_scale && <div className="note note-warn" style={{ marginTop: 12 }}>
      <span className="claim">Skala citra {num(det.scale_ratio)}×, di luar jendela terlatih.</span>
      <span className="detail">Rentang data latih 0,80–1,25×. Angka graf di atas tidak
        sebanding dengan angka mana pun di repositori ini.</span>
    </div>}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div className="grid g4" style={{ gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div className="card" style={{ padding: 16 }}>
            <div className="muted" style={{ fontSize: 12 }}>Jarak tanam</div>
            <div className="mono" style={{ fontSize: 21, fontWeight: 500 }}>
              {det.spacing_px ? num(det.spacing_px, 0) + " px" : "—"}</div>
          </div>
          <div className="card" style={{ padding: 16 }}>
            <div className="muted" style={{ fontSize: 12 }}>Derajat graf</div>
            <div className="mono" style={{ fontSize: 21, fontWeight: 500 }}>
              {num(det.deg_inner)}</div>
          </div>
        </div>

        {ok ? <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          <div style={{ padding: "16px 18px 10px", display: "flex",
            justifyContent: "space-between", alignItems: "baseline" }}>
            <div className="h2">Daftar prioritas</div>
            <div className="muted" style={{ fontSize: 12 }}>10 dari {risk.n_risk}</div>
          </div>
          <table>
            <thead><tr><th>#</th><th>Prioritas</th><th>Tetangga</th>
              <th>Ada sawit sakit di sekitarnya</th></tr></thead>
            <tbody>{risk.top10.map(r =>
              <tr key={r.rank}>
                <td><span className="rank-pill">{r.rank}</span></td>
                <td><Tingkat lvl={r.lvl} n={risk.n_lev} /></td>
                <td className="num">{r.nb}</td>
                <td><YaTidak ada={r.nb_sick > 0} n={r.nb_sick} /></td>
              </tr>)}</tbody>
          </table>
          <div style={{ padding: "12px 18px 18px" }}>
            <div className="note note-soft">
              Sawit yang bersentuhan dengan sawit sakit: <b>{num(risk.nb_sick_top5)}</b> rata-rata
              pada 5 teratas, lawan <b>{num(risk.nb_sick_all)}</b> pada seluruh sawit
              yang dinilai. Sistem membaca <b>keadaan tetangga</b>, bukan pohon itu sendiri.
            </div>
          </div>
        </div> : <div className="card">
          <div className="h2" style={{ marginBottom: 12 }}>
            Belum ada peringkat</div>
          <div className="note note-warn">
            {/* Penjelasan panjangnya sengaja dibuang: pada petak yang seluruhnya
                sehat, satu kalimat sudah cukup dan sisanya terbaca seperti
                pembelaan. Fakta laju dasar (~0,85 sawit sakit per ubin) tetap
                tercatat di 00_HASIL.md dan di expander batasan teknis. */}
            {risk ? <b>Tidak ada tajuk bergejala terdeteksi di citra ini.</b>
                  : "Graf tidak dapat dibangun dari citra ini."}
          </div>
        </div>}

        {d.foci && d.foci.n_fokus > 0 && <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between",
            alignItems: "baseline" }}>
            <div className="h2">Pusat wabah</div>
            <span className="badge neutral">tanpa model</span>
          </div>
          <p className="muted" style={{ fontSize: 12, margin: "6px 0 12px" }}>
            Tajuk bergejala yang saling bersentuhan lewat graf, dikelompokkan.
            Pernyataan geometris, tidak meramal apa pun.
          </p>
          {d.foci.fokus.map((f, i) =>
            <div className="kv" key={i}>
              <span><b>Pusat {i + 1}</b> · {f.n_sakit} tajuk bergejala</span>
              <b>{f.n_terpapar} sehat bersentuhan</b>
            </div>)}
          <div className="kv" style={{ borderTop: "1px solid var(--ink-200)" }}>
            <span className="muted">Total</span>
            <b>{d.foci.n_fokus} pusat · {d.foci.n_terpapar} sawit terpapar langsung</b>
          </div>
        </div>}

        <EnvMini onOpen={onOpenEnv} />

      </div>
    </div>

  </div>;
}

/* --------------------------------------------------------- layar bukti */
function Lattice({ d }) {
  const W = 1000, H = Math.max(220, W * d.aspect), GREY = "#A9AFA3";
  const box = "-14 -14 " + (W + 28) + " " + (H + 28);
  return <svg className="lattice" viewBox={box}>
    {d.points.map((p, i) => {
      const x = p.x * W, y = (1 - p.y) * H;
      if (p.q) return <circle key={i} cx={x} cy={y} r="5.2" fill={QHEX[p.q - 1]}
        stroke="#fff" strokeWidth="1" />;
      if (p.st === "D") return <path key={i} stroke={GREY} strokeWidth="1.5"
        d={"M" + (x - 3.4) + " " + (y - 3.4) + "l6.8 6.8M" + (x + 3.4) + " " + (y - 3.4) + "l-6.8 6.8"} />;
      if (p.st === "C") return <rect key={i} x={x - 3} y={y - 3} width="6" height="6"
        fill="none" stroke={GREY} strokeWidth="1.3" />;
      return <path key={i} fill={GREY}
        d={"M" + x + " " + (y - 4) + "l3.8 6.6h-7.6z"} />;
    })}
  </svg>;
}

/* Animasi 25 tahun (DEMO_BRIEF.md §8, butir pertama). Warna = STATUS APA ADANYA
   dari catatan lapangan Eg9PP (A/S/D/C), sensus demi sensus, BUKAN skor model
   dan bukan kuintil risiko di peta `Lattice` sebelah kanan. Ini pernyataan
   historis murni: memutar ulang apa yang sungguh terekam, bukan menjalankan
   checkpoint 45 kali (yang akan mengundang klaim performa tak terukur, lihat
   larangan checkpoint demo di `DEMO_BRIEF.md` §7). */
function Timeline() {
  const [tl, setTl] = useState(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    fetch("/api/eg9pp_timeline").then(r => r.json()).then(setTl);
  }, []);

  useEffect(() => {
    if (!playing || !tl) return;
    const id = setInterval(() => {
      setFrame(f => {
        if (f >= tl.censuses.length - 1) { setPlaying(false); return f; }
        return f + 1;
      });
    }, 160);
    return () => clearInterval(id);
  }, [playing, tl]);

  if (!tl) return <div className="card"><div className="muted">Memuat animasi…</div></div>;

  const n = tl.censuses.length;
  const W = 1000, H = Math.max(220, W * tl.aspect), GREY = "#A9AFA3";
  const box = "-14 -14 " + (W + 28) + " " + (H + 28);
  const counts = { A: 0, S: 0, D: 0, C: 0 };
  tl.status.forEach(s => { const c = s[frame]; counts[c] = (counts[c] || 0) + 1; });

  return <div className="card">
    <div style={{ display: "flex", justifyContent: "space-between",
      alignItems: "baseline", flexWrap: "wrap", gap: 8 }}>
      <div className="h2">
        Animasi 25 tahun: penyakit merambat antar tetangga
      </div>
      <span className="badge neutral">status apa adanya, bukan skor model</span>
    </div>
    <p className="fine" style={{ margin: "6px 0 14px" }}>
      A = asimtomatik · S = bergejala · D = mati · C = disensor. Diputar ulang
      dari catatan lapangan Eg9PP, sensus demi sensus.
    </p>

    <svg className="lattice" viewBox={box}>
      {tl.palms.map((p, i) => {
        const x = p.x * W, y = (1 - p.y) * H, s = tl.status[i][frame];
        if (s === "A") return <circle key={i} cx={x} cy={y} r="4.6" fill={GREEN}
          stroke="#fff" strokeWidth="0.8" />;
        if (s === "S") return <circle key={i} cx={x} cy={y} r="6.2" fill={DANGER}
          stroke="#fff" strokeWidth="1" />;
        if (s === "D") return <path key={i} stroke={GREY} strokeWidth="1.5"
          d={"M" + (x - 3.4) + " " + (y - 3.4) + "l6.8 6.8M" + (x + 3.4) + " " + (y - 3.4) + "l-6.8 6.8"} />;
        return <rect key={i} x={x - 3} y={y - 3} width="6" height="6"
          fill="none" stroke={GREY} strokeWidth="1.3" />;
      })}
    </svg>

    <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 16 }}>
      <button className="btn btn-primary btn-sm" onClick={() => {
        if (frame >= n - 1) setFrame(0);
        setPlaying(p => !p);
      }}>{playing ? "⏸ Jeda" : "▶ Putar"}</button>
      <input type="range" min="0" max={n - 1} value={frame}
        onChange={e => { setPlaying(false); setFrame(Number(e.target.value)); }}
        style={{ flex: 1 }} />
      <span className="mono" style={{ fontSize: 13, minWidth: 168, textAlign: "right" }}>
        sensus {frame + 1}/{n} · tahun {num(tl.censuses[frame], 1)}
      </span>
    </div>

    <div className="legend" style={{ marginTop: 12 }}>
      <span className="sw2"><i className="dot" style={{ background: GREEN }} />
        asimtomatik ({counts.A || 0})</span>
      <span className="sw2"><i className="dot" style={{ background: DANGER }} />
        bergejala ({counts.S || 0})</span>
      <span>✕ mati ({counts.D || 0})</span>
      <span>▫ disensor ({counts.C || 0})</span>
    </div>
  </div>;
}

function Evidence({ d }) {
  if (!d) return <div className="page"><div className="muted">Memuat…</div></div>;
  const f = d.facts, mx = Math.max.apply(null, d.levels.map(l => l.n));
  const pair = t => num(t[0], 4) + " ± " + num(t[1], 4);
  return <div className="page-wide fade">
    <h1 className="h1">Kebun yang membuktikan modelnya</h1>
    <p className="sec" style={{ margin: "10px 0 24px", maxWidth: 720 }}>
      Layar aplikasi bekerja dari satu foto. Angka performa yang sebenarnya diukur di
      kebun Eg9PP: {d.n_total} pohon sawit yang dipantau selama 45 kali sensus sepanjang
      25 tahun, dengan status Ganoderma yang diperiksa langsung di lapangan, bukan ditebak.
    </p>

    <div className="grid g4" style={{ marginBottom: 20 }}>
      {[["Sawit dipantau", d.n_total], ["Dinilai model", d.n_risk],
        ["Sudah sakit, mati, atau disensor", d.n_out],
        ["Persentase yang pernah bergejala", num(100 * d.sick_rate, 1) + "%"]].map(kv =>
        <div className="card" key={kv[0]} style={{ padding: 16 }}>
          <div className="muted" style={{ fontSize: 12 }}>{kv[0]}</div>
          <div className="mono" style={{ fontSize: 22, fontWeight: 500 }}>{kv[1]}</div>
        </div>)}
    </div>

    <div className="grid g-results">
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div className="card">
        <div className="h2" style={{ marginBottom: 12 }}>Peta risiko kebun Eg9PP</div>
        <Lattice d={d} />
        <div style={{ marginTop: 16 }}>
          <div className="ramp">{QHEX.map(c => <i key={c} style={{ background: c }} />)}</div>
          <div className="ramp-lab"><span>Risiko lebih rendah</span>
            <span>Risiko lebih tinggi</span></div>
        </div>
        <div className="legend" style={{ marginTop: 12 }}>
          <span>▲ bergejala ({d.status_out.S})</span>
          <span>✕ mati ({d.status_out.D})</span>
          <span>▫ disensor ({d.status_out.C})</span>
        </div>
      </div>
      <div className="card">
        <div className="h2" style={{ marginBottom: 12 }}>Peta kontak akar terbukti membantu</div>
        <p className="sec" style={{ fontSize: 13, margin: "0 0 12px" }}>
          Model yang tahu pohon mana bersentuhan dengan pohon mana jauh lebih akurat
          daripada model yang tidak tahu apa-apa soal tetangganya, dan keunggulan itu
          bertahan pada pengujian statistik yang ketat, bukan kebetulan.
        </p>
        <div className="note note-soft">
          <span className="claim">Peta kontak yang benar menambah akurasi secara nyata.</span>
          <span className="detail">Diuji lewat 200 percobaan acak yang mengontrol faktor
            keluarga dan lokasi petak. Hasil aslinya mengalahkan seluruhnya: hanya{" "}
            {f.perm_strict[2]} percobaan acak yang mencapai nilai setinggi hasil
            sungguhan.</span>
        </div>
        <details className="tech" style={{ marginTop: 12 }}>
          <summary>Angka statistik lengkap</summary>
          <div className="body">
            <div className="cmp">
              <span className="h">Model</span><span className="h n">AP dalam-sensus</span><span className="h n">Lift</span>
              <span>Tanpa graf</span><span className="n">{pair(f.ap_nograph_within)}</span><span className="n muted">acak</span>
              <span>Foto, 1 kolom</span><span className="n">{pair(f.ap_1col_within)}</span><span className="n muted">1,45×</span>
              <span>Foto, 6 kolom</span><span className="n">{pair(f.ap_photo_within)}</span><span className="n muted">1,61×</span>
              <span>Penuh, 24 kolom</span><span className="n">{pair(f.ap_full_within)}</span><span className="n muted">1,54×</span>
            </div>
            <p className="fine" style={{ marginTop: 10 }}>
              Sumbangan peta kontak: +{num(f.struktur[0], 4)} ± {num(f.struktur[1], 4)}
              ({f.struktur[2]} tanda searah). Uji permutasi terketat: z = +{num(f.perm_strict[1], 2)}.
            </p>
          </div>
        </details>
      </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          <div className="h2" style={{ padding: "16px 18px 8px" }}>
            Sepuluh pohon paling berisiko</div>
          <table>
            <thead><tr><th>#</th><th>Sawit</th><th>Prioritas</th>
              <th>Ada sawit sakit di sekitarnya</th></tr></thead>
            <tbody>{d.top10.map(r => <tr key={r.rank}>
              <td><span className="rank-pill">{r.rank}</span></td>
              <td className="num">{r.id}</td>
              <td className="num"><Prioritas pct={r.pct} /></td>
              <td><YaTidak ada={r.nb_sick > 0} n={r.nb_sick} /></td></tr>)}</tbody>
          </table>
          <div style={{ padding: "10px 18px 18px" }}>
            <div className="note note-soft">
              <span className="claim">Model membaca tetangga, bukan pohon itu sendiri.</span>
              <span className="detail">Rata-rata tetangga sakit: {num(d.nb_top10, 2)} pada 10
                pohon paling berisiko, {num(d.nb_all, 2)} pada semua pohon yang dinilai,
                dan hanya {num(d.nb_bot10, 2)} pada 10 pohon paling aman.</span>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="h2" style={{ marginBottom: 8 }}>Kenapa peringkatnya lebih halus di sini</div>
          <p className="sec" style={{ fontSize: 13, margin: "0 0 12px" }}>
            Foto drone hanya bisa membedakan dua tingkat risiko. Di kebun Eg9PP, riwayat
            25 tahun memberi {d.levels.length} tingkat berbeda, jadi urutan prioritasnya
            jauh lebih detail.
          </p>
          <div className="levbar">
            {d.levels.map(l => <div key={l.nb}
              style={{ height: (100 * l.n / mx) + "%" }}><span>{l.nb}</span></div>)}
          </div>
          <div className="fine" style={{ marginTop: 24 }}>jumlah tetangga bergejala →</div>
        </div>
      </div>
    </div>

    <div style={{ marginTop: 16 }}><Timeline /></div>

    <div className="note note-limit" style={{ marginTop: 20 }}>
      <span className="claim">Memeriksa 5% pohon paling berisiko menemukan satu kasus
        per {num(f.per_case_model, 1)} pohon, bukan {num(f.per_case_random, 1)} kalau
        memeriksa acak. {num(f.lift_top5, 2)} kali lebih efisien.</span>
      <span className="detail">Batasnya: efek graf mengandung {f.kinship_pct}% kontaminasi
        kekerabatan, jalur foto menyisakan {f.signal_kept_pct}% sinyal, dan seluruh angka
        ini datang dari satu kebun percobaan dengan dua petak yang hasilnya sendiri
        berbeda 2,6 kali satu sama lain.</span>
    </div>
  </div>;
}


/* ---------------------------------------------------- konteks lingkungan
   BUKAN masukan model. Lihat docstring env_context.py / ENV_CONTEXT.md.
   Ganoderma di paket ini menyebar lewat graf kontak akar yang divalidasi di
   Lapisan 2; angin dan tekstur tanah tidak pernah dilatih atau diuji terhadap
   kejadian BSR nyata, karena ds_B dan Eg9PP sama-sama tidak bergeoreferensi.
   Panel ini murni konteks tambahan dari data ASLI (Open-Meteo + ISRIC
   SoilGrids) untuk satu titik koordinat, ditampilkan di samping peringkat
   risiko, bukan di dalamnya. */
const ENV_DEFAULT_LAT = -0.5272, ENV_DEFAULT_LON = 101.4174;
const ENV_COORDS_KEY = "sawitguard_coords";

/* Koordinat kebun disimpan di localStorage supaya SATU kali diisi di layar
   Konteks lingkungan, lalu ikut muncul sebagai info ringkas di layar Hasil -
   dua layar yang sebelumnya sama sekali tidak saling tahu. */
function loadSavedCoords() {
  try {
    const j = JSON.parse(localStorage.getItem(ENV_COORDS_KEY) || "null");
    if (j && typeof j.lat === "number" && typeof j.lon === "number") return j;
  } catch (e) { /* localStorage tidak tersedia atau isinya rusak - abaikan */ }
  return null;
}
function saveCoords(lat, lon) {
  try { localStorage.setItem(ENV_COORDS_KEY, JSON.stringify({ lat, lon })); } catch (e) {}
}

function EnvContext() {
  const saved0 = loadSavedCoords();
  const [lat, setLat] = useState(saved0 ? saved0.lat : ENV_DEFAULT_LAT);
  const [lon, setLon] = useState(saved0 ? saved0.lon : ENV_DEFAULT_LON);
  const [d, setD] = useState(null);
  const [busy, setBusy] = useState(false);

  // Batas waktu di sisi peramban sendiri, terpisah dari batas 6 detik per
  // panggilan di server (angin dan tanah diambil BERSAMAAN, bukan berurutan,
  // jadi kasus terburuk ~6 detik bukan ~12). Jaring pengaman kedua: kalau
  // server pernah macet karena alasan lain, layar ini tidak boleh terjebak
  // "Memuat…" selamanya - 15 detik lalu ditampilkan sebagai galat yang bisa
  // dicoba ulang lewat tombol "Perbarui".
  const load = useCallback((la, lo) => {
    setBusy(true);
    const ac = new AbortController();
    const bom = setTimeout(() => ac.abort(), 15000);
    fetch("/api/env_context?lat=" + la + "&lon=" + lo, { signal: ac.signal })
      .then(r => r.json())
      .then(j => { clearTimeout(bom); setD(j); setBusy(false); })
      .catch(e => {
        clearTimeout(bom);
        const msg = e.name === "AbortError"
          ? "Server tidak merespons dalam 15 detik. Coba lagi."
          : String(e);
        setD({ ok: false, error: msg }); setBusy(false);
      });
  }, []);
  useEffect(() => { load(lat, lon); }, []);   // muat sekali dengan lokasi bawaan

  if (!d) return <div className="page"><div className="muted">Memuat…</div></div>;

  return <div className="page-wide fade">
    <h1 className="h1" style={{ marginBottom: 16 }}>
      Angin, hujan, dan tekstur tanah di kebunmu
    </h1>

    <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
      <div className="field">
        <label>Lintang kebun</label>
        <input type="number" step="0.0001" value={lat}
          onChange={e => setLat(parseFloat(e.target.value))} />
      </div>
      <div className="field">
        <label>Bujur kebun</label>
        <input type="number" step="0.0001" value={lon}
          onChange={e => setLon(parseFloat(e.target.value))} />
      </div>
      <button className="btn btn-primary btn-sm" disabled={busy}
        onClick={() => { saveCoords(lat, lon); load(lat, lon); }}>
        {busy ? "Memuat…" : "Perbarui"}</button>
    </div>

    {!d.ok ? <div className="note note-warn" style={{ marginTop: 20 }}>{d.error}</div> :
    <React.Fragment>
      <p className="fine" style={{ margin: "10px 0 20px" }}>
        {(lat === ENV_DEFAULT_LAT && lon === ENV_DEFAULT_LON)
          ? "Bawaan: Riau, Sumatra, contoh generik. Ganti dengan koordinat kebunmu "
            + "sendiri kalau ada."
          : "Koordinat kebunmu sendiri. Ganti kapan saja lewat kolom di atas."}
      </p>

      {d.using_cache && <div className="note note-soft" style={{ marginBottom: 16 }}>
        Tidak ada koneksi internet saat ini.
      </div>}

      <div className="grid g4" style={{ marginBottom: 20 }}>
        <div className="card" style={{ padding: 16 }}>
          <div className="muted" style={{ fontSize: 12 }}>Angin</div>
          <div className="mono" style={{ fontSize: 22, fontWeight: 500 }}>
            {num(d.weather.wind_speed_kmh, 1)} km/h</div>
          <div className="fine">arah {Math.round(d.weather.wind_dir_deg)}°</div>
        </div>
        <div className="card" style={{ padding: 16 }}>
          <div className="muted" style={{ fontSize: 12 }}>Hujan 30 hari</div>
          <div className="mono" style={{ fontSize: 22, fontWeight: 500 }}>
            {Math.round(d.weather.rain_30d_mm)} mm</div>
          <div className="fine">{d.rain.level}</div>
        </div>
        <div className="card" style={{ padding: 16 }}>
          <div className="muted" style={{ fontSize: 12 }}>Liat tanah</div>
          <div className="mono" style={{ fontSize: 22, fontWeight: 500 }}>
            {num(d.soil.clay_pct, 1)}%</div>
        </div>
        <div className="card" style={{ padding: 16 }}>
          <div className="muted" style={{ fontSize: 12 }}>Drainase (tekstur)</div>
          <div className="mono" style={{ fontSize: 22, fontWeight: 500,
            textTransform: "uppercase" }}>{d.drainage.level}</div>
        </div>
      </div>

    </React.Fragment>}
  </div>;
}

/* ------------------------------------------------- dialog syarat foto ----
   Muncul setelah analisis, HANYA kalau ada syarat foto yang tidak lolos, dan
   muncul di atas layar UNGGAH - bukan layar Hasil. Analisisnya sendiri sudah
   terlanjur jalan (syaratnya diukur dari hasil deteksi), tetapi mendaratkan
   pengguna di Hasil akan menyiratkan angkanya layak dipakai.

   `berat` menentukan pilihannya. Gagal berat = tidak ada peringkat sama sekali,
   jadi tidak ada yang bisa dilihat. Gagal ringan = peringkatnya ada tapi daftar
   pohonnya belum tentu lengkap, jadi pengguna boleh memilih tetap melihatnya. */
function SyaratDialog({ checks, adaHasil, onClose, onTetap }) {
  const semua = (checks || []).filter(c => !c.ok);
  const syarat = semua.filter(c => c.syarat_foto);
  if (!syarat.length) return null;
  const berat = syarat.some(c => c.berat);
  return <div className="modal-bg" onClick={onClose}>
    <div className="modal" onClick={e => e.stopPropagation()}>
      <div className="modal-h">
        <div>
          <div style={{ font: "600 17px var(--font-display)" }}>
            {berat ? "Foto ini belum bisa dibaca" : "Foto ini di luar rentang yang diuji"}
          </div>
          <div className="muted" style={{ fontSize: 12, marginTop: 3 }}>
            {berat
              ? "Tidak ada daftar prioritas yang bisa dibuat dari foto ini."
              : "Peringkatnya bisa dibuat, tapi keandalannya berkurang."}
          </div>
        </div>
        <button className="modal-x" onClick={onClose} aria-label="Tutup">✕</button>
      </div>
      {semua.map((c, i) => <div className="modal-r" key={i}>
        <div className="modal-rh">
          <b>{c.judul}</b>
          <span className={"yn " + (c.syarat_foto ? "yn-ya" : "yn-no")}>{c.punyamu}</span>
        </div>
        <div className="fine">Dibutuhkan: {c.syarat}</div>
        <div className="modal-s">{c.saran}</div>
      </div>)}
      <div className="modal-f">
        {!berat && adaHasil &&
          <button className="btn btn-ghost btn-sm" onClick={onTetap}>
            Tetap lihat hasilnya
          </button>}
        <button className="btn btn-primary btn-sm" onClick={onClose}>
          Pilih foto lain
        </button>
      </div>
    </div>
  </div>;
}

/* ------------------------------------------------------------------- app */
function App() {
  const [samples, setSamples] = useState([]);
  const [screen, setScreen] = useState("upload");
  const [phase, setPhase] = useState(0);
  const [data, setData] = useState(null);
  const [name, setName] = useState("");
  const [err, setErr] = useState(null);
  const [view, setView] = useState("app");
  const [eg, setEg] = useState(null);
  const [syarat, setSyarat] = useState(null);   // null = sudah ditutup / tak perlu
  const [batchData, setBatchData] = useState(null);
  const [batchProgress, setBatchProgress] = useState({ done: 0, total: 0, name: "" });
  const [fromBatch, setFromBatch] = useState(false);

  useEffect(() => {
    fetch("/api/samples").then(r => r.json()).then(j => setSamples(j.samples));
  }, []);
  useEffect(() => {
    if (view === "bukti" && !eg) fetch("/api/eg9pp").then(r => r.json()).then(setEg);
  }, [view, eg]);

  const run = async (opt) => {
    setScreen("proc"); setPhase(0); setErr(null); setFromBatch(false);
    setName(opt.file ? opt.file.name.slice(0, 18) : (samples[opt.sample] || {}).label || "");
    const tick = setInterval(() => setPhase(p => Math.min(p + 1, 3)), 430);
    const fd = new FormData();
    if (opt.file) fd.append("file", opt.file); else fd.append("sample", String(opt.sample));
    try {
      const r = await fetch("/api/analyze", { method: "POST", body: fd });
      if (!r.ok) throw new Error(await apiErrorMessage(r));
      const j = await r.json();
      clearInterval(tick); setData(j);
      // Hanya SYARAT foto yang memicu dialog. Butir informatif (mis. tidak ada
      // tajuk bergejala) ikut ditampilkan kalau dialognya terlanjur terbuka, tapi
      // tidak pernah memunculkannya sendiri - petak sehat bukan foto yang cacat.
      //
      // KENAPA LAYAR PROSES TETAP LEWAT. Syaratnya diukur DARI hasil deteksi -
      // jumlah sawit dan jarak tanam tidak bisa diketahui dari berkas mentah, jadi
      // detektor harus jalan lebih dulu. Yang bisa dijamin adalah ia tidak MENDARAT
      // di layar Hasil: kalau ada syarat yang gagal, kita kembali ke Unggah dan
      // dialognya muncul di sana.
      const gagal = (j.checks || []).filter(c => !c.ok && c.syarat_foto);
      setSyarat(gagal.length ? j.checks : null);
      setScreen(gagal.length ? "upload" : "res");
    } catch (e) { clearInterval(tick); setErr(String(e)); setScreen("upload"); }
  };

  // Sekuensial dengan sengaja, bukan Promise.all - detektornya jalan di CPU,
  // dan beberapa inferensi sekaligus akan berebut core yang sama alih-alih
  // benar-benar lebih cepat, sambil membuat progres per-foto tidak berarti.
  const runBatch = async (files) => {
    setScreen("bproc"); setErr(null);
    setBatchProgress({ done: 0, total: files.length, name: files[0].name });
    const results = [];
    for (let i = 0; i < files.length; i++) {
      setBatchProgress({ done: i, total: files.length, name: files[i].name });
      const fd = new FormData();
      fd.append("file", files[i]);
      try {
        const r = await fetch("/api/analyze", { method: "POST", body: fd });
        if (!r.ok) throw new Error(await apiErrorMessage(r));
        const j = await r.json();
        results.push(Object.assign({ srcName: files[i].name }, j));
      } catch (e) {
        results.push({ srcName: files[i].name, error: String(e) });
      }
    }
    setBatchProgress({ done: files.length, total: files.length, name: "" });
    setBatchData(results);
    setScreen("batch");
  };

  const step = screen === "upload" ? 0 : (screen === "proc" || screen === "bproc") ? 1 : 2;
  return <React.Fragment>
    <AppBar step={step} view={view} onView={setView} />
    {view === "app" && syarat && <SyaratDialog
      checks={syarat}
      adaHasil={!!(data && data.risk)}
      onClose={() => setSyarat(null)}
      onTetap={() => { setSyarat(null); setScreen("res"); }} />}
    {err && <div className="page"><div className="note note-warn">Gagal: {err}</div></div>}
    {view === "bukti" ? <Evidence d={eg} />
      : view === "env" ? <EnvContext />
      : <React.Fragment>
      {screen === "upload" && <Upload samples={samples} onRun={run} onRunBatch={runBatch} />}
      {screen === "proc" && <Processing phase={phase} name={name} />}
      {screen === "bproc" && <BatchProcessing progress={batchProgress} />}
      {screen === "res" && data && <Results d={data}
        onReset={() => setScreen(fromBatch ? "batch" : "upload")}
        backLabel={fromBatch ? "← Kembali ke survei" : undefined}
        onOpenEnv={() => setView("env")} />}
      {screen === "batch" && batchData && <BatchResults batchData={batchData}
        onReset={() => setScreen("upload")}
        onOpenPhoto={i => { setData(batchData[i]); setFromBatch(true); setScreen("res"); }} />}
    </React.Fragment>}
  </React.Fragment>;
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
