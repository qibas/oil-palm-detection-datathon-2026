/* SawitGuard — frontend React.
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
const { useState, useEffect, useRef, useCallback } = React;

const Q = ["var(--q1)", "var(--q2)", "var(--q3)", "var(--q4)", "var(--q5)"];
const QHEX = ["#EE9A87", "#E5484D", "#B32B30", "#822024", "#4F1315"];
const GREEN = "#4EC75B", DANGER = "#E5484D", INK = "#1A1C17";
const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v))
  ? "—" : Number(v).toFixed(d).replace(".", ",");

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
function AppBar({ step, view, onView }) {
  const S = ["Unggah", "Proses", "Hasil"];
  return <div className="appbar">
    <div className="mark">Prediksi <i>Pohon Berisiko</i></div>
    <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
      {view === "app" && <div className="stepper">
        {S.map((label, i) => <React.Fragment key={label}>
          {i > 0 && <span className="sep" />}
          <div className={"s " + (step > i ? "done" : step === i ? "active" : "")}>
            <span className="n">{step > i ? "✓" : i + 1}</span>{label}
          </div>
        </React.Fragment>)}
      </div>}
      <button className={"navlink" + (view === "bukti" ? " on" : "")}
        onClick={() => onView(view === "bukti" ? "app" : "bukti")}>
        {view === "bukti" ? "← Kembali ke aplikasi" : "Bukti & validasi"}
      </button>
    </div>
  </div>;
}

/* --------------------------------------------------------------- layar 1 */
function Upload({ samples, onRun }) {
  const [sel, setSel] = useState(0);
  const [over, setOver] = useState(false);
  const fileRef = useRef();

  return <div className="page fade">
    <h1 style={{ font: "700 44px/1.08 var(--font-display)", letterSpacing: "-.02em" }}>
      Di mana tim kamu<br />harus melihat duluan?
    </h1>
    <p className="sec" style={{ margin: "14px 0 32px", maxWidth: 520 }}>
      Unggah satu foto drone kebunmu. Sistem menemukan setiap sawit, membangun
      graf kontak akarnya, lalu memeringkat sawit sehat mana yang paling berisiko
      tertular berikutnya.
    </p>

    <div className={"drop" + (over ? " over" : "")}
      onDragOver={e => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={e => { e.preventDefault(); setOver(false);
        const f = e.dataTransfer.files[0]; if (f) onRun({ file: f }); }}
      onClick={() => fileRef.current.click()}>
      <div className="ico"><Icon d={I_UP} size={22} /></div>
      <div className="big">Jatuhkan citra drone di sini</div>
      {/* Syaratnya sengaja TIDAK dijabarkan di sini. Empat kartu teks membuat layar
          pertama terasa seperti formulir, dan pengguna belum punya konteks untuk
          menilainya. Pemeriksaan lengkap dijalankan SESUDAH unggah dan hanya muncul
          sebagai dialog kalau ada yang tidak lolos - lihat `SyaratDialog`. */}
      <div className="muted" style={{ marginTop: 6, fontSize: 13 }}>
        atau klik untuk memilih berkas &nbsp;·&nbsp; JPG / PNG &nbsp;·&nbsp;
        tegak dari atas, minimal 20 sawit
      </div>
      <input ref={fileRef} type="file" accept="image/*" hidden
        onChange={e => e.target.files[0] && onRun({ file: e.target.files[0] })} />
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

    <p className="fine" style={{ marginTop: 28 }}>
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
    <div className="card" style={{ padding: 36, borderRadius: "var(--radius-xl)" }}>
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
      <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
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

/* ---------------------------------------------------------- kanvas overlay */
function Overlay({ src, w, h, mode, data, showGreys }) {
  const cv = useRef(), box = useRef();
  const draw = useCallback(() => {
    const c = cv.current, wrap = box.current;
    if (!c || !wrap) return;
    const W = wrap.clientWidth, H = W * h / w;
    c.width = W * devicePixelRatio; c.height = H * devicePixelRatio;
    c.style.height = H + "px";
    const g = c.getContext("2d");
    g.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    g.clearRect(0, 0, W, H);

    if (mode === "graph") {
      g.strokeStyle = "rgba(255,255,255,.45)"; g.lineWidth = 1.3;
      (data.edges || []).forEach(([x1, y1, x2, y2]) => {
        g.beginPath(); g.moveTo(x1 * W, y1 * H); g.lineTo(x2 * W, y2 * H); g.stroke();
      });
    }
    const dot = (x, y, fill, r = 5.5) => {
      g.beginPath(); g.arc(x * W, y * H, r, 0, 7);
      g.fillStyle = fill; g.fill();
      g.lineWidth = 1.5; g.strokeStyle = "#fff"; g.stroke();
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
      (data.crowns || []).forEach(c2 =>
        dot(c2.x, c2.y, c2.cls === "Unhealthy" ? DANGER : GREEN));
    }
  }, [mode, data, w, h, showGreys]);

  useEffect(() => { draw(); }, [draw]);
  useEffect(() => {
    const r = () => draw(); window.addEventListener("resize", r);
    return () => window.removeEventListener("resize", r);
  }, [draw]);

  return <div className="viz" ref={box}>
    <img src={src} alt="" onLoad={draw} />
    <canvas ref={cv} />
  </div>;
}

/* --------------------------------------------------------------- layar 3 */
function Results({ d, onReset }) {
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
    <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between",
      gap: 16, flexWrap: "wrap" }}>
      <div>
        <h1 style={{ font: "700 34px/1.1 var(--font-display)", letterSpacing: "-.02em" }}>
          Peta risiko
        </h1>
        <div style={{ marginTop: 8, display: "flex", gap: 8, flexWrap: "wrap" }}>
          <span className="badge inverse">{det.n} sawit</span>
          {ok && <span className="badge lime">{risk.n_risk} dinilai</span>}
          <span className="badge neutral">{d.name.slice(0, 22)}…</span>
        </div>
      </div>
      <div style={{ display: "flex", gap: 10 }}>
        <button className="btn btn-ghost btn-sm" onClick={onReset}>Foto baru</button>
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

    <div style={{ display: "flex", gap: 8, margin: "20px 0 16px", alignItems: "center",
      flexWrap: "wrap" }}>
      {[["risk", "Peringkat risiko"], ["graph", "Graf kontak"], ["crowns", "Deteksi"]]
        .map(([k, l]) =>
          <button key={k} className={"chip" + (mode === k ? " on" : "")}
            onClick={() => setMode(k)}>{l}</button>)}
      <div style={{ marginLeft: "auto", display: "flex", gap: 18,
        alignItems: "center", flexWrap: "wrap" }}>
        <div className={"sw" + (soft ? " on" : "")} onClick={() => setSoft(!soft)}>
          <span className="track"><i /></span>Skor gejala kontinu
        </div>
        <div className={"sw" + (greys ? " on" : "")} onClick={() => setGreys(!greys)}>
          <span className="track"><i /></span>Tampilkan yang sudah bergejala
        </div>
      </div>
    </div>

    {soft && d.risk_soft && <div className="note note-warn" style={{ marginBottom: 14 }}>
      <b>Mode skor kontinu — belum tervalidasi.</b> Kolom gejala diisi
      <b> keyakinan detektor</b> (0–1), bukan 0/1. Petanya jadi lebih halus
      ({d.risk_soft.n_tingkat} tingkat, bukan {d.risk.n_tingkat}) — tetapi model
      dilatih pada status <b>biner</b> terverifikasi lapangan, dan Eg9PP tidak punya
      ground truth kontinu untuk menguji apakah gradasi ini <b>lebih benar</b>.
      Yang bertambah adalah resolusi tampilan; bukti bahwa peringkatnya membaik
      <b> belum ada</b>. Matikan sakelar untuk kembali ke mode biner yang terukur.
    </div>}

    <div className="grid g-results">
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div className="card" style={{ padding: 20 }}>
        <Overlay src={d.image} w={d.w} h={d.h} mode={mode} data={dd} showGreys={greys} />
        <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 16 }}>
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
              <span>✕ sudah bergejala — jadi sumber penularan, tidak ikut diperingkat</span>
            </div>
          </React.Fragment> : <div className="legend">
            <span className="sw2"><i className="dot" style={{ background: GREEN }} />sehat</span>
            <span className="sw2"><i className="dot" style={{ background: DANGER }} />tidak sehat</span>
            {mode === "graph" && <span>garis = akar berpotensi bersentuhan</span>}
          </div>}
        </div>
      </div>
    {det.ok_n && <div className="note note-soft">
      {det.ok_scale
        ? <React.Fragment>Skala citra <b>{num(det.scale_ratio)}×</b> — di dalam jendela
          data latih (0,80–1,25×), jadi derajat graf sebanding dengan acuan
          5,54 ± 0,12.</React.Fragment>
        : <React.Fragment><b>Skala citra {num(det.scale_ratio)}× di luar jendela
          0,80–1,25×.</b> Angka graf di atas tidak sebanding dengan angka mana pun di
          repositori ini.</React.Fragment>}
    </div>}

    {/* Bahasa pengguna di depan. Angka performa SENGAJA tidak ada di layar utama:
        aturan paket ini menyatakan checkpoint demo dilatih tanpa held-out sehingga
        tidak ada angka performa yang boleh dikutip darinya - memasang 1,45x/1,27x
        di sini mengundang persis salah-baca itu. Rinciannya pindah ke expander,
        dengan keterangan bahwa angkanya dari lari tervalidasi, bukan dari layar ini. */}
    <div className="note note-soft">
      Sistem menandai sawit yang <b>kondisinya terlihat buruk dari udara</b> —
      itu belum tentu Ganoderma. Peringkatnya untuk <b>memandu urutan pemeriksaan</b>,
      bukan menggantikan cek lapangan.
    </div>

    <details className="tech">
      <summary>Lihat batasan teknis</summary>
      <div className="body">
        <dl>
          <dt>Label citra</dt>
          <dd>Kelas detektor adalah kesehatan tajuk umum, bukan Ganoderma
            terverifikasi lapangan. Tidak ada diagnosis di sini.</dd>
          <dt>Asal model</dt>
          <dd>Dilatih di kebun percobaan pemuliaan (Eg9PP, 2 parcel), bukan kebun
            produksi. Efeknya sendiri berbeda 2,6× antar kedua parcel itu.</dd>
          <dt>Kekerabatan</dt>
          <dd>Efek graf mengandung 36% kontaminasi kekerabatan — famili sekandung
            ditanam berdampingan (null dalam-famili+petak, 200 permutasi, 0/200).</dd>
          <dt>Masukan gejala</dt>
          <dd>Model dilatih pada status terverifikasi lapangan; di layar ini kolomnya
            diisi kelas detektor. Ongkos substitusi itu terukur pada lari
            leave-one-parcel-out — bukan pada layar ini.</dd>
          <dt>Kalibrasi</dt>
          <dd>Model memeringkat baik tetapi <b>menaksir buruk</b>, dan itu terukur:
            pada uji leave-one-parcel-out, sawit ber-sigmoid(skor) 0,50–0,60
            sesungguhnya sakit <b>23,6%</b>, bukan 55% — meleset 31 poin. Penyebabnya
            focal loss (α 0,75) yang sengaja membobot kelas langka agar model belajar
            <i>membedakan</i>, bukan agar angkanya benar. Karena itu keluarannya
            peringkat, dan sigmoid-nya tidak boleh disajikan sebagai persentase.</dd>
          <dt>Checkpoint layar ini</dt>
          <dd>Dilatih pada seluruh 1.200 sawit <b>tanpa kumpulan uji</b>. Ia artefak
            inferensi, bukan evaluasi: <b>tidak ada angka performa yang boleh dikutip
            dari layar ini</b>. Angka tervalidasi ada di <b>Bukti &amp; validasi</b>.</dd>
        </dl>
        {risk && <div className="fine" style={{ marginTop: 12 }}>
          checkpoint: {risk.ckpt}</div>}
      </div>
    </details>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className="grid g4" style={{ gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          <div className="card" style={{ padding: 14 }}>
            <div className="muted" style={{ fontSize: 12 }}>Jarak tanam</div>
            <div className="mono" style={{ fontSize: 21, fontWeight: 500 }}>
              {det.spacing_px ? num(det.spacing_px, 0) + " px" : "—"}</div>
          </div>
          <div className="card" style={{ padding: 14 }}>
            <div className="muted" style={{ fontSize: 12 }}>Derajat graf</div>
            <div className="mono" style={{ fontSize: 21, fontWeight: 500 }}>
              {num(det.deg_inner)}</div>
          </div>
        </div>

        {ok ? <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          <div style={{ padding: "16px 18px 10px", display: "flex",
            justifyContent: "space-between", alignItems: "baseline" }}>
            <div style={{ font: "600 17px var(--font-display)" }}>Daftar prioritas</div>
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
          <div style={{ font: "600 17px var(--font-display)", marginBottom: 10 }}>
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
            <div style={{ font: "600 17px var(--font-display)" }}>Pusat wabah</div>
            <span className="badge neutral">tanpa model</span>
          </div>
          <p className="muted" style={{ fontSize: 12, margin: "6px 0 12px" }}>
            Tajuk bergejala yang saling bersentuhan lewat graf, dikelompokkan.
            Pernyataan geometris — tidak meramal apa pun.
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

function Evidence({ d }) {
  if (!d) return <div className="page"><div className="muted">Memuat…</div></div>;
  const f = d.facts, mx = Math.max.apply(null, d.levels.map(l => l.n));
  const pair = t => num(t[0], 4) + " ± " + num(t[1], 4);
  return <div className="page-wide fade">
    <h1 style={{ font: "700 34px/1.1 var(--font-display)", letterSpacing: "-.02em" }}>
      Di mana angkanya benar-benar diukur
    </h1>
    <p className="sec" style={{ margin: "10px 0 24px", maxWidth: 720 }}>
      Layar aplikasi berjalan di atas <b>satu foto</b>, dan di sana model hanya
      menerima satu angka — jumlah tetangga bergejala, sehingga peringkatnya
      identik dengan menghitung tetangga. Nilai grafnya baru terlihat di kebun
      <b> Eg9PP</b>: {d.n_total} sawit, 45 sensus, 25 tahun, Ganoderma terverifikasi
      lapangan. Seluruh angka performa di paket ini datang dari sini.
    </p>

    <div className="grid g4" style={{ marginBottom: 20 }}>
      {[["Sawit dipantau", d.n_total], ["Dinilai", d.n_risk],
        ["Sakit / mati / disensor", d.n_out],
        ["Laju gejala di kebun", num(100 * d.sick_rate, 1) + "%"]].map(kv =>
        <div className="card" key={kv[0]} style={{ padding: 16 }}>
          <div className="muted" style={{ fontSize: 12 }}>{kv[0]}</div>
          <div className="mono" style={{ fontSize: 22, fontWeight: 500 }}>{kv[1]}</div>
        </div>)}
    </div>

    <div className="grid g-results">
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div className="card">
        <div style={{ font: "600 17px var(--font-display)", marginBottom: 12 }}>
          Peta risiko Eg9PP · sensus 41 (tahun ke-24)
        </div>
        <Lattice d={d} />
        <div style={{ marginTop: 14 }}>
          <div className="ramp">{QHEX.map(c => <i key={c} style={{ background: c }} />)}</div>
          <div className="ramp-lab"><span>Risiko lebih rendah</span>
            <span>Risiko lebih tinggi</span></div>
        </div>
        <div className="legend" style={{ marginTop: 10 }}>
          <span>▲ bergejala ({d.status_out.S})</span>
          <span>✕ mati ({d.status_out.D})</span>
          <span>▫ disensor ({d.status_out.C})</span>
        </div>
      </div>
      <div className="card">
        <div style={{ font: "600 17px var(--font-display)", marginBottom: 10 }}>
          Yang graf sumbangkan</div>
        <div className="cmp">
          <span className="h">Model</span><span className="h n">AP dalam-sensus</span><span className="h n">Lift</span>
          <span>Tanpa graf</span><span className="n">{pair(f.ap_nograph_within)}</span><span className="n muted">acak</span>
          <span>Foto, 1 kolom</span><span className="n">{pair(f.ap_1col_within)}</span><span className="n muted">1,45×</span>
          <span>Foto, 6 kolom</span><span className="n">{pair(f.ap_photo_within)}</span><span className="n muted">1,61×</span>
          <span>Penuh, 24 kolom</span><span className="n">{pair(f.ap_full_within)}</span><span className="n muted">1,54×</span>
        </div>
        <div className="note note-soft" style={{ marginTop: 12, padding: 14 }}>
          Sumbangan <b>peta kontak yang benar</b>: +{num(f.struktur[0], 4)} ±
          {" "}{num(f.struktur[1], 4)}, <b>{f.struktur[2]}</b> tanda searah. Bertahan
          di <b>{f.perm_strict[2]}</b> permutasi dalam-famili+petak
          (z +{num(f.perm_strict[1], 2)}).
        </div>
      </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className="card">
          <div style={{ font: "600 17px var(--font-display)" }}>
            Kenapa di sini 5 pita, di foto cuma 2</div>
          <p className="sec" style={{ fontSize: 13, margin: "8px 0 0" }}>
            Sebaran tetangga bergejala per sawit. Di sini ada {d.levels.length} hitungan
            berbeda; di ubin drone yang sehat cuma dua.
          </p>
          <div className="levbar">
            {d.levels.map(l => <div key={l.nb}
              style={{ height: (100 * l.n / mx) + "%" }}><span>{l.nb}</span></div>)}
          </div>
          <div className="fine" style={{ marginTop: 24 }}>jumlah tetangga bergejala →</div>
        </div>

        <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          <div style={{ padding: "16px 18px 8px", font: "600 17px var(--font-display)" }}>
            Sepuluh teratas Eg9PP</div>
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
              Tetangga sakit rata-rata: <b>{num(d.nb_top10, 2)}</b> pada 10 teratas,
              {" "}<b>{num(d.nb_all, 2)}</b> pada semua, <b>{num(d.nb_bot10, 2)}</b> pada
              10 teraman. Model membaca tetangga, bukan pohon itu sendiri.
            </div>
          </div>
        </div>
      </div>
    </div>

    <div className="note note-limit" style={{ marginTop: 18 }}>
      <b>Manfaat operasionalnya.</b> Memeriksa 5% teratas menemukan satu kasus per
      {" "}<b>{num(f.per_case_model, 1)} pohon</b>, bukan {num(f.per_case_random, 1)}
      {" "}— <b>{num(f.lift_top5, 2)}× lebih efisien</b>.
      {" "}<b>Batasnya:</b> efek graf mengandung {f.kinship_pct}% kontaminasi
      kekerabatan; lewat jalur foto {f.signal_kept_pct}% sinyal bertahan; dan seluruh
      angka ini dari <b>satu kebun percobaan, dua parcel</b> — efeknya sendiri
      berbeda 2,6× antar kedua parcel itu.
    </div>
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
          <div style={{ font: "600 18px var(--font-display)" }}>
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

  useEffect(() => {
    fetch("/api/samples").then(r => r.json()).then(j => setSamples(j.samples));
  }, []);
  useEffect(() => {
    if (view === "bukti" && !eg) fetch("/api/eg9pp").then(r => r.json()).then(setEg);
  }, [view, eg]);

  const run = async (opt) => {
    setScreen("proc"); setPhase(0); setErr(null);
    setName(opt.file ? opt.file.name.slice(0, 18) : (samples[opt.sample] || {}).label || "");
    const tick = setInterval(() => setPhase(p => Math.min(p + 1, 3)), 430);
    const fd = new FormData();
    if (opt.file) fd.append("file", opt.file); else fd.append("sample", String(opt.sample));
    try {
      const r = await fetch("/api/analyze", { method: "POST", body: fd });
      if (!r.ok) throw new Error("HTTP " + r.status);
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

  const step = screen === "upload" ? 0 : screen === "proc" ? 1 : 2;
  return <React.Fragment>
    <AppBar step={step} view={view} onView={setView} />
    {view === "app" && syarat && <SyaratDialog
      checks={syarat}
      adaHasil={!!(data && data.risk)}
      onClose={() => setSyarat(null)}
      onTetap={() => { setSyarat(null); setScreen("res"); }} />}
    {err && <div className="page"><div className="note note-warn">Gagal: {err}</div></div>}
    {view === "bukti" ? <Evidence d={eg} /> : <React.Fragment>
      {screen === "upload" && <Upload samples={samples} onRun={run} />}
      {screen === "proc" && <Processing phase={phase} name={name} />}
      {screen === "res" && data && <Results d={data} onReset={() => setScreen("upload")} />}
    </React.Fragment>}
  </React.Fragment>;
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
