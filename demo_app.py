"""UI demonstrasi SawitGuard-GNN.

    streamlit run demo_app.py

Berkas ini HANYA menyusun tata letak. Seluruh perhitungan DAN seluruh pembuatan
gambar ada di `demo_core.py`, yang bisa dijalankan sendiri:

    python demo_core.py     # cetak semua angka + render keempat gambar ke PNG

Pemisahan itu disengaja. Grafiknya bisa diperiksa dengan mata tanpa menyalakan
server, dan tidak mungkin ada versi UI yang berbeda dari versi pratinjau.

Empat layar mengikuti DEMO_BRIEF.md. Layar 3 bukan pelengkap - di situlah letak
kontribusinya, dan ia sengaja diletakkan DI ANTARA sisi citra dan sisi peramalan
supaya tidak mungkin dilewati saat presentasi.
"""
import os
import tempfile

import numpy as np
import streamlit as st

import demo_core as core

P = core.PALETTE
n = core._num
st.set_page_config(page_title="SawitGuard", page_icon="🌴", layout="wide")

# --- Chrome: token SawitGuard Design System (claude.ai/design).
#
# Streamlit tidak bisa memuat komponen JSX design system itu, jadi yang dipinjam
# adalah BAHASA VISUAL-nya - font, warna, radius, bentuk pil - lewat CSS. Hasilnya
# mirip, bukan identik; komponen aslinya tetap jadi acuan kalau nanti aplikasi ini
# dibangun ulang sebagai web app.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --lime-500:#C3EC3C; --lime-600:#A8D122; --green-500:#4EC75B; --green-700:#27803A;
  --ink-900:#1A1C17; --ink-700:#3D423A; --ink-500:#6A7065; --ink-200:#D4D9CF;
  --mint-50:#EDF2E9; --mint-100:#E2EADB; --white:#FFFFFF;
  --radius-md:16px; --radius-lg:24px; --radius-pill:999px;
  --shadow-card:0 1px 2px rgba(26,28,23,.05),0 4px 16px rgba(26,28,23,.06);
  --ease-out:cubic-bezier(.16,1,.3,1);
}
html,body,[class*="css"]{font-family:'Instrument Sans',system-ui,sans-serif;}
h1,h2,h3,h4{font-family:'Space Grotesk',system-ui,sans-serif;
  letter-spacing:-.02em; color:var(--ink-900);}
h1{font-weight:700;}
code,pre,[data-testid="stMetricValue"]{font-family:'JetBrains Mono',ui-monospace,monospace;}

/* Kartu putih di atas tint mint - dipisahkan oleh keputihan, bukan bayangan */
[data-testid="stMetric"]{background:var(--white); border:1px solid var(--ink-200);
  border-radius:var(--radius-md); padding:14px 16px; box-shadow:var(--shadow-card);}
[data-testid="stMetricLabel"]{color:var(--ink-500); font-size:13px;}
[data-testid="stMetricValue"]{color:var(--ink-900); font-size:26px;}

/* Tombol: pil, satu aksen keras */
.stButton>button{border-radius:var(--radius-pill); border:1px solid transparent;
  font-family:'Instrument Sans',sans-serif; font-weight:600; padding:.5rem 1.25rem;
  transition:all var(--ease-out) 160ms;}
.stButton>button[kind="primary"]{background:var(--lime-500); color:var(--ink-900);}
.stButton>button[kind="primary"]:hover{background:var(--lime-600); color:var(--ink-900);}
.stButton>button:active{transform:scale(.98);}

/* Tab sebagai chip pil */
.stTabs [data-baseweb="tab-list"]{gap:8px; border-bottom:none;}
.stTabs [data-baseweb="tab"]{background:var(--mint-100); border-radius:var(--radius-pill);
  padding:8px 18px; color:var(--ink-700); font-weight:500;}
.stTabs [aria-selected="true"]{background:var(--ink-900) !important; color:var(--mint-50) !important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none;}

/* Kotak pesan & tabel */
[data-testid="stAlert"]{border-radius:var(--radius-md); border:1px solid var(--ink-200);}
[data-testid="stDataFrame"]{border-radius:var(--radius-md); overflow:hidden;}
[data-testid="stSidebar"]{background:var(--white); border-right:1px solid var(--ink-200);}
[data-testid="stImage"] img,[data-testid="stPyplot"] img{border-radius:var(--radius-lg);}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Batas yang selalu berlaku")
    st.markdown(
        "- Keluaran adalah **peringkat**, bukan peluang. Skor 0,51 berarti *lebih "
        "berisiko daripada* yang berskor 0,30 — **bukan** 51% kemungkinan sakit.\n"
        "- Label citra adalah **kesehatan tajuk umum**, bukan BSR/Ganoderma "
        "terverifikasi lapangan.\n"
        "- Model dilatih di **kebun percobaan pemuliaan**, bukan kebun produksi.\n"
        "- Sawit **disensor** tidak dihitung sehat — ia keluar dari kumpulan berisiko."
    )
    st.divider()
    st.caption("Seluruh demo berjalan di CPU. GPU hanya dibutuhkan untuk melatih.")

st.title("SawitGuard-GNN")
st.markdown(
    "**Lapisan 1 adalah mata** — diberi foto drone, ia menemukan setiap sawit dan "
    "letaknya.  \n**Lapisan 2 adalah ingatan** — diberi satu kebun yang dipantau "
    "25 tahun, ia menebak sawit sehat mana yang paling mungkin sakit berikutnya.  \n"
    "**Keduanya dari kebun berbeda dan tidak kami sambung.** Yang kami lakukan: "
    "mengukur apakah keduanya *akan* cocok."
)

t1, t2, t3, t4 = st.tabs(["Langkah 1 · Daftarkan kebun",
                          "Langkah 1 · Peta kontak",
                          "Langkah 2 · Apa yang masih kurang",
                          "Langkah 3 · Peringkat risiko"])

# ---------------------------------------------------------------- 1. deteksi
with t1:
    st.subheader("Dari foto drone ke pusat tajuk")
    samples = core.sample_images(6)
    c1, c2 = st.columns([2, 1])
    with c2:
        up = st.file_uploader("Unggah citra sendiri", type=["jpg", "jpeg", "png"])
        pick = st.selectbox("atau pilih contoh", samples,
                            format_func=lambda p: os.path.basename(p)[:26] + "…")
        go = st.button("Jalankan deteksi", type="primary")

    if go or "det" not in st.session_state:
        path = pick
        if up is not None:
            path = os.path.join(tempfile.mkdtemp(prefix="sg_"), up.name)
            with open(path, "wb") as fh:
                fh.write(up.getbuffer())
        with st.spinner("mendeteksi…"):
            st.session_state["det"] = core.detect_image(path)
            st.session_state["det_path"] = path

    df, info = st.session_state["det"]
    path = st.session_state["det_path"]

    with c1:
        from PIL import Image
        st.pyplot(core.fig_detection(np.array(Image.open(path).convert("RGB")), df),
                  width='stretch')

    with c2:
        st.metric("Pohon terdeteksi", info["n"])
        if info.get("ok_n"):
            st.metric("Jarak tanam (estimasi)", "%s px" % n(info["spacing_px"], 0))
            st.markdown(
                ("✅ **Skala %s×** — di dalam jendela data latih (0,80–1,25×)"
                 if info["ok_scale"] else
                 "⚠️ **Skala %s×** — DI LUAR jendela 0,80–1,25×. Citra harus "
                 "di-resample; angka graf tidak sebanding.") % n(info["scale_ratio"]))
        else:
            st.warning("Terlalu sedikit pohon untuk mengestimasi jarak tanam; "
                       "graf tidak dibangun.")
        st.caption("Bobot `%s` · conf %s" % (info["weights"], n(info["conf"])))

    st.info(
        "F1 pusat tajuk **0,960 ± 0,024** diukur *leave-one-ortho-out* di dalam **satu "
        "kebun**. Citra dari kebun, sensor, atau ketinggian lain adalah domain yang "
        "belum pernah diukur — angka itu tidak berpindah ke sana."
    )
    with st.expander("Lihat tabel deteksi"):
        st.dataframe(df, width='stretch', height=260)

# ---------------------------------------------------------------- 2. graf
with t2:
    st.subheader("Pusat tajuk menjadi jaring tetangga")
    df, info = st.session_state["det"]
    if not info.get("ok_n"):
        st.warning("Jalankan deteksi dulu di layar 1 pada citra yang cukup luas.")
    else:
        xy = df[["cx", "cy"]].values
        edges = core.edges_within(xy, info["r_graph_px"])
        c1, c2 = st.columns([2, 1])
        with c1:
            st.pyplot(core.fig_graph(xy, edges), width='stretch')
        with c2:
            st.metric("Sisi (pasangan tetangga)", len(edges))
            st.metric("Derajat rata-rata", n(info["deg_all"]))
            st.metric("Pohon bagian dalam", n(info["deg_inner"]),
                      help="Hanya pohon yang jaraknya > radius dari tepi. Pohon tepi "
                           "berderajat rendah karena kebunnya habis, bukan karena "
                           "grafnya beda.")
            st.caption("Radius = 1,5 × jarak tanam = %s px" % n(info["r_graph_px"], 0))
        st.markdown(
            "Satu garis berarti dua sawit cukup dekat untuk **akarnya bersentuhan**. "
            "Angka “derajat” inilah yang dibandingkan dengan kebun Eg9PP di layar 3."
        )

# ---------------------------------------------------------------- 3. antarmuka
with t3:
    st.subheader("Seberapa siap kebun kamu?")
    st.markdown(
        "Dua hal diperiksa di sini. Pertama, **apakah geometri kebun kamu cocok** dengan "
        "kebun tempat model dilatih — kalau jarak tanamnya beda jauh, model tidak akan "
        "berlaku. Kedua, **bahan apa yang masih kurang** sebelum peringkat risiko bisa "
        "dihitung untuk kebunmu."
    )
    _, uinfo = st.session_state["det"]
    u_deg = uinfo.get("deg_inner") if uinfo.get("ok_scale") else None
    u_n = uinfo.get("n_inner", 0)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(core.fig_interface(u_deg, u_n), width='stretch')
    with c2:
        st.metric("Selisih antarmuka", "%s%%" % n(core.FACTS["gap_pct"], 1),
                  help="acuan Lapisan 1 (5,54) terhadap Eg9PP (5,74)")
        if u_deg is not None:
            st.markdown(
                "Titik hijau adalah **citra yang barusan kamu proses**, diukur dengan "
                "cara yang sama. Ia dihitung dari %d pohon bagian dalam saja — satu "
                "ubin jauh lebih sempit daripada satu ortomosaik penuh, jadi wajar "
                "kalau ia bergoyang di sekitar acuan 5,54." % u_n)
        else:
            st.markdown(
                "Citra yang diproses di Langkah 1 **tidak lolos pemeriksaan skala** "
                "atau terlalu sempit, jadi ia tidak ditempatkan di sumbu ini.")
        st.markdown(
            "Angka Lapisan 1 berasal dari **prediksi detektor**, bukan kotak acuan — "
            "jadi ini yang benar-benar dicapai, bukan batas atas.\n\n"
            "**Yang harus dinyatakan:** 5,74 jatuh **di luar** pita ±0,12. Kalimat yang "
            "sah adalah “berselisih 3,5%”, **bukan** “keduanya sama”."
        )
    st.divider()
    rd = core.readiness(uinfo)
    have = sum(1 for r in rd if r["ada"])
    st.markdown("#### Kesiapan kebun kamu: **%d dari %d bahan**" % (have, len(rd)))
    st.markdown(
        "Peringkat risiko butuh empat bahan. Foto drone memberi dua di antaranya "
        "seketika; dua sisanya datang dari waktu dan arsip, bukan dari kamera."
    )
    for r in rd:
        c_a, c_b = st.columns([1, 2])
        c_a.markdown("%s **%s**" % ("✅" if r["ada"] else "⬜", r["bahan"]))
        c_b.markdown("<span style='color:%s'>%s — <em>%s</em></span>"
                     % (P["ink"] if r["ada"] else P["muted"], r["punyamu"], r["cara"]),
                     unsafe_allow_html=True)

    st.info(
        "Begitu dua bahan yang tersisa tersedia, peringkat risiko bisa dihitung untuk "
        "kebun kamu sendiri. **Langkah 3 memperlihatkan bentuk keluarannya** — memakai "
        "kebun Eg9PP, satu-satunya kebun yang datanya sudah lengkap 25 tahun."
    )
    with st.expander("Rincian teknis: 24 kolom masukan model"):
        st.dataframe(
            [{"Blok": b, "Kolom": k, "Bisa dari satu foto?": "sebagian" if ok else "TIDAK",
              "Alasan": why} for b, k, ok, why in core.FEATURE_BLOCKS],
            width='stretch', hide_index=True)
        st.markdown(
            "Memaksa checkpoint menerima tajuk hasil deteksi dengan mengisi **nol** pada "
            "kolom yang hilang akan tetap mengeluarkan angka, dan angka itu akan tampak "
            "masuk akal. Ia tidak berarti apa-apa — jangan dilakukan."
        )

# ---------------------------------------------------------------- 4. risiko
with t4:
    st.subheader("Peringkat risiko di kebun Eg9PP")
    st.info(
        "**Inilah bentuk keluaran yang akan kamu terima** begitu kebunmu punya keempat "
        "bahan di Langkah 2.  \nYang ditampilkan di bawah adalah kebun **Eg9PP** — "
        "1.200 sawit, 45 sensus, 25 tahun, Ganoderma terverifikasi lapangan — **bukan** "
        "kebun dari foto yang kamu proses. Satu foto belum bisa menghasilkan peringkat "
        "ini karena “memburuk” butuh minimal dua titik waktu."
    )
    dfr = core.load_risk()
    s = core.risk_summary(dfr)
    a, b, c, d = st.columns(4)
    a.metric("Sawit dipantau", s["n_total"])
    b.metric("Dinilai sekarang", s["n_risk"])
    c.metric("Keluar dari penilaian", s["n_out"],
             help="sudah bergejala, mati, atau disensor — bukan sehat")
    d.metric("Lama pemantauan", "%d sensus · %d th"
             % (core.FACTS["n_censuses"], core.FACTS["n_years"]))

    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(core.fig_risk_map(dfr), width='stretch')
    with c2:
        st.markdown("**Sepuluh paling berisiko**")
        st.dataframe(s["top10"].rename(columns={
            "rank": "#", "palm_id": "sawit", "parcel": "petak", "logit": "skor",
            "risk_percentile": "persentil", "n_sick_neighbours": "tetangga sakit",
            "n_neighbours": "tetangga"}), width='stretch', hide_index=True)
        st.markdown(
            "**Model membaca tetangga, bukan pohon itu sendiri.** Setiap sawit yang "
            "dinilai berstatus sehat, jadi statusnya sendiri tidak membawa informasi "
            "apa pun — semuanya datang lewat tetangganya."
        )
        e, f_, g = st.columns(3)
        e.metric("10 teratas", n(s["sick_nb_top10"], 1), help="rata-rata tetangga sakit")
        f_.metric("semua", n(s["sick_nb_all"], 1))
        g.metric("10 teraman", n(s["sick_nb_bot10"], 1))

    st.warning(
        "Warna adalah **kuintil**, bukan peluang. Skor berkisar %s…%s dan hanya "
        "berarti urutan. Checkpoint ini dilatih pada seluruh 1.200 sawit tanpa kumpulan "
        "uji tersendiri — **tidak ada angka performa yang boleh dikutip darinya**; "
        "angka performa datang dari `results_real.csv` (leave-one-parcel-out)."
        % (n(s["logit_min"]), n(s["logit_max"]))
    )
