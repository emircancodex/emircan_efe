import os
import io
import tempfile
import numpy as np
import wave
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import median_filter


# ============================================================
# Sayfa Ayarlari
# ============================================================
st.set_page_config(
    page_title="VAD Analiz Araci",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Ozel CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp, .stApp > header, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], .main .block-container {
        background-color: #ffffff !important;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #333;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 4px;
    }

    .metric-card.green { background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%); }
    .metric-card.blue  { background: linear-gradient(135deg, #bee3f8 0%, #90cdf4 100%); }
    .metric-card.purple { background: linear-gradient(135deg, #e9d8fd 0%, #d6bcfa 100%); }
    .metric-card.orange { background: linear-gradient(135deg, #fefcbf 0%, #fbd38d 100%); }

    .info-box {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2d3748;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7f8fc 0%, #eef0f7 100%);
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Yardimci Fonksiyonlar
# ============================================================
def read_wav(file_bytes):
    """WAV dosyasini oku ve numpy array olarak dondur."""
    with io.BytesIO(file_bytes) as buf:
        with wave.open(buf, 'r') as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw_data = wf.readframes(n_frames)

    if sampwidth == 2:
        y = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        y = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        y = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

    if n_channels > 1:
        y = y.reshape(-1, n_channels).mean(axis=1)

    return y, sr


def vad_analysis(y, sr, frame_ms, hop_ms, noise_ms, energy_threshold, hangover_ms, zcr_threshold):
    """VAD analizi yap ve sonuclari dondur."""
    # Normalizasyon
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    hangover_frames = int(hangover_ms / hop_ms)
    noise_samples = int(sr * noise_ms / 1000)

    noise_energy = np.mean(y[:noise_samples] ** 2)

    flags = []
    energies = []
    zcrs = []

    hamming_window = np.hamming(frame_len)

    for i in range(0, len(y) - frame_len, hop_len):
        raw_frame = y[i:i + frame_len]
        windowed_frame = raw_frame * hamming_window

        energy = np.mean(windowed_frame ** 2)
        energies.append(energy)

        zcr = np.sum(np.abs(np.diff(np.sign(windowed_frame)))) / (2 * frame_len)
        zcrs.append(zcr)

        if energy > noise_energy * energy_threshold or zcr > zcr_threshold:
            flags.append(1)
        else:
            flags.append(0)

    # Medyan Filtreleme
    flags = list(median_filter(np.array(flags), size=5).astype(int))

    # Hangover
    i = 0
    while i < len(flags):
        if flags[i] == 0:
            start = i
            while i < len(flags) and flags[i] == 0:
                i += 1
            end = i
            if (end - start) <= hangover_frames:
                for j in range(start, end):
                    flags[j] = 1
        else:
            i += 1

    # Voiced / Unvoiced
    voiced_flags = np.zeros(len(flags))
    unvoiced_flags = np.zeros(len(flags))
    for i in range(len(flags)):
        if flags[i] == 1:
            if zcrs[i] > zcr_threshold:
                unvoiced_flags[i] = 1
            else:
                voiced_flags[i] = 1

    # Cikis sinyali
    mask = np.zeros(len(y))
    for idx, flag in enumerate(flags):
        if flag == 1:
            s = idx * hop_len
            e = s + frame_len
            mask[s:e] = 1
    output = y[mask == 1]

    return {
        'y': y,
        'sr': sr,
        'output': output,
        'energies': energies,
        'zcrs': zcrs,
        'flags': flags,
        'voiced_flags': voiced_flags,
        'unvoiced_flags': unvoiced_flags,
        'frame_len': frame_len,
        'hop_len': hop_len,
        'noise_energy': noise_energy,
        'mask': mask
    }


def make_output_wav(output_signal, sr):
    """Cikis sinyalini WAV formatinda byte olarak dondur."""
    out_data = (output_signal * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(out_data.tobytes())
    return buf.getvalue()


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🎙️ Ses Etkinligi Algılama (VAD) Analiz Araci</h1>
    <p>Voice Activity Detection • Voiced/Unvoiced Siniflandirma • Sessizlik Kaldirma</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR - Parametreler
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Analiz Parametreleri")
    st.markdown("---")

    st.markdown("**🪟 Pencereleme**")
    frame_ms = st.slider("Pencere Uzunlugu (ms)", 10, 50, 20, 5,
                         help="Her bir analiz penceresinin suresi")
    hop_ms = st.slider("Adim Boyutu (ms)", 5, 25, 10, 5,
                       help="Pencereler arasi kayma miktari. Pencere/2 = %50 ortusme")

    overlap_pct = (1 - hop_ms / frame_ms) * 100
    st.info(f"📐 Ortusme: %{overlap_pct:.0f}")

    st.markdown("---")
    st.markdown("**🎯 Esik Degerleri**")
    noise_ms = st.slider("Gurultu Tahmin Suresi (ms)", 50, 500, 200, 50,
                         help="Baslangictaki sessizlik suresi (gurultu tahmini icin)")
    energy_threshold = st.slider("Enerji Carpani", 1.0, 10.0, 2.0, 0.5,
                                 help="Gurultu enerjisinin kac kati esik olacak")
    zcr_threshold = st.slider("ZCR Esigi", 0.05, 0.40, 0.15, 0.01,
                              help="Sifir gecis orani esigi")

    st.markdown("---")
    st.markdown("**⏳ Hangover**")
    hangover_ms = st.slider("Hangover Suresi (ms)", 50, 1000, 500, 50,
                            help="Kisa sessizlikleri konusmaya dahil etme suresi")


# ============================================================
# DOSYA YUKLEME
# ============================================================
st.markdown('<div class="section-title">📁 Ses Dosyasi Yukle</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "WAV formatinda ses dosyasi secin",
    type=["wav"],
    help="Konusma ve sessizlik iceren bir WAV dosyasi yukleyin"
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    try:
        y, sr = read_wav(file_bytes)
    except Exception as e:
        st.error(f"Dosya okunamadi: {e}")
        st.stop()

    # --------------------------------------------------------
    # Analiz
    # --------------------------------------------------------
    with st.spinner("🔍 Analiz yapiliyor..."):
        result = vad_analysis(y, sr, frame_ms, hop_ms, noise_ms,
                              energy_threshold, hangover_ms, zcr_threshold)

    original_dur = len(result['y']) / result['sr']
    new_dur = len(result['output']) / result['sr']
    removed_dur = original_dur - new_dur
    compression = (removed_dur / original_dur) * 100 if original_dur > 0 else 0

    # --------------------------------------------------------
    # Metrik Kartlari
    # --------------------------------------------------------
    st.markdown('<div class="section-title">📊 Sonuclar</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card blue">
            <div class="value">{original_dur:.2f}s</div>
            <div class="label">Orijinal Sure</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card green">
            <div class="value">{new_dur:.2f}s</div>
            <div class="label">Yeni Sure</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card orange">
            <div class="value">{removed_dur:.2f}s</div>
            <div class="label">Kesilen Sure</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card purple">
            <div class="value">%{compression:.1f}</div>
            <div class="label">Sikistirma Orani</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Grafikler
    # --------------------------------------------------------
    st.markdown('<div class="section-title">📈 Grafikler</div>', unsafe_allow_html=True)

    time_axis = np.linspace(0, len(result['y']) / sr, num=len(result['y']))
    frames_time = np.linspace(0, len(result['y']) / sr, num=len(result['energies']))
    hop_len = result['hop_len']
    frame_len = result['frame_len']

    # --- Grafik 1: Orijinal Sinyal ---
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=time_axis, y=result['y'],
        mode='lines', line=dict(color='#718096', width=0.8),
        name='Sinyal'
    ))
    fig1.update_layout(
        title=dict(text="1. Orijinal Ses Sinyali (Normalize Edilmis)", font=dict(size=16)),
        xaxis_title="Zaman (s)", yaxis_title="Genlik",
        height=300, template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=40)
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- Grafik 2: Enerji + ZCR ---
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(
        x=frames_time, y=result['energies'],
        mode='lines', line=dict(color='#4299e1', width=1.5),
        name='Enerji', fill='tozeroy', fillcolor='rgba(66,153,225,0.15)'
    ), secondary_y=False)
    fig2.add_trace(go.Scatter(
        x=frames_time, y=result['zcrs'],
        mode='lines', line=dict(color='#e53e3e', width=1.5),
        name='ZCR'
    ), secondary_y=True)
    # Esik cizgileri
    fig2.add_hline(y=result['noise_energy'] * energy_threshold,
                   line_dash="dash", line_color="#4299e1", opacity=0.5,
                   annotation_text="Enerji Esigi", secondary_y=False)
    fig2.add_hline(y=zcr_threshold,
                   line_dash="dash", line_color="#e53e3e", opacity=0.5,
                   annotation_text="ZCR Esigi", secondary_y=True)
    fig2.update_layout(
        title=dict(text="2. Pencere Bazli Enerji ve ZCR", font=dict(size=16)),
        xaxis_title="Zaman (s)", height=350, template="plotly_white",
        margin=dict(l=60, r=60, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig2.update_yaxes(title_text="Enerji", secondary_y=False, color="#4299e1")
    fig2.update_yaxes(title_text="ZCR", secondary_y=True, color="#e53e3e")
    st.plotly_chart(fig2, use_container_width=True)

    # --- Grafik 3: VAD + Voiced/Unvoiced ---
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=time_axis, y=result['y'],
        mode='lines', line=dict(color='#CBD5E0', width=0.8),
        name='Orijinal Sinyal'
    ))

    # Voiced bolgeleri (yesil)
    for i, v in enumerate(result['voiced_flags']):
        if v == 1:
            t_start = (i * hop_len) / sr
            t_end = (i * hop_len + frame_len) / sr
            fig3.add_vrect(x0=t_start, x1=t_end,
                          fillcolor="rgba(72,187,120,0.3)", line_width=0)

    # Unvoiced bolgeleri (sari)
    for i, u in enumerate(result['unvoiced_flags']):
        if u == 1:
            t_start = (i * hop_len) / sr
            t_end = (i * hop_len + frame_len) / sr
            fig3.add_vrect(x0=t_start, x1=t_end,
                          fillcolor="rgba(236,201,75,0.4)", line_width=0)

    # Legend icin dummy trace'ler
    fig3.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                              marker=dict(size=12, color='rgba(72,187,120,0.5)'),
                              name='Voiced (Sesli: A, O, U)'))
    fig3.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                              marker=dict(size=12, color='rgba(236,201,75,0.6)'),
                              name='Unvoiced (Sessiz: S, S, F)'))
    fig3.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                              marker=dict(size=12, color='rgba(203,213,224,0.8)'),
                              name='Sessizlik'))

    fig3.update_layout(
        title=dict(text="3. VAD: Voiced / Unvoiced / Sessizlik Bolgeleri", font=dict(size=16)),
        xaxis_title="Zaman (s)", yaxis_title="Genlik",
        height=350, template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig3, use_container_width=True)

    # --------------------------------------------------------
    # Voiced / Unvoiced Istatistik Tablosu
    # --------------------------------------------------------
    st.markdown('<div class="section-title">📋 Voiced vs Unvoiced Istatistikleri</div>', unsafe_allow_html=True)

    voiced_mask = result['voiced_flags'] == 1
    unvoiced_mask = result['unvoiced_flags'] == 1
    energies = np.array(result['energies'])
    zcrs_arr = np.array(result['zcrs'])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🟢 Sesli Harfler (Voiced)")
        if np.any(voiced_mask):
            st.metric("Ortalama Enerji", f"{np.mean(energies[voiced_mask]):.6f}")
            st.metric("Ortalama ZCR", f"{np.mean(zcrs_arr[voiced_mask]):.4f}")
            st.metric("Pencere Sayisi", f"{int(np.sum(voiced_mask))}")
        else:
            st.info("Voiced bolge bulunamadi")

    with col2:
        st.markdown("#### 🟡 Sessiz Harfler (Unvoiced)")
        if np.any(unvoiced_mask):
            st.metric("Ortalama Enerji", f"{np.mean(energies[unvoiced_mask]):.6f}")
            st.metric("Ortalama ZCR", f"{np.mean(zcrs_arr[unvoiced_mask]):.4f}")
            st.metric("Pencere Sayisi", f"{int(np.sum(unvoiced_mask))}")
        else:
            st.info("Unvoiced bolge bulunamadi")

    # Karsilastirma tablosu
    st.markdown("#### Harf Turu Karsilastirmasi")
    st.markdown("""
    | Metrik | Sesli Harfler (A, O, U) | Sessiz Harfler (S, Ş, F) |
    |---|---|---|
    | **Enerji** | ✅ Yuksek | ❌ Dusuk |
    | **ZCR** | ❌ Dusuk | ✅ Yuksek |

    > **Aciklama:** Sesli harflerde ses telleri titresir → periyodik sinyal → **yuksek enerji, dusuk ZCR**.
    > Sessiz harflerde hava surtunmesi → rastgele sinyal → **dusuk enerji, yuksek ZCR**.
    """)

    # --------------------------------------------------------
    # Indirme Butonu
    # --------------------------------------------------------
    st.markdown('<div class="section-title">💾 Cikti Dosyasi</div>', unsafe_allow_html=True)

    wav_bytes = make_output_wav(result['output'], sr)
    st.download_button(
        label="⬇️ Sessizligi Kaldirilmis Sesi Indir (output.wav)",
        data=wav_bytes,
        file_name="output.wav",
        mime="audio/wav"
    )

    # Teknik detaylar
    with st.expander("🔧 Teknik Detaylar"):
        st.markdown(f"""
        | Parametre | Deger |
        |---|---|
        | Ornekleme Hizi (Fs) | {sr} Hz |
        | Pencere Uzunlugu | {frame_ms} ms ({result['frame_len']} ornek) |
        | Adim Boyutu (Hop) | {hop_ms} ms ({hop_len} ornek) |
        | Ortusme | %{overlap_pct:.0f} |
        | Pencere Tipi | Hamming |
        | Gurultu Tahmini | Ilk {noise_ms} ms |
        | Gurultu Enerji Esigi | {result['noise_energy']:.8f} × {energy_threshold} = {result['noise_energy'] * energy_threshold:.8f} |
        | ZCR Esigi | {zcr_threshold} |
        | Hangover | {hangover_ms} ms |
        | Medyan Filtre | Boyut = 5 |
        | Toplam Pencere | {len(result['energies'])} |
        | Voiced Pencere | {int(np.sum(voiced_mask))} |
        | Unvoiced Pencere | {int(np.sum(unvoiced_mask))} |
        """)

else:
    # Dosya yuklenmediginde bilgi goster
    st.markdown("""
    <div class="info-box">
        <strong>📌 Baslamak icin</strong> sol taraftaki parametreleri ayarlayin ve yukaridan bir 
        <strong>.wav</strong> dosyasi yukleyin.<br><br>
        <strong>Odev Gereksinimleri:</strong><br>
        ✅ Normalizasyon [-1, 1]<br>
        ✅ Hamming Pencereleme<br>
        ✅ %50 Ortusme (Overlap)<br>
        ✅ Dinamik Esikleme (Adaptive Thresholding)<br>
        ✅ Medyan Filtreleme<br>
        ✅ Hangover Time<br>
        ✅ Voiced/Unvoiced Siniflandirma (Enerji + ZCR)<br>
        ✅ 3 Grafik (Sinyal, Enerji+ZCR, VAD Bölgeleri)
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 🎯 Adim 1: VAD
        Konusma bolgelerini sessizlikten ayir.
        Dinamik esikleme + Hangover ile
        dogal sonuc elde et.
        """)
    with col2:
        st.markdown("""
        #### 🔊 Adim 2: Voiced/Unvoiced
        ZCR ve Enerji ile sesli/sessiz
        harf ayrimi yap. Periyodik mi
        yoksa gurultu mu?
        """)
    with col3:
        st.markdown("""
        #### 📊 Adim 3: Gorsellestirme
        3 interaktif grafik: Orijinal sinyal,
        Enerji+ZCR, VAD bolgeleri
        renkli maskelerle.
        """)
