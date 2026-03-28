import streamlit as st
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
mplstyle.use("dark_background")
import librosa
import io

from audio_utils import load_audio, compute_autocorrelation_pitch, compute_fft_pitch, analyze_audio
from classifier import classify_gender
from dataset_manager import process_dataset, evaluate_metrics, scan_dataset

st.set_page_config(
    page_title="Ses Analizi & Cinsiyet Sınıflandırma",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0d0f1a 0%, #111827 50%, #0a0e1a 100%); color: #e2e8f0; }
.main-header { text-align: center; padding: 2.5rem 0 1.5rem 0; }
.main-header h1 { font-size: 2.8rem; font-weight: 900;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
.main-header p { color: #94a3b8; font-size: 1.05rem; }
.metric-card { background: linear-gradient(135deg, #1e2d40 0%, #1a2535 100%);
    border: 1px solid #2d4a6a; border-radius: 16px; padding: 1.5rem;
    text-align: center; margin-bottom: 10px; }
.metric-value { font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px;
    text-transform: uppercase; letter-spacing: 0.08em; }
.pred-badge { display: inline-block; padding: 0.6rem 2.5rem; border-radius: 50px;
    font-size: 1.8rem; font-weight: 800; margin: 1rem 0; }
.pred-erkek  { background: linear-gradient(135deg, #1d4ed8, #2563eb); color: white; }
.pred-kadin  { background: linear-gradient(135deg, #be185d, #ec4899); color: white; }
.pred-cocuk  { background: linear-gradient(135deg, #15803d, #22c55e); color: white; }
.pred-bilinmeyen { background: linear-gradient(135deg, #374151, #4b5563); color: white; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: rgba(15,23,42,.8);
    padding: 8px; border-radius: 12px; border: 1px solid #1e3a5f; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 24px;
    color: #94a3b8; font-weight: 600; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg,#1d4ed8,#7c3aed) !important;
    color: white !important; }
.stButton > button { border:none; background: linear-gradient(135deg,#1d4ed8,#7c3aed);
    color:white; font-weight:700; border-radius:10px; padding:.65rem 2rem;
    font-size:1rem; transition:all .2s ease; width:100%; }
.stButton > button:hover { transform:translateY(-2px);
    box-shadow:0 8px 25px rgba(124,58,237,.4); }
div[data-testid="stMetric"] { background:rgba(30,41,59,.7);
    border:1px solid #1e3a5f; border-radius:12px; padding:1rem 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🎙️ Ses Analizi ve Cinsiyet Sınıflandırma</h1>
    <p>Zaman Düzlemi Analizi &amp; Otokorelasyon Yöntemi ile Erkek / Kadın / Çocuk Sınıflandırması</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎵  Tekil Ses Analizi", "📊  Veri Seti Değerlendirme"])

# ─── TAB 1: Tekil Ses ─────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Bir .wav dosyası yükle, analiz et!")
    uploaded_file = st.file_uploader("Ses Dosyası Seçin (.wav)", type="wav", label_visibility="collapsed")

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

        col_wave, col_pred = st.columns([2, 1])
        with col_wave:
            st.markdown("**📈 Dalga Formu**")
            fig_wave, ax_wave = plt.subplots(figsize=(8, 3), facecolor='#0f172a')
            ax_wave.set_facecolor('#0f172a')
            t = np.linspace(0, len(y)/sr, len(y))
            ax_wave.plot(t, y, color='#60a5fa', linewidth=0.8, alpha=0.9)
            ax_wave.fill_between(t, y, alpha=0.15, color='#60a5fa')
            ax_wave.set_xlabel("Zaman (s)", color='#94a3b8')
            ax_wave.set_ylabel("Genlik", color='#94a3b8')
            ax_wave.tick_params(colors='#94a3b8')
            for s in ax_wave.spines.values(): s.set_edgecolor('#1e3a5f')
            st.pyplot(fig_wave); plt.close(fig_wave)

        mean_f0, mean_zcr, mean_ste, f0_list = analyze_audio(y, sr)
        pred = classify_gender(mean_f0)
        css_class = {"Erkek":"pred-erkek","Kadın":"pred-kadin","Çocuk":"pred-cocuk"}.get(pred,"pred-bilinmeyen")
        emoji = {"Erkek":"👨","Kadın":"👩","Çocuk":"👧"}.get(pred,"❓")

        with col_pred:
            st.markdown("**🎯 Tahmin Sonucu**")
            st.markdown(f'<div class="pred-badge {css_class}">{emoji} {pred}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-value">{mean_f0:.0f} Hz</div><div class="metric-label">Ortalama F0 (Pitch)</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-value">{mean_zcr:.4f}</div><div class="metric-label">Ortalama ZCR</div></div>', unsafe_allow_html=True)

        # Otokorelasyon vs FFT
        st.markdown("---")
        st.markdown("### 🔬 Otokorelasyon vs. FFT Karşılaştırması")
        mid_idx = len(y) // 2
        frame_len = int(sr * 0.03)
        if mid_idx + frame_len < len(y):
            frame = y[mid_idx:mid_idx + frame_len]
            f0_ac, autocorr = compute_autocorrelation_pitch(frame, sr)
            f0_fft, spectrum, freqs = compute_fft_pitch(frame, sr)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**📉 Otokorelasyon R(τ)** — F0 = `{f0_ac:.1f} Hz`")
                fig_ac, ax_ac = plt.subplots(figsize=(6, 3.5), facecolor='#0f172a')
                ax_ac.set_facecolor('#0f172a')
                if len(autocorr) > 0:
                    lag_ms = np.arange(len(autocorr)) / sr * 1000
                    ax_ac.plot(lag_ms[:300], autocorr[:300], color='#a78bfa', linewidth=1.5)
                    if f0_ac > 0:
                        ax_ac.axvline(1000/f0_ac, color='#f472b6', linestyle='--', label=f'F0={f0_ac:.0f}Hz')
                    ax_ac.set_xlabel("Gecikme τ (ms)", color='#94a3b8')
                    ax_ac.set_ylabel("R(τ)", color='#94a3b8')
                    ax_ac.tick_params(colors='#94a3b8')
                    for s in ax_ac.spines.values(): s.set_edgecolor('#1e3a5f')
                    ax_ac.legend(facecolor='#0f172a', edgecolor='#1e3a5f', labelcolor='white')
                st.pyplot(fig_ac); plt.close(fig_ac)
            with c2:
                st.markdown(f"**📊 FFT Spektrumu** — F0 = `{f0_fft:.1f} Hz`")
                fig_fft, ax_fft = plt.subplots(figsize=(6, 3.5), facecolor='#0f172a')
                ax_fft.set_facecolor('#0f172a')
                if len(spectrum) > 0:
                    zoom = np.where(freqs <= 800)[0]
                    ax_fft.plot(freqs[zoom], spectrum[zoom], color='#34d399', linewidth=1.5)
                    if f0_fft > 0:
                        ax_fft.axvline(f0_fft, color='#f472b6', linestyle='--', label=f'F0={f0_fft:.0f}Hz')
                    ax_fft.set_xlabel("Frekans (Hz)", color='#94a3b8')
                    ax_fft.set_ylabel("|X(f)|", color='#94a3b8')
                    ax_fft.tick_params(colors='#94a3b8')
                    for s in ax_fft.spines.values(): s.set_edgecolor('#1e3a5f')
                    ax_fft.legend(facecolor='#0f172a', edgecolor='#1e3a5f', labelcolor='white')
                st.pyplot(fig_fft); plt.close(fig_fft)

        if len(f0_list) > 0:
            st.markdown("---")
            st.markdown("### 🎶 F0 Zaman Grafiği")
            fig_f0, ax_f0 = plt.subplots(figsize=(10, 3), facecolor='#0f172a')
            ax_f0.set_facecolor('#0f172a')
            ax_f0.plot(f0_list, color='#f59e0b', linewidth=1.5, marker='o', markersize=3)
            ax_f0.axhline(mean_f0, color='#f472b6', linestyle='--', linewidth=1.2, label=f'Ort. F0={mean_f0:.0f} Hz')
            ax_f0.set_xlabel("Çerçeve No", color='#94a3b8')
            ax_f0.set_ylabel("F0 (Hz)", color='#94a3b8')
            ax_f0.tick_params(colors='#94a3b8')
            for s in ax_f0.spines.values(): s.set_edgecolor('#1e3a5f')
            ax_f0.legend(facecolor='#0f172a', edgecolor='#1e3a5f', labelcolor='white')
            st.pyplot(fig_f0); plt.close(fig_f0)
    else:
        st.info("⬆️ Analiz için bir `.wav` dosyası yükleyin.")

# ─── TAB 2: Veri Seti ──────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📂 Veri Seti Performans Testi")

    dataset_path = st.text_input(
        "Veri Seti Klasör Yolu:",
        r"C:\Users\onlin\Downloads\Midterm_Dataset_2026-20260328T161509Z-3-001\Midterm_Dataset_2026"
    )

    # Dataset önizleme
    if dataset_path and os.path.isdir(dataset_path):
        records = scan_dataset(dataset_path)
        if records:
            import pandas as pd
            preview = pd.DataFrame(records)[['group','basename','gender_true']]
            n_erkek = (preview['gender_true']=='Erkek').sum()
            n_kadin = (preview['gender_true']=='Kadın').sum()
            n_cocuk = (preview['gender_true']=='Çocuk').sum()
            p1,p2,p3,p4 = st.columns(4)
            p1.metric("📁 Toplam WAV", len(records))
            p2.metric("👨 Erkek", n_erkek)
            p3.metric("👩 Kadın", n_kadin)
            p4.metric("👧 Çocuk", n_cocuk)

    if st.button("🚀 Tüm Veri Setini Analiz Et"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(done, total):
            if total > 0:
                pct = done / total
                progress_bar.progress(pct)
                status_text.text(f"İşleniyor: {done}/{total} dosya (%{pct*100:.0f})")

        with st.spinner("Lütfen bekleyin — tüm WAV dosyaları analiz ediliyor..."):
            results_df, err = process_dataset(dataset_path, progress_callback=update_progress)

        progress_bar.progress(1.0)
        status_text.text("✅ İşlem tamamlandı!")

        if err:
            st.error(f"❌ Hata: {err}")
        elif results_df is None or len(results_df) == 0:
            st.warning("⚠️ Hiç sonuç üretilemedi.")
        else:
            acc, conf_matrix, stats = evaluate_metrics(results_df)

            st.markdown("---")
            k1, k2, k3, k4 = st.columns(4)
            correct = results_df['Doğru mu'].sum()
            k1.metric("🎯 Genel Başarı", f"%{acc:.1f}")
            k2.metric("📁 İşlenen Dosya", len(results_df))
            k3.metric("✅ Doğru Tahmin", int(correct))
            k4.metric("❌ Yanlış Tahmin", int(len(results_df) - correct))

            # İstatistiksel Tablo
            st.markdown("---")
            st.markdown("### 📈 İstatistiksel F0 Tablosu (Sınıf Bazlı)")
            st.dataframe(
                stats.style.format({
                    "Ortalama F0 (Hz)": "{:.1f}",
                    "Standart Sapma": "{:.1f}",
                    "Başarı (%)": "{:.1f}%"
                }).background_gradient(subset=["Başarı (%)"], cmap="RdYlGn"),
                use_container_width=True
            )

            # Confusion Matrix (görsel)
            st.markdown("### 🔄 Karışıklık Matrisi")
            if conf_matrix is not None:
                import numpy as np
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4), facecolor='#0f172a')
                ax_cm.set_facecolor('#0f172a')
                labels = conf_matrix.index.tolist()
                matrix_vals = conf_matrix.values
                im = ax_cm.imshow(matrix_vals, cmap='Blues')
                ax_cm.set_xticks(range(len(conf_matrix.columns)))
                ax_cm.set_yticks(range(len(labels)))
                ax_cm.set_xticklabels(conf_matrix.columns, color='#e2e8f0')
                ax_cm.set_yticklabels(labels, color='#e2e8f0')
                ax_cm.set_xlabel("Tahmin", color='#94a3b8', fontsize=12)
                ax_cm.set_ylabel("Gerçek", color='#94a3b8', fontsize=12)
                ax_cm.tick_params(colors='#94a3b8')
                max_val = np.max(matrix_vals) if matrix_vals.size > 0 else 1
                for i in range(len(labels)):
                    for j in range(len(conf_matrix.columns)):
                        val = matrix_vals[i, j]
                        ax_cm.text(j, i, str(val), ha='center', va='center',
                                   color='white' if val > max_val/2 else '#aaa',
                                   fontsize=14, fontweight='bold')
                plt.colorbar(im, ax=ax_cm)
                st.pyplot(fig_cm); plt.close(fig_cm)

            # F0 Dağılım Histogramı
            st.markdown("### 🎻 Sınıf Bazlı F0 Dağılımı")
            colors_map = {"Erkek": "#60a5fa", "Kadın": "#f472b6", "Çocuk": "#34d399"}
            fig_dist, ax_dist = plt.subplots(figsize=(10, 4), facecolor='#0f172a')
            ax_dist.set_facecolor('#0f172a')
            for cls in results_df['Gerçek Sınıf'].unique():
                subset = results_df[results_df['Gerçek Sınıf'] == cls]['F0 (Hz)']
                subset = subset[subset > 0]
                if len(subset) > 1:
                    ax_dist.hist(subset, bins=25, alpha=0.65, label=cls,
                                 color=colors_map.get(cls,'#94a3b8'), edgecolor='none')
            ax_dist.set_xlabel("F0 (Hz)", color='#94a3b8')
            ax_dist.set_ylabel("Dosya Sayısı", color='#94a3b8')
            ax_dist.tick_params(colors='#94a3b8')
            for s in ax_dist.spines.values(): s.set_edgecolor('#1e3a5f')
            ax_dist.legend(facecolor='#0f172a', edgecolor='#1e3a5f', labelcolor='white')
            st.pyplot(fig_dist); plt.close(fig_dist)

            # Grup Bazlı Başarı Grafiği
            st.markdown("### 🏆 Grup Bazlı Doğruluk")
            grp_acc = results_df.groupby('Grup')['Doğru mu'].mean() * 100
            grp_acc = grp_acc.sort_values()
            fig_grp, ax_grp = plt.subplots(figsize=(12, 4), facecolor='#0f172a')
            ax_grp.set_facecolor('#0f172a')
            bars = ax_grp.bar(grp_acc.index, grp_acc.values,
                              color=['#22c55e' if v >= 70 else '#f59e0b' if v >= 40 else '#ef4444'
                                     for v in grp_acc.values])
            ax_grp.axhline(acc, color='#a78bfa', linestyle='--', linewidth=1.5,
                           label=f'Ortalama: %{acc:.1f}')
            ax_grp.set_xlabel("Grup", color='#94a3b8')
            ax_grp.set_ylabel("Başarı (%)", color='#94a3b8')
            ax_grp.tick_params(colors='#94a3b8', axis='both')
            plt.xticks(rotation=45, ha='right', color='#94a3b8', fontsize=8)
            for s in ax_grp.spines.values(): s.set_edgecolor('#1e3a5f')
            ax_grp.legend(facecolor='#0f172a', edgecolor='#1e3a5f', labelcolor='white')
            st.pyplot(fig_grp); plt.close(fig_grp)

            # Ham Sonuçlar Tablosu
            with st.expander("📋 Tüm Dosya Sonuçlarını Göster"):
                st.dataframe(
                    results_df.style.apply(
                        lambda row: ['background-color: #14532d' if row['Doğru mu']
                                     else 'background-color: #7f1d1d'] * len(row),
                        axis=1
                    ),
                    use_container_width=True
                )
