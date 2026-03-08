import os
import numpy as np
import wave
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Scriptin bulunduğu klasörü otomatik tespit et
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def analyze_and_remove_silence(input_wav, output_wav,
                               frame_ms=20,
                               hop_ms=10,
                               noise_ms=200,
                               energy_threshold=2,
                               hangover_ms=500,
                               zcr_threshold=0.15):
    # 1. Sinyali Oku
    try:
        with wave.open(input_wav, 'r') as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw_data = wf.readframes(n_frames)
        # Byte verisini numpy array'e çevir
        if sampwidth == 2:
            y = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            y = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            y = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        # Stereo ise mono'ya çevir
        if n_channels > 1:
            y = y.reshape(-1, n_channels).mean(axis=1)
    except FileNotFoundError:
        print(f"HATA: '{input_wav}' dosyasi bulunamadi. Lutfen dosya yolunu kontrol edin.")
        return

    original_duration = len(y) / sr

    # --- SINYAL ISLEME KURALLARI ---

    # KURAL 1: Normalizasyon (Sinyali [-1, 1] araligina cekme)
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    # KURAL 3: Ornekleme Hizi (Fs) Donusumleri
    frame_len = int(sr * frame_ms / 1000)     # 20ms = pencere uzunlugu
    hop_len = int(sr * hop_ms / 1000)          # 10ms = %50 ortusme (overlap)
    hangover_frames = int(hangover_ms / hop_ms)
    noise_samples = int(sr * noise_ms / 1000)

    # Baslangictaki gurultu enerjisini hesapla (Normalize sinyal uzerinden)
    noise_energy = np.mean(y[:noise_samples] ** 2)

    flags = []
    energies = []
    zcrs = []

    # KURAL 2: Pencereleme (Hamming Window) - Spektral sizintiyi onler
    hamming_window = np.hamming(frame_len)

    # Frame bazli analiz
    for i in range(0, len(y) - frame_len, hop_len):
        raw_frame = y[i:i + frame_len]

        # Hamming penceresini uygula
        windowed_frame = raw_frame * hamming_window

        # Enerji ve ZCR hesabi
        energy = np.mean(windowed_frame ** 2)
        energies.append(energy)

        zcr = np.sum(np.abs(np.diff(np.sign(windowed_frame)))) / (2 * frame_len)
        zcrs.append(zcr)

        # VAD: Enerji esigi asilirsa VEYA ZCR yuksekse (sessiz harf)
        if energy > noise_energy * energy_threshold or zcr > zcr_threshold:
            flags.append(1)
        else:
            flags.append(0)

    # Medyan Filtreleme: Anlik hatali kararlari temizle (tek pencerelik yanlis kararlar)
    from scipy.ndimage import median_filter
    flags = list(median_filter(np.array(flags), size=5).astype(int))

    # Kisa sessizlikleri konusmaya dahil etme (Hangover)
    i = 0
    while i < len(flags):
        if flags[i] == 0:
            start = i
            while i < len(flags) and flags[i] == 0:
                i += 1
            end = i
            length = end - start
            if length <= hangover_frames:
                for j in range(start, end):
                    flags[j] = 1
        else:
            i += 1

    # Gorsellestirme icin Voiced/Unvoiced ayrimi
    voiced_flags = np.zeros(len(flags))
    unvoiced_flags = np.zeros(len(flags))

    for i in range(len(flags)):
        if flags[i] == 1:
            if zcrs[i] > zcr_threshold:
                unvoiced_flags[i] = 1  # Sessiz harfler (Yuksek ZCR)
            else:
                voiced_flags[i] = 1    # Sesli harfler (Yuksek Enerji)

    # Kesme islemi icin Maske olustur
    mask = np.zeros(len(y))
    for idx, flag in enumerate(flags):
        if flag == 1:
            start = idx * hop_len
            end = start + frame_len
            mask[start:end] = 1

    # Yeni sesi olustur ve kaydet
    output = y[mask == 1]
    new_duration = len(output) / sr
    out_data = (output * 32767).astype(np.int16)
    with wave.open(output_wav, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(out_data.tobytes())

    # --- KONSOL CIKTILARI ---
    print("=" * 40)
    print(f"Orijinal sure : {original_duration:.2f} saniye")
    print(f"Yeni sure     : {new_duration:.2f} saniye")
    print(f"Kesilen sure  : {(original_duration - new_duration):.2f} saniye")
    print(f"Sikistirma    : %{100 * (original_duration - new_duration) / original_duration:.2f}")
    print("=" * 40)

    # --- GORSELLESTIRME ---
    time_axis = np.linspace(0, len(y) / sr, num=len(y))
    frames_time_axis = np.linspace(0, len(y) / sr, num=len(energies))

    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1. Grafik: Orijinal sinyal
    axs[0].plot(time_axis, y, color='gray', alpha=0.7)
    axs[0].set_title("1. Orijinal Ses Sinyali (Zaman Domeni - Normalize Edilmis)")
    axs[0].set_ylabel("Genlik")
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # 2. Grafik: Enerji ve ZCR
    color_en = 'tab:blue'
    axs[1].set_ylabel("Enerji (Hamming)", color=color_en)
    axs[1].plot(frames_time_axis, energies, color=color_en, label="Enerji", alpha=0.8)
    axs[1].tick_params(axis='y', labelcolor=color_en)

    ax2 = axs[1].twinx()
    color_zcr = 'tab:red'
    ax2.set_ylabel("ZCR", color=color_zcr)
    ax2.plot(frames_time_axis, zcrs, color=color_zcr, label="ZCR", alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color_zcr)

    axs[1].set_title("2. Pencere Bazli Enerji ve ZCR Grafikleri")
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # 3. Grafik: VAD + Voiced/Unvoiced bolgeleri
    axs[2].plot(time_axis, y, color='lightgray', alpha=0.8, label="Orijinal Sinyal")

    for i, is_voiced in enumerate(voiced_flags):
        if is_voiced == 1:
            start_t = (i * hop_len) / sr
            end_t = (i * hop_len + frame_len) / sr
            axs[2].axvspan(start_t, end_t, color='green', alpha=0.3, lw=0)

    for i, is_unvoiced in enumerate(unvoiced_flags):
        if is_unvoiced == 1:
            start_t = (i * hop_len) / sr
            end_t = (i * hop_len + frame_len) / sr
            axs[2].axvspan(start_t, end_t, color='yellow', alpha=0.4, lw=0)

    voiced_patch = mpatches.Patch(color='green', alpha=0.3, label='Voiced (A, O, U)')
    unvoiced_patch = mpatches.Patch(color='yellow', alpha=0.4, label='Unvoiced (S, S, F)')
    silence_patch = mpatches.Patch(color='lightgray', alpha=0.8, label='Sessizlik')

    axs[2].legend(handles=[voiced_patch, unvoiced_patch, silence_patch], loc="upper right")
    axs[2].set_title("3. VAD ve Voiced/Unvoiced Bolgeleri")
    axs[2].set_xlabel("Zaman (saniye)")
    axs[2].set_ylabel("Genlik")

    plot_path = os.path.join(SCRIPT_DIR, "vad_analysis_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Grafik '{plot_path}' olarak basariyla kaydedildi.\n")


if __name__ == "__main__":
    input_path = os.path.join(SCRIPT_DIR, "input.wav")
    output_path = os.path.join(SCRIPT_DIR, "output.wav")
    analyze_and_remove_silence(input_path, output_path)
