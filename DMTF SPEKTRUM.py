import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# Örnekleme parametreleri
fs = 44100
duration = 0.5

# DTMF Frekans Tablosu
dtmf_freq = {
    '1': (697, 1209), '2': (697, 1336), '3': (697, 1477), 'A': (697, 1633),
    '4': (770, 1209), '5': (770, 1336), '6': (770, 1477), 'B': (770, 1633),
    '7': (852, 1209), '8': (852, 1336), '9': (852, 1477), 'C': (852, 1633),
    '*': (941, 1209), '0': (941, 1336), '#': (941, 1477), 'D': (941, 1633)
}

def play_tone(key):
    f1, f2 = dtmf_freq[key]

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # İki frekansın toplamı
    signal = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)

    # Normalizasyon (distortion olmasın diye)
    signal = signal / np.max(np.abs(signal))

    # Ses çal
    sd.play(signal, fs)

    # --- 1. Zaman Domeni Grafiği ---
    plt.figure("DTMF Sinyali")
    plt.clf()
    plt.plot(t[:1000], signal[:1000])
    plt.title(f"{key} Tuşu - Zaman Domeni ({f1} Hz + {f2} Hz)")
    plt.xlabel("Zaman (s)")
    plt.ylabel("Genlik")
    plt.grid(True)
    plt.pause(0.001)

    # --- 2. Frekans Domeni Grafiği (FFT) ---
    # FFT işlemleri fonksiyonun içine alındı, böylece her tuş basımında yenilenir
    fft_signal = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), 1/fs)

    # Sadece pozitif frekansları al
    positive_freqs = fft_freq[:len(signal)//2]
    magnitude = np.abs(fft_signal[:len(signal)//2])

    plt.figure("DTMF Spektrumu")
    plt.clf()
    plt.plot(positive_freqs, magnitude)
    plt.title(f"{key} Tuşu - Frekans Spektrumu (FFT)")
    plt.xlabel("Frekans (Hz)")
    plt.ylabel("Genlik")
    plt.xlim(0, 2000)  # DTMF frekans aralığı
    
    # Zirve yapan frekansları grafikte daha net görmek için x ekseninde ufak bir ayarlama
    plt.xticks(list(range(0, 2001, 200)), rotation=45) 
    plt.grid(True)
    plt.pause(0.001)

# İnteraktif mod açılışı
plt.ion()

# Tkinter Arayüz
root = tk.Tk()
root.title("DTMF Telefon Tuş Takımı")

buttons = [
    ['1','2','3','A'],
    ['4','5','6','B'],
    ['7','8','9','C'],
    ['*','0','#','D']
]

for r in range(4):
    for c in range(4):
        key = buttons[r][c]
        btn = tk.Button(root, text=key, width=8, height=3,
                        command=lambda k=key: play_tone(k))
        btn.grid(row=r, column=c, padx=5, pady=5)

root.mainloop()
