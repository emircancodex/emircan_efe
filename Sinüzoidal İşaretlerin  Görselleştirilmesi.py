import numpy as np
import matplotlib.pyplot as plt

signals = [
    (16, 32),
    (32, 64),
    (320, 640)
]

plt.figure(figsize=(12,10))

for i, (f, fs) in enumerate(signals):

    T = 1 / f
    t_max = 3 * T

    # Sürekli sinüs (yüksek çözünürlük)
    t_cont = np.linspace(0, t_max, 5000)
    x_cont = np.sin(2*np.pi*f*t_cont)

    # Ayrık örnekler (Nyquist)
    n = np.arange(0, int(t_max * fs))
    t_sample = n / fs
    x_sample = np.sin(2*np.pi*f*t_sample)

    plt.subplot(3,1,i+1)
    plt.plot(t_cont, x_cont, label="Sürekli")
    plt.stem(t_sample, x_sample, basefmt=" ", linefmt="r-", markerfmt="ro", label="Örnekler")
    plt.title(f"{f} Hz Sinyal - {fs} Hz Örnekleme (3 Periyot)")
    plt.xlabel("Zaman (s)")
    plt.ylabel("Genlik")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
