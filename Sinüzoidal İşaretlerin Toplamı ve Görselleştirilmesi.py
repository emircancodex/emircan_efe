import numpy as np
import matplotlib.pyplot as plt

# Frekanslar
f1 = 16
f2 = 32
f3 = 320

# Örnekleme frekansı (en yükseğe göre)
fs = 640

# 3 periyot (en yavaş sinyale göre)
t_max = 3 * (1 / 16)

# Sürekli referans
t_cont = np.linspace(0, t_max, 5000)
x_cont = (np.sin(2*np.pi*f1*t_cont) +
          np.sin(2*np.pi*f2*t_cont) +
          np.sin(2*np.pi*f3*t_cont))

# Ayrık örnekler
n = np.arange(0, int(t_max * fs))
t_sample = n / fs
x_sample = (np.sin(2*np.pi*f1*t_sample) +
            np.sin(2*np.pi*f2*t_sample) +
            np.sin(2*np.pi*f3*t_sample))

plt.figure(figsize=(10,5))
plt.plot(t_cont, x_cont, label="Sürekli Toplam")
plt.stem(t_sample, x_sample, basefmt=" ", linefmt="r-", markerfmt="ro", label="Örnekler (640 Hz)")
plt.title("Toplam Sinyal (16 + 32 + 320 Hz)")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.legend()
plt.grid(True)
plt.show()
