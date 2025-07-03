import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
fs = 10000           # Frecuencia de muestreo (Hz)
T = .5             # Duración total (s)
f_c = 300           # Frecuencia de la portadora (Hz)

t = np.linspace(0, T, int(fs*T), endpoint=False)

# Tren de pulsos: duración del pulso = 10 ms
pulse_width = 0.01  # en segundos
pulse_period = 0.1  # repetición cada 100 ms
m_t = ((t % pulse_period) < pulse_width).astype(float)
M_f = np.fft.fft(m_t)
M_f = np.abs(M_f) / len(M_f)
freqs = np.fft.fftfreq(len(M_f), 1/fs)
# Genero la señal de portadora a la frecuencia definida por f_c
c_t = np.cos(2 * np.pi * f_c * t)

# Multiplico el mensaje por la señal de portadora ya que en doble banda lateral, la amplitud de la portadora es directamente proporcional al mensaje
s_t = m_t * c_t

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(t, m_t)
plt.title("m(t)")
plt.xlabel("t")
plt.ylabel("A")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(freqs, M_f)
plt.title("M(f)")
plt.xlabel("f")
plt.ylabel("M")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, c_t)
plt.title("c(t)")
plt.xlabel("t")
plt.ylabel("A")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, s_t)
plt.title("AM")
plt.xlabel("t")
plt.ylabel("A")
plt.grid(True)

plt.tight_layout()
plt.show()

S_f = np.fft.fft(s_t)
S_f = np.abs(S_f) / len(s_t)
freqs = np.fft.fftfreq(len(s_t), 1/fs)

plt.figure(figsize=(10, 4))
plt.plot(freqs, S_f)
plt.title("Espectro de la señal modulada s(t)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid(True)
plt.tight_layout()
plt.show()

# Parámetros del filtro
f_m = 1 / pulse_period  # 10 Hz
lowcut = f_c - f_m  # 290 Hz
highcut = f_c + f_m # 310 Hz

# Diseño del filtro pasabanda Butterworth
order = 1 # orden del filtro (puede ajustarse)
b, a = butter(order, [lowcut/(fs/2), highcut/(fs/2)], btype='band')

# Filtrado de la señal modulada
s_t_filtered = filtfilt(b, a, s_t)

# Espectro de la señal filtrada
S_f_filtered = np.fft.fft(s_t_filtered)
S_f_filtered = np.abs(S_f_filtered) / len(s_t_filtered)

# Gráfica del espectro de la señal filtrada
plt.figure(figsize=(10, 4))
plt.plot(freqs, S_f_filtered)
plt.title("Espectro de la señal AM filtrada (pasabanda)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid(True)
plt.tight_layout()
plt.show()