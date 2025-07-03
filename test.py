import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

fs = 50000           # Frecuencia de muestreo (Hz)
T = 0.5              # Duración total (s)
f_c = 300            # Frecuencia de la portadora (Hz)
f_m = 100            # Ancho de banda del mensaje

t = np.linspace(0, T, int(fs*T), endpoint=False)

# Tren de pulsos
pulse_width = 0.01
pulse_period = 0.1
m_t = ((t % pulse_period) < pulse_width).astype(float)

# Portadora
c_t = np.cos(2 * np.pi * f_c * t)

# Señal modulada
s_t = m_t * c_t

# Filtro pasabanda Butterworth
lowcut = f_c - f_m
highcut = f_c + f_m
nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq
b, a = butter(N=5, Wn=[low, high], btype='band')
s_t_filt = filtfilt(b, a, s_t)

# Espectro
S_f = np.fft.fft(s_t)
S_f_filt = np.fft.fft(s_t_filt)
freqs = np.fft.fftfreq(len(t), 1/fs)

# Solo el lado positivo
idx = freqs > 0
freqs_pos = freqs[idx]
S_f = np.abs(S_f) / len(S_f)
S_f_filt = np.abs(S_f_filt) / len(S_f_filt)

# --- PLOTEO ---

plt.figure(figsize=(12, 6))
plt.plot(freqs_pos, S_f[idx], label='Original', alpha=0.5)
plt.plot(freqs_pos, S_f_filt[idx], label='Filtrada (Pasabanda)', color='red')
plt.xlim(0, 1000)
plt.title("Espectro de s(t) antes y después del filtro pasabanda")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# También mostramos la señal en el tiempo
plt.figure(figsize=(10, 4))
plt.plot(t, s_t, label="s(t) original", alpha=0.5)
plt.plot(t, s_t_filt, label="s(t) filtrada", color='red')
plt.xlim(0, 0.1)
plt.title("Señal en el tiempo antes y después del filtro")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
