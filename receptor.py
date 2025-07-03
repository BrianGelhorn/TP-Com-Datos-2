import matplotlib.pyplot as plt
import numpy as np
import transmisor
from scipy.signal import butter, filtfilt

fs = 10000           # Frecuencia de muestreo (Hz)
T = 1             # Duración total (s)
f_c = 300           # Frecuencia de la portadora (Hz)
f_m = 10  # 10 Hz
t = np.linspace(0, T, int(fs*T), endpoint=False)

#Simulo ruido en el sistema
noise = np.random.normal(0, .5, transmisor.señalModuladaCF().shape)
s_t = transmisor.señalModuladaCF() + noise

S_f = np.fft.fft(s_t)
S_f = np.abs(S_f) / len(s_t)
freqs = np.fft.fftfreq(len(s_t), 1/fs)

# Parámetros del filtro
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

plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
plt.plot(t, s_t)
plt.title("s(t)+ruido")
plt.xlabel("t")
plt.ylabel("A")
plt.grid(True)
plt.tight_layout()


plt.subplot(3,1,2)
plt.plot(freqs, S_f)
plt.title("Espectro de la señal modulada s(t)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid(True)
plt.tight_layout()

# Gráfica del espectro de la señal filtrada
plt.subplot(3,1,3)
plt.plot(freqs, s_t_filtered)
plt.title("Espectro de la señal AM filtrada (pasabanda)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.show()

# Genero la señal de portadora a la frecuencia definida por f_c
c_t = 2*np.cos(2 * np.pi * f_c * t)

plt.subplot(2,1,1)
s_t_filtered = s_t_filtered*c_t
S_f_filtered = np.fft.fft(s_t_filtered)
S_f_filtered = np.abs(S_f_filtered) / len(s_t_filtered)
plt.plot(t, s_t_filtered)
plt.title("Espectro de la señal AM filtrada (pasabanda)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid(True)
plt.tight_layout()
# Gráfica del espectro de la señal filtrada
plt.subplot(2,1,2)
plt.plot(freqs, S_f_filtered)
plt.title("Espectro de la señal AM filtrada (pasabanda)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.show()

# Filtro pasabajo para extraer la señal de mensaje
def filtro_pasabajo(data, cutoff, fs, orden=5):
    nyq = fs / 2
    b, a = butter(orden, cutoff / nyq, btype='low')
    return filtfilt(b, a, data)

# Parámetro del filtro pasabajo
cutoff = f_m + 20  # Un poco más del ancho del mensaje (por ejemplo, 15 Hz)

# Aplicar filtro pasabajo
mensaje_recuperado = filtro_pasabajo(s_t_filtered, cutoff, fs)

# Mostrar señal recuperada
plt.plot(t, mensaje_recuperado)
plt.title("Señal demodulada (mensaje recuperado)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.tight_layout()
plt.show()