import numpy as np
import matplotlib.pyplot as plt

fs = 2000            # Frecuencia de muestreo en Hz
T = 0.1              # Duración total en segundos
f_m = 100            # Frecuencia de la señal mensaje (Hz)

t = np.linspace(0, T, int(fs*T), endpoint=False) 
m_t = np.sin(2 * np.pi * f_m * t)       

# Señal en dominio del tiempo
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t, m_t)
plt.title("m(t)")
plt.xlabel("t")
plt.ylabel("A")
plt.grid(True)


# Obtengo la transformada de Fourier para tener el mensaje en dominio de frecuencia.
M_f = np.fft.fft(m_t)
M_f = np.abs(M_f) / len(m_t)            # Magnitud normalizada
freqs = np.fft.fftfreq(len(m_t), 1/fs)  # Vector de frecuencias

# Nos quedamos con la parte positiva
half = len(m_t) // 2
plt.subplot(2, 1, 2)
plt.plot(freqs[:half], M_f[:half])
plt.title("M(f)")
plt.xlabel("f")
plt.ylabel("M")
plt.grid(True)

plt.tight_layout()
plt.show()