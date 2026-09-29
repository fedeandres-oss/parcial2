import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, firwin, filtfilt, hilbert

# ==========================================
# 0. CONTROL DE TIEMPO Y CONFIGURACIÓN
# ==========================================
t_graficar_ms = 1.0  # Tiempo en ms a visualizar en el eje X

fs_kHz = 200.0   # Frecuencia de muestreo en kHz (200 kHz)
fs = fs_kHz * 1000  # 200,000 Hz
archivo_audio = '03_1_modulated_signal_20.0_200.0_40.0.csv'

# Cargar señal del CSV
y_recibida = np.loadtxt(archivo_audio, delimiter=',', skiprows=1, usecols=1)

dt = 1 / fs
N = len(y_recibida)
t_ms = (np.arange(0, N) * dt) * 1000
T_total_ms = t_ms[-1]

if t_graficar_ms > T_total_ms:
    t_graficar_ms = T_total_ms

nyquist = fs / 2

# ==========================================
# 1. (a) y (b) FFT, DETECCIÓN Y GRÁFICA DEL ESPECTRO
# ==========================================
fft_valores = np.fft.fft(y_recibida)
frecuencias_kHz = np.fft.fftfreq(N, dt) / 1000

# Considerar rango espectral positivo hasta Nyquist
pos_mask = (frecuencias_kHz >= 0) & (frecuencias_kHz <= nyquist / 1000)
frecuencias_pos = frecuencias_kHz[pos_mask]
magnitud = np.abs(fft_valores[pos_mask]) / N

# Detección del pico principal (Portadora)
idx_max = np.argmax(magnitud)
fc_transmitida_kHz = frecuencias_pos[idx_max]
fc_transmitida_Hz = fc_transmitida_kHz * 1000  # Ej: 20,000 Hz
pico_max = magnitud[idx_max]

# VERIFICACIÓN ESPECTRAL AM
mascara_sin_portadora = np.ones(len(magnitud), dtype=bool)
radio_exclusion = int(len(magnitud) * 0.005)
idx_min_excl = max(0, idx_max - radio_exclusion)
idx_max_excl = min(len(magnitud), idx_max + radio_exclusion)
mascara_sin_portadora[idx_min_excl:idx_max_excl] = False

nivel_base = np.mean(magnitud[mascara_sin_portadora])
relacion_pico_base = pico_max / (nivel_base + 1e-12)

# Es AM si presenta un pico de portadora prominente
es_AM = relacion_pico_base > 10.0

print("=" * 65)
if es_AM:
    print(f"(a) La señal SÍ es una Modulación AM (Pico dominante detectado).")
    print(f"    -> Prominencia: {relacion_pico_base:.2f}x sobre el fondo espectral.")
else:
    print(f"(a) La señal NO presenta las características de una Modulación AM.")

print(f"(b) Frecuencia de la señal transmitida (fc): {fc_transmitida_kHz:.2f} kHz ({fc_transmitida_Hz:.0f} Hz)")
print("=" * 65)

# --- FIGURA PUNTOS (a) y (b): Espectro de Frecuencias ---
plt.figure(figsize=(10, 4))
plt.plot(frecuencias_pos, magnitud, label='Espectro de la señal', color='tab:blue')
plt.axvline(x=fc_transmitida_kHz, color='tab:red', linestyle='--', label=f'Portadora fc = {fc_transmitida_kHz:.2f} kHz')
plt.title('Verificación espectral de la modulación AM')
plt.xlabel('Frecuencia (kHz)')
plt.ylabel('Magnitud')
plt.grid(True)

# =========================================================
# MODIFICACIÓN DE ESCALA (Ajusta los valores según tu gusto)
# =========================================================
# Opción 1: ZOOM en el eje X (por ejemplo, centrado entre 0 y 40 kHz para ver bien los picos)
plt.xlim([0, 40])  

# Opción 2: Si quieres un zoom súper cercano alrededor de los 20 kHz (de 10 a 30 kHz)
# plt.xlim([10, 30])

# Opción 3: Limitar el eje Y si la portadora es muy alta y no deja ver las bandas laterales
# plt.ylim([0, pico_max * 1.1]) 
# =========================================================

plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# 2. TRANSFORMADA DE HILBERT Y FILTROS SEGÚN PUNTO (e)
# ==========================================
magnitud_hilbert = np.abs(hilbert(y_recibida))

# Frecuencia de corte = Frecuencia transmitida (fc = 20 kHz) según punto (e)
fc_corte = fc_transmitida_Hz     #20,000 (frecuencia de la portadora) o 5kHz frecuencia de la moduladora

# (c) y (e) FIR Orden 50 (51 numtaps)
b_fir_50 = firwin(51, fc_corte / nyquist, pass_zero=True)
mag_fir_e = filtfilt(b_fir_50, [1.0], magnitud_hilbert)

# (d) y (e) IIR Butterworth Orden < 10 (Orden 4)
b_iir_e, a_iir_e = butter(4, fc_corte / nyquist, btype='low')
mag_iir_e = filtfilt(b_iir_e, a_iir_e, magnitud_hilbert)

# ==========================================
# 3. FIGURAS SEPARADAS PARA PUNTO (c) Y PUNTO (d)
# ==========================================

# --- GRAFICA PARA EL PUNTO (c) ---
fig_c, (ax1_c, ax2_c, ax3_c) = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

ax1_c.plot(t_ms, y_recibida, color='tab:purple', linewidth=0.8, label='Señal Original Recibida')
ax1_c.set_title('(c) Estructura: Original -> Hilbert -> Filtrada FIR Orden 50')
ax1_c.set_ylabel('Amplitud')
ax1_c.grid(True)
ax1_c.legend(loc='upper right')

ax2_c.plot(t_ms, magnitud_hilbert, color='tab:orange', linewidth=1.0, label='Magnitud Transformada de Hilbert')
ax2_c.set_ylabel('Amplitud')
ax2_c.grid(True)
ax2_c.legend(loc='upper right')

ax3_c.plot(t_ms, mag_fir_e, color='tab:red', linewidth=1.2, label=f'Señal Filtrada FIR (fc = {fc_transmitida_kHz:.1f} kHz)')
ax3_c.set_xlabel('Tiempo (ms)')
ax3_c.set_ylabel('Amplitud')
ax3_c.set_xlim([0, t_graficar_ms])
ax3_c.grid(True)
ax3_c.legend(loc='upper right')

plt.tight_layout()
plt.show()

# --- GRAFICA PARA EL PUNTO (d) ---
fig_d, (ax1_d, ax2_d, ax3_d) = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

ax1_d.plot(t_ms, y_recibida, color='tab:purple', linewidth=0.8, label='Señal Original Recibida')
ax1_d.set_title('(d) Estructura: Original -> Hilbert -> Filtrada IIR Orden 4')
ax1_d.set_ylabel('Amplitud')
ax1_d.grid(True)
ax1_d.legend(loc='upper right')

ax2_d.plot(t_ms, magnitud_hilbert, color='tab:orange', linewidth=1.0, label='Magnitud Transformada de Hilbert')
ax2_d.set_ylabel('Amplitud')
ax2_d.grid(True)
ax2_d.legend(loc='upper right')

ax3_d.plot(t_ms, mag_iir_e, color='tab:green', linewidth=1.2, label=f'Señal Filtrada IIR (fc = {fc_transmitida_kHz:.1f} kHz)')
ax3_d.set_xlabel('Tiempo (ms)')
ax3_d.set_ylabel('Amplitud')
ax3_d.set_xlim([0, t_graficar_ms])
ax3_d.grid(True)
ax3_d.legend(loc='upper right')

plt.tight_layout()
plt.show()

# ==========================================
# 4. TEORÍA PUNTOS (e) Y (f)
# ==========================================
print("=" * 65)
print("(e) EXPLICACIÓN:")
print("    - Al usar fc = 20 kHz (frecuencia de la portadora), el filtro deja pasar")
print("      componentes hasta esa frecuencia. Sin embargo, la magnitud de Hilbert")
print("      ya realizó la detección de envolvente, eliminando la portadora y dejando")
print("      la señal moduladora (seno de ~5 kHz) más un leve rizo residual.")
print("\n(f) DIFERENCIAS FIR vs IIR:")
print("    - Filtro FIR (Orden 50): Presenta respuesta de fase lineal pero una")
print("      zona de transición más ancha.")
print("    - Filtro IIR (Butterworth Orden 4): Ofrece una pendiente de atenuación")
print("      más pronunciada con un orden menor, atenuando ligeramente mejor el rizo.")
print("=" * 65)