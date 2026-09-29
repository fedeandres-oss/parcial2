import sys
import os
import glob
import re
import numpy as np
import pandas as pd
import scipy.signal as signal

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    """Clase para integrar gráficos de Matplotlib dentro de la interfaz PyQt5."""
    def __init__(self, parent=None, width=10, height=7, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)


class VentanaPunto2(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Demodulación y Decodificación de Señal (Punto 2)")
        self.resize(1100, 850)

        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout = QVBoxLayout()

        top_layout = QHBoxLayout()
        self.btn_run_p2 = QPushButton("Cargar Archivo (*_2_modulated_signal_*.csv) y Procesar")
        self.btn_run_p2.clicked.connect(self.procesar_punto2)
        top_layout.addWidget(self.btn_run_p2)
        layout.addLayout(top_layout)

        self.txt_p2_results = QTextEdit()
        self.txt_p2_results.setMaximumHeight(200)
        self.txt_p2_results.setReadOnly(True)
        layout.addWidget(self.txt_p2_results)

        self.canvas_p2 = MplCanvas(self, width=10, height=7, dpi=100)
        layout.addWidget(self.canvas_p2)

        widget_central.setLayout(layout)

    def procesar_punto2(self):
        # Búsqueda del archivo según la nomenclatura de la guía
        matched_files = glob.glob("*_2_modulated_signal_*.csv")
        if not matched_files:
            self.txt_p2_results.setText("Error: No se encontró ningún archivo *_2_modulated_signal_*.csv en la carpeta.")
            return

        filename = matched_files[0]
        # Regex para extraer <f>, <fs> y <T>
        pattern = r".*?_2_modulated_signal_([\d\.]+)_([\d\.]+)_([\d\.]+)\.csv"
        match = re.search(pattern, filename)

        if not match:
            self.txt_p2_results.setText(f"Error: El archivo '{filename}' no cumple la nomenclatura dada.")
            return

        f_carrier_kHz = float(match.group(1))
        fs_kHz = float(match.group(2))
        T_bit_ms = float(match.group(3))

        fs = fs_kHz * 1000.0
        T_bit = T_bit_ms / 1000.0
        samples_per_bit = int(round(fs * T_bit))

        # Lectura robusta del archivo CSV
        df = pd.read_csv(filename, header=None)
        try:
            signal_val = df.iloc[:, 1].astype(float).values if df.shape[1] > 1 else df.iloc[:, 0].astype(float).values
        except ValueError:
            signal_val = df.iloc[1:, 1].astype(float).values if df.shape[1] > 1 else df.iloc[1:, 0].astype(float).values

        # ---------------------------------------------------------------------
        # (b) MAGNITUD DE LA TRANSFORMADA DE HILBERT
        # ---------------------------------------------------------------------
        analytic_signal = signal.hilbert(signal_val)
        envelope_raw = np.abs(analytic_signal)
        envelope = envelope_raw - np.min(envelope_raw)  # Remoción de offset DC

        # ---------------------------------------------------------------------
        # (a) FILTRADO FIR (Orden 50) e IIR Butterworth (Orden < 10)
        # ---------------------------------------------------------------------
        f_bit = 1.0 / T_bit
        cutoff_freq = f_bit * 2.0  # Ancho de banda para preservar componentes del bit

        # Filtro FIR Orden 50
        num_fir = signal.firwin(51, cutoff=cutoff_freq, fs=fs)
        env_fir = signal.lfilter(num_fir, 1.0, envelope)

        # Filtro IIR Butterworth Orden 4 (< 10)
        b_iir, a_iir = signal.butter(4, Wn=cutoff_freq, btype='low', fs=fs)
        env_iir = signal.lfilter(b_iir, a_iir, envelope)

        # ---------------------------------------------------------------------
        # (c) MUESTREO DE BITS, SEÑAL CUADRADA Y DECODIFICACIÓN ASCII
        # ---------------------------------------------------------------------
        threshold = (np.max(envelope) + np.min(envelope)) / 2.0
        num_bits = len(envelope) // samples_per_bit

        bits = []
        square_wave = np.zeros_like(envelope)

        for i in range(num_bits):
            start_idx = i * samples_per_bit
            end_idx = (i + 1) * samples_per_bit
            center_idx = int((i + 0.5) * samples_per_bit)

            bit_val = 1 if envelope[center_idx] > threshold else 0
            bits.append(bit_val)
            square_wave[start_idx:end_idx] = bit_val

        # Conversión de bits a ASCII
        chars = []
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            if len(byte_bits) == 8:
                byte_str = "".join(map(str, byte_bits))
                chars.append(chr(int(byte_str, 2)))

        phrase = "".join(chars)

        # Texto explicativo (Punto A, B y C)
        res = f"=== RESULTADOS Y ANÁLISIS DE LA SEÑAL ===\n"
        res += f"• Archivo: {filename}\n"
        res += f"• Parámetros: f_portadora = {f_carrier_kHz} kHz | fs = {fs_kHz} kHz | T_bit = {T_bit_ms} ms\n"
        res += f"• Bits extraídos: {''.join(map(str, bits))}\n"
        res += f"• Frase decodificada (ASCII): \"{phrase}\"\n\n"
        res += f"--- Explicación del Inciso (a) ---\n"
        res += (
            "¿Por qué no se logra ver una señal cuadrada directamente al aplicar filtros FIR/IIR?\n"
            "1. Ancho de Banda Limitado: Una señal cuadrada perfecta requiere armónicos infinitos. "
            "Al pasar por un filtro Pasa-Bajas (FIR o IIR), se eliminan los armónicos de alta frecuencia, redondeando las esquinas.\n"
            "2. Distorsión de Fase y Retardo: Los filtros IIR alteran la fase no linealmente, dispersando en el tiempo los bordes de los pulsos. "
            "Por estas razones se requiere una etapa de umbralizado o decisión para recuperar la señal cuadrada digital."
        )
        self.txt_p2_results.setText(res)

        # ---------------------------------------------------------------------
        # (b) GRÁFICAS REQUERIDAS (Señal Original -> Magnitud Hilbert -> Señal Cuadrada)
        # ---------------------------------------------------------------------
        t = np.arange(len(signal_val)) / fs
        samples_view = min(2000, len(signal_val))

        self.canvas_p2.fig.clear()

        # 1. Señal Original Modulada
        ax1 = self.canvas_p2.fig.add_subplot(311)
        ax1.plot(t[:samples_view]*1000, signal_val[:samples_view], color='tab:blue', alpha=0.8)
        ax1.set_title("1. Señal Original Modulada", fontsize=10, fontweight='bold')
        ax1.set_ylabel("Amplitud")
        ax1.grid(True)

        # 2. Magnitud de la Transformada de Hilbert (con Envolventes Filtradas)
        ax2 = self.canvas_p2.fig.add_subplot(312, sharex=ax1)
        ax2.plot(t[:samples_view]*1000, envelope[:samples_view], color='tab:orange', label="Magnitud Hilbert (Sin Offset)")
        ax2.plot(t[:samples_view]*1000, env_fir[:samples_view], color='green', linestyle='--', label="Filtrada FIR (Ord. 50)")
        ax2.plot(t[:samples_view]*1000, env_iir[:samples_view], color='red', linestyle=':', label="Filtrada IIR Butterworth (Ord. 4)")
        ax2.axhline(threshold, color='black', linestyle='-.', alpha=0.6, label="Umbral de Decisión")
        ax2.set_title("2. Magnitud de la Transformada de Hilbert (Envolvente)", fontsize=10, fontweight='bold')
        ax2.set_ylabel("Amplitud")
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True)

        # 3. Señal Cuadrada Digital Reconstruida
        ax3 = self.canvas_p2.fig.add_subplot(313, sharex=ax1)
        ax3.plot(t[:samples_view]*1000, square_wave[:samples_view], color='tab:green', linewidth=2)
        ax3.set_title("3. Señal Cuadrada Digital Reconstruida", fontsize=10, fontweight='bold')
        ax3.set_xlabel("Tiempo (ms)")
        ax3.set_ylabel("Nivel Lógico")
        ax3.set_ylim(-0.2, 1.2)
        ax3.grid(True)

        self.canvas_p2.fig.tight_layout()
        self.canvas_p2.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaPunto2()
    ventana.show()
    sys.exit(app.exec_())