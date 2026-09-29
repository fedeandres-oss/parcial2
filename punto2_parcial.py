import sys
import os
import glob
import re
import numpy as np
import pandas as pd
import scipy.signal as signal
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QPushButton, QLabel, QTextEdit, QFileDialog, QSlider, QGroupBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    """Clase para integrar gráficos de Matplotlib dentro de la interfaz PyQt5."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solución Parcial - Procesamiento de Señales e Imágenes")
        self.resize(1300, 920)

        # Gotero: Lista para almacenar hasta 5 colores seleccionados por el usuario
        self.colores_gotero = []
        self.max_colores = 5
        self.tolerancia_color = 40.0
        
        # Color Gris Claro de reemplazo [R, G, B]
        self.gris_claro = np.array([210, 210, 210], dtype=np.float32)

        # Panel de pestañas
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_punto1 = QWidget()
        self.tab_punto2 = QWidget()
        self.tab_punto3 = QWidget()

        self.tabs.addTab(self.tab_punto1, "Punto 1: Análisis AM (FIR e IIR)")
        self.tabs.addTab(self.tab_punto2, "Punto 2: Demodulación de Señal")
        self.tabs.addTab(self.tab_punto3, "Punto 3: Filtrado Espacial en Imagen Modificada")

        self.init_punto1()
        self.init_punto2()
        self.init_punto3()

    # =========================================================================
    # PESTAÑA 1: MODULACIÓN AM Y FILTRADO FIR / IIR
    # =========================================================================
    def init_punto1(self):
        layout = QVBoxLayout()

        top_layout = QHBoxLayout()
        self.btn_run_p1 = QPushButton("Buscar CSV (*_1_modulated_signal_*.csv) y Procesar")
        self.btn_run_p1.clicked.connect(self.procesar_punto1)
        top_layout.addWidget(self.btn_run_p1)
        layout.addLayout(top_layout)

        self.txt_p1_results = QTextEdit()
        self.txt_p1_results.setMaximumHeight(160)
        self.txt_p1_results.setReadOnly(True)
        layout.addWidget(self.txt_p1_results)

        self.canvas_p1 = MplCanvas(self, width=10, height=6, dpi=100)
        layout.addWidget(self.canvas_p1)

        self.tab_punto1.setLayout(layout)

    def procesar_punto1(self):
        matched_files = glob.glob("*_1_modulated_signal_*.csv")
        if not matched_files:
            self.txt_p1_results.setText("Error: No se encontró ningún archivo *_1_modulated_signal_*.csv en la carpeta.")
            return

        filename = matched_files[0]
        pattern = r".*_1_modulated_signal_([\d.]+)_([\d.]+)_([\d.]+)\.csv"
        match = re.search(pattern, filename)

        if not match:
            self.txt_p1_results.setText("Error: El archivo no cumple con la nomenclatura de parámetros.")
            return

        f_carrier_kHz = float(match.group(1))
        fs_kHz = float(match.group(2))
        T_ms = float(match.group(3))

        fc = f_carrier_kHz * 1000.0
        fs = fs_kHz * 1000.0

        df = pd.read_csv(filename, header=None)
        if isinstance(df.iloc[0, 0], str) and not df.iloc[0, 0].replace('.','',1).replace('e','',1).replace('-','',1).isdigit():
            signal_val = df.iloc[1:, 1].astype(float).values
        else:
            signal_val = df.iloc[:, 1].astype(float).values if df.shape[1] > 1 else df.iloc[:, 0].astype(float).values

        analytic_signal = signal.hilbert(signal_val)
        envelope_raw = np.abs(analytic_signal)
        envelope = envelope_raw - np.min(envelope_raw)

        freqs = np.fft.rfftfreq(len(envelope), d=1/fs)
        fft_env = np.abs(np.fft.rfft(envelope - np.mean(envelope)))
        idx_peak = np.argmax(fft_env[1:]) + 1
        f_mensaje = freqs[idx_peak]

        cutoff = fc / 2.0  

        num_fir = signal.firwin(51, cutoff=cutoff, fs=fs)
        env_fir = signal.lfilter(num_fir, 1.0, envelope)

        b_iir, a_iir = signal.butter(4, Wn=cutoff, btype='low', fs=fs)
        env_iir = signal.lfilter(b_iir, a_iir, envelope)

        res = f"--- RESULTADOS PUNTO 1 ---\n"
        res += f"• Archivo: {filename}\n"
        res += f"• Parámetros: f_portadora = {f_carrier_kHz} kHz | fs = {fs_kHz} kHz | T = {T_ms} ms\n"
        res += f"(a) Modulación AM: La amplitud varía proporcionalmente a la señal mensaje.\n"
        res += f"(b) Frecuencia del mensaje: {f_mensaje:.2f} Hz ({f_mensaje/1000:.2f} kHz)\n"
        res += f"(e) Offset eliminado: Mínimo ajustado en y = 0.\n"
        res += f"(f) FIR vs IIR: FIR conserva fase lineal; IIR ofrece corte empinado con menor orden."

        self.txt_p1_results.setText(res)

        t = np.arange(len(signal_val)) / fs
        samples_view = min(1500, len(signal_val))

        self.canvas_p1.fig.clear()

        ax1 = self.canvas_p1.fig.add_subplot(311)
        ax1.plot(t[:samples_view]*1000, signal_val[:samples_view], label="Señal Original AM", color='navy', alpha=0.7)
        ax1.set_title("(c) Señal AM Original")
        ax1.set_ylabel("Amplitud")
        ax1.grid(True)

        ax2 = self.canvas_p1.fig.add_subplot(312, sharex=ax1)
        ax2.plot(t[:samples_view]*1000, envelope[:samples_view], label="Envolvente (Sin Offset)", color='orange')
        ax2.plot(t[:samples_view]*1000, env_fir[:samples_view], label="Filtrada FIR (Orden 50)", color='darkgreen', linewidth=1.5)
        ax2.set_title("(c) Hilbert + Filtro FIR")
        ax2.set_ylabel("Amplitud")
        ax2.legend(loc='upper right')
        ax2.grid(True)

        ax3 = self.canvas_p1.fig.add_subplot(313, sharex=ax1)
        ax3.plot(t[:samples_view]*1000, envelope[:samples_view], label="Envolvente (Sin Offset)", color='orange')
        ax3.plot(t[:samples_view]*1000, env_iir[:samples_view], label="Filtrada IIR (Butterworth Ord. 4)", color='crimson', linewidth=1.5)
        ax3.set_title("(d) Hilbert + Filtro IIR")
        ax3.set_xlabel("Tiempo (ms)")
        ax3.set_ylabel("Amplitud")
        ax3.legend(loc='upper right')
        ax3.grid(True)

        self.canvas_p1.fig.tight_layout()
        self.canvas_p1.draw()

    # =========================================================================
    # PESTAÑA 2: DEMODULACIÓN HILBERT
    # =========================================================================
    def init_punto2(self):
        layout = QVBoxLayout()

        top_layout = QHBoxLayout()
        self.btn_run_p2 = QPushButton("Buscar CSV y Procesar Señal")
        self.btn_run_p2.clicked.connect(self.procesar_punto2)
        top_layout.addWidget(self.btn_run_p2)
        layout.addLayout(top_layout)

        self.txt_p2_results = QTextEdit()
        self.txt_p2_results.setMaximumHeight(110)
        self.txt_p2_results.setReadOnly(True)
        layout.addWidget(self.txt_p2_results)

        self.canvas_p2 = MplCanvas(self, width=10, height=6, dpi=100)
        layout.addWidget(self.canvas_p2)

        self.tab_punto2.setLayout(layout)

    def procesar_punto2(self):
        matched_files = glob.glob("*_2_modulated_signal_*.csv")
        if not matched_files:
            self.txt_p2_results.setText("Error: No se encontró ningún archivo *_2_modulated_signal_*.csv en la carpeta.")
            return

        filename = matched_files[0]
        pattern = r".*_2_modulated_signal_([\d.]+)_([\d.]+)_([\d.]+)\.csv"
        match = re.search(pattern, filename)

        if not match:
            self.txt_p2_results.setText("Error: El archivo no cumple con la nomenclatura de parámetros.")
            return

        f_carrier_kHz = float(match.group(1))
        fs_kHz = float(match.group(2))
        T_bit_ms = float(match.group(3))

        fs = fs_kHz * 1000.0
        T_bit = T_bit_ms / 1000.0
        samples_per_bit = int(round(fs * T_bit))

        df = pd.read_csv(filename, header=None)
        if isinstance(df.iloc[0, 0], str) and not df.iloc[0, 0].replace('.','',1).replace('e','',1).replace('-','',1).isdigit():
            signal_val = df.iloc[1:, 1].astype(float).values
        else:
            signal_val = df.iloc[:, 1].astype(float).values if df.shape[1] > 1 else df.iloc[:, 0].astype(float).values

        analytic_signal = signal.hilbert(signal_val)
        envelope_raw = np.abs(analytic_signal)
        envelope = envelope_raw - np.min(envelope_raw)

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

        chars = []
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            if len(byte_bits) == 8:
                byte_str = "".join(map(str, byte_bits))
                chars.append(chr(int(byte_str, 2)))

        phrase = "".join(chars)

        res = f"Archivo detectado: {filename}\n"
        res += f"Parámetros: f = {f_carrier_kHz} kHz | fs = {fs_kHz} kHz | T = {T_bit_ms} ms\n"
        res += f"Bits extraídos: {''.join(map(str, bits))}\n"
        res += f"Frase asignada decodificada: {phrase}"
        self.txt_p2_results.setText(res)

        t = np.arange(len(signal_val)) / fs
        self.canvas_p2.fig.clear()

        ax1 = self.canvas_p2.fig.add_subplot(311)
        ax1.plot(t[:1000], signal_val[:1000], color='tab:blue')
        ax1.set_title("1. Señal Original Modulada")
        ax1.grid(True)

        ax2 = self.canvas_p2.fig.add_subplot(312, sharex=ax1)
        ax2.plot(t[:1000], envelope[:1000], color='tab:orange')
        ax2.set_title("2. Magnitud de Hilbert (Sin Offset)")
        ax2.grid(True)

        ax3 = self.canvas_p2.fig.add_subplot(313, sharex=ax1)
        ax3.plot(t[:1000], square_wave[:1000], color='tab:green')
        ax3.set_title("3. Señal Cuadrada Digital")
        ax3.set_xlabel("Tiempo (s)")
        ax3.grid(True)

        self.canvas_p2.fig.tight_layout()
        self.canvas_p2.draw()

    # =========================================================================
    # PESTAÑA 3: FILTRADO ESPACIAL SOBRE LA IMAGEN MODIFICADA
    # =========================================================================
    def init_punto3(self):
        layout = QVBoxLayout()

        top_layout = QHBoxLayout()
        self.btn_load_img = QPushButton("Buscar Imagen en PC")
        self.btn_load_img.clicked.connect(self.seleccionar_imagen)
        
        self.btn_direct_img = QPushButton("Cargar 'images (1).jpg' Directo")
        self.btn_direct_img.clicked.connect(lambda: self.ejecutar_filtrado_imagen("images (1).jpg"))

        self.btn_clear_colors = QPushButton("🧹 Limpiar Selección del Gotero")
        self.btn_clear_colors.clicked.connect(self.limpiar_colores)

        top_layout.addWidget(self.btn_load_img)
        top_layout.addWidget(self.btn_direct_img)
        top_layout.addWidget(self.btn_clear_colors)
        layout.addLayout(top_layout)

        ctrl_group = QGroupBox("Controles Interactivos")
        ctrl_layout = QHBoxLayout()

        self.lbl_slider_kernel = QLabel("Tamaño Kernel (NxN): 5x5")
        self.slider_order = QSlider(Qt.Horizontal)
        self.slider_order.setMinimum(3)
        self.slider_order.setMaximum(15)
        self.slider_order.setSingleStep(2)
        self.slider_order.setValue(5)
        self.slider_order.valueChanged.connect(self.actualizar_controles_p3)

        self.lbl_slider_tol = QLabel("Tolerancia Gotero: 40")
        self.slider_tol = QSlider(Qt.Horizontal)
        self.slider_tol.setMinimum(5)
        self.slider_tol.setMaximum(150)
        self.slider_tol.setValue(40)
        self.slider_tol.valueChanged.connect(self.actualizar_controles_p3)

        ctrl_layout.addWidget(self.lbl_slider_kernel)
        ctrl_layout.addWidget(self.slider_order)
        ctrl_layout.addWidget(self.lbl_slider_tol)
        ctrl_layout.addWidget(self.slider_tol)
        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)

        self.lbl_img = QLabel("💡 Azul a GRIS CLARO automático. Los filtros espaciales se aplican SOBRE la imagen modificada.")
        layout.addWidget(self.lbl_img)

        self.canvas_p3 = MplCanvas(self, width=11, height=7, dpi=100)
        layout.addWidget(self.canvas_p3)

        self.canvas_p3.mpl_connect('button_press_event', self.on_click_gotero)

        self.tab_punto3.setLayout(layout)

        self.ruta_imagen_actual = "images (1).jpg" if os.path.exists("images (1).jpg") else None
        if self.ruta_imagen_actual:
            self.ejecutar_filtrado_imagen(self.ruta_imagen_actual)

    def actualizar_controles_p3(self, val):
        val_kernel = self.slider_order.value()
        if val_kernel % 2 == 0:
            val_kernel += 1
            self.slider_order.setValue(val_kernel)

        self.lbl_slider_kernel.setText(f"Tamaño Kernel (NxN): {val_kernel}x{val_kernel}")
        self.tolerancia_color = float(self.slider_tol.value())
        self.lbl_slider_tol.setText(f"Tolerancia Gotero: {int(self.tolerancia_color)}")

        if hasattr(self, 'ruta_imagen_actual') and self.ruta_imagen_actual:
            self.ejecutar_filtrado_imagen(self.ruta_imagen_actual)

    def limpiar_colores(self):
        self.colores_gotero.clear()
        self.lbl_img.setText("🧹 Colores del gotero limpiados. Reaplicando filtros en la imagen ajustada.")
        if hasattr(self, 'ruta_imagen_actual') and self.ruta_imagen_actual:
            self.ejecutar_filtrado_imagen(self.ruta_imagen_actual)

    def seleccionar_imagen(self):
        archivo, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Imagen", "", "Archivos de Imagen (*.jpg *.jpeg *.png *.bmp)"
        )
        if archivo:
            self.colores_gotero.clear()
            self.ruta_imagen_actual = archivo
            self.ejecutar_filtrado_imagen(archivo)

    def on_click_gotero(self, event):
        """Captura hasta 5 colores al hacer clic con el ratón."""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        if event.inaxes == self.ax_original and hasattr(self, 'img_rgb'):
            col = int(round(event.xdata))
            row = int(round(event.ydata))

            if 0 <= row < self.img_rgb.shape[0] and 0 <= col < self.img_rgb.shape[1]:
                color_click = self.img_rgb[row, col, :3].astype(float)
                
                if len(self.colores_gotero) < self.max_colores:
                    self.colores_gotero.append(color_click)
                else:
                    self.colores_gotero.pop(0)
                    self.colores_gotero.append(color_click)

                r, g, b = color_click.astype(int)
                msg = f"🎯 Gotero ({len(self.colores_gotero)}/5): RGB({r},{g},{b}) -> Gris Claro (Filtros actualizados)."
                self.lbl_img.setText(msg)
                self.ejecutar_filtrado_imagen(self.ruta_imagen_actual)

    def ejecutar_filtrado_imagen(self, ruta_archivo):
        if not os.path.exists(ruta_archivo):
            self.lbl_img.setText(f"Error: No se encontró el archivo {os.path.basename(ruta_archivo)}")
            return

        self.ruta_imagen_actual = ruta_archivo
        N = self.slider_order.value()
        
        try:
            pil_img_rgb = Image.open(ruta_archivo).convert('RGB')
            self.img_rgb = np.array(pil_img_rgb, dtype=np.float32)
        except Exception as e:
            self.lbl_img.setText(f"Error al cargar imagen: {str(e)}")
            return

        # ---------------------------------------------------------------------
        # 1. DETECCIÓN DE TONOS AZULES
        # ---------------------------------------------------------------------
        r_chan = self.img_rgb[:, :, 0]
        g_chan = self.img_rgb[:, :, 1]
        b_chan = self.img_rgb[:, :, 2]

        mask_azul = (b_chan > 1.2 * r_chan) & (b_chan > 1.1 * g_chan) & (b_chan > 50)

        # ---------------------------------------------------------------------
        # 2. GOTERO HASTA 5 COLORES ADICIONALES
        # ---------------------------------------------------------------------
        mask_gotero = np.zeros((self.img_rgb.shape[0], self.img_rgb.shape[1]), dtype=bool)

        for col_selected in self.colores_gotero:
            dist = np.sqrt(np.sum((self.img_rgb[:, :, :3] - col_selected)**2, axis=2))
            mask_gotero |= (dist < self.tolerancia_color)

        mask_total = mask_azul | mask_gotero

        # Generar Imagen Modificada (Azul + Gotero -> Gris Claro [210, 210, 210])
        img_modificada = self.img_rgb.copy()
        img_modificada[mask_total] = self.gris_claro

        # ---------------------------------------------------------------------
        # 3. ESCALA DE GRISES A PARTIR DE LA IMAGEN MODIFICADA
        # ---------------------------------------------------------------------
        # Convertimos la imagen ya modificada a escala de grises para alimentar los filtros
        imagen_gray_modificada = (
            0.299 * img_modificada[:, :, 0] + 
            0.587 * img_modificada[:, :, 1] + 
            0.114 * img_modificada[:, :, 2]
        )

        # ---------------------------------------------------------------------
        # 4. KERNELS Y CONVOLUCIONES 2D SOBRE LA IMAGEN MODIFICADA
        # ---------------------------------------------------------------------
        k_pb = np.ones((N, N), dtype=np.float32) / float(N * N)

        k_pa = -np.ones((N, N), dtype=np.float32)
        k_pa[N // 2, N // 2] = (N * N) - 1.0

        k_cruz = np.zeros((N, N), dtype=np.float32)
        k_cruz[N // 2, :] = -1.0
        k_cruz[:, N // 2] = -1.0
        k_cruz[N // 2, N // 2] = 2 * N - 2.0

        k_mas_agresivo = -2.0 * np.ones((N, N), dtype=np.float32)
        k_mas_agresivo[N // 2, N // 2] = 2.0 * (N * N - 1)

        # Convoluciones usando la versión en grises de la IMAGEN MODIFICADA
        img_pb = signal.convolve2d(imagen_gray_modificada, k_pb, mode='same', boundary='symm')
        img_pa = signal.convolve2d(imagen_gray_modificada, k_pa, mode='same', boundary='symm')
        img_bandas = signal.convolve2d(img_pb, k_pa, mode='same', boundary='symm')
        img_cruz = signal.convolve2d(imagen_gray_modificada, k_cruz, mode='same', boundary='symm')
        img_mas_agr = signal.convolve2d(imagen_gray_modificada, k_mas_agresivo, mode='same', boundary='symm')

        # DIBUJO EN MATPLOTLIB
        self.canvas_p3.fig.clear()

        # Subplot 1: Original en Color (Captura de clics)
        self.ax_original = self.canvas_p3.fig.add_subplot(2, 4, 1)
        self.ax_original.imshow(self.img_rgb.astype(np.uint8))
        self.ax_original.set_title("Original (Clic: Gotero)", fontsize=8, color='navy', fontweight='bold')
        self.ax_original.axis('off')

        # Subplot 2: Resultado Azul + Gotero en Gris Claro
        ax_gotero = self.canvas_p3.fig.add_subplot(2, 4, 2)
        ax_gotero.imshow(img_modificada.astype(np.uint8))
        ax_gotero.set_title(f"Imagen Modificada (Base para Filtros)", fontsize=8, color='crimson', fontweight='bold')
        ax_gotero.axis('off')

        # Subplots 3 al 8: Filtros espaciales sobre la imagen modificada
        filtros_restantes = [
            (f"Pasa Bajas ({N}x{N})", img_pb),
            (f"Pasa Altas (Suma=0)", img_pa),
            (f"Pasa Bandas", img_bandas),
            (f"Filtro Cruz (Suma=0)", img_cruz),
            (f"Más Agresivo", img_mas_agr),
            (f"Grises (Modificada)", imagen_gray_modificada)
        ]

        for idx, (titulo, img_proc) in enumerate(filtros_restantes, 3):
            ax = self.canvas_p3.fig.add_subplot(2, 4, idx)
            ax.imshow(img_proc, cmap='gray')
            ax.set_title(titulo, fontsize=8)
            ax.axis('off')

        self.canvas_p3.fig.tight_layout()
        self.canvas_p3.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec_())
