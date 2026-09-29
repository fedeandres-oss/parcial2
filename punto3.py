import cv2
import numpy as np
from PIL import Image
import os
import glob

# 1. Obtener la carpeta donde se encuentra este script
carpeta = os.path.dirname(os.path.abspath(__file__))

# 2. Buscar automáticamente archivos de imagen en la misma carpeta
extensiones = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
archivos_encontrados = []
for ext in extensiones:
    archivos_encontrados.extend(glob.glob(os.path.join(carpeta, ext)))

if not archivos_encontrados:
    raise FileNotFoundError("No se encontró ninguna imagen (.jpg, .png, etc.) en la carpeta del script.")

# Prioriza 'flower.jpg' si existe; si no, toma la primera imagen que encuentre
ruta_imagen = os.path.join(carpeta, 'flower.jpg')
if not os.path.exists(ruta_imagen):
    ruta_imagen = archivos_encontrados[0]

print(f"Imagen seleccionada: {os.path.basename(ruta_imagen)}")

# 3. Convertir la imagen a escala de grises usando PIL y guardarla temporalmente
ruta_gris = os.path.join(carpeta, 'imagen_gris.jpg')
imagen = Image.open(ruta_imagen)
imagen_gris = imagen.convert('L')
imagen_gris.save(ruta_gris)

# 4. Cargar con OpenCV en escala de grises
imagen = cv2.imread(ruta_gris, cv2.IMREAD_GRAYSCALE)

if imagen is None:
    raise FileNotFoundError("No se pudo cargar la imagen en escala de grises.")

# Redimensionar la imagen base
alto, ancho = 200, 200
imagen = cv2.resize(imagen, (ancho, alto))

def preparar(img, titulo):
    img_visible = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_visible = img_visible.astype('uint8')
    img_visible = cv2.resize(img_visible, (ancho, alto))
    img_color = cv2.cvtColor(img_visible, cv2.COLOR_GRAY2BGR)
    cv2.putText(img_color, titulo, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img_color

# DEFINICIÓN DE KERNELS DE FILTRADO
kernel_pasa_bajas = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                               [0.04, 0.04, 0.04, 0.04, 0.04],
                               [0.04, 0.04, 0.04, 0.04, 0.04],
                               [0.04, 0.04, 0.04, 0.04, 0.04],
                               [0.04, 0.04, 0.04, 0.04, 0.04]])

kernel_pasa_altas02 = np.array([[0,  0,  0,  0, 0],
                                 [0, -3, -3, -3, 0],
                                 [0, -3, 24, -3, 0],
                                 [0, -3, -3, -3, 0],
                                 [0,  0,  0,  0, 0]])

kernel_nuevo = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])

kernel_variacion1 = np.array([[-2, -2, -2],
                               [-2, 16, -2],
                               [-2, -2, -2]])

kernel_variacion2 = np.array([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]])

print("Suma kernel_nuevo:", kernel_nuevo.sum())
print("Suma kernel_variacion1:", kernel_variacion1.sum())
print("Suma kernel_variacion2:", kernel_variacion2.sum())

# APLICACIÓN DE FILTROS Y PREPARACIÓN DE VISUALIZACIÓN
original = preparar(imagen, 'Original')

pasa_altas = cv2.filter2D(imagen, -1, kernel_pasa_altas02)
pasa_altas = preparar(pasa_altas, 'Pasa Altas')

pasa_bajas = cv2.filter2D(imagen, -1, kernel_pasa_bajas)
pasa_bajas = preparar(pasa_bajas, 'Pasa Bajas')

pasa_bandas = cv2.filter2D(imagen, -1, kernel_pasa_bajas)
pasa_bandas = cv2.filter2D(pasa_bandas, -1, kernel_pasa_altas02)
pasa_bandas = preparar(pasa_bandas, 'Pasa Bandas')

nuevo = cv2.filter2D(imagen, -1, kernel_nuevo)
nuevo = preparar(nuevo, 'Suma=0')

variacion1 = cv2.filter2D(imagen, -1, kernel_variacion1)
variacion1 = preparar(variacion1, 'Mas agresivo')

variacion2 = cv2.filter2D(imagen, -1, kernel_variacion2)
variacion2 = preparar(variacion2, 'Solo cruz')

# ARREGLO Y MOSTRADO EN GRILLA DE OPENCV
fila1 = np.hstack([original, pasa_altas, pasa_bajas, pasa_bandas])
fila2 = np.hstack([nuevo, variacion1, variacion2, np.zeros_like(original)])
grid = np.vstack([fila1, fila2])

cv2.imshow('Resultados de Filtros', grid)
cv2.waitKey(0)
cv2.destroyAllWindows()