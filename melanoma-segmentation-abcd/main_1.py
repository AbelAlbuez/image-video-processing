import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocesar_y_segmentar(ruta_imagen):
    """
    Carga una imagen, aplica el pipeline de preprocesamiento y segmentación.
    """
    # Cargar la imagen
    # OpenCV carga las imágenes en formato BGR (Blue, Green, Red)
    imagen_bgr = cv2.imread(ruta_imagen)
    if imagen_bgr is None:
        print(f"Error: No se pudo cargar la imagen en la ruta: {ruta_imagen}")
        return None, None, None

    # --- FASE 1: PREPROCESAMIENTO ---

    # 1. Convertir de BGR a HSV [cite: 42, 96]
    imagen_hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)

    # 2. Aplicar un Filtrado Mediana para reducir ruido [cite: 45]
    # El '9' es el tamaño del kernel, se puede ajustar.
    imagen_filtrada = cv2.medianBlur(imagen_hsv, 9)

    # 3. Realce de Contraste con CLAHE [cite: 45]
    h, s, v = cv2.split(imagen_filtrada) # Separar canales
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v)
    imagen_hsv_mejorada = cv2.merge([h, s, v_clahe]) # Unir canales de nuevo

    # --- FASE 2: SEGMENTACIÓN ---

    # Definir el rango de color para la segmentación en HSV.
    # Estos valores son un punto de partida para tonos de piel y lesiones marrones/rojizas.
    # ¡NECESITARÁS AJUSTARLOS!
    limite_inferior = np.array([0, 40, 50])
    limite_superior = np.array([30, 255, 255])

    # Crear la máscara binaria [cite: 98]
    mascara = cv2.inRange(imagen_hsv_mejorada, limite_inferior, limite_superior)

    # --- GENERACIÓN DE VISUALIZACIONES ---

    # Imagen original convertida a RGB para mostrarla correctamente con Matplotlib
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

    # Encontrar contornos en la máscara para dibujarlos
    contornos, _ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos sobre la imagen original
    imagen_con_contorno = imagen_rgb.copy()
    cv2.drawContours(imagen_con_contorno, contornos, -1, (0, 255, 0), 3) # Dibuja en color verde y grosor 3

    return imagen_rgb, mascara, imagen_con_contorno

# --- EJECUCIÓN DEL SCRIPT ---

# Reemplaza 'ruta/a/tu/imagen.jpg' con la ruta de una imagen del dataset HAM10000
ruta_de_ejemplo = 'img/ISIC_0024307.jpg'

original, mascara_generada, contorno_superpuesto = preprocesar_y_segmentar(ruta_de_ejemplo)

if original is not None:
    # Mostrar los resultados usando Matplotlib
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mascara_generada, cmap='gray')
    plt.title('Máscara Generada')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(contorno_superpuesto)
    plt.title('Contorno Superpuesto')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
