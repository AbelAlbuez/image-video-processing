import cv2
import numpy as np
import matplotlib.pyplot as plt

def refinar_mascara(mascara):
    """
    Aplica operaciones morfológicas para limpiar la máscara, eliminando ruido
    y rellenando huecos internos.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara_refinada = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=2)
    mascara_refinada = cv2.morphologyEx(mascara_refinada, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mascara_refinada

def analizar_lesion(ruta_imagen):
    """
    Pipeline completo: Carga, segmenta, refina y extrae descriptores de una imagen.
    """
    imagen_bgr = cv2.imread(ruta_imagen)
    if imagen_bgr is None:
        print(f"Error: No se pudo cargar la imagen. Revisa la ruta: {ruta_imagen}")
        return

    # --- FASE 1: PREPROCESAMIENTO ---
    canal_azul = imagen_bgr[:, :, 0]
    blur = cv2.GaussianBlur(canal_azul, (5, 5), 0)

    # --- FASE 2: SEGMENTACIÓN ADAPTATIVA (OTSU) ---
    _, mascara_inicial = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- FASE 3: REFINAMIENTO ---
    mascara_refinada = refinar_mascara(mascara_inicial)

    # --- FASE 4: EXTRACCIÓN DE DESCRIPTORES ---
    contornos, _ = cv2.findContours(mascara_refinada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    imagen_con_contorno = imagen_rgb.copy()
    
    print(f"\n--- 📊 Resultados para: {ruta_imagen} ---")

    if len(contornos) > 0:
        cnt = max(contornos, key=cv2.contourArea)
        cv2.drawContours(imagen_con_contorno, [cnt], -1, (0, 255, 0), 3)
        
        # Cálculos de Área y Perímetro
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)
        
        # Descriptor B: Borde (Índice de Circularidad)
        if perimetro > 0:
            circularidad = (4 * np.pi * area) / (perimetro**2)
        else:
            circularidad = 0
            
        # Descriptor D: Diámetro
        (_, _), radio = cv2.minEnclosingCircle(cnt)
        diametro = radio * 2
        
        # Imprimir resultados para la tabla
        print(f"Área (píxeles²): {area:.2f}")
        print(f"Perímetro (píxeles): {perimetro:.2f}")
        print(f"Índice de Circularidad (Borde): {circularidad:.4f}")
        print(f"Diámetro Estimado (D): {diametro:.2f}")
        
    else:
        print("⚠️ No se encontraron contornos.")
    
    print("---------------------------------------")

    # --- FASE 5: VISUALIZACIÓN ---
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(imagen_rgb)
    plt.title('1. Imagen Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mascara_refinada, cmap='gray')
    plt.title('2. Máscara Refinada (Otsu)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(imagen_con_contorno)
    plt.title('3. Contorno Superpuesto')
    plt.axis('off')

    plt.suptitle('Resultados con Segmentación Adaptativa (Otsu)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- PUNTO DE ENTRADA ---
if __name__ == '__main__':
    imagenes_a_procesar = [
        'img/ISIC_0024306.jpg',
        'img/ISIC_0024307.jpg',
        'img/ISIC_0024308.jpg'
    ]
    
    for ruta in imagenes_a_procesar:
        analizar_lesion(ruta)