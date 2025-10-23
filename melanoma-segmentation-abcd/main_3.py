import cv2
import numpy as np
import matplotlib.pyplot as plt

def refinar_mascara(mascara):
    """
    Aplica operaciones morfológicas para limpiar la máscara.
    (Esta función sigue siendo muy necesaria y no cambia).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara_refinada = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=2)
    mascara_refinada = cv2.morphologyEx(mascara_refinada, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mascara_refinada

def procesar_imagen_lesion_otsu(ruta_imagen):
    """
    Pipeline mejorado que usa la Binarización de Otsu para una segmentación adaptativa.
    """
    imagen_bgr = cv2.imread(ruta_imagen)
    if imagen_bgr is None:
        print(f"❌ Error: No se pudo cargar la imagen. Revisa la ruta: {ruta_imagen}")
        return

    # --- FASE 1: PREPROCESAMIENTO ---
    # Extraer el canal Azul, que suele tener buen contraste para lesiones pigmentadas.
    canal_azul = imagen_bgr[:, :, 0]
    
    # Aplicar un desenfoque Gaussiano para suavizar la imagen antes de Otsu.
    # Esto ayuda al algoritmo a encontrar un mejor umbral.
    blur = cv2.GaussianBlur(canal_azul, (5, 5), 0)

    # --- FASE 2: SEGMENTACIÓN ADAPTATIVA (MÉTODO DE OTSU) ---
    # cv2.threshold calcula el umbral óptimo automáticamente gracias a THRESH_OTSU.
    # Usamos THRESH_BINARY_INV porque las lesiones son oscuras (valores bajos)
    # y queremos que se conviertan en blanco (255) en la máscara.
    _, mascara_inicial = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- FASE 3: REFINAMIENTO ---
    # La máscara de Otsu también se beneficia de la limpieza morfológica.
    mascara_refinada = refinar_mascara(mascara_inicial)

    # --- FASE 4: EXTRACCIÓN Y VISUALIZACIÓN ---
    contornos, _ = cv2.findContours(mascara_refinada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    imagen_con_contorno = imagen_rgb.copy()
    
    if len(contornos) > 0:
        cnt = max(contornos, key=cv2.contourArea)
        cv2.drawContours(imagen_con_contorno, [cnt], -1, (0, 255, 0), 3)
        
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)
        
        print("\n--- 📊 Resultados del Análisis (Otsu) ---")
        print(f"Área de la lesión: {area:.2f} píxeles cuadrados")
        print(f"Perímetro de la lesión: {perimetro:.2f} píxeles")
        print("---------------------------------------")
    else:
        print("\n⚠️ No se encontraron contornos después del refinamiento.")

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
    # ⬇️ PRUEBA CON LA MISMA IMAGEN QUE DIO PROBLEMAS ⬇️
    ruta_de_imagen_prueba = 'img/ISIC_0024307.jpg' 
    procesar_imagen_lesion_otsu(ruta_de_imagen_prueba)