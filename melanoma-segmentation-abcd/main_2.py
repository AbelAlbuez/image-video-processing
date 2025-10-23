import cv2
import numpy as np
import matplotlib.pyplot as plt

def refinar_mascara(mascara):
    """
    Aplica operaciones morfológicas para limpiar la máscara, eliminando ruido
    [cite_start]y rellenando huecos internos. [cite: 15]
    """
    # Se define un 'kernel' o elemento estructurante.
    # Una forma de elipse es generalmente buena para objetos orgánicos/redondeados.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # 1. Operación de Apertura (Opening): Elimina el ruido exterior (píxeles blancos aislados).
    # Consiste en una erosión seguida de una dilatación.
    mascara_refinada = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)

    # 2. Operación de Cierre (Closing): Rellena los huecos interiores (píxeles negros dentro del objeto).
    # Consiste en una dilatación seguida de una erosión.
    mascara_refinada = cv2.morphologyEx(mascara_refinada, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mascara_refinada

def procesar_imagen_lesion(ruta_imagen):
    """
    Ejecuta el pipeline completo: carga, preprocesa, segmenta, refina y analiza una imagen de lesión cutánea.
    """
    # Cargar la imagen desde la ruta especificada.
    # OpenCV la carga en formato BGR por defecto.
    imagen_bgr = cv2.imread(ruta_imagen)
    if imagen_bgr is None:
        print(f"❌ Error: No se pudo cargar la imagen. Revisa la ruta: {ruta_imagen}")
        return

    # --- FASE 1: PREPROCESAMIENTO ---
    # [cite_start]Convertir de BGR a HSV para una mejor segmentación por color. [cite: 42, 91]
    imagen_hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)

    # Aplicar un Filtrado Mediana para suavizar y reducir ruido como vellos finos.
    imagen_filtrada = cv2.medianBlur(imagen_hsv, 9)

    # Realzar el contraste en el canal de Valor (Brillo) usando CLAHE.
    h, s, v = cv2.split(imagen_filtrada)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v)
    imagen_hsv_mejorada = cv2.merge([h, s, v_clahe])

    # --- FASE 2: SEGMENTACIÓN ---
    # Definir el rango de color en HSV para aislar la lesión.
    # ⚠️ ¡Estos valores son un punto de partida y puede que necesites ajustarlos!
    limite_inferior = np.array([0, 40, 50])
    limite_superior = np.array([35, 255, 255]) # Ajustado ligeramente para capturar más tonos rojizos

    # [cite_start]Crear la máscara binaria inicial basada en el umbral de color. [cite: 42]
    mascara_inicial = cv2.inRange(imagen_hsv_mejorada, limite_inferior, limite_superior)

    # --- FASE 3: REFINAMIENTO ---
    # [cite_start]Limpiar la máscara usando las operaciones morfológicas. [cite: 15]
    mascara_refinada = refinar_mascara(mascara_inicial)

    # --- FASE 4: EXTRACCIÓN ---
    # Encontrar los contornos en la máscara ya limpia y refinada.
    contornos, _ = cv2.findContours(mascara_refinada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Preparar la imagen original para visualización convirtiéndola a RGB.
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    imagen_con_contorno = imagen_rgb.copy()
    
    # Proceder solo si se encontraron contornos.
    if len(contornos) > 0:
        # Seleccionar el contorno más grande, asumiendo que es la lesión.
        cnt = max(contornos, key=cv2.contourArea)
        
        # Dibujar el contorno seleccionado sobre la imagen.
        cv2.drawContours(imagen_con_contorno, [cnt], -1, (0, 255, 0), 3) # Contorno verde, grosor 3
        
        # [cite_start]Calcular descriptores básicos (un adelanto de la regla ABCD). [cite: 43]
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)
        
        print("\n--- 📊 Resultados del Análisis ---")
        print(f"Área de la lesión: {area:.2f} píxeles cuadrados")
        print(f"Perímetro de la lesión: {perimetro:.2f} píxeles")
        print("---------------------------------")

    else:
        print("\n⚠️ No se encontraron contornos después del refinamiento.")

    # --- FASE 5: VISUALIZACIÓN ---
    # [cite_start]Mostrar las tres vistas clave para evaluar la calidad de la segmentación. [cite: 19]
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(imagen_rgb)
    plt.title('1. Imagen Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mascara_refinada, cmap='gray')
    plt.title('2. Máscara Refinada')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(imagen_con_contorno)
    plt.title('3. Contorno Superpuesto')
    plt.axis('off')

    plt.suptitle('Resultados del Pipeline de Segmentación', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- PUNTO DE ENTRADA DEL SCRIPT ---
if __name__ == '__main__':
    # ⬇️ REEMPLAZA ESTA LÍNEA CON LA RUTA A TU IMAGEN ⬇️
    ruta_de_imagen_prueba = 'img/ISIC_0024306.jpg' 
    procesar_imagen_lesion(ruta_de_imagen_prueba)