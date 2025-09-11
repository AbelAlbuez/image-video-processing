"""
Taller 1 - Procesamiento de Imágenes y Video
Extracción de Objetos por Color
Autores: Abel Albuez Sanchez
Fecha: 11/09/2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import os
from datetime import datetime

class ExtractorObjetosColor:
    """
    Clase para extraer objetos de una imagen basándose en su color.
    """
    
    def __init__(self, ruta_imagen):
        """
        Inicializa el extractor con la imagen especificada.
        
        Parámetros:
        -----------
        ruta_imagen : str
            Ruta al archivo de imagen
        """
        self.ruta_imagen = ruta_imagen
        self.nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
        self.imagen_bgr = cv2.imread(ruta_imagen)
        if self.imagen_bgr is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
        
        self.imagen_rgb = cv2.cvtColor(self.imagen_bgr, cv2.COLOR_BGR2RGB)
        self.imagen_hsv = cv2.cvtColor(self.imagen_bgr, cv2.COLOR_BGR2HSV)
        self.altura, self.ancho = self.imagen_bgr.shape[:2]
        
        print(f"\nImagen cargada: {os.path.basename(ruta_imagen)}")
        print(f"Dimensiones: {self.ancho}x{self.altura} píxeles")
    
    def identificar_colores_dominantes(self, n_colores=6, mostrar=True):
        """
        Identifica los colores dominantes en la imagen usando K-means.
        
        Parámetros:
        -----------
        n_colores : int
            Número de colores a identificar
        mostrar : bool
            Si se debe mostrar la paleta de colores
            
        Retorna:
        --------
        colores_bgr : numpy.ndarray
            Array con los colores dominantes en formato BGR
        """
        # Reducir tamaño para eficiencia
        factor_reduccion = max(1, int(np.sqrt(self.altura * self.ancho / 10000)))
        imagen_reducida = self.imagen_bgr[::factor_reduccion, ::factor_reduccion]
        
        # Reshape para K-means
        pixeles = imagen_reducida.reshape(-1, 3)
        
        # Aplicar K-means
        print(f"Aplicando K-means con {n_colores} clusters...")
        kmeans = KMeans(n_clusters=n_colores, random_state=42, n_init=10)
        etiquetas = kmeans.fit_predict(pixeles)
        
        # Obtener colores centroides
        colores_bgr = kmeans.cluster_centers_.astype(int)
        
        # Ordenar por frecuencia
        contador_etiquetas = Counter(etiquetas)
        indices_ordenados = [i for i, _ in contador_etiquetas.most_common()]
        colores_bgr = colores_bgr[indices_ordenados]
        
        if mostrar:
            self._mostrar_paleta_colores(colores_bgr)
        
        return colores_bgr
    
    def _mostrar_paleta_colores(self, colores_bgr):
        """Muestra la paleta de colores identificados."""
        n_colores = len(colores_bgr)
        fig, axes = plt.subplots(1, n_colores, figsize=(2*n_colores, 2))
        
        if n_colores == 1:
            axes = [axes]
        
        for i, color_bgr in enumerate(colores_bgr):
            color_rgb = color_bgr[::-1]
            axes[i].imshow([[color_rgb/255.0]])
            axes[i].set_title(f"Color {i+1}", fontsize=10)
            axes[i].axis('off')
            
            # Mostrar valores RGB
            axes[i].text(0.5, -0.1, f"RGB({color_rgb[0]},{color_rgb[1]},{color_rgb[2]})",
                        transform=axes[i].transAxes, ha='center', fontsize=8)
        
        plt.suptitle(f"Colores Dominantes - {self.nombre_imagen}")
        plt.tight_layout()
        
        # Guardar la paleta
        carpeta_resultado = os.path.join("resultados", self.nombre_imagen)
        os.makedirs(carpeta_resultado, exist_ok=True)
        plt.savefig(os.path.join(carpeta_resultado, "paleta_colores.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def calcular_rangos_color(self, color_bgr, metodo='hsv', tolerancia=None):
        """
        Calcula los rangos de valores para segmentar un color específico.
        
        Parámetros:
        -----------
        color_bgr : array-like
            Color en formato BGR
        metodo : str
            Método de segmentación ('hsv', 'rgb', 'lab')
        tolerancia : dict
            Tolerancias para cada canal. Si es None, usa valores por defecto
            
        Retorna:
        --------
        rangos : list
            Lista de tuplas (min, max) para cada rango
        """
        if tolerancia is None:
            if metodo == 'hsv':
                tolerancia = {'h': 10, 's': 50, 'v': 50}
            elif metodo == 'rgb':
                tolerancia = {'r': 30, 'g': 30, 'b': 30}
            elif metodo == 'lab':
                tolerancia = {'l': 30, 'a': 20, 'b': 20}
        
        if metodo == 'hsv':
            return self._rangos_hsv(color_bgr, tolerancia)
        elif metodo == 'rgb':
            return self._rangos_rgb(color_bgr, tolerancia)
        elif metodo == 'lab':
            return self._rangos_lab(color_bgr, tolerancia)
        else:
            raise ValueError(f"Método no soportado: {metodo}")
    
    def _rangos_hsv(self, color_bgr, tolerancia):
        """Calcula rangos en espacio HSV."""
        # Convertir color a HSV
        color_hsv = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = color_hsv
        
        tol_h = tolerancia['h']
        tol_s = tolerancia['s']
        tol_v = tolerancia['v']
        
        # Manejar caso especial del rojo (H cerca de 0 o 180)
        if h < tol_h or h > 180 - tol_h:
            # Dos rangos para el rojo
            rango1 = (np.array([0, max(0, s-tol_s), max(0, v-tol_v)]),
                     np.array([tol_h, min(255, s+tol_s), min(255, v+tol_v)]))
            rango2 = (np.array([180-tol_h, max(0, s-tol_s), max(0, v-tol_v)]),
                     np.array([180, min(255, s+tol_s), min(255, v+tol_v)]))
            return [rango1, rango2]
        else:
            # Un solo rango para otros colores
            return [(np.array([max(0, h-tol_h), max(0, s-tol_s), max(0, v-tol_v)]),
                    np.array([min(180, h+tol_h), min(255, s+tol_s), min(255, v+tol_v)]))]
    
    def _rangos_rgb(self, color_bgr, tolerancia):
        """Calcula rangos en espacio RGB."""
        b, g, r = color_bgr
        tol_r = tolerancia['r']
        tol_g = tolerancia['g']
        tol_b = tolerancia['b']
        
        return [(np.array([max(0, b-tol_b), max(0, g-tol_g), max(0, r-tol_r)]),
                np.array([min(255, b+tol_b), min(255, g+tol_g), min(255, r+tol_r)]))]
    
    def _rangos_lab(self, color_bgr, tolerancia):
        """Calcula rangos en espacio LAB."""
        # Convertir a LAB
        color_lab = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
        l, a, b = color_lab
        
        tol_l = tolerancia['l']
        tol_a = tolerancia['a']
        tol_b = tolerancia['b']
        
        return [(np.array([max(0, l-tol_l), max(0, a-tol_a), max(0, b-tol_b)]),
                np.array([min(255, l+tol_l), min(255, a+tol_a), min(255, b+tol_b)]))]
    
    def segmentar_color(self, color_bgr, metodo='hsv', tolerancia=None, 
                       limpiar=True, area_minima=100):
        """
        Segmenta objetos de un color específico.
        
        Parámetros:
        -----------
        color_bgr : array-like
            Color objetivo en BGR
        metodo : str
            Espacio de color a usar
        tolerancia : dict
            Tolerancias personalizadas
        limpiar : bool
            Si aplicar operaciones morfológicas
        area_minima : int
            Área mínima de objetos a considerar (en píxeles)
            
        Retorna:
        --------
        mascara : numpy.ndarray
            Máscara binaria
        contornos : list
            Lista de contornos encontrados
        estadisticas : dict
            Estadísticas de la segmentación
        """
        # Calcular rangos
        rangos = self.calcular_rangos_color(color_bgr, metodo, tolerancia)
        
        # Seleccionar imagen en el espacio correcto
        if metodo == 'hsv':
            imagen_espacio = self.imagen_hsv
        elif metodo == 'rgb':
            imagen_espacio = self.imagen_bgr
        elif metodo == 'lab':
            imagen_espacio = cv2.cvtColor(self.imagen_bgr, cv2.COLOR_BGR2LAB)
        
        # Crear máscara combinando todos los rangos
        mascara = np.zeros((self.altura, self.ancho), dtype=np.uint8)
        for rango_min, rango_max in rangos:
            mascara_parcial = cv2.inRange(imagen_espacio, rango_min, rango_max)
            mascara = cv2.bitwise_or(mascara, mascara_parcial)
        
        # Limpiar máscara si se solicita
        if limpiar:
            mascara = self._limpiar_mascara(mascara, area_minima)
        
        # Encontrar contornos
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Calcular estadísticas
        areas = [cv2.contourArea(c) for c in contornos]
        estadisticas = {
            'num_objetos': len(contornos),
            'area_total': np.sum(mascara > 0),
            'porcentaje_imagen': 100 * np.sum(mascara > 0) / (self.altura * self.ancho),
            'areas_objetos': areas,
            'area_promedio': np.mean(areas) if areas else 0,
            'area_maxima': np.max(areas) if areas else 0,
            'area_minima': np.min(areas) if areas else 0
        }
        
        return mascara, contornos, estadisticas
    
    def _limpiar_mascara(self, mascara, area_minima):
        """Aplica operaciones morfológicas para limpiar la máscara."""
        # Operaciones morfológicas
        kernel_pequeno = np.ones((3,3), np.uint8)
        kernel_grande = np.ones((5,5), np.uint8)
        
        # Eliminar ruido pequeño
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel_pequeno)
        
        # Cerrar huecos
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_grande)
        
        # Filtrar por área mínima
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mascara)
        mascara_filtrada = np.zeros_like(mascara)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= area_minima:
                mascara_filtrada[labels == i] = 255
        
        return mascara_filtrada
    
    def extraer_multiples_colores(self, colores_bgr, num_colores=4, 
                                 metodo='hsv', mostrar=True):
        """
        Extrae objetos de múltiples colores.
        
        Parámetros:
        -----------
        colores_bgr : numpy.ndarray
            Array de colores en BGR
        num_colores : int
            Número de colores a extraer
        metodo : str
            Método de segmentación
        mostrar : bool
            Si mostrar resultados
            
        Retorna:
        --------
        resultados : dict
            Diccionario con resultados por color
        """
        resultados = {}
        num_colores = min(num_colores, len(colores_bgr))
        
        print(f"\nExtrayendo objetos de {num_colores} colores...")
        
        for i in range(num_colores):
            print(f"\nProcesando color {i+1}/{num_colores}...")
            color_bgr = colores_bgr[i]
            
            # Probar con diferentes tolerancias si es necesario
            mejor_resultado = None
            mejor_num_objetos = 0
            
            for factor_tolerancia in [1.0, 1.5, 0.7]:
                if metodo == 'hsv':
                    tolerancia = {
                        'h': int(10 * factor_tolerancia),
                        's': int(50 * factor_tolerancia),
                        'v': int(50 * factor_tolerancia)
                    }
                else:
                    tolerancia = None
                
                mascara, contornos, stats = self.segmentar_color(
                    color_bgr, metodo, tolerancia
                )
                
                if stats['num_objetos'] > mejor_num_objetos:
                    mejor_resultado = (mascara, contornos, stats)
                    mejor_num_objetos = stats['num_objetos']
            
            mascara, contornos, stats = mejor_resultado
            
            resultados[f'color_{i+1}'] = {
                'color_bgr': color_bgr,
                'color_rgb': color_bgr[::-1],
                'mascara': mascara,
                'contornos': contornos,
                'estadisticas': stats
            }
            
            print(f"  - Objetos encontrados: {stats['num_objetos']}")
            print(f"  - Área total: {stats['area_total']} píxeles")
            print(f"  - Porcentaje de imagen: {stats['porcentaje_imagen']:.2f}%")
        
        if mostrar:
            self.visualizar_resultados(resultados)
        
        return resultados
    
    def visualizar_resultados(self, resultados):
        """Visualiza los resultados de la segmentación."""
        n_colores = len(resultados)
        fig, axes = plt.subplots(n_colores, 4, figsize=(16, 4*n_colores))
        
        if n_colores == 1:
            axes = axes.reshape(1, -1)
        
        for i, (nombre, data) in enumerate(resultados.items()):
            # Imagen original
            axes[i, 0].imshow(self.imagen_rgb)
            axes[i, 0].set_title(f"{nombre} - Original")
            
            # Máscara
            axes[i, 1].imshow(data['mascara'], cmap='gray')
            axes[i, 1].set_title(f"Máscara ({data['estadisticas']['num_objetos']} objetos)")
            
            # Objetos extraídos
            resultado = cv2.bitwise_and(self.imagen_bgr, self.imagen_bgr, 
                                      mask=data['mascara'])
            axes[i, 2].imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
            axes[i, 2].set_title("Objetos Extraídos")
            
            # Contornos
            imagen_contornos = self.imagen_rgb.copy()
            cv2.drawContours(imagen_contornos, data['contornos'], -1, (255, 0, 0), 2)
            axes[i, 3].imshow(imagen_contornos)
            axes[i, 3].set_title("Contornos Detectados")
            
            # Mostrar color
            color_rgb = data['color_rgb']
            fig.text(0.02, 1 - (i + 0.5) / n_colores, 
                    f"RGB({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})",
                    ha='left', va='center', fontsize=10, 
                    transform=fig.transFigure)
            
            for ax in axes[i]:
                ax.axis('off')
        
        plt.tight_layout()
        
        # Guardar visualización
        carpeta_resultado = os.path.join("resultados", self.nombre_imagen)
        plt.savefig(os.path.join(carpeta_resultado, "resultados_visualizacion.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def guardar_mascaras(self, resultados):
        """Guarda las máscaras individuales."""
        carpeta_resultado = os.path.join("resultados", self.nombre_imagen)
        os.makedirs(carpeta_resultado, exist_ok=True)
        
        for nombre, data in resultados.items():
            cv2.imwrite(os.path.join(carpeta_resultado, f"mascara_{nombre}.png"), 
                       data['mascara'])
    
    def generar_reporte(self, resultados, archivo_salida=None):
        """
        Genera un reporte con los resultados de la extracción.
        
        Parámetros:
        -----------
        resultados : dict
            Resultados de la extracción
        archivo_salida : str
            Nombre del archivo de reporte
        """
        if archivo_salida is None:
            carpeta_resultado = os.path.join("resultados", self.nombre_imagen)
            os.makedirs(carpeta_resultado, exist_ok=True)
            archivo_salida = os.path.join(carpeta_resultado, "reporte.txt")
        
        with open(archivo_salida, 'w', encoding='utf-8') as f:
            f.write("REPORTE - TALLER 1: EXTRACCIÓN DE OBJETOS POR COLOR\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Imagen: {self.nombre_imagen}\n")
            f.write(f"Dimensiones: {self.ancho}x{self.altura} píxeles\n")
            f.write(f"Colores procesados: {len(resultados)}\n\n")
            
            total_objetos = 0
            
            for nombre, data in resultados.items():
                stats = data['estadisticas']
                color_rgb = data['color_rgb']
                
                f.write(f"{nombre.upper()}\n")
                f.write("-" * 20 + "\n")
                f.write(f"Color RGB: ({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})\n")
                f.write(f"Objetos detectados: {stats['num_objetos']}\n")
                f.write(f"Área total: {stats['area_total']} píxeles\n")
                f.write(f"Porcentaje de imagen: {stats['porcentaje_imagen']:.2f}%\n")
                
                if stats['num_objetos'] > 0:
                    f.write(f"Área promedio por objeto: {stats['area_promedio']:.2f} píxeles\n")
                    f.write(f"Área máxima: {stats['area_maxima']:.2f} píxeles\n")
                    f.write(f"Área mínima: {stats['area_minima']:.2f} píxeles\n")
                
                f.write("\n")
                total_objetos += stats['num_objetos']
            
            f.write(f"TOTAL DE OBJETOS DETECTADOS: {total_objetos}\n")


def analizar_problemas_segmentacion(extractor, color_bgr, nombre_color=""):
    """
    Analiza y visualiza problemas comunes en la segmentación por color.
    
    Parámetros:
    -----------
    extractor : ExtractorObjetosColor
        Instancia del extractor
    color_bgr : array-like
        Color a analizar
    nombre_color : str
        Nombre descriptivo del color
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Análisis de Problemas - {extractor.nombre_imagen} - {nombre_color}", fontsize=16)
    
    # 1. Segmentación básica
    mascara_basica, _, stats_basica = extractor.segmentar_color(
        color_bgr, metodo='hsv', limpiar=False
    )
    axes[0, 0].imshow(mascara_basica, cmap='gray')
    axes[0, 0].set_title(f"Segmentación Básica\n({stats_basica['num_objetos']} objetos)")
    
    # 2. Con limpieza morfológica
    mascara_limpia, _, stats_limpia = extractor.segmentar_color(
        color_bgr, metodo='hsv', limpiar=True
    )
    axes[0, 1].imshow(mascara_limpia, cmap='gray')
    axes[0, 1].set_title(f"Con Limpieza Morfológica\n({stats_limpia['num_objetos']} objetos)")
    
    # 3. Diferencia (ruido eliminado)
    diferencia = mascara_basica - mascara_limpia
    axes[0, 2].imshow(diferencia, cmap='hot')
    axes[0, 2].set_title("Ruido Eliminado")
    
    # 4. Histograma de áreas
    _, contornos, _ = extractor.segmentar_color(color_bgr, metodo='hsv')
    areas = [cv2.contourArea(c) for c in contornos]
    if areas:
        axes[1, 0].hist(areas, bins=20, edgecolor='black')
        axes[1, 0].set_xlabel('Área (píxeles)')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].set_title('Distribución de Áreas')
    
    # 5. Efectos de iluminación
    imagen_lab = cv2.cvtColor(extractor.imagen_bgr, cv2.COLOR_BGR2LAB)
    l_channel = imagen_lab[:, :, 0]
    mascara_aplicada = cv2.bitwise_and(l_channel, l_channel, mask=mascara_limpia)
    axes[1, 1].imshow(mascara_aplicada, cmap='gray')
    axes[1, 1].set_title('Variación de Iluminación\nen Objetos Detectados')
    
    # 6. Comparación RGB vs HSV
    mascara_rgb, _, stats_rgb = extractor.segmentar_color(
        color_bgr, metodo='rgb', limpiar=True
    )
    comparacion = np.zeros((extractor.altura, extractor.ancho, 3), dtype=np.uint8)
    comparacion[:, :, 0] = mascara_limpia  # HSV en rojo
    comparacion[:, :, 1] = mascara_rgb      # RGB en verde
    comparacion[:, :, 2] = mascara_limpia & mascara_rgb  # Intersección en azul
    axes[1, 2].imshow(comparacion)
    axes[1, 2].set_title(f'HSV (rojo) vs RGB (verde)\nHSV: {stats_limpia["num_objetos"]}, RGB: {stats_rgb["num_objetos"]}')
    
    for ax in axes.flat:
        if ax.get_xlabel() == '':
            ax.axis('off')
    
    plt.tight_layout()
    
    # Guardar análisis
    carpeta_resultado = os.path.join("resultados", extractor.nombre_imagen)
    plt.savefig(os.path.join(carpeta_resultado, f"analisis_problemas_{nombre_color}.png"), 
               dpi=150, bbox_inches='tight')
    plt.close()


def procesar_imagen(ruta_imagen, num_colores_extraer=4):
    """Procesa una imagen completa."""
    try:
        print("\n" + "="*70)
        print(f"PROCESANDO: {os.path.basename(ruta_imagen)}")
        print("="*70)
        
        # Crear extractor
        extractor = ExtractorObjetosColor(ruta_imagen)
        
        # Crear carpeta de resultados para esta imagen
        carpeta_resultado = os.path.join("resultados", extractor.nombre_imagen)
        os.makedirs(carpeta_resultado, exist_ok=True)
        
        # Paso 1: Identificar colores dominantes
        print("\nPaso 1: Identificando colores dominantes...")
        colores = extractor.identificar_colores_dominantes(n_colores=6)
        
        # Paso 2: Extraer objetos de colores
        print(f"\nPaso 2: Extrayendo objetos de {num_colores_extraer} colores...")
        resultados = extractor.extraer_multiples_colores(colores, num_colores=num_colores_extraer)
        
        # Paso 3: Análisis de problemas (solo para el primer color)
        print("\nPaso 3: Analizando problemas de segmentación...")
        analizar_problemas_segmentacion(extractor, colores[0], "Color_1")
        
        # Paso 4: Guardar resultados
        print("\nPaso 4: Guardando resultados...")
        extractor.guardar_mascaras(resultados)
        extractor.generar_reporte(resultados)
        
        print(f"\n✅ Procesamiento completado para {extractor.nombre_imagen}")
        print(f"   Resultados guardados en: resultados/{extractor.nombre_imagen}/")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error procesando {ruta_imagen}: {e}")
        import traceback
        traceback.print_exc()
        return False


def generar_reporte_general(imagenes_procesadas):
    """Genera un reporte general de todas las imágenes procesadas."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo_reporte = os.path.join("resultados", f"reporte_general_{timestamp}.txt")
    
    with open(archivo_reporte, 'w', encoding='utf-8') as f:
        f.write("REPORTE GENERAL - TALLER 1: EXTRACCIÓN DE OBJETOS POR COLOR\n")
        f.write("="*70 + "\n\n")
        f.write(f"Fecha de procesamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de imágenes procesadas: {len(imagenes_procesadas)}\n\n")
        
        for i, imagen in enumerate(imagenes_procesadas, 1):
            f.write(f"{i}. {imagen}\n")
            carpeta = os.path.join("resultados", os.path.splitext(os.path.basename(imagen))[0])
            if os.path.exists(carpeta):
                archivos = os.listdir(carpeta)
                f.write(f"   Archivos generados: {len(archivos)}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Procesamiento completado exitosamente\n")
    
    print(f"\n📄 Reporte general guardado en: {archivo_reporte}")


# Función principal para ejecutar el taller
def main():
    """Función principal del programa."""
    print("TALLER 1 - EXTRACCIÓN DE OBJETOS POR COLOR")
    print("="*70)
    print("Autor: Abel Albuez Sanchez")
    print("Fecha: 11/09/2025")
    print("="*70)
    
    # Verificar que existe la carpeta de imágenes
    if not os.path.exists("imagenes"):
        print("\n❌ Error: No se encuentra la carpeta 'imagenes/'")
        print("   Ejecute primero el script setup.py para generar las imágenes de ejemplo")
        return
    
    # Obtener lista de imágenes a procesar
    imagenes = []
    extensiones = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for archivo in os.listdir("imagenes"):
        if any(archivo.lower().endswith(ext) for ext in extensiones):
            imagenes.append(os.path.join("imagenes", archivo))
    
    if not imagenes:
        print("\n❌ No se encontraron imágenes en la carpeta 'imagenes/'")
        return
    
    print(f"\n📁 Se encontraron {len(imagenes)} imágenes para procesar:")
    for img in imagenes:
        print(f"   • {os.path.basename(img)}")
    
    # Crear carpeta principal de resultados
    os.makedirs("resultados", exist_ok=True)
    
    # Procesar cada imagen
    imagenes_procesadas = []
    print("\n🔄 Iniciando procesamiento de imágenes...")
    
    for imagen in imagenes:
        if procesar_imagen(imagen):
            imagenes_procesadas.append(imagen)
    
    # Generar reporte general
    if imagenes_procesadas:
        generar_reporte_general(imagenes_procesadas)
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    print(f"✅ Imágenes procesadas exitosamente: {len(imagenes_procesadas)}/{len(imagenes)}")
    print("\n📂 Estructura de resultados generada:")
    print("   resultados/")
    
    for imagen in imagenes_procesadas:
        nombre = os.path.splitext(os.path.basename(imagen))[0]
        print(f"   ├── {nombre}/")
        print(f"   │   ├── paleta_colores.png")
        print(f"   │   ├── resultados_visualizacion.png")
        print(f"   │   ├── analisis_problemas_Color_1.png")
        print(f"   │   ├── mascara_color_1.png")
        print(f"   │   ├── mascara_color_2.png")
        print(f"   │   ├── mascara_color_3.png")
        print(f"   │   ├── mascara_color_4.png")
        print(f"   │   └── reporte.txt")
    
    print(f"   └── reporte_general_*.txt")
    
    print("\n✨ ¡Procesamiento completado exitosamente!")


if __name__ == "__main__":
    main()