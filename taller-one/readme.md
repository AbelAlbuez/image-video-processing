# Taller 1 - Extracción de Objetos por Color

## 📋 Descripción

Este proyecto implementa un sistema de procesamiento de imágenes para extraer y segmentar objetos basándose en su color. El objetivo es identificar los colores dominantes en una imagen y extraer todos los objetos que pertenezcan a cada color específico.

### Características principales:
- Identificación automática de colores dominantes usando K-means
- Segmentación por color en múltiples espacios (HSV, RGB, LAB)
- Operaciones morfológicas para limpieza de máscaras
- Análisis estadístico de objetos detectados
- Generación automática de reportes
- Visualización detallada de resultados

## 🗂️ Estructura del Proyecto

```
taller-one/
├── imagenes/                    # Imágenes de ejemplo
│   ├── ejemplo_circulos.jpg    # Círculos de diferentes colores
│   ├── ejemplo_formas.jpg      # Formas geométricas variadas
│   ├── ejemplo_realista.jpg    # Escena con iluminación compleja
│   └── ejemplo_ruido.jpg       # Objetos con ruido
├── informes/                   # Carpeta para reportes generados
├── resultados/                 # Máscaras y resultados de segmentación
├── venv_taller1/              # Entorno virtual Python
├── .gitignore                 # Archivos ignorados por Git
├── imagen_taller1.jpg         # Imagen principal de prueba
├── requirements.txt           # Dependencias del proyecto
├── setup.py                   # Script de configuración automática
├── taller1_extraccion.py      # Código principal
└── README.md                  # Este archivo
```

## 🚀 Instalación

### Prerrequisitos
- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Pasos de instalación

1. **Clonar o descargar el proyecto**
   ```bash
   git clone https://github.com/AbelAlbuez/image-video-processing.git
   cd image-video-processing/taller-one
   ```

2. **Crear entorno virtual**
   ```bash
   # En macOS/Linux
   python3 -m venv venv_taller1
   source venv_taller1/bin/activate
   
   # En Windows
   python -m venv venv_taller1
   venv_taller1\Scripts\activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

### Instalación automática (alternativa)
```bash
python setup.py
```
Este script automático realiza las siguientes acciones:
1. Crea el entorno virtual `venv_taller1`
2. Instala todas las dependencias desde `requirements.txt`
3. Crea la estructura de directorios (`imagenes/`, `informes/`, `resultados/`)
4. Genera archivo `.gitignore`
5. **Crea 2 imágenes de ejemplo automáticamente:**
   - `imagen_taller1.jpg`: Bloques de 6 colores puros
   - `imagenes/ejemplo_circulos.jpg`: Múltiples objetos de colores
6. Verifica que la instalación sea correcta

Una vez ejecutado el setup, puedes proceder directamente a ejecutar el programa principal.

## 📦 Dependencias

- **opencv-python**: Procesamiento de imágenes
- **numpy**: Operaciones con arrays
- **matplotlib**: Visualización
- **scikit-learn**: Algoritmo K-means
- **scipy**: Operaciones científicas
- **Pillow**: Manejo adicional de imágenes

## 🎯 Uso

### Uso básico

1. **Activar el entorno virtual** (si no está activo)
   ```bash
   source venv_taller1/bin/activate  # macOS/Linux
   # o
   venv_taller1\Scripts\activate     # Windows
   ```

2. **Ejecutar el programa principal**
   ```bash
   python taller1_extraccion.py
   ```

### Cambiar la imagen a procesar

Edita la línea en `taller1_extraccion.py`:
```python
RUTA_IMAGEN = "imagen_taller1.jpg"  # Cambiar por tu imagen
```

### Imágenes de ejemplo disponibles

- `imagen_taller1.jpg`: 6 bloques de colores puros
- `imagenes/ejemplo_circulos.jpg`: Múltiples círculos de colores
- `imagenes/ejemplo_formas.jpg`: Formas geométricas variadas
- `imagenes/ejemplo_ruido.jpg`: Objetos con ruido añadido
- `imagenes/ejemplo_realista.jpg`: Escena con iluminación compleja

## 📊 Resultados

El programa genera:

1. **Visualizaciones en pantalla**:
   - Paleta de colores identificados
   - Máscaras de segmentación
   - Objetos extraídos
   - Contornos detectados
   - Análisis de problemas de segmentación

2. **Archivos generados**:
   - `reporte_taller1.txt`: Informe detallado con estadísticas
   - `resultados/mascara_color_X.png`: Máscaras binarias para cada color

## 🔧 Configuración Avanzada

### Parámetros ajustables

En el código puedes modificar:

- **Número de colores a detectar**: Por defecto 6
- **Número de colores a extraer**: Por defecto 4
- **Tolerancias de color**: En espacios HSV, RGB o LAB
- **Área mínima de objetos**: Por defecto 100 píxeles
- **Método de segmentación**: HSV (recomendado), RGB o LAB

### Ejemplo de modificación:
```python
# En la función main()
colores = extractor.identificar_colores_dominantes(n_colores=8)  # Detectar 8 colores
resultados = extractor.extraer_multiples_colores(colores, num_colores=5)  # Extraer 5
```

## 🐛 Solución de Problemas

### Error: "No module named 'cv2'"
```bash
pip uninstall opencv-python
pip install opencv-python==4.8.1.78
```

### Error: "No se pudo cargar la imagen"
- Verifica que el nombre del archivo sea correcto
- Asegúrate de que la imagen esté en la ruta especificada
- Comprueba que la extensión sea .jpg, .png, etc.

### Las ventanas no se muestran (Linux)
```bash
sudo apt-get install python3-tk
```

### Problemas con macOS M1/M2
```bash
pip install numpy --upgrade --force-reinstall
```

## 📈 Explicación del Algoritmo

### 1. Procesamiento por lotes (NUEVO)
- Detecta automáticamente todas las imágenes en `imagenes/`
- Procesa cada imagen de forma independiente
- Organiza resultados en carpetas separadas
- Genera reportes individuales y consolidados

### 2. Identificación de colores (K-means)
- Reduce el tamaño de la imagen para eficiencia
- Aplica K-means para encontrar los 6 colores dominantes
- Ordena por frecuencia de aparición

### 3. Segmentación por color
- Convierte al espacio de color seleccionado (HSV recomendado)
- Define rangos de tolerancia para cada color
- Crea máscaras binarias usando `cv2.inRange()`

### 4. Limpieza morfológica
- Apertura: elimina ruido pequeño
- Cierre: rellena huecos en objetos
- Filtrado por área mínima

### 5. Extracción de objetos
- Encuentra contornos usando `cv2.findContours()`
- Calcula estadísticas (área, número de objetos)
- Genera visualizaciones y reportes

### 6. Guardado automático
- Todas las visualizaciones se guardan como PNG (150 DPI)
- Sin ventanas emergentes para mejor automatización
- Nombres descriptivos para fácil identificación

## 📝 Estructura del Reporte

El reporte generado incluye:
- Dimensiones de la imagen
- Colores procesados (valores RGB)
- Por cada color:
  - Número de objetos detectados
  - Área total y porcentaje de la imagen
  - Estadísticas de áreas (promedio, máxima, mínima)
- Total de objetos detectados

## 🎓 Conceptos Teóricos

### ¿Por qué HSV es mejor que RGB?
- **H (Hue)**: Representa el color puro, independiente de iluminación
- **S (Saturation)**: Pureza del color
- **V (Value)**: Brillo/iluminación
- Permite segmentar por color ignorando variaciones de luz

### Problemas comunes en segmentación:
1. **Variaciones de iluminación**: Sombras y reflejos
2. **Ruido**: Píxeles aislados con valores atípicos
3. **Bordes difusos**: Antialiasing y mezcla de colores
4. **Objetos superpuestos**: Se detectan como uno solo

## 👥 Autores

- Abel Albuez Sanchez
- Curso: Procesamiento de Imágenes y Video
- Profesor: Carlos Alberto Parra Rodríguez
- Pontificia Universidad Javeriana

## 📄 Licencia

Este proyecto es de uso académico para el curso de Procesamiento de Imágenes y Video.

## 🙏 Agradecimientos

- Al profesor Carlos Alberto Parra por el material del curso
- OpenCV por la librería de procesamiento de imágenes
- Scikit-learn por la implementación de K-means

---

**Nota**: Para cualquier duda o problema, revisar la documentación del código o contactar a los autores.