# 🩺 Detección Temprana de Melanomas mediante Segmentación Adaptativa

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Universidad:** Pontificia Universidad Javeriana  
**Curso:** Procesamiento de Imágenes y Video - 2025  
**Autores:** Abel Albuez Sanchez, Daniel Felipe Rios Caro

---

## 📋 Descripción

Sistema de procesamiento digital de imágenes para la segmentación automática de lesiones cutáneas y extracción de descriptores clínicos ABCD (Asimetría, Borde, Color, Diámetro). El objetivo es apoyar la detección temprana de melanomas mediante técnicas clásicas de visión por computador.

### 🎯 Objetivos

- ✅ Segmentación adaptativa de lesiones cutáneas usando el método de Otsu
- ✅ Extracción completa de descriptores ABCD
- ✅ Evaluación cuantitativa con métricas IoU y Dice
- ✅ Análisis automatizado de múltiples imágenes dermatoscópicas

---

## 🚀 Características

### Descriptores ABCD Implementados

| Descriptor | Métrica | Descripción |
|-----------|---------|-------------|
| **A - Asimetría** | Score 0-1 | Análisis por ejes principales (horizontal/vertical) |
| **B - Borde** | Circularidad | Irregularidad del contorno (4πA/P²) |
| **C - Color** | Std Lab | Variación cromática en espacio Lab |
| **D - Diámetro** | px y mm | Estimación del tamaño de la lesión |

### Métricas de Evaluación

- **IoU (Intersection over Union)**: Mide el solapamiento con ground truth
- **Coeficiente de Dice**: Métrica F1 para segmentación médica

---

## 📦 Instalación y Configuración

### Requisitos Previos

- **Python 3.8 o superior** - [Descargar Python](https://www.python.org/downloads/)
- **pip** (gestor de paquetes de Python - viene con Python)
- **git** (opcional, para clonar el repositorio)

---

## 🚀 Guía de Instalación Paso a Paso

### **PASO 1️⃣: Clonar o Descargar el Proyecto**

**Opción A - Con Git:**
```bash
git clone https://github.com/tu-usuario/melanoma-segmentation.git
cd melanoma-segmentation
```

**Opción B - Sin Git:**
1. Descarga el proyecto como ZIP
2. Extrae el contenido
3. Abre la terminal en la carpeta del proyecto

---

### **PASO 2️⃣: Crear el Entorno Virtual (venv)**

Un entorno virtual aísla las dependencias del proyecto para evitar conflictos.

#### **En Windows (CMD o PowerShell):**
```bash
# Crear el entorno virtual
python -m venv venv

# Activar el entorno virtual (CMD)
venv\Scripts\activate

# Activar el entorno virtual (PowerShell)
venv\Scripts\Activate.ps1
```

**⚠️ Si tienes error de permisos en PowerShell:**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1
```

#### **En Linux/macOS:**
```bash
# Crear el entorno virtual
python3 -m venv venv

# Activar el entorno virtual
source venv/bin/activate
```

**✅ Verificar que el entorno esté activo:**
- Deberías ver `(venv)` al inicio de tu línea de comandos
- Ejemplo: `(venv) C:\Users\TuUsuario\melanoma-segmentation>`

---

### **PASO 3️⃣: Instalar las Dependencias**

Con el entorno virtual **activado**, ejecuta:

```bash
# Actualizar pip a la última versión
python -m pip install --upgrade pip

# Instalar TODAS las dependencias del proyecto
pip install -r requirements.txt
```

**⏱️ Esto puede tomar 2-3 minutos.** Se instalarán:
- ✅ `opencv-python` - Procesamiento de imágenes
- ✅ `numpy` - Computación numérica
- ✅ `pandas` - Análisis de datos
- ✅ `matplotlib` - Visualización
- ✅ `seaborn` - Gráficos estadísticos
- ✅ `scikit-image` - Procesamiento científico de imágenes
- ✅ `scipy` - Algoritmos científicos
- ✅ `tqdm` - Barras de progreso
- ✅ `jinja2` - Generación de tablas LaTeX
- ✅ `Pillow` - Manejo de imágenes

---

### **PASO 4️⃣: Verificar la Instalación**

```bash
python verify_setup.py
```

**Deberías ver:**
```
✅ Python 3.9.x detectado
✅ Entorno virtual activo
✅ opencv-python instalado
✅ numpy instalado
✅ pandas instalado
✅ jinja2 instalado
...
🎉 ¡Todo está configurado correctamente!
```

**⚠️ Si hay errores:**
- Verifica que el entorno virtual esté activo (`(venv)` visible)
- Reinstala las dependencias: `pip install -r requirements.txt`

---

### **PASO 5️⃣: Preparar el Dataset**

Crea la siguiente estructura de carpetas en tu proyecto:

```
melanoma-segmentation/
├── dataset/
│   ├── images/          # ← Coloca aquí tus imágenes .jpg o .png
│   └── masks/           # ← Opcional: máscaras ground truth (para evaluar con IoU/Dice)
├── src/
├── venv/
└── ...
```

**Crear las carpetas:**
```bash
# Windows
mkdir dataset\images dataset\masks

# Linux/macOS
mkdir -p dataset/images dataset/masks
```

**Agregar imágenes:**
- Copia tus imágenes dermatoscópicas a `dataset/images/`
- Si tienes máscaras ground truth, cópialas a `dataset/masks/`
- Los nombres deben coincidir: `imagen.jpg` ↔ `imagen.png` (máscara)

---

### **PASO 6️⃣: Ejecutar el Proyecto**

#### **Opción A: Demo Interactiva (Recomendada) 🎮**

```bash
python src/demo_avance2.py
```

**Menú con 3 opciones:**
1. 📷 **Analizar UNA imagen individual** - Muestra todos los descriptores ABCD
2. 📊 **Procesamiento BATCH** - Procesa múltiples imágenes y genera reportes
3. 🧪 **Crear dataset sintético** - Genera imágenes de prueba

#### **Opción B: Procesar Imagen Específica 🔬**

```bash
python
```

```python
from src.melanoma_descriptors import MelanomaDescriptors

# Analizar una imagen
analyzer = MelanomaDescriptors('dataset/images/mi_imagen.jpg')
descriptors = analyzer.calculate_all_descriptors()
analyzer.visualize_results(save_path='resultado.png')

# Ver descriptores
print(f"Asimetría: {descriptors['A_asymmetry']['asymmetry_score']:.4f}")
print(f"Circularidad: {descriptors['B_border']['circularity']:.4f}")
print(f"Variación de Color: {descriptors['C_color']['color_std_lab']:.4f}")
print(f"Diámetro: {descriptors['D_diameter']['diameter_mm']:.2f} mm")
```

#### **Opción C: Procesamiento Masivo (Batch) 📊**

```bash
python src/batch_evaluation.py
```

**Genera automáticamente en la carpeta `results/`:**
- 📄 `resultados_completos.csv` - Todos los descriptores
- 📊 `distribucion_descriptores.png` - Histogramas de A, B, C, D
- 🔥 `matriz_correlacion.png` - Heatmap de correlaciones
- 📋 `tabla_resultados.tex` - Tabla LaTeX para tu informe
- 📈 `reporte_evaluacion.json` - Métricas de rendimiento
- 📉 `estadisticas_descriptivas.csv` - Estadísticas del dataset

---

### **PASO 7️⃣: Desactivar el Entorno Virtual** (cuando termines)

```bash
deactivate
```

El `(venv)` desaparecerá de tu terminal.

---

## ⚡ Resumen Rápido (Cheat Sheet)

```bash
# 1️⃣ Navegar al proyecto
cd melanoma-segmentation

# 2️⃣ Crear entorno virtual
python -m venv venv                 # Windows/Linux/macOS

# 3️⃣ Activar entorno virtual
venv\Scripts\activate               # Windows
source venv/bin/activate            # Linux/macOS

# 4️⃣ Instalar dependencias
pip install -r requirements.txt

# 5️⃣ Verificar instalación
python verify_setup.py

# 6️⃣ Ejecutar demo
python src/demo_avance2.py

# 7️⃣ Desactivar al terminar
deactivate
```

---

## 📋 Lista Completa de Dependencias

```
opencv-python>=4.8.0          # Procesamiento de imágenes
opencv-contrib-python>=4.8.0  # Algoritmos adicionales de OpenCV
numpy>=1.24.0                 # Computación numérica
scipy>=1.11.0                 # Algoritmos científicos
pandas>=2.0.0                 # Análisis de datos
matplotlib>=3.7.0             # Visualización básica
seaborn>=0.12.0               # Gráficos estadísticos
scikit-image>=0.21.0          # Procesamiento científico de imágenes
Pillow>=10.0.0                # Manejo de formatos de imagen
tqdm>=4.65.0                  # Barras de progreso
jinja2>=3.0.0                 # Generación de tablas LaTeX
```

---

## 📁 Estructura del Proyecto

```
melanoma-segmentation/
├── src/
│   ├── melanoma_descriptors.py    # Clase principal con descriptores ABCD
│   ├── batch_evaluation.py        # Evaluación en batch
│   └── demo_avance2.py            # Script de demostración
├── dataset/
│   ├── images/                    # Imágenes dermatoscópicas
│   └── masks/                     # Ground truth (opcional)
├── results/                       # Resultados generados
├── docs/
│   ├── latex/                     # Informe LaTeX
│   └── Proyecto - Melanoma Segmentacion.docx
├── venv/                          # Entorno virtual (no subir a Git)
├── requirements.txt               # Dependencias
├── verify_setup.py                # Script de verificación
├── .gitignore                     # Archivos a ignorar en Git
├── README.md                      # Este archivo
└── LICENSE                        # Licencia MIT
```

---

## 🎮 Uso Detallado

### ⚙️ Preparación Inicial

Antes de ejecutar cualquier script, asegúrate de:

1. **Activar el entorno virtual:**

   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

2. **Verificar que el entorno esté activo** (deberías ver `(venv)` al inicio del prompt)

### Opción 1: Demostración Interactiva

```bash
# Asegúrate de estar en el directorio raíz del proyecto
python src/demo_avance2.py
```

El script presenta un menú con 3 opciones:
1. Analizar una imagen individual
2. Procesamiento en batch (múltiples imágenes)
3. Crear dataset sintético de ejemplo

### Opción 2: Uso Programático

#### Analizar una imagen individual

```python
from src.melanoma_descriptors import MelanomaDescriptors

# Cargar imagen
analyzer = MelanomaDescriptors("dataset/images/ISIC_0024306.jpg")

# Calcular descriptores
descriptors = analyzer.calculate_all_descriptors()

# Visualizar resultados
analyzer.visualize_results(save_path="resultado.png")

# Evaluar con ground truth (opcional)
import cv2
gt_mask = cv2.imread("dataset/masks/ISIC_0024306_mask.png", 0)
metrics = analyzer.evaluate_segmentation(gt_mask)
print(f"IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}")
```

#### Procesamiento en batch

```python
from src.batch_evaluation import BatchEvaluator

# Configurar evaluador
evaluator = BatchEvaluator(
    images_dir="dataset/images",
    masks_dir="dataset/masks",
    output_dir="results"
)

# Procesar dataset
df_results = evaluator.process_dataset(max_images=50)

# Generar reporte completo
evaluator.generate_full_report()
```

---

## 📊 Resultados

El sistema genera automáticamente:

### Archivos CSV
- `resultados_completos.csv` - Todos los descriptores por imagen
- `estadisticas_descriptivas.csv` - Estadísticas del dataset

### Tablas LaTeX
- `tabla_resultados.tex` - Tabla formateada para informe

### Visualizaciones
- `distribucion_descriptores.png` - Histogramas de A, B, C, D
- `matriz_correlacion.png` - Heatmap de correlaciones

### Reportes
- `reporte_evaluacion.json` - Métricas de rendimiento

### Ejemplo de Resultados

| Imagen | Asimetría | Circularidad | Color Std | Diámetro (mm) | IoU | Dice |
|--------|-----------|--------------|-----------|---------------|-----|------|
| ISIC_0024306 | 0.3245 | 0.5569 | 28.45 | 19.5 | 0.7234 | 0.8401 |
| ISIC_0024307 | 0.4123 | 0.3854 | 35.22 | 16.9 | 0.6891 | 0.8156 |

---

## 🔬 Metodología

### Pipeline de Segmentación

1. **Preprocesamiento**
   - Extracción del canal azul (mayor contraste)
   - Filtrado Gaussiano para reducir ruido

2. **Segmentación Adaptativa**
   - Binarización de Otsu (umbral automático)
   - No requiere parámetros manuales

3. **Refinamiento Morfológico**
   - Apertura: elimina ruido de "sal y pimienta"
   - Cierre: rellena agujeros internos

4. **Extracción de Descriptores**
   - Análisis de asimetría por ejes principales
   - Cálculo de circularidad del contorno
   - Análisis de variación cromática en Lab
   - Estimación del diámetro

### Datasets Utilizados

- **HAM10000** (Kaggle): 10,015 imágenes dermatoscópicas clasificadas
- **ISIC Archive**: Imágenes con máscaras ground truth para evaluación

---

## 📈 Métricas de Rendimiento

**Objetivo del proyecto:** IoU ≥ 0.6 en ≥ 70% de las imágenes

**Resultados obtenidos:** *(Actualizar con tus resultados después de ejecutar)*
- IoU promedio: X.XXXX ± X.XXXX
- Dice promedio: X.XXXX ± X.XXXX
- Porcentaje IoU ≥ 0.6: XX.X%
- ✅ Objetivo cumplido / ⚠️ En progreso

---

## 🎓 Casos de Uso

### Educación Médica
- Herramienta de apoyo para el aprendizaje de dermatología
- Demostración de características clínicas del melanoma

### Investigación
- Baseline para comparar con métodos de deep learning
- Análisis de características clínicas en grandes datasets

### Telemedicina
- Sistema de triaje inicial para consultas remotas
- Apoyo a la decisión clínica

---

## ⚠️ Limitaciones

- No reemplaza el diagnóstico médico profesional
- Rendimiento variable según calidad de imagen
- Sensible a artefactos (vello, burbujas de aire)
- Requiere imágenes dermatoscópicas de buena calidad

---

## 🔧 Troubleshooting (Solución de Problemas)

### Problemas Comunes con Virtual Environment

#### **Problema:** `venv\Scripts\activate` no funciona en Windows

**Solución:**
```bash
# Usar PowerShell en lugar de CMD
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1
```

#### **Problema:** Python no encuentra los módulos instalados

**Solución:**
```bash
# Verifica que el entorno esté activado (debe aparecer (venv) en el prompt)
# Si no está activado, actívalo primero
# Verifica qué Python estás usando:
which python  # Linux/macOS
where python  # Windows

# Debe apuntar a venv/bin/python o venv\Scripts\python.exe
```

#### **Problema:** Error al instalar OpenCV

**Solución:**
```bash
# Intenta con la versión headless (sin GUI)
pip install opencv-python-headless

# O actualiza pip primero
python -m pip install --upgrade pip setuptools wheel
pip install opencv-python
```

#### **Problema:** ModuleNotFoundError al ejecutar demo

**Solución:**
```bash
# Asegúrate de estar en el directorio raíz del proyecto
cd melanoma-segmentation

# Y que el entorno virtual esté activado
# Luego ejecuta con:
python src/demo_avance2.py
```

#### **Problema:** Error "Missing optional dependency 'Jinja2'"

**Solución:**
```bash
# Instala jinja2 manualmente
pip install jinja2

# O reinstala todas las dependencias
pip install -r requirements.txt
```

### Problemas con el Dataset

#### **Problema:** "No se encontró la imagen"

**Solución:**
```bash
# Verifica la estructura de directorios:
ls dataset/images/     # Linux/macOS
dir dataset\images\    # Windows

# Ajusta las rutas en el código según tu estructura
```

#### **Problema:** Ground truth tiene dimensiones diferentes

**Solución:** El código ya maneja esto automáticamente. Asegúrate de que las máscaras sean binarias (0 y 255).

#### **Problema:** Warnings de RuntimeWarning en color conversion

**Solución:** Son advertencias normales cuando hay píxeles con valores extremos. No afectan los resultados. Para suprimirlas, agrega al inicio de tu script:
```python
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
```

---

## 🛣️ Roadmap

- [x] Implementación de segmentación adaptativa
- [x] Descriptores A, B, C, D completos
- [x] Métricas IoU y Dice
- [x] Evaluación en batch
- [ ] Interfaz gráfica (GUI)
- [ ] API REST para integración
- [ ] Comparación con métodos deep learning
- [ ] Publicación de paper

---

## 📚 Referencias

1. **Otsu, N. (1979).** "A threshold selection method from gray-level histograms." *IEEE Transactions on Systems, Man, and Cybernetics.*

2. **Nachbar, F. et al. (1994).** "The ABCD rule of dermatoscopy: High prospective value in the diagnosis of doubtful melanocytic skin lesions." *Journal of the American Academy of Dermatology.*

3. **Messadi, M., Cherifi, H., & Bessaid, A. (2021).** "Segmentation and ABCD rule extraction for skin tumors classification."

4. **ISIC Archive.** International Skin Imaging Collaboration. https://www.isic-archive.com/

---

## 👥 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

---

## 📧 Contacto

**Abel Albuez Sanchez** - [correo@javeriana.edu.co](mailto:correo@javeriana.edu.co)  
**Daniel Felipe Rios Caro** - [correo@javeriana.edu.co](mailto:correo@javeriana.edu.co)

**Link del Proyecto:** [Detección Temprana de Melanomas mediante Segmentación Adaptativa](https://github.com/AbelAlbuez/image-video-processing/tree/main/melanoma-segmentation-abcd)

---

## 🙏 Agradecimientos

- Pontificia Universidad Javeriana
- Curso de Procesamiento de Imágenes y Video
- ISIC Archive por proporcionar datos abiertos
- Comunidad de OpenCV y Python

---

<div align="center">

**⭐ Si este proyecto te fue útil, considera darle una estrella ⭐**

</div>