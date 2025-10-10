# 🔬 Taller 2: Aplicación de la Transformada de Fourier

## 📋 Descripción

Implementación de técnicas de procesamiento de imágenes usando la Transformada de Fourier para detectar:

1. **Desplazamiento** de objetos en imágenes
2. **Rotación** de objetos en imágenes

Este proyecto utiliza las propiedades fundamentales de la Transformada de Fourier para analizar transformaciones geométricas en imágenes digitales.

---

## 🎯 Objetivos

- Aplicar el teorema del desplazamiento de Fourier
- Detectar rotaciones usando el espectro de magnitud
- Verificar resultados mediante correlación cruzada de fase
- Visualizar espectros de Fourier y transformaciones

---

## 🛠️ Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Conocimientos básicos de procesamiento de imágenes

---

## 📦 Instalación

### 1. Clonar o descargar el proyecto

```bash
# Si usas git
git clone <url-del-repositorio>
cd taller2-fourier

# O simplemente descarga los archivos en una carpeta
```

### 2. Crear entorno virtual (Recomendado)

#### En Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### En macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 🚀 Uso

### Ejecución de la demostración completa

```bash
python fourier_shift_rotation.py
```

Este comando ejecutará:
- ✅ Detección de desplazamiento con imagen de ejemplo
- ✅ Detección de rotación con imagen de ejemplo
- ✅ Detección combinada (rotación + desplazamiento)
- ✅ Generación de gráficas explicativas

### Analizar tus propias imágenes

Edita el archivo `fourier_shift_rotation.py` y descomenta la última línea:

```python
if __name__ == "__main__":
    demo_completo()
    
    # Descomenta esta línea y reemplaza con tus imágenes:
    shift, angulo = analizar_imagenes_propias('imagen_original.png', 'imagen_transformada.png')
```

---

## 📁 Estructura del Proyecto

```
taller2-fourier/
│
├── fourier_shift_rotation.py    # Código principal
├── requirements.txt              # Dependencias
├── README.md                     # Este archivo
│
├── desplazamiento_fft.png       # Gráfica generada (desplazamiento)
└── rotacion_fft.png             # Gráfica generada (rotación)
```

---

## 🧮 Fundamento Teórico

### Detección de Desplazamiento

**Teorema del Desplazamiento:**
```
f(x-x₀, y-y₀) ↔ F(u,v) · e^(-j2π(ux₀+vy₀))
```

- La magnitud del espectro **NO cambia** con desplazamientos
- Solo cambia la **fase**
- Usamos **correlación cruzada de fase** para detectar el desplazamiento

### Detección de Rotación

**Propiedad de Rotación:**
```
Si f(x,y) rota θ° → |F(u,v)| también rota θ°
```

- El espectro de magnitud se rota el **mismo ángulo**
- Transformamos a coordenadas polares
- Correlación en el eje angular detecta la rotación

---

## 🔍 Funciones Principales

### `detectar_desplazamiento(img1, img2)`
Detecta el desplazamiento entre dos imágenes en píxeles.

**Retorna:** `(shift_y, shift_x)` en píxeles

### `detectar_rotacion_correlacion(img1, img2)`
Detecta el ángulo de rotación entre dos imágenes.

**Retorna:** Ángulo en grados

### `calcular_espectro_magnitud(imagen)`
Calcula y visualiza el espectro de magnitud de la FFT.

**Retorna:** Transformada de Fourier y espectro logarítmico

---

## 📊 Resultados Esperados

El programa generará:

1. **desplazamiento_fft.png**: Visualización de detección de desplazamiento
   - Imágenes original y desplazada
   - Espectros de Fourier (invariantes en magnitud)
   - Valores detectados vs. reales

2. **rotacion_fft.png**: Visualización de detección de rotación
   - Imágenes original y rotada
   - Espectros rotados (misma rotación)
   - Ángulo detectado vs. real

---

## 🔧 Personalización

### Cambiar parámetros de transformación

En `demo_completo()`, puedes modificar:

```python
# Desplazamiento (en píxeles)
shift_real = (30, 50)  # (y, x)

# Rotación (en grados)
angulo_real = 25

# Cambiar imagen de ejemplo
img_original = rgb2gray(data.astronaut())  # Cambiar por tu imagen
```

### Usar diferentes imágenes

```python
from skimage.io import imread

# Cargar tu imagen
img = imread('tu_imagen.jpg')
img = rgb2gray(img)  # Convertir a escala de grises
```

---

## ⚠️ Solución de Problemas

### Error: "No module named 'skimage'"
```bash
pip install --upgrade scikit-image
```

### Error: "Cannot allocate memory"
- Reduce el tamaño de las imágenes
- Usa imágenes más pequeñas para pruebas

### Las gráficas no se muestran
- Asegúrate de ejecutar en un entorno con interfaz gráfica
- Si usas Jupyter, agrega: `%matplotlib inline`

---

## 📚 Referencias

- Gonzalez & Woods - "Digital Image Processing"
- Teorema del Desplazamiento de Fourier
- Documentación de scikit-image: https://scikit-image.org/
- NumPy FFT: https://numpy.org/doc/stable/reference/routines.fft.html

---

## 👥 Autores

Taller 2 - Procesamiento de Imágenes y Video  
[Tu Nombre]  
[Tu Universidad/Institución]

---

## 📝 Notas Adicionales

- Las imágenes deben estar en formato compatible (PNG, JPG, BMP)
- Se recomienda usar imágenes en escala de grises
- Para mejor precisión, usa imágenes con características distintivas
- El método funciona mejor con objetos que no salen del marco de la imagen

---

## 🎓 Ejercicios Propuestos

1. Prueba con diferentes ángulos de rotación (-180° a 180°)
2. Combina rotación y desplazamiento en diferentes órdenes
3. Analiza la precisión con diferentes tipos de imágenes
4. Implementa detección para imágenes a color (procesa cada canal)
5. Agrega manejo de ruido en las imágenes

---

## ✅ Verificación del Setup

Para verificar que todo está correctamente instalado:

```bash
python -c "import numpy; import matplotlib; import scipy; import skimage; print('✓ Todas las dependencias instaladas correctamente')"
```

Si ves el mensaje de éxito, ¡estás listo para comenzar! 🎉