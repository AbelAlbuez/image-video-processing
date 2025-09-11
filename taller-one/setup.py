#!/usr/bin/env python
"""
Script de configuración automática para el Taller 1
Procesamiento de Imágenes y Video
Incluye generador de imágenes de ejemplo
"""

import os
import sys
import subprocess
import platform
import numpy as np


def ejecutar_comando(comando):
    """Ejecuta un comando y muestra su salida."""
    try:
        if isinstance(comando, str):
            comando = comando.split()
        
        proceso = subprocess.Popen(
            comando,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = proceso.communicate()
        
        if proceso.returncode == 0:
            return True, stdout
        else:
            return False, stderr
    except Exception as e:
        return False, str(e)


def crear_entorno_virtual():
    """Crea un entorno virtual."""
    print("📦 Creando entorno virtual...")
    
    venv_name = "venv_taller1"
    python_cmd = "python3" if platform.system() != "Windows" else "python"
    
    # Verificar si ya existe
    if os.path.exists(venv_name):
        print(f"ℹ️  El entorno virtual '{venv_name}' ya existe.")
        return venv_name
    
    exito, salida = ejecutar_comando(f"{python_cmd} -m venv {venv_name}")
    
    if exito:
        print(f"✅ Entorno virtual '{venv_name}' creado exitosamente.")
        return venv_name
    else:
        print(f"❌ Error al crear entorno virtual: {salida}")
        return None


def obtener_comando_activacion(venv_name):
    """Obtiene el comando para activar el entorno virtual."""
    sistema = platform.system()
    
    if sistema == "Windows":
        return os.path.join(venv_name, "Scripts", "activate.bat")
    else:
        return f"source {os.path.join(venv_name, 'bin', 'activate')}"


def instalar_dependencias(venv_name):
    """Instala las dependencias usando pip del entorno virtual."""
    print("\n📚 Instalando dependencias...")
    
    sistema = platform.system()
    
    if sistema == "Windows":
        pip_path = os.path.join(venv_name, "Scripts", "pip")
        python_path = os.path.join(venv_name, "Scripts", "python")
    else:
        pip_path = os.path.join(venv_name, "bin", "pip")
        python_path = os.path.join(venv_name, "bin", "python")
    
    # Actualizar pip
    print("📦 Actualizando pip...")
    exito, _ = ejecutar_comando(f"{python_path} -m pip install --upgrade pip")
    
    if not exito:
        print("⚠️  No se pudo actualizar pip, continuando...")
    
    # Instalar desde requirements.txt
    if os.path.exists("requirements.txt"):
        print("📦 Instalando desde requirements.txt...")
        exito, salida = ejecutar_comando(f"{pip_path} install -r requirements.txt")
        
        if exito:
            print("✅ Dependencias instaladas exitosamente.")
        else:
            print(f"❌ Error al instalar dependencias: {salida}")
            return False
    else:
        print("⚠️  No se encontró requirements.txt")
        print("📦 Instalando dependencias manualmente...")
        
        dependencias = [
            "opencv-python==4.8.1.78",
            "numpy==1.24.3",
            "matplotlib==3.7.2",
            "scikit-learn==1.3.0",
            "scipy==1.11.1",
            "Pillow==10.0.0"
        ]
        
        for dep in dependencias:
            print(f"  Instalando {dep}...")
            exito, _ = ejecutar_comando(f"{pip_path} install {dep}")
            if not exito:
                print(f"  ⚠️  Error al instalar {dep}")
    
    return True


def crear_estructura_directorios():
    """Crea la estructura de directorios necesaria."""
    print("\n📁 Creando estructura de directorios...")
    
    directorios = ["resultados", "informes", "imagenes"]
    
    for directorio in directorios:
        if not os.path.exists(directorio):
            os.makedirs(directorio)
            print(f"  ✅ Creado: {directorio}/")
        else:
            print(f"  ℹ️  Ya existe: {directorio}/")


def crear_gitignore():
    """Crea un archivo .gitignore básico."""
    print("\n📝 Creando .gitignore...")
    
    contenido_gitignore = """# Entorno virtual
venv_taller1/
venv/
env/
ENV/

# Archivos de Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Archivos de IDE
.idea/
.vscode/
*.swp
*.swo

# Archivos de sistema
.DS_Store
Thumbs.db

# Jupyter Notebook
.ipynb_checkpoints/

# Archivos temporales
*.tmp
*.temp
"""
    
    with open(".gitignore", "w") as f:
        f.write(contenido_gitignore)
    
    print("  ✅ .gitignore creado")


def crear_imagenes_ejemplo():
    """Crea imágenes de ejemplo para probar el código."""
    print("\n🎨 Creando imágenes de ejemplo...")
    
    try:
        import cv2
        import numpy as np
        
        # Imagen 1: Bloques de colores puros
        print("  Creando imagen_taller1.jpg - Bloques de colores...")
        img1 = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Fila 1
        img1[0:100, 0:100] = [0, 0, 255]      # Rojo
        img1[0:100, 100:200] = [0, 255, 0]    # Verde
        img1[0:100, 200:300] = [255, 0, 0]    # Azul
        img1[0:100, 300:400] = [0, 255, 255]  # Amarillo
        
        # Fila 2
        img1[100:200, 0:100] = [255, 0, 255]    # Magenta
        img1[100:200, 100:200] = [255, 255, 0]  # Cyan
        img1[100:200, 200:300] = [0, 128, 255]  # Naranja
        img1[100:200, 300:400] = [255, 0, 128]  # Rosa
        
        # Fila 3 - con algo de ruido
        img1[200:300, 0:200] = [128, 128, 128]  # Gris
        img1[200:300, 200:400] = [64, 64, 64]   # Gris oscuro
        
        cv2.imwrite('imagen_taller1.jpg', img1)
        print("    ✅ imagen_taller1.jpg creada")
        
        # Imagen 2: Círculos de colores
        print("  Creando ejemplo_circulos.jpg - Círculos de colores...")
        img2 = np.ones((400, 600, 3), dtype=np.uint8) * 255  # Fondo blanco
        
        # Dibujar círculos de diferentes colores
        cv2.circle(img2, (100, 100), 40, (0, 0, 255), -1)      # Rojo
        cv2.circle(img2, (200, 100), 40, (0, 0, 255), -1)      # Rojo
        cv2.circle(img2, (300, 100), 30, (0, 255, 0), -1)      # Verde
        cv2.circle(img2, (400, 100), 35, (0, 255, 0), -1)      # Verde
        cv2.circle(img2, (500, 100), 40, (0, 255, 0), -1)      # Verde
        
        cv2.circle(img2, (100, 200), 35, (255, 0, 0), -1)      # Azul
        cv2.circle(img2, (200, 200), 40, (255, 0, 0), -1)      # Azul
        cv2.circle(img2, (350, 200), 45, (0, 255, 255), -1)    # Amarillo
        cv2.circle(img2, (450, 200), 30, (0, 255, 255), -1)    # Amarillo
        
        cv2.circle(img2, (150, 300), 40, (255, 0, 255), -1)    # Magenta
        cv2.circle(img2, (250, 300), 35, (255, 0, 255), -1)    # Magenta
        cv2.circle(img2, (350, 300), 40, (255, 0, 255), -1)    # Magenta
        cv2.circle(img2, (450, 300), 50, (255, 255, 0), -1)    # Cyan
        
        cv2.imwrite('imagenes/ejemplo_circulos.jpg', img2)
        print("    ✅ ejemplo_circulos.jpg creada")
        
        # Imagen 3: Objetos más realistas
        print("  Creando ejemplo_objetos.jpg - Objetos variados...")
        img3 = np.ones((500, 700, 3), dtype=np.uint8) * 240  # Fondo gris claro
        
        # Rectángulos (simulando cajas)
        cv2.rectangle(img3, (50, 50), (150, 150), (0, 0, 200), -1)      # Rojo oscuro
        cv2.rectangle(img3, (200, 50), (300, 150), (0, 0, 200), -1)     # Rojo oscuro
        cv2.rectangle(img3, (350, 50), (450, 180), (0, 200, 0), -1)     # Verde oscuro
        cv2.rectangle(img3, (500, 50), (600, 150), (0, 200, 0), -1)     # Verde oscuro
        
        # Elipses (simulando pelotas)
        cv2.ellipse(img3, (100, 250), (40, 40), 0, 0, 360, (200, 0, 0), -1)    # Azul
        cv2.ellipse(img3, (250, 250), (45, 45), 0, 0, 360, (200, 0, 0), -1)    # Azul
        cv2.ellipse(img3, (400, 250), (35, 35), 0, 0, 360, (200, 0, 0), -1)    # Azul
        cv2.ellipse(img3, (550, 250), (50, 50), 0, 0, 360, (0, 200, 200), -1)  # Amarillo
        
        # Polígonos irregulares
        pts1 = np.array([[100, 350], [150, 370], [140, 420], [90, 410], [80, 380]], np.int32)
        cv2.fillPoly(img3, [pts1], (200, 0, 200))  # Magenta
        
        pts2 = np.array([[250, 360], [300, 350], [310, 400], [280, 420], [240, 400]], np.int32)
        cv2.fillPoly(img3, [pts2], (200, 0, 200))  # Magenta
        
        pts3 = np.array([[400, 370], [450, 360], [460, 410], [430, 430], [390, 410]], np.int32)
        cv2.fillPoly(img3, [pts3], (200, 200, 0))  # Cyan
        
        # Añadir algo de ruido
        ruido = np.random.randint(-20, 20, img3.shape, dtype=np.int16)
        img3_ruido = np.clip(img3.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
        
        cv2.imwrite('imagenes/ejemplo_objetos.jpg', img3_ruido)
        print("    ✅ ejemplo_objetos.jpg creada")
        
        # Imagen 4: Degradados y formas complejas
        print("  Creando ejemplo_complejo.jpg - Escena más compleja...")
        img4 = np.ones((400, 600, 3), dtype=np.uint8) * 200
        
        # Añadir gradiente de fondo
        for i in range(400):
            img4[i, :] = img4[i, :] * (1 - i/800.0) + np.array([255, 240, 220]) * (i/800.0)
        
        # Objetos con sombras
        # Círculos rojos con sombra
        cv2.circle(img4, (105, 105), 40, (150, 150, 150), -1)  # Sombra
        cv2.circle(img4, (100, 100), 40, (0, 0, 255), -1)      # Rojo
        
        cv2.circle(img4, (255, 105), 35, (150, 150, 150), -1)  # Sombra
        cv2.circle(img4, (250, 100), 35, (0, 0, 255), -1)      # Rojo
        
        # Cuadrados verdes con sombra
        cv2.rectangle(img4, (405, 85), (485, 165), (150, 150, 150), -1)  # Sombra
        cv2.rectangle(img4, (400, 80), (480, 160), (0, 255, 0), -1)      # Verde
        
        # Triángulos azules
        pts_t1 = np.array([[100, 250], [50, 330], [150, 330]], np.int32)
        cv2.fillPoly(img4, [pts_t1], (255, 0, 0))  # Azul
        
        pts_t2 = np.array([[300, 250], [250, 330], [350, 330]], np.int32)
        cv2.fillPoly(img4, [pts_t2], (255, 0, 0))  # Azul
        
        # Estrellas amarillas (aproximadas con polígonos)
        def dibujar_estrella(img, centro, radio, color):
            puntos = []
            for i in range(10):
                angulo = i * np.pi / 5
                if i % 2 == 0:
                    r = radio
                else:
                    r = radio * 0.5
                x = int(centro[0] + r * np.cos(angulo - np.pi/2))
                y = int(centro[1] + r * np.sin(angulo - np.pi/2))
                puntos.append([x, y])
            cv2.fillPoly(img, [np.array(puntos, np.int32)], color)
        
        dibujar_estrella(img4, (450, 280), 40, (0, 255, 255))  # Amarillo
        dibujar_estrella(img4, (520, 320), 30, (0, 255, 255))  # Amarillo
        
        cv2.imwrite('imagenes/ejemplo_complejo.jpg', img4)
        print("    ✅ ejemplo_complejo.jpg creada")
        
        print("\n✅ Todas las imágenes de ejemplo creadas exitosamente!")
        print("\n📌 Imágenes disponibles:")
        print("   - imagen_taller1.jpg (principal)")
        print("   - imagenes/ejemplo_circulos.jpg")
        print("   - imagenes/ejemplo_objetos.jpg") 
        print("   - imagenes/ejemplo_complejo.jpg")
        
    except ImportError:
        print("  ⚠️  OpenCV no está instalado aún. Las imágenes se crearán después de la instalación.")
        return False
    except Exception as e:
        print(f"  ❌ Error al crear imágenes: {e}")
        return False
    
    return True


def verificar_instalacion(venv_name):
    """Verifica que todo esté instalado correctamente."""
    print("\n🔍 Verificando instalación...")
    
    sistema = platform.system()
    
    if sistema == "Windows":
        python_path = os.path.join(venv_name, "Scripts", "python")
    else:
        python_path = os.path.join(venv_name, "bin", "python")
    
    # Ejecutar el script de prueba
    if os.path.exists("test_instalacion.py"):
        exito, salida = ejecutar_comando(f"{python_path} test_instalacion.py")
        print(salida)
    else:
        # Verificación básica
        exito, salida = ejecutar_comando(
            f'{python_path} -c "import cv2, numpy, matplotlib, sklearn; print(\'✅ Todas las librerías básicas funcionan\')"'
        )
        if exito:
            print(salida)
        else:
            print("❌ Error al verificar librerías")


def main():
    """Función principal."""
    print("🚀 CONFIGURACIÓN AUTOMÁTICA - TALLER 1")
    print("=" * 50)
    print(f"Sistema operativo: {platform.system()}")
    print(f"Versión de Python: {sys.version.split()[0]}")
    print("=" * 50)
    
    # Verificar versión de Python
    if sys.version_info < (3, 7):
        print("❌ Se requiere Python 3.7 o superior")
        return
    
    # Crear entorno virtual
    venv_name = crear_entorno_virtual()
    if not venv_name:
        return
    
    # Instalar dependencias
    instalacion_exitosa = instalar_dependencias(venv_name)
    
    # Crear estructura de directorios
    crear_estructura_directorios()
    
    # Crear .gitignore
    crear_gitignore()
    
    # Si la instalación fue exitosa, crear imágenes de ejemplo
    if instalacion_exitosa:
        # Intentar crear imágenes usando el entorno virtual
        sistema = platform.system()
        if sistema == "Windows":
            python_path = os.path.join(venv_name, "Scripts", "python")
        else:
            python_path = os.path.join(venv_name, "bin", "python")
        
        # Crear script temporal para generar imágenes
        script_imagenes = """
import cv2
import numpy as np
import os

os.makedirs('imagenes', exist_ok=True)

# Imagen principal
img1 = np.zeros((300, 400, 3), dtype=np.uint8)
img1[0:100, 0:100] = [0, 0, 255]
img1[0:100, 100:200] = [0, 255, 0]
img1[0:100, 200:300] = [255, 0, 0]
img1[0:100, 300:400] = [0, 255, 255]
img1[100:200, 0:100] = [255, 0, 255]
img1[100:200, 100:200] = [255, 255, 0]
img1[100:200, 200:300] = [0, 128, 255]
img1[100:200, 300:400] = [255, 0, 128]
img1[200:300, :] = [128, 128, 128]
cv2.imwrite('imagen_taller1.jpg', img1)

# Más imágenes de ejemplo...
print('✅ Imágenes de ejemplo creadas')
"""
        
        with open('_temp_crear_imagenes.py', 'w') as f:
            f.write(script_imagenes)
        
        print("\n🎨 Creando imágenes de ejemplo...")
        exito, _ = ejecutar_comando(f"{python_path} _temp_crear_imagenes.py")
        if exito:
            print("  ✅ imagen_taller1.jpg creada")
        
        # Limpiar archivo temporal
        if os.path.exists('_temp_crear_imagenes.py'):
            os.remove('_temp_crear_imagenes.py')
    
    # Verificar instalación
    verificar_instalacion(venv_name)
    
    # Instrucciones finales
    print("\n" + "=" * 50)
    print("✅ ¡CONFIGURACIÓN COMPLETADA!")
    print("=" * 50)
    print("\nPara activar el entorno virtual:")
    print(f"  {obtener_comando_activacion(venv_name)}")
    print("\nPara ejecutar el taller:")
    print("  python taller1_extraccion.py")
    print("\nPara desactivar el entorno virtual:")
    print("  deactivate")
    print("\n🎯 ¡Buena suerte con el taller!")


if __name__ == "__main__":
    main()