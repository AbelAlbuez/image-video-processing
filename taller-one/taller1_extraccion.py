#!/usr/bin/env python
"""
Script de configuración automática para el Taller 1
Procesamiento de Imágenes y Video
Genera 2 imágenes de ejemplo
"""

import os
import sys
import subprocess
import platform


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
            return True
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


def crear_dos_imagenes_ejemplo(venv_name):
    """Crea dos imágenes de ejemplo usando el entorno virtual."""
    print("\n🎨 Creando 2 imágenes de ejemplo...")
    
    sistema = platform.system()
    if sistema == "Windows":
        python_path = os.path.join(venv_name, "Scripts", "python")
    else:
        python_path = os.path.join(venv_name, "bin", "python")
    
    # Script para crear las imágenes
    script_imagenes = '''
import cv2
import numpy as np
import os

# Crear carpeta imagenes si no existe
os.makedirs('imagenes', exist_ok=True)

print("Generando imágenes de ejemplo...")

# ========== IMAGEN 1: Bloques de colores puros ==========
print("  1. Creando imagen_taller1.jpg - Bloques de 6 colores...")
img1 = np.zeros((300, 600, 3), dtype=np.uint8)

# 6 colores diferentes en bloques
# Fila superior
img1[0:150, 0:200] = [0, 0, 255]      # Rojo
img1[0:150, 200:400] = [0, 255, 0]    # Verde  
img1[0:150, 400:600] = [255, 0, 0]    # Azul

# Fila inferior
img1[150:300, 0:200] = [0, 255, 255]    # Amarillo
img1[150:300, 200:400] = [255, 0, 255]  # Magenta
img1[150:300, 400:600] = [255, 255, 0]  # Cyan

cv2.imwrite('imagen_taller1.jpg', img1)
print("    ✓ imagen_taller1.jpg creada")

# ========== IMAGEN 2: Círculos de colores ==========
print("  2. Creando ejemplo_circulos.jpg - Múltiples objetos...")
img2 = np.ones((400, 600, 3), dtype=np.uint8) * 240  # Fondo gris claro

# Círculos rojos (3 objetos)
cv2.circle(img2, (100, 100), 40, (0, 0, 255), -1)
cv2.circle(img2, (200, 100), 35, (0, 0, 255), -1)
cv2.circle(img2, (300, 100), 45, (0, 0, 255), -1)

# Círculos verdes (4 objetos)
cv2.circle(img2, (400, 100), 35, (0, 255, 0), -1)
cv2.circle(img2, (500, 100), 40, (0, 255, 0), -1)
cv2.circle(img2, (100, 200), 35, (0, 255, 0), -1)
cv2.circle(img2, (200, 200), 30, (0, 255, 0), -1)

# Círculos azules (2 objetos)
cv2.circle(img2, (350, 200), 45, (255, 0, 0), -1)
cv2.circle(img2, (450, 200), 40, (255, 0, 0), -1)

# Rectángulos amarillos (3 objetos)
cv2.rectangle(img2, (50, 280), (120, 350), (0, 255, 255), -1)
cv2.rectangle(img2, (150, 280), (220, 350), (0, 255, 255), -1)
cv2.rectangle(img2, (250, 280), (320, 350), (0, 255, 255), -1)

# Elipses magenta (2 objetos)
cv2.ellipse(img2, (400, 320), (40, 30), 0, 0, 360, (255, 0, 255), -1)
cv2.ellipse(img2, (500, 320), (35, 40), 45, 0, 360, (255, 0, 255), -1)

# Hexágono cyan (1 objeto)
pts = []
for i in range(6):
    angle = i * 2 * np.pi / 6
    x = int(300 + 40 * np.cos(angle))
    y = int(320 + 40 * np.sin(angle))
    pts.append([x, y])
cv2.fillPoly(img2, [np.array(pts, np.int32)], (255, 255, 0))

cv2.imwrite('imagenes/ejemplo_circulos.jpg', img2)
print("    ✓ ejemplo_circulos.jpg creada")

print("\\n✅ 2 imágenes de ejemplo creadas exitosamente!")
'''
    
    # Guardar el script temporal
    with open('_temp_crear_imagenes.py', 'w') as f:
        f.write(script_imagenes)
    
    # Ejecutar el script
    exito, salida = ejecutar_comando(f"{python_path} _temp_crear_imagenes.py")
    
    if exito:
        print("\n✅ Imágenes creadas exitosamente:")
        print("   • imagen_taller1.jpg - Bloques de 6 colores puros")
        print("   • imagenes/ejemplo_circulos.jpg - Múltiples objetos de colores")
    else:
        print(f"\n❌ Error al crear imágenes: {salida}")
    
    # Limpiar archivo temporal
    if os.path.exists('_temp_crear_imagenes.py'):
        os.remove('_temp_crear_imagenes.py')
    
    return exito


def verificar_instalacion(venv_name):
    """Verifica que todo esté instalado correctamente."""
    print("\n🔍 Verificando instalación...")
    
    sistema = platform.system()
    
    if sistema == "Windows":
        python_path = os.path.join(venv_name, "Scripts", "python")
    else:
        python_path = os.path.join(venv_name, "bin", "python")
    
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
    
    # Si la instalación fue exitosa, crear las 2 imágenes de ejemplo
    if instalacion_exitosa:
        crear_dos_imagenes_ejemplo(venv_name)
    
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
    print("\nImágenes disponibles:")
    print("  • imagen_taller1.jpg (principal)")
    print("  • imagenes/ejemplo_circulos.jpg")
    print("\n🎯 ¡Buena suerte con el taller!")


if __name__ == "__main__":
    main()