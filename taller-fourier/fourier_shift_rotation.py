import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift
from skimage import data, transform
from skimage.color import rgb2gray
from skimage.registration import phase_cross_correlation

# ===== FUNCIONES AUXILIARES =====

def calcular_espectro_magnitud(imagen):
    """Calcula el espectro de magnitud de la FFT"""
    f_transform = fft2(imagen)
    f_shift = fftshift(f_transform)
    magnitud = np.abs(f_shift)
    # Escala logarítmica para mejor visualización
    espectro_log = np.log1p(magnitud)
    return f_shift, espectro_log

def detectar_desplazamiento(img_original, img_desplazada):
    """
    Detecta el desplazamiento usando correlación de fase cruzada.
    Retorna: (shift_y, shift_x) en píxeles
    """
    # Método 1: Usando phase_cross_correlation de scikit-image
    shift, error, diffphase = phase_cross_correlation(
        img_original, img_desplazada, upsample_factor=10
    )
    return shift

def detectar_rotacion_correlacion(img_original, img_rotada):
    """
    Detecta la rotación usando correlación en coordenadas polares
    del espectro de magnitud.
    """
    # Calcular espectros de magnitud
    _, espectro1 = calcular_espectro_magnitud(img_original)
    _, espectro2 = calcular_espectro_magnitud(img_rotada)
    
    # Convertir espectros a coordenadas polares
    h, w = espectro1.shape
    cy, cx = h // 2, w // 2
    
    # Crear grid de coordenadas
    y, x = np.ogrid[:h, :w]
    y, x = y - cy, x - cx
    
    # Radio y ángulo
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Parámetros para transformación polar
    r_max = min(cy, cx) - 1
    n_r = int(r_max)
    n_theta = 360
    
    # Remapeo a coordenadas polares
    r_i = np.linspace(0, r_max, n_r)
    theta_i = np.linspace(-np.pi, np.pi, n_theta)
    
    polar1 = np.zeros((n_r, n_theta))
    polar2 = np.zeros((n_r, n_theta))
    
    for i, ri in enumerate(r_i):
        for j, thetaj in enumerate(theta_i):
            # Coordenadas cartesianas
            xi = int(cx + ri * np.cos(thetaj))
            yi = int(cy + ri * np.sin(thetaj))
            
            if 0 <= xi < w and 0 <= yi < h:
                polar1[i, j] = espectro1[yi, xi]
                polar2[i, j] = espectro2[yi, xi]
    
    # Correlación en la dimensión angular (suma sobre radios)
    perfil1 = np.sum(polar1, axis=0)
    perfil2 = np.sum(polar2, axis=0)
    
    # Correlación cruzada
    correlacion = np.correlate(perfil2, perfil1, mode='same')
    pico = np.argmax(correlacion)
    
    # Calcular ángulo
    angulo = (pico - n_theta // 2) * 360 / n_theta
    
    return angulo

# ===== EJEMPLO COMPLETO =====

def demo_completo():
    """Demostración completa con desplazamiento y rotación"""
    
    # Cargar imagen de ejemplo
    img_original = rgb2gray(data.astronaut())
    
    # ===== PARTE 1: DESPLAZAMIENTO =====
    print("=" * 60)
    print("DETECCIÓN DE DESPLAZAMIENTO")
    print("=" * 60)
    
    # Crear imagen desplazada
    shift_real = (30, 50)  # (y, x) en píxeles
    img_shifted = ndimage.shift(img_original, shift_real, mode='constant')
    
    # Detectar desplazamiento
    shift_detectado = detectar_desplazamiento(img_original, img_shifted)
    
    print(f"\nDesplazamiento real: {shift_real}")
    print(f"Desplazamiento detectado: ({shift_detectado[0]:.2f}, {shift_detectado[1]:.2f})")
    print(f"Error: ({abs(shift_detectado[0] - shift_real[0]):.2f}, "
          f"{abs(shift_detectado[1] - shift_real[1]):.2f}) píxeles")
    
    # Visualización desplazamiento
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Imágenes originales
    axes[0, 0].imshow(img_original, cmap='gray')
    axes[0, 0].set_title('Imagen Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_shifted, cmap='gray')
    axes[0, 1].set_title(f'Desplazada {shift_real} px')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.abs(img_original - img_shifted), cmap='hot')
    axes[0, 2].set_title('Diferencia')
    axes[0, 2].axis('off')
    
    # Espectros de magnitud
    _, espectro_orig = calcular_espectro_magnitud(img_original)
    _, espectro_shift = calcular_espectro_magnitud(img_shifted)
    
    axes[1, 0].imshow(espectro_orig, cmap='viridis')
    axes[1, 0].set_title('Espectro Original')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(espectro_shift, cmap='viridis')
    axes[1, 1].set_title('Espectro Desplazada\n(invariante en magnitud)')
    axes[1, 1].axis('off')
    
    axes[1, 2].text(0.5, 0.5, 
                    f'Desplazamiento real:\n{shift_real}\n\n'
                    f'Detectado:\n({shift_detectado[0]:.1f}, {shift_detectado[1]:.1f})',
                    ha='center', va='center', fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('desplazamiento_fft.png', dpi=150, bbox_inches='tight')
    print("\n✓ Gráfica guardada: desplazamiento_fft.png")
    
    # ===== PARTE 2: ROTACIÓN =====
    print("\n" + "=" * 60)
    print("DETECCIÓN DE ROTACIÓN")
    print("=" * 60)
    
    # Crear imagen rotada
    angulo_real = 25  # grados
    img_rotada = transform.rotate(img_original, angulo_real, mode='constant')
    
    # Detectar rotación
    angulo_detectado = detectar_rotacion_correlacion(img_original, img_rotada)
    
    print(f"\nÁngulo real: {angulo_real}°")
    print(f"Ángulo detectado: {angulo_detectado:.2f}°")
    print(f"Error: {abs(angulo_detectado - angulo_real):.2f}°")
    
    # Visualización rotación
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(img_original, cmap='gray')
    axes[0, 0].set_title('Imagen Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_rotada, cmap='gray')
    axes[0, 1].set_title(f'Rotada {angulo_real}°')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.abs(img_original - img_rotada), cmap='hot')
    axes[0, 2].set_title('Diferencia')
    axes[0, 2].axis('off')
    
    # Espectros de magnitud
    _, espectro_orig = calcular_espectro_magnitud(img_original)
    _, espectro_rot = calcular_espectro_magnitud(img_rotada)
    
    axes[1, 0].imshow(espectro_orig, cmap='viridis')
    axes[1, 0].set_title('Espectro Original')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(espectro_rot, cmap='viridis')
    axes[1, 1].set_title('Espectro Rotado\n(rotado igual ángulo)')
    axes[1, 1].axis('off')
    
    axes[1, 2].text(0.5, 0.5, 
                    f'Rotación real:\n{angulo_real}°\n\n'
                    f'Detectada:\n{angulo_detectado:.1f}°',
                    ha='center', va='center', fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('rotacion_fft.png', dpi=150, bbox_inches='tight')
    print("\n✓ Gráfica guardada: rotacion_fft.png")
    
    # ===== PARTE 3: COMBINADO =====
    print("\n" + "=" * 60)
    print("TRANSFORMACIÓN COMBINADA (Rotación + Desplazamiento)")
    print("=" * 60)
    
    # Primero rotar, luego desplazar
    img_rot_shift = transform.rotate(img_original, angulo_real, mode='constant')
    img_rot_shift = ndimage.shift(img_rot_shift, shift_real, mode='constant')
    
    # Para detectar ambos, primero detectamos rotación usando espectros
    angulo_det = detectar_rotacion_correlacion(img_original, img_rot_shift)
    
    # Luego corregimos rotación y detectamos desplazamiento
    img_corregida_rot = transform.rotate(img_rot_shift, -angulo_det, mode='constant')
    shift_det = detectar_desplazamiento(img_original, img_corregida_rot)
    
    print(f"\nTransformación real: Rotación {angulo_real}° + Desplazamiento {shift_real}")
    print(f"Detectado: Rotación {angulo_det:.2f}° + Desplazamiento ({shift_det[0]:.2f}, {shift_det[1]:.2f})")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("ANÁLISIS COMPLETADO")
    print("=" * 60)

# ===== FUNCIÓN PARA USAR CON TUS PROPIAS IMÁGENES =====

def analizar_imagenes_propias(img1_path, img2_path):
    """
    Analiza dos imágenes para detectar desplazamiento y rotación.
    
    Args:
        img1_path: Ruta a la imagen original
        img2_path: Ruta a la imagen transformada
    """
    from skimage.io import imread
    
    # Cargar imágenes
    img1 = rgb2gray(imread(img1_path))
    img2 = rgb2gray(imread(img2_path))
    
    # Detectar transformaciones
    shift = detectar_desplazamiento(img1, img2)
    angulo = detectar_rotacion_correlacion(img1, img2)
    
    print(f"\nDesplazamiento detectado: ({shift[0]:.2f}, {shift[1]:.2f}) píxeles")
    print(f"Rotación detectada: {angulo:.2f}°")
    
    # Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title('Imagen 1 (Original)')
    axes[0].axis('off')
    
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title('Imagen 2 (Transformada)')
    axes[1].axis('off')
    
    axes[2].text(0.5, 0.5,
                 f'Desplazamiento:\n({shift[0]:.1f}, {shift[1]:.1f}) px\n\n'
                 f'Rotación:\n{angulo:.1f}°',
                 ha='center', va='center', fontsize=16,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return shift, angulo

# ===== EJECUTAR DEMO =====
if __name__ == "__main__":
    demo_completo()
    
    # Para usar con tus propias imágenes, descomenta:
    # shift, angulo = analizar_imagenes_propias('imagen1.png', 'imagen2.png')