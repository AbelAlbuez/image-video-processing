"""
DEMO - AVANCE 2: Descriptores ABCD + Métricas IoU/Dice
Pontificia Universidad Javeriana - Procesamiento de Imágenes y Video

Autores: Abel Albuez Sanchez, Daniel Felipe Rios
Fecha: Octubre 2025

Este script demuestra el uso completo del sistema de análisis de melanomas.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Importar las clases (asegúrate de que estén en la misma carpeta)
try:
    from melanoma_descriptors import MelanomaDescriptors
    from batch_evaluation import BatchEvaluator
except ImportError:
    print("❌ Error: No se encontraron los módulos necesarios.")
    print("   Asegúrate de que melanoma_descriptors.py y batch_evaluation.py")
    print("   estén en la misma carpeta que este script.")
    sys.exit(1)


def demo_imagen_individual():
    """
    DEMO 1: Procesar una imagen individual
    Muestra todos los descriptores ABCD
    """
    print("\n" + "="*70)
    print("DEMO 1: ANÁLISIS DE UNA IMAGEN INDIVIDUAL")
    print("="*70)
    
    # Ruta de ejemplo - AJUSTA SEGÚN TU DATASET
    image_path = "dataset/images/ISIC_0024306.jpg"
    
    if not os.path.exists(image_path):
        print(f"⚠️  No se encontró la imagen: {image_path}")
        print("   Ajusta la ruta en el código o proporciona una imagen de prueba.")
        return
    
    print(f"📷 Procesando: {image_path}")
    
    # Crear analizador
    analyzer = MelanomaDescriptors(image_path)
    
    # Calcular todos los descriptores
    print("🔄 Calculando descriptores ABCD...")
    descriptors = analyzer.calculate_all_descriptors()
    
    # Mostrar resultados
    print("\n" + "-"*70)
    print("📊 RESULTADOS:")
    print("-"*70)
    
    print("\n🔹 A - ASIMETRÍA:")
    print(f"   Score Total:           {descriptors['A_asymmetry']['asymmetry_score']:.4f}")
    print(f"   Asimetría Horizontal:  {descriptors['A_asymmetry']['horizontal_asymmetry']:.4f}")
    print(f"   Asimetría Vertical:    {descriptors['A_asymmetry']['vertical_asymmetry']:.4f}")
    
    print("\n🔹 B - BORDE:")
    print(f"   Circularidad:          {descriptors['B_border']['circularity']:.4f}")
    print(f"   Área (px²):            {descriptors['B_border']['area']:.2f}")
    print(f"   Perímetro (px):        {descriptors['B_border']['perimeter']:.2f}")
    
    print("\n🔹 C - COLOR:")
    print(f"   Std Lab:               {descriptors['C_color']['color_std_lab']:.4f}")
    print(f"   Número de colores:     {descriptors['C_color']['num_colors']}")
    print(f"   Varianza:              {descriptors['C_color']['color_variance']:.4f}")
    print(f"   Score (0-1):           {descriptors['C_color']['color_score']:.4f}")
    
    print("\n🔹 D - DIÁMETRO:")
    print(f"   Píxeles:               {descriptors['D_diameter']['diameter_px']:.2f} px")
    print(f"   Milímetros (est.):     {descriptors['D_diameter']['diameter_mm']:.2f} mm")
    
    # Visualizar
    print("\n📊 Generando visualización...")
    analyzer.visualize_results(save_path="demo_resultado_individual.png")
    print("✅ Visualización guardada en: demo_resultado_individual.png")
    
    # Si hay ground truth, evaluar
    gt_path = "dataset/masks/ISIC_0024306_segmentation.png"
    if os.path.exists(gt_path):
        print("\n📈 Evaluando contra ground truth...")
        gt_mask = cv2.imread(gt_path, 0)
        metrics = analyzer.evaluate_segmentation(gt_mask)
        print(f"   IoU:  {metrics['iou']:.4f}")
        print(f"   Dice: {metrics['dice']:.4f}")
    
    print("-"*70)


def demo_batch_processing():
    """
    DEMO 2: Procesar múltiples imágenes (batch)
    Genera reporte completo con estadísticas
    """
    print("\n" + "="*70)
    print("DEMO 2: PROCESAMIENTO EN BATCH (MÚLTIPLES IMÁGENES)")
    print("="*70)
    
    # Configurar rutas - AJUSTA SEGÚN TU ESTRUCTURA
    images_dir = "dataset/images"
    masks_dir = "dataset/masks"  # Opcional, comentar si no tienes
    output_dir = "results_demo"
    
    # Verificar que exista el directorio de imágenes
    if not os.path.exists(images_dir):
        print(f"⚠️  No se encontró el directorio: {images_dir}")
        print("   Ajusta las rutas en el código según tu estructura.")
        return
    
    # Verificar si hay máscaras ground truth
    has_masks = os.path.exists(masks_dir)
    if not has_masks:
        print("⚠️  No se encontraron máscaras ground truth.")
        print("   Se procesarán las imágenes sin métricas de evaluación.")
        masks_dir = None
    
    # Crear evaluador
    print(f"\n📁 Imágenes:  {images_dir}")
    print(f"📁 Máscaras:  {masks_dir if masks_dir else 'N/A'}")
    print(f"📁 Salida:    {output_dir}")
    
    evaluator = BatchEvaluator(
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_dir=output_dir
    )
    
    # Procesar dataset (limitar a 10 imágenes para demo rápida)
    print("\n🔄 Procesando imágenes...")
    df_results = evaluator.process_dataset(max_images=10)
    
    if len(df_results) == 0:
        print("❌ No se pudieron procesar imágenes. Verifica las rutas.")
        return
    
    print(f"\n✅ Procesadas {len(df_results)} imágenes exitosamente")
    
    # Generar reporte completo
    print("\n📝 Generando reporte completo...")
    evaluator.generate_full_report()
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETADA")
    print(f"📂 Revisa los resultados en: {output_dir}/")
    print("="*70)
    
    # Mostrar preview de resultados
    print("\n📊 PREVIEW DE RESULTADOS:")
    print(df_results[['image_name', 'asymmetry_score', 'circularity', 
                      'color_std_lab', 'diameter_px']].head())


def demo_crear_dataset_ejemplo():
    """
    DEMO 3: Crear un dataset de ejemplo sintético
    Útil para probar si no tienes imágenes
    """
    print("\n" + "="*70)
    print("DEMO 3: CREAR DATASET DE EJEMPLO SINTÉTICO")
    print("="*70)
    
    output_dir = Path("dataset_ejemplo")
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Generando imágenes sintéticas...")
    
    for i in range(5):
        # Crear imagen sintética (simula lesión oscura en piel clara)
        img = np.random.randint(180, 220, (600, 450, 3), dtype=np.uint8)
        
        # Agregar "lesión" oscura circular
        center = (np.random.randint(150, 300), np.random.randint(150, 450))
        radius = np.random.randint(50, 100)
        color = (np.random.randint(20, 80), np.random.randint(30, 90), 
                np.random.randint(40, 100))
        cv2.circle(img, center, radius, color, -1)
        
        # Agregar ruido
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # Guardar imagen
        img_path = images_dir / f"synthetic_{i:03d}.jpg"
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Crear máscara ground truth
        mask = np.zeros((600, 450), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        mask_path = masks_dir / f"synthetic_{i:03d}.png"
        cv2.imwrite(str(mask_path), mask)
    
    print(f"✅ Dataset sintético creado en: {output_dir}/")
    print(f"   - 5 imágenes en: {images_dir}/")
    print(f"   - 5 máscaras en: {masks_dir}/")
    print("\n💡 Ahora puedes usar estas rutas para probar las demos 1 y 2")


def menu_principal():
    """Menú interactivo para elegir demo"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   SISTEMA DE ANÁLISIS DE MELANOMAS - DEMO AVANCE 2          ║
    ║   Pontificia Universidad Javeriana                           ║
    ║   Procesamiento de Imágenes y Video                          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Selecciona una opción:
    
    1. Demo 1: Analizar UNA imagen individual
    2. Demo 2: Procesamiento en BATCH (múltiples imágenes)
    3. Demo 3: Crear dataset sintético de ejemplo
    4. Ejecutar TODAS las demos
    5. Salir
    """)
    
    while True:
        try:
            opcion = input("\n👉 Ingresa el número de opción (1-5): ").strip()
            
            if opcion == "1":
                demo_imagen_individual()
                break
            elif opcion == "2":
                demo_batch_processing()
                break
            elif opcion == "3":
                demo_crear_dataset_ejemplo()
                break
            elif opcion == "4":
                demo_crear_dataset_ejemplo()
                demo_imagen_individual()
                demo_batch_processing()
                break
            elif opcion == "5":
                print("\n👋 ¡Hasta luego!")
                break
            else:
                print("❌ Opción inválida. Ingresa un número del 1 al 5.")
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break


# ==================== PUNTO DE ENTRADA ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("INICIANDO SISTEMA DE ANÁLISIS DE MELANOMAS")
    print("="*70)
    
    # Verificar dependencias
    try:
        import cv2
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from skimage import color
        from scipy import ndimage
        from tqdm import tqdm
        print("✅ Todas las dependencias están instaladas")
    except ImportError as e:
        print(f"❌ Falta instalar dependencias: {e}")
        print("\n📦 Instala con: pip install opencv-python numpy pandas matplotlib seaborn scikit-image scipy tqdm")
        sys.exit(1)
    
    # Mostrar menú
    menu_principal()
    
    print("\n" + "="*70)
    print("🎉 DEMO FINALIZADA")
    print("="*70)
    print("""
    📚 SIGUIENTE PASO:
    
    Para tu Avance 2, necesitas:
    1. ✅ Código funcionando (ya lo tienes)
    2. ✅ Procesar tu dataset real (ISIC)
    3. ✅ Generar tablas y gráficos para el informe
    4. ✅ Actualizar tu documento LaTeX con los resultados
    
    💡 Usa el BatchEvaluator con tu dataset completo para obtener
       todos los resultados necesarios para el informe.
    
    🚀 ¡Éxito en tu entrega!
    """)