import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json

# Importar la clase anterior (asumiendo que está en melanoma_descriptors.py)
# from melanoma_descriptors import MelanomaDescriptors

class BatchEvaluator:
    """
    Clase para evaluar múltiples imágenes y generar reportes completos
    """
    
    def __init__(self, images_dir, masks_dir=None, output_dir='results'):
        """
        Args:
            images_dir: directorio con imágenes de lesiones
            masks_dir: directorio con máscaras ground truth (opcional)
            output_dir: directorio para guardar resultados
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = []
        self.df_results = None
        
    def process_dataset(self, max_images=None):
        """
        Procesa todas las imágenes del dataset
        
        Args:
            max_images: número máximo de imágenes a procesar (None = todas)
        """
        # Obtener lista de imágenes
        image_files = list(self.images_dir.glob('*.jpg')) + \
                     list(self.images_dir.glob('*.png'))
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"🔍 Procesando {len(image_files)} imágenes...")
        
        for img_path in tqdm(image_files, desc="Procesando"):
            try:
                result = self._process_single_image(img_path)
                if result:
                    self.results.append(result)
            except Exception as e:
                print(f"❌ Error procesando {img_path.name}: {str(e)}")
        
        # Crear DataFrame con resultados
        self.df_results = pd.DataFrame(self.results)
        
        print(f"✅ Procesamiento completo: {len(self.results)} imágenes exitosas")
        
        return self.df_results
    
    def _process_single_image(self, img_path):
        """Procesa una imagen individual"""
        from melanoma_descriptors import MelanomaDescriptors
        
        # Crear analizador
        analyzer = MelanomaDescriptors(str(img_path))
        
        # Calcular descriptores
        descriptors = analyzer.calculate_all_descriptors()
        
        # Preparar resultado
        result = {
            'image_name': img_path.name,
            'image_path': str(img_path),
        }
        
        # Agregar descriptores A
        result['asymmetry_score'] = descriptors['A_asymmetry']['asymmetry_score']
        result['h_asymmetry'] = descriptors['A_asymmetry']['horizontal_asymmetry']
        result['v_asymmetry'] = descriptors['A_asymmetry']['vertical_asymmetry']
        
        # Agregar descriptores B
        result['circularity'] = descriptors['B_border']['circularity']
        result['area_px2'] = descriptors['B_border']['area']
        result['perimeter_px'] = descriptors['B_border']['perimeter']
        
        # Agregar descriptores C
        result['color_std_lab'] = descriptors['C_color']['color_std_lab']
        result['num_colors'] = descriptors['C_color']['num_colors']
        result['color_variance'] = descriptors['C_color']['color_variance']
        result['color_score'] = descriptors['C_color']['color_score']
        
        # Agregar descriptores D
        result['diameter_px'] = descriptors['D_diameter']['diameter_px']
        result['diameter_mm'] = descriptors['D_diameter']['diameter_mm']
        
        # Si hay ground truth, calcular métricas
        if self.masks_dir:
            mask_path = self.masks_dir / img_path.name
            if mask_path.exists():
                gt_mask = cv2.imread(str(mask_path), 0)
                metrics = analyzer.evaluate_segmentation(gt_mask)
                result['iou'] = metrics['iou']
                result['dice'] = metrics['dice']
            else:
                result['iou'] = None
                result['dice'] = None
        
        return result
    
    def generate_summary_statistics(self):
        """Genera estadísticas descriptivas del dataset"""
        if self.df_results is None:
            print("⚠️ No hay resultados para analizar. Ejecuta process_dataset() primero.")
            return None
        
        print("\n" + "="*70)
        print("📊 ESTADÍSTICAS DESCRIPTIVAS DEL DATASET")
        print("="*70)
        
        stats = self.df_results.describe()
        print(stats)
        
        # Guardar estadísticas
        stats.to_csv(self.output_dir / 'estadisticas_descriptivas.csv')
        
        return stats
    
    def generate_latex_table(self, num_samples=10):
        """
        Genera tabla en formato LaTeX para el informe
        
        Args:
            num_samples: número de ejemplos a incluir en la tabla
        """
        if self.df_results is None:
            print("⚠️ No hay resultados. Ejecuta process_dataset() primero.")
            return
        
        # Seleccionar columnas principales
        cols = ['image_name', 'asymmetry_score', 'circularity', 
                'color_std_lab', 'diameter_px']
        
        if 'iou' in self.df_results.columns:
            cols.extend(['iou', 'dice'])
        
        df_sample = self.df_results[cols].head(num_samples)
        
        # Generar LaTeX
        latex_table = df_sample.to_latex(
            index=False,
            float_format="%.4f",
            column_format='l' + 'r' * (len(cols) - 1),
            caption='Resultados de descriptores ABCD y métricas de evaluación',
            label='tab:resultados_descriptores'
        )
        
        # Guardar
        output_path = self.output_dir / 'tabla_resultados.tex'
        with open(output_path, 'w') as f:
            f.write(latex_table)
        
        print(f"\n✅ Tabla LaTeX guardada en: {output_path}")
        print("\n" + "="*70)
        print("TABLA LATEX:")
        print("="*70)
        print(latex_table)
        
        return latex_table
    
    def plot_descriptors_distribution(self):
        """Genera gráficos de distribución de los descriptores"""
        if self.df_results is None:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Distribución de Descriptores ABCD', fontsize=16, fontweight='bold')
        
        # A - Asimetría
        axes[0, 0].hist(self.df_results['asymmetry_score'], bins=20, color='steelblue', edgecolor='black')
        axes[0, 0].set_title('A - Asimetría')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].axvline(self.df_results['asymmetry_score'].mean(), 
                          color='red', linestyle='--', label='Media')
        axes[0, 0].legend()
        
        # B - Circularidad (Borde)
        axes[0, 1].hist(self.df_results['circularity'], bins=20, color='green', edgecolor='black')
        axes[0, 1].set_title('B - Borde (Circularidad)')
        axes[0, 1].set_xlabel('Circularidad')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].axvline(self.df_results['circularity'].mean(), 
                          color='red', linestyle='--', label='Media')
        axes[0, 1].legend()
        
        # C - Variación de Color
        axes[0, 2].hist(self.df_results['color_std_lab'], bins=20, color='orange', edgecolor='black')
        axes[0, 2].set_title('C - Color (Std Lab)')
        axes[0, 2].set_xlabel('Desviación Estándar')
        axes[0, 2].set_ylabel('Frecuencia')
        axes[0, 2].axvline(self.df_results['color_std_lab'].mean(), 
                          color='red', linestyle='--', label='Media')
        axes[0, 2].legend()
        
        # D - Diámetro
        axes[1, 0].hist(self.df_results['diameter_mm'], bins=20, color='purple', edgecolor='black')
        axes[1, 0].set_title('D - Diámetro')
        axes[1, 0].set_xlabel('Diámetro (mm)')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].axvline(self.df_results['diameter_mm'].mean(), 
                          color='red', linestyle='--', label='Media')
        axes[1, 0].legend()
        
        # IoU (si existe)
        if 'iou' in self.df_results.columns:
            iou_data = self.df_results['iou'].dropna()
            axes[1, 1].hist(iou_data, bins=20, color='teal', edgecolor='black')
            axes[1, 1].set_title('IoU (Intersection over Union)')
            axes[1, 1].set_xlabel('IoU Score')
            axes[1, 1].set_ylabel('Frecuencia')
            axes[1, 1].axvline(iou_data.mean(), color='red', linestyle='--', label='Media')
            axes[1, 1].axvline(0.6, color='green', linestyle=':', label='Objetivo (0.6)')
            axes[1, 1].legend()
        
        # Dice (si existe)
        if 'dice' in self.df_results.columns:
            dice_data = self.df_results['dice'].dropna()
            axes[1, 2].hist(dice_data, bins=20, color='coral', edgecolor='black')
            axes[1, 2].set_title('Coeficiente de Dice')
            axes[1, 2].set_xlabel('Dice Score')
            axes[1, 2].set_ylabel('Frecuencia')
            axes[1, 2].axvline(dice_data.mean(), color='red', linestyle='--', label='Media')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distribucion_descriptores.png', dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado en: {self.output_dir / 'distribucion_descriptores.png'}")
        plt.show()
    
    def plot_correlation_matrix(self):
        """Genera matriz de correlación entre descriptores"""
        if self.df_results is None:
            return
        
        # Seleccionar columnas numéricas
        numeric_cols = ['asymmetry_score', 'circularity', 'color_std_lab', 
                       'diameter_px', 'area_px2']
        
        if 'iou' in self.df_results.columns:
            numeric_cols.extend(['iou', 'dice'])
        
        corr_matrix = self.df_results[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Matriz de Correlación entre Descriptores', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'matriz_correlacion.png', dpi=300, bbox_inches='tight')
        print(f"✅ Matriz guardada en: {self.output_dir / 'matriz_correlacion.png'}")
        plt.show()
    
    def evaluate_performance(self):
        """Evalúa el rendimiento de la segmentación contra objetivos"""
        if 'iou' not in self.df_results.columns:
            print("⚠️ No hay métricas de evaluación. Proporciona máscaras ground truth.")
            return
        
        iou_data = self.df_results['iou'].dropna()
        dice_data = self.df_results['dice'].dropna()
        
        # Objetivo: IoU >= 0.6 en >= 70% de las imágenes
        target_iou = 0.6
        target_percentage = 0.7
        
        images_above_target = (iou_data >= target_iou).sum()
        percentage_above = images_above_target / len(iou_data)
        
        print("\n" + "="*70)
        print("🎯 EVALUACIÓN DE RENDIMIENTO")
        print("="*70)
        print(f"Objetivo del proyecto: IoU >= {target_iou} en >= {target_percentage*100}% de imágenes")
        print(f"\nResultados obtenidos:")
        print(f"  • IoU promedio: {iou_data.mean():.4f} (±{iou_data.std():.4f})")
        print(f"  • IoU mediana: {iou_data.median():.4f}")
        print(f"  • IoU mínimo: {iou_data.min():.4f}")
        print(f"  • IoU máximo: {iou_data.max():.4f}")
        print(f"  • Imágenes con IoU >= {target_iou}: {images_above_target}/{len(iou_data)} ({percentage_above*100:.1f}%)")
        
        print(f"\n  • Dice promedio: {dice_data.mean():.4f} (±{dice_data.std():.4f})")
        print(f"  • Dice mediana: {dice_data.median():.4f}")
        
        # Evaluación del objetivo
        if percentage_above >= target_percentage:
            print(f"\n✅ OBJETIVO CUMPLIDO: {percentage_above*100:.1f}% >= {target_percentage*100}%")
        else:
            print(f"\n⚠️ OBJETIVO NO CUMPLIDO: {percentage_above*100:.1f}% < {target_percentage*100}%")
            print(f"   Faltan {int((target_percentage - percentage_above) * len(iou_data))} imágenes")
        
        print("="*70)
        
        # Guardar reporte
        report = {
            'iou_mean': float(iou_data.mean()),
            'iou_std': float(iou_data.std()),
            'iou_median': float(iou_data.median()),
            'iou_min': float(iou_data.min()),
            'iou_max': float(iou_data.max()),
            'dice_mean': float(dice_data.mean()),
            'dice_std': float(dice_data.std()),
            'images_above_target': int(images_above_target),
            'total_images': int(len(iou_data)),
            'percentage_above_target': float(percentage_above),
            'target_met': bool(percentage_above >= target_percentage)
        }
        
        with open(self.output_dir / 'reporte_evaluacion.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        return report
    
    def generate_full_report(self):
        """Genera reporte completo con todas las visualizaciones y tablas"""
        print("\n" + "="*70)
        print("📝 GENERANDO REPORTE COMPLETO")
        print("="*70)
        
        # 1. Estadísticas descriptivas
        self.generate_summary_statistics()
        
        # 2. Tabla LaTeX
        self.generate_latex_table(num_samples=15)
        
        # 3. Distribuciones
        self.plot_descriptors_distribution()
        
        # 4. Correlaciones
        self.plot_correlation_matrix()
        
        # 5. Evaluación de rendimiento
        if 'iou' in self.df_results.columns:
            self.evaluate_performance()
        
        # 6. Exportar CSV completo
        csv_path = self.output_dir / 'resultados_completos.csv'
        self.df_results.to_csv(csv_path, index=False)
        print(f"\n✅ CSV completo guardado en: {csv_path}")
        
        print("\n" + "="*70)
        print("✅ REPORTE COMPLETO GENERADO")
        print(f"📁 Todos los archivos en: {self.output_dir}")
        print("="*70)


# ==================== SCRIPT PRINCIPAL ====================
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║   SISTEMA DE EVALUACIÓN DE DESCRIPTORES ABCD - MELANOMA     ║
    ║              Pontificia Universidad Javeriana                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Configuración de rutas (AJUSTAR SEGÚN TU ESTRUCTURA)
    IMAGES_DIR = "dataset/images"  # Carpeta con imágenes
    MASKS_DIR = "dataset/masks"    # Carpeta con ground truth (opcional)
    OUTPUT_DIR = "results_avance2"  # Carpeta de salida
    
    # Crear evaluador
    evaluator = BatchEvaluator(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,  # Comentar si no tienes ground truth
        output_dir=OUTPUT_DIR
    )
    
    # Procesar dataset (ajustar max_images según necesites)
    df_results = evaluator.process_dataset(max_images=50)
    
    # Generar reporte completo
    evaluator.generate_full_report()
    
    print("\n🎉 ¡Proceso completado exitosamente!")
    print(f"📊 Se procesaron {len(df_results)} imágenes")
    print(f"📁 Revisa los resultados en: {OUTPUT_DIR}/")