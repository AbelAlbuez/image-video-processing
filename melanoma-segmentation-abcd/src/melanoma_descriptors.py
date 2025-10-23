import cv2
import numpy as np
from scipy import ndimage
from skimage import color
import matplotlib.pyplot as plt

class MelanomaDescriptors:
    """
    Clase para calcular los descriptores ABCD completos y métricas de evaluación
    para segmentación de lesiones cutáneas (melanoma)
    """
    
    def __init__(self, image_path, mask=None):
        """
        Inicializa con la imagen original y opcionalmente una máscara
        
        Args:
            image_path: ruta a la imagen o imagen RGB como numpy array
            mask: máscara binaria (opcional, si None se calculará)
        """
        if isinstance(image_path, str):
            self.original_image = cv2.imread(image_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        else:
            self.original_image = image_path
            
        if mask is None:
            self.mask = self._segment_lesion()
        else:
            self.mask = mask
            
        self.contours = None
        self.descriptors = {}
        
    def _segment_lesion(self):
        """Segmentación adaptativa usando Otsu (tu método actual)"""
        # Extraer canal azul
        blue_channel = self.original_image[:, :, 2]
        
        # Filtrado Gaussiano
        blurred = cv2.GaussianBlur(blue_channel, (5, 5), 0)
        
        # Binarización de Otsu
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return closing
    
    def _get_main_contour(self):
        """Obtiene el contorno principal (más grande) de la máscara"""
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        # Retornar el contorno más grande
        return max(contours, key=cv2.contourArea)
    
    # ==================== DESCRIPTOR A: ASIMETRÍA ====================
    def calculate_asymmetry(self):
        """
        Calcula la asimetría de la lesión dividiendo por ejes principales
        
        Returns:
            dict: {
                'asymmetry_score': float (0-1, donde 0 es simétrico),
                'horizontal_asymmetry': float,
                'vertical_asymmetry': float
            }
        """
        contour = self._get_main_contour()
        if contour is None:
            return {'asymmetry_score': 0, 'horizontal_asymmetry': 0, 'vertical_asymmetry': 0}
        
        # Obtener momentos para encontrar el centroide
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return {'asymmetry_score': 0, 'horizontal_asymmetry': 0, 'vertical_asymmetry': 0}
            
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Crear máscaras para cada mitad
        h, w = self.mask.shape
        
        # Asimetría horizontal (dividir verticalmente por el centro)
        left_half = self.mask[:, :cx].copy()
        right_half = self.mask[:, cx:].copy()
        
        # Voltear la mitad derecha para comparar
        right_half_flipped = cv2.flip(right_half, 1)
        
        # Ajustar tamaños si son diferentes
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, -min_width:]
        right_half_flipped = right_half_flipped[:, :min_width]
        
        # Calcular solapamiento horizontal
        intersection_h = np.logical_and(left_half > 0, right_half_flipped > 0).sum()
        union_h = np.logical_or(left_half > 0, right_half_flipped > 0).sum()
        horizontal_asymmetry = 1 - (intersection_h / union_h if union_h > 0 else 0)
        
        # Asimetría vertical (dividir horizontalmente por el centro)
        top_half = self.mask[:cy, :].copy()
        bottom_half = self.mask[cy:, :].copy()
        
        # Voltear la mitad inferior para comparar
        bottom_half_flipped = cv2.flip(bottom_half, 0)
        
        # Ajustar tamaños si son diferentes
        min_height = min(top_half.shape[0], bottom_half_flipped.shape[0])
        top_half = top_half[-min_height:, :]
        bottom_half_flipped = bottom_half_flipped[:min_height, :]
        
        # Calcular solapamiento vertical
        intersection_v = np.logical_and(top_half > 0, bottom_half_flipped > 0).sum()
        union_v = np.logical_or(top_half > 0, bottom_half_flipped > 0).sum()
        vertical_asymmetry = 1 - (intersection_v / union_v if union_v > 0 else 0)
        
        # Puntuación final de asimetría (promedio de ambos ejes)
        asymmetry_score = (horizontal_asymmetry + vertical_asymmetry) / 2
        
        return {
            'asymmetry_score': round(asymmetry_score, 4),
            'horizontal_asymmetry': round(horizontal_asymmetry, 4),
            'vertical_asymmetry': round(vertical_asymmetry, 4)
        }
    
    # ==================== DESCRIPTOR B: BORDE (ya lo tienen) ====================
    def calculate_border(self):
        """
        Calcula la irregularidad del borde usando circularidad
        
        Returns:
            dict: {
                'circularity': float (1.0 = círculo perfecto),
                'area': float,
                'perimeter': float
            }
        """
        contour = self._get_main_contour()
        if contour is None:
            return {'circularity': 0, 'area': 0, 'perimeter': 0}
        
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            circularity = 0
        else:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        return {
            'circularity': round(circularity, 4),
            'area': round(area, 2),
            'perimeter': round(perimeter, 2)
        }
    
    # ==================== DESCRIPTOR C: COLOR ====================
    def calculate_color_variation(self):
        """
        Calcula la variación de color dentro de la lesión
        Utiliza el espacio de color Lab para mejor análisis
        
        Returns:
            dict: {
                'color_std_lab': float (desviación estándar en Lab),
                'num_colors': int (número de colores dominantes),
                'color_variance': float,
                'color_range': float
            }
        """
        # Convertir imagen a Lab
        lab_image = color.rgb2lab(self.original_image / 255.0)
        
        # Extraer solo los píxeles de la lesión
        lesion_pixels = lab_image[self.mask > 0]
        
        if len(lesion_pixels) == 0:
            return {
                'color_std_lab': 0,
                'num_colors': 0,
                'color_variance': 0,
                'color_range': 0
            }
        
        # Calcular desviación estándar en cada canal Lab
        l_std = np.std(lesion_pixels[:, 0])
        a_std = np.std(lesion_pixels[:, 1])
        b_std = np.std(lesion_pixels[:, 2])
        
        # Métrica combinada de variación de color
        color_std_lab = np.sqrt(l_std**2 + a_std**2 + b_std**2)
        
        # Varianza total
        color_variance = np.var(lesion_pixels)
        
        # Rango de colores (diferencia entre máximo y mínimo)
        color_range = np.ptp(lesion_pixels)  # peak to peak
        
        # Estimación de número de colores dominantes usando clustering simplificado
        # Cuantizar los valores para contar colores distintos
        quantized = np.round(lesion_pixels).astype(int)
        unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
        
        # Normalizar para tener una métrica más interpretable (0-1)
        # Mayor variación = más colores = más sospechoso
        num_colors_normalized = min(unique_colors / 100, 1.0)  # normalizado a 100 colores
        
        return {
            'color_std_lab': round(color_std_lab, 4),
            'num_colors': unique_colors,
            'color_variance': round(color_variance, 4),
            'color_range': round(color_range, 4),
            'color_score': round(num_colors_normalized, 4)  # Score de 0-1
        }
    
    # ==================== DESCRIPTOR D: DIÁMETRO (ya lo tienen) ====================
    def calculate_diameter(self):
        """
        Calcula el diámetro de la lesión
        
        Returns:
            dict: {
                'diameter_px': float (en píxeles),
                'diameter_mm': float (estimado, asumiendo escala)
            }
        """
        contour = self._get_main_contour()
        if contour is None:
            return {'diameter_px': 0, 'diameter_mm': 0}
        
        # Calcular diámetro como la raíz del área
        area = cv2.contourArea(contour)
        diameter_px = 2 * np.sqrt(area / np.pi)
        
        # Conversión aproximada a mm (ajustar según resolución conocida)
        # Asumiendo ~20 píxeles por mm en imágenes dermatoscópicas típicas
        pixels_per_mm = 20
        diameter_mm = diameter_px / pixels_per_mm
        
        return {
            'diameter_px': round(diameter_px, 2),
            'diameter_mm': round(diameter_mm, 2)
        }
    
    # ==================== CALCULAR TODOS LOS DESCRIPTORES ====================
    def calculate_all_descriptors(self):
        """
        Calcula todos los descriptores ABCD
        
        Returns:
            dict: Diccionario completo con todos los descriptores
        """
        self.descriptors = {
            'A_asymmetry': self.calculate_asymmetry(),
            'B_border': self.calculate_border(),
            'C_color': self.calculate_color_variation(),
            'D_diameter': self.calculate_diameter()
        }
        return self.descriptors
    
    # ==================== MÉTRICAS DE EVALUACIÓN: IoU y DICE ====================
    def calculate_iou(self, ground_truth_mask):
        """
        Calcula Intersection over Union (IoU)
        
        Args:
            ground_truth_mask: máscara de referencia (ground truth)
            
        Returns:
            float: IoU score (0-1)
        """
        # Asegurar que ambas máscaras sean binarias
        pred_mask = (self.mask > 0).astype(np.uint8)
        gt_mask = (ground_truth_mask > 0).astype(np.uint8)
        
        # Calcular intersección y unión
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            return 0.0
        
        iou = intersection / union
        return round(iou, 4)
    
    def calculate_dice(self, ground_truth_mask):
        """
        Calcula el coeficiente de Dice (F1-score para segmentación)
        
        Args:
            ground_truth_mask: máscara de referencia (ground truth)
            
        Returns:
            float: Dice coefficient (0-1)
        """
        # Asegurar que ambas máscaras sean binarias
        pred_mask = (self.mask > 0).astype(np.uint8)
        gt_mask = (ground_truth_mask > 0).astype(np.uint8)
        
        # Calcular intersección
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        
        # Calcular suma de ambas máscaras
        pred_sum = pred_mask.sum()
        gt_sum = gt_mask.sum()
        
        if (pred_sum + gt_sum) == 0:
            return 0.0
        
        dice = (2 * intersection) / (pred_sum + gt_sum)
        return round(dice, 4)
    
    def evaluate_segmentation(self, ground_truth_mask):
        """
        Evalúa la segmentación usando ambas métricas
        
        Args:
            ground_truth_mask: máscara de referencia (ground truth)
            
        Returns:
            dict: {'iou': float, 'dice': float}
        """
        return {
            'iou': self.calculate_iou(ground_truth_mask),
            'dice': self.calculate_dice(ground_truth_mask)
        }
    
    # ==================== VISUALIZACIÓN ====================
    def visualize_results(self, save_path=None):
        """
        Visualiza la imagen original, máscara y resultados
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Imagen original
        axes[0, 0].imshow(self.original_image)
        axes[0, 0].set_title('Imagen Original')
        axes[0, 0].axis('off')
        
        # Máscara de segmentación
        axes[0, 1].imshow(self.mask, cmap='gray')
        axes[0, 1].set_title('Máscara de Segmentación')
        axes[0, 1].axis('off')
        
        # Superposición
        overlay = self.original_image.copy()
        overlay[self.mask > 0] = overlay[self.mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
        axes[0, 2].imshow(overlay.astype(np.uint8))
        axes[0, 2].set_title('Superposición')
        axes[0, 2].axis('off')
        
        # Análisis de asimetría
        contour = self._get_main_contour()
        if contour is not None:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                asym_img = self.original_image.copy()
                cv2.line(asym_img, (cx, 0), (cx, asym_img.shape[0]), (255, 0, 0), 2)
                cv2.line(asym_img, (0, cy), (asym_img.shape[1], cy), (255, 0, 0), 2)
                axes[1, 0].imshow(asym_img)
                axes[1, 0].set_title('Ejes de Asimetría')
                axes[1, 0].axis('off')
        
        # Visualización de color
        lab_image = color.rgb2lab(self.original_image / 255.0)
        lesion_mask_3d = np.stack([self.mask > 0] * 3, axis=-1)
        color_viz = lab_image.copy()
        color_viz[~lesion_mask_3d] = 0
        axes[1, 1].imshow(color_viz[:,:,0], cmap='hot')
        axes[1, 1].set_title('Variación de Color (Canal L)')
        axes[1, 1].axis('off')
        
        # Texto con descriptores
        if self.descriptors:
            desc_text = "Descriptores ABCD:\n\n"
            desc_text += f"A - Asimetría: {self.descriptors['A_asymmetry']['asymmetry_score']:.3f}\n"
            desc_text += f"B - Borde (Circ.): {self.descriptors['B_border']['circularity']:.3f}\n"
            desc_text += f"C - Color (Var.): {self.descriptors['C_color']['color_std_lab']:.3f}\n"
            desc_text += f"D - Diámetro: {self.descriptors['D_diameter']['diameter_px']:.2f} px\n"
            desc_text += f"            {self.descriptors['D_diameter']['diameter_mm']:.2f} mm\n"
            
            axes[1, 2].text(0.1, 0.5, desc_text, fontsize=12, family='monospace',
                           verticalalignment='center')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_summary_table(self):
        """
        Retorna un resumen en formato de tabla para el informe
        """
        if not self.descriptors:
            self.calculate_all_descriptors()
        
        summary = []
        summary.append(["Descriptor", "Métrica", "Valor"])
        summary.append(["A - Asimetría", "Score Total", f"{self.descriptors['A_asymmetry']['asymmetry_score']:.4f}"])
        summary.append(["", "Asimetría Horizontal", f"{self.descriptors['A_asymmetry']['horizontal_asymmetry']:.4f}"])
        summary.append(["", "Asimetría Vertical", f"{self.descriptors['A_asymmetry']['vertical_asymmetry']:.4f}"])
        summary.append(["B - Borde", "Circularidad", f"{self.descriptors['B_border']['circularity']:.4f}"])
        summary.append(["", "Área (px²)", f"{self.descriptors['B_border']['area']:.2f}"])
        summary.append(["", "Perímetro (px)", f"{self.descriptors['B_border']['perimeter']:.2f}"])
        summary.append(["C - Color", "Std Lab", f"{self.descriptors['C_color']['color_std_lab']:.4f}"])
        summary.append(["", "Núm. Colores", f"{self.descriptors['C_color']['num_colors']}"])
        summary.append(["", "Varianza", f"{self.descriptors['C_color']['color_variance']:.4f}"])
        summary.append(["D - Diámetro", "Píxeles", f"{self.descriptors['D_diameter']['diameter_px']:.2f}"])
        summary.append(["", "Milímetros", f"{self.descriptors['D_diameter']['diameter_mm']:.2f}"])
        
        return summary


# ==================== EJEMPLO DE USO ====================
if __name__ == "__main__":
    # Ejemplo 1: Procesar una imagen
    print("="*60)
    print("EJEMPLO: Procesamiento de imagen de melanoma")
    print("="*60)
    
    # Cargar imagen (ajusta la ruta)
    image_path = "path/to/your/image.jpg"
    
    # Crear analizador
    analyzer = MelanomaDescriptors(image_path)
    
    # Calcular todos los descriptores
    descriptors = analyzer.calculate_all_descriptors()
    
    # Mostrar resultados
    print("\n📊 DESCRIPTORES ABCD:")
    print("\nA - ASIMETRÍA:")
    for key, value in descriptors['A_asymmetry'].items():
        print(f"  {key}: {value}")
    
    print("\nB - BORDE:")
    for key, value in descriptors['B_border'].items():
        print(f"  {key}: {value}")
    
    print("\nC - COLOR:")
    for key, value in descriptors['C_color'].items():
        print(f"  {key}: {value}")
    
    print("\nD - DIÁMETRO:")
    for key, value in descriptors['D_diameter'].items():
        print(f"  {key}: {value}")
    
    # Visualizar resultados
    analyzer.visualize_results(save_path="resultado_analisis.png")
    
    # Si tienes ground truth para evaluar
    # ground_truth = cv2.imread("path/to/ground_truth_mask.png", 0)
    # metrics = analyzer.evaluate_segmentation(ground_truth)
    # print("\n📈 MÉTRICAS DE EVALUACIÓN:")
    # print(f"  IoU: {metrics['iou']}")
    # print(f"  Dice: {metrics['dice']}")
    
    # Obtener tabla resumen
    print("\n📋 TABLA RESUMEN:")
    table = analyzer.get_summary_table()
    for row in table:
        print(f"{row[0]:20} | {row[1]:25} | {row[2]:>10}")