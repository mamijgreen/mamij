import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageStat
import io
from typing import Tuple, List, Dict
import cv2
from collections import Counter

class NSFWDetector:
    def __init__(self):
        # طیف گسترده‌ای از رنگ‌های پوست
        self.skin_tones = [
            ([255, 228, 196], 45),  # پوست بسیار روشن
            ([255, 219, 172], 50),  # پوست روشن
            ([241, 194, 125], 55),  # پوست متوسط روشن
            ([224, 172, 105], 60),  # پوست متوسط
            ([198, 134, 66], 50),   # پوست متوسط تیره
            ([141, 85, 36], 45),    # پوست تیره
            ([92, 51, 23], 40),     # پوست خیلی تیره
            ([70, 35, 18], 35),     # پوست بسیار تیره
        ]
        
        # رنگ‌های مشکوک (اعضای حساس، لب‌ها، ...)
        self.sensitive_colors = [
            ([255, 182, 193], 35),  # صورتی ملایم
            ([255, 160, 180], 40),  # صورتی متوسط
            ([255, 105, 180], 45),  # صورتی پر رنگ
            ([255, 20, 147], 35),   # صورتی عمیق
            ([220, 20, 60], 30),    # قرمز کریمسون
            ([178, 34, 34], 25),    # قرمز آتشین
            ([255, 69, 0], 28),     # قرمز نارنجی
            ([255, 99, 71], 32),    # گوجه‌ای
        ]
        
        # آستانه‌های تشخیص
        self.thresholds = {
            'skin_basic': 0.22,
            'skin_clustered': 0.12,
            'sensitive_colors': 0.06,
            'combined_score': 0.28,
            'high_confidence': 0.45
        }
    
    def preprocess_image(self, image: Image.Image) -> Tuple[np.ndarray, Dict]:
        """پیش‌پردازش و آنالیز اولیه تصویر"""
        # تبدیل به RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # محاسبه آمار اولیه
        stats = ImageStat.Stat(image)
        brightness = sum(stats.mean) / 3
        contrast = sum(stats.stddev) / 3
        
        # بهبود کیفیت برای تشخیص بهتر
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(1.3)
        
        # کاهش نویز
        smoothed = enhanced.filter(ImageFilter.MedianFilter(size=3))
        
        image_array = np.array(smoothed)
        
        info = {
            'brightness': brightness,
            'contrast': contrast,
            'size': image.size,
            'aspect_ratio': image.size[0] / image.size[1]
        }
        
        return image_array, info
    
    def detect_skin_pixels(self, image_array: np.ndarray) -> Tuple[float, np.ndarray]:
        """تشخیص پیکسل‌های پوستی با دقت بالا"""
        skin_mask = np.zeros(image_array.shape[:2], dtype=bool)
        
        for skin_tone, threshold in self.skin_tones:
            skin_color = np.array(skin_tone)
            
            # محاسبه فاصله اقلیدسی
            distances = np.sqrt(np.sum((image_array - skin_color) ** 2, axis=2))
            current_mask = distances < threshold
            skin_mask |= current_mask
        
        # تشخیص اضافی با HSV
        hsv_mask = self._detect_skin_hsv(image_array)
        skin_mask |= hsv_mask
        
        # تشخیص اضافی با YCbCr
        ycbcr_mask = self._detect_skin_ycbcr(image_array)
        skin_mask |= ycbcr_mask
        
        skin_percentage = np.sum(skin_mask) / skin_mask.size
        
        return skin_percentage, skin_mask
    
    def _detect_skin_hsv(self, image_array: np.ndarray) -> np.ndarray:
        """تشخیص پوست در فضای HSV"""
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # محدوده‌های HSV برای پوست
        lower1 = np.array([0, 10, 60])
        upper1 = np.array([20, 150, 255])
        
        lower2 = np.array([0, 10, 60])
        upper2 = np.array([25, 180, 255])
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        
        return (mask1 | mask2) > 0
    
    def _detect_skin_ycbcr(self, image_array: np.ndarray) -> np.ndarray:
        """تشخیص پوست در فضای YCbCr"""
        ycbcr = cv2.cvtColor(image_array, cv2.COLOR_RGB2YCrCb)
        
        # محدوده‌های YCbCr برای پوست
        lower = np.array([0, 133, 77])
        upper = np.array([255, 173, 127])
        
        mask = cv2.inRange(ycbcr, lower, upper)
        return mask > 0
    
    def analyze_skin_distribution(self, skin_mask: np.ndarray) -> Dict:
        """تحلیل توزیع نواحی پوستی"""
        # تبدیل به uint8 برای opencv
        skin_uint8 = skin_mask.astype(np.uint8) * 255
        
        # پیدا کردن contourها
        contours, _ = cv2.findContours(skin_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'largest_region': 0, 'num_regions': 0, 'concentration': 0}
        
        # محاسبه مساحت‌ها
        areas = [cv2.contourArea(contour) for contour in contours]
        total_area = skin_uint8.shape[0] * skin_uint8.shape[1]
        
        largest_region = max(areas) / total_area if areas else 0
        num_regions = len([area for area in areas if area > 100])
        
        # محاسبه تمرکز (چقدر نواحی پوستی متصل هستند)
        concentration = largest_region / (np.sum(skin_mask) / total_area) if np.sum(skin_mask) > 0 else 0
        
        return {
            'largest_region': largest_region,
            'num_regions': num_regions,
            'concentration': min(concentration, 1.0)
        }
    
    def detect_sensitive_colors(self, image_array: np.ndarray) -> float:
        """تشخیص رنگ‌های حساس و مشکوک"""
        sensitive_pixels = 0
        total_pixels = image_array.shape[0] * image_array.shape[1]
        
        for color, threshold in self.sensitive_colors:
            target_color = np.array(color)
            distances = np.sqrt(np.sum((image_array - target_color) ** 2, axis=2))
            sensitive_pixels += np.sum(distances < threshold)
        
        return sensitive_pixels / total_pixels
    
    def analyze_color_distribution(self, image_array: np.ndarray) -> Dict:
        """تحلیل توزیع رنگ‌ها"""
        # تبدیل به فرمت مناسب برای تحلیل
        pixels = image_array.reshape(-1, 3)
        
        # محاسبه میانگین رنگ‌ها
        mean_color = np.mean(pixels, axis=0)
        
        # محاسبه انحراف معیار
        color_std = np.std(pixels, axis=0)
        
        # تشخیص رنگ‌های غالب
        dominant_colors = self._get_dominant_colors(pixels, k=5)
        
        return {
            'mean_color': mean_color,
            'color_diversity': np.mean(color_std),
            'dominant_colors': dominant_colors
        }
    
    def _get_dominant_colors(self, pixels: np.ndarray, k: int = 5) -> List:
        """استخراج رنگ‌های غالب با K-means"""
        from sklearn.cluster import KMeans
        
        try:
            # نمونه‌برداری برای سرعت بیشتر
            sample_size = min(10000, len(pixels))
            sample_pixels = pixels[np.random.choice(len(pixels), sample_size, replace=False)]
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(sample_pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            return colors.tolist()
        except:
            # اگر sklearn در دسترس نباشد، روش ساده
            return []
    
    def detect_body_shapes(self, image_array: np.ndarray, skin_mask: np.ndarray) -> float:
        """تشخیص اشکال مشکوک بدن"""
        # تبدیل به grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # اعمال ماسک پوست
        masked_gray = cv2.bitwise_and(gray, gray, mask=skin_mask.astype(np.uint8) * 255)
        
        # تشخیص لبه‌ها
        edges = cv2.Canny(masked_gray, 30, 100)
        
        # تشخیص دایره‌ها
        circles = cv2.HoughCircles(
            masked_gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        
        circle_score = 0
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # وزن‌دهی بر اساس اندازه و موقعیت دایره‌ها
            for (x, y, r) in circles:
                if r > 15:  # دایره‌های بزرگ مشکوک‌تر
                    circle_score += r / 50
        
        return min(circle_score, 1.0)
    
    def calculate_texture_analysis(self, image_array: np.ndarray) -> Dict:
        """تحلیل بافت و پیچیدگی تصویر"""
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # محاسبه گرادیان
        grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # محاسبه انرژی بافت
        texture_energy = np.mean(gradient_magnitude)
        
        # محاسبه یکنواختی
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        uniformity = np.sum(hist**2) / (gray.shape[0] * gray.shape[1])**2
        
        return {
            'texture_energy': texture_energy / 100,  # نرمال‌سازی
            'uniformity': uniformity,
            'complexity': min(texture_energy / 50, 1.0)
        }
    
    def calculate_risk_score(self, features: Dict) -> float:
        """محاسبه امتیاز ریسک نهایی"""
        # وزن‌های مختلف برای هر ویژگی
        weights = {
            'skin_percentage': 0.30,
            'skin_concentration': 0.20,
            'sensitive_colors': 0.25,
            'body_shapes': 0.10,
            'texture_complexity': 0.08,
            'color_diversity': 0.07
        }
        
        # محاسبه امتیاز وزن‌دار
        score = 0
        for feature, weight in weights.items():
            if feature in features:
                score += features[feature] * weight
        
        # جریمه برای تصاویر با نسبت ابعاد خاص
        if 'aspect_ratio' in features:
            ratio = features['aspect_ratio']
            if 0.7 < ratio < 1.4:  # نسبت‌های مربعی مشکوک‌تر
                score *= 1.1
        
        return min(score, 1.0)
    
    def is_nsfw(self, image_bytes: bytes) -> bool:
        """
        تشخیص NSFW با الگوریتم پیشرفته و چندلایه
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_array, image_info = self.preprocess_image(image)
            
            # تشخیص پوست
            skin_percentage, skin_mask = self.detect_skin_pixels(image_array)
            
            # تحلیل توزیع پوست
            skin_distribution = self.analyze_skin_distribution(skin_mask)
            
            # تشخیص رنگ‌های حساس
            sensitive_colors = self.detect_sensitive_colors(image_array)
            
            # تحلیل رنگ‌ها
            color_analysis = self.analyze_color_distribution(image_array)
            
            # تشخیص اشکال
            body_shapes = self.detect_body_shapes(image_array, skin_mask)
            
            # تحلیل بافت
            texture_analysis = self.calculate_texture_analysis(image_array)
            
            # ترکیب تمام ویژگی‌ها
            features = {
                'skin_percentage': skin_percentage,
                'skin_concentration': skin_distribution['concentration'],
                'sensitive_colors': sensitive_colors,
                'body_shapes': body_shapes,
                'texture_complexity': texture_analysis['complexity'],
                'color_diversity': color_analysis['color_diversity'] / 100,
                'aspect_ratio': image_info['aspect_ratio']
            }
            
            # محاسبه امتیاز نهایی
            risk_score = self.calculate_risk_score(features)
            
            # تصمیم‌گیری نهایی
            is_nsfw = (
                skin_percentage > self.thresholds['skin_basic'] or
                skin_distribution['concentration'] > self.thresholds['skin_clustered'] or
                sensitive_colors > self.thresholds['sensitive_colors'] or
                risk_score > self.thresholds['combined_score']
            )
            
            # اعتماد بالا برای موارد واضح
            high_confidence = risk_score > self.thresholds['high_confidence']
            
            return is_nsfw or high_confidence
            
        except Exception as e:
            # در صورت خطا، محافظه‌کارانه False
            print(f"خطا در تشخیص NSFW: {e}")
            return False
    
    def get_detailed_analysis(self, image_bytes: bytes) -> Dict:
        """تحلیل کامل و جزئی برای بررسی و تنظیم"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_array, image_info = self.preprocess_image(image)
            
            # تمام تحلیل‌ها
            skin_percentage, skin_mask = self.detect_skin_pixels(image_array)
            skin_distribution = self.analyze_skin_distribution(skin_mask)
            sensitive_colors = self.detect_sensitive_colors(image_array)
            color_analysis = self.analyze_color_distribution(image_array)
            body_shapes = self.detect_body_shapes(image_array, skin_mask)
            texture_analysis = self.calculate_texture_analysis(image_array)
            
            features = {
                'skin_percentage': skin_percentage,
                'skin_concentration': skin_distribution['concentration'],
                'sensitive_colors': sensitive_colors,
                'body_shapes': body_shapes,
                'texture_complexity': texture_analysis['complexity'],
                'color_diversity': color_analysis['color_diversity'] / 100,
                'aspect_ratio': image_info['aspect_ratio']
            }
            
            risk_score = self.calculate_risk_score(features)
            
            return {
                'image_info': image_info,
                'skin_analysis': {
                    'percentage': skin_percentage,
                    'distribution': skin_distribution
                },
                'color_analysis': color_analysis,
                'sensitive_colors': sensitive_colors,
                'body_shapes': body_shapes,
                'texture_analysis': texture_analysis,
                'risk_score': risk_score,
                'is_nsfw': self.is_nsfw(image_bytes),
                'features': features
            }
            
        except Exception as e:
            return {'error': str(e)}

# مثال استفاده:
# detector = NSFWDetector()
# 
# # تشخیص ساده
# with open('image.jpg', 'rb') as f:
#     result = detector.is_nsfw(f.read())
#     print(f"NSFW: {result}")
# 
# # تحلیل کامل
# with open('image.jpg', 'rb') as f:
#     analysis = detector.get_detailed_analysis(f.read())
#     print(f"امتیاز ریسک: {analysis['risk_score']:.2f}")
#     print(f"درصد پوست: {analysis['skin_analysis']['percentage']:.2f}")
