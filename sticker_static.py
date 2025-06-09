import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io
from typing import Tuple, List
import cv2

class NSFWStaticStickerDetector:
    def __init__(self):
        # رنج‌های مختلف رنگ پوست
        self.skin_tones = [
            np.array([255, 219, 172]),  # روشن
            np.array([241, 194, 125]),  # متوسط روشن
            np.array([224, 172, 105]),  # متوسط
            np.array([198, 134, 66]),   # متوسط تیره
            np.array([141, 85, 36]),    # تیره
            np.array([92, 51, 23]),     # خیلی تیره
        ]
        
        # محدوده رنگ‌های مشکوک (رنگ‌های صورتی/قرمز برای اندام‌های حساس)
        self.suspicious_colors = [
            (np.array([255, 182, 193]), 40),  # صورتی روشن
            (np.array([255, 105, 180]), 35),  # صورتی داغ
            (np.array([255, 20, 147]), 30),   # صورتی عمیق
            (np.array([220, 20, 60]), 25),    # قرمز
        ]
        
        # آستانه‌های تشخیص
        self.skin_threshold = 0.25
        self.suspicious_threshold = 0.08
        self.skin_cluster_threshold = 0.15
        
    def preprocess_image(self, image: Image.Image) -> Tuple[np.ndarray, Image.Image]:
        """پیش‌پردازش تصویر برای تشخیص بهتر"""
        # تبدیل به RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # افزایش کنتراست برای تشخیص بهتر رنگ‌ها
        enhancer = ImageEnhance.Contrast(image)
        enhanced_image = enhancer.enhance(1.2)
        
        # کاهش نویز
        smoothed_image = enhanced_image.filter(ImageFilter.MedianFilter(size=3))
        
        return np.array(smoothed_image), smoothed_image
    
    def detect_skin_advanced(self, image_array: np.ndarray) -> float:
        """تشخیص پیشرفته رنگ پوست با در نظر گیری رنج‌های مختلف"""
        skin_mask = np.zeros(image_array.shape[:2], dtype=bool)
        
        # بررسی هر رنگ پوست
        for skin_tone in self.skin_tones:
            # محاسبه فاصله رنگی
            distances = np.sqrt(np.sum((image_array - skin_tone) ** 2, axis=2))
            
            # آستانه تطبیقی بر اساس روشنی رنگ پوست
            brightness = np.mean(skin_tone)
            if brightness > 200:
                threshold = 60  # رنگ‌های روشن
            elif brightness > 150:
                threshold = 50  # رنگ‌های متوسط
            else:
                threshold = 40  # رنگ‌های تیره
                
            skin_mask |= distances < threshold
        
        # تشخیص بر اساس HSV نیز
        hsv_mask = self._detect_skin_hsv(image_array)
        skin_mask |= hsv_mask
        
        return np.sum(skin_mask) / skin_mask.size
    
    def _detect_skin_hsv(self, image_array: np.ndarray) -> np.ndarray:
        """تشخیص رنگ پوست در فضای رنگی HSV"""
        # تبدیل به HSV
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # محدوده‌های HSV برای رنگ پوست
        lower_skin1 = np.array([0, 20, 70])
        upper_skin1 = np.array([20, 255, 255])
        
        lower_skin2 = np.array([0, 20, 70])
        upper_skin2 = np.array([255, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        return (mask1 | mask2) > 0
    
    def detect_suspicious_colors(self, image_array: np.ndarray) -> float:
        """تشخیص رنگ‌های مشکوک"""
        suspicious_pixels = 0
        total_pixels = image_array.shape[0] * image_array.shape[1]
        
        for color, threshold in self.suspicious_colors:
            distances = np.sqrt(np.sum((image_array - color) ** 2, axis=2))
            suspicious_pixels += np.sum(distances < threshold)
            
        return suspicious_pixels / total_pixels
    
    def analyze_skin_clusters(self, image_array: np.ndarray) -> float:
        """تحلیل خوشه‌بندی نواحی پوستی"""
        skin_mask = np.zeros(image_array.shape[:2], dtype=bool)
        
        # ایجاد ماسک کل پوست
        for skin_tone in self.skin_tones:
            distances = np.sqrt(np.sum((image_array - skin_tone) ** 2, axis=2))
            skin_mask |= distances < 55
            
        # تبدیل به uint8 برای opencv
        skin_mask_uint8 = skin_mask.astype(np.uint8) * 255
        
        # پیدا کردن contourها
        contours, _ = cv2.findContours(skin_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
            
        # محاسبه نسبت بزرگترین ناحیه پوستی
        largest_area = max(cv2.contourArea(contour) for contour in contours)
        total_area = image_array.shape[0] * image_array.shape[1]
        
        return largest_area / total_area
    
    def detect_body_shapes(self, image_array: np.ndarray) -> float:
        """تشخیص اشکال مشکوک بدن"""
        # تبدیل به grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # تشخیص لبه‌ها
        edges = cv2.Canny(gray, 50, 150)
        
        # تشخیص دایره‌ها (ممکن است نشان‌دهنده اعضای بدن باشد)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=5, maxRadius=50)
        
        circle_score = 0
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            circle_score = len(circles) / 10  # نرمال‌سازی
            
        return min(circle_score, 1.0)
    
    def calculate_texture_features(self, image_array: np.ndarray) -> float:
        """محاسبه ویژگی‌های بافت تصویر"""
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # محاسبه واریانس محلی (نشان‌دهنده پیچیدگی بافت)
        kernel = np.ones((9, 9), np.float32) / 81
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        variance = cv2.filter2D((gray.astype(np.float32) - mean) ** 2, -1, kernel)
        
        # نرمال‌سازی
        texture_complexity = np.mean(variance) / 10000
        
        return min(texture_complexity, 1.0)
    
    def is_nsfw(self, sticker_bytes: bytes) -> bool:
        """
        تشخیص NSFW با الگوریتم پیشرفته
        ترکیب چندین معیار برای تشخیص دقیق‌تر
        """
        try:
            image = Image.open(io.BytesIO(sticker_bytes))
            image_array, processed_image = self.preprocess_image(image)
            
            # معیارهای مختلف تشخیص
            skin_percentage = self.detect_skin_advanced(image_array)
            suspicious_colors = self.detect_suspicious_colors(image_array)
            skin_cluster_ratio = self.analyze_skin_clusters(image_array)
            body_shapes = self.detect_body_shapes(image_array)
            texture_complexity = self.calculate_texture_features(image_array)
            
            # محاسبه امتیاز نهایی با وزن‌دهی
            weights = {
                'skin': 0.35,
                'suspicious': 0.25,
                'cluster': 0.20,
                'shapes': 0.10,
                'texture': 0.10
            }
            
            final_score = (
                skin_percentage * weights['skin'] +
                suspicious_colors * weights['suspicious'] +
                skin_cluster_ratio * weights['cluster'] +
                body_shapes * weights['shapes'] +
                texture_complexity * weights['texture']
            )
            
            # تشخیص نهایی
            is_nsfw = (
                skin_percentage > self.skin_threshold or
                suspicious_colors > self.suspicious_threshold or
                skin_cluster_ratio > self.skin_cluster_threshold or
                final_score > 0.3
            )
            
            return is_nsfw
            
        except Exception as e:
            # در صورت خطا، محافظه‌کارانه False برمی‌گرداند
            print(f"Error in NSFW detection: {e}")
            return False
    
    def get_detailed_analysis(self, sticker_bytes: bytes) -> dict:
        """تحلیل جزئی برای دیباگ و بهینه‌سازی"""
        try:
            image = Image.open(io.BytesIO(sticker_bytes))
            image_array, processed_image = self.preprocess_image(image)
            
            analysis = {
                'skin_percentage': self.detect_skin_advanced(image_array),
                'suspicious_colors': self.detect_suspicious_colors(image_array),
                'skin_cluster_ratio': self.analyze_skin_clusters(image_array),
                'body_shapes': self.detect_body_shapes(image_array),
                'texture_complexity': self.calculate_texture_features(image_array),
                'is_nsfw': self.is_nsfw(sticker_bytes)
            }
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}

# نمونه استفاده:
# detector = NSFWStaticStickerDetector()
# 
# # تشخیص ساده
# with open('sticker.webp', 'rb') as f:
#     result = detector.is_nsfw(f.read())
#     print(f"NSFW: {result}")
# 
# # تحلیل جزئی
# with open('sticker.webp', 'rb') as f:
#     analysis = detector.get_detailed_analysis(f.read())
#     print(f"Analysis: {analysis}")
