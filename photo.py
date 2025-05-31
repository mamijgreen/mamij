import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io
import cv2

class NSFWDetector:
    def __init__(self):
        # تعریف رنج‌های مختلف رنگ پوست برای نژادهای مختلف
        self.skin_ranges = [
            ([255, 219, 172], [255, 255, 255]),  # پوست روشن  
            ([241, 194, 125], [255, 219, 172]),  # پوست متوسط روشن
            ([198, 134, 66], [241, 194, 125]),   # پوست متوسط
            ([141, 85, 36], [198, 134, 66]),     # پوست تیره
            ([83, 45, 20], [141, 85, 36])        # پوست خیلی تیره
        ]
        
        # رنگ‌های مو
        self.hair_colors = [
            ([0, 0, 0], [50, 50, 50]),          # مو مشکی
            ([101, 67, 33], [165, 107, 70]),    # مو قهوه‌ای
            ([255, 255, 0], [255, 255, 100]),   # مو بلوند
            ([165, 42, 42], [200, 100, 100])    # مو قرمز
        ]
        
        # رنگ‌های عمومی لباس
        self.clothing_colors = [
            ([0, 0, 0], [100, 100, 100]),      # تیره (مشکی/خاکستری)
            ([0, 0, 100], [100, 100, 255]),    # آبی
            ([100, 0, 0], [255, 100, 100]),    # قرمز
            ([0, 100, 0], [100, 255, 100]),    # سبز
            ([200, 200, 200], [255, 255, 255]) # سفید/روشن
        ]

    def detect_skin_regions(self, image_array):
        """تشخیص نواحی پوست با دقت بالا"""
        skin_mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
        
        # تبدیل به فضای رنگی HSV برای تشخیص بهتر پوست
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # رنج HSV برای پوست انسان
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_hsv = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # تبدیل به YCrCb برای دقت بیشتر
        ycrcb = cv2.cvtColor(image_array, cv2.COLOR_RGB2YCrCb)
        lower_skin_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        upper_skin_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        skin_ycrcb = cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)
        
        # ترکیب دو روش
        skin_mask = cv2.bitwise_and(skin_hsv, skin_ycrcb)
        
        # حذف نویز
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        return skin_mask

    def detect_edges_and_shapes(self, image_array):
        """تشخیص اشکال مشکوک با تحلیل لبه‌ها"""
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # تشخیص لبه‌ها
        edges = cv2.Canny(gray, 50, 150)
        
        # پیدا کردن کانتورها
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        suspicious_shapes = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # فقط اشکال بزرگ
                # محاسبه نسبت طول به عرض
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                # بررسی اشکال مشکوک (مثل بدن انسان)
                if 0.3 < aspect_ratio < 3.0 and area > 5000:
                    suspicious_shapes += 1
        
        return suspicious_shapes

    def analyze_color_distribution(self, image_array):
        """تحلیل توزیع رنگ در تصویر"""
        # محاسبه هیستوگرام رنگ
        hist_r = cv2.calcHist([image_array], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image_array], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([image_array], [2], None, [256], [0, 256])
        
        # بررسی غالب بودن رنگ‌های پوست
        skin_dominance = 0
        for i in range(150, 255):  # رنج رنگ‌های پوست
            skin_dominance += (hist_r[i] + hist_g[i] + hist_b[i])[0]
        
        total_pixels = image_array.shape[0] * image_array.shape[1]
        skin_ratio = skin_dominance / (total_pixels * 3)
        
        return skin_ratio

    def detect_clothing_ratio(self, image_array):
        """تشخیص نسبت لباس در تصویر"""
        clothing_pixels = 0
        total_pixels = image_array.shape[0] * image_array.shape[1]
        
        for lower, upper in self.clothing_colors:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(image_array, lower, upper)
            clothing_pixels += np.sum(mask > 0)
        
        clothing_ratio = clothing_pixels / total_pixels
        return clothing_ratio

    def check_image_quality_indicators(self, image_array):
        """بررسی شاخص‌های کیفیت تصویر که ممکن است نشان‌دهنده محتوای نامناسب باشد"""
        # بررسی تاری (blur) - تصاویر نامناسب غالباً تار هستند
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # بررسی کنتراست
        contrast = gray.std()
        
        # بررسی روشنایی متوسط
        brightness = gray.mean()
        
        return {
            'blur': blur_value,
            'contrast': contrast,
            'brightness': brightness
        }

    def is_nsfw(self, image_bytes):
        """
        تشخیص هوشمند تصاویر نامناسب با استفاده از چندین الگوریتم
        """
        try:
            # باز کردن تصویر
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # بررسی حداقل سایز تصویر
            if image.size[0] < 100 or image.size[1] < 100:
                return False  # تصاویر خیلی کوچک معمولاً ایموجی یا آیکون هستند
            
            image_array = np.array(image)
            
            # 1. تشخیص نواحی پوست پیشرفته
            skin_mask = self.detect_skin_regions(image_array)
            skin_percentage = np.sum(skin_mask > 0) / (image_array.shape[0] * image_array.shape[1])
            
            # 2. تحلیل توزیع رنگ
            color_analysis = self.analyze_color_distribution(image_array)
            
            # 3. تشخیص نسبت لباس
            clothing_ratio = self.detect_clothing_ratio(image_array)
            
            # 4. تشخیص اشکال مشکوک
            suspicious_shapes = self.detect_edges_and_shapes(image_array)
            
            # 5. بررسی کیفیت تصویر
            quality_indicators = self.check_image_quality_indicators(image_array)
            
            # محاسبه امتیاز نهایی
            risk_score = 0
            
            # امتیازدهی بر اساس پوست
            if skin_percentage > 0.4:
                risk_score += 40
            elif skin_percentage > 0.25:
                risk_score += 25
            elif skin_percentage > 0.15:
                risk_score += 10
            
            # امتیازدهی بر اساس کمبود لباس
            if clothing_ratio < 0.1:
                risk_score += 35
            elif clothing_ratio < 0.2:
                risk_score += 20
            
            # امتیازدهی بر اساس اشکال مشکوک
            if suspicious_shapes > 3:
                risk_score += 20
            elif suspicious_shapes > 1:
                risk_score += 10
            
            # امتیازدهی بر اساس کیفیت (تصاویر نامناسب اغلب کیفیت پایین دارند)
            if quality_indicators['blur'] < 100:  # خیلی تار
                risk_score += 15
            if quality_indicators['contrast'] < 30:  # کنتراست پایین
                risk_score += 10
            if 50 < quality_indicators['brightness'] < 200:  # روشنایی مشکوک
                risk_score += 5
            
            # تنظیم حساسیت بر اساس اندازه تصویر
            # تصاویر بزرگتر نیاز به حساسیت کمتری دارند
            image_size_factor = min(image.size[0] * image.size[1] / (500 * 500), 1.0)
            threshold = 55 + (10 * image_size_factor)
            
            # گزارش تفصیلی برای دیباگ
            print(f"تحلیل تصویر:")
            print(f"- درصد پوست: {skin_percentage:.3f}")
            print(f"- نسبت لباس: {clothing_ratio:.3f}")
            print(f"- اشکال مشکوک: {suspicious_shapes}")
            print(f"- تاری: {quality_indicators['blur']:.2f}")
            print(f"- کنتراست: {quality_indicators['contrast']:.2f}")
            print(f"- روشنایی: {quality_indicators['brightness']:.2f}")
            print(f"- امتیاز ریسک: {risk_score}")
            print(f"- آستانه: {threshold}")
            
            return risk_score > threshold
            
        except Exception as e:
            print(f"خطا در تحلیل تصویر: {e}")
            return False

    def get_detailed_analysis(self, image_bytes):
        """گزارش تفصیلی از تحلیل تصویر"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_array = np.array(image)
            
            skin_mask = self.detect_skin_regions(image_array)
            skin_percentage = np.sum(skin_mask > 0) / (image_array.shape[0] * image_array.shape[1])
            
            return {
                'skin_percentage': skin_percentage,
                'clothing_ratio': self.detect_clothing_ratio(image_array),
                'suspicious_shapes': self.detect_edges_and_shapes(image_array),
                'quality': self.check_image_quality_indicators(image_array),
                'is_nsfw': self.is_nsfw(image_bytes)
            }
        except Exception as e:
            return {'error': str(e)}
