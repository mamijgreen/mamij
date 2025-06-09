import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io
import cv2
import random
from typing import List, Dict, Tuple
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class AdvancedNSFWDetector:
    def __init__(self):
        # طیف گسترده رنگ‌های پوست
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
        
        # رنگ‌های حساس
        self.sensitive_colors = [
            ([255, 182, 193], 35),  # صورتی ملایم
            ([255, 160, 180], 40),  # صورتی متوسط
            ([255, 105, 180], 45),  # صورتی پر رنگ
            ([255, 20, 147], 35),   # صورتی عمیق
            ([220, 20, 60], 30),    # قرمز کریمسون
            ([255, 69, 0], 28),     # نارنجی قرمز
        ]
        
        # آستانه‌های تشخیص
        self.thresholds = {
            'skin_basic': 0.22,
            'skin_clustered': 0.12,
            'sensitive_colors': 0.06,
            'combined_score': 0.28
        }
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """پیش‌پردازش تصویر"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # بهبود کیفیت
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(1.2)
        smoothed = enhanced.filter(ImageFilter.MedianFilter(size=3))
        
        return np.array(smoothed)
    
    def detect_skin_advanced(self, image_array: np.ndarray) -> Tuple[float, np.ndarray]:
        """تشخیص پیشرفته پوست"""
        skin_mask = np.zeros(image_array.shape[:2], dtype=bool)
        
        # تشخیص با رنگ‌های مختلف پوست
        for skin_tone, threshold in self.skin_tones:
            skin_color = np.array(skin_tone)
            distances = np.sqrt(np.sum((image_array - skin_color) ** 2, axis=2))
            skin_mask |= distances < threshold
        
        # تشخیص HSV
        hsv_mask = self._detect_skin_hsv(image_array)
        skin_mask |= hsv_mask
        
        # تشخیص YCbCr
        ycbcr_mask = self._detect_skin_ycbcr(image_array)
        skin_mask |= ycbcr_mask
        
        skin_percentage = np.sum(skin_mask) / skin_mask.size
        return skin_percentage, skin_mask
    
    def _detect_skin_hsv(self, image_array: np.ndarray) -> np.ndarray:
        """تشخیص پوست در HSV"""
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        lower = np.array([0, 15, 70])
        upper = np.array([25, 170, 255])
        mask = cv2.inRange(hsv, lower, upper)
        return mask > 0
    
    def _detect_skin_ycbcr(self, image_array: np.ndarray) -> np.ndarray:
        """تشخیص پوست در YCbCr"""
        ycbcr = cv2.cvtColor(image_array, cv2.COLOR_RGB2YCrCb)
        lower = np.array([0, 133, 77])
        upper = np.array([255, 173, 127])
        mask = cv2.inRange(ycbcr, lower, upper)
        return mask > 0
    
    def analyze_skin_clusters(self, skin_mask: np.ndarray) -> float:
        """تحلیل خوشه‌های پوستی"""
        skin_uint8 = skin_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(skin_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        areas = [cv2.contourArea(contour) for contour in contours]
        total_area = skin_uint8.shape[0] * skin_uint8.shape[1]
        largest_region = max(areas) / total_area if areas else 0
        
        return largest_region
    
    def detect_sensitive_colors(self, image_array: np.ndarray) -> float:
        """تشخیص رنگ‌های حساس"""
        sensitive_pixels = 0
        total_pixels = image_array.shape[0] * image_array.shape[1]
        
        for color, threshold in self.sensitive_colors:
            target_color = np.array(color)
            distances = np.sqrt(np.sum((image_array - target_color) ** 2, axis=2))
            sensitive_pixels += np.sum(distances < threshold)
        
        return sensitive_pixels / total_pixels
    
    def detect_motion_patterns(self, image_array: np.ndarray, prev_array: np.ndarray = None) -> float:
        """تشخیص الگوهای حرکتی مشکوک"""
        if prev_array is None:
            return 0.0
        
        # محاسبه تفاوت بین فریم‌ها
        diff = cv2.absdiff(image_array, prev_array)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        
        # تشخیص حرکت
        motion_threshold = 30
        motion_mask = gray_diff > motion_threshold
        motion_percentage = np.sum(motion_mask) / motion_mask.size
        
        return motion_percentage
    
    def is_nsfw(self, image_bytes: bytes, prev_frame: np.ndarray = None) -> Dict:
        """تشخیص NSFW با اطلاعات کامل"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_array = self.preprocess_image(image)
            
            # تشخیص پوست
            skin_percentage, skin_mask = self.detect_skin_advanced(image_array)
            
            # تحلیل خوشه‌ها
            skin_clusters = self.analyze_skin_clusters(skin_mask)
            
            # رنگ‌های حساس
            sensitive_colors = self.detect_sensitive_colors(image_array)
            
            # الگوهای حرکتی
            motion_score = self.detect_motion_patterns(image_array, prev_frame)
            
            # محاسبه امتیاز نهایی
            weights = {
                'skin': 0.35,
                'clusters': 0.25,
                'sensitive': 0.25,
                'motion': 0.15
            }
            
            final_score = (
                skin_percentage * weights['skin'] +
                skin_clusters * weights['clusters'] +
                sensitive_colors * weights['sensitive'] +
                motion_score * weights['motion']
            )
            
            # تشخیص نهایی
            is_nsfw = (
                skin_percentage > self.thresholds['skin_basic'] or
                skin_clusters > self.thresholds['skin_clustered'] or
                sensitive_colors > self.thresholds['sensitive_colors'] or
                final_score > self.thresholds['combined_score']
            )
            
            return {
                'is_nsfw': is_nsfw,
                'confidence': final_score,
                'skin_percentage': skin_percentage,
                'skin_clusters': skin_clusters,
                'sensitive_colors': sensitive_colors,
                'motion_score': motion_score,
                'image_array': image_array
            }
            
        except Exception as e:
            return {
                'is_nsfw': False,
                'confidence': 0.0,
                'error': str(e),
                'image_array': None
            }

class NSFWGifDetector:
    def __init__(self):
        self.detector = AdvancedNSFWDetector()
        self.frame_cache = deque(maxlen=100)  # کش برای فریم‌ها
        
    def extract_smart_frames(self, video_path: str, strategy: str = 'adaptive') -> List[bytes]:
        """استخراج هوشمند فریم‌ها"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return []
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 10
        duration = frame_count / fps if fps > 0 else 0
        
        if frame_count == 0:
            cap.release()
            return []
        
        # انتخاب استراتژی بر اساس طول ویدیو
        if strategy == 'adaptive':
            if duration < 3:  # کوتاه
                num_frames = min(8, frame_count)
                strategy = 'uniform'
            elif duration < 10:  # متوسط
                num_frames = min(12, frame_count)
                strategy = 'weighted'
            else:  # طولانی
                num_frames = min(15, frame_count)
                strategy = 'smart'
        
        frames = self._extract_by_strategy(cap, frame_count, strategy, num_frames)
        cap.release()
        
        return frames
    
    def _extract_by_strategy(self, cap, frame_count: int, strategy: str, num_frames: int) -> List[bytes]:
        """استخراج فریم بر اساس استراتژی"""
        if strategy == 'uniform':
            # توزیع یکنواخت
            indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        elif strategy == 'weighted':
            # تمرکز بر وسط و انتها
            start_frames = np.linspace(0, frame_count * 0.2, num_frames // 3, dtype=int)
            middle_frames = np.linspace(frame_count * 0.3, frame_count * 0.7, num_frames // 3, dtype=int)
            end_frames = np.linspace(frame_count * 0.8, frame_count - 1, num_frames - 2 * (num_frames // 3), dtype=int)
            indices = np.concatenate([start_frames, middle_frames, end_frames])
        elif strategy == 'smart':
            # ترکیب یکنواخت و تصادفی
            uniform_count = num_frames // 2
            random_count = num_frames - uniform_count
            
            uniform_indices = np.linspace(0, frame_count - 1, uniform_count, dtype=int)
            random_indices = sorted(random.sample(range(frame_count), random_count))
            indices = sorted(set(list(uniform_indices) + random_indices))
        else:
            # پیش‌فرض: تصادفی
            indices = sorted(random.sample(range(frame_count), min(num_frames, frame_count)))
        
        return self._extract_frames_by_indices(cap, indices)
    
    def _extract_frames_by_indices(self, cap, indices: List[int]) -> List[bytes]:
        """استخراج فریم‌ها بر اساس ایندکس‌ها"""
        results = []
        current_frame = 0
        
        for target_index in indices:
            # جهش به فریم مورد نظر
            if target_index != current_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
                current_frame = target_index
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            # تبدیل و ذخیره
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # بهینه‌سازی کیفیت
            if pil_img.size[0] * pil_img.size[1] > 1000000:  # اگر خیلی بزرگ بود
                pil_img = pil_img.resize((800, 600), Image.Resampling.LANCZOS)
            
            with io.BytesIO() as output:
                pil_img.save(output, format="JPEG", quality=85, optimize=True)
                results.append(output.getvalue())
            
            current_frame += 1
        
        return results
    
    def analyze_frame_transitions(self, frame_results: List[Dict]) -> Dict:
        """تحلیل انتقال‌های بین فریم‌ها"""
        if len(frame_results) < 2:
            return {'transition_score': 0, 'suspicious_transitions': 0}
        
        suspicious_transitions = 0
        total_transitions = len(frame_results) - 1
        transition_scores = []
        
        for i in range(1, len(frame_results)):
            prev_result = frame_results[i-1]
            curr_result = frame_results[i]
            
            # بررسی تغییرات ناگهانی در امتیاز
            if prev_result.get('confidence', 0) < 0.3 and curr_result.get('confidence', 0) > 0.6:
                suspicious_transitions += 1
            
            # محاسبه امتیاز انتقال
            confidence_diff = abs(curr_result.get('confidence', 0) - prev_result.get('confidence', 0))
            transition_scores.append(confidence_diff)
        
        return {
            'transition_score': np.mean(transition_scores) if transition_scores else 0,
            'suspicious_transitions': suspicious_transitions / total_transitions if total_transitions > 0 else 0
        }
    
    def is_nsfw_parallel(self, gif_path: str, max_workers: int = 4) -> Dict:
        """تشخیص NSFW با پردازش موازی"""
        frames = self.extract_smart_frames(gif_path, strategy='adaptive')
        
        if not frames:
            return {
                'is_nsfw': False,
                'confidence': 0.0,
                'reason': 'No frames extracted',
                'frame_count': 0
            }
        
        # پردازش موازی فریم‌ها
        frame_results = []
        prev_frame = None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # ارسال کارها
            future_to_index = {
                executor.submit(self.detector.is_nsfw, frame, prev_frame): i 
                for i, frame in enumerate(frames)
            }
            
            # جمع‌آوری نتایج
            results_dict = {}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results_dict[index] = result
                    if result.get('image_array') is not None:
                        prev_frame = result['image_array']
                except Exception as e:
                    results_dict[index] = {
                        'is_nsfw': False,
                        'confidence': 0.0,
                        'error': str(e)
                    }
            
            # مرتب‌سازی نتایج
            frame_results = [results_dict[i] for i in sorted(results_dict.keys())]
        
        return self._calculate_final_result(frame_results)
    
    def is_nsfw(self, gif_path: str) -> Dict:
        """تشخیص NSFW اصلی"""
        frames = self.extract_smart_frames(gif_path, strategy='adaptive')
        
        if not frames:
            return {
                'is_nsfw': False,
                'confidence': 0.0,
                'reason': 'No frames extracted',
                'frame_count': 0
            }
        
        # پردازش فریم‌ها
        frame_results = []
        prev_frame = None
        
        for frame in frames:
            result = self.detector.is_nsfw(frame, prev_frame)
            frame_results.append(result)
            
            if result.get('image_array') is not None:
                prev_frame = result['image_array']
        
        return self._calculate_final_result(frame_results)
    
    def _calculate_final_result(self, frame_results: List[Dict]) -> Dict:
        """محاسبه نتیجه نهایی"""
        if not frame_results:
            return {'is_nsfw': False, 'confidence': 0.0}
        
        # آمار کلی
        nsfw_count = sum(1 for r in frame_results if r.get('is_nsfw', False))
        total_frames = len(frame_results)
        nsfw_ratio = nsfw_count / total_frames
        
        # میانگین اعتماد
        confidences = [r.get('confidence', 0) for r in frame_results]
        avg_confidence = np.mean(confidences)
        max_confidence = max(confidences)
        
        # تحلیل انتقال‌ها
        transition_analysis = self.analyze_frame_transitions(frame_results)
        
        # امتیاز نهایی
        final_weights = {
            'nsfw_ratio': 0.4,
            'avg_confidence': 0.3,
            'max_confidence': 0.2,
            'transitions': 0.1
        }
        
        final_score = (
            nsfw_ratio * final_weights['nsfw_ratio'] +
            avg_confidence * final_weights['avg_confidence'] +
            max_confidence * final_weights['max_confidence'] +
            transition_analysis['suspicious_transitions'] * final_weights['transitions']
        )
        
        # تصمیم نهایی
        is_nsfw = (
            nsfw_ratio > 0.3 or  # 30% فریم‌ها مشکوک
            avg_confidence > 0.35 or  # میانگین اعتماد بالا
            max_confidence > 0.7 or  # حداقل یک فریم با اعتماد خیلی بالا
            final_score > 0.4
        )
        
        return {
            'is_nsfw': is_nsfw,
            'confidence': final_score,
            'frame_count': total_frames,
            'nsfw_frames': nsfw_count,
            'nsfw_ratio': nsfw_ratio,
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'transition_analysis': transition_analysis,
            'frame_details': frame_results[:5]  # فقط 5 فریم اول برای جزئیات
        }
    
    def get_detailed_analysis(self, gif_path: str) -> Dict:
        """تحلیل کامل و جزئی"""
        result = self.is_nsfw(gif_path)
        
        # اطلاعات اضافی فایل
        try:
            cap = cv2.VideoCapture(gif_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 10
                duration = frame_count / fps
                
                result.update({
                    'total_frames': frame_count,
                    'fps': fps,
                    'duration_seconds': duration
                })
            cap.release()
        except:
            pass
        
        return result

# مثال استفاده:
# detector = NSFWGifDetector()
# 
# # تشخیص ساده
# result = detector.is_nsfw('animation.gif')
# print(f"NSFW: {result['is_nsfw']}")
# print(f"اعتماد: {result['confidence']:.2f}")
# 
# # تشخیص با پردازش موازی
# result = detector.is_nsfw_parallel('animation.gif', max_workers=6)
# 
# # تحلیل کامل
# analysis = detector.get_detailed_analysis('animation.gif')
# print(f"تعداد فریم‌ها: {analysis.get('frame_count', 0)}")
# print(f"نسبت فریم‌های مشکوک: {analysis.get('nsfw_ratio', 0):.2f}")
# print(f"مدت زمان: {analysis.get('duration_seconds', 0):.1f} ثانیه")
