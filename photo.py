from nudenet import NudeDetector
import io
from PIL import Image
import numpy as np

class NSFWDetector:
    def __init__(self):
        self.detector = NudeDetector()

    def is_nsfw(self, image_bytes):
        """
        بررسی می‌کند آیا تصویر نامناسب است یا خیر
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image)
            
            # بررسی با nudenet
            result = self.detector.detect(image_array)
            
            # بررسی ساده رنگ پوست
            skin_tone = np.array([220, 180, 140])  # یک نمونه رنگ پوست
            skin_pixels = np.sum(np.all(np.abs(image_array - skin_tone) < 50, axis=-1))
            skin_percentage = skin_pixels / (image_array.shape[0] * image_array.shape[1])
            
            # ترکیب نتایج
            nsfw_score = max([item['score'] for item in result]) if result else 0
            is_nsfw = (nsfw_score > 0.8) or (skin_percentage > 0.5 and nsfw_score > 0.3)
            
            return is_nsfw, max(nsfw_score, skin_percentage)
        except Exception as e:
            print(f"خطا در پردازش تصویر: {e}")
            return False, 0.0
