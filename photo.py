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
            result = self.detector.detect(image_array)
            
            # اگر هر گونه شیء نامناسب تشخیص داده شود، تصویر را نامناسب در نظر می‌گیریم
            is_nsfw = len(result) > 0
            nsfw_score = max([item['score'] for item in result]) if result else 0
            
            return is_nsfw, nsfw_score
        except Exception as e:
            print(f"خطا در پردازش تصویر: {e}")
            return False, 0.0
