import numpy as np
from PIL import Image
import io

class NSFWDetector:
    def __init__(self):
        # این یک مدل بسیار ساده است و فقط برای نمایش عملکرد استفاده می‌شود
        # در یک سناریوی واقعی، باید از یک مدل پیچیده‌تر استفاده کنید
        pass

    def is_nsfw(self, image_bytes):
        """
        بررسی می‌کند آیا تصویر نامناسب است یا خیر
        این تابع فقط یک نمونه ساده است و دقت کافی ندارد
        """
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_array = np.array(image)
        
        # بررسی ساده رنگ پوست
        # این روش بسیار ساده است و نمی‌تواند به طور دقیق تصاویر نامناسب را تشخیص دهد
        skin_tone = np.array([220, 180, 140])  # یک نمونه رنگ پوست
        skin_pixels = np.sum(np.all(np.abs(image_array - skin_tone) < 50, axis=-1))
        skin_percentage = skin_pixels / (image_array.shape[0] * image_array.shape[1])
        
        # اگر بیش از 30% تصویر شبیه رنگ پوست باشد، آن را نامناسب در نظر می‌گیریم
        return skin_percentage > 0.3
