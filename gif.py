import numpy as np
from PIL import Image
import io

class NSFWGifDetector:
    def __init__(self):
        # این یک مدل بسیار ساده است و فقط برای نمایش عملکرد استفاده می‌شود
        # در یک سناریوی واقعی، باید از یک مدل پیچیده‌تر استفاده کنید
        pass

    def is_nsfw(self, gif_bytes):
        """
        بررسی می‌کند آیا گیف نامناسب است یا خیر
        این تابع 5 فریم تصادفی از گیف را بررسی می‌کند
        """
        try:
            # تبدیل بایت‌های گیف به یک شیء Image
            with Image.open(io.BytesIO(gif_bytes)) as gif:
                # تعداد کل فریم‌ها
                n_frames = getattr(gif, "n_frames", 1)
                
                # انتخاب 5 فریم تصادفی (یا کمتر اگر گیف کمتر از 5 فریم دارد)
                frames_to_check = min(5, n_frames)
                frame_indices = np.random.choice(n_frames, frames_to_check, replace=False)
                
                for frame_index in frame_indices:
                    gif.seek(frame_index)
                    frame = gif.convert('RGB')
                    
                    # بررسی ساده رنگ پوست
                    image_array = np.array(frame)
                    skin_tone = np.array([220, 180, 140])  # یک نمونه رنگ پوست
                    skin_pixels = np.sum(np.all(np.abs(image_array - skin_tone) < 50, axis=-1))
                    skin_percentage = skin_pixels / (image_array.shape[0] * image_array.shape[1])
                    
                    # اگر بیش از 30% تصویر شبیه رنگ پوست باشد، آن را نامناسب در نظر می‌گیریم
                    if skin_percentage > 0.3:
                        return True
            
            return False
        except Exception as e:
            print(f"خطا در پردازش گیف: {e}")
            return False
