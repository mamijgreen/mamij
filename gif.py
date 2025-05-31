import numpy as np
from PIL import Image
import io
import random
from moviepy.editor import VideoFileClip

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
        # تبدیل بایت‌های گیف به یک فایل موقت
        with io.BytesIO(gif_bytes) as gif_file:
            # استفاده از moviepy برای خواندن گیف
            clip = VideoFileClip(gif_file.name)
            
            # انتخاب 5 فریم تصادفی
            total_frames = int(clip.fps * clip.duration)
            random_frames = random.sample(range(total_frames), min(5, total_frames))
            
            for frame_num in random_frames:
                # گرفتن فریم به عنوان یک آرایه numpy
                frame = clip.get_frame(frame_num / clip.fps)
                
                # تبدیل فریم به تصویر PIL
                image = Image.fromarray(frame)
                
                # بررسی ساده رنگ پوست
                image_array = np.array(image)
                skin_tone = np.array([220, 180, 140])  # یک نمونه رنگ پوست
                skin_pixels = np.sum(np.all(np.abs(image_array - skin_tone) < 50, axis=-1))
                skin_percentage = skin_pixels / (image_array.shape[0] * image_array.shape[1])
                
                # اگر بیش از 30% تصویر شبیه رنگ پوست باشد، آن را نامناسب در نظر می‌گیریم
                if skin_percentage > 0.3:
                    clip.close()
                    return True
            
            clip.close()
            return False
