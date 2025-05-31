from nudenet import NudeDetector
from PIL import Image
import io
import random
import numpy as np

class NSFWGifDetector:
    def __init__(self):
        self.detector = NudeDetector()

    def is_nsfw(self, gif_bytes):
        """
        بررسی می‌کند آیا گیف نامناسب است یا خیر
        این تابع 5 فریم تصادفی از گیف را بررسی می‌کند
        """
        try:
            gif = Image.open(io.BytesIO(gif_bytes))
            n_frames = getattr(gif, 'n_frames', 1)
            frames_to_check = min(5, n_frames)
            frame_indices = random.sample(range(n_frames), frames_to_check)
            
            nsfw_scores = []
            
            for frame_index in frame_indices:
                gif.seek(frame_index)
                frame = gif.convert('RGB')
                frame_array = np.array(frame)
                
                # بررسی با nudenet
                result = self.detector.detect(frame_array)
                
                # بررسی ساده رنگ پوست
                skin_tone = np.array([220, 180, 140])
                skin_pixels = np.sum(np.all(np.abs(frame_array - skin_tone) < 50, axis=-1))
                skin_percentage = skin_pixels / (frame_array.shape[0] * frame_array.shape[1])
                
                # ترکیب نتایج
                frame_score = max([item['score'] for item in result]) if result else 0
                frame_score = max(frame_score, skin_percentage)
                nsfw_scores.append(frame_score)
            
            avg_nsfw_score = np.mean(nsfw_scores)
            is_nsfw = avg_nsfw_score > 0.5  # آستانه را می‌توانید تنظیم کنید
            
            return is_nsfw, avg_nsfw_score
        
        except Exception as e:
            print(f"خطا در پردازش گیف: {e}")
            return False, 0.0
