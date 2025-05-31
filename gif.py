from nudenet import NudeDetector
import imageio
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
            with imageio.get_reader(io.BytesIO(gif_bytes)) as reader:
                n_frames = reader.get_length()
                frames_to_check = min(5, n_frames)
                frame_indices = random.sample(range(n_frames), frames_to_check)
                
                nsfw_scores = []
                
                for frame_index in frame_indices:
                    frame = reader.get_data(frame_index)
                    result = self.detector.detect(frame)
                    
                    # اگر هر گونه شیء نامناسب تشخیص داده شود، فریم را نامناسب در نظر می‌گیریم
                    frame_score = max([item['score'] for item in result]) if result else 0
                    nsfw_scores.append(frame_score)
                
                avg_nsfw_score = np.mean(nsfw_scores)
                is_nsfw = avg_nsfw_score > 0.5  # آستانه را می‌توانید تنظیم کنید
                
                return is_nsfw, avg_nsfw_score
            
        except Exception as e:
            print(f"خطا در پردازش گیف: {e}")
            return False, 0.0
