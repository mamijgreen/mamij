from PIL import Image
import imageio.v3 as iio
import numpy as np
import random
import os

class NSFWAnimatedStickerDetector:
    def __init__(self):
        pass

    def is_nsfw(self, video_path: str) -> bool:
        """
        بررسی استیکر متحرک (مثلاً webm) با گرفتن چند فریم رندوم و میانگین بررسی رنگ پوست
        مشابه gif.py ولی برای استیکرهای متحرک
        """

        try:
            with iio.imopen(video_path, 'r') as file:
                frames = list(file)
                total_frames = len(frames)
                if total_frames == 0:
                    return False

                sample_frames_count = min(5, total_frames)
                frame_indices = random.sample(range(total_frames), sample_frames_count)

                skin_percentages = []
                skin_tone = np.array([220, 180, 140])

                for idx in frame_indices:
                    frame = frames[idx]
                    image_array = np.array(frame)

                    # اگر تصویر RGBA بود، حذف کانال آلفا
                    if image_array.shape[2] == 4:
                        image_array = image_array[:, :, :3]

                    skin_pixels = np.sum(np.all(np.abs(image_array - skin_tone) < 50, axis=-1))
                    skin_percentage = skin_pixels / (image_array.shape[0] * image_array.shape[1])
                    skin_percentages.append(skin_percentage)

                avg_skin_percentage = sum(skin_percentages) / len(skin_percentages)

            return avg_skin_percentage > 0.3

        except Exception as e:
            print(f"Error reading animated sticker: {e}")
            return False
