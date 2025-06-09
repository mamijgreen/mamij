import numpy as np
from PIL import Image
import io

class NSFWStaticStickerDetector:
    def __init__(self):
        pass

    def is_nsfw(self, sticker_bytes: bytes) -> bool:
        """
        بررسی استیکرهای ثابت (WebP) بر اساس رنگ پوست
        مشابه photo.py اما فقط مخصوص استیکرهای ثابت
        """
        image = Image.open(io.BytesIO(sticker_bytes)).convert('RGB')
        image_array = np.array(image)

        skin_tone = np.array([220, 180, 140])
        skin_pixels = np.sum(np.all(np.abs(image_array - skin_tone) < 50, axis=-1))
        skin_percentage = skin_pixels / (image_array.shape[0] * image_array.shape[1])

        return skin_percentage > 0.3
