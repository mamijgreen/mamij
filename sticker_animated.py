import imageio
import io

class AnimatedStickerNSFWDetector:
    def __init__(self):
        pass

    def is_nsfw(self, sticker_bytes):
        try:
            frames = imageio.mimread(sticker_bytes, memtest=False)
        except Exception as e:
            print(f"Error reading animated sticker: {e}")
            return False

        frames_to_check = frames[:5]  # حداکثر 5 فریم اول

        for frame in frames_to_check:
            # تبدیل frame (numpy array) به bytes PNG برای بررسی عکس
            with io.BytesIO() as output:
                imageio.imwrite(output, frame, format='png')
                image_bytes = output.getvalue()

            if self.check_frame_nsfw(image_bytes):
                return True

        return False

    def check_frame_nsfw(self, image_bytes):
        # کد چک nsfw برای یک فریم عکس
        from PIL import Image
        import numpy as np

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_array = np.array(image)

        skin_tone = np.array([220, 180, 140])
        skin_pixels = np.sum(np.all(np.abs(image_array - skin_tone) < 50, axis=-1))
        skin_percentage = skin_pixels / (image_array.shape[0] * image_array.shape[1])

        return skin_percentage > 0.3
