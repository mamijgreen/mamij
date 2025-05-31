# gif.py

import numpy as np
from PIL import Image
import io
import cv2
import random

class NSFWDetector:
    def __init__(self):
        pass

    def is_nsfw(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_array = np.array(image)

        skin_tone = np.array([220, 180, 140])  # رنگ پوست ساده
        skin_pixels = np.sum(np.all(np.abs(image_array - skin_tone) < 50, axis=-1))
        skin_percentage = skin_pixels / (image_array.shape[0] * image_array.shape[1])

        return skin_percentage > 0.3

class NSFWGifDetector:
    def __init__(self):
        self.detector = NSFWDetector()

    def extract_frames(self, video_path, num_frames=5):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count == 0:
            return []

        chosen_frames = sorted(random.sample(range(frame_count), min(num_frames, frame_count)))
        results = []
        current = 0

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if current in chosen_frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                with io.BytesIO() as output:
                    pil_img.save(output, format="JPEG")
                    results.append(output.getvalue())
            current += 1

        cap.release()
        return results

    def is_nsfw(self, gif_path):
        frames = self.extract_frames(gif_path)
        if not frames:
            return False

        scores = [self.detector.is_nsfw(f) for f in frames]
        score_ratio = sum(scores) / len(scores)
        return score_ratio > 0.3
