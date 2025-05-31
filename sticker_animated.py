from PIL import Image
import imageio
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
            # استفاده از context manager برای مدیریت بهتر منابع
            with imageio.get_reader(video_path) as reader:
                # بررسی وجود metadata برای تعداد فریم‌ها
                total_frames = 0
                
                # تلاش برای گرفتن تعداد فریم‌ها با روش‌های مختلف
                try:
                    # روش اول: استفاده از count_frames اگر موجود باشد
                    if hasattr(reader, 'count_frames'):
                        total_frames = reader.count_frames()
                    elif hasattr(reader, '_meta') and 'nframes' in reader._meta:
                        total_frames = reader._meta['nframes']
                    else:
                        # روش دوم: شمارش دستی فریم‌ها
                        total_frames = 0
                        try:
                            while True:
                                reader.get_data(total_frames)
                                total_frames += 1
                                # محدودیت برای جلوگیری از حلقه بی‌نهایت
                                if total_frames > 1000:
                                    break
                        except (IndexError, RuntimeError, OSError):
                            pass
                except Exception:
                    # اگر هیچ‌کدام کار نکرد، حداقل یک فریم را بررسی کنیم
                    total_frames = 1
                
                if total_frames == 0:
                    print("No frames found in the file")
                    return False
                
                # انتخاب تعداد فریم‌های نمونه
                sample_frames_count = min(5, total_frames)
                
                # انتخاب ایندکس‌های فریم
                if total_frames == 1:
                    frame_indices = [0]
                else:
                    try:
                        frame_indices = random.sample(range(total_frames), sample_frames_count)
                    except ValueError:
                        # اگر تعداد فریم‌ها کمتر از نمونه‌ها باشد
                        frame_indices = list(range(min(sample_frames_count, total_frames)))
                
                skin_percentages = []
                skin_tone = np.array([220, 180, 140])
                
                successful_frames = 0
                
                for idx in frame_indices:
                    try:
                        frame = reader.get_data(idx)
                        
                        # تبدیل به numpy array
                        if hasattr(frame, 'shape'):
                            image_array = np.array(frame)
                        else:
                            # اگر PIL Image باشد
                            image_array = np.array(frame)
                        
                        # بررسی ابعاد تصویر
                        if len(image_array.shape) < 3:
                            print(f"Frame {idx} has unexpected dimensions: {image_array.shape}")
                            continue
                        
                        # اگر تصویر RGBA بود، حذف کانال آلفا
                        if image_array.shape[2] == 4:
                            image_array = image_array[:, :, :3]
                        elif image_array.shape[2] != 3:
                            print(f"Frame {idx} has unexpected color channels: {image_array.shape[2]}")
                            continue
                        
                        # محاسبه درصد پیکسل‌های رنگ پوست
                        skin_mask = np.all(np.abs(image_array - skin_tone) < 50, axis=-1)
                        skin_pixels = np.sum(skin_mask)
                        total_pixels = image_array.shape[0] * image_array.shape[1]
                        
                        if total_pixels > 0:
                            skin_percentage = skin_pixels / total_pixels
                            skin_percentages.append(skin_percentage)
                            successful_frames += 1
                        
                    except (IndexError, RuntimeError, OSError, ValueError) as e:
                        print(f"Error processing frame {idx}: {e}")
                        continue
                
                # بررسی اینکه آیا حداقل یک فریم با موفقیت پردازش شده
                if successful_frames == 0:
                    print("No frames could be processed successfully")
                    return False
                
                # محاسبه میانگین درصد رنگ پوست
                avg_skin_percentage = sum(skin_percentages) / len(skin_percentages)
                
                print(f"Processed {successful_frames} frames, average skin percentage: {avg_skin_percentage:.3f}")
                
                return avg_skin_percentage > 0.3
                
        except Exception as e:
            print(f"Error reading animated sticker: {e}")
            return False
    
    def is_nsfw_alternative(self, video_path: str) -> bool:
        """
        روش جایگزین با استفاده از opencv-python اگر imageio مشکل داشت
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Could not open video file with OpenCV")
                return False
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return False
            
            sample_frames_count = min(5, total_frames)
            frame_indices = random.sample(range(total_frames), sample_frames_count) if total_frames > 1 else [0]
            
            skin_percentages = []
            skin_tone = np.array([140, 180, 220])  # BGR format for OpenCV
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # محاسبه درصد پیکسل‌های رنگ پوست
                skin_mask = np.all(np.abs(frame - skin_tone) < 50, axis=-1)
                skin_pixels = np.sum(skin_mask)
                total_pixels = frame.shape[0] * frame.shape[1]
                
                if total_pixels > 0:
                    skin_percentage = skin_pixels / total_pixels
                    skin_percentages.append(skin_percentage)
            
            cap.release()
            
            if not skin_percentages:
                return False
            
            avg_skin_percentage = sum(skin_percentages) / len(skin_percentages)
            return avg_skin_percentage > 0.3
            
        except ImportError:
            print("OpenCV not available, falling back to imageio method")
            return self.is_nsfw(video_path)
        except Exception as e:
            print(f"Error with OpenCV method: {e}")
            return False
