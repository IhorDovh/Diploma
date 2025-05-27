import cv2
import time
import random
import argparse
import numpy as np
import threading
import queue
import os
from collections import deque

def load_source(source_file):
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    cap = None
    
    if source_file == "0":
        source_file = 0  # Веб-камера
        image_type = False
    else:
        image_type = source_file.split('.')[-1].lower() in img_formats
    
    if image_type:
        frame = cv2.imread(source_file)
    else:
        cap = cv2.VideoCapture(source_file)
    
    return image_type, frame if image_type else None, cap

def sharpen_image(image, strength=1):
    """
    Підвищення різкості зображення з можливістю регулювання сили ефекту.
    """
    result = image.copy()
    
    if strength == 1:
        kernel = np.array([[-1,-1,-1], 
                           [-1, 9,-1],
                           [-1,-1,-1]])
        result = cv2.filter2D(image, -1, kernel)
    
    elif strength == 2:
        kernel = np.array([[-2,-2,-2], 
                           [-2, 17,-2],
                           [-2,-2,-2]])
        result = cv2.filter2D(image, -1, kernel)
    
    elif strength == 3:
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        usm = cv2.addWeighted(image, 2.0, gaussian, -1.0, 0)
        
        kernel = np.array([[-2,-2,-2], 
                           [-2, 17,-2],
                           [-2,-2,-2]])
        result = cv2.filter2D(usm, -1, kernel)
        
        if len(image.shape) == 2:
            lab = cv2.cvtColor(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    return result

def apply_preprocessing(image, preprocessing_method):
    """
    Застосування різних методів попередньої обробки зображення.
    """
    if preprocessing_method == 0:
        return image
    
    elif preprocessing_method == 1:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    elif preprocessing_method == 2:
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    elif preprocessing_method == 3:
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        return cv2.addWeighted(image, 0.7, binary, 0.3, 0)
    
    elif preprocessing_method == 4:
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return cv2.addWeighted(image, 0.7, sobel, 0.3, 0)
    
    elif preprocessing_method == 5:
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        return closing
    
    elif preprocessing_method == 6:
        norm_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return norm_img
    
    return image

def yolo_detection(frame, model, IMAGE_SIZE, NAMES, COLORS, current_settings):
    """
    Виконання YOLO детекції на кадрі.
    """
    overlay = np.zeros_like(frame, dtype=np.uint8)
    
    # Конвертація зображення в чорно-біле для моделі
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Застосування попередньої обробки зображення
    preprocessed_gray = apply_preprocessing(gray_frame, current_settings["preprocessing"])
    
    # Застосування різкості
    sharpened_gray = sharpen_image(preprocessed_gray, current_settings["sharpness"])
    
    # Зміна розмірності для моделі
    gray_frame_3ch = cv2.cvtColor(sharpened_gray, cv2.COLOR_GRAY2BGR)
    
    # Зберігаємо зображення для відображення
    model_input = cv2.resize(gray_frame_3ch, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Підготовка даних для моделі
    blob = cv2.dnn.blobFromImage(gray_frame_3ch, 1/255.0, (IMAGE_SIZE, IMAGE_SIZE), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward().transpose((0, 2, 1))
    
    image_height, image_width, _ = frame.shape
    x_factor = image_width / IMAGE_SIZE
    y_factor = image_height / IMAGE_SIZE
    
    class_ids, confs, boxes = [], [], []
    for row in preds[0]:
        conf = row[4]
        classes_score = row[4:]
        _, _, _, max_idx = cv2.minMaxLoc(classes_score)
        class_id = max_idx[1]
        if classes_score[class_id] > current_settings["tresh"]:
            confs.append(classes_score[class_id])
            class_ids.append(class_id)
            x, y, w, h = row[:4]
            left, top = int((x - 0.5 * w) * x_factor), int((y - 0.5 * h) * y_factor)
            width, height = int(w * x_factor), int(h * y_factor)
            boxes.append([left, top, width, height])
    
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.2, 0.5)
    if len(indexes) > 0:
        for i in indexes.flatten():
            left, top, width, height = boxes[i]
            class_id, score = class_ids[i], round(float(confs[i]), 3)
            color = COLORS[class_id]
            cv2.rectangle(overlay, (left, top), (left + width, top + height), color, current_settings["thickness"])
            text = f'{NAMES[class_id]} {score:.2f}'
            cv2.putText(overlay, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return overlay, sharpened_gray, model_input

class OptimizedYOLODetectorWithDownsampling:
    """
    Оптимізований YOLO детектор з різними алгоритмами downsampling
    """
    def __init__(self, model_path, names_path, IMAGE_SIZE=640):
        # Основна ініціалізація
        self.model = cv2.dnn.readNet(model_path)
        self.IMAGE_SIZE = IMAGE_SIZE
        
        with open(names_path, "r") as f:
            self.NAMES = [cname.strip() for cname in f.readlines()]
        self.COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in self.NAMES]
        
        # Налаштування downsampling
        self.spatial_scale = 1.0  # Коефіцієнт зменшення просторової роздільності
        self.temporal_skip = 1    # Пропуск кадрів
        self.use_roi = False      # Використання ROI
        self.adaptive_skipping = True  # Адаптивний пропуск кадрів
        
        # Змінні для адаптивного пропуску
        self.frame_counter = 0
        self.target_detection_fps = 15
        self.current_skip_rate = 1
        
        # Для ROI детекції
        self.prev_frame = None
        self.motion_threshold = 30
        
        # Черги та потоки
        self.frame_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=3)
        
        self.processing_fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        
        self.current_settings = {
            "tresh": 0.25,
            "thickness": 2,
            "show_grayscale": True,
            "show_model_view": False,
            "sharpness": 1,
            "preprocessing": 0
        }
        
        self.running = True
        self.latest_overlay = None
        self.latest_gray_frame = None
        self.latest_model_input = None
        self.result_lock = threading.Lock()
        
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
    
    def set_downsampling_params(self, spatial_scale=1.0, temporal_skip=1, use_roi=False, adaptive_skipping=True):
        """
        Налаштування параметрів downsampling
        """
        self.spatial_scale = spatial_scale
        self.temporal_skip = temporal_skip
        self.use_roi = use_roi
        self.adaptive_skipping = adaptive_skipping
        print(f"Downsampling params: spatial={spatial_scale:.2f}, temporal={temporal_skip}, roi={use_roi}, adaptive={adaptive_skipping}")
    
    def _apply_spatial_downsampling(self, frame):
        """
        Застосування просторового downsampling
        """
        if self.spatial_scale >= 1.0:
            return frame, 1.0
            
        height, width = frame.shape[:2]
        new_width = int(width * self.spatial_scale)
        new_height = int(height * self.spatial_scale)
        
        # INTER_AREA найкраще для downsampling
        downsampled = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return downsampled, self.spatial_scale
    
    def _detect_motion_roi(self, current_frame):
        """
        Визначення області з рухом
        """
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return None
            
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(self.prev_frame, current_gray)
        _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Морфологічна обробка для усунення шуму
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.prev_frame = current_gray.copy()
        
        if contours:
            # Фільтрація малих контурів
            significant_contours = [c for c in contours if cv2.contourArea(c) > 500]
            
            if significant_contours:
                # Об'єднання всіх значущих областей
                x_coords = []
                y_coords = []
                for contour in significant_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    x_coords.extend([x, x+w])
                    y_coords.extend([y, y+h])
                
                roi = (int(min(x_coords)), int(min(y_coords)), 
                       int(max(x_coords) - min(x_coords)), 
                       int(max(y_coords) - min(y_coords)))
                
                # Розширення ROI на 20% для захоплення контексту
                height, width = current_frame.shape[:2]
                x, y, w, h = roi
                expand_x = max(0, x - int(w * 0.1))
                expand_y = max(0, y - int(h * 0.1))
                expand_w = min(width - expand_x, w + int(w * 0.2))
                expand_h = min(height - expand_y, h + int(h * 0.2))
                
                return (expand_x, expand_y, expand_w, expand_h)
        
        return None
    
    def _should_process_frame(self):
        """
        Визначення, чи потрібно обробляти поточний кадр (temporal downsampling)
        """
        self.frame_counter += 1
        
        if self.adaptive_skipping:
            # Адаптивне налаштування частоти обробки
            if self.processing_fps > 0:
                if self.processing_fps < self.target_detection_fps * 0.8:
                    self.current_skip_rate = min(self.current_skip_rate + 1, 5)
                elif self.processing_fps > self.target_detection_fps * 1.2:
                    self.current_skip_rate = max(self.current_skip_rate - 1, 1)
            
            return self.frame_counter % self.current_skip_rate == 0
        else:
            return self.frame_counter % self.temporal_skip == 0
    
    def _processing_loop(self):
        """
        Основний цикл обробки з підтримкою downsampling
        """
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # Перевірка temporal downsampling
                if not self._should_process_frame():
                    continue
                
                settings = self.current_settings.copy()
                start_time = time.time()
                
                # Застосування ROI якщо потрібно
                processing_frame = frame
                roi_offset = (0, 0)
                roi = None
                
                if self.use_roi:
                    roi = self._detect_motion_roi(frame)
                    if roi:
                        x, y, w, h = roi
                        processing_frame = frame[y:y+h, x:x+w]
                        roi_offset = (x, y)
                
                # Застосування spatial downsampling
                downsampled_frame, scale_factor = self._apply_spatial_downsampling(processing_frame)
                
                # Виконання детекції на зменшеному кадрі
                overlay, gray_frame, model_input = yolo_detection(
                    downsampled_frame, self.model, self.IMAGE_SIZE, self.NAMES, self.COLORS, settings
                )
                
                # Створення overlay для оригінального розміру
                full_overlay = np.zeros_like(frame, dtype=np.uint8)
                full_gray_frame = None
                
                if overlay is not None:
                    # Масштабування overlay назад
                    if scale_factor < 1.0:
                        overlay_upscaled = cv2.resize(overlay, 
                                                    (processing_frame.shape[1], processing_frame.shape[0]), 
                                                    interpolation=cv2.INTER_NEAREST)
                    else:
                        overlay_upscaled = overlay
                    
                    # Розміщення overlay з урахуванням ROI
                    if self.use_roi and roi:
                        x, y = roi_offset
                        h_overlay, w_overlay = overlay_upscaled.shape[:2]
                        full_overlay[y:y+h_overlay, x:x+w_overlay] = overlay_upscaled
                    else:
                        if overlay_upscaled.shape[:2] == frame.shape[:2]:
                            full_overlay = overlay_upscaled
                        else:
                            full_overlay = cv2.resize(overlay_upscaled, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Обробка gray_frame для відображення
                if gray_frame is not None:
                    if scale_factor < 1.0:
                        gray_frame = cv2.resize(gray_frame, 
                                              (processing_frame.shape[1], processing_frame.shape[0]), 
                                              interpolation=cv2.INTER_CUBIC)
                    
                    if self.use_roi and roi:
                        full_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        x, y = roi_offset
                        h_gray, w_gray = gray_frame.shape[:2]
                        full_gray_frame[y:y+h_gray, x:x+w_gray] = gray_frame
                    else:
                        if gray_frame.shape == frame.shape[:2]:
                            full_gray_frame = gray_frame
                        else:
                            full_gray_frame = cv2.resize(gray_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
                
                # Збереження результатів
                with self.result_lock:
                    self.latest_overlay = full_overlay
                    self.latest_gray_frame = full_gray_frame
                    self.latest_model_input = model_input
                
                # Оновлення статистики
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed >= 1.0:
                    self.processing_fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")
    
    def process_frame(self, frame):
        """
        Додавання кадру в чергу обробки
        """
        try:
            self.frame_queue.put(frame.copy(), block=False)
        except queue.Full:
            # Видаляємо найстарший кадр і додаємо новий
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put(frame.copy(), block=False)
            except queue.Empty:
                pass
    
    def get_latest_results(self):
        """
        Отримання останніх результатів обробки
        """
        with self.result_lock:
            return self.latest_overlay, self.latest_gray_frame, self.latest_model_input
    
    def update_settings(self, **kwargs):
        """
        Оновлення налаштувань детектора
        """
        self.current_settings.update(kwargs)
    
    def get_performance_stats(self):
        """
        Отримання статистики продуктивності
        """
        return {
            'processing_fps': self.processing_fps,
            'spatial_scale': self.spatial_scale,
            'temporal_skip': self.current_skip_rate if self.adaptive_skipping else self.temporal_skip,
            'using_roi': self.use_roi,
            'adaptive_skipping': self.adaptive_skipping
        }
    
    def stop(self):
        """
        Зупинка детектора
        """
        self.running = False
        self.processing_thread.join(timeout=1.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/idiots3.mp4", help="Video")
    parser.add_argument("--names", type=str, default="data/class.names", help="Object Names")
    parser.add_argument("--model", type=str, default="./models/yolo11n-old.onnx", help="Pretrained Model")
    parser.add_argument("--tresh", type=float, default=0.25, help="Confidence Threshold")
    parser.add_argument("--thickness", type=int, default=2, help="Bounding Box Thickness")
    parser.add_argument("--show_grayscale", default=True, action="store_true", help="Show grayscale input")
    parser.add_argument("--show_model_view", action="store_true", help="Show what the model sees")
    parser.add_argument("--sharpness", type=int, default=1, choices=[1, 2, 3], 
                        help="Sharpness level (1=Standard, 2=Strong, 3=Extreme)")
    parser.add_argument("--preprocessing", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6],
                        help="Preprocessing method")
    
    # Нові аргументи для downsampling
    parser.add_argument("--spatial_scale", type=float, default=1, 
                        help="Spatial downsampling scale (0.5 = half resolution)")
    parser.add_argument("--temporal_skip", type=int, default=1, 
                        help="Process every N-th frame")
    parser.add_argument("--use_roi", action="store_true",  default=True,
                        help="Use motion-based ROI detection")
    parser.add_argument("--adaptive_fps", action="store_true", default=True,
                        help="Use adaptive frame skipping")
    parser.add_argument("--target_fps", type=int, default=7,
                        help="Target detection FPS for adaptive mode")
    
    args = parser.parse_args()
    
    # Ініціалізація детектора з downsampling
    detector = OptimizedYOLODetectorWithDownsampling(args.model, args.names)
    
    # Налаштування downsampling параметрів
    detector.set_downsampling_params(
        spatial_scale=args.spatial_scale,
        temporal_skip=args.temporal_skip,
        use_roi=args.use_roi,
        adaptive_skipping=args.adaptive_fps
    )
    
    detector.target_detection_fps = args.target_fps
    
    # Оновлення основних налаштувань
    detector.update_settings(
        tresh=args.tresh,
        thickness=args.thickness,
        show_grayscale=args.show_grayscale,
        show_model_view=args.show_model_view,
        sharpness=args.sharpness,
        preprocessing=args.preprocessing
    )
    
    # Завантаження відео
    image_type, frame, cap = load_source(args.source)
    
    if image_type:
        print("Processing static image...")
        # Для статичного зображення
        detector.process_frame(frame)
        time.sleep(1)  # Даємо час на обробку
        overlay, gray_frame, model_input = detector.get_latest_results()
        
        if overlay is not None:
            result = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)
            cv2.imshow("YOLO Detection", result)
            cv2.waitKey(0)
        
        detector.stop()
        cv2.destroyAllWindows()
        return
    
    # Для відео
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / video_fps if video_fps > 0 else 0
    
    print(f"Starting video processing...")
    print(f"Video FPS: {video_fps:.2f}")
    print(f"Downsampling settings:")
    print(f"  - Spatial scale: {args.spatial_scale}")
    print(f"  - Temporal skip: {args.temporal_skip}")
    print(f"  - Use ROI: {args.use_roi}")
    print(f"  - Adaptive FPS: {args.adaptive_fps}")
    print(f"  - Target detection FPS: {args.target_fps}")
    
    try:
        while True:
            start_frame_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Відправляємо кадр на обробку
            detector.process_frame(frame)
            
            # Отримуємо останні результати
            overlay, gray_frame, model_input = detector.get_latest_results()
            
            # Формування результату для відображення
            if overlay is not None:
                if detector.current_settings["show_model_view"] and model_input is not None:
                    # Відображаємо те, що бачить модель
                    model_view_resized = cv2.resize(model_input, (frame.shape[1], frame.shape[0]))
                    result = cv2.addWeighted(model_view_resized, 1.0, overlay, 1.0, 0)
                    display_mode = "Model View Mode"
                elif detector.current_settings["show_grayscale"] and gray_frame is not None:
                    # Показуємо чорно-біле зображення
                    gray_frame_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                    result = cv2.addWeighted(gray_frame_3ch, 1.0, overlay, 1.0, 0)
                    display_mode = "Grayscale Sharpened Mode"
                else:
                    # Показуємо кольорове зображення
                    result = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)
                    display_mode = "Color Mode"
            else:
                # Якщо немає результатів, показуємо просто кадр
                result = frame.copy()
                display_mode = "Waiting for detection..."
            
            # Отримання статистики продуктивності
            perf_stats = detector.get_performance_stats()
            
            # Відображення інформації
            fps_text = f"Video FPS: {video_fps:.2f} | Detection FPS: {perf_stats['processing_fps']:.2f}"
            cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Інформація про downsampling
            downsampling_text = f"Spatial: {perf_stats['spatial_scale']:.2f} | Temporal: {perf_stats['temporal_skip']} | ROI: {perf_stats['using_roi']}"
            cv2.putText(result, downsampling_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Інформація про режим та налаштування
            preproc_names = ["None", "CLAHE", "Bilateral", "Adaptive Threshold", "Sobel", "Morphological", "Normalization"]
            preproc_name = preproc_names[detector.current_settings["preprocessing"]]
            cv2.putText(result, f"{display_mode} (Sharpness: {detector.current_settings['sharpness']}, Preproc: {preproc_name})", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Підказки
            cv2.putText(result, "Press 'g' for grayscale, 'm' for model view, 'c' for color", 
                       (10, result.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, "Press '1', '2', '3' to change sharpness level", 
                       (10, result.shape[0] - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, "Press '0'-'6' to change preprocessing method", 
                       (10, result.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, "Press 's' to toggle spatial scale | 'r' to toggle ROI | 'q' to quit", 
                       (10, result.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("YOLO Detection", result)
            
            # Обробка клавіатури
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                detector.update_settings(show_grayscale=True, show_model_view=False)
                print("Switched to Grayscale mode")
            elif key == ord('m'):
                detector.update_settings(show_model_view=True, show_grayscale=False)
                print("Switched to Model View mode")
            elif key == ord('c'):
                detector.update_settings(show_grayscale=False, show_model_view=False)
                print("Switched to Color mode")
            elif key == ord('1'):
                detector.update_settings(sharpness=1)
                print("Sharpness level set to 1 (Standard)")
            elif key == ord('2'):
                detector.update_settings(sharpness=2)
                print("Sharpness level set to 2 (Strong)")
            elif key == ord('3'):
                detector.update_settings(sharpness=3)
                print("Sharpness level set to 3 (Extreme)")
            elif key in [ord('0')]:
                if chr(key) == '0':
                    preproc_idx = 0
                    detector.update_settings(preprocessing=preproc_idx)
                    print(f"Preprocessing method set to {preproc_idx} ({preproc_names[preproc_idx]})")
            elif key in [ord('4'), ord('5'), ord('6')]:
                preproc_idx = int(chr(key))
                if 0 <= preproc_idx <= 6:
                    detector.update_settings(preprocessing=preproc_idx)
                    print(f"Preprocessing method set to {preproc_idx} ({preproc_names[preproc_idx]})")
            
            # Нові hotkeys для downsampling
            elif key == ord('s'):
                # Переключення spatial scale між 1.0, 0.75, 0.5, 0.25
                current_scale = detector.spatial_scale
                if current_scale >= 1.0:
                    new_scale = 0.75
                elif current_scale >= 0.75:
                    new_scale = 0.5
                elif current_scale >= 0.5:
                    new_scale = 0.25
                else:
                    new_scale = 1.0
                detector.set_downsampling_params(
                    spatial_scale=new_scale,
                    temporal_skip=detector.temporal_skip,
                    use_roi=detector.use_roi,
                    adaptive_skipping=detector.adaptive_skipping
                )
                print(f"Spatial scale changed to {new_scale}")
            
            elif key == ord('r'):
                # Переключення ROI detection
                new_roi = not detector.use_roi
                detector.set_downsampling_params(
                    spatial_scale=detector.spatial_scale,
                    temporal_skip=detector.temporal_skip,
                    use_roi=new_roi,
                    adaptive_skipping=detector.adaptive_skipping
                )
                print(f"ROI detection {'enabled' if new_roi else 'disabled'}")
            
            elif key == ord('t'):
                # Переключення temporal skip між 1, 2, 3, 4
                current_skip = detector.temporal_skip
                new_skip = current_skip + 1 if current_skip < 4 else 1
                detector.set_downsampling_params(
                    spatial_scale=detector.spatial_scale,
                    temporal_skip=new_skip,
                    use_roi=detector.use_roi,
                    adaptive_skipping=False  # Вимикаємо adaptive при ручному налаштуванні
                )
                print(f"Temporal skip changed to {new_skip}")
            
            elif key == ord('a'):
                # Переключення adaptive frame skipping
                new_adaptive = not detector.adaptive_skipping
                detector.set_downsampling_params(
                    spatial_scale=detector.spatial_scale,
                    temporal_skip=detector.temporal_skip,
                    use_roi=detector.use_roi,
                    adaptive_skipping=new_adaptive
                )
                print(f"Adaptive frame skipping {'enabled' if new_adaptive else 'disabled'}")
            
            # Контроль FPS
            elapsed_time = time.time() - start_frame_time
            sleep_time = max(frame_time - elapsed_time, 0)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Очищення ресурсів
        print("\nCleaning up...")
        
        # Виводимо фінальну статистику
        final_stats = detector.get_performance_stats()
        print(f"Final Performance Statistics:")
        print(f"  - Detection FPS: {final_stats['processing_fps']:.2f}")
        print(f"  - Spatial Scale: {final_stats['spatial_scale']:.2f}")
        print(f"  - Temporal Skip: {final_stats['temporal_skip']}")
        print(f"  - ROI Detection: {final_stats['using_roi']}")
        print(f"  - Adaptive Skipping: {final_stats['adaptive_skipping']}")
        
        detector.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()