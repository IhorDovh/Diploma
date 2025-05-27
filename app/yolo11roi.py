import cv2
import time
import random
import argparse
import numpy as np
import threading
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

class ROIDetector:
    """
    Клас для детекції областей інтересу (ROI) на основі руху.
    """
    
    def __init__(self, motion_threshold=30, min_area=1000, expand_ratio=0.2):
        self.motion_threshold = motion_threshold
        self.min_area = min_area
        self.expand_ratio = expand_ratio
        self.prev_frame = None
        self.roi_history = deque(maxlen=5)  # Історія ROI для згладжування
    
    def detect_roi(self, current_frame):
        """
        Детекція ROI на основі руху між кадрами.
        """
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return None
            
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Додаємо Gaussian blur для зменшення шуму
        prev_blur = cv2.GaussianBlur(self.prev_frame, (5, 5), 0)
        curr_blur = cv2.GaussianBlur(current_gray, (5, 5), 0)
        
        diff = cv2.absdiff(prev_blur, curr_blur)
        _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Морфологічна обробка
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.prev_frame = current_gray.copy()
        
        if contours:
            # Фільтрація малих контурів
            significant_contours = [c for c in contours if cv2.contourArea(c) > self.min_area]
            
            if significant_contours:
                # Об'єднання всіх значущих областей
                all_points = []
                for contour in significant_contours:
                    points = contour.reshape(-1, 2)
                    all_points.extend(points)
                
                if all_points:
                    all_points = np.array(all_points)
                    x_min, y_min = np.min(all_points, axis=0)
                    x_max, y_max = np.max(all_points, axis=0)
                    
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Розширення ROI
                    frame_height, frame_width = current_frame.shape[:2]
                    expand_w = int(width * self.expand_ratio)
                    expand_h = int(height * self.expand_ratio)
                    
                    x_expand = max(0, x_min - expand_w)
                    y_expand = max(0, y_min - expand_h)
                    w_expand = min(frame_width - x_expand, width + 2 * expand_w)
                    h_expand = min(frame_height - y_expand, height + 2 * expand_h)
                    
                    # Мінімальний розмір ROI
                    min_roi_size = min(frame_width, frame_height) // 4
                    if w_expand < min_roi_size or h_expand < min_roi_size:
                        return None
                    
                    roi = (x_expand, y_expand, w_expand, h_expand)
                    
                    # Згладжування ROI
                    return self._smooth_roi(roi, current_frame.shape[:2])
        
        return None
    
    def _smooth_roi(self, new_roi, frame_shape):
        """
        Згладжування ROI через історію
        """
        self.roi_history.append(new_roi)
        
        if len(self.roi_history) < 3:
            return new_roi
        
        # Усереднюємо координати
        x_coords = [roi[0] for roi in self.roi_history]
        y_coords = [roi[1] for roi in self.roi_history]
        w_coords = [roi[2] for roi in self.roi_history]
        h_coords = [roi[3] for roi in self.roi_history]
        
        avg_x = int(sum(x_coords) / len(x_coords))
        avg_y = int(sum(y_coords) / len(y_coords))
        avg_w = int(sum(w_coords) / len(w_coords))
        avg_h = int(sum(h_coords) / len(h_coords))
        
        # Перевірка меж
        frame_h, frame_w = frame_shape
        avg_x = max(0, min(avg_x, frame_w - avg_w))
        avg_y = max(0, min(avg_y, frame_h - avg_h))
        avg_w = max(1, min(avg_w, frame_w - avg_x))
        avg_h = max(1, min(avg_h, frame_h - avg_y))
        
        return (avg_x, avg_y, avg_w, avg_h)

def yolo_detection(frame, model, IMAGE_SIZE, NAMES, COLORS, args, roi=None):
    """
    Виконання YOLO детекції з підтримкою ROI
    """
    overlay = np.zeros_like(frame, dtype=np.uint8)
    
    # Визначаємо область для обробки
    if roi:
        x, y, w, h = roi
        processing_frame = frame[y:y+h, x:x+w]
        roi_offset = (x, y)
    else:
        processing_frame = frame
        roi_offset = (0, 0)
    
    # YOLO детекція
    blob = cv2.dnn.blobFromImage(processing_frame, 1/255.0, (IMAGE_SIZE, IMAGE_SIZE), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward().transpose((0, 2, 1))
    
    image_height, image_width, _ = processing_frame.shape
    x_factor = image_width / IMAGE_SIZE
    y_factor = image_height / IMAGE_SIZE
    
    class_ids, confs, boxes = [], [], []
    for row in preds[0]:
        conf = row[4]
        classes_score = row[4:]
        _, _, _, max_idx = cv2.minMaxLoc(classes_score)
        class_id = max_idx[1]
        if classes_score[class_id] > args.tresh:
            confs.append(classes_score[class_id])
            class_ids.append(class_id)
            x_det, y_det, w_det, h_det = row[:4]
            left = int((x_det - 0.5 * w_det) * x_factor)
            top = int((y_det - 0.5 * h_det) * y_factor)
            width = int(w_det * x_factor)
            height = int(h_det * y_factor)
            boxes.append([left, top, width, height])
    
    # NMS
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.2, 0.5)
    if len(indexes) > 0:
        for i in indexes.flatten():
            left, top, width, height = boxes[i]
            class_id, score = class_ids[i], round(float(confs[i]), 3)
            color = COLORS[class_id]
            
            # Коригуємо координати з урахуванням ROI offset
            actual_left = left + roi_offset[0]
            actual_top = top + roi_offset[1]
            
            cv2.rectangle(overlay, (actual_left, actual_top), 
                         (actual_left + width, actual_top + height), color, args.thickness)
            text = f'{NAMES[class_id]} {score:.2f}'
            cv2.putText(overlay, text, (actual_left, actual_top - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return overlay, roi

def async_yolo_processing(model, IMAGE_SIZE, NAMES, COLORS, args, roi_detector, use_roi):
    """
    Асинхронна обробка YOLO з ROI
    """
    global latest_frame, processed_overlay, processing_fps, current_roi
    frame_count = 0
    start_processing_time = time.time()
    
    while True:
        if latest_frame is not None:
            frame_copy = latest_frame.copy()
            
            # Детекція ROI якщо увімкнено
            roi = None
            if use_roi:
                roi = roi_detector.detect_roi(frame_copy)
                current_roi = roi  # Зберігаємо для візуалізації
            
            # YOLO детекція
            processed_overlay, _ = yolo_detection(frame_copy, model, IMAGE_SIZE, NAMES, COLORS, args, roi)
            frame_count += 1
            
            # Обчислюємо FPS розпізнавання
            elapsed_processing_time = time.time() - start_processing_time
            if elapsed_processing_time >= 1.0:
                processing_fps = frame_count / elapsed_processing_time
                frame_count = 0
                start_processing_time = time.time()

if __name__ == '__main__':
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/idiots3.mp4", help="Video")
    parser.add_argument("--names", type=str, default="data/class.names", help="Object Names")
    parser.add_argument("--model", type=str, default="./models/yolo11n-old.onnx", help="Pretrained Model")
    parser.add_argument("--tresh", type=float, default=0.25, help="Confidence Threshold")
    parser.add_argument("--thickness", type=int, default=2, help="Bounding Box Thickness")
    parser.add_argument("--use_roi", action="store_true", help="Enable ROI detection")
    parser.add_argument("--motion_threshold", type=int, default=30, help="Motion detection threshold")
    parser.add_argument("--min_area", type=int, default=1000, help="Minimum area for ROI")
    args = parser.parse_args()
    
    # Завантаження моделі
    model = cv2.dnn.readNet(args.model)
    try:
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using CUDA for inference")
    except:
        print("CUDA not available, using CPU")
    
    IMAGE_SIZE = 640
    with open(args.names, "r") as f:
        NAMES = [cname.strip() for cname in f.readlines()]
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in NAMES]
    
    # Ініціалізація ROI детектора
    roi_detector = ROIDetector(
        motion_threshold=args.motion_threshold,
        min_area=args.min_area,
        expand_ratio=0.2
    )
    
    # Глобальні змінні
    image_type, frame, cap = load_source(args.source)
    latest_frame, processed_overlay, current_roi = None, None, None
    processing_fps = 0.0
    
    # Запускаємо фоновий потік для обробки YOLO
    processing_thread = threading.Thread(
        target=async_yolo_processing, 
        args=(model, IMAGE_SIZE, NAMES, COLORS, args, roi_detector, args.use_roi), 
        daemon=True
    )
    processing_thread.start()

    video_fps = cap.get(cv2.CAP_PROP_FPS) if not image_type else 0
    frame_time = 1 / video_fps if video_fps > 0 else 0
    
    print(f"Video FPS: {video_fps:.2f}")
    print(f"ROI Detection: {'Enabled' if args.use_roi else 'Disabled'}")
    print("Press 'r' to toggle ROI, 'q' to quit")

    while True:
        start_frame_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        latest_frame = frame.copy()
        overlay = processed_overlay if processed_overlay is not None else np.zeros_like(frame, dtype=np.uint8)
        result = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)

        # Візуалізація ROI
        if args.use_roi and current_roi:
            x, y, w, h = current_roi
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, f"ROI: {w}x{h}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Відображення інформації
        fps_text = f"Video FPS: {video_fps:.2f} | Detection FPS: {processing_fps:.2f}"
        cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        roi_text = f"ROI: {'ON' if args.use_roi else 'OFF'}"
        cv2.putText(result, roi_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Підказки
        cv2.putText(result, "Press 'r' to toggle ROI | 'q' to quit", 
                   (10, result.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLO Detection with ROI", result)

        # Обробка клавіатури
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            args.use_roi = not args.use_roi
            print(f"ROI Detection: {'Enabled' if args.use_roi else 'Disabled'}")
            if not args.use_roi:
                current_roi = None  # Очищаємо ROI при вимкненні

        # Контроль FPS
        elapsed_time = time.time() - start_frame_time
        sleep_time = max(frame_time - elapsed_time, 0)
        time.sleep(sleep_time)
    
    cap.release()
    cv2.destroyAllWindows()
    
    end_time = time.time()
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")
    print(f"Video FPS: {video_fps:.2f}")
    print(f"Average Detection FPS: {processing_fps:.2f}")
    if args.use_roi:
        print("ROI detection was enabled")