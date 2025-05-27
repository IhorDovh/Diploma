"""
YOLO Detection Module
Містить функції для YOLO детекції об'єктів
"""

import cv2
import numpy as np
from image_processing import sharpen_image, apply_preprocessing, apply_color_preprocessing, sharpen_color_image


def yolo_detection(frame, model, IMAGE_SIZE, NAMES, COLORS, current_settings):
    """
    Виконання YOLO детекції на кадрі.
    
    Args:
        frame: Вхідний кадр
        model: Завантажена YOLO модель
        IMAGE_SIZE: Розмір зображення для моделі
        NAMES: Список назв класів
        COLORS: Список кольорів для класів
        current_settings: Словник з налаштуваннями
        
    Returns:
        Tuple (overlay, processed_frame, model_input)
    """
    overlay = np.zeros_like(frame, dtype=np.uint8)
    
    # Вибір між кольоровою та чорно-білою обробкою
    if current_settings.get("use_color_processing", False):
        # Кольорова обробка
        processed_frame = frame.copy()
        
        # Застосування preprocessing для кольорового зображення
        if current_settings["preprocessing"] != 0:
            processed_frame = apply_color_preprocessing(processed_frame, current_settings["preprocessing"])
        
        # Застосування різкості для кольорового зображення
        if current_settings["sharpness"] != 1:
            processed_frame = sharpen_color_image(processed_frame, current_settings["sharpness"])
        
        # Підготовка для моделі (залишаємо кольоровим)
        model_input_frame = processed_frame
    else:
        # Стандартна чорно-біла обробка
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Застосування попередньої обробки зображення
        preprocessed_gray = apply_preprocessing(gray_frame, current_settings["preprocessing"])
        
        # Застосування різкості
        sharpened_gray = sharpen_image(preprocessed_gray, current_settings["sharpness"])
        
        # Конвертація назад в 3-канальне для моделі
        model_input_frame = cv2.cvtColor(sharpened_gray, cv2.COLOR_GRAY2BGR)
        processed_frame = sharpened_gray
    
    # Зберігаємо зображення для відображення
    model_input = cv2.resize(model_input_frame, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Підготовка даних для моделі
    blob = cv2.dnn.blobFromImage(model_input_frame, 1/255.0, (IMAGE_SIZE, IMAGE_SIZE), swapRB=True, crop=False)
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
    
    return overlay, processed_frame, model_input


def load_yolo_model(model_path, names_path):
    """
    Завантаження YOLO моделі та назв класів.
    
    Args:
        model_path: Шлях до файлу моделі
        names_path: Шлях до файлу з назвами класів
        
    Returns:
        Tuple (model, names, colors)
    """
    # Завантаження моделі
    model = cv2.dnn.readNet(model_path)
    
    # Завантаження назв класів
    with open(names_path, "r") as f:
        names = [cname.strip() for cname in f.readlines()]
    
    # Генерація кольорів для класів
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]
    
    return model, names, colors


def post_process_detections(boxes, confidences, class_ids, conf_threshold=0.2, nms_threshold=0.5):
    """
    Пост-обробка детекцій з Non-Maximum Suppression.
    
    Args:
        boxes: Список bounding boxes
        confidences: Список довірчих оцінок
        class_ids: Список ID класів
        conf_threshold: Поріг довірчої оцінки
        nms_threshold: Поріг для NMS
        
    Returns:
        Список індексів відфільтрованих детекцій
    """
    if len(boxes) == 0:
        return []
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    if len(indexes) > 0:
        return indexes.flatten()
    else:
        return []


def draw_detections(frame, boxes, confidences, class_ids, names, colors, indexes, thickness=2):
    """
    Малювання детекцій на кадрі.
    
    Args:
        frame: Кадр для малювання
        boxes: Список bounding boxes
        confidences: Список довірчих оцінок
        class_ids: Список ID класів
        names: Список назв класів
        colors: Список кольорів
        indexes: Індекси відфільтрованих детекцій
        thickness: Товщина ліній
        
    Returns:
        Overlay з намальованими детекціями
    """
    overlay = np.zeros_like(frame, dtype=np.uint8)
    
    for i in indexes:
        left, top, width, height = boxes[i]
        class_id, score = class_ids[i], round(float(confidences[i]), 3)
        color = colors[class_id]
        
        # Малювання прямокутника
        cv2.rectangle(overlay, (left, top), (left + width, top + height), color, thickness)
        
        # Малювання тексту
        text = f'{names[class_id]} {score:.2f}'
        cv2.putText(overlay, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return overlay


class YOLODetector:
    """
    Клас для роботи з YOLO детекцією.
    """
    
    def __init__(self, model_path, names_path, image_size=640):
        """
        Ініціалізація YOLO детектора.
        
        Args:
            model_path: Шлях до моделі
            names_path: Шлях до файлу з назвами класів
            image_size: Розмір зображення для моделі
        """
        self.model, self.names, self.colors = load_yolo_model(model_path, names_path)
        self.image_size = image_size
        
        # Налаштування за замовчуванням
        self.settings = {
            "tresh": 0.25,
            "thickness": 2,
            "preprocessing": 0,
            "sharpness": 1
        }
    
    def update_settings(self, **kwargs):
        """
        Оновлення налаштувань детектора.
        
        Args:
            **kwargs: Нові налаштування
        """
        self.settings.update(kwargs)
    
    def detect(self, frame):
        """
        Виконання детекції на кадрі.
        
        Args:
            frame: Вхідний кадр
            
        Returns:
            Tuple (overlay, processed_gray, model_input)
        """
        return yolo_detection(frame, self.model, self.image_size, 
                            self.names, self.colors, self.settings)
    
    def detect_objects_only(self, frame):
        """
        Детекція об'єктів без створення overlay.
        
        Args:
            frame: Вхідний кадр
            
        Returns:
            Dict з результатами детекції
        """
        # Конвертація зображення в чорно-біле для моделі
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Застосування попередньої обробки зображення
        preprocessed_gray = apply_preprocessing(gray_frame, self.settings["preprocessing"])
        
        # Застосування різкості
        sharpened_gray = sharpen_image(preprocessed_gray, self.settings["sharpness"])
        
        # Зміна розмірності для моделі
        gray_frame_3ch = cv2.cvtColor(sharpened_gray, cv2.COLOR_GRAY2BGR)
        
        # Підготовка даних для моделі
        blob = cv2.dnn.blobFromImage(gray_frame_3ch, 1/255.0, (self.image_size, self.image_size), swapRB=True, crop=False)
        self.model.setInput(blob)
        preds = self.model.forward().transpose((0, 2, 1))
        
        image_height, image_width, _ = frame.shape
        x_factor = image_width / self.image_size
        y_factor = image_height / self.image_size
        
        class_ids, confs, boxes = [], [], []
        for row in preds[0]:
            conf = row[4]
            classes_score = row[4:]
            _, _, _, max_idx = cv2.minMaxLoc(classes_score)
            class_id = max_idx[1]
            if classes_score[class_id] > self.settings["tresh"]:
                confs.append(classes_score[class_id])
                class_ids.append(class_id)
                x, y, w, h = row[:4]
                left, top = int((x - 0.5 * w) * x_factor), int((y - 0.5 * h) * y_factor)
                width, height = int(w * x_factor), int(h * y_factor)
                boxes.append([left, top, width, height])
        
        # Застосування NMS
        indexes = post_process_detections(boxes, confs, class_ids)
        
        # Формування результату
        detections = []
        for i in indexes:
            detections.append({
                'bbox': boxes[i],
                'confidence': float(confs[i]),
                'class_id': class_ids[i],
                'class_name': self.names[class_ids[i]]
            })
        
        return {
            'detections': detections,
            'processed_frame': sharpened_gray,
            'total_objects': len(detections)
        }