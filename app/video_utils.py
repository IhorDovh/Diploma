"""
Video Processing Utilities Module
Містить функції для роботи з відео та джерелами даних
"""

import cv2
import os
import time
from typing import Tuple, Optional, Union


def load_source(source_file: str) -> Tuple[bool, Optional[cv2.Mat], Optional[cv2.VideoCapture]]:
    """
    Завантаження джерела даних (зображення або відео).
    
    Args:
        source_file: Шлях до файлу або "0" для веб-камери
        
    Returns:
        Tuple (is_image, frame_or_none, capture_or_none)
    """
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    cap = None
    
    if source_file == "0":
        source_file = 0  # Веб-камера
        image_type = False
    else:
        image_type = source_file.split('.')[-1].lower() in img_formats
    
    if image_type:
        frame = cv2.imread(source_file)
        if frame is None:
            raise FileNotFoundError(f"Could not load image: {source_file}")
    else:
        cap = cv2.VideoCapture(source_file)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source_file}")
    
    return image_type, frame if image_type else None, cap


def get_video_info(cap: cv2.VideoCapture) -> dict:
    """
    Отримання інформації про відео.
    
    Args:
        cap: VideoCapture об'єкт
        
    Returns:
        Словник з інформацією про відео
    """
    if not cap.isOpened():
        return {}
    
    return {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
    }


def create_video_writer(output_path: str, fps: float, width: int, height: int, codec: str = 'mp4v') -> cv2.VideoWriter:
    """
    Створення VideoWriter для запису відео.
    
    Args:
        output_path: Шлях для збереження відео
        fps: Кадри за секунду
        width: Ширина відео
        height: Висота відео
        codec: Кодек для відео
        
    Returns:
        VideoWriter об'єкт
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


class FPSCounter:
    """
    Лічильник FPS для моніторингу продуктивності.
    """
    
    def __init__(self, window_size: int = 30):
        """
        Ініціалізація лічильника FPS.
        
        Args:
            window_size: Розмір вікна для усереднення FPS
        """
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
        self.fps = 0.0
    
    def update(self) -> float:
        """
        Оновлення лічильника FPS.
        
        Returns:
            Поточний FPS
        """
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(frame_time)
        
        # Обмеження розміру вікна
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        # Обчислення FPS
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        return self.fps
    
    def get_fps(self) -> float:
        """
        Отримання поточного FPS.
        
        Returns:
            Поточний FPS
        """
        return self.fps


class VideoProcessor:
    """
    Клас для обробки відео з різними налаштуваннями.
    """
    
    def __init__(self, source: str):
        """
        Ініціалізація обробника відео.
        
        Args:
            source: Джерело відео
        """
        self.is_image, self.frame, self.cap = load_source(source)
        self.fps_counter = FPSCounter()
        
        if not self.is_image:
            self.video_info = get_video_info(self.cap)
            self.frame_time = 1.0 / self.video_info['fps'] if self.video_info['fps'] > 0 else 0
        else:
            self.video_info = {}
            self.frame_time = 0
    
    def read_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Читання наступного кадру.
        
        Returns:
            Tuple (success, frame)
        """
        if self.is_image:
            return True, self.frame
        else:
            return self.cap.read()
    
    def get_video_info(self) -> dict:
        """
        Отримання інформації про відео.
        
        Returns:
            Словник з інформацією
        """
        return self.video_info
    
    def release(self):
        """
        Звільнення ресурсів.
        """
        if self.cap is not None:
            self.cap.release()


class DisplayManager:
    """
    Менеджер для відображення результатів на екрані.
    """
    
    def __init__(self, window_name: str = "YOLO Detection"):
        """
        Ініціалізація менеджера відображення.
        
        Args:
            window_name: Назва вікна
        """
        self.window_name = window_name
        self.fps_counter = FPSCounter()
    
    def create_info_overlay(self, frame: cv2.Mat, detector_stats: dict, video_fps: float, 
                          display_mode: str, preprocessing_name: str, sharpness: int) -> cv2.Mat:
        """
        Створення overlay з інформацією на кадрі.
        
        Args:
            frame: Вхідний кадр
            detector_stats: Статистика детектора
            video_fps: FPS відео
            display_mode: Режим відображення
            preprocessing_name: Назва методу preprocessing
            sharpness: Рівень різкості
            
        Returns:
            Кадр з overlay інформацією
        """
        result = frame.copy()
        
        # Оновлення FPS лічильника
        current_fps = self.fps_counter.update()
        
        # Основна інформація про FPS
        fps_text = f"Video FPS: {video_fps:.2f} | Detection FPS: {detector_stats.get('processing_fps', 0):.2f} | Display FPS: {current_fps:.2f}"
        cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Інформація про режим та налаштування
        mode_text = f"{display_mode} (Sharpness: {sharpness}, Preproc: {preprocessing_name})"
        cv2.putText(result, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Інформація про downsampling (якщо доступна)
        if 'spatial_scale' in detector_stats:
            downsampling_text = f"Spatial: {detector_stats['spatial_scale']:.2f} | Temporal: {detector_stats['temporal_skip']} | ROI: {detector_stats.get('using_roi', False)}"
            cv2.putText(result, downsampling_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return result
    
    def add_help_overlay(self, frame: cv2.Mat) -> cv2.Mat:
        """
        Додавання overlay з підказками.
        
        Args:
            frame: Вхідний кадр
            
        Returns:
            Кадр з підказками
        """
        result = frame.copy()
        height = result.shape[0]
        
        help_texts = [
            "Press 'g' for grayscale, 'c' for color, 'p' for color processed, 'm' for model view",
            "Press '1', '2', '3' to change sharpness level", 
            "Press '0'-'6' to change preprocessing method",
            "Press 's' to toggle spatial scale | 'r' to toggle ROI | 'q' to quit"
        ]
        
        for i, text in enumerate(help_texts):
            y_pos = height - (len(help_texts) - i) * 25 - 15
            cv2.putText(result, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return result
    
    def show(self, frame: cv2.Mat) -> int:
        """
        Відображення кадру та обробка клавіатурного вводу.
        
        Args:
            frame: Кадр для відображення
            
        Returns:
            Код натиснутої клавіші
        """
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1) & 0xFF
    
    def close(self):
        """
        Закриття вікна відображення.
        """
        cv2.destroyAllWindows()


class FrameBuffer:
    """
    Буфер кадрів для згладжування відео потоку.
    """
    
    def __init__(self, buffer_size: int = 5):
        """
        Ініціалізація буфера кадрів.
        
        Args:
            buffer_size: Розмір буфера
        """
        self.buffer_size = buffer_size
        self.frames = []
        self.current_index = 0
    
    def add_frame(self, frame: cv2.Mat):
        """
        Додавання кадру в буфер.
        
        Args:
            frame: Кадр для додавання
        """
        if len(self.frames) < self.buffer_size:
            self.frames.append(frame.copy())
        else:
            self.frames[self.current_index] = frame.copy()
            self.current_index = (self.current_index + 1) % self.buffer_size
    
    def get_latest_frame(self) -> Optional[cv2.Mat]:
        """
        Отримання останнього кадру з буфера.
        
        Returns:
            Останній кадр або None
        """
        if not self.frames:
            return None
        
        latest_index = (self.current_index - 1) % len(self.frames)
        return self.frames[latest_index]
    
    def clear(self):
        """
        Очищення буфера.
        """
        self.frames.clear()
        self.current_index = 0


def validate_source(source: str) -> bool:
    """
    Перевірка валідності джерела відео/зображення.
    
    Args:
        source: Шлях до джерела
        
    Returns:
        True якщо джерело валідне
    """
    if source == "0":
        return True  # Веб-камера
    
    if not os.path.exists(source):
        return False
    
    # Перевірка чи це зображення або відео
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    video_formats = ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm']
    
    ext = source.split('.')[-1].lower()
    return ext in img_formats + video_formats