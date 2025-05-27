"""
Downsampling Module
Містить алгоритми downsampling для оптимізації продуктивності
"""

import cv2
import numpy as np
import time


class SpatialDownsampler:
    """
    Клас для просторового downsampling зображень.
    """
    
    def __init__(self, scale_factor=1.0):
        """
        Ініціалізація з коефіцієнтом масштабування.
        
        Args:
            scale_factor: Коефіцієнт масштабування (0.1-1.0)
        """
        self.scale_factor = scale_factor
    
    def downsample(self, frame):
        """
        Зменшення роздільності кадру.
        
        Args:
            frame: Вхідний кадр
            
        Returns:
            Tuple (downsampled_frame, actual_scale_factor)
        """
        if self.scale_factor >= 1.0:
            return frame, 1.0
            
        height, width = frame.shape[:2]
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        
        # INTER_AREA найкраще для downsampling
        downsampled = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return downsampled, self.scale_factor
    
    def upsample(self, frame, target_size):
        """
        Збільшення роздільності назад до оригінального розміру.
        
        Args:
            frame: Зменшений кадр
            target_size: Цільовий розмір (width, height)
            
        Returns:
            Збільшений кадр
        """
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)


class TemporalDownsampler:
    """
    Клас для temporal downsampling (пропуск кадрів).
    """
    
    def __init__(self, skip_frames=1, adaptive=False, target_fps=15):
        """
        Ініціалізація temporal downsampler.
        
        Args:
            skip_frames: Кількість кадрів для пропуску
            adaptive: Використовувати адаптивний режим
            target_fps: Цільовий FPS для адаптивного режиму
        """
        self.skip_frames = skip_frames
        self.adaptive = adaptive
        self.target_fps = target_fps
        
        self.frame_counter = 0
        self.current_skip_rate = skip_frames
        self.processing_fps = 0.0
    
    def should_process_frame(self, current_fps=None):
        """
        Визначення, чи потрібно обробляти поточний кадр.
        
        Args:
            current_fps: Поточний FPS обробки
            
        Returns:
            True якщо кадр потрібно обробляти
        """
        self.frame_counter += 1
        
        if self.adaptive and current_fps is not None:
            self.processing_fps = current_fps
            # Адаптивне налаштування частоти обробки
            if self.processing_fps < self.target_fps * 0.8:
                self.current_skip_rate = min(self.current_skip_rate + 1, 5)
            elif self.processing_fps > self.target_fps * 1.2:
                self.current_skip_rate = max(self.current_skip_rate - 1, 1)
            
            return self.frame_counter % self.current_skip_rate == 0
        else:
            return self.frame_counter % self.skip_frames == 0
    
    def get_current_skip_rate(self):
        """
        Отримання поточної частоти пропуску кадрів.
        
        Returns:
            Поточна частота пропуску
        """
        return self.current_skip_rate if self.adaptive else self.skip_frames


class ROIDetector:
    """
    Клас для детекції областей інтересу (ROI) на основі руху.
    """
    
    def __init__(self, motion_threshold=30, min_area=1000, expand_ratio=0.2):
        """
        Ініціалізація ROI detector.
        
        Args:
            motion_threshold: Поріг для детекції руху
            min_area: Мінімальна площа контуру для врахування  
            expand_ratio: Коефіцієнт розширення ROI
        """
        self.motion_threshold = motion_threshold
        self.min_area = min_area
        self.expand_ratio = expand_ratio
        self.prev_frame = None
        self.roi_history = []  # Історія ROI для згладжування
        self.history_size = 5
    
    def detect_roi(self, current_frame):
        """
        Детекція ROI на основі руху між кадрами.
        
        Args:
            current_frame: Поточний кадр
            
        Returns:
            ROI координати (x, y, width, height) або None
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
        
        # Більш агресивна морфологічна обробка
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Додаткове розмиття для об'єднання близьких областей
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
                    
                    # Розширення ROI для захоплення контексту
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
                    
                    # Згладжування ROI через історію
                    return self._smooth_roi(roi, current_frame.shape[:2])
        
        return None
    
    def _smooth_roi(self, new_roi, frame_shape):
        """
        Згладжування ROI через історію для зменшення тремтіння
        """
        self.roi_history.append(new_roi)
        
        # Обмежуємо розмір історії
        if len(self.roi_history) > self.history_size:
            self.roi_history.pop(0)
        
        # Якщо історії недостатньо, повертаємо поточний ROI
        if len(self.roi_history) < 3:
            return new_roi
        
        # Усереднюємо координати з історії
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
    
    def crop_roi(self, frame, roi):
        """
        Обрізка кадру за ROI координатами.
        
        Args:
            frame: Вхідний кадр
            roi: ROI координати (x, y, width, height)
            
        Returns:
            Обрізаний кадр та offset координати
        """
        if roi is None:
            return frame, (0, 0)
        
        x, y, w, h = roi
        cropped = frame[y:y+h, x:x+w]
        return cropped, (x, y)


class DownsamplingManager:
    """
    Головний менеджер для управління всіма видами downsampling.
    """
    
    def __init__(self):
        """
        Ініціалізація менеджера downsampling.
        """
        self.spatial_downsampler = SpatialDownsampler()
        self.temporal_downsampler = TemporalDownsampler()
        self.roi_detector = ROIDetector()
        
        self.use_spatial = False
        self.use_temporal = False
        self.use_roi = False
    
    def configure(self, spatial_scale=1.0, temporal_skip=1, use_roi=False, 
                 adaptive_temporal=True, target_fps=15):
        """
        Конфігурація параметрів downsampling.
        
        Args:
            spatial_scale: Коефіцієнт просторового downsampling
            temporal_skip: Пропуск кадрів
            use_roi: Використовувати ROI детекцію
            adaptive_temporal: Адаптивний temporal downsampling
            target_fps: Цільовий FPS
        """
        # Spatial downsampling
        self.spatial_downsampler.scale_factor = spatial_scale
        self.use_spatial = spatial_scale < 1.0
        
        # Temporal downsampling
        self.temporal_downsampler.skip_frames = temporal_skip
        self.temporal_downsampler.adaptive = adaptive_temporal
        self.temporal_downsampler.target_fps = target_fps
        self.use_temporal = temporal_skip > 1 or adaptive_temporal
        
        # ROI detection
        self.use_roi = use_roi
    
    def process_frame(self, frame, current_fps=None):
        """
        Обробка кадру з застосуванням всіх налаштованих методів downsampling.
        
        Args:
            frame: Вхідний кадр
            current_fps: Поточний FPS обробки
            
        Returns:
            Dict з результатами обробки
        """
        result = {
            'should_process': True,
            'processed_frame': frame,
            'spatial_scale': 1.0,
            'roi': None,
            'roi_offset': (0, 0)
        }
        
        # Temporal downsampling
        if self.use_temporal:
            result['should_process'] = self.temporal_downsampler.should_process_frame(current_fps)
            if not result['should_process']:
                return result
        
        # ROI detection
        if self.use_roi:
            roi = self.roi_detector.detect_roi(frame)
            if roi:
                result['processed_frame'], result['roi_offset'] = self.roi_detector.crop_roi(frame, roi)
                result['roi'] = roi
        
        # Spatial downsampling
        if self.use_spatial:
            result['processed_frame'], result['spatial_scale'] = self.spatial_downsampler.downsample(
                result['processed_frame']
            )
        
        return result
    
    def restore_results(self, processed_results, original_frame_shape):
        """
        Відновлення результатів до оригінального розміру кадру.
        
        Args:
            processed_results: Результати обробки (overlay, тощо)
            original_frame_shape: Форма оригінального кадру
            
        Returns:
            Відновлені результати
        """
        overlay, gray_frame = processed_results
        result = {
            'overlay': np.zeros(original_frame_shape, dtype=np.uint8),
            'gray_frame': None
        }
        
        if overlay is not None:
            # Відновлення overlay
            if self.use_spatial and self.spatial_downsampler.scale_factor < 1.0:
                # Spatial upsampling
                target_size = (original_frame_shape[1], original_frame_shape[0])
                if self.use_roi:
                    # Враховуємо ROI
                    # Тут потрібна додаткова логіка для правильного відновлення
                    pass
                else:
                    result['overlay'] = self.spatial_downsampler.upsample(overlay, target_size)
            else:
                result['overlay'] = overlay
        
        if gray_frame is not None:
            # Аналогічне відновлення для gray_frame
            result['gray_frame'] = gray_frame
        
        return result
    
    def get_stats(self):
        """
        Отримання статистики downsampling.
        
        Returns:
            Dict зі статистикою
        """
        return {
            'spatial_scale': self.spatial_downsampler.scale_factor,
            'temporal_skip': self.temporal_downsampler.get_current_skip_rate(),
            'using_roi': self.use_roi,
            'adaptive_temporal': self.temporal_downsampler.adaptive,
            'target_fps': self.temporal_downsampler.target_fps
        }