"""
Optimized YOLO Detector Module
Містить оптимізований детектор з підтримкою downsampling та багатопоточності
"""

import cv2
import time
import threading
import queue
import numpy as np
from yolo_detection import yolo_detection
from downsampling import DownsamplingManager
from threated_pipeline import ThreadedPipelineDetector

class OptimizedYOLODetectorWithDownsampling:
    """
    Оптимізований YOLO детектор з різними алгоритмами downsampling
    """
    def __init__(self, model_path, names_path, IMAGE_SIZE=640, use_threaded_pipeline=False):
        # Основна ініціалізація
        self.model = cv2.dnn.readNet(model_path)
        self.IMAGE_SIZE = IMAGE_SIZE
        
        with open(names_path, "r") as f:
            self.NAMES = [cname.strip() for cname in f.readlines()]
        self.COLORS = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.NAMES]
        
        # Ініціалізація downsampling менеджера
        self.downsampling_manager = DownsamplingManager()
        
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
            "show_color": False,
            "use_color_processing": False,
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
    
    def set_downsampling_params(self, spatial_scale=1.0, temporal_skip=1, use_roi=False, adaptive_skipping=True, target_fps=15):
        """
        Налаштування параметрів downsampling
        """
        self.downsampling_manager.configure(
            spatial_scale=spatial_scale,
            temporal_skip=temporal_skip,
            use_roi=use_roi,
            adaptive_temporal=adaptive_skipping,
            target_fps=target_fps
        )
        print(f"Downsampling params: spatial={spatial_scale:.2f}, temporal={temporal_skip}, roi={use_roi}, adaptive={adaptive_skipping}")
    
    def _processing_loop(self):
        """
        Основний цикл обробки з підтримкою downsampling
        """
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                settings = self.current_settings.copy()
                start_time = time.time()
                
                # Застосування downsampling
                downsampling_result = self.downsampling_manager.process_frame(frame, self.processing_fps)
                
                if not downsampling_result['should_process']:
                    continue
                
                processing_frame = downsampling_result['processed_frame']
                
                # Виконання детекції на обробленому кадрі
                overlay, gray_frame, model_input = yolo_detection(
                    processing_frame, self.model, self.IMAGE_SIZE, self.NAMES, self.COLORS, settings
                )
                
                # Відновлення результатів до оригінального розміру
                full_overlay = np.zeros_like(frame, dtype=np.uint8)
                full_gray_frame = None
                
                if overlay is not None:
                    full_overlay = self._restore_overlay(
                        overlay, frame.shape, downsampling_result
                    )
                
                if gray_frame is not None:
                    full_gray_frame = self._restore_gray_frame(
                        gray_frame, frame.shape, downsampling_result
                    )
                
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
    
    def _restore_overlay(self, overlay, original_shape, downsampling_result):
        """
        Відновлення overlay до оригінального розміру
        """
        full_overlay = np.zeros(original_shape, dtype=np.uint8)
        
        if overlay is None:
            return full_overlay
        
        current_overlay = overlay.copy()
        
        # Спочатку відновлюємо spatial downsampling
        if downsampling_result['spatial_scale'] < 1.0:
            # Визначаємо цільовий розмір для upsampling
            if downsampling_result['roi'] is not None:
                # Якщо використовувався ROI, відновлюємо до розміру ROI
                roi = downsampling_result['roi']
                target_size = (roi[2], roi[3])  # width, height з ROI
            else:
                # Якщо ROI не використовувався, відновлюємо до повного розміру
                target_size = (original_shape[1], original_shape[0])
            
            current_overlay = cv2.resize(current_overlay, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Потім розміщуємо в правильному місці (ROI restoration)
        if downsampling_result['roi'] is not None:
            x, y = downsampling_result['roi_offset']
            h_overlay, w_overlay = current_overlay.shape[:2]
            
            # Перевіряємо межі, щоб не вийти за кадр
            end_y = min(y + h_overlay, original_shape[0])
            end_x = min(x + w_overlay, original_shape[1])
            
            # Обрізаємо overlay якщо потрібно
            crop_h = end_y - y
            crop_w = end_x - x
            
            full_overlay[y:end_y, x:end_x] = current_overlay[:crop_h, :crop_w]
        else:
            # Без ROI - просто копіюємо весь overlay
            if current_overlay.shape[:2] == original_shape[:2]:
                full_overlay = current_overlay
            else:
                full_overlay = cv2.resize(current_overlay, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        return full_overlay
    
    def _restore_gray_frame(self, gray_frame, original_shape, downsampling_result):
        """
        Відновлення gray frame до оригінального розміру
        """
        if gray_frame is None:
            return None
        
        # Створюємо повний gray frame з оригінального кадру
        full_gray_frame = np.zeros(original_shape[:2], dtype=np.uint8)
        current_gray = gray_frame.copy()
        
        # Відновлення spatial downsampling
        if downsampling_result['spatial_scale'] < 1.0:
            if downsampling_result['roi'] is not None:
                # Відновлюємо до розміру ROI
                roi = downsampling_result['roi']
                target_size = (roi[2], roi[3])  # width, height
            else:
                # Відновлюємо до повного розміру
                target_size = (original_shape[1], original_shape[0])
            
            current_gray = cv2.resize(current_gray, target_size, interpolation=cv2.INTER_CUBIC)
        
        # ROI restoration
        if downsampling_result['roi'] is not None:
            # Заповнюємо тільки ROI область, решта залишається чорною
            x, y = downsampling_result['roi_offset']
            h_gray, w_gray = current_gray.shape[:2]
            
            # Перевіряємо межі
            end_y = min(y + h_gray, original_shape[0])
            end_x = min(x + w_gray, original_shape[1])
            
            # Обрізаємо якщо потрібно
            crop_h = end_y - y
            crop_w = end_x - x
            
            full_gray_frame[y:end_y, x:end_x] = current_gray[:crop_h, :crop_w]
        else:
            # Без ROI - використовуємо весь кадр
            if current_gray.shape == original_shape[:2]:
                full_gray_frame = current_gray
            else:
                full_gray_frame = cv2.resize(current_gray, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
        
        return full_gray_frame
    
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
        downsampling_stats = self.downsampling_manager.get_stats()
        return {
            'processing_fps': self.processing_fps,
            **downsampling_stats
        }
    
    def stop(self):
        """
        Зупинка детектора
        """
        self.running = False
        self.processing_thread.join(timeout=1.0)


class ThreadedYOLODetector:
    """
    Багатопоточна версія YOLO детектора з окремими потоками для preprocessing та detection
    """
    def __init__(self, model_path, names_path, IMAGE_SIZE=640):
        # Тимчасово відключаємо threaded_pipeline через ImportError
        try:
            self.pipeline = ThreadedPipelineDetector(model_path, names_path, IMAGE_SIZE)
            self.use_pipeline = True
        except ImportError:
            print("Warning: threaded_pipeline module not available, falling back to standard detector")
        
        # Використовуємо стандартний детектор
        self.detector = OptimizedYOLODetectorWithDownsampling(model_path, names_path, IMAGE_SIZE)
        self.use_pipeline = False
        
        # Додаємо current_settings для сумісності
        self.current_settings = {
            "tresh": 0.25,
            "thickness": 2,
            "show_grayscale": True,
            "show_model_view": False,
            "show_color": False,
            "use_color_processing": False,
            "sharpness": 1,
            "preprocessing": 0
        }
        
        # Додаємо downsampling_manager для сумісності з ROI візуалізацією
        if self.use_pipeline:
            self.downsampling_manager = self.pipeline.downsampling_manager
        else:
            self.downsampling_manager = self.detector.downsampling_manager
    
    def set_downsampling_params(self, **kwargs):
        """
        Налаштування параметрів downsampling
        """
        if self.use_pipeline:
            self.pipeline.configure_downsampling(**kwargs)
        else:
            self.detector.set_downsampling_params(**kwargs)
    
    def update_settings(self, **kwargs):
        """
        Оновлення налаштувань детекції
        """
        # Оновлюємо локальні настройки
        self.current_settings.update(kwargs)
        
        if self.use_pipeline:
            self.pipeline.update_detection_settings(**kwargs)
        else:
            self.detector.update_settings(**kwargs)
    
    def process_frame(self, frame):
        """
        Обробка кадру
        """
        if self.use_pipeline:
            self.pipeline.process_frame(frame)
        else:
            self.detector.process_frame(frame)
    
    def get_latest_results(self):
        """
        Отримання результатів
        """
        if self.use_pipeline:
            return self.pipeline.get_latest_results()
        else:
            return self.detector.get_latest_results()
    
    def get_performance_stats(self):
        """
        Отримання статистики
        """
        if self.use_pipeline:
            stats = self.pipeline.get_performance_stats()
            # Перетворюємо статистику для сумісності
            return {
                'processing_fps': stats.get('overall_fps', 0),
                'spatial_scale': stats.get('downsampling', {}).get('spatial_scale', 1.0),
                'temporal_skip': stats.get('downsampling', {}).get('temporal_skip', 1),
                'using_roi': stats.get('downsampling', {}).get('using_roi', False),
                'adaptive_skipping': stats.get('downsampling', {}).get('adaptive_skipping', False)
            }
        else:
            return self.detector.get_performance_stats()
    
    def stop(self):
        """
        Зупинка детектора
        """
        if self.use_pipeline:
            self.pipeline.stop()
        else:
            self.detector.stop()


class SimpleYOLODetector:
    """
    Простий YOLO детектор без downsampling для порівняння продуктивності
    """
    def __init__(self, model_path, names_path, IMAGE_SIZE=640):
        self.model = cv2.dnn.readNet(model_path)
        self.IMAGE_SIZE = IMAGE_SIZE
        
        with open(names_path, "r") as f:
            self.NAMES = [cname.strip() for cname in f.readlines()]
        self.COLORS = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.NAMES]
        
        self.current_settings = {
            "tresh": 0.25,
            "thickness": 2,
            "show_grayscale": True,
            "show_model_view": False,
            "show_color": False,
            "use_color_processing": False,
            "sharpness": 1,
            "preprocessing": 0
        }
    
    def update_settings(self, **kwargs):
        """
        Оновлення налаштувань детектора
        """
        self.current_settings.update(kwargs)
    
    def detect(self, frame):
        """
        Виконання детекції на кадрі
        """
        return yolo_detection(frame, self.model, self.IMAGE_SIZE, 
                            self.NAMES, self.COLORS, self.current_settings)


class DetectorBenchmark:
    """
    Клас для бенчмаркінгу різних детекторів
    """
    def __init__(self):
        self.results = {}
    
    def benchmark_detector(self, detector, frames, detector_name="detector"):
        """
        Бенчмаркінг детектора на наборі кадрів
        
        Args:
            detector: Детектор для тестування
            frames: Список кадрів для тестування
            detector_name: Назва детектора
            
        Returns:
            Dict з результатами бенчмаркінгу
        """
        start_time = time.time()
        processed_frames = 0
        
        for frame in frames:
            if hasattr(detector, 'process_frame'):
                # Для оптимізованого детектора
                detector.process_frame(frame)
                time.sleep(0.01)  # Невелика затримка для обробки
                detector.get_latest_results()
            else:
                # Для простого детектора
                detector.detect(frame)
            
            processed_frames += 1
        
        total_time = time.time() - start_time
        avg_fps = processed_frames / total_time
        
        result = {
            'detector_name': detector_name,
            'total_frames': processed_frames,
            'total_time': total_time,
            'average_fps': avg_fps,
            'ms_per_frame': (total_time / processed_frames) * 1000
        }
        
        self.results[detector_name] = result
        return result
    
    def compare_results(self):
        """
        Порівняння результатів бенчмаркінгу
        
        Returns:
            Відформатований звіт
        """
        if not self.results:
            return "No benchmark results available"
        
        report = "\n=== DETECTOR BENCHMARK RESULTS ===\n"
        for name, result in self.results.items():
            report += f"\n{result['detector_name']}:\n"
            report += f"  Average FPS: {result['average_fps']:.2f}\n"
            report += f"  MS per frame: {result['ms_per_frame']:.2f}\n"
            report += f"  Total frames: {result['total_frames']}\n"
            report += f"  Total time: {result['total_time']:.2f}s\n"
        
        # Знаходження найшвидшого детектора
        fastest = max(self.results.values(), key=lambda x: x['average_fps'])
        report += f"\nFastest detector: {fastest['detector_name']} ({fastest['average_fps']:.2f} FPS)\n"
        
        return report