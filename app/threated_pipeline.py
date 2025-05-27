"""
Multi-threaded Processing Pipeline
Пайплайн з окремими потоками для кожного етапу обробки
"""

import cv2
import time
import threading
import queue
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from downsampling import DownsamplingManager
from yolo_detection import yolo_detection
from image_processing import get_preprocessing_name


@dataclass
class FramePacket:
    """
    Пакет даних для передачі між потоками
    """
    frame_id: int
    original_frame: np.ndarray
    processed_frame: Optional[np.ndarray] = None
    roi_info: Optional[Dict] = None
    downsampling_info: Optional[Dict] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class FramePreprocessor:
    """
    Потік для попередньої обробки кадрів (ROI, downsampling, фільтри)
    """
    
    def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue, 
                 downsampling_manager: DownsamplingManager, max_workers=2):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.downsampling_manager = downsampling_manager
        self.max_workers = max_workers
        
        self.running = True
        self.workers = []
        
        # Статистика
        self.processed_frames = 0
        self.processing_times = deque(maxlen=30)
        self.avg_processing_time = 0.0
        
        # Запуск воркерів
        for i in range(max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self, worker_id: int):
        """
        Основний цикл воркера для попередньої обробки
        """
        while self.running:
            try:
                packet = self.input_queue.get(timeout=0.1)
                start_time = time.time()
                
                # Застосування downsampling
                downsampling_result = self.downsampling_manager.process_frame(
                    packet.original_frame, 
                    self.get_current_fps()
                )
                
                if downsampling_result['should_process']:
                    # Оновлення пакету з результатами обробки
                    packet.processed_frame = downsampling_result['processed_frame']
                    packet.downsampling_info = downsampling_result
                    packet.roi_info = {
                        'roi': downsampling_result.get('roi'),
                        'roi_offset': downsampling_result.get('roi_offset', (0, 0)),
                        'spatial_scale': downsampling_result.get('spatial_scale', 1.0)
                    }
                    
                    # Передача на наступний етап
                    try:
                        self.output_queue.put(packet, block=False)
                    except queue.Full:
                        # Пропускаємо кадр якщо черга переповнена
                        pass
                
                # Оновлення статистики
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                self.avg_processing_time = sum(self.processing_times) / len(self.processing_times)
                self.processed_frames += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in preprocessor worker {worker_id}: {e}")
    
    def get_current_fps(self):
        """
        Отримання поточного FPS обробки
        """
        if len(self.processing_times) > 1:
            return 1.0 / self.avg_processing_time
        return 0.0
    
    def get_stats(self):
        """
        Отримання статистики preprocessor
        """
        return {
            'processed_frames': self.processed_frames,
            'avg_processing_time': self.avg_processing_time,
            'current_fps': self.get_current_fps(),
            'queue_size': self.input_queue.qsize()
        }
    
    def stop(self):
        """
        Зупинка preprocessor
        """
        self.running = False
        for worker in self.workers:
            worker.join(timeout=1.0)


class YOLOProcessor:
    """
    Потік для YOLO детекції
    """
    
    def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue,
                 model_path: str, names_path: str, IMAGE_SIZE=640, max_workers=1):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.IMAGE_SIZE = IMAGE_SIZE
        self.max_workers = max_workers
        
        # Завантаження моделі для кожного воркера
        self.models = []
        self.names = []
        self.colors = []
        
        for _ in range(max_workers):
            model = cv2.dnn.readNet(model_path)
            self.models.append(model)
            
            with open(names_path, "r") as f:
                names = [cname.strip() for cname in f.readlines()]
            self.names.append(names)
            
            colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]
            self.colors.append(colors)
        
        # Налаштування детекції
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
        self.workers = []
        
        # Статистика
        self.processed_frames = 0
        self.detection_times = deque(maxlen=30)
        self.avg_detection_time = 0.0
        
        # Запуск воркерів
        for i in range(max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self, worker_id: int):
        """
        Основний цикл воркера для YOLO детекції
        """
        model = self.models[worker_id]
        names = self.names[worker_id]
        colors = self.colors[worker_id]
        
        while self.running:
            try:
                packet = self.input_queue.get(timeout=0.1)
                start_time = time.time()
                
                # Виконання YOLO детекції
                overlay, processed_frame, model_input = yolo_detection(
                    packet.processed_frame or packet.original_frame,
                    model, self.IMAGE_SIZE, names, colors, self.current_settings
                )
                
                # Відновлення overlay до оригінального розміру якщо потрібно
                if packet.roi_info and overlay is not None:
                    overlay = self._restore_overlay_to_original(
                        overlay, packet.original_frame.shape, packet.roi_info
                    )
                
                # Створення результуючого пакету
                result_packet = FramePacket(
                    frame_id=packet.frame_id,
                    original_frame=packet.original_frame,
                    processed_frame=processed_frame,
                    roi_info=packet.roi_info,
                    downsampling_info=packet.downsampling_info,
                    timestamp=packet.timestamp
                )
                
                # Додавання результатів детекції
                result_packet.overlay = overlay
                result_packet.model_input = model_input
                
                # Передача результату
                try:
                    self.output_queue.put(result_packet, block=False)
                except queue.Full:
                    # Видаляємо найстарший результат і додаємо новий
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put(result_packet, block=False)
                    except queue.Empty:
                        pass
                
                # Оновлення статистики
                detection_time = time.time() - start_time
                self.detection_times.append(detection_time)
                self.avg_detection_time = sum(self.detection_times) / len(self.detection_times)
                self.processed_frames += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in YOLO worker {worker_id}: {e}")
    
    def _restore_overlay_to_original(self, overlay, original_shape, roi_info):
        """
        Відновлення overlay до оригінального розміру з урахуванням ROI
        """
        full_overlay = np.zeros(original_shape, dtype=np.uint8)
        
        if roi_info and roi_info.get('roi'):
            # Spatial upsampling якщо потрібно
            if roi_info.get('spatial_scale', 1.0) < 1.0:
                roi = roi_info['roi']
                target_size = (roi[2], roi[3])  # width, height
                overlay = cv2.resize(overlay, target_size, interpolation=cv2.INTER_NEAREST)
            
            # Розміщення в ROI області
            x, y = roi_info.get('roi_offset', (0, 0))
            h_overlay, w_overlay = overlay.shape[:2]
            
            end_y = min(y + h_overlay, original_shape[0])
            end_x = min(x + w_overlay, original_shape[1])
            
            crop_h = end_y - y
            crop_w = end_x - x
            
            full_overlay[y:end_y, x:end_x] = overlay[:crop_h, :crop_w]
        else:
            # Без ROI - просто масштабування
            if overlay.shape[:2] != original_shape[:2]:
                full_overlay = cv2.resize(overlay, (original_shape[1], original_shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
            else:
                full_overlay = overlay
        
        return full_overlay
    
    def update_settings(self, **kwargs):
        """
        Оновлення налаштувань детекції
        """
        self.current_settings.update(kwargs)
    
    def get_stats(self):
        """
        Отримання статистики YOLO processor
        """
        return {
            'processed_frames': self.processed_frames,
            'avg_detection_time': self.avg_detection_time,
            'detection_fps': 1.0 / self.avg_detection_time if self.avg_detection_time > 0 else 0.0,
            'queue_size': self.input_queue.qsize()
        }
    
    def stop(self):
        """
        Зупинка YOLO processor
        """
        self.running = False
        for worker in self.workers:
            worker.join(timeout=1.0)


class ThreadedPipelineDetector:
    """
    Головний клас багатопоточного пайплайну детекції
    """
    
    def __init__(self, model_path: str, names_path: str, IMAGE_SIZE=640):
        # Черги для передачі даних між етапами
        self.input_queue = queue.Queue(maxsize=5)          # Вхідні кадри
        self.preprocessed_queue = queue.Queue(maxsize=3)   # Оброблені кадри
        self.result_queue = queue.Queue(maxsize=3)         # Результати детекції
        
        # Ініціалізація компонентів пайплайну
        self.downsampling_manager = DownsamplingManager()
        
        self.preprocessor = FramePreprocessor(
            self.input_queue, 
            self.preprocessed_queue,
            self.downsampling_manager,
            max_workers=2  # 2 потоки для preprocessing
        )
        
        self.yolo_processor = YOLOProcessor(
            self.preprocessed_queue,
            self.result_queue,
            model_path,
            names_path,
            IMAGE_SIZE,
            max_workers=1  # 1 потік для YOLO (можна збільшити якщо є кілька GPU)
        )
        
        # Глобальна статистика
        self.frame_counter = 0
        self.start_time = time.time()
        self.overall_fps = 0.0
        
        # Останні результати
        self.latest_result = None
        self.result_lock = threading.Lock()
        
        # Потік для збору результатів
        self.result_collector = threading.Thread(target=self._collect_results, daemon=True)
        self.result_collector.start()
    
    def _collect_results(self):
        """
        Потік для збору результатів з YOLO processor
        """
        while True:
            try:
                result_packet = self.result_queue.get(timeout=0.1)
                
                with self.result_lock:
                    self.latest_result = result_packet
                
                # Оновлення загальної статистики
                self.frame_counter += 1
                elapsed = time.time() - self.start_time
                if elapsed >= 1.0:
                    self.overall_fps = self.frame_counter / elapsed
                    self.frame_counter = 0
                    self.start_time = time.time()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in result collector: {e}")
    
    def configure_downsampling(self, spatial_scale=1.0, temporal_skip=1, use_roi=False, 
                             adaptive_temporal=True, target_fps=15):
        """
        Конфігурація параметрів downsampling
        """
        self.downsampling_manager.configure(
            spatial_scale=spatial_scale,
            temporal_skip=temporal_skip,
            use_roi=use_roi,
            adaptive_temporal=adaptive_temporal,
            target_fps=target_fps
        )
    
    def update_detection_settings(self, **kwargs):
        """
        Оновлення налаштувань детекції
        """
        self.yolo_processor.update_settings(**kwargs)
    
    def process_frame(self, frame: np.ndarray):
        """
        Додавання кадру в пайплайн обробки
        """
        packet = FramePacket(
            frame_id=self.frame_counter,
            original_frame=frame
        )
        
        try:
            self.input_queue.put(packet, block=False)
        except queue.Full:
            # Видаляємо найстарший кадр і додаємо новий
            try:
                self.input_queue.get_nowait()
                self.input_queue.put(packet, block=False)
            except queue.Empty:
                pass
    
    def get_latest_results(self):
        """
        Отримання останніх результатів обробки
        """
        with self.result_lock:
            if self.latest_result:
                return (
                    getattr(self.latest_result, 'overlay', None),
                    self.latest_result.processed_frame,
                    getattr(self.latest_result, 'model_input', None)
                )
            return None, None, None
    
    def get_performance_stats(self):
        """
        Отримання детальної статистики продуктивності
        """
        preprocessor_stats = self.preprocessor.get_stats()
        yolo_stats = self.yolo_processor.get_stats()
        downsampling_stats = self.downsampling_manager.get_stats()
        
        return {
            'overall_fps': self.overall_fps,
            'preprocessor': preprocessor_stats,
            'yolo_processor': yolo_stats,
            'downsampling': downsampling_stats,
            'queue_sizes': {
                'input': self.input_queue.qsize(),
                'preprocessed': self.preprocessed_queue.qsize(),
                'results': self.result_queue.qsize()
            }
        }
    
    def print_performance_stats(self):
        """
        Виведення детальної статистики продуктивності
        """
        stats = self.get_performance_stats()
        
        print("\n=== PIPELINE PERFORMANCE STATS ===")
        print(f"Overall FPS: {stats['overall_fps']:.2f}")
        print(f"Preprocessor FPS: {stats['preprocessor']['current_fps']:.2f}")
        print(f"YOLO Detection FPS: {stats['yolo_processor']['detection_fps']:.2f}")
        print(f"Queue sizes - Input: {stats['queue_sizes']['input']}, "
              f"Preprocessed: {stats['queue_sizes']['preprocessed']}, "
              f"Results: {stats['queue_sizes']['results']}")
        print("=" * 35)
    
    def stop(self):
        """
        Зупинка всього пайплайну
        """
        print("Stopping threaded pipeline...")
        self.preprocessor.stop()
        self.yolo_processor.stop()
        
        # Очищення черг
        self._clear_queue(self.input_queue)
        self._clear_queue(self.preprocessed_queue)
        self._clear_queue(self.result_queue)
    
    def _clear_queue(self, q):
        """
        Очищення черги
        """
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass


class AdaptiveThreadedPipelineDetector(ThreadedPipelineDetector):
    """
    Адаптивна версія пайплайну з автоматичним налаштуванням кількості потоків
    """
    
    def __init__(self, model_path: str, names_path: str, IMAGE_SIZE=640):
        super().__init__(model_path, names_path, IMAGE_SIZE)
        
        # Система моніторингу та адаптації
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
        self.adaptation_thread.start()
        
        self.target_overall_fps = 20
        self.adaptation_interval = 5.0  # секунд
        self.last_adaptation = time.time()
    
    def _adaptation_loop(self):
        """
        Цикл адаптації кількості потоків під навантаження
        """
        while True:
            try:
                time.sleep(self.adaptation_interval)
                
                if time.time() - self.last_adaptation < self.adaptation_interval:
                    continue
                
                stats = self.get_performance_stats()
                
                # Аналіз вузьких місць
                if stats['overall_fps'] < self.target_overall_fps * 0.8:
                    # Система працює повільно
                    if stats['preprocessor']['current_fps'] < stats['yolo_processor']['detection_fps']:
                        print("Bottleneck detected in preprocessor - consider increasing workers")
                    elif stats['yolo_processor']['detection_fps'] < stats['preprocessor']['current_fps']:
                        print("Bottleneck detected in YOLO processor - consider optimizing model")
                
                # Адаптація параметрів downsampling
                if stats['overall_fps'] < self.target_overall_fps * 0.6:
                    # Критично низька продуктивність - агресивні налаштування
                    current_scale = stats['downsampling']['spatial_scale']
                    if current_scale > 0.3:
                        new_scale = max(0.25, current_scale - 0.1)
                        self.downsampling_manager.spatial_downsampler.scale_factor = new_scale
                        print(f"Adaptive: reducing spatial scale to {new_scale}")
                
                self.last_adaptation = time.time()
                
            except Exception as e:
                print(f"Error in adaptation loop: {e}")
    
    def set_target_fps(self, target_fps: int):
        """
        Встановлення цільового FPS для адаптації
        """
        self.target_overall_fps = target_fps