"""
Main Application Module with Metrics Support
Головний файл для запуску YOLO детектора з модульною архітектурою та метриками
"""

import argparse
import time
import cv2
import sys
import os
import numpy as np

# Імпорт модулів
from optimized_detector import OptimizedYOLODetectorWithDownsampling, SimpleYOLODetector, DetectorBenchmark, ThreadedYOLODetector
from video_utils import VideoProcessor, DisplayManager, validate_source
from image_processing import get_preprocessing_name


def calculate_iou(box1, box2):
    """Обчислює IoU (Intersection over Union) між двома прямокутниками."""
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2
    
    # Координати перетину
    xi1 = max(x1, x1_gt)
    yi1 = max(y1, y1_gt)
    xi2 = min(x2, x2_gt)
    yi2 = min(y2, y2_gt)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    # Площі
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def calculate_average_precision(confidences, ious, iou_threshold=0.5):
    """Обчислює Average Precision для одного класу."""
    if not confidences:
        return 0.0
    
    # Сортуємо за confidence (від найвищого до найнижчого)
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_confidences = [confidences[i] for i in sorted_indices]
    sorted_ious = [ious[i] for i in sorted_indices]
    
    # True positives та false positives
    tp = []
    fp = []
    
    for iou in sorted_ious:
        if iou >= iou_threshold:
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    
    # Накопичувальні суми
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Precision і Recall
    precisions = []
    recalls = []
    
    total_gt = len([iou for iou in ious if iou >= iou_threshold])
    if total_gt == 0:
        return 0.0
    
    for i in range(len(tp)):
        precision = tp_cumsum[i] / (tp_cumsum[i] + fp_cumsum[i]) if (tp_cumsum[i] + fp_cumsum[i]) > 0 else 0
        recall = tp_cumsum[i] / total_gt
        precisions.append(precision)
        recalls.append(recall)
    
    # Обчислення AP методом інтерполяції (11-point interpolation)
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        p_max = 0.0
        for i in range(len(recalls)):
            if recalls[i] >= t:
                p_max = max(p_max, precisions[i])
        ap += p_max / 11.0
    
    return ap


def calculate_map(all_detections, confidence_threshold=0.25, iou_threshold=0.5):
    """Обчислює mAP для всіх детекцій."""
    if not all_detections:
        return 0.0
    
    # Групуємо детекції за класами
    detections_by_class = {}
    for detection in all_detections:
        class_id = detection['class_id']
        if class_id not in detections_by_class:
            detections_by_class[class_id] = []
        detections_by_class[class_id].append(detection)
    
    # Обчислюємо AP для кожного класу
    average_precisions = []
    
    for class_id, detections in detections_by_class.items():
        if len(detections) < 2:
            continue
            
        # Сортуємо за confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Для спрощення, використовуємо топ 50% детекцій як ground truth
        ground_truth_count = max(1, len(detections) // 2)
        ground_truth = detections[:ground_truth_count]
        
        confidences = []
        ious = []
        
        for detection in detections:
            confidences.append(detection['confidence'])
            # Обчислюємо максимальний IoU з ground truth
            max_iou = 0.0
            for gt in ground_truth:
                iou = calculate_iou(detection['box'], gt['box'])
                max_iou = max(max_iou, iou)
            ious.append(max_iou)
        
        ap = calculate_average_precision(confidences, ious, iou_threshold)
        average_precisions.append(ap)
    
    # mAP - середнє значення AP по всіх класах
    return np.mean(average_precisions) if average_precisions else 0.0


class MetricsCollector:
    """Клас для збору метрик детекції"""
    
    def __init__(self):
        self.all_detections = []
        self.total_processed_frames = 0
        self.detected_frames_count = 0
        self.total_detected_confidences = 0.0
        self.start_time = time.time()
    
    def add_detections(self, detections):
        """Додавання детекцій з кадру"""
        if detections:
            self.all_detections.extend(detections)
            frame_confidences = [det['confidence'] for det in detections]
            if frame_confidences:
                self.total_detected_confidences += np.mean(frame_confidences)
                self.detected_frames_count += 1
    
    def add_frame(self):
        """Додавання обробленого кадру"""
        self.total_processed_frames += 1
    
    def get_final_metrics(self, confidence_threshold=0.25):
        """Отримання фінальних метрик"""
        overall_elapsed_time = time.time() - self.start_time
        average_detection_fps = self.total_processed_frames / overall_elapsed_time if overall_elapsed_time > 0 else 0
        map_score = calculate_map(self.all_detections, confidence_threshold)
        avg_confidence = self.total_detected_confidences / self.detected_frames_count if self.detected_frames_count > 0 else 0.0
        
        return {
            'detection_fps': average_detection_fps,
            'map_score': map_score,
            'total_time': overall_elapsed_time,
            'total_frames': self.total_processed_frames,
            'total_detections': len(self.all_detections),
            'avg_confidence': avg_confidence
        }


class YOLOApplication:
    """
    Головний клас додатку YOLO детектора з підтримкою метрик
    """
    
    def __init__(self, args):
        """
        Ініціалізація додатку з аргументами командного рядка
        """
        self.args = args
        self.detector = None
        self.video_processor = None
        self.display_manager = DisplayManager()
        self.running = True
        self.metrics_collector = MetricsCollector()
        
        # Валідація джерела
        if not validate_source(args.source):
            raise ValueError(f"Invalid source: {args.source}")
    
    def setup_detector(self):
        """
        Налаштування детектора згідно з параметрами
        """
        print("Initializing YOLO detector...")
        
        if self.args.use_simple_detector:
            # Простий детектор без downsampling
            self.detector = SimpleYOLODetector(self.args.model, self.args.names, 640)
            self.detector_type = "Simple Detector"
        elif self.args.use_threaded_pipeline:
            # Багатопоточний пайплайн
            self.detector = ThreadedYOLODetector(self.args.model, self.args.names, 640)
            self.detector_type = "Threaded Detector"
            
            # Налаштування downsampling параметрів
            self.detector.set_downsampling_params(
                spatial_scale=self.args.spatial_scale,
                temporal_skip=self.args.temporal_skip,
                use_roi=self.args.use_roi,
                adaptive_skipping=self.args.adaptive_fps,
                target_fps=self.args.target_fps
            )
        else:
            # Оптимізований детектор з downsampling
            self.detector = OptimizedYOLODetectorWithDownsampling(self.args.model, self.args.names, 640)
            self.detector_type = "Optimized Detector"
            
            # Налаштування downsampling параметрів
            self.detector.set_downsampling_params(
                spatial_scale=self.args.spatial_scale,
                temporal_skip=self.args.temporal_skip,
                use_roi=self.args.use_roi,
                adaptive_skipping=self.args.adaptive_fps,
                target_fps=self.args.target_fps
            )
        
        # Оновлення основних налаштувань
        self.detector.update_settings(
            tresh=self.args.tresh,
            thickness=self.args.thickness,
            show_grayscale=self.args.show_grayscale,
            show_model_view=self.args.show_model_view,
            show_color=self.args.show_color,
            use_color_processing=self.args.use_color_processing,
            sharpness=self.args.sharpness,
            preprocessing=self.args.preprocessing
        )
        
        print("Detector initialized successfully!")
    
    def setup_video_processor(self):
        """
        Налаштування обробника відео
        """
        print("Setting up video processor...")
        self.video_processor = VideoProcessor(self.args.source)
        
        if not self.video_processor.is_image:
            video_info = self.video_processor.get_video_info()
            print(f"Video info: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.2f} FPS")
            print(f"Duration: {video_info['duration']:.2f} seconds")
        else:
            print("Processing static image")
    
    def extract_detections_from_overlay(self, overlay, confidence_threshold=0.25):
        """
        Витягання детекцій з overlay для збору метрик
        Це спрощена версія - в реальній реалізації потрібно отримувати детекції безпосередньо з детектора
        """
        detections = []
        
        # Якщо детектор має метод для отримання детекцій, використовуємо його
        if hasattr(self.detector, 'get_latest_detections'):
            return self.detector.get_latest_detections()
        
        # Інакше намагаємося виділити детекції з overlay (обмежений підхід)
        if overlay is not None:
            # Знаходимо контури на overlay
            gray_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray_overlay, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) > 100:  # Фільтр малих областей
                    x, y, w, h = cv2.boundingRect(contour)
                    # Створюємо псевдо-детекцію
                    detections.append({
                        'box': [x, y, x + w, y + h],
                        'confidence': confidence_threshold + 0.1,  # Псевдо-confidence
                        'class_id': 0  # Невідомий клас
                    })
        
        return detections
    
    def process_static_image(self):
        """
        Обробка статичного зображення
        """
        print("Processing static image...")
        
        if hasattr(self.detector, 'process_frame'):
            # Оптимізований детектор
            self.detector.process_frame(self.video_processor.frame)
            time.sleep(1)  # Даємо час на обробку
            overlay, processed_frame, model_input = self.detector.get_latest_results()
        else:
            # Простий детектор
            overlay, processed_frame, model_input = self.detector.detect(self.video_processor.frame)
        
        # Збираємо метрики
        detections = self.extract_detections_from_overlay(overlay, self.args.tresh)
        self.metrics_collector.add_detections(detections)
        self.metrics_collector.add_frame()
        
        if overlay is not None:
            result = cv2.addWeighted(self.video_processor.frame, 1.0, overlay, 1.0, 0)
            
            # Додавання інформації
            if hasattr(self.detector, 'get_performance_stats'):
                stats = self.detector.get_performance_stats()
            else:
                stats = {}
            
            result = self.display_manager.create_info_overlay(
                result, stats, 0, "Static Image", 
                get_preprocessing_name(self.args.preprocessing), 
                self.args.sharpness
            )
            
            cv2.imshow("YOLO Detection", result)
            cv2.waitKey(0)
    
    def process_video(self):
        """
        Обробка відео потоку
        """
        print("Starting video processing...")
        
        video_info = self.video_processor.get_video_info()
        video_fps = video_info.get('fps', 30)
        frame_time = 1.0 / video_fps if video_fps > 0 else 0
        
        # Виведення початкової інформації
        if not self.args.use_simple_detector:
            print(f"Downsampling settings:")
            print(f"  - Spatial scale: {self.args.spatial_scale}")
            print(f"  - Temporal skip: {self.args.temporal_skip}")
            print(f"  - Use ROI: {self.args.use_roi}")
            print(f"  - Adaptive FPS: {self.args.adaptive_fps}")
            print(f"  - Target detection FPS: {self.args.target_fps}")
        
        try:
            while self.running:
                start_frame_time = time.time()
                
                # Читання кадру
                ret, frame = self.video_processor.read_frame()
                if not ret:
                    break
                
                # Обробка кадру детектором
                if hasattr(self.detector, 'process_frame'):
                    # Оптимізований детектор
                    self.detector.process_frame(frame)
                    overlay, processed_frame, model_input = self.detector.get_latest_results()
                    detector_stats = self.detector.get_performance_stats()
                else:
                    # Простий детектор
                    overlay, processed_frame, model_input = self.detector.detect(frame)
                    detector_stats = {}
                
                # Збираємо метрики
                detections = self.extract_detections_from_overlay(overlay, self.args.tresh)
                self.metrics_collector.add_detections(detections)
                self.metrics_collector.add_frame()
                
                # Формування результату для відображення
                result = self._create_display_frame(frame, overlay, processed_frame, model_input)
                
                # Додавання інформаційного overlay
                result = self.display_manager.create_info_overlay(
                    result, detector_stats, video_fps,
                    self._get_display_mode(), 
                    get_preprocessing_name(self.detector.current_settings["preprocessing"]),
                    self.detector.current_settings["sharpness"]
                )
                
                # Додавання підказок
                result = self.display_manager.add_help_overlay(result)
                
                # Відображення та обробка клавіатури
                key = self.display_manager.show(result)
                if not self._handle_keyboard_input(key):
                    break
                
                # Контроль FPS
                self._control_fps(start_frame_time, frame_time)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self._cleanup()
    
    def _create_display_frame(self, frame, overlay, processed_frame, model_input):
        """
        Створення кадру для відображення залежно від режиму
        """
        if overlay is not None:
            settings = self.detector.current_settings if hasattr(self.detector, 'current_settings') else {}
            
            if settings.get("show_model_view", False) and model_input is not None:
                # Відображаємо те, що бачить модель
                model_view_resized = cv2.resize(model_input, (frame.shape[1], frame.shape[0]))
                result = cv2.addWeighted(model_view_resized, 1.0, overlay, 1.0, 0)
            elif settings.get("show_color", False):
                # Показуємо кольорове оброблене зображення або оригінал
                if settings.get("use_color_processing", False) and processed_frame is not None and len(processed_frame.shape) == 3:
                    result = cv2.addWeighted(processed_frame, 1.0, overlay, 1.0, 0)
                else:
                    result = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)
            elif settings.get("show_grayscale", True) and processed_frame is not None:
                # Показуємо чорно-біле зображення
                if len(processed_frame.shape) == 2:
                    processed_frame_3ch = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
                else:
                    processed_frame_3ch = processed_frame
                result = cv2.addWeighted(processed_frame_3ch, 1.0, overlay, 1.0, 0)
            else:
                # Показуємо кольорове зображення за замовчуванням
                result = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)
        else:
            # Якщо немає результатів, показуємо просто кадр
            result = frame.copy()
        
        # Додавання візуалізації ROI якщо він активний
        if (hasattr(self.detector, 'downsampling_manager') and 
            self.detector.downsampling_manager.use_roi and 
            hasattr(self.detector.downsampling_manager.roi_detector, 'roi_history') and
            self.detector.downsampling_manager.roi_detector.roi_history):
            
            # Отримуємо останній ROI з історії
            last_roi = self.detector.downsampling_manager.roi_detector.roi_history[-1]
            if last_roi:
                x, y, w, h = last_roi
                # Малюємо ROI рамку зеленим кольором
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result, f"ROI: {w}x{h}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return result
    
    def _get_display_mode(self):
        """
        Отримання поточного режиму відображення
        """
        if not hasattr(self.detector, 'current_settings'):
            return "Simple Detector Mode"
        
        settings = self.detector.current_settings
        if settings.get("show_model_view", False):
            return "Model View Mode"
        elif settings.get("show_color", False):
            if settings.get("use_color_processing", False):
                return "Color Processed Mode"
            else:
                return "Color Original Mode"
        elif settings.get("show_grayscale", True):
            return "Grayscale Processed Mode"
        else:
            return "Default Mode"
    
    def _handle_keyboard_input(self, key):
        """
        Обробка клавіатурного вводу
        """
        if key == ord('q'):
            return False
        
        # Базові команди для всіх детекторів
        if key == ord('g'):
            self.detector.update_settings(show_grayscale=True, show_model_view=False, show_color=False)
            print("Switched to Grayscale mode")
        elif key == ord('m'):
            self.detector.update_settings(show_model_view=True, show_grayscale=False, show_color=False)
            print("Switched to Model View mode")
        elif key == ord('c'):
            self.detector.update_settings(show_grayscale=False, show_model_view=False, show_color=True, use_color_processing=False)
            print("Switched to Color Original mode")
        elif key == ord('p'):  # Color Processed mode
            self.detector.update_settings(show_grayscale=False, show_model_view=False, show_color=True, use_color_processing=True)
            print("Switched to Color Processed mode")
        elif key == ord('1'):
            self.detector.update_settings(sharpness=1)
            print("Sharpness level set to 1 (Standard)")
        elif key == ord('2'):
            self.detector.update_settings(sharpness=2)
            print("Sharpness level set to 2 (Strong)")
        elif key == ord('3'):
            self.detector.update_settings(sharpness=3)
            print("Sharpness level set to 3 (Extreme)")
        elif key in [ord('0'), ord('4'), ord('5'), ord('6')]:
            preproc_idx = int(chr(key))
            if 0 <= preproc_idx <= 6:
                self.detector.update_settings(preprocessing=preproc_idx)
                print(f"Preprocessing method set to {preproc_idx} ({get_preprocessing_name(preproc_idx)})")
        
        # Команди для оптимізованого детектора
        elif hasattr(self.detector, 'set_downsampling_params'):
            self._handle_downsampling_keys(key)
        
        return True
    
    def _handle_downsampling_keys(self, key):
        """
        Обробка клавіш для downsampling (тільки для оптимізованого детектора)
        """
        if key == ord('s'):
            # Переключення spatial scale
            current_scale = self.detector.downsampling_manager.spatial_downsampler.scale_factor
            if current_scale >= 1.0:
                new_scale = 0.75
            elif current_scale >= 0.75:
                new_scale = 0.5
            elif current_scale >= 0.5:
                new_scale = 0.25
            else:
                new_scale = 1.0
            
            self.detector.set_downsampling_params(
                spatial_scale=new_scale,
                temporal_skip=self.args.temporal_skip,
                use_roi=self.args.use_roi,
                adaptive_skipping=self.args.adaptive_fps,
                target_fps=self.args.target_fps
            )
            print(f"Spatial scale changed to {new_scale}")
        
        elif key == ord('r'):
            # Переключення ROI detection
            new_roi = not self.args.use_roi
            self.args.use_roi = new_roi
            self.detector.set_downsampling_params(
                spatial_scale=self.args.spatial_scale,
                temporal_skip=self.args.temporal_skip,
                use_roi=new_roi,
                adaptive_skipping=self.args.adaptive_fps,
                target_fps=self.args.target_fps
            )
            print(f"ROI detection {'enabled' if new_roi else 'disabled'}")
        
        elif key == ord('t'):
            # Переключення temporal skip
            current_skip = self.args.temporal_skip
            new_skip = current_skip + 1 if current_skip < 4 else 1
            self.args.temporal_skip = new_skip
            self.detector.set_downsampling_params(
                spatial_scale=self.args.spatial_scale,
                temporal_skip=new_skip,
                use_roi=self.args.use_roi,
                adaptive_skipping=False,  # Вимикаємо adaptive при ручному налаштуванні
                target_fps=self.args.target_fps
            )
            print(f"Temporal skip changed to {new_skip}")
        
        elif key == ord('a'):
            # Переключення adaptive frame skipping
            new_adaptive = not self.args.adaptive_fps
            self.args.adaptive_fps = new_adaptive
            self.detector.set_downsampling_params(
                spatial_scale=self.args.spatial_scale,
                temporal_skip=self.args.temporal_skip,
                use_roi=self.args.use_roi,
                adaptive_skipping=new_adaptive,
                target_fps=self.args.target_fps
            )
            print(f"Adaptive frame skipping {'enabled' if new_adaptive else 'disabled'}")
    
    def _control_fps(self, start_time, target_frame_time):
        """
        Контроль FPS для відповідності швидкості відео
        """
        elapsed_time = time.time() - start_time
        sleep_time = max(target_frame_time - elapsed_time, 0)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    def _cleanup(self):
        """
        Очищення ресурсів та виведення фінальних метрик
        """
        print("\nCleaning up...")
        
        # Отримання фінальних метрик
        final_metrics = self.metrics_collector.get_final_metrics(self.args.tresh)
        
        # Виведення статистики детектора
        if hasattr(self.detector, 'get_performance_stats'):
            detector_stats = self.detector.get_performance_stats()
            print(f"Detector Performance Statistics:")
            print(f"  - Current Detection FPS: {detector_stats.get('processing_fps', 0):.2f}")
            if 'spatial_scale' in detector_stats:
                print(f"  - Spatial Scale: {detector_stats['spatial_scale']:.2f}")
                print(f"  - Temporal Skip: {detector_stats['temporal_skip']}")
                print(f"  - ROI Detection: {detector_stats.get('using_roi', False)}")
                print(f"  - Adaptive Skipping: {detector_stats.get('adaptive_skipping', False)}")
        
        # --- Виведення фінальних результатів у стандартному форматі ---
        print("\n" + "="*50)
        print("ФІНАЛЬНІ РЕЗУЛЬТАТИ")
        print("="*50)
        print(f"Загальний показник Detection FPS: {final_metrics['detection_fps']:.2f}")
        print(f"Загальний показник mAP: {final_metrics['map_score']:.4f}")
        print(f"Загальний час обробки: {final_metrics['total_time']:.2f} секунд")
        print(f"Загальна кількість оброблених кадрів: {final_metrics['total_frames']}")
        print(f"Загальна кількість детекцій: {final_metrics['total_detections']}")
        print(f"Середня впевненість детекцій: {final_metrics['avg_confidence']:.4f}")
        print(f"Backend: {self.detector_type}")
        print("="*50)
        print("Програма завершена.")
        
        # Зупинка детектора
        if hasattr(self.detector, 'stop'):
            self.detector.stop()
        
        # Звільнення ресурсів
        self.video_processor.release()
        self.display_manager.close()
    
    def run(self):
        """
        Запуск додатку
        """
        try:
            self.setup_detector()
            self.setup_video_processor()
            
            if self.video_processor.is_image:
                self.process_static_image()
            else:
                self.process_video()
                
        except Exception as e:
            print(f"Error running application: {e}")
            sys.exit(1)


def run_benchmark(args):
    """
    Запуск бенчмаркінгу різних детекторів з метриками
    """
    print("Starting benchmark...")
    
    # Завантаження тестових кадрів
    video_processor = VideoProcessor(args.source)
    test_frames = []
    
    # Читання перших 100 кадрів для тестування
    for _ in range(100):
        ret, frame = video_processor.read_frame()
        if not ret:
            break
        test_frames.append(frame)
    
    video_processor.release()
    
    if not test_frames:
        print("No frames to benchmark")
        return
    
    print(f"Loaded {len(test_frames)} frames for benchmarking")
    
    # Ініціалізація бенчмарка
    benchmark = DetectorBenchmark()
    
    # Тестування простого детектора з метриками
    print("Benchmarking Simple Detector...")
    simple_metrics = MetricsCollector()
    simple_detector = SimpleYOLODetector(args.model, args.names)
    simple_detector.update_settings(
        tresh=args.tresh,
        thickness=args.thickness,
        sharpness=args.sharpness,
        preprocessing=args.preprocessing
    )
    
    # Прогін кадрів через простий детектор
    for frame in test_frames:
        overlay, _, _ = simple_detector.detect(frame)
        # Псевдо-детекції для простого детектора
        if overlay is not None:
            gray_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray_overlay, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detections = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    detections.append({
                        'box': [x, y, x + w, y + h],
                        'confidence': args.tresh + 0.1,
                        'class_id': 0
                    })
            simple_metrics.add_detections(detections)
        simple_metrics.add_frame()
    
    simple_final_metrics = simple_metrics.get_final_metrics(args.tresh)
    
    # Тестування оптимізованого детектора з метриками
    print("Benchmarking Optimized Detector...")
    optimized_metrics = MetricsCollector()
    optimized_detector = OptimizedYOLODetectorWithDownsampling(args.model, args.names)
    optimized_detector.set_downsampling_params(
        spatial_scale=args.spatial_scale,
        temporal_skip=args.temporal_skip,
        use_roi=args.use_roi,
        adaptive_skipping=args.adaptive_fps,
        target_fps=args.target_fps
    )
    optimized_detector.update_settings(
        tresh=args.tresh,
        thickness=args.thickness,
        sharpness=args.sharpness,
        preprocessing=args.preprocessing
    )
    
    # Прогін кадрів через оптимізований детектор
    for frame in test_frames:
        optimized_detector.process_frame(frame)
        time.sleep(0.01)  # Невелика пауза для обробки
        overlay, _, _ = optimized_detector.get_latest_results()
        
        # Витягування детекцій з оптимізованого детектора
        if hasattr(optimized_detector, 'get_latest_detections'):
            detections = optimized_detector.get_latest_detections()
        else:
            # Псевдо-детекції
            detections = []
            if overlay is not None:
                gray_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
                contours, _ = cv2.findContours(gray_overlay, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 100:
                        x, y, w, h = cv2.boundingRect(contour)
                        detections.append({
                            'box': [x, y, x + w, y + h],
                            'confidence': args.tresh + 0.1,
                            'class_id': 0
                        })
        
        optimized_metrics.add_detections(detections)
        optimized_metrics.add_frame()
    
    optimized_final_metrics = optimized_metrics.get_final_metrics(args.tresh)
    
    # Виведення стандартизованих результатів бенчмарку
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТИ БЕНЧМАРКУ")
    print("="*60)
    
    print("\nSIMPLE DETECTOR:")
    print("-" * 30)
    print(f"Загальний показник Detection FPS: {simple_final_metrics['detection_fps']:.2f}")
    print(f"Загальний показник mAP: {simple_final_metrics['map_score']:.4f}")
    print(f"Загальний час обробки: {simple_final_metrics['total_time']:.2f} секунд")
    print(f"Загальна кількість оброблених кадрів: {simple_final_metrics['total_frames']}")
    print(f"Загальна кількість детекцій: {simple_final_metrics['total_detections']}")
    print(f"Середня впевненість детекцій: {simple_final_metrics['avg_confidence']:.4f}")
    print(f"Backend: Simple Detector")
    
    print("\nOPTIMIZED DETECTOR:")
    print("-" * 30)
    print(f"Загальний показник Detection FPS: {optimized_final_metrics['detection_fps']:.2f}")
    print(f"Загальний показник mAP: {optimized_final_metrics['map_score']:.4f}")
    print(f"Загальний час обробки: {optimized_final_metrics['total_time']:.2f} секунд")
    print(f"Загальна кількість оброблених кадрів: {optimized_final_metrics['total_frames']}")
    print(f"Загальна кількість детекцій: {optimized_final_metrics['total_detections']}")
    print(f"Середня впевненість детекцій: {optimized_final_metrics['avg_confidence']:.4f}")
    print(f"Backend: Optimized Detector")
    
    print("\nПОРІВНЯННЯ:")
    print("-" * 30)
    fps_improvement = ((optimized_final_metrics['detection_fps'] - simple_final_metrics['detection_fps']) / simple_final_metrics['detection_fps']) * 100 if simple_final_metrics['detection_fps'] > 0 else 0
    map_difference = optimized_final_metrics['map_score'] - simple_final_metrics['map_score']
    
    print(f"Покращення FPS: {fps_improvement:+.1f}%")
    print(f"Різниця mAP: {map_difference:+.4f}")
    print(f"Кращий за FPS: {'Optimized' if optimized_final_metrics['detection_fps'] > simple_final_metrics['detection_fps'] else 'Simple'}")
    print(f"Кращий за mAP: {'Optimized' if optimized_final_metrics['map_score'] > simple_final_metrics['map_score'] else 'Simple'}")
    
    print("="*60)
    print("Бенчмарк завершено.")
    
    # Зупинка детекторів
    optimized_detector.stop()


def parse_arguments():
    """
    Парсинг аргументів командного рядка
    """
    parser = argparse.ArgumentParser(description="YOLO Object Detection with Downsampling and Metrics")
    
    # Основні параметри
    parser.add_argument("--source", type=str, default="data/videos/IsomCar.mp4", help="Video source")
    parser.add_argument("--names", type=str, default="data/class.names", help="Object names file")
    parser.add_argument("--model", type=str, default="./models/yolo11n-old.onnx", help="YOLO model path")
    parser.add_argument("--tresh", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--thickness", type=int, default=2, help="Bounding box thickness")
    parser.add_argument("--show_grayscale", action="store_true", default=True, help="Show grayscale input")
    parser.add_argument("--show_model_view", action="store_true", help="Show what the model sees")
    parser.add_argument("--show_color", action="store_true", help="Show color mode")
    parser.add_argument("--use_color_processing", action="store_true", help="Use color processing instead of grayscale")
    parser.add_argument("--sharpness", type=int, default=1, choices=[1, 2, 3], 
                        help="Sharpness level (1=Standard, 2=Strong, 3=Extreme)")
    parser.add_argument("--preprocessing", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6],
                        help="Preprocessing method")
    
    # Параметри downsampling
    parser.add_argument("--spatial_scale", type=float, default=0.3, 
                        help="Spatial downsampling scale (0.1-1.0)")
    parser.add_argument("--temporal_skip", type=int, default=1, 
                        help="Process every N-th frame")
    parser.add_argument("--use_roi", action="store_true", default=False,
                        help="Use motion-based ROI detection")
    parser.add_argument("--adaptive_fps", action="store_true", default=False,
                        help="Use adaptive frame skipping")
    parser.add_argument("--target_fps", type=int, default=7,
                        help="Target detection FPS for adaptive mode")
    
    # Додаткові опції
    parser.add_argument("--use_simple_detector", action="store_true", 
                        help="Use simple detector without downsampling", default=False)
    parser.add_argument("--use_threaded_pipeline", action="store_true",
                        help="Use multi-threaded processing pipeline", default=False)
    parser.add_argument("--benchmark", action="store_true", 
                        help="Run benchmark comparison with metrics")
    
    return parser.parse_args()


def main():
    """
    Головна функція
    """
    args = parse_arguments()
    
    # Перевірка наявності файлів
    required_files = [args.model, args.names]
    if args.source != "0":  # Якщо не веб-камера
        required_files.append(args.source)
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            sys.exit(1)
    
    # Запуск бенчмарка або головного додатку
    if args.benchmark:
        run_benchmark(args)
    else:
        app = YOLOApplication(args)
        app.run()


if __name__ == '__main__':
    main()