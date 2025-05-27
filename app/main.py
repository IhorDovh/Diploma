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
            return "Default Mode""""
Main Application Module
Головний файл для запуску YOLO детектора з модульною архітектурою
"""

import argparse
import time
import cv2
import sys
import os

# Імпорт модулів
from optimized_detector import OptimizedYOLODetectorWithDownsampling, SimpleYOLODetector, DetectorBenchmark, ThreadedYOLODetector
from video_utils import VideoProcessor, DisplayManager, validate_source
from image_processing import get_preprocessing_name


class YOLOApplication:
    """
    Головний клас додатку YOLO детектора
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
        elif self.args.use_threaded_pipeline:
            # Багатопоточний пайплайн
            self.detector = ThreadedYOLODetector(self.args.model, self.args.names, 640)
            
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
        elif settings.get("show_grayscale", True):
            return "Grayscale Sharpened Mode"
        else:
            return "Color Mode"
    
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
        Очищення ресурсів
        """
        print("\nCleaning up...")
        
        # Виведення фінальної статистики
        if hasattr(self.detector, 'get_performance_stats'):
            final_stats = self.detector.get_performance_stats()
            print(f"Final Performance Statistics:")
            print(f"  - Detection FPS: {final_stats.get('processing_fps', 0):.2f}")
            if 'spatial_scale' in final_stats:
                print(f"  - Spatial Scale: {final_stats['spatial_scale']:.2f}")
                print(f"  - Temporal Skip: {final_stats['temporal_skip']}")
                print(f"  - ROI Detection: {final_stats.get('using_roi', False)}")
                print(f"  - Adaptive Skipping: {final_stats.get('adaptive_skipping', False)}")
        
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
    Запуск бенчмаркінгу різних детекторів
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
    
    # Тестування простого детектора
    print("Benchmarking Simple Detector...")
    simple_detector = SimpleYOLODetector(args.model, args.names)
    simple_detector.update_settings(
        tresh=args.tresh,
        thickness=args.thickness,
        sharpness=args.sharpness,
        preprocessing=args.preprocessing
    )
    benchmark.benchmark_detector(simple_detector, test_frames, "Simple Detector")
    
    # Тестування оптимізованого детектора
    print("Benchmarking Optimized Detector...")
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
    benchmark.benchmark_detector(optimized_detector, test_frames, "Optimized Detector")
    
    # Виведення результатів
    print(benchmark.compare_results())
    
    # Зупинка детекторів
    optimized_detector.stop()


def parse_arguments():
    """
    Парсинг аргументів командного рядка
    """
    parser = argparse.ArgumentParser(description="YOLO Object Detection with Downsampling")
    
    # Основні параметри
    parser.add_argument("--source", type=str, default="data/videos/idiots3.mp4", help="Video source")
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
                        help="Run benchmark comparison")
    
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