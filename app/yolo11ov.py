import cv2
import time
import random
import argparse
import numpy as np
import threading
import os
from openvino.runtime import Core

def load_source(source_file):
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    cap = None
    frame = None

    if source_file == "0" or source_file == 0:
        source_file = 0  # Веб-камера
        image_type = False
    else:
        image_type = source_file.split('.')[-1].lower() in img_formats

    if image_type:
        frame = cv2.imread(source_file)
        if frame is None:
            print(f"Помилка: Не вдалося прочитати зображення {source_file}")
            return True, None, None
    else:
        cap = cv2.VideoCapture(source_file)
        if not cap.isOpened():
            print(f"Помилка: Не вдалося відкрити джерело відео {source_file}")
            return False, None, None

    return image_type, frame if image_type else None, cap

def preprocess_frame(frame, input_shape):
    """Preprocessing для OpenVINO моделі"""
    height, width = input_shape[2], input_shape[3]
    
    # Зміна розміру зображення
    resized = cv2.resize(frame, (width, height))
    
    # Перетворення до формату (1, 3, H, W) та нормалізація
    input_tensor = resized.transpose((2, 0, 1))  # HWC -> CHW
    input_tensor = input_tensor.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Додати batch dimension
    
    return input_tensor

def postprocess_outputs(outputs, frame_shape, input_shape, conf_threshold=0.25):
    """Постобробка результатів OpenVINO"""
    # Отримати результати (припускаємо формат YOLO)
    predictions = outputs[0]  # Перший вихід
    
    # Якщо форма [1, 84, 8400] - транспонувати до [1, 8400, 84]
    if len(predictions.shape) == 3 and predictions.shape[1] < predictions.shape[2]:
        predictions = predictions.transpose((0, 2, 1))
    
    frame_height, frame_width = frame_shape[:2]
    input_height, input_width = input_shape[2], input_shape[3]
    
    # Коефіцієнти масштабування
    x_factor = frame_width / input_width
    y_factor = frame_height / input_height
    
    boxes = []
    confidences = []
    class_ids = []
    
    for detection in predictions[0]:
        # Отримати координати центру та розміри
        x, y, w, h = detection[:4]
        
        # Отримати впевненості класів (починаючи з 4-го елемента)
        class_scores = detection[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        if confidence > conf_threshold:
            # Перетворити центральні координати в координати кутів
            left = int((x - w/2) * x_factor)
            top = int((y - h/2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            boxes.append([left, top, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)
    
    return boxes, confidences, class_ids

def yolo_detection_openvino(frame, compiled_model, input_layer, output_layer, NAMES, COLORS, args):
    """YOLO детекція з OpenVINO"""
    overlay = np.zeros_like(frame, dtype=np.uint8)
    
    # Отримати форму входу
    input_shape = input_layer.shape
    
    # Preprocessing
    input_tensor = preprocess_frame(frame, input_shape)
    
    # Інференс
    results = compiled_model([input_tensor])
    
    # Постобробка
    boxes, confidences, class_ids = postprocess_outputs(
        results, frame.shape, input_shape, args.tresh
    )
    
    # Non-Maximum Suppression
    if boxes:
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, args.tresh, 0.4)
    else:
        indexes = []
    
    current_frame_confidences = []
    
    if len(indexes) > 0:
        for i in indexes.flatten():
            left, top, width, height = boxes[i]
            class_id = class_ids[i]
            confidence = confidences[i]
            
            # Перевірка валідності class_id
            if 0 <= class_id < len(NAMES):
                color = COLORS[class_id]
                cv2.rectangle(overlay, (left, top), (left + width, top + height), color, args.thickness)
                text = f'{NAMES[class_id]} {confidence:.2f}'
                cv2.putText(overlay, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                current_frame_confidences.append(confidence)
    
    avg_confidence_this_frame = np.mean(current_frame_confidences) if current_frame_confidences else 0.0
    return overlay, avg_confidence_this_frame

def benchmark_openvino_config(model_path, device="CPU", num_iterations=10):
    """Бенчмарк тест різних конфігурацій OpenVINO"""
    print("\n" + "="*60)
    print("БЕНЧМАРК ТЕСТ OPENVINO КОНФІГУРАЦІЙ")
    print("="*60)
    
    # Тестовий вхід
    test_input = np.random.random((1, 3, 640, 640)).astype(np.float32)
    
    # Тест 1: Без оптимізацій
    print("🔄 Тест 1: Базова конфігурація...")
    core1 = Core()
    model1 = core1.read_model(os.path.join(model_path, "yolo11n.xml"))
    compiled_model1 = core1.compile_model(model1, device)
    
    start_time = time.time()
    for _ in range(num_iterations):
        results = compiled_model1([test_input])
    base_time = time.time() - start_time
    print(f"⏱️ Базовий час: {base_time:.3f} сек ({base_time/num_iterations*1000:.1f} мс/інференс)")
    
    # Тест 2: З оптимізаціями
    print("🔄 Тест 2: Оптимізована конфігурація...")
    core2 = Core()
    model2 = core2.read_model(os.path.join(model_path, "yolo11n.xml"))
    
    # Оптимальна конфігурація для ARM64
    optimized_config = {
        "CPU_THREADS_NUM": "4",
        "CPU_BIND_THREAD": "YES", 
        "CPU_THROUGHPUT_STREAMS": "4",
        "INFERENCE_NUM_THREADS": "4",
        "PERFORMANCE_HINT": "LATENCY"
    }
    
    # Застосування конфігурації
    for key, value in optimized_config.items():
        try:
            core2.set_property("CPU", {key: value})
        except:
            pass
    
    compiled_model2 = core2.compile_model(model2, device, optimized_config)
    
    start_time = time.time()
    for _ in range(num_iterations):
        results = compiled_model2([test_input])
    optimized_time = time.time() - start_time
    print(f"⏱️ Оптимізований час: {optimized_time:.3f} сек ({optimized_time/num_iterations*1000:.1f} мс/інференс)")
    
    # Результати
    improvement = ((base_time - optimized_time) / base_time) * 100
    print(f"📈 Покращення продуктивності: {improvement:.1f}%")
    if improvement > 0:
        print("✅ Оптимізації працюють!")
    else:
        print("⚠️ Оптимізації не дали покращення")
    print("="*60)
    
    return improvement


    global latest_frame, processed_overlay, processing_fps, total_detected_confidences, detected_frames_count
    frame_count = 0
    start_processing_time = time.time()

    while True:
        if latest_frame is not None:
            current_frame_to_process = latest_frame.copy()
            if current_frame_to_process is not None:
                overlay, avg_confidence_this_frame = yolo_detection_openvino(
                    current_frame_to_process, compiled_model_ref, input_layer_ref, 
                    output_layer_ref, NAMES_ref, COLORS_ref, args_ref
                )
                processed_overlay = overlay
                
                if avg_confidence_this_frame > 0:
                    total_detected_confidences += avg_confidence_this_frame
                    detected_frames_count += 1

                frame_count += 1

                elapsed_processing_time = time.time() - start_processing_time
                if elapsed_processing_time >= 1.0:
                    processing_fps = frame_count / elapsed_processing_time
                    frame_count = 0
                    start_processing_time = time.time()
        else:
            time.sleep(0.01)

if __name__ == '__main__':
    overall_start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/idiots3.mp4", help="Відео або зображення. '0' для веб-камери.")
    parser.add_argument("--names", type=str, default="data/class.names", help="Шлях до файлу імен класів.")
    parser.add_argument("--model", type=str, default="yolo11n_openvino_model", help="Шлях до OpenVINO моделі (папка).")
    parser.add_argument("--tresh", type=float, default=0.25, help="Поріг впевненості для детекції.")
    parser.add_argument("--thickness", type=int, default=2, help="Товщина рамки.")
    parser.add_argument("--device", type=str, default="AUTO", help="Пристрій OpenVINO (AUTO, CPU, GPU).")
    parser.add_argument("--benchmark", action="store_true", help="Запустити бенчмарк тест оптимізацій.")
    args = parser.parse_args()

    # Якщо запрошено бенчмарк
    if args.benchmark:
        benchmark_openvino_config(args.model, args.device)
        exit()

    # Ініціалізація OpenVINO
    try:
        print("Ініціалізація OpenVINO...")
        core = Core()
        
        # Перевірка доступних пристроїв
        available_devices = core.available_devices
        print(f"Доступні пристрої: {available_devices}")
        
        # Завантаження моделі
        model_xml = os.path.join(args.model, "yolo11n.xml")
        if not os.path.exists(model_xml):
            print(f"Помилка: Файл моделі не знайдено: {model_xml}")
            print("Переконайтеся, що модель експортована в формат OpenVINO")
            print("Очікуваний файл: yolo11n.xml")
            exit()
            
        model = core.read_model(model_xml)
        
        # Оптимізація конфігурації для ARM64 (Radxa Zero 3E)
        config = {}
        if args.device == "CPU" or args.device == "AUTO":
            print("Застосування оптимізацій для ARM64 CPU...")
            config = {
                # Використовувати всі 4 ядра RK3566
                "CPU_THREADS_NUM": "4",
                "CPU_BIND_THREAD": "YES",
                "CPU_THROUGHPUT_STREAMS": "4",
                # Оптимізація для ARM Cortex-A55
                "INFERENCE_NUM_THREADS": "4",
                "INFERENCE_PRECISION_HINT": "f32",
                # Додаткові оптимізації
                "PERFORMANCE_HINT": "LATENCY",  # або "THROUGHPUT"
                "EXECUTION_MODE_HINT": "PERFORMANCE"
            }
            
            # Встановити властивості для CPU
            for key, value in config.items():
                try:
                    core.set_property("CPU", {key: value})
                    print(f"✅ {key}: {value}")
                except Exception as e:
                    print(f"⚠️ Не вдалося встановити {key}: {e}")
        
        compiled_model = core.compile_model(model, args.device, config)
        
        # Отримати інформацію про входи та виходи
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        
        print(f"Модель завантажена на пристрій: {args.device}")
        print(f"Форма входу: {input_layer.shape}")
        print(f"Форма виходу: {output_layer.shape}")
        
        # Виведення інформації про застосовані оптимізації
        if args.device == "CPU" or args.device == "AUTO":
            try:
                actual_threads = core.get_property("CPU", "CPU_THREADS_NUM")
                print(f"✅ Фактична кількість потоків CPU: {actual_threads}")
            except:
                print("⚠️ Не вдалося отримати інформацію про потоки CPU")
                
            try:
                streams = core.get_property("CPU", "CPU_THROUGHPUT_STREAMS")
                print(f"✅ Кількість потоків throughput: {streams}")
            except:
                print("⚠️ Не вдалося отримати інформацію про throughput streams")
        
        print("="*50)
        
    except Exception as e:
        print(f"Помилка ініціалізації OpenVINO: {e}")
        exit()

    # Завантаження імен класів
    try:
        with open(args.names, "r", encoding='utf-8') as f:
            NAMES = [cname.strip() for cname in f.readlines()]
    except FileNotFoundError:
        print(f"Помилка: Файл імен класів не знайдено за шляхом {args.names}")
        print("Використання стандартних імен класів COCO як резервний варіант.")
        NAMES = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in NAMES]

    # Завантаження джерела
    image_type, initial_frame, cap = load_source(args.source)

    if initial_frame is None and cap is None:
        print("Не вдалося завантажити джерело. Вихід.")
        exit()

    # Глобальні змінні
    latest_frame, processed_overlay = None, None
    processing_fps = 0.0
    total_detected_confidences = 0.0
    detected_frames_count = 0

    # Запуск потоку обробки
    processing_thread = threading.Thread(
        target=async_yolo_processing,
        args=(compiled_model, input_layer, output_layer, NAMES, COLORS, args),
        daemon=True
    )
    processing_thread.start()

    # Ініціалізація бенчмарку
    benchmark_results = {
        "source_file": args.source,
        "model_path": args.model,
        "device": args.device,
        "confidence_threshold": args.tresh,
        "input_image_size": f"{input_layer.shape[2]}x{input_layer.shape[3]}",
        "total_runtime_seconds": 0.0,
        "average_detection_fps": 0.0,
        "average_display_fps": 0.0,
        "source_fps": 0.0,
        "overall_average_confidence": 0.0,
        "available_devices": str(available_devices)
    }

    total_frames_read = 0
    display_start_time = time.time()
    total_display_frames = 0

    # Обробка зображення або відео
    if image_type:
        if initial_frame is not None:
            latest_frame = initial_frame.copy()
            print("Обробка зображення...")
            while processed_overlay is None and processing_thread.is_alive():
                time.sleep(0.01)

            if processed_overlay is not None:
                result = cv2.addWeighted(initial_frame, 1.0, processed_overlay, 1.0, 0)
                cv2.imshow("OpenVINO YOLO Detection", result)
                output_filename = "output_openvino_" + os.path.basename(args.source)
                try:
                    cv2.imwrite(output_filename, result)
                    print(f"Оброблено та збережено зображення як {output_filename}")
                except Exception as e:
                    print(f"Не вдалося зберегти зображення: {e}")
                cv2.waitKey(0)
            else:
                print("Обробка зображення не вдалася.")
            
            total_frames_read = 1
            total_display_frames = 1
            if detected_frames_count > 0:
                 benchmark_results["overall_average_confidence"] = total_detected_confidences / detected_frames_count
        else:
            print("Завантажене зображення було None, неможливо обробити.")
    else:
        if cap is None or not cap.isOpened():
            print("Захоплення відео не відкрито. Вихід.")
            exit()

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        benchmark_results["source_fps"] = video_fps
        frame_time = 1 / video_fps if video_fps > 0 else 0

        print(f"Обробка відео (FPS: {video_fps:.2f})...")
        print("Натисніть 'q' для виходу")

        while True:
            start_frame_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Кінець відео потоку або неможливо прочитати кадр.")
                break

            total_frames_read += 1
            latest_frame = frame.copy()
            
            overlay = processed_overlay if processed_overlay is not None else np.zeros_like(frame, dtype=np.uint8)
            result = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)
            total_display_frames += 1

            # Відображення FPS та інформації про пристрій
            fps_text = f"Video FPS: {video_fps:.2f} | Detection FPS: {processing_fps:.2f} | Device: {args.device}"
            cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Додаткова інформація
            info_text = f"Frames: {total_frames_read} | Detections: {detected_frames_count}"
            cv2.putText(result, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("OpenVINO YOLO Detection", result)

            elapsed_time = time.time() - start_frame_time
            sleep_time = max(frame_time - elapsed_time, 0)
            time.sleep(sleep_time)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if total_display_frames > 0:
            benchmark_results["average_display_fps"] = total_display_frames / (time.time() - display_start_time)
        if detected_frames_count > 0:
            benchmark_results["overall_average_confidence"] = total_detected_confidences / detected_frames_count

    # Закриття ресурсів
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    latest_frame = None

    # Фінальні результати
    overall_end_time = time.time()
    benchmark_results["total_runtime_seconds"] = overall_end_time - overall_start_time
    benchmark_results["average_detection_fps"] = processing_fps

    # Виведення результатів бенчмарку
    print("\n" + "="*40)
    print("Результати бенчмарку OpenVINO:")
    print("="*40)
    for key, value in benchmark_results.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")
    print("="*40)