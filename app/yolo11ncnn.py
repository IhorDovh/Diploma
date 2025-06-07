import cv2
import time
import random
import argparse
import numpy as np
import threading
import os
import ncnn # Імпортуємо ncnn тут
from ultralytics import YOLO

# --- Функція завантаження джерела (без змін) ---
def load_source(source_file):
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    cap = None
    frame = None
    if source_file.isdigit():
        source_file = int(source_file)
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

# --- Детекція через Ultralytics (без змін) ---
def yolo_detection_ultralytics(frame, model, NAMES, COLORS, args):
    """YOLO детекція з Ultralytics API (для .pt або NCNN через YOLO)"""
    overlay = np.zeros_like(frame, dtype=np.uint8)
    results = model.predict(frame, conf=args.tresh, verbose=False)
    current_frame_confidences = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(boxes)):
                if confidences[i] > args.tresh:
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    if 0 <= class_id < len(NAMES):
                        color = COLORS[class_id]
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, args.thickness)
                        text = f'{NAMES[class_id]} {confidence:.2f}'
                        cv2.putText(overlay, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        current_frame_confidences.append(confidence)
    avg_confidence_this_frame = np.mean(current_frame_confidences) if current_frame_confidences else 0.0
    return overlay, avg_confidence_this_frame

# --- ОПТИМІЗОВАНА функція детекції для NCNN ---
def yolo_detection_ncnn_native(frame, net, NAMES, COLORS, args):
    """
    Оптимізована YOLO детекція з NCNN native API.
    Приймає вже завантажений об'єкт 'net', а не шлях до моделі.
    """
    overlay = np.zeros_like(frame, dtype=np.uint8)
    
    # Preprocessing
    height, width = frame.shape[:2]
    resized = cv2.resize(frame, (640, 640))
    mat_in = ncnn.Mat.from_pixels(resized, ncnn.Mat.PixelType.PIXEL_BGR, 640, 640)
    mean_vals = [0, 0, 0]
    norm_vals = [1/255.0, 1/255.0, 1/255.0]
    mat_in.substract_mean_normalize(mean_vals, norm_vals)
    
    # Інференс
    ex = net.create_extractor()
    ex.input("in0", mat_in)
    mat_out = ncnn.Mat()
    ex.extract("out0", mat_out)
    
    # Постобробка
    outputs = np.array(mat_out).reshape(-1, 84) # 8400x84
    current_frame_confidences = []
    boxes, confidences, class_ids = [], [], []
    x_factor, y_factor = width / 640.0, height / 640.0
    
    for row in outputs:
        cx, cy, w, h = row[:4]
        class_scores = row[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        if confidence > args.tresh:
            x1 = int((cx - w/2) * x_factor)
            y1 = int((cy - h/2) * y_factor)
            x2 = int((cx + w/2) * x_factor)
            y2 = int((cy + h/2) * y_factor)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            
    if boxes:
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, args.tresh, 0.4)
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                class_id, confidence = class_ids[i], confidences[i]
                if 0 <= class_id < len(NAMES):
                    color = COLORS[class_id]
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, args.thickness)
                    text = f'{NAMES[class_id]} {confidence:.2f}'
                    cv2.putText(overlay, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    current_frame_confidences.append(confidence)
                    
    avg_confidence_this_frame = np.mean(current_frame_confidences) if current_frame_confidences else 0.0
    return overlay, avg_confidence_this_frame

# --- ОПТИМІЗОВАНА функція асинхронної обробки ---
def async_yolo_processing(model_ref, NAMES_ref, COLORS_ref, args_ref, use_native_ncnn_flag):
    global latest_frame, processed_overlay, processing_fps, total_detected_confidences, detected_frames_count
    frame_count = 0
    start_processing_time = time.time()
    
    while True:
        if latest_frame is not None:
            current_frame_to_process = latest_frame.copy()
            if current_frame_to_process is not None:
                # Вибір функції детекції на основі прапора
                if use_native_ncnn_flag:
                    overlay, avg_confidence_this_frame = yolo_detection_ncnn_native(
                        current_frame_to_process, model_ref, NAMES_ref, COLORS_ref, args_ref
                    )
                else:
                    overlay, avg_confidence_this_frame = yolo_detection_ultralytics(
                        current_frame_to_process, model_ref, NAMES_ref, COLORS_ref, args_ref
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
            # Зменшуємо навантаження на CPU, коли немає кадрів
            time.sleep(0.01)

# --- Експорт моделі (без змін) ---
def export_model_to_ncnn(pt_model_path):
    print(f"Експорт моделі {pt_model_path} в NCNN формат...")
    model = YOLO(pt_model_path)
    model.export(format="ncnn", device="cpu")
    base_name = os.path.splitext(os.path.basename(pt_model_path))[0]
    return f"./{base_name}_ncnn_model"

if __name__ == '__main__':
    overall_start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/IsomCar.mp4", help="Відео або зображення. '0' для веб-камери.")
    parser.add_argument("--names", type=str, default="data/class.names", help="Шлях до файлу імен класів.")
    parser.add_argument("--model", type=str, default="yolo11n_ncnn_model", help="Шлях до моделі (папка NCNN або .pt файл).")
    parser.add_argument("--tresh", type=float, default=0.25, help="Поріг впевненості для детекції.")
    parser.add_argument("--thickness", type=int, default=2, help="Товщина рамки.")
    parser.add_argument("--export", action="store_true", help="Експортувати PT модель в NCNN формат перед запуском.")
    # --- НОВИЙ АРГУМЕНТ для вибору нативного режиму ---
    parser.add_argument("--native", action="store_true", help="Використовувати нативний pyncnn для максимальної продуктивності.")
    args = parser.parse_args()

    model_to_process = None
    use_native_ncnn = args.native
    backend_name = ""

    try:
        if use_native_ncnn:
            print("Ініціалізація нативної NCNN моделі...")
            model_path = args.model
            if not os.path.isdir(model_path):
                raise FileNotFoundError(f"Директорія нативної NCNN моделі не знайдена: {model_path}")
            
            param_path = os.path.join(model_path, "model.ncnn.param")
            bin_path = os.path.join(model_path, "model.ncnn.bin")
            
            if not os.path.exists(param_path) or not os.path.exists(bin_path):
                 # Спроба знайти файли з назвою моделі (стандарт експорту Ultralytics)
                 model_name = os.path.basename(model_path).replace("_ncnn_model", "")
                 param_path = os.path.join(model_path, f"{model_name}.ncnn.param")
                 bin_path = os.path.join(model_path, f"{model_name}.ncnn.bin")
                 if not os.path.exists(param_path) or not os.path.exists(bin_path):
                     raise FileNotFoundError(f"Файли .param та .bin не знайдені в {model_path}")

            # --- ОСНОВНА ЗМІНА: Завантажуємо модель тут, один раз ---
            ncnn_net = ncnn.Net()
            ncnn_net.load_param(param_path)
            ncnn_net.load_model(bin_path)
            model_to_process = ncnn_net
            backend_name = "NCNN (Native)"
            print("Нативна NCNN модель успішно ініціалізована!")
        else:
            print("Ініціалізація моделі через Ultralytics...")
            model_path = args.model
            if args.export and model_path.endswith('.pt'):
                model_path = export_model_to_ncnn(model_path)
                backend_name = "NCNN (Ultralytics Exported)"
            
            model_to_process = YOLO(model_path)
            
            # Тестовий запуск для ініціалізації
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = model_to_process.predict(test_img, verbose=False)
            
            if "ncnn" in str(model_path):
                backend_name = "NCNN (Ultralytics)"
            else:
                backend_name = "PyTorch"
            print(f"Модель ({backend_name}) успішно ініціалізована!")

    except Exception as e:
        print(f"Помилка ініціалізації моделі: {e}")
        exit()

    # Завантаження імен класів
    try:
        with open(args.names, "r", encoding='utf-8') as f:
            NAMES = [cname.strip() for cname in f.readlines()]
    except FileNotFoundError:
        print(f"Файл імен класів '{args.names}' не знайдено. Спроба отримати імена з моделі.")
        if use_native_ncnn:
            print("Неможливо отримати імена класів з нативної моделі. Вкажіть правильний --names файл.")
            exit()
        NAMES = model_to_process.names
        if isinstance(NAMES, dict):
            NAMES = [NAMES[i] for i in range(len(NAMES))]
    
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in NAMES]

    # Завантаження джерела
    image_type, initial_frame, cap = load_source(args.source)
    if initial_frame is None and cap is None:
        exit()

    # Глобальні змінні
    latest_frame, processed_overlay = None, None
    processing_fps = 0.0
    total_detected_confidences = 0.0
    detected_frames_count = 0

    # Запуск потоку обробки
    processing_thread = threading.Thread(
        target=async_yolo_processing,
        args=(model_to_process, NAMES, COLORS, args, use_native_ncnn), # Передаємо модель і прапор
        daemon=True
    )
    processing_thread.start()

    # Ініціалізація бенчмарку
    benchmark_results = {
        "source_file": args.source, "model_path": args.model, "backend": backend_name,
        "confidence_threshold": args.tresh, "input_image_size": "640x640",
        "total_runtime_seconds": 0.0, "average_detection_fps": 0.0,
        "average_display_fps": 0.0, "source_fps": 0.0, "overall_average_confidence": 0.0
    }

    # --- Головний цикл (майже без змін) ---
    total_frames_read = 0
    display_start_time = time.time()
    total_display_frames = 0

    if image_type:
        latest_frame = initial_frame.copy()
        print("Обробка зображення...")
        time.sleep(1) # Даємо час потоку на обробку
        if processed_overlay is not None:
            result = cv2.addWeighted(initial_frame, 1.0, processed_overlay, 1.0, 0)
            output_filename = "output_ncnn_" + os.path.basename(args.source)
            cv2.imwrite(output_filename, result)
            print(f"Оброблено та збережено зображення як {output_filename}")
        else:
            print("Обробка зображення не вдалася.")
    else:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        benchmark_results["source_fps"] = video_fps
        print(f"Обробка відео (Source FPS: {video_fps:.2f})... Натисніть 'q' для виходу.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Кінець відео потоку.")
                break
            
            total_frames_read += 1
            latest_frame = frame.copy()
            
            overlay = processed_overlay if processed_overlay is not None else np.zeros_like(frame, dtype=np.uint8)
            result = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)
            total_display_frames += 1
            
            fps_text = f"Detection FPS: {processing_fps:.2f} | Backend: {backend_name}"
            cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("NCNN YOLO Detection", result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Завершення та підрахунок результатів
    latest_frame = None # Сигнал для завершення потоку
    time.sleep(0.5)
    if cap: cap.release()
    cv2.destroyAllWindows()
    
    overall_end_time = time.time()
    total_runtime = overall_end_time - overall_start_time
    benchmark_results["total_runtime_seconds"] = total_runtime
    benchmark_results["average_detection_fps"] = processing_fps
    if total_display_frames > 0:
        benchmark_results["average_display_fps"] = total_display_frames / total_runtime
    if detected_frames_count > 0:
        benchmark_results["overall_average_confidence"] = total_detected_confidences / detected_frames_count

    print("\n" + "="*40)
    print("Результати бенчмарку NCNN:")
    print("="*40)
    for key, value in benchmark_results.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")
    print("="*40)