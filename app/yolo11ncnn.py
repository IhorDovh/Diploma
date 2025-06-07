import cv2
import time
import random
import argparse
import numpy as np
import threading
import os
from ultralytics import YOLO

def load_source(source_file):
    """Завантажує джерело відео або зображення."""
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    cap = None
    frame = None

    if source_file.isdigit():
        source_file = int(source_file) # Веб-камера
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

def yolo_detection_ultralytics(frame, model, NAMES, COLORS, args):
    """
    YOLO детекція з використанням Ultralytics API.
    Ця функція обробляє як .pt, так і NCNN моделі, завантажені через YOLO().
    """
    overlay = np.zeros_like(frame, dtype=np.uint8)
    
    # Виконати детекцію
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

def async_yolo_processing(model_ref, NAMES_ref, COLORS_ref, args_ref):
    """
    Асинхронний потік для обробки відео.
    Працює в нескінченному циклі, обробляючи найсвіжіший кадр.
    """
    global latest_frame, processed_overlay, processing_fps, total_detected_confidences, detected_frames_count
    frame_count = 0
    start_processing_time = time.time()
    
    while True:
        if latest_frame is not None:
            # Копіюємо кадр, щоб уникнути проблем з одночасним доступом
            current_frame_to_process = latest_frame.copy()
            
            overlay, avg_confidence_this_frame = yolo_detection_ultralytics(
                current_frame_to_process, model_ref, NAMES_ref, COLORS_ref, args_ref
            )
            
            # Оновлюємо глобальну змінну з результатом
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
            # Якщо кадрів немає, потік "спить", щоб не навантажувати CPU
            time.sleep(0.01)

def export_model_to_ncnn(pt_model_path):
    """Експорт моделі з PT в NCNN формат."""
    print(f"Експорт моделі {pt_model_path} в NCNN формат...")
    model = YOLO(pt_model_path)
    # Експортуємо з опцією half=False для кращої сумісності та точності на CPU
    model.export(format="ncnn", half=False, device="cpu")
    base_name = os.path.splitext(os.path.basename(pt_model_path))[0]
    return f"./{base_name}_ncnn_model"

if __name__ == '__main__':
    overall_start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/IsomCar.mp4", help="Відео або зображення. '0' для веб-камери.")
    parser.add_argument("--names", type=str, default="data/coco.names", help="Шлях до файлу імен класів.")
    parser.add_argument("--model", type=str, default="yolo11n_ncnn_model", help="Шлях до моделі (папка NCNN або .pt файл).")
    parser.add_argument("--tresh", type=float, default=0.25, help="Поріг впевненості для детекції.")
    parser.add_argument("--thickness", type=int, default=2, help="Товщина рамки.")
    parser.add_argument("--export", action="store_true", help="Експортувати PT модель в NCNN формат перед запуском.")
    args = parser.parse_args()

    model = None
    backend_name = ""

    # --- Ініціалізація моделі (один раз) ---
    try:
        print("Ініціалізація моделі через Ultralytics...")
        model_path = args.model
        if args.export and model_path.endswith('.pt'):
            model_path = export_model_to_ncnn(model_path)
        
        model = YOLO(model_path)
        
        # Тестовий запуск для "прогріву" та ініціалізації моделі
        print("Прогрів моделі...")
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model.predict(test_img, verbose=False)
        
        if "ncnn" in model_path:
            backend_name = "NCNN (via Ultralytics)"
        else:
            backend_name = "PyTorch"
        print(f"Модель ({backend_name}) успішно ініціалізована!")

    except Exception as e:
        print(f"Помилка ініціалізації моделі: {e}")
        exit()

    # --- Завантаження імен класів ---
    try:
        with open(args.names, "r", encoding='utf-8') as f:
            NAMES = [cname.strip() for cname in f.readlines()]
    except FileNotFoundError:
        print(f"Файл імен класів '{args.names}' не знайдено. Імена буде взято з моделі.")
        NAMES = model.names
        if isinstance(NAMES, dict):
            NAMES = [NAMES[i] for i in range(len(NAMES))]
    
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in NAMES]

    # --- Завантаження джерела ---
    image_type, initial_frame, cap = load_source(args.source)
    if initial_frame is None and cap is None:
        exit()

    # --- Глобальні змінні для зв'язку між потоками ---
    latest_frame = None
    processed_overlay = None
    processing_fps = 0.0
    total_detected_confidences = 0.0
    detected_frames_count = 0

    # --- Запуск потоку обробки ---
    processing_thread = threading.Thread(
        target=async_yolo_processing,
        args=(model, NAMES, COLORS, args),
        daemon=True # Потік завершиться разом з основною програмою
    )
    processing_thread.start()

    # --- Головний цикл для зчитування та відображення відео ---
    if image_type:
        latest_frame = initial_frame.copy()
        print("Обробка зображення...")
        time.sleep(1) # Даємо час потоку на обробку
        if processed_overlay is not None:
            result = cv2.addWeighted(initial_frame, 1.0, processed_overlay, 1.0, 0)
            output_filename = "output_" + os.path.basename(args.source)
            cv2.imwrite(output_filename, result)
            print(f"Оброблено та збережено зображення як {output_filename}")
            cv2.imshow("Detection Result", result)
            cv2.waitKey(0)
        else:
            print("Обробка зображення не вдалася.")
    else:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Обробка відео (Source FPS: {video_fps:.2f})... Натисніть 'q' для виходу.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Кінець відео потоку.")
                break
            
            # Оновлюємо глобальний кадр для потоку обробки
            latest_frame = frame
            
            # Беремо останній готовий результат (оверлей) з потоку обробки
            # Якщо результату ще немає, використовуємо порожній оверлей
            overlay = processed_overlay if processed_overlay is not None else np.zeros_like(frame, dtype=np.uint8)
            
            # Накладаємо результат на поточний кадр
            result = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)
            
            fps_text = f"Detection FPS: {processing_fps:.2f} | Backend: {backend_name}"
            cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Asynchronous YOLO Detection", result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # --- Завершення роботи ---
    print("Завершення роботи...")
    latest_frame = None # Сигнал для завершення циклу в потоці
    time.sleep(0.5)
    if cap: cap.release()
    cv2.destroyAllWindows()
    print("Програма завершена.")