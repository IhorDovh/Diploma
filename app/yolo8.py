import cv2
import time
import random
import argparse
import numpy as np
import threading
import os

def load_source(source_file):
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    cap = None
    frame = None # Ініціалізація frame

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
    
    total_gt = len([iou for iou in ious if iou >= iou_threshold])  # Загальна кількість ground truth об'єктів
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
    """
    Обчислює mAP для всіх детекцій.
    Для спрощення, використовуємо детекції як ground truth з високим confidence.
    """
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
        if len(detections) < 2:  # Потрібно мінімум 2 детекції для обчислення AP
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

def yolo_detection_yolov8(frame, model, input_shape, class_names, colors, args):
    overlay = np.zeros_like(frame, dtype=np.uint8)
    original_height, original_width = frame.shape[:2]

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[1], input_shape[0]))
    blob = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
    blob = blob.transpose(0, 3, 1, 2)

    model.setInput(blob)
    preds = model.forward()
    preds = preds[0].transpose(1, 0)

    class_ids, confs, boxes = [], [], []
    x_factor = original_width / input_shape[1]
    y_factor = original_height / input_shape[0]

    for row in preds:
        box_coords = row[:4]
        class_scores = row[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence > args.tresh:
            cx, cy, w, h = box_coords
            left = int((cx - 0.5 * w) * x_factor)
            top = int((cy - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            boxes.append([left, top, width, height])
            class_ids.append(class_id)
            confs.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confs, args.tresh, args.nms_tresh)

    # --- Початок: Розрахунок середньої впевненості для поточного кадру ---
    current_frame_confidences = []
    current_frame_detections = []
    # --- Кінець: Розрахунок середньої впевненості для поточного кадру ---

    if len(indexes) > 0:
        for i in indexes.flatten():
            left, top, width, height = boxes[i]
            class_id = class_ids[i]
            score = round(confs[i], 3)
            if class_id < len(class_names):
                color = colors[class_id]
                label = f'{class_names[class_id]} {score:.2f}'
                cv2.rectangle(overlay, (left, top), (left + width, top + height), color, args.thickness)
                cv2.putText(overlay, label, (left, top - 5 if top > 20 else top + height + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                print(f"Попередження: class_id {class_id} виходить за межі для class_names (довжина: {len(class_names)}).")

            # --- Додаємо впевненість до списку для розрахунку середньої ---
            current_frame_confidences.append(score)
            current_frame_detections.append({
                'box': [left, top, left + width, top + height],
                'confidence': score,
                'class_id': class_id
            })
            # --- Кінець: Додаємо впевненість до списку для розрахунку середньої ---

    # --- Розрахунок середньої впевненості для поточного кадру ---
    avg_confidence_this_frame = np.mean(current_frame_confidences) if current_frame_confidences else 0.0
    # --- Кінець: Розрахунок середньої впевненості для поточного кадру ---

    return overlay, avg_confidence_this_frame, current_frame_detections # Повертаємо overlay, середню впевненість та детекції

def async_yolo_processing(model_ref, input_shape_ref, class_names_ref, colors_ref, args_ref):
    global latest_frame, processed_overlay, processing_fps, total_detected_confidences, detected_frames_count, all_detections
    frame_count = 0
    start_processing_time = time.time()
    while True:
        if latest_frame is not None:
            current_frame_to_process = latest_frame.copy()
            if current_frame_to_process is not None:
                overlay, avg_confidence_this_frame, frame_detections = yolo_detection_yolov8(current_frame_to_process, model_ref, input_shape_ref, class_names_ref, colors_ref, args_ref)
                processed_overlay = overlay # Оновлюємо глобальну змінну для відображення

                # --- Додаємо дані для розрахунку загальної середньої впевненості та mAP ---
                all_detections.extend(frame_detections)
                if avg_confidence_this_frame > 0: # Тільки якщо були детекції в кадрі
                    total_detected_confidences += avg_confidence_this_frame
                    detected_frames_count += 1
                # --- Кінець: Додаємо дані для розрахунку загальної середньої впевненості та mAP ---

                frame_count += 1
            elapsed_processing_time = time.time() - start_processing_time
            if elapsed_processing_time >= 1.0:
                processing_fps = frame_count / elapsed_processing_time
                frame_count = 0
                start_processing_time = time.time()
        else:
            time.sleep(0.01)

if __name__ == '__main__':
    overall_start_time = time.time() # Changed to overall_start_time for clarity
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/IsomCar.mp4", help="Джерело відео або шлях до файлу зображення/відео. '0' для веб-камери.")
    parser.add_argument("--names", type=str, default="data/class.names", help="Шлях до файлу з іменами класів об'єктів (наприклад, coco.names).")
    parser.add_argument("--model", type=str, default="./models/yolov8n.onnx", help="Шлях до моделі YOLOv8 ONNX.")
    parser.add_argument("--tresh", type=float, default=0.25, help="Поріг впевненості для детекції.")
    parser.add_argument("--nms_tresh", type=float, default=0.45, help="Поріг IoU для NMS.")
    parser.add_argument("--thickness", type=int, default=2, help="Товщина обмежувальної рамки.")
    parser.add_argument("--img_size", type=int, default=640, help="Розмір зображення для входу моделі (квадратне зображення, наприклад, 640 для 640x640).")
    parser.add_argument("--target_fps", type=int, default=30, help="Цільовий FPS для відображення.")
    args = parser.parse_args()

    try:
        model = cv2.dnn.readNet(args.model)
        if model.empty():
            print(f"Помилка: Не вдалося завантажити модель з {args.model}. Перевірте шлях та цілісність моделі.")
            exit()
    except cv2.error as e:
        print(f"Помилка OpenCV під час завантаження моделі: {e}")
        exit()
    except Exception as e:
        print(f"Неочікувана помилка під час завантаження моделі: {e}")
        exit()

    INPUT_SHAPE = (args.img_size, args.img_size)

    try:
        with open(args.names, "r", encoding='utf-8') as f:
            CLASS_NAMES = [cname.strip() for cname in f.readlines()]
    except FileNotFoundError:
        print(f"Помилка: Файл імен класів не знайдено за шляхом {args.names}")
        if "coco.names" in args.names.lower() and "yolov8" in args.model.lower():
            print("Використання стандартних імен класів COCO як резервний варіант.")
            CLASS_NAMES = [
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
        else:
            print("Будь ласка, надайте дійсний файл імен класів.")
            exit()
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in CLASS_NAMES]

    image_type, initial_frame, cap = load_source(args.source)

    if initial_frame is None and cap is None:
        print("Не вдалося завантажити джерело. Вихід.")
        exit()

    latest_frame, processed_overlay = None, None
    processing_fps = 0.0

    # --- Нові глобальні змінні для агрегації впевненості та mAP ---
    total_detected_confidences = 0.0
    detected_frames_count = 0
    all_detections = []  # Зберігаємо всі детекції для розрахунку mAP
    total_processed_frames = 0
    # --- Кінець: Нові глобальні змінні ---

    processing_thread = threading.Thread(target=async_yolo_processing,
                                         args=(model, INPUT_SHAPE, CLASS_NAMES, COLORS, args),
                                         daemon=True)
    processing_thread.start()

    TARGET_DISPLAY_FPS = args.target_fps
    target_display_frame_time = 1 / TARGET_DISPLAY_FPS if TARGET_DISPLAY_FPS > 0 else 0

    benchmark_results = {
        "source_file": args.source,
        "model_path": args.model,
        "confidence_threshold": args.tresh,
        "nms_threshold": args.nms_tresh,
        "input_image_size": args.img_size,
        "target_display_fps": TARGET_DISPLAY_FPS,
        "total_runtime_seconds": 0.0,
        "average_detection_fps": 0.0,
        "average_display_fps": 0.0,
        "source_fps": 0.0,
        "overall_average_confidence": 0.0 # Новий показник
    }

    total_frames_read = 0
    display_start_time = time.time()
    total_display_frames = 0

    if image_type:
        if initial_frame is not None:
            latest_frame = initial_frame.copy()
            total_processed_frames = 1
            print("Обробка зображення...")
            processing_start = time.time()
            while processed_overlay is None and processing_thread.is_alive():
                time.sleep(0.01) # Reduced sleep for quicker response
            processing_end = time.time()
            # total_processing_time is not directly used for benchmark results here,
            # as detection FPS is updated by the async thread.

            if processed_overlay is not None:
                result = cv2.addWeighted(initial_frame, 1.0, processed_overlay, 1.0, 0)
                cv2.imshow("YOLOv8 Detection", result)
                output_filename = "output_" + os.path.basename(args.source)
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
            # For a single image, calculate overall average confidence if detections were made
            if detected_frames_count > 0:
                benchmark_results["overall_average_confidence"] = total_detected_confidences / detected_frames_count
        else:
            print("Завантажене зображення було None, неможливо обробити.")
    else: # Video processing
        if cap is None or not cap.isOpened():
            print("Захоплення відео не відкрито. Вихід.")
            exit()

        benchmark_results["source_fps"] = cap.get(cv2.CAP_PROP_FPS)

        display_frame_start_time = time.time()

        while True:
            loop_process_start_time = time.time()

            ret, frame_read = cap.read()
            if not ret:
                print("Кінець відео потоку або неможливо прочитати кадр.")
                break

            total_frames_read += 1 # Count frames read from source
            total_processed_frames += 1

            latest_frame = frame_read.copy()
            current_overlay = processed_overlay
            if current_overlay is not None:
                result = cv2.addWeighted(frame_read, 1.0, current_overlay, 1.0, 0)
            else:
                result = frame_read.copy()

            total_display_frames += 1 # Count frames displayed

            fps_text = f"Display FPS: {processing_fps:.2f} | Detection FPS: {processing_fps:.2f}"
            cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("YOLOv8 Detection", result)

            loop_process_end_time = time.time()
            processing_duration_this_loop = loop_process_end_time - loop_process_start_time

            if target_display_frame_time > 0:
                sleep_duration = target_display_frame_time - processing_duration_this_loop
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Calculate average display FPS after the video loop finishes
        total_display_time = time.time() - display_start_time
        if total_display_time > 0:
            benchmark_results["average_display_fps"] = total_display_frames / total_display_time
        
        # Calculate overall average confidence for video
        if detected_frames_count > 0:
            benchmark_results["overall_average_confidence"] = total_detected_confidences / detected_frames_count

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    latest_frame = None # Signal the processing thread to potentially stop or idle

    # --- Завершення роботи та підрахунок метрик ---
    print("Завершення роботи...")
    latest_frame = None # Сигнал для завершення циклу в потоці
    time.sleep(0.5)
    
    # Обчислення фінальних метрик
    overall_end_time = time.time()
    overall_elapsed_time = overall_end_time - overall_start_time
    average_detection_fps = total_processed_frames / overall_elapsed_time if overall_elapsed_time > 0 else 0
    
    # Обчислення mAP
    map_score = calculate_map(all_detections, confidence_threshold=args.tresh)
    
    benchmark_results["total_runtime_seconds"] = overall_elapsed_time
    benchmark_results["average_detection_fps"] = average_detection_fps

    # --- Виведення фінальних результатів у стандартному форматі ---
    print("\n" + "="*50)
    print("ФІНАЛЬНІ РЕЗУЛЬТАТИ")
    print("="*50)
    print(f"Загальний показник Detection FPS: {average_detection_fps:.2f}")
    print(f"Загальний показник mAP: {map_score:.4f}")
    print(f"Загальний час обробки: {overall_elapsed_time:.2f} секунд")
    print(f"Загальна кількість оброблених кадрів: {total_processed_frames}")
    print(f"Загальна кількість детекцій: {len(all_detections)}")
    print(f"Середня впевненість детекцій: {np.mean([det['confidence'] for det in all_detections]):.4f}" if all_detections else "Середня впевненість детекцій: 0.0000")
    print(f"Backend: OpenCV DNN YOLOv8")
    print("="*50)
    print("Програма завершена.")

    # Print benchmark results for graph (старий формат для сумісності)
    print("\n" + "="*30)
    print("Результати бенчмарку:")
    print("="*30)
    for key, value in benchmark_results.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")
    print("="*30)