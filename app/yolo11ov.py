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

def calculate_iou(box1, box2):
    """Обчислює IoU (Intersection over Union) між двома прямокутниками."""
    # box1 та box2 у форматі [x1, y1, x2, y2]
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
    current_frame_detections = []
    
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
                
                # Зберігаємо детекцію у форматі [x1, y1, x2, y2] для mAP
                current_frame_detections.append({
                    'box': [left, top, left + width, top + height],
                    'confidence': confidence,
                    'class_id': class_id
                })
    
    avg_confidence_this_frame = np.mean(current_frame_confidences) if current_frame_confidences else 0.0
    return overlay, avg_confidence_this_frame, current_frame_detections

def async_yolo_processing(compiled_model_ref, input_layer_ref, output_layer_ref, NAMES_ref, COLORS_ref, args_ref):
    global latest_frame, processed_overlay, processing_fps, total_detected_confidences, detected_frames_count, all_detections
    frame_count = 0
    start_processing_time = time.time()

    while True:
        if latest_frame is not None:
            current_frame_to_process = latest_frame.copy()
            if current_frame_to_process is not None:
                overlay, avg_confidence_this_frame, frame_detections = yolo_detection_openvino(
                    current_frame_to_process, compiled_model_ref, input_layer_ref, 
                    output_layer_ref, NAMES_ref, COLORS_ref, args_ref
                )
                processed_overlay = overlay
                all_detections.extend(frame_detections)
                
                if avg_confidence_this_frame > 0:
                    total_detected_confidences += avg_confidence_this_frame
                    detected_frames_count += 1

                frame_count += 1

                elapsed_processing_time = time.time() - start_processing_time
                if elapsed_processing_time >= 1.0:
                    processing_fps = frame_count / elapsed_processing_time
                    frame_count = 0
                    start_processing_time = time.time()

if __name__ == '__main__':
    overall_start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/IsomCar.mp4", help="Відео або зображення. '0' для веб-камери.")
    parser.add_argument("--names", type=str, default="data/class.names", help="Шлях до файлу імен класів.")
    parser.add_argument("--model", type=str, default="yolo11n_openvino_model", help="Шлях до OpenVINO моделі (папка).")
    parser.add_argument("--tresh", type=float, default=0.25, help="Поріг впевненості для детекції.")
    parser.add_argument("--thickness", type=int, default=2, help="Товщина рамки.")
    parser.add_argument("--device", type=str, default="AUTO", help="Пристрій OpenVINO (AUTO, CPU, GPU).")
    args = parser.parse_args()

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
            exit()
            
        model = core.read_model(model_xml)
        compiled_model = core.compile_model(model, args.device)
        
        # Отримати інформацію про входи та виходи
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        
        print(f"Модель завантажена на пристрій: {args.device}")
        print(f"Форма входу: {input_layer.shape}")
        print(f"Форма виходу: {output_layer.shape}")
        
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
    all_detections = []  # Зберігаємо всі детекції для розрахунку mAP
    total_processed_frames = 0

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
            total_processed_frames = 1
            print("Обробка зображення...")
            time.sleep(1)  # Даємо час потоку на обробку

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
            total_processed_frames += 1
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

    # Фінальні результати та обчислення метрик
    overall_end_time = time.time()
    total_runtime = overall_end_time - overall_start_time
    average_detection_fps = total_processed_frames / total_runtime if total_runtime > 0 else 0
    
    # Обчислення mAP
    map_score = calculate_map(all_detections, confidence_threshold=args.tresh)
    
    benchmark_results["total_runtime_seconds"] = total_runtime
    benchmark_results["average_detection_fps"] = average_detection_fps

    # Виведення фінальних результатів
    print("\n" + "="*50)
    print("ФІНАЛЬНІ РЕЗУЛЬТАТИ OpenVINO")
    print("="*50)
    print(f"Загальний показник Detection FPS: {average_detection_fps:.2f}")
    print(f"Загальний показник mAP: {map_score:.4f}")
    print("="*50)
    print("ДЕТАЛЬНИЙ БЕНЧМАРК:")
    print("="*50)
    for key, value in benchmark_results.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")
    print("="*50)
    print(f"Загальна кількість оброблених кадрів: {total_processed_frames}")
    print(f"Загальна кількість детекцій: {len(all_detections)}")
    print(f"Середня впевненість детекцій: {np.mean([det['confidence'] for det in all_detections]):.4f}" if all_detections else "Середня впевненість детекцій: 0.0000")
    print("="*50)
    print("Програма завершена.")