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
    return overlay

def async_yolo_processing(model_ref, input_shape_ref, class_names_ref, colors_ref, args_ref):
    global latest_frame, processed_overlay, processing_fps
    frame_count = 0
    start_processing_time = time.time()
    while True:
        if latest_frame is not None:
            current_frame_to_process = latest_frame.copy()
            if current_frame_to_process is not None:
                processed_overlay = yolo_detection_yolov8(current_frame_to_process, model_ref, input_shape_ref, class_names_ref, colors_ref, args_ref)
                frame_count += 1
            elapsed_processing_time = time.time() - start_processing_time
            if elapsed_processing_time >= 1.0:
                processing_fps = frame_count / elapsed_processing_time
                frame_count = 0
                start_processing_time = time.time()
        else:
            time.sleep(0.01)

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/idiots3.mp4", help="Джерело відео або шлях до файлу зображення/відео. '0' для веб-камери.")
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

    processing_thread = threading.Thread(target=async_yolo_processing,
                                         args=(model, INPUT_SHAPE, CLASS_NAMES, COLORS, args),
                                         daemon=True)
    processing_thread.start()

    TARGET_DISPLAY_FPS = args.target_fps
    target_display_frame_time = 1 / TARGET_DISPLAY_FPS if TARGET_DISPLAY_FPS > 0 else 0

    # Variables to collect benchmark results
    total_frames_processed = 0
    total_display_frames = 0
    total_processing_time = 0
    total_display_time = 0
    
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
        "source_fps": 0.0
    }

    if image_type:
        if initial_frame is not None:
            latest_frame = initial_frame.copy()
            print("Обробка зображення...")
            processing_start = time.time()
            while processed_overlay is None and processing_thread.is_alive():
                time.sleep(0.01) # Reduced sleep for quicker response
            processing_end = time.time()
            total_processing_time = processing_end - processing_start
            total_frames_processed = 1 # Only one frame for an image

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

            latest_frame = frame_read.copy()
            current_overlay = processed_overlay
            if current_overlay is not None:
                result = cv2.addWeighted(frame_read, 1.0, current_overlay, 1.0, 0)
            else:
                result = frame_read.copy()

            total_display_frames += 1
            
            fps_text = f"Display FPS: {processing_fps:.2f} | Detection FPS: {processing_fps:.2f}" # processing_fps is already updated by the thread
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
        
        # Calculate average display FPS after the loop
        total_display_time = time.time() - display_frame_start_time
        if total_display_time > 0:
            benchmark_results["average_display_fps"] = total_display_frames / total_display_time

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    latest_frame = None # Signal the processing thread to potentially stop or idle

    end_time = time.time()
    benchmark_results["total_runtime_seconds"] = end_time - start_time
    benchmark_results["average_detection_fps"] = processing_fps # Last recorded processing FPS

    # Print benchmark results for graph
    print("\n" + "="*30)
    print("Результати бенчмарку:")
    print("="*30)
    for key, value in benchmark_results.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")
    print("="*30)

    # Optional: You can also return these results if this code were part of a larger function
    # return benchmark_results