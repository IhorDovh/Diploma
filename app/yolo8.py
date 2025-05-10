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
            return True, None, None # Повертаємо тип зображення, але з помилкою завантаження
    else:
        cap = cv2.VideoCapture(source_file)
        if not cap.isOpened():
            print(f"Помилка: Не вдалося відкрити джерело відео {source_file}")
            return False, None, None # Повертаємо тип відео, але з помилкою відкриття

    return image_type, frame if image_type else None, cap

def yolo_detection_yolov8(frame, model, input_shape, class_names, colors, args):
    overlay = np.zeros_like(frame, dtype=np.uint8)
    original_height, original_width = frame.shape[:2]

    # Попередня обробка
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # YOLOv8 зазвичай очікує RGB
    img = cv2.resize(img, (input_shape[1], input_shape[0])) # Зміна розміру до вхідного розміру моделі (h, w)
    blob = np.expand_dims(img.astype(np.float32) / 255.0, axis=0) # Додавання розмірності пакету та нормалізація
    blob = blob.transpose(0, 3, 1, 2) # Формат HWC в CHW (batch, channels, height, width)

    model.setInput(blob)
    preds = model.forward() # Для YOLOv8 це може бути основний вихідний шар

    # Вихід YOLOv8 ONNX зазвичай (batch_size, 4 + num_classes, num_detections)
    # наприклад, (1, 84, 8400) для COCO (80 класів), де 84 = 4 (cx, cy, w, h) + 80 (class_scores)
    # Вихід потрібно транспонувати до (batch_size, num_detections, 4 + num_classes)
    preds = preds[0].transpose(1, 0) # З (84, 8400) до (8400, 84)

    class_ids, confs, boxes = [], [], []

    # Розрахунок коефіцієнтів масштабування
    x_factor = original_width / input_shape[1]
    y_factor = original_height / input_shape[0]

    for row in preds:
        # Перші 4 значення - це bbox: cx, cy, w, h
        # Решта - оцінки класів
        box_coords = row[:4]
        class_scores = row[4:]

        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence > args.tresh:
            cx, cy, w, h = box_coords
            # Перетворення центральних координат та розмірів у top-left, width, height
            left = int((cx - 0.5 * w) * x_factor)
            top = int((cy - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            boxes.append([left, top, width, height])
            class_ids.append(class_id)
            confs.append(float(confidence))

    # Non-Maximum Suppression
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
            time.sleep(0.01) # Невеликий відпочинок, щоб не завантажувати CPU


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
    args = parser.parse_args()

    try:
        model = cv2.dnn.readNet(args.model)
        if model.empty():
            print(f"Помилка: Не вдалося завантажити модель з {args.model}. Перевірте шлях та цілісність моделі.")
            exit()
        # model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) # Розкоментуйте для CUDA
        # model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)   # Розкоментуйте для CUDA
    except cv2.error as e:
        print(f"Помилка OpenCV під час завантаження моделі: {e}")
        exit()
    except Exception as e:
        print(f"Неочікувана помилка під час завантаження моделі: {e}")
        exit()

    INPUT_SHAPE = (args.img_size, args.img_size) # (height, width)

    try:
        with open(args.names, "r", encoding='utf-8') as f: # Додано encoding
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
            ] # 80 класів COCO
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

    if image_type:
        if initial_frame is not None:
            latest_frame = initial_frame.copy()
            # Зачекайте, поки завершиться перша обробка
            print("Обробка зображення...")
            while processed_overlay is None and processing_thread.is_alive():
                time.sleep(0.1)

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
    else: # Відео або веб-камера
        if cap is None or not cap.isOpened():
            print("Захоплення відео не відкрито. Вихід.")
            exit()

        video_fps_cap = cap.get(cv2.CAP_PROP_FPS)
        frame_time = 1 / video_fps_cap if video_fps_cap > 0 else 0.033 # ~30 FPS за замовчуванням

        actual_frames_displayed = 0
        display_start_time = time.time()
        display_fps = 0.0

        while True:
            loop_start_time = time.time()

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

            actual_frames_displayed += 1
            elapsed_display_time = time.time() - display_start_time
            if elapsed_display_time >= 1.0:
                display_fps = actual_frames_displayed / elapsed_display_time
                actual_frames_displayed = 0
                display_start_time = time.time()

            fps_text = f"Display FPS: {display_fps:.2f} | Detection FPS: {processing_fps:.2f}"
            cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("YOLOv8 Detection", result)

            elapsed_loop_time = time.time() - loop_start_time
            # sleep_time = max(0, frame_time - elapsed_loop_time) # Може робити ривки, якщо детекція повільна
            # time.sleep(sleep_time)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    latest_frame = None # Сигнал потоку для чистого завершення
    # if processing_thread.is_alive():
    # print("Очікування завершення потоку обробки...")
    # processing_thread.join(timeout=2.0) # Для daemon=True це не так критично

    end_time = time.time()
    print(f"Загальний час виконання: {end_time - start_time:.2f} секунд")
    if not image_type and 'video_fps_cap' in locals() and video_fps_cap > 0:
        print(f"FPS захоплення відео: {video_fps_cap:.2f}")
    print(f"Останнє зареєстроване FPS детекції: {processing_fps:.2f} (приблизне середнє за останню секунду обробки)")