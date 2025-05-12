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
    
    if source_file == "0":
        source_file = 0  # Веб-камера
        image_type = False
    else:
        image_type = source_file.split('.')[-1].lower() in img_formats
    
    if image_type:
        frame = cv2.imread(source_file)
    else:
        cap = cv2.VideoCapture(source_file)
    
    return image_type, frame if image_type else None, cap

def yolo_detection(frame, model, IMAGE_SIZE, NAMES, COLORS, args):
    overlay = np.zeros_like(frame, dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (IMAGE_SIZE, IMAGE_SIZE), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward().transpose((0, 2, 1))
    
    image_height, image_width, _ = frame.shape
    x_factor = image_width / IMAGE_SIZE
    y_factor = image_height / IMAGE_SIZE
    
    class_ids, confs, boxes = [], [], []
    for row in preds[0]:
        conf = row[4]
        classes_score = row[4:]
        _, _, _, max_idx = cv2.minMaxLoc(classes_score)
        class_id = max_idx[1]
        if classes_score[class_id] > args.tresh:
            confs.append(classes_score[class_id])
            class_ids.append(class_id)
            x, y, w, h = row[:4]
            left, top = int((x - 0.5 * w) * x_factor), int((y - 0.5 * h) * y_factor)
            width, height = int(w * x_factor), int(h * y_factor)
            boxes.append([left, top, width, height])
    
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.2, 0.5)
    if len(indexes) > 0:
        for i in indexes.flatten():
            left, top, width, height = boxes[i]
            class_id, score = class_ids[i], round(float(confs[i]), 3)
            color = COLORS[class_id]
            cv2.rectangle(overlay, (left, top), (left + width, top + height), color, args.thickness)
            text = f'{NAMES[class_id]} {score:.2f}'
            cv2.putText(overlay, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return overlay

def async_yolo_processing(model, IMAGE_SIZE, NAMES, COLORS, args):
    global latest_frame, processed_overlay, processing_fps
    frame_count = 0
    start_processing_time = time.time()
    
    while True:
        if latest_frame is not None:
            processed_overlay = yolo_detection(latest_frame.copy(), model, IMAGE_SIZE, NAMES, COLORS, args)
            frame_count += 1
            
            # Обчислюємо FPS розпізнавання кожну секунду
            elapsed_processing_time = time.time() - start_processing_time
            if elapsed_processing_time >= 1.0:
                processing_fps = frame_count / elapsed_processing_time
                frame_count = 0
                start_processing_time = time.time()

if __name__ == '__main__':
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/idiots3.mp4", help="Video")
    parser.add_argument("--names", type=str, default="data/class.names", help="Object Names")
    parser.add_argument("--model", type=str, default="./models/yolo11n-old.onnx", help="Pretrained Model")
    parser.add_argument("--tresh", type=float, default=0.25, help="Confidence Threshold")
    parser.add_argument("--thickness", type=int, default=2, help="Bounding Box Thickness")
    args = parser.parse_args()
    
    model = cv2.dnn.readNet(args.model)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    IMAGE_SIZE = 640
    with open(args.names, "r") as f:
        NAMES = [cname.strip() for cname in f.readlines()]
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in NAMES]
    
    image_type, frame, cap = load_source(args.source)
    latest_frame, processed_overlay = None, None
    processing_fps = 0.0  # Змінна для FPS розпізнавання

    # Запускаємо фоновий потік для обробки YOLO
    processing_thread = threading.Thread(target=async_yolo_processing, args=(model, IMAGE_SIZE, NAMES, COLORS, args), daemon=True)
    processing_thread.start()

    video_fps = cap.get(cv2.CAP_PROP_FPS) if not image_type else 0
    frame_time = 1 / video_fps if video_fps > 0 else 0

    while True:
        start_frame_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        latest_frame = frame.copy()
        overlay = processed_overlay if processed_overlay is not None else np.zeros_like(frame, dtype=np.uint8)
        result = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)

        # Відображення FPS на відео
        fps_text = f"Video FPS: {video_fps:.2f} | Detection FPS: {processing_fps:.2f}"
        cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("YOLO Detection", result)

        elapsed_time = time.time() - start_frame_time
        sleep_time = max(frame_time - elapsed_time, 0)  # Уникаємо від’ємного часу сну
        time.sleep(sleep_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Video FPS: {video_fps:.2f}")
    print(f"Average Detection FPS: {processing_fps:.2f}")