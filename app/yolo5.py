import cv2
import time
import random
import argparse
import numpy as np
import threading
import os
import onnxruntime as ort

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

def yolo_detection(frame, session, IMAGE_SIZE, NAMES, COLORS, args):
    overlay = np.zeros_like(frame, dtype=np.uint8)
    
    # Preprocess input
    img = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension: [1, 3, 640, 640]
    
    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})[0]  # Shape: [1, 25200, 85]
    
    # Image dimensions for scaling
    image_height, image_width, _ = frame.shape
    x_factor = image_width / IMAGE_SIZE
    y_factor = image_height / IMAGE_SIZE
    
    # Process predictions
    class_ids, confs, boxes = [], [], []
    for pred in outputs[0]:
        obj_conf = pred[4]  # Objectness score
        class_scores = pred[5:]  # Class probabilities
        obj_conf = 1 / (1 + np.exp(-obj_conf))  # Sigmoid
        class_scores = 1 / (1 + np.exp(-class_scores))  # Sigmoid
        
        max_score = np.max(class_scores)
        class_id = np.argmax(class_scores)
        
        conf = obj_conf * max_score
        if conf > args.tresh:
            x, y, w, h = pred[:4]
            left = int((x - 0.5 * w) * x_factor)
            top = int((y - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            boxes.append([left, top, width, height])
            confs.append(float(conf))
            class_ids.append(class_id)
    
    # Apply Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confs, args.tresh, 0.45)
    
    # Draw detections
    if len(indexes) > 0:
        for i in indexes.flatten():
            left, top, width, height = boxes[i]
            class_id, score = class_ids[i], round(float(confs[i]), 3)
            color = COLORS[class_id]
            cv2.rectangle(overlay, (left, top), (left + width, top + height), color, args.thickness)
            text = f'{NAMES[class_id]} {score:.2f}'
            cv2.putText(overlay, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return overlay

def async_yolo_processing():
    global latest_frame, processed_overlay, processing_fps
    frame_count = 0
    start_processing_time = time.time()
    
    while True:
        if latest_frame is not None:
            frame_start_time = time.time()
            processed_overlay = yolo_detection(latest_frame.copy(), session, IMAGE_SIZE, NAMES, COLORS, args)
            frame_count += 1
            
            elapsed_processing_time = time.time() - start_processing_time
            if elapsed_processing_time >= 1.0:
                processing_fps = frame_count / elapsed_processing_time
                frame_count = 0
                start_processing_time = time.time()

if __name__ == '__main__':
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/idiots3.mp4", help="Video or image file")
    parser.add_argument("--names", type=str, default="data/class.names", help="Path to class names file")
    parser.add_argument("--model", type=str, default="./models/yolov5s.onnx", help="Path to YOLOv5s ONNX model")
    parser.add_argument("--tresh", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--thickness", type=int, default=2, help="Bounding box thickness")
    args = parser.parse_args()
    
    # Load ONNX model with ONNX Runtime
    session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    
    # Model input size
    IMAGE_SIZE = 640
    
    # Load class names
    with open(args.names, "r") as f:
        NAMES = [cname.strip() for cname in f.readlines()]
    
    # Generate random colors for each class
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in NAMES]
    
    # Load source (image or video)
    image_type, frame, cap = load_source(args.source)
    latest_frame, processed_overlay = None, None
    processing_fps = 0.0  # Variable for detection FPS

    # Start async processing thread
    processing_thread = threading.Thread(target=async_yolo_processing, daemon=True)
    processing_thread.start()

    # Get video FPS (if applicable)
    video_fps = cap.get(cv2.CAP_PROP_FPS) if not image_type else 0
    frame_time = 1 / video_fps if video_fps > 0 else 0

    while True:
        start_frame_time = time.time()

        if image_type:
            latest_frame = frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                break
            latest_frame = frame.copy()

        # Use latest processed overlay or empty overlay
        overlay = processed_overlay if processed_overlay is not None else np.zeros_like(frame, dtype=np.uint8)
        result = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)

        # Display FPS
        fps_text = f"Video FPS: {video_fps:.2f} | Detection FPS: {processing_fps:.2f}"
        cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show result
        cv2.imshow("YOLOv5s Detection", result)

        # Control frame rate
        elapsed_time = time.time() - start_frame_time
        sleep_time = max(frame_time - elapsed_time, 0)
        time.sleep(sleep_time)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    if not image_type:
        cap.release()
    cv2.destroyAllWindows()
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Video FPS: {video_fps:.2f}")
    print(f"Average Detection FPS: {processing_fps:.2f}")