import argparse
import cv2
import numpy as np
from openvino.runtime import Core
import yaml

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/road.mp4", help="Video")
    parser.add_argument("--names", type=str, default="data/class.names", help="Object Names")
    parser.add_argument("--xml", type=str, default="./models/openvino_model/yolo11n.xml", help="xml")
    parser.add_argument("--bin", type=str, default="./models/openvino_model/yolo11n.bin", help="bin")
    parser.add_argument("--tresh", type=float, default=0.25, help="Confidence Threshold")
    parser.add_argument("--thickness", type=int, default=2, help="Bounding Box Thickness")
    return parser.parse_args()

def load_names(yaml_file):
    with open(yaml_file, "r") as f:
        return [cname.strip() for cname in f.readlines()]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process_output(output, frame, class_names, conf_threshold, thickness):
    height, width = frame.shape[:2]
    boxes = []
    confidences = []
    class_ids = []
    
    # Транспонуємо вивід: з (1, 84, 8400) до (8400, 84)
    detections = output[0].T  # Тепер [8400, 84]
    
    for detection in detections:
        # Перші 4 значення - координати, 5-е - впевненість, решта - оцінки класів
        confidence = sigmoid(detection[4])  # Застосовуємо сигмоїду до впевненості
        
        if confidence > conf_threshold:
            scores = sigmoid(detection[5:])  # Сигмоїда до оцінок класів
            class_id = np.argmax(scores)
            class_conf = scores[class_id] * confidence
            
            if class_conf > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                if x >= 0 and y >= 0 and w > 0 and h > 0:
                    boxes.append([x, y, w, h])
                    confidences.append(float(class_conf))
                    if class_id < len(class_names):
                        class_ids.append(class_id)
                    else:
                        class_ids.append(0)

    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.45)
        if len(indices) > 0:
            for i in indices.flatten() if indices.ndim > 1 else indices:
                if i < len(boxes):
                    box = boxes[i]
                    x, y, w, h = box
                    label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
                    color = (0, 255, 0)
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, color, thickness)

    return frame

def main():
    args = parse_arguments()
    
    ie = Core()
    model = ie.read_model(model=args.xml, weights=args.bin)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    
    class_names = load_names(args.names)
    if not class_names:
        print("Не вдалося завантажити назви класів з metadata.yaml")
        return
    print(f"Loaded {len(class_names)} class names: {class_names}")
    
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Не вдалося відкрити відео: {args.source}")
        return
    
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        input_height, input_width = 640, 640
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_width, input_height), 
                                   swapRB=True, crop=False)
        
        outputs = compiled_model([blob])[output_layer]
        
        frame = process_output(outputs, frame, class_names, 
                             args.tresh, args.thickness)
        
        cv2.imshow('YOLOv11 OpenVINO', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()