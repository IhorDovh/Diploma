import cv2
import time
import random
import argparse
import numpy as np
# import onnxruntime as ort

def loadSource(source_file):
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    key = 1 # 1 = Video, 0 = Image
    frame = None
    cap = None

    # Source from webcam
    if(source_file == "0"):
        image_type = False
        source_file = 0    
    else:
        image_type = source_file.split('.')[-1].lower() in img_formats

    # Open Image or Video
    if(image_type):
        frame = cv2.imread(source_file)
        key = 0
    else:
        cap = cv2.VideoCapture(source_file)

    return image_type, key, frame, cap

if __name__ == '__main__':
    # Add Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/road.mp4", help="Video")
    parser.add_argument("--names", type=str, default="data/class.names", help="Object Names")
    parser.add_argument("--model", type=str, default="./models/yolo11n-old.onnx", help="Pretrained Model")
    parser.add_argument("--tresh", type=float, default=0.25, help="Confidence Threshold")
    parser.add_argument("--thickness", type=int, default=2, help="Line Thickness on Bounding Boxes")
    args = parser.parse_args()

    # Load Model
    model = cv2.dnn.readNet(args.model)

    IMAGE_SIZE = 640
    NAMES = []
    with open(args.names, "r") as f:
        NAMES = [cname.strip() for cname in f.readlines()]
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in NAMES]

    source_file = args.source
    # Load Source
    image_type, key, frame, cap = loadSource(source_file)
    grabbed = True

    while(1):
        if not image_type:
            (grabbed, frame) = cap.read()

        if not grabbed:
            exit()

        image = frame.copy()
    
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (IMAGE_SIZE, IMAGE_SIZE), swapRB=True, crop=False)

        # Extract output detection
        class_ids, confs, boxes = list(), list(), list()

        model.setInput(blob)
        preds = model.forward()
        preds = preds.transpose((0, 2, 1))

        image_height, image_width, _ = image.shape
        x_factor = image_width / IMAGE_SIZE
        y_factor = image_height / IMAGE_SIZE

        rows = preds[0].shape[0]

        for i in range(rows):
            row = preds[0][i]
            conf = row[4]
            
            classes_score = row[4:]
            _,_,_, max_idx = cv2.minMaxLoc(classes_score)
            class_id = max_idx[1]
            if (classes_score[class_id] > .25):
                confs.append(classes_score[class_id])
                label = NAMES[int(class_id)]
                class_ids.append(class_id)
                
                #extract boxes
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)                

        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.2, 0.5)         
        
        for i in indexes:
            box = boxes[i]
            class_id = class_ids[i]
            score = confs[i]

            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            # Draw Bounding box            
            cv2.rectangle(image, (left, top), (left + width, top + height), COLORS[class_id], args.thickness)
            # Draw class name and confidence score

            # Name
            name = NAMES[class_id]    
            # Confidence Score
            score = round(float(score), 3)            
            name += f' {str(score)}'

            font_size = args.thickness / 2.5
            margin = args.thickness * 2
            cv2.putText(image, name, (left, top - margin), cv2.FONT_HERSHEY_SIMPLEX, font_size, COLORS[class_id], args.thickness)

        grabbed = False
        cv2.imshow("Detected",image)
        if cv2.waitKey(key) ==  ord('q'):
            break        