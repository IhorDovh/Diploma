import ncnn
import cv2
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/road.mp4", help="Video")
    parser.add_argument("--names", type=str, default="data/class.names", help="Object Names")
    parser.add_argument("--param", type=str, default="./models/ncnn_model/model.ncnn.param", help="NCNN param file")
    parser.add_argument("--bin", type=str, default="./models/ncnn_model/model.ncnn.bin", help="NCNN bin file")
    parser.add_argument("--thresh", type=float, default=0.25, help="Confidence Threshold")
    parser.add_argument("--thickness", type=int, default=2, help="Bounding Box Thickness")
    return parser.parse_args()

def load_class_names(names_file):
    with open(names_file, "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]
    print("Завантажено класів:", len(class_names))
    print("Класи:", class_names)
    return class_names

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    args = parse_args()

    # Завантаження класів
    class_names = load_class_names(args.names)

    # Ініціалізація NCNN
    net = ncnn.Net()
    net.load_param(args.param)
    net.load_model(args.bin)

    # Відкриття відеофайлу
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Помилка: не вдалося відкрити відео.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Підготовка зображення
        input_size = 640
        img = cv2.resize(frame, (input_size, input_size))
        img = img.astype(np.float32) / 255.0

        # Перетворення у формат NCNN
        mat_in = ncnn.Mat.from_pixels(img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, input_size, input_size)

        # Виконання інференсу
        extractor = net.create_extractor()
        extractor.input("in0", mat_in)
        ret, mat_out = extractor.extract("out0")

        # Обробка результатів
        output = np.array(mat_out).reshape(8400, 84)  # Перетворюємо в (8400, 84)
        for detection in output:
            confidence = sigmoid(detection[4])  # Застосовуємо sigmoid до впевненості
            if confidence > args.thresh:
                class_scores = sigmoid(detection[5:])  # Останні 80 елементів — сирі оцінки класів
                class_id = np.argmax(class_scores)  # Знаходимо клас із найвищою ймовірністю
                class_confidence = class_scores[class_id] * confidence  # Загальна впевненість

                if class_id >= len(class_names) or class_id < 0:
                    print(f"Помилка: class_id {class_id} виходить за межі списку class_names (довжина: {len(class_names)})")
                    continue

                # Координати
                x = int(detection[0] * frame.shape[1] / input_size)
                y = int(detection[1] * frame.shape[0] / input_size)
                w = int(detection[2] * frame.shape[1] / input_size)
                h = int(detection[3] * frame.shape[0] / input_size)

                # Малювання
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), args.thickness)
                label = f"{class_names[class_id]}: {class_confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Показати результат
        cv2.imshow("NCNN Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()