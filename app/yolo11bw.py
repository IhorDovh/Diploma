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

def sharpen_image(image, strength=1):
    """
    Підвищення різкості зображення з можливістю регулювання сили ефекту.
    
    :param image: Вхідне зображення
    :param strength: Рівень різкості (1 - стандартний, 2 - сильний, 3 - екстремальний)
    :return: Зображення з підвищеною різкістю
    """
    # Базове зображення для результату
    result = image.copy()
    
    if strength == 1:
        # Стандартне ядро різкості
        kernel = np.array([[-1,-1,-1], 
                           [-1, 9,-1],
                           [-1,-1,-1]])
        result = cv2.filter2D(image, -1, kernel)
    
    elif strength == 2:
        # Більш сильне ядро різкості
        kernel = np.array([[-2,-2,-2], 
                           [-2, 17,-2],
                           [-2,-2,-2]])
        result = cv2.filter2D(image, -1, kernel)
    
    elif strength == 3:
        # Екстремальне підвищення різкості з використанням комбінації фільтрів
        
        # Спочатку застосуємо UnSharp Mask (USM) фільтр
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        usm = cv2.addWeighted(image, 2.0, gaussian, -1.0, 0)
        
        # Далі застосуємо сильне ядро різкості
        kernel = np.array([[-2,-2,-2], 
                           [-2, 17,-2],
                           [-2,-2,-2]])
        result = cv2.filter2D(usm, -1, kernel)
        
        # Додамо корекцію контрасту
        if len(image.shape) == 2:  # Якщо зображення вже монохромне
            lab = cv2.cvtColor(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    return result

def apply_preprocessing(image, preprocessing_method):
    """
    Застосування різних методів попередньої обробки зображення для покращення продуктивності.
    
    :param image: Вхідне зображення (монохромне)
    :param preprocessing_method: Метод обробки (0-6)
    :return: Оброблене зображення
    """
    if preprocessing_method == 0:
        # Без додаткової обробки, тільки підвищення різкості
        return image
    
    elif preprocessing_method == 1:
        # Адаптивна еквалізація гістограми з обмеженням контрасту (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    elif preprocessing_method == 2:
        # Видалення шуму з використанням двостороннього фільтру
        # Зберігає краї при видаленні шуму
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    elif preprocessing_method == 3:
        # Адаптивний пороговий фільтр (бінаризація)
        # Допомагає виділити краї та контури
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        # Поєднуємо з оригіналом для збереження деталей
        return cv2.addWeighted(image, 0.7, binary, 0.3, 0)
    
    elif preprocessing_method == 4:
        # Фільтр Собеля для виділення країв
        # Виділяє градієнти яскравості (краї об'єктів)
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        # Нормалізація результату
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # Поєднуємо з оригіналом для збереження деталей
        return cv2.addWeighted(image, 0.7, sobel, 0.3, 0)
    
    elif preprocessing_method == 5:
        # Морфологічні операції для видалення шуму та підкреслення форм
        kernel = np.ones((3, 3), np.uint8)
        # Відкриття (ерозія з подальшою дилатацією) для видалення шуму
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        # Закриття (дилатація з подальшою ерозією) для заповнення малих отворів
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        return closing
    
    elif preprocessing_method == 6:
        # Нормалізація зображення для підвищення контрасту
        # Розтягування гістограми на повний діапазон
        norm_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return norm_img
    
    return image

def yolo_detection(frame, model, IMAGE_SIZE, NAMES, COLORS, current_settings):
    overlay = np.zeros_like(frame, dtype=np.uint8)
    
    # Конвертація зображення в чорно-біле для моделі
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Застосування попередньої обробки зображення
    preprocessed_gray = apply_preprocessing(gray_frame, current_settings["preprocessing"])
    
    # Застосування різкості до вхідного зображення з вказаним рівнем інтенсивності
    sharpened_gray = sharpen_image(preprocessed_gray, current_settings["sharpness"])
    
    # Зміна розмірності для моделі (з одного каналу в три)
    gray_frame_3ch = cv2.cvtColor(sharpened_gray, cv2.COLOR_GRAY2BGR)
    
    # Зберігаємо зображення, яке бачить модель для відображення
    model_input = cv2.resize(gray_frame_3ch, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Використовуємо покращене чорно-біле зображення як вхідні дані для моделі
    blob = cv2.dnn.blobFromImage(gray_frame_3ch, 1/255.0, (IMAGE_SIZE, IMAGE_SIZE), swapRB=True, crop=False)
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
        if classes_score[class_id] > current_settings["tresh"]:
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
            cv2.rectangle(overlay, (left, top), (left + width, top + height), color, current_settings["thickness"])
            text = f'{NAMES[class_id]} {score:.2f}'
            cv2.putText(overlay, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return overlay, sharpened_gray, model_input

def async_yolo_processing():
    global latest_frame, processed_overlay, processing_fps, gray_frame_global, model_input_global
    global current_settings
    
    frame_count = 0
    start_processing_time = time.time()
    
    while True:
        if latest_frame is not None:
            # Використовуємо копію поточних налаштувань, щоб уникнути конфліктів між потоками
            settings_copy = current_settings.copy()
            
            frame_start_time = time.time()
            processed_overlay, gray_frame_global, model_input_global = yolo_detection(
                latest_frame.copy(), model, IMAGE_SIZE, NAMES, COLORS, settings_copy)
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
    parser.add_argument("--show_grayscale", default=True, action="store_true", help="Show grayscale input")
    parser.add_argument("--show_model_view", action="store_true", help="Show what the model sees")
    parser.add_argument("--sharpness", type=int, default=1, choices=[1, 2, 3], 
                        help="Sharpness level (1=Standard, 2=Strong, 3=Extreme)")
    parser.add_argument("--preprocessing", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6],
                        help="Preprocessing method (0=None, 1=CLAHE, 2=Bilateral, 3=Adaptive Threshold, 4=Sobel, 5=Morphological, 6=Normalization)")
    args = parser.parse_args()
    
    model = cv2.dnn.readNet(args.model)
    
    IMAGE_SIZE = 640
    with open(args.names, "r") as f:
        NAMES = [cname.strip() for cname in f.readlines()]
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in NAMES]
    
    image_type, frame, cap = load_source(args.source)
    latest_frame, processed_overlay = None, None
    gray_frame_global = None  # Глобальна змінна для зберігання чорно-білого кадру
    model_input_global = None  # Глобальна змінна для зберігання зображення, яке бачить модель
    processing_fps = 0.0  # Змінна для FPS розпізнавання
    
    # Центральний словник налаштувань для спільного доступу між потоками
    current_settings = {
        "tresh": args.tresh,
        "thickness": args.thickness,
        "show_grayscale": args.show_grayscale,
        "show_model_view": args.show_model_view,
        "sharpness": args.sharpness,
        "preprocessing": args.preprocessing
    }

    processing_thread = threading.Thread(target=async_yolo_processing, daemon=True)
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
        
        # Формуємо результат відображення в залежності від обраного режиму
        if current_settings["show_model_view"] and model_input_global is not None:
            # Відображаємо те, що бачить модель (640x640 з підвищеною різкістю)
            # Змінюємо розмір для відображення в повному вікні
            model_view_resized = cv2.resize(model_input_global, (frame.shape[1], frame.shape[0]))
            result = model_view_resized.copy()
            display_mode = "Model View Mode"
        elif current_settings["show_grayscale"] and gray_frame_global is not None:
            # Показувати чорно-біле зображення з підвищеною різкістю
            gray_frame_3ch = cv2.cvtColor(gray_frame_global, cv2.COLOR_GRAY2BGR)
            result = cv2.addWeighted(gray_frame_3ch, 1.0, overlay, 1.0, 0)
            display_mode = "Grayscale Sharpened Mode"
        else:
            # Показувати звичайне кольорове зображення з накладеною детекцією
            result = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)
            display_mode = "Color Mode"

        # Відображення FPS на відео
        fps_text = f"Video FPS: {video_fps:.2f} | Detection FPS: {processing_fps:.2f}"
        cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Додаємо текст про режим відображення, рівень різкості та метод попередньої обробки
        preproc_names = ["None", "CLAHE", "Bilateral", "Adaptive Threshold", "Sobel", "Morphological", "Normalization"]
        preproc_name = preproc_names[current_settings["preprocessing"]]
        cv2.putText(result, f"{display_mode} (Sharpness: {current_settings['sharpness']}, Preproc: {preproc_name})", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Додамо підказку щодо зміни режимів перегляду та різкості
        cv2.putText(result, "Press 'g' for grayscale, 'm' for model view, 'c' for color", 
                   (10, result.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, "Press '1', '2', '3' to change sharpness level", 
                   (10, result.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, "Press '0'-'6' to change preprocessing method", 
                   (10, result.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLO Detection", result)

        elapsed_time = time.time() - start_frame_time
        sleep_time = max(frame_time - elapsed_time, 0)  # Уникаємо від'ємного часу сну
        time.sleep(sleep_time)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            current_settings["show_grayscale"] = True
            current_settings["show_model_view"] = False
            print("Switched to Grayscale mode")
        elif key == ord('m'):
            current_settings["show_model_view"] = True
            current_settings["show_grayscale"] = False
            print("Switched to Model View mode")
        elif key == ord('c'):
            current_settings["show_grayscale"] = False
            current_settings["show_model_view"] = False
            print("Switched to Color mode")
        # Зміна рівня різкості
        elif key == ord('1'):
            current_settings["sharpness"] = 1
            print("Sharpness level set to 1 (Standard)")
        elif key == ord('2'):
            current_settings["sharpness"] = 2
            print("Sharpness level set to 2 (Strong)")
        elif key == ord('3'):
            current_settings["sharpness"] = 3
            print("Sharpness level set to 3 (Extreme)")
        # Зміна методу попередньої обробки
        elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
            preproc_idx = int(chr(key))
            if 0 <= preproc_idx <= 6:
                current_settings["preprocessing"] = preproc_idx
                preproc_names = ["None", "CLAHE", "Bilateral", "Adaptive Threshold", "Sobel", "Morphological", "Normalization"]
                print(f"Preprocessing method set to {preproc_idx} ({preproc_names[preproc_idx]})")
    
    cap.release()
    cv2.destroyAllWindows()
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Video FPS: {video_fps:.2f}")
    print(f"Average Detection FPS: {processing_fps:.2f}")