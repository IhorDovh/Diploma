import ncnn # Потрібні встановлені Python-біндінги для NCNN
import cv2   # Для роботи з відео та зображеннями
import numpy as np # Для обробки даних
import yaml   # Для читання YAML файлу

# --- Налаштування ---
# Шляхи до файлів вашої NCNN моделі
NCNN_PARAM_PATH = './models/yolo11n_ncnn_model/model.ncnn.param'
NCNN_BIN_PATH = './models/yolo11n_ncnn_model/model.ncnn.bin'

# Шлях до YAML файлу з класами (COCO8)
YAML_PATH = './coco8.yaml'

# Шлях до вхідного відеофайлу (або 0 для веб-камери)
INPUT_VIDEO_PATH = './data/videos/idiots3.mp4' # Змініть на шлях до вашого відеофайлу

# Шлях для збереження вихідного відео (або None, якщо не потрібно зберігати)
OUTPUT_VIDEO_PATH = None # Змініть або залиште None

# Розмір входу моделі (має відповідати тому, що очікує модель)
MODEL_INPUT_SIZE = (640, 640) # Ширина, Висота

# Параметри для постобробки YOLO
# Тимчасово знижуємо поріг для налагодження
CONFIDENCE_THRESHOLD = 0.05 # Поріг впевненості для боксів (знижено для тестування)
NMS_THRESHOLD = 0.45      # Поріг для Non-Maximum Suppression (NMS)

# Кількість потоків для інференсу (CPU)
NUM_THREADS = 1

# Завантаження імен класів з YAML файлу
def load_class_names(yaml_path):
    try:
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        # Extract class names from the 'names' field
        class_names = list(data['names'].values()) if isinstance(data['names'], dict) else data['names']
        return class_names
    except Exception as e:
        print(f"Помилка завантаження YAML файлу {yaml_path}: {e}")
        exit()

CLASS_NAMES = load_class_names(YAML_PATH)
print(f"Завантажено {len(CLASS_NAMES)} класів з {YAML_PATH}: {CLASS_NAMES}")

# Кольори для рамок (можна згенерувати динамічно або задати вручну)
COLORS = np.random.uniform(0, 255, size=(len(CLASS_NAMES), 3))

# --- Ініціалізація NCNN ---
net = ncnn.Net()

# Налаштування опцій мережі
# net.opt.use_vulkan_compute = True # Розкоментуйте, якщо скомпільовано з Vulkan та хочете використовувати GPU
net.opt.num_threads = NUM_THREADS # Встановлюємо кількість потоків для CPU

print(f"Завантажуємо модель NCNN з {NCNN_PARAM_PATH} та {NCNN_BIN_PATH}...")
try:
    net.load_param(NCNN_PARAM_PATH)
    net.load_model(NCNN_BIN_PATH)
    print("Модель NCNN завантажено успішно.")
except Exception as e:
    print(f"Помилка завантаження моделі NCNN: {e}")
    exit()

# --- Ініціалізація відео ---
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

if not cap.isOpened():
    print(f"Помилка: Не вдалося відкрити відеофайл {INPUT_VIDEO_PATH}")
    exit()

# Отримання інформації про відео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Кодек для mp4

# Ініціалізація запису відео, якщо потрібно
out_writer = None
if OUTPUT_VIDEO_PATH is not None:
    out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    if not out_writer.isOpened():
         print(f"Помилка: Не вдалося створити вихідний відеофайл {OUTPUT_VIDEO_PATH}")
         out_writer = None # Вимикаємо запис, якщо не вдалося створити файл

print(f"Обробка відео: {INPUT_VIDEO_PATH}")
print(f"Роздільна здатність: {frame_width}x{frame_height}, FPS: {fps}")

# --- Обробка кадрів відео ---
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read() # Читаємо кадр

    if not ret:
        print("Відео потік завершено або помилка читання.")
        break

    frame_count += 1
    #print(f"Обробка кадру {frame_count}...")

    # 1. Передобробка кадру для NCNN
    # Зміна розміру
    img_resized = cv2.resize(frame, MODEL_INPUT_SIZE)

    # Конвертація в ncnn::Mat
    # YOLOv8/v11 NCNN експорт часто очікує RGB, хоча OpenCV читає як BGR за замовчуванням
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    in_mat = ncnn.Mat.from_pixels(img_rgb.data, 2, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])

    # Нормалізація (має відповідати тренувальній)
    # Ultralytics зазвичай ділить на 255
    normal_vals = [1/255.0, 1/255.0, 1/255.0]
    in_mat.substract_mean_normalize([], normal_vals) # Без віднімання середнього, тільки ділення

    # 2. Виконання інференсу
    extractor = net.create_extractor()

    # Введення даних та отримання виходу
    # Перевірте імена входу/виходу в .param файлі або документації моделі
    input_name = 'in0'
    output_name = 'out0'

    extractor.input(input_name, in_mat)
    out_mat = ncnn.Mat()
    ret_code = extractor.extract(output_name, out_mat)

    if ret_code != 0:
        print(f"Помилка при виконанні інференсу на кадрі {frame_count}, код: {ret_code}")
        continue # Пропустити кадр або обробити помилку

    # 3. Постобробка вихідних даних NCNN

    # --- Додано для налагодження ---
    # print(f"--- Налагодження кадру {frame_count} ---")
    # print(f"out_mat.w: {out_mat.w}")
    # print(f"out_mat.h: {out_mat.h}")
    # print(f"out_mat.c: {out_mat.c}")
    # print(f"out_mat.d: {out_mat.d}") # Додатковий вимір, якщо є
    # print(f"out_mat.shape: ({out_mat.c}, {out_mat.d}, {out_mat.h}, {out_mat.w})") # Порядок NCNN
    # print(f"out_mat.elemsize: {out_mat.elemsize}") # Розмір елемента в байтах
    # print(f"out_mat.elempack: {out_mat.elempack}") # Упаковка елементів
    
    # Перетворюємо в numpy масив
    np_out_mat = np.array(out_mat)
    # print(f"np_out_mat.shape: {np_out_mat.shape}")
    # print(f"np_out_mat.dtype: {np_out_mat.dtype}")
    # print("-------------------------------")
    # --- Кінець налагодження ---

    # Визначення кількості пропозицій та ознак з numpy масиву
    # За виводом налагодження, np_out_mat.shape: (84, 8400)
    # Припускаємо, що це (num_features, num_proposals)
    if np_out_mat.shape[0] == 84 and np_out_mat.shape[1] == 8400:
        num_features = np_out_mat.shape[0] # 84
        num_proposals = np_out_mat.shape[1] # 8400
        
        # Транспонуємо, щоб отримати форму (num_proposals, num_features)
        data = np_out_mat.T
        # print(f"Припущено формат (84, 8400), транспоновано до (8400, 84). Форма даних: {data.shape}")
    else:
        print(f"Помилка: Неочікуваний формат вихідних даних NCNN на кадрі {frame_count}.")
        print(f"Очікувана форма np_out_mat: (84, 8400), отримана: {np_out_mat.shape}")
        continue # Пропустити кадр

    # Забезпечуємо суміжність пам'яті
    data = np.ascontiguousarray(data)

    boxes = []
    confidences = []
    class_ids = []

    # Визначення кількості класів у виводі моделі
    # За припущенням, features = bbox (4) + objectness (1) + classes
    # Кількість класів у моделі = загальна кількість ознак - 5
    actual_num_model_classes = data.shape[1] - 5 # 84 - 5 = 79

    # Перевірка сумісності кількості класів
    if actual_num_model_classes != len(CLASS_NAMES):
        # print(f"Помилка: Кількість класів у виводі моделі ({actual_num_model_classes}) не відповідає кількості класів у YAML файлі ({len(CLASS_NAMES)}) на кадрі {frame_count}.")
        # print("Будь ласка, переконайтеся, що модель та YAML файл класів сумісні.")
        if actual_num_model_classes > len(CLASS_NAMES):
            print(f"Помилка: Модель виводить більше класів ({actual_num_model_classes}), ніж є в YAML ({len(CLASS_NAMES)}). Неможливо правильно відобразити всі класи.")
            break # Зупиняємо виконання, якщо модель виводить більше класів
        else:
            # print(f"Попередження: Кількість класів у виводі моделі ({actual_num_model_classes}) менша, ніж у YAML ({len(CLASS_NAMES)}). Будуть відображені тільки перші {actual_num_model_classes} класів з YAML.")
            num_classes_to_use = actual_num_model_classes
    else:
        num_classes_to_use = len(CLASS_NAMES)


    # Прохід по детекціях
    if data is not None:
        # Додано лічильник для виводу перших кількох детекцій для налагодження
        debug_detection_count = 0
        max_debug_detections = 10 # Виводити інформацію лише для перших 10 детекцій

        for detection in data:
            # Assuming structure is (bbox (4), objectness (1), classes (actual_num_model_classes))
            bbox_coords = detection[0:4]
            confidence = detection[4] # Об'єктна впевненість
            # Нарізаємо оцінки класів відповідно до фактичної кількості класів моделі
            class_scores = detection[5 : 5 + actual_num_model_classes]

            # --- Додано для налагодження сирих оцінок ---
            if debug_detection_count < max_debug_detections:
                print(f"Кадр {frame_count}, Детекція {debug_detection_count}:")
                print(f"  Об'єктна впевненість: {confidence:.6f}")
                if actual_num_model_classes > 0:
                    max_class_score = np.max(class_scores) if len(class_scores) > 0 else 0.0
                    max_class_id = np.argmax(class_scores) if len(class_scores) > 0 else -1
                    print(f"  Макс. оцінка класу: {max_class_score:.6f} (ID: {max_class_id})")
                    print(f"  Фінальна впевненість (об'єктність * клас): {confidence * max_class_score:.6f}")
                else:
                    print("  Модель не має класів.")
                print("-" * 20)
                debug_detection_count += 1
            # --- Кінець налагодження сирих оцінок ---


            # Перевірка довжини нарізаних class_scores
            if len(class_scores) != actual_num_model_classes:
                print(f"Помилка внутрішнього нарізання class_scores на кадрі {frame_count}. Очікувана довжина: {actual_num_model_classes}, отримана: {len(class_scores)}")
                continue # Пропустити цю детекцію

            # Визначення класу з найвищою оцінкою
            # Використовуємо тільки ті класи, які є в моделі
            class_id = -1
            class_score = 0.0
            final_confidence = 0.0

            if actual_num_model_classes > 0:
                class_id = np.argmax(class_scores)
                class_score = class_scores[class_id]
                final_confidence = confidence * class_score # Типово для YOLO

            # Основна перевірка порогу впевненості
            if final_confidence >= CONFIDENCE_THRESHOLD:
                # Координати боксу - потрібно перетворити з вихідного формату моделі
                # наприклад, з центру (cx, cy, w, h) до верхнього лівого кута (x1, y1, x2, y2)
                # та масштабувати до оригінального розміру кадру

                # Використовуємо bbox_coords, які ми вже нарізали
                center_x = bbox_coords[0] * frame_width / MODEL_INPUT_SIZE[0] # Масштабування X
                center_y = bbox_coords[1] * frame_height / MODEL_INPUT_SIZE[1] # Масштабування Y
                box_width = bbox_coords[2] * frame_width / MODEL_INPUT_SIZE[0] # Масштабування ширини
                box_height = bbox_coords[3] * frame_height / MODEL_INPUT_SIZE[1] # Масштабування висоти

                x1 = int(center_x - box_width / 2)
                y1 = int(center_y - box_height / 2)
                x2 = int(x1 + box_width)
                y2 = int(y1 + box_height)

                boxes.append([x1, y1, x2, y2])
                confidences.append(float(final_confidence))
                # Використовуємо class_id, отриманий з class_scores моделі
                class_ids.append(class_id)

    # Застосування Non-Maximum Suppression (NMS)
    # OpenCV має функцію для NMS, яка працює з форматом [x1, y1, x2, y2] або [x, y, w, h]
    # Використовуємо формат [x1, y1, w, h] для функції NMSBoxes
    boxes_xywh = [[x1, y1, x2 - x1, y2 - y1] for [x1, y1, x2, y2] in boxes]

    indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # --- Додано для налагодження після NMS ---
    print(f"--- Налагодження після NMS на кадрі {frame_count} ---")
    print(f"Кількість детекцій до NMS: {len(boxes)}")
    print(f"Кількість детекцій після NMS: {len(indices)}")
    if len(indices) > 0:
        print("Перші 5 детекцій після NMS:")
        for j in range(min(5, len(indices))):
            idx = np.array(indices).flatten()[j]
            print(f"  Бокс: {boxes[idx]}, Впевненість: {confidences[idx]:.6f}, Class ID (моделі): {class_ids[idx]}")
    print("---------------------------------------")
    # --- Кінець налагодження ---


    # 4. Відображення результатів на кадрі
    if len(indices) > 0:
        # Індекси повертаються як список списків або numpy масив залежно від OpenCV
        indices = np.array(indices).flatten()

        for i in indices:
            x1, y1, x2, y2 = boxes[i]
            confidence = confidences[i]
            class_id = class_ids[i] # Це class_id з моделі (0-78)

            # Перевіряємо, чи class_id знаходиться в межах доступних класів з YAML
            if class_id < len(CLASS_NAMES):
                # Малювання рамки
                color = COLORS[class_id % len(COLORS)] # Використовуємо class_id для кольору
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Малювання тексту (клас та впевненість)
                label = f"{CLASS_NAMES[class_id]}: {confidence:.2f}" # Використовуємо class_id для імені класу
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Текст чорний
            else:
                # Якщо class_id виходить за межі YAML, це означає, що модель виявила клас, якого немає в нашому списку.
                # Це не повинно статися, якщо actual_num_model_classes <= len(CLASS_NAMES).
                # Але якщо сталося, можемо відобразити загальну рамку без імені класу або з попередженням.
                print(f"Попередження: Виявлено class_id ({class_id}), який виходить за межі CLASS_NAMES ({len(CLASS_NAMES)}) після NMS на кадрі {frame_count}. Пропускаємо відображення цієї детекції.")
                # Або можна намалювати рамку без тексту:
                # color = (0, 0, 255) # Червоний колір для невідомих класів
                # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


    # 5. Відображення кадру або запис у файл
    cv2.imshow("NCNN YOLOv11 Inference", frame)

    if out_writer is not None:
        out_writer.write(frame)

    # Перервати відео, якщо натиснуто 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Завершення ---
cap.release()
if out_writer is not None:
    out_writer.release()
cv2.destroyAllWindows()
print("Обробку відео завершено.")
