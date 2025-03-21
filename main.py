import os
import cv2
import numpy as np
import pandas as pd
from rembg import remove
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

# Константы
TEST_DIR = "TEST"
SUBMISSION_PATH = "submission.csv"
TRUE_DATA_PATH = "test_data.csv"  # Файл с истинными метками

CATEGORY_NAMES = ["стол", "стул", "сумка", "детская одежда"]
COLOR_NAMES = ["бежевый", "белый", "бирюзовый", "бордовый", "голубой", "жёлтый",
               "зелёный", "золотой", "коричневый", "красный", "оранжевый",
               "розовый", "серебристый", "серый", "синий", "фиолетовый",
               "чёрный", "разноцветный"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CATEGORY_CLASSES = 4
NUM_COLOR_CLASSES = 18

COLOR_RGB_MAP = {
    "бежевый": (200, 173, 127),
    "белый": (255, 255, 255),
    "бирюзовый": (64, 224, 208),
    "бордовый": (128, 0, 32),
    "голубой": (135, 206, 250),
    "жёлтый": (255, 255, 0),
    "зелёный": (0, 128, 0),
    "золотой": (212, 175, 55),
    "коричневый": (139, 69, 19),
    "красный": (255, 0, 0),
    "оранжевый": (255, 165, 0),
    "розовый": (255, 192, 203),
    "серебристый": (192, 192, 192),
    "серый": (128, 128, 128),
    "синий": (0, 0, 255),
    "фиолетовый": (128, 0, 128),
    "чёрный": (0, 0, 0),
    "разноцветный": (0, 0, 0)  # Разноцветный не имеет конкретного RGB
}

# Загрузка модели для категорий
def load_category_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель категорий не найдена: {model_path}")
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, NUM_CATEGORY_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Загрузка модели для цветов
def load_color_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель цветов не найдена: {model_path}")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_COLOR_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Трансформации изображений
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Функция предсказания категории
def predict_category(image, category_model):
    try:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = category_model(image)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        predicted_category = CATEGORY_NAMES[np.argmax(probabilities)]
        return predicted_category
    except Exception as e:
        print(f"Ошибка предсказания категории: {e}")
        return "unknown"

# Функция предсказания цвета
def predict_color(image, color_model, threshold=0.5):
    try:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = color_model(image)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        predicted_color_index = np.argmax(probabilities)
        max_probability = probabilities[predicted_color_index]

        # Если вероятность ниже порога, считаем объект разноцветным
        if max_probability < threshold:
            return "разноцветный"

        predicted_color = COLOR_NAMES[predicted_color_index]
        return predicted_color
    except Exception as e:
        print(f"Ошибка предсказания цвета: {e}")
        return "разноцветный"

# Функция для получения доминирующего цвета
def get_dominant_color(image):
    """
    Функция для получения доминирующего цвета на изображении.
    Возвращает среднее значение RGB.
    """
    # Преобразуем изображение в массив RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    # Убираем фон
    image_no_bg = remove(image)

    # Извлекаем непрозрачные пиксели
    mask = image_no_bg[:, :, 3] > 0  # Маска для непрозрачных пикселей
    rgb_values = image_no_bg[mask][:, :3]  # Берем только RGB-компоненты

    if len(rgb_values) == 0:
        return None  # Если нет видимых пикселей

    # Вычисляем средний цвет
    mean_color = np.mean(rgb_values, axis=0).astype(int)
    return tuple(mean_color)

# Функция предсказания цвета с RGB-проверкой
def predict_color_with_rgb_check(image, color_model, threshold=0.5):
    """
    Функция предсказания цвета с дополнительной проверкой по RGB.
    """
    try:
        # Получаем предсказание модели
        predicted_color = predict_color(image, color_model, threshold)

        # Получаем доминирующий цвет
        dominant_color = get_dominant_color(image)
        if dominant_color is None:
            return predicted_color  # Если не удалось определить доминирующий цвет

        # Сравниваем доминирующий цвет с RGB-значениями из словаря
        min_distance = float('inf')
        closest_color = None
        for color_name, rgb in COLOR_RGB_MAP.items():
            distance = np.linalg.norm(np.array(dominant_color) - np.array(rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name

        # Если расстояние между доминирующим цветом и предсказанным слишком большое,
        # используем доминирующий цвет как основное предсказание
        if closest_color != predicted_color and min_distance > 50:  # Пороговое значение
            return closest_color

        return predicted_color
    except Exception as e:
        print(f"Ошибка предсказания цвета с RGB-проверкой: {e}")
        return "разноцветный"

# Генерация CSV-файла
def create_submission(category_model, color_model):
    image_files = [f for f in os.listdir(TEST_DIR) if f.endswith((".png", ".jpg", ".jpeg"))]
    if not image_files:
        raise ValueError(f"В папке {TEST_DIR} нет изображений.")

    results = []
    for img_file in image_files:
        img_path = Path(TEST_DIR) / img_file
        img_id = os.path.splitext(img_file)[0]

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Ошибка загрузки: {img_file}")
            continue

        img_no_bg = remove(img)

        # Предсказание цвета с RGB-проверкой
        predicted_color = predict_color_with_rgb_check(img_no_bg, color_model)

        # Предсказание категории
        predicted_category = predict_category(img_no_bg, category_model)

        # Отладочная информация
        print(f"Image: {img_file}")
        print(f"Predicted Category: {predicted_category}")
        print(f"Predicted Color: {predicted_color}")

        # Формируем запись для CSV
        results.append({
            "id": img_id,
            "category": predicted_category,
            "predict_color": predicted_color  # Только предсказанный цвет
        })

    # Создаем DataFrame и сохраняем в CSV
    df = pd.DataFrame(results)
    df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Результат сохранен: {SUBMISSION_PATH}")

# Расчет метрик
def calculate_metrics(true_labels, predicted_labels):
    # Преобразуем метки в числовые индексы
    label_to_index = {label: idx for idx, label in enumerate(COLOR_NAMES)}
    y_true_idx = [label_to_index[label] for label in true_labels]
    y_pred_idx = [label_to_index[label] for label in predicted_labels]

    # Расчет метрик
    recall = recall_score(y_true_idx, y_pred_idx, average='macro')
    precision = precision_score(y_true_idx, y_pred_idx, average='macro')
    accuracy = accuracy_score(y_true_idx, y_pred_idx)
    f1 = f1_score(y_true_idx, y_pred_idx, average='macro')

    print(f"Recall (macro): {recall}")
    print(f"Precision (macro): {precision}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 (macro): {f1}")

if __name__ == "__main__":
    # Загрузка моделей
    category_model = load_category_model("category_model.pth")
    color_model = load_color_model("color_model.pth")

    # Создание submission.csv
    create_submission(category_model, color_model)

    # Проверка существования файла submission.csv
    if not os.path.exists(SUBMISSION_PATH):
        raise FileNotFoundError(f"Файл {SUBMISSION_PATH} не найден.")

    # Загрузка истинных меток из файла test_data.csv
    if not os.path.exists(TRUE_DATA_PATH):
        raise FileNotFoundError(f"Файл {TRUE_DATA_PATH} не найден.")
    true_data = pd.read_csv(TRUE_DATA_PATH)
    true_labels = true_data["color"].tolist()  # Предположим, что столбец называется "color"

    # Загрузка предсказанных меток
    predicted_labels = pd.read_csv(SUBMISSION_PATH)["predict_color"].tolist()

    # Проверка длины списков
    if len(true_labels) != len(predicted_labels):
        raise ValueError(f"Количество истинных меток ({len(true_labels)}) "
                         f"не совпадает с количеством предсказанных меток ({len(predicted_labels)}).")

    # Расчет метрик
    calculate_metrics(true_labels, predicted_labels)