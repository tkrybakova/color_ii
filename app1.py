import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from rembg import remove
import json

# Константы
CATEGORY_NAMES = ["стол", "стул", "сумка", "детская одежда"]
COLOR_NAMES = ["бежевый", "белый", "бирюзовый", "бордовый", "голубой", "жёлтый",
               "зелёный", "золотой", "коричневый", "красный", "оранжевый",
               "розовый", "серебристый", "серый", "синий", "фиолетовый",
               "чёрный", "разноцветный"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CATEGORY_CLASSES = 4
NUM_COLOR_CLASSES = 18

# Загрузка модели для категорий
def load_category_model(model_path):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, NUM_CATEGORY_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Загрузка модели для цветов
def load_color_model(model_path):
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
def predict_color(image, color_model):
    try:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = color_model(image)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        # Создаем словарь вероятностей для всех цветов
        color_probabilities = {COLOR_NAMES[i]: float(probabilities[i]) for i in range(NUM_COLOR_CLASSES)}
        
        # Сортируем цвета по вероятности
        sorted_colors = sorted(color_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Наиболее подходящий цвет
        predicted_color = sorted_colors[0][0]
        
        # Топ-5 цветов
        top_5_colors = sorted_colors[:5]
        
        return predicted_color, top_5_colors
    except Exception as e:
        print(f"Ошибка предсказания цвета: {e}")
        return "разноцветный", []

# Основная функция для обработки одного файла
def process_image(file_path, category_model, color_model):
    # Загрузка изображения
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {file_path}")

    # Удаление фона
    img_no_bg = remove(img)

    # Предсказание категории
    predicted_category = predict_category(img_no_bg, category_model)

    # Предсказание цвета
    predicted_color, top_5_colors = predict_color(img_no_bg, color_model)

    # Возвращаем результаты
    return {
        "category": predicted_category,
        "predicted_color": predicted_color,
        "top_5_colors": top_5_colors
    }

if __name__ == "__main__":
    # Загрузка моделей
    category_model = load_category_model("category_model.pth")
    color_model = load_color_model("color_model.pth")

    # Запрос пути к файлу у пользователя
    file_path = input("Введите путь к изображению: ").strip()

    try:
        # Обработка изображения
        result = process_image(file_path, category_model, color_model)

        # Вывод результатов
        print("\nРезультаты анализа:")
        print("Категория:", result["category"])
        print("Наиболее подходящий цвет:", result["predicted_color"])
        print("Топ-5 цветов:")
        for color, prob in result["top_5_colors"]:
            print(f"  {color}: {prob:.2f}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")