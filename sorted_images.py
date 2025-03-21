import os
import shutil

# Константы
IMAGES_DIR = "TEST_DATA"  # Папка с исходными изображениями
TEXT_FILE = "train_data.csv"  # Текстовый файл с данными
OUTPUT_DIR = "categories"  # Папка для сортированных изображений

def sort_images_by_category():
    # Создаем выходную директорию, если её нет
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Читаем данные из текстового файла
    try:
        with open(TEXT_FILE, "r", encoding="utf-8") as file:
            lines = [line.strip().split(",") for line in file.readlines()]
            file_data = {line[0]: (line[1], line[2]) for line in lines if len(line) == 3}
    except Exception as e:
        print(f"Ошибка чтения текстового файла: {e}")
        return

    # Получаем список всех файлов в папке с изображениями
    image_files = [f for f in os.listdir(IMAGES_DIR) if os.path.isfile(os.path.join(IMAGES_DIR, f))]

    # Сортируем файлы
    for img_file in image_files:
        img_name, img_ext = os.path.splitext(img_file)
        if img_name in file_data:  # Если название файла есть в текстовом файле
            # Получаем категорию и цвет из текстового файла
            category, color = file_data[img_name]

            # Создаем папку для категории
            category_folder = os.path.join(OUTPUT_DIR, category)
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)

            # Перемещаем файл в соответствующую папку
            src_path = os.path.join(IMAGES_DIR, img_file)
            dst_path = os.path.join(category_folder, img_file)
            try:
                shutil.move(src_path, dst_path)
                print(f"Файл {img_file} перемещен в папку {category} (Цвет: {color})")
            except Exception as e:
                print(f"Ошибка перемещения файла {img_file}: {e}")
        else:
            print(f"Файл {img_file} не найден в текстовом файле")

if __name__ == "__main__":
    sort_images_by_category()