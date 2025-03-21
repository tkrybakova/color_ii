import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.amp import GradScaler, autocast
import multiprocessing

# Константы
DATA_DIR = "data/categories"  # Папка с категориями (стол, стул, сумка, детская одежда)
MODEL_PATH = "category_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4  # Стол, стул, сумка, детская одежда

def main():
    # Подготовка данных
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=multiprocessing.cpu_count()  # Используем все доступные ядра
    )

    # Загрузка модели
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # Оптимизатор и функция потерь
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = GradScaler(enabled=DEVICE.type == 'cuda')  # Включаем GradScaler только для CUDA

    # Обучение модели
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            with autocast(device_type=DEVICE.type, enabled=DEVICE.type == 'cuda'):  # Включаем autocast только для CUDA
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {running_loss / len(train_loader)}")

    # Сохранение модели
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Модель категорий сохранена в {MODEL_PATH}")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Необходимо для Windows
    main()