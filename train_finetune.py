import os
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def has_class_folders(path):
    path = Path(path)
    if not path.exists():
        return False
    subfolders = [p for p in path.iterdir() if p.is_dir()]
    return len(subfolders) > 0

if has_class_folders(TRAIN_DIR) and has_class_folders(VAL_DIR):
    print("Режим: отдельные папки train и val")

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform_val)

    class_names = train_dataset.classes

elif has_class_folders(TRAIN_DIR):
    print("Режим: есть только train, val создаётся автоматически")

    full_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
    class_names = full_dataset.classes

    indices = list(range(len(full_dataset)))
    random.shuffle(indices)

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_dataset, train_indices)

    full_dataset_val = datasets.ImageFolder(TRAIN_DIR, transform=transform_val)
    val_dataset = Subset(full_dataset_val, val_indices)

elif has_class_folders(DATA_DIR):
    print("Режим: классы лежат сразу в папке data, val создаётся автоматически")

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform_train)
    class_names = full_dataset.classes

    indices = list(range(len(full_dataset)))
    random.shuffle(indices)

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_dataset, train_indices)

    full_dataset_val = datasets.ImageFolder(DATA_DIR, transform=transform_val)
    val_dataset = Subset(full_dataset_val, val_indices)

else:
    raise FileNotFoundError(
        "Не найдены папки классов.\n"
        "Нужен один из вариантов:\n"
        "1) data/train/class_name/...\n"
        "2) data/class_name/..."
    )

num_classes = len(class_names)
print("Classes:", class_names)
print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

for param in model.parameters():
    param.requires_grad = False

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"  batch {batch_idx}/{len(train_loader)}")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total if total > 0 else 0.0
    avg_loss = train_loss / len(train_loader)

    print(f"loss = {avg_loss:.4f}")
    print(f"val_acc = {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        save_path = MODELS_DIR / "classifier_finetune.pth"

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "img_size": IMG_SIZE,
            "model_type": "resnet18"
        }, save_path)

        print("Model saved:", save_path)

print("\nTraining finished")
print("Best val_acc:", best_acc)