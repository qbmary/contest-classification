import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from config import *


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Что оцениваем:
# "classifier_finetune.pth" или "classifier_scratch.pth"
MODEL_FILENAME = "classifier_scratch.pth"

EVAL_DIR = OUTPUTS_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, img_size: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        feature_size = img_size // 8

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * feature_size * feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)

    class_names = checkpoint["class_names"]
    img_size = checkpoint["img_size"]

    if model_path.name == "classifier_finetune.pth":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(class_names))
    elif model_path.name == "classifier_scratch.pth":
        model = SimpleCNN(num_classes=len(class_names), img_size=img_size)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_path.name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names, img_size


def plot_confusion_matrix(cm, class_names, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest")
    plt.colorbar(im)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = cm.max() / 2 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_binary_roc(y_true, y_score, class_names, save_path: Path):
    # Для 2 классов
    # считаем ROC для положительного класса с индексом 1
    fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
    auc_value = roc_auc_score(y_true, y_score[:, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(f"ROC Curve ({class_names[1]} vs rest)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_multiclass_roc(y_true, y_score, class_names, save_path: Path):
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        auc_value = roc_auc_score(y_true_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, label=f"{class_name} (AUC = {auc_value:.4f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    print("Device:", DEVICE)

    model_path = MODELS_DIR / MODEL_FILENAME
    model, class_names, img_size = load_model(model_path)

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Папка test не найдена: {TEST_DIR}\n"
            f"Создай структуру data/test/имя_класса/*.jpg"
        )

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    if test_dataset.classes != class_names:
        print("ВНИМАНИЕ:")
        print("Классы в модели:", class_names)
        print("Классы в test:", test_dataset.classes)
        raise ValueError("Порядок классов в test не совпадает с классами модели.")

    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
            y_score.extend(probabilities.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)

    print(f"Model: {MODEL_FILENAME}")
    print(f"Classes: {class_names}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 (weighted): {f1:.4f}")

    cm_path = EVAL_DIR / f"{model_path.stem}_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)

    roc_path = EVAL_DIR / f"{model_path.stem}_roc_curve.png"
    if len(class_names) == 2:
        plot_binary_roc(y_true, y_score, class_names, roc_path)
        roc_auc = roc_auc_score(y_true, y_score[:, 1])
    else:
        plot_multiclass_roc(y_true, y_score, class_names, roc_path)
        y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        roc_auc = roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")

    metrics = {
        "model": MODEL_FILENAME,
        "classes": class_names,
        "test_samples": int(len(test_dataset)),
        "accuracy": float(accuracy),
        "f1_weighted": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_image": str(cm_path),
        "roc_curve_image": str(roc_path),
    }

    metrics_path = EVAL_DIR / f"{model_path.stem}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print("Saved:")
    print(" ", cm_path)
    print(" ", roc_path)
    print(" ", metrics_path)


if __name__ == "__main__":
    main()