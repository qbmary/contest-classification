import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import config
from dataset_loader import load_datasets
from models import load_model_from_checkpoint


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Что проверять:
# config.SCRATCH_MODEL_NAME
# config.FINETUNE_MODEL_NAME
MODEL_FILENAME = config.FINETUNE_MODEL_NAME


def plot_confusion_matrix(cm, class_names, save_path):
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


def plot_binary_roc(y_true, y_score, class_names, save_path):
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


def plot_multiclass_roc(y_true, y_score, class_names, save_path):
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


def build_test_loader_from_config(img_size):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    _, _, test_dataset, _ = load_datasets(config, train_transform, eval_transform)

    if test_dataset is None:
        raise ValueError(
            "Тестовый набор не найден. "
            "Либо добавь TEST_DIR / TEST_CSV, либо используй авторазбиение в single-режиме."
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    return test_dataset, test_loader


def main():
    print("Device:", DEVICE)

    eval_dir = config.OUTPUTS_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    model_path = config.MODELS_DIR / MODEL_FILENAME
    model, class_names, img_size, model_type = load_model_from_checkpoint(model_path, DEVICE)

    test_dataset, test_loader = build_test_loader_from_config(img_size)

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
    print(f"Model type: {model_type}")
    print(f"Classes: {class_names}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 (weighted): {f1:.4f}")

    cm_path = eval_dir / f"{model_path.stem}_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)

    roc_path = eval_dir / f"{model_path.stem}_roc_curve.png"
    if len(class_names) == 2:
        plot_binary_roc(y_true, y_score, class_names, roc_path)
        roc_auc = roc_auc_score(y_true, y_score[:, 1])
    else:
        plot_multiclass_roc(y_true, y_score, class_names, roc_path)
        y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        roc_auc = roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")

    metrics = {
        "model": MODEL_FILENAME,
        "model_type": model_type,
        "classes": class_names,
        "test_samples": int(len(test_dataset)),
        "accuracy": float(accuracy),
        "f1_weighted": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_image": str(cm_path),
        "roc_curve_image": str(roc_path),
    }

    metrics_path = eval_dir / f"{model_path.stem}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print("Saved:")
    print(" ", cm_path)
    print(" ", roc_path)
    print(" ", metrics_path)


if __name__ == "__main__":
    main()