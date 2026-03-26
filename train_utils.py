import json

import torch
from sklearn.metrics import accuracy_score, f1_score


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


def evaluate_classification(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())

    avg_loss = running_loss / max(len(loader), 1)
    accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0
    f1 = f1_score(y_true, y_pred, average="weighted") if y_true else 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_weighted": f1,
    }


def save_checkpoint(save_path, model, class_names, img_size, model_type):
    payload = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "img_size": img_size,
        "model_type": model_type,
    }
    torch.save(payload, save_path)


def save_training_summary(save_path, history):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)