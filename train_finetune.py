import os

import torch
from torch import nn
from torchvision import transforms

import config
from dataset_loader import load_datasets, create_dataloaders, set_seed
from models import create_finetune_model, get_trainable_parameters
from train_utils import train_one_epoch, evaluate_classification, save_checkpoint, save_training_summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    set_seed(config.SEED)
    torch.manual_seed(config.SEED)

    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset, val_dataset, test_dataset, class_names = load_datasets(
        config, train_transform, eval_transform
    )

    train_loader, val_loader, _ = create_dataloaders(
        config, train_dataset, val_dataset, test_dataset
    )

    print("Classes:", class_names)
    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    model = create_finetune_model(
        num_classes=len(class_names),
        freeze_backbone=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        get_trainable_parameters(model),
        lr=config.LEARNING_RATE
    )

    best_acc = 0.0
    history = []

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_classification(model, val_loader, criterion, device)

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_weighted": val_metrics["f1_weighted"],
        }
        history.append(epoch_info)

        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        print(f"train_loss = {train_loss:.4f}")
        print(f"val_loss   = {val_metrics['loss']:.4f}")
        print(f"val_acc    = {val_metrics['accuracy']:.4f}")
        print(f"val_f1     = {val_metrics['f1_weighted']:.4f}")

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            save_path = config.MODELS_DIR / config.FINETUNE_MODEL_NAME
            save_checkpoint(
                save_path,
                model,
                class_names,
                config.IMG_SIZE,
                model_type="finetune"
            )
            print("Model saved:", save_path)

    history_path = config.OUTPUTS_DIR / "finetune_training_history.json"
    save_training_summary(history_path, history)

    print("\nTraining finished")
    print("Best val_acc:", best_acc)
    print("History saved:", history_path)


if __name__ == "__main__":
    main()