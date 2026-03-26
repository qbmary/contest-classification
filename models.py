import torch
from torch import nn
from torchvision import models


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


def create_scratch_model(num_classes: int, img_size: int):
    return SimpleCNN(num_classes=num_classes, img_size=img_size)


def create_finetune_model(num_classes: int, freeze_backbone: bool = True):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def get_trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def load_model_from_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    model_type = checkpoint["model_type"]
    class_names = checkpoint["class_names"]
    img_size = checkpoint.get("img_size", 224)

    if model_type == "scratch":
        model = create_scratch_model(len(class_names), img_size)

    elif model_type in ["finetune", "resnet18"]:
        model = create_finetune_model(len(class_names), freeze_backbone=False)

    else:
        raise ValueError(f"Неизвестный model_type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, img_size, model_type