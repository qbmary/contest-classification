from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from PIL import Image

from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

IMAGE_PATH = Path(r"C:\Users\admin\Desktop\video_templates\classification\samples\cat.webp")
MODEL_PATH = MODELS_DIR / MODEL_NAME

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, img_size):
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
            nn.MaxPool2d(2)
        )

        feature_size = img_size // 8

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * feature_size * feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Изображение не найдено: {IMAGE_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=device)

class_names = checkpoint["class_names"]
img_size = checkpoint["img_size"]

model = SimpleCNN(num_classes=len(class_names), img_size=img_size).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probabilities, dim=1)

predicted_class = class_names[predicted_idx.item()]
confidence_value = confidence.item()

print("Image:", IMAGE_PATH.name)
print("Predicted class:", predicted_class)
print("Confidence:", round(confidence_value, 4))