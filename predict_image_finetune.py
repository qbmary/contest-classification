from pathlib import Path

import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

IMAGE_PATH = Path(r"C:\Users\admin\Desktop\video_templates\classification\samples\cat.webp")
MODEL_PATH = MODELS_DIR / "classifier_finetune.pth"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Изображение не найдено: {IMAGE_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=device)

class_names = checkpoint["class_names"]
img_size = checkpoint["img_size"]

model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(class_names))

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
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