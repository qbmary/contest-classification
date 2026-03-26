from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

import config
from models import load_model_from_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ===== НАСТРОЙКИ =====
IMAGE_PATH = Path(r"C:\Users\admin\Desktop\video_templates\classification\samples\cat.webp")

# какую модель использовать:
# config.SCRATCH_MODEL_NAME
# config.FINETUNE_MODEL_NAME
MODEL_PATH = config.MODELS_DIR / config.FINETUNE_MODEL_NAME
# =====================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Изображение не найдено: {IMAGE_PATH}")

# ===== ЗАГРУЗКА МОДЕЛИ =====
model, class_names, img_size, model_type = load_model_from_checkpoint(
    MODEL_PATH,
    device
)

print("Model type:", model_type)
print("Classes:", class_names)

# ===== ПРЕОБРАЗОВАНИЕ =====
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# ===== ПРЕДСКАЗАНИЕ =====
image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probabilities, dim=1)

predicted_class = class_names[predicted_idx.item()]
confidence_value = confidence.item()

print("\nImage:", IMAGE_PATH.name)
print("Predicted class:", predicted_class)
print("Confidence:", round(confidence_value, 4))