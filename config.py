from pathlib import Path

BASE_DIR = Path(r"C:\Users\admin\Desktop\classification")

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ===== Формат датасета =====
# Варианты:
# "folder_separate" -> data/train/class_x, data/val/class_x, data/test/class_x
# "folder_single"   -> data/class_x, всё делится автоматически
# "csv_separate"    -> отдельные train.csv / val.csv / test.csv
# "csv_single"      -> один annotations.csv, всё делится автоматически
DATASET_FORMAT = "folder_single"

# ===== Пути для формата folder_separate =====
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# ===== Пути для формата folder_single =====
# Если в одной папке лежат изображения, то после = должно быть DATA_DIR / "images"
FULL_DATA_DIR = DATA_DIR

# ===== Пути для формата csv =====
# CSV должен содержать минимум 2 колонки:
# image_path,label
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"
ANNOTATIONS_CSV = DATA_DIR / "annotations.csv"

# Корневая папка для картинок в csv-режиме.
# Если в csv пути относительные, они будут считаться от этой папки.
CSV_IMAGES_ROOT = DATA_DIR

IMAGE_COLUMN = "image_path"
LABEL_COLUMN = "label"

# Можно задать классы вручную, например:
# CLASS_NAMES = ["cat", "dog"]
# Если None, классы определяются автоматически
CLASS_NAMES = None

# ===== Разбиение =====
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1
SEED = 42

# ===== Обучение =====
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_WORKERS = 0

# ===== Модели =====
SCRATCH_MODEL_NAME = "classifier_scratch.pth"
FINETUNE_MODEL_NAME = "classifier_finetune.pth"