from pathlib import Path

BASE_DIR = Path(r"C:\Users\admin\Desktop\classification")

DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_WORKERS = 0

VAL_SPLIT = 0.2
SEED = 42

MODEL_NAME = "classifier_scratch.pth"