import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 4

EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_CLASSES = None

RANDOM_SEED = 42

MODEL_DIR = PROJECT_ROOT / "saved_models"

LOG_DIR = PROJECT_ROOT / "logs"

RESULT_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)

LOG_DIR.mkdir(exist_ok=True)

RESULT_DIR.mkdir(exist_ok=True)

IMAGENET_MEAN = [0.485,0.456,0.406]

IMAGENET_STD = [0.229,0.224,0.225]
