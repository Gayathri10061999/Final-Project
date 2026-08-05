import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 4

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

EPOCHS = 20

NUM_CLASSES = None

MODEL_NAME = "resnet18"

DATASET_PATH = "dataset"

CHECKPOINT_DIR = "checkpoints"

OUTPUT_DIR = "outputs"

MEAN = [0.485, 0.456, 0.406]

STD = [0.229, 0.224, 0.225]
