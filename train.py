import os
import copy
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.config import *
from datasets.dataset import create_dataloaders
from models.model_factory import get_model

# Device
device = torch.device(DEVICE)

# TensorBoard
writer = SummaryWriter("runs/image_classification")

# Dataset
train_loader, val_loader, test_loader, classes = create_dataloaders(
    DATASET_PATH,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    image_size=IMAGE_SIZE
)

NUM_CLASSES = len(classes)

# Model
model = get_model(MODEL_NAME, NUM_CLASSES)
model = model.to(device)

# Loss
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# Scheduler
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# Mixed Precision
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

best_accuracy = 0.0
best_model = copy.deepcopy(model.state_dict())

early_stop_counter = 0
patience = 5
