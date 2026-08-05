"""
Training Utility Functions

Author : Gayathri
Project : Multi Dataset Image Classification
"""

import os
import random
import numpy as np
import torch

from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau
)


# --------------------------------------------------
# Device
# --------------------------------------------------

def get_device():

    if torch.cuda.is_available():
        print("Using GPU :", torch.cuda.get_device_name(0))
        return torch.device("cuda")

    print("Using CPU")
    return torch.device("cpu")


# --------------------------------------------------
# Random Seed
# --------------------------------------------------

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# --------------------------------------------------
# Count Parameters
# --------------------------------------------------

def count_parameters(model):

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("-"*50)

    print(f"Total Parameters     : {total:,}")

    print(f"Trainable Parameters : {trainable:,}")

    print("-"*50)

    return total, trainable


# --------------------------------------------------
# Save Checkpoint
# --------------------------------------------------

def save_checkpoint(
        model,
        optimizer,
        epoch,
        loss,
        path):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "loss": loss

    }

    torch.save(checkpoint, path)

    print(f"Checkpoint Saved : {path}")


# --------------------------------------------------
# Load Checkpoint
# --------------------------------------------------

def load_checkpoint(
        path,
        model,
        optimizer=None,
        device="cpu"):

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    epoch = checkpoint["epoch"]

    loss = checkpoint["loss"]

    print(f"Checkpoint Loaded : {path}")

    return model, optimizer, epoch, loss


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

def get_optimizer(
        model,
        optimizer_name="adamw",
        lr=1e-4,
        weight_decay=1e-4):

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":

        return Adam(
            model.parameters(),
            lr=lr
        )

    elif optimizer_name == "sgd":

        return SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9
        )

    else:

        return AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )


# --------------------------------------------------
# Scheduler
# --------------------------------------------------

def get_scheduler(
        optimizer,
        scheduler_name="cosine",
        epochs=20):

    scheduler_name = scheduler_name.lower()

    if scheduler_name == "step":

        return StepLR(
            optimizer,
            step_size=5,
            gamma=0.1
        )

    elif scheduler_name == "reduce":

        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=3,
            factor=0.5
        )

    else:

        return CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )


# --------------------------------------------------
# Average Meter
# --------------------------------------------------

class AverageMeter:

    def __init__(self):

        self.reset()

    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0

    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count


# --------------------------------------------------
# Accuracy
# --------------------------------------------------

def accuracy(outputs, labels):

    _, predicted = torch.max(outputs, 1)

    correct = predicted.eq(labels).sum().item()

    return 100 * correct / labels.size(0)


# --------------------------------------------------
# Gradient Clipping
# --------------------------------------------------

def clip_gradients(model, max_norm=1.0):

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm
    )


# --------------------------------------------------
# Create Folder
# --------------------------------------------------

def create_directory(path):

    if not os.path.exists(path):

        os.makedirs(path)

        print(f"Created Folder : {path}")


# --------------------------------------------------
# Save Training History
# --------------------------------------------------

def save_history(history, filepath):

    np.save(filepath, history)

    print("Training history saved.")
