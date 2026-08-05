import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from configs.config import *
from datasets.dataset import create_dataloaders
from models.model_factory import get_model

device = torch.device(DEVICE)

_, _, test_loader, classes = create_dataloaders(
    DATASET_PATH,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    image_size=IMAGE_SIZE
)

num_classes = len(classes)

model = get_model(MODEL_NAME, num_classes)
model.load_state_dict(
    torch.load(
        f"{CHECKPOINT_DIR}/best_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

true_labels = []
pred_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        true_labels.extend(labels.numpy())

        pred_labels.extend(predictions.cpu().numpy())
