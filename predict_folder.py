"""
Batch Image Prediction

Predict all images inside a folder.

Author : Gayathri
"""

import os
import torch
from PIL import Image
from torchvision import transforms

from configs.config import *
from models.model_factory import get_model


device = torch.device(DEVICE)

classes = [
    "class1",
    "class2",
    "class3"
]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

model = get_model(
    MODEL_NAME,
    len(classes)
)

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()


def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities,1)

    return classes[prediction.item()], confidence.item()


folder = "sample_images"

for file in os.listdir(folder):

    if file.lower().endswith((".jpg",".png",".jpeg")):

        path = os.path.join(folder,file)

        label, confidence = predict(path)

        print(f"{file:30} -> {label:20} ({confidence:.2%})")
