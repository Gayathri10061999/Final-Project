import torch
from PIL import Image
from torchvision import transforms

from configs.config import *
from models.model_factory import get_model

device = torch.device(DEVICE)

classes = [
    "cat",
    "dog",
    "horse"
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

image = Image.open("sample.jpg").convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)

with torch.no_grad():

    output = model(image)

prediction = torch.argmax(output,1)

print("Prediction :", classes[prediction.item()])
