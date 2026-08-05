import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

def build_model(num_classes):
    model = resnet34(weights=ResNet34_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features

    model.fc = nn.Linear(in_features, num_classes)

    return model
