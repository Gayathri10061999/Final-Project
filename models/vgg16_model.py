import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

def build_model(num_classes):
    model = vgg16(weights=VGG16_Weights.DEFAULT)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[6].in_features

    model.classifier[6] = nn.Linear(in_features, num_classes)

    return model
