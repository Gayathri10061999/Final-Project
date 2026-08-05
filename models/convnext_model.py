import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

def build_model(num_classes):
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[2].in_features

    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model
