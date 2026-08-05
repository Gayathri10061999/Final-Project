from models.alexnet_model import build_model as alexnet
from models.vgg16_model import build_model as vgg16
from models.resnet18_model import build_model as resnet18
from models.resnet34_model import build_model as resnet34
from models.convnext_model import build_model as convnext

MODEL_FACTORY = {
    "alexnet": alexnet,
    "vgg16": vgg16,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "convnext": convnext,
}


def get_model(model_name, num_classes):
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Unsupported model: {model_name}")

    return MODEL_FACTORY[model_name](num_classes)
