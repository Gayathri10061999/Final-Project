from torchvision import transforms

from config import IMAGE_SIZE
from config import IMAGENET_MEAN
from config import IMAGENET_STD


def get_train_transform():

    return transforms.Compose([

        transforms.Resize((256,256)),

        transforms.RandomResizedCrop(IMAGE_SIZE),

        transforms.RandomHorizontalFlip(),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),

        transforms.RandomRotation(10),

        transforms.ToTensor(),

        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD
        )

    ])


def get_valid_transform():

    return transforms.Compose([

        transforms.Resize((256,256)),

        transforms.CenterCrop(IMAGE_SIZE),

        transforms.ToTensor(),

        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD
        )

    ])
