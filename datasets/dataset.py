from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from datasets.transforms import (
    get_train_transforms,
    get_val_transforms
)


def create_dataloaders(
    dataset_path,
    batch_size=32,
    num_workers=4,
    image_size=224
):
    train_dataset = ImageFolder(
        root=f"{dataset_path}/train",
        transform=get_train_transforms(image_size)
    )

    val_dataset = ImageFolder(
        root=f"{dataset_path}/val",
        transform=get_val_transforms(image_size)
    )

    test_dataset = ImageFolder(
        root=f"{dataset_path}/test",
        transform=get_val_transforms(image_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.classes
    )
