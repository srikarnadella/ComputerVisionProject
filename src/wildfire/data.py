from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _build_transform(image_size: int, train: bool):
    resize = transforms.Resize((image_size, image_size))
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose(
            [
                resize,
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            resize,
            transforms.ToTensor(),
            normalize,
        ]
    )


def create_dataloader(config: dict, split: str) -> DataLoader:
    split_name = config["data"][f"{split}_split"]
    root = Path(config["data"]["root_dir"]) / split_name
    dataset = datasets.ImageFolder(
        root=str(root),
        transform=_build_transform(config["data"]["image_size"], train=(split == "train")),
    )
    return DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=config["data"]["num_workers"],
    )

