"""Image preprocessing for ImageNet-pretrained transformer backbones."""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_feature_transform(image_size: int = 224) -> transforms.Compose:
    """Build deterministic preprocessing for cached frozen features."""

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_finetune_train_transform(image_size: int = 224) -> transforms.Compose:
    """Build conservative train-time augmentation for partial fine-tuning."""

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
