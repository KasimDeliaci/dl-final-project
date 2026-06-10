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

