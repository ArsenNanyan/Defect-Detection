"""Image transforms for NEU-DET classification.

Design notes
------------
* Source images are 200x200 grayscale steel textures; we convert to 3-channel RGB and
  resize to 224x224 (the size torchvision ImageNet backbones expect).
* Pretrained backbones use **ImageNet mean/std**; the scratch CNN uses plain [0.5] norm.
* Train-only augmentation is deliberately mild and defect-preserving:
  horizontal + vertical flips (steel texture has no canonical orientation) and small
  rotations/brightness. We do NOT use aggressive crops, blur, or cutout that could
  erase thin scratches or tiny pits.
"""
from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PLAIN_MEAN = (0.5, 0.5, 0.5)
PLAIN_STD = (0.5, 0.5, 0.5)


def _norm(pretrained: bool) -> transforms.Normalize:
    if pretrained:
        return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    return transforms.Normalize(PLAIN_MEAN, PLAIN_STD)


def build_transforms(image_size: int = 224, pretrained: bool = True):
    """Return (train_tf, eval_tf)."""
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            _norm(pretrained),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            _norm(pretrained),
        ]
    )
    return train_tf, eval_tf
