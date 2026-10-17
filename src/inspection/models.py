"""Classification model factory: custom CNN + torchvision transfer-learning backbones.

Models
------
* custom_cnn      : 3x (Conv-BN-ReLU-MaxPool) -> AdaptiveAvgPool -> Linear(6)
* resnet18/50     : torchvision, ImageNet weights, 6-way head
* efficientnet_b0 : torchvision, ImageNet weights, 6-way head

Transfer modes (ignored by custom_cnn, which always trains fully)
-----------------------------------------------------------------
* frozen         : freeze backbone, train only the new head (feature extraction)
* finetune_last  : train the final backbone block + head
* full           : train end-to-end (optionally with a smaller backbone LR)

`build_model` returns (model, pretrained_flag). pretrained_flag drives normalization
choice in transforms.build_transforms.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


def custom_cnn(num_classes: int = 6) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes),
    )


def _set_backbone_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable


def _apply_transfer_mode(net: nn.Module, mode: str, last_block: nn.Module, head: nn.Module):
    if mode == "frozen":
        _set_backbone_trainable(net, False)
        _set_backbone_trainable(head, True)
    elif mode == "finetune_last":
        _set_backbone_trainable(net, False)
        _set_backbone_trainable(last_block, True)
        _set_backbone_trainable(head, True)
    elif mode == "full":
        _set_backbone_trainable(net, True)
    else:
        raise ValueError(f"Unknown transfer_mode: {mode}")


def build_model(name: str, num_classes: int = 6, transfer_mode: str = "full"):
    """Return (model, pretrained_flag)."""
    name = name.lower()
    if name == "custom_cnn":
        return custom_cnn(num_classes), False

    if name == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        _apply_transfer_mode(net, transfer_mode, net.layer4, net.fc)
        return net, True

    if name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        _apply_transfer_mode(net, transfer_mode, net.layer4, net.fc)
        return net, True

    if name == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_f = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_f, num_classes)
        _apply_transfer_mode(net, transfer_mode, net.features[-1], net.classifier)
        return net, True

    raise ValueError(f"Unknown model: {name}")
