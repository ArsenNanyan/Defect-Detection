"""U-Net for the supervised segmentation track (Step 6, Track B — NONSTANDARD).

Encoder: repeated (3x3 conv -> BN -> ReLU) x2 then 2x2 max-pool.
Decoder: transpose-conv upsample + skip concat + double conv.
Binary output: one logit per pixel. Train with BCEWithLogits + soft Dice.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32) -> None:
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.d1 = _DoubleConv(in_ch, c[0])
        self.d2 = _DoubleConv(c[0], c[1])
        self.d3 = _DoubleConv(c[1], c[2])
        self.bott = _DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.u3 = _DoubleConv(c[3], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.u2 = _DoubleConv(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.u1 = _DoubleConv(c[1], c[0])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        s1 = self.d1(x)
        s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2))
        b = self.bott(self.pool(s3))
        x = self.u3(torch.cat([self.up3(b), s3], 1))
        x = self.u2(torch.cat([self.up2(x), s2], 1))
        x = self.u1(torch.cat([self.up1(x), s1], 1))
        return self.head(x)  # (B,1,H,W) logits


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bw, self.dw = bce_weight, dice_weight

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        num = 2 * (prob * target).sum(dim=(1, 2, 3)) + 1.0
        den = (prob + target).sum(dim=(1, 2, 3)) + 1.0
        dice = 1 - (num / den).mean()
        return self.bw * bce + self.dw * dice


@torch.no_grad()
def dice_iou(logits, target, thr: float = 0.5) -> tuple[float, float]:
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = ((2 * inter + 1) / (union + 1)).mean().item()
    iou = ((inter + 1) / (union - inter + 1)).mean().item()
    return dice, iou
