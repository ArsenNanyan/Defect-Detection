"""Track A — PatchCore-style normal-only anomaly localization (Step 6, benchmark-compatible).

Pipeline (standard PatchCore):
1. Extract mid-level ImageNet features (ResNet layer2 + layer3), locally average-pooled,
   layer3 upsampled to layer2's grid and concatenated -> one feature vector per spatial cell.
2. Build a memory bank of these patch features over all train/good images, then subsample
   it (coreset). We use a reproducible RANDOM subset for simplicity and document it; the
   original greedy coreset is a drop-in replacement.
3. Score a test image: each patch's distance to its nearest memory feature -> anomaly map;
   bilinearly upsample to image size and Gaussian-blur. Image score = max patch distance.

Metrics are threshold-free AUROC (image-level and pixel-level), so no threshold is tuned on
test. Fit uses train/good only; thresholds/calibration never look at test outcomes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class _Backbone(nn.Module):
    def __init__(self, name: str = "resnet18") -> None:
        super().__init__()
        if name == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif name == "wide_resnet50_2":
            net = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        else:
            raise ValueError(name)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        # local smoothing, then align layer3 to layer2 grid and concat on channels
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)  # (B, C, Hf, Wf)


class PatchCore:
    def __init__(self, backbone: str = "resnet18", coreset_fraction: float = 0.1,
                 n_neighbors: int = 1, blur_sigma: float = 4.0,
                 image_size: int = 256, seed: int = 42, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.backbone = _Backbone(backbone).to(self.device)
        self.coreset_fraction = coreset_fraction
        self.n_neighbors = n_neighbors
        self.blur_sigma = blur_sigma
        self.image_size = image_size
        self.seed = seed
        self.memory: torch.Tensor | None = None
        self.feat_hw: tuple[int, int] | None = None

    @torch.no_grad()
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x.to(self.device))            # (B,C,Hf,Wf)
        b, c, h, w = f.shape
        self.feat_hw = (h, w)
        return f.permute(0, 2, 3, 1).reshape(b * h * w, c)  # (B*Hf*Wf, C)

    @torch.no_grad()
    def fit(self, loader) -> None:
        feats = []
        for x in loader:
            feats.append(self._embed(x).cpu())
        bank = torch.cat(feats, dim=0)
        # random coreset subsample (reproducible)
        n_keep = max(1, int(round(bank.shape[0] * self.coreset_fraction)))
        g = torch.Generator().manual_seed(self.seed)
        idx = torch.randperm(bank.shape[0], generator=g)[:n_keep]
        self.memory = bank[idx].to(self.device)

    @torch.no_grad()
    def _patch_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-image anomaly map (B,1,H,W) at image resolution."""
        assert self.memory is not None, "call fit() first"
        b = x.shape[0]
        emb = self._embed(x)                             # (B*Hf*Wf, C)
        h, w = self.feat_hw
        # nearest-neighbor distance to memory (chunked to bound memory use)
        dmin = torch.empty(emb.shape[0], device=self.device)
        step = 4096
        for i in range(0, emb.shape[0], step):
            d = torch.cdist(emb[i:i + step], self.memory)  # (chunk, M)
            dmin[i:i + step] = d.topk(self.n_neighbors, largest=False).values.mean(1)
        amap = dmin.reshape(b, 1, h, w)
        amap = F.interpolate(amap, size=(self.image_size, self.image_size),
                             mode="bilinear", align_corners=False)
        return self._gaussian_blur(amap)

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        if self.blur_sigma <= 0:
            return x
        radius = int(round(3 * self.blur_sigma))
        ks = 2 * radius + 1
        coords = torch.arange(ks, device=x.device) - radius
        g1 = torch.exp(-(coords ** 2) / (2 * self.blur_sigma ** 2))
        g1 = (g1 / g1.sum()).to(x.dtype)
        kx = g1.view(1, 1, 1, ks)
        ky = g1.view(1, 1, ks, 1)
        x = F.conv2d(x, kx, padding=(0, radius))
        x = F.conv2d(x, ky, padding=(radius, 0))
        return x

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        """loader yields (image, mask, label). Returns image/pixel AUROC."""
        from sklearn.metrics import roc_auc_score

        img_scores, img_labels = [], []
        pix_scores, pix_labels = [], []
        for img, mask, label in loader:
            amap = self._patch_scores(img).cpu()          # (B,1,H,W)
            b = amap.shape[0]
            img_scores.extend(amap.view(b, -1).max(1).values.tolist())
            img_labels.extend([int(v) for v in label])
            # subsample pixels to keep pixel-AUROC tractable
            m = mask.view(b, -1).numpy()
            s = amap.view(b, -1).numpy()
            for bi in range(b):
                pix_scores.append(s[bi])
                pix_labels.append((m[bi] > 0.5).astype(np.int32))
        out = {"image_auroc": float(roc_auc_score(img_labels, img_scores))}
        pl = np.concatenate(pix_labels)
        ps = np.concatenate(pix_scores)
        if pl.max() > 0 and pl.min() == 0:
            # cap at ~2M pixels for speed
            if pl.shape[0] > 2_000_000:
                g = np.random.default_rng(self.seed)
                sel = g.choice(pl.shape[0], 2_000_000, replace=False)
                pl, ps = pl[sel], ps[sel]
            out["pixel_auroc"] = float(roc_auc_score(pl, ps))
        out["memory_bank_size"] = int(self.memory.shape[0])
        return out
