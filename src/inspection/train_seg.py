"""Track B U-Net training + evaluation (Step 6, NONSTANDARD supervised segmentation).

Trains on the self-constructed defect split, selects the binarization threshold on the
validation split (best Dice), and reports Dice/IoU on the held-out test split plus the
false-positive rate on clean (all-zero-mask) images.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .seg_models import BCEDiceLoss, UNet, dice_iou


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_unet(train_ds, val_ds, cfg: dict) -> dict:
    device = _device()
    model = UNet(base=cfg["base_channels"]).to(device)
    dl_tr = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    dl_va = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    crit = BCEDiceLoss(cfg["bce_weight"], cfg["dice_weight"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_dice, best_state = -1.0, None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for img, mask in dl_tr:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(model(img), mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        vd = _epoch_dice(model, dl_va, device)
        if vd > best_dice:
            best_dice = vd
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"[unet] epoch {epoch}/{cfg['epochs']} val_dice={vd:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    thr = _pick_threshold(model, dl_va, device)
    return {"model": model, "threshold": thr, "best_val_dice": round(best_dice, 4),
            "device": device.type}


@torch.no_grad()
def _epoch_dice(model, loader, device) -> float:
    model.eval()
    ds = []
    for img, mask in loader:
        d, _ = dice_iou(model(img.to(device)), mask.to(device))
        ds.append(d)
    return float(np.mean(ds)) if ds else 0.0


@torch.no_grad()
def _pick_threshold(model, loader, device) -> float:
    model.eval()
    best_t, best_d = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 17):
        ds = []
        for img, mask in loader:
            d, _ = dice_iou(model(img.to(device)), mask.to(device), thr=float(t))
            ds.append(d)
        md = float(np.mean(ds)) if ds else 0.0
        if md > best_d:
            best_d, best_t = md, float(t)
    return round(best_t, 3)


@torch.no_grad()
def eval_unet(model, threshold: float, test_ds, cfg: dict) -> dict:
    device = _device()
    model.eval()
    dl = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    dices, ious, fp_clean = [], [], []
    for img, mask in dl:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        pred = (torch.sigmoid(logits) > threshold).float()
        for bi in range(img.shape[0]):
            gt = mask[bi]
            pr = pred[bi]
            if gt.sum() > 0:  # defective image -> Dice/IoU
                inter = (pr * gt).sum()
                dices.append(((2 * inter + 1) / (pr.sum() + gt.sum() + 1)).item())
                ious.append(((inter + 1) / (pr.sum() + gt.sum() - inter + 1)).item())
            else:             # clean image -> false-positive pixel rate
                fp_clean.append((pr.sum() / pr.numel()).item())
    return {
        "threshold": threshold,
        "test_dice": round(float(np.mean(dices)), 4) if dices else None,
        "test_iou": round(float(np.mean(ious)), 4) if ious else None,
        "clean_fp_pixel_rate": round(float(np.mean(fp_clean)), 5) if fp_clean else None,
        "n_defective": len(dices), "n_clean": len(fp_clean),
    }
