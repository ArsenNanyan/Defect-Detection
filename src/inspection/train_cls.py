"""STEP 2 — classification training pipeline for NEU-DET.

Trains custom_cnn / resnet18 / resnet50 / efficientnet_b0 on the frozen split.
AdamW + CrossEntropy + cosine LR, early stopping on validation macro-F1, best
checkpoint saved. Appends a row to the shared experiment table artifacts/reports/metrics.csv.

Examples
--------
    python -m inspection.train_cls --model resnet18 --transfer-mode full --epochs 20
    python -m inspection.train_cls --model custom_cnn --epochs 30
    python -m inspection.train_cls --config configs/classification.yaml

Run from repo root with `src` on PYTHONPATH, e.g.:
    set PYTHONPATH=src   (PowerShell: $env:PYTHONPATH="src")
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import NEUClassificationDataset
from .models import build_model
from .transforms import build_transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def make_optimizer(model, name, pretrained, transfer_mode, lr_scratch, lr_finetune, wd):
    """Lower LR for a pretrained backbone under full fine-tuning; single LR otherwise."""
    if not pretrained:
        return torch.optim.AdamW(model.parameters(), lr=lr_scratch, weight_decay=wd)
    if transfer_mode == "full":
        head_params, backbone_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (head_params if n.startswith(("fc", "classifier")) else backbone_params).append(p)
        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr_finetune * 0.1},
                {"params": head_params, "lr": lr_finetune},
            ],
            weight_decay=wd,
        )
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=lr_finetune, weight_decay=wd)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(y.tolist())
    return y_true, y_pred


def train(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])
    model_name = cfg["model"]
    transfer_mode = cfg["transfer_mode"]
    image_size = cfg["data"]["image_size"]

    model, pretrained = build_model(model_name, cfg["num_classes"], transfer_mode)
    model.to(device)
    train_tf, eval_tf = build_transforms(image_size, pretrained)

    raw_root = cfg["data"].get("raw_root", "data/raw/neu_det")
    splits_dir = cfg["data"]["splits_dir"]
    ds_train = NEUClassificationDataset("train", raw_root, splits_dir, train_tf)
    ds_val = NEUClassificationDataset("val", raw_root, splits_dir, eval_tf)

    nw = cfg["train"]["num_workers"]
    pin = device.type == "cuda"
    dl_train = DataLoader(ds_train, batch_size=cfg["train"]["batch_size"], shuffle=True,
                          num_workers=nw, pin_memory=pin, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=nw, pin_memory=pin)

    optim = make_optimizer(model, model_name, pretrained, transfer_mode,
                           cfg["train"]["lr_scratch"], cfg["train"]["lr_finetune"],
                           cfg["train"]["weight_decay"])
    epochs = cfg["train"]["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run = f"{model_name}_{transfer_mode}" if pretrained else model_name
    ckpt_dir = Path("artifacts/checkpoints")
    rep_dir = Path("artifacts/reports")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    history_path = rep_dir / f"history_{run}.csv"

    best_f1 = -1.0
    best_epoch = -1
    patience = cfg["train"]["early_stopping_patience"]
    bad_epochs = 0
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in tqdm(dl_train, desc=f"{run} e{epoch}/{epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += loss.item() * x.size(0)
        sched.step()
        train_loss = running / len(ds_train)

        y_true, y_pred = evaluate(model, dl_val, device)
        val_f1 = f1_score(y_true, y_pred, average="macro")
        val_bal = balanced_accuracy_score(y_true, y_pred)
        history.append(dict(epoch=epoch, train_loss=round(train_loss, 4),
                            val_macro_f1=round(val_f1, 4),
                            val_balanced_acc=round(val_bal, 4),
                            lr=optim.param_groups[0]["lr"]))
        print(f"[{run}] epoch {epoch}: loss={train_loss:.4f} "
              f"val_f1={val_f1:.4f} val_bal_acc={val_bal:.4f}")

        if val_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
            torch.save(
                dict(model_state=model.state_dict(), model_name=model_name,
                     transfer_mode=transfer_mode, num_classes=cfg["num_classes"],
                     image_size=image_size, pretrained=pretrained,
                     class_names=ds_train.class_names, val_macro_f1=best_f1),
                ckpt_dir / f"{run}_best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best f1={best_f1:.4f}"
                      f" @ epoch {best_epoch}).")
                break

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    train_minutes = round((time.time() - t0) / 60, 2)
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = round((ckpt_dir / f"{run}_best.pt").stat().st_size / 1e6, 2)
    result = dict(run=run, model=model_name, transfer_mode=transfer_mode,
                  best_val_macro_f1=round(best_f1, 4), best_epoch=best_epoch,
                  params_M=round(n_params / 1e6, 3), ckpt_size_MB=size_mb,
                  device=device.type, train_minutes=train_minutes,
                  checkpoint=str(ckpt_dir / f"{run}_best.pt"))

    _append_metrics(rep_dir / "metrics.csv", result)
    print(json.dumps(result, indent=2))
    return result


def _append_metrics(path: Path, row: dict) -> None:
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def load_config(path: str | None) -> dict:
    default = Path("configs/classification.yaml")
    with open(path or default, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None,
                    choices=["custom_cnn", "resnet18", "resnet50", "efficientnet_b0"])
    ap.add_argument("--transfer-mode", default=None,
                    choices=["frozen", "finetune_last", "full"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"] = args.model
    if args.transfer_mode:
        cfg["transfer_mode"] = args.transfer_mode
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.device:
        cfg["train"]["device"] = args.device
    train(cfg)


if __name__ == "__main__":
    main()
