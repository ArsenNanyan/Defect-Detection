"""STEP 6 orchestrator — run both segmentation tracks on one MVTec AD category.

Track A (PatchCore, benchmark-compatible): fit normal-only, report image/pixel AUROC.
Track B (U-Net, NONSTANDARD supervised): self-split defect images, report Dice/IoU + clean FP.

Writes artifacts/reports/segmentation_metrics.json, SEGMENTATION_REPORT.md, and a
qualitative overlay figure. Requires MVTec AD category downloaded under configs' mvtec_root.

Usage (PYTHONPATH=src):
    python scripts/run_seg.py --config configs/segmentation.yaml
    python scripts/run_seg.py --category carpet --skip-unet   # PatchCore only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inspection.mvtec_data import (  # noqa: E402
    MVTecAnomalyDataset,
    MVTecSegDataset,
    MVTecTrainGood,
    list_test_items,
    list_train_good,
    supervised_unet_split,
)
from inspection.patchcore import PatchCore  # noqa: E402
from inspection.train_seg import eval_unet, train_unet  # noqa: E402

REP = Path("artifacts/reports")
FIG = REP / "figures" / "segmentation"


def track_a(cfg: dict) -> dict:
    root, cat, size = cfg["mvtec_root"], cfg["category"], cfg["image_size"]
    pc_cfg = cfg["patchcore"]
    pc = PatchCore(backbone=pc_cfg["backbone"], coreset_fraction=pc_cfg["coreset_fraction"],
                   n_neighbors=pc_cfg["n_neighbors"], blur_sigma=pc_cfg["blur_sigma"],
                   image_size=size, seed=cfg["seed"])
    train_good = MVTecTrainGood(list_train_good(root, cat), size)
    pc.fit(DataLoader(train_good, batch_size=8, shuffle=False, num_workers=0))
    test_ds = MVTecAnomalyDataset(list_test_items(root, cat), size)
    metrics = pc.evaluate(DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0))
    metrics["backbone"] = pc_cfg["backbone"]
    metrics["coreset_fraction"] = pc_cfg["coreset_fraction"]
    return metrics


def track_b(cfg: dict) -> dict:
    root, cat, size = cfg["mvtec_root"], cfg["category"], cfg["image_size"]
    u = cfg["unet"]
    splits = supervised_unet_split(root, cat, cfg["seed"], u["val_frac"], u["test_frac"],
                                   u["include_good_frac"])
    tr = MVTecSegDataset(splits["train"], size, augment=True)
    va = MVTecSegDataset(splits["val"], size)
    te = MVTecSegDataset(splits["test"], size)
    trained = train_unet(tr, va, u)
    result = eval_unet(trained["model"], trained["threshold"], te, u)
    result["best_val_dice"] = trained["best_val_dice"]
    result["split_counts"] = {k: len(v) for k, v in splits.items()}
    result["protocol"] = ("NONSTANDARD supervised split of MVTec test defects; NOT the "
                          "official unsupervised benchmark. Whole-image disjoint splits.")
    _overlay_figure(trained["model"], trained["threshold"], te, size)
    return result


@torch.no_grad()
def _overlay_figure(model, thr, test_ds, size, n: int = 4) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    picks = [i for i in range(len(test_ds))][:n]
    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    for r, i in enumerate(picks):
        img, mask = test_ds[i]
        pred = (torch.sigmoid(model(img.unsqueeze(0).to(device))) > thr).float()[0, 0].cpu()
        disp = (img.permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
        axes[r, 0].imshow(disp); axes[r, 0].set_title("image", fontsize=8)
        axes[r, 1].imshow(mask[0], cmap="gray"); axes[r, 1].set_title("GT mask", fontsize=8)
        axes[r, 2].imshow(disp); axes[r, 2].imshow(pred, cmap="Reds", alpha=0.5)
        axes[r, 2].set_title("pred overlay", fontsize=8)
        for c in range(3):
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "unet_overlays.png", dpi=110)
    plt.close(fig)


def _write_report(cfg: dict, a: dict | None, b: dict | None) -> None:
    lines = [
        "# MVTec AD Segmentation / Anomaly Localization (Step 6)",
        "",
        f"Category: **{cfg['category']}** · image size {cfg['image_size']} · seed {cfg['seed']}.",
        "MVTec AD is **CC BY-NC-SA 4.0 (non-commercial)**; images are not redistributed here.",
        "",
        "> The two NEU-DET tasks and this MVTec task are **separate datasets and protocols** "
        "and are never ranked against each other.",
        "",
    ]
    if a:
        lines += [
            "## Track A — PatchCore (benchmark-compatible, normal-only)",
            f"Backbone {a['backbone']}, coreset {a['coreset_fraction']}, "
            f"memory bank {a.get('memory_bank_size','?')} patches. Fit on train/good only; "
            "AUROC is threshold-free so nothing is tuned on test.",
            "",
            f"- **Image AUROC: {a['image_auroc']:.4f}**",
            f"- **Pixel AUROC: {a.get('pixel_auroc', 'n/a')}**",
            "",
        ]
    if b:
        lines += [
            "## Track B — U-Net (NONSTANDARD supervised)",
            f"{b['protocol']}",
            f"Split (whole-image disjoint): {b['split_counts']}.",
            "",
            f"- Threshold (picked on val): {b['threshold']}",
            f"- **Test Dice: {b['test_dice']}**, **Test IoU: {b['test_iou']}** "
            f"(over {b['n_defective']} defective images)",
            f"- Clean false-positive pixel rate: {b['clean_fp_pixel_rate']} "
            f"(over {b['n_clean']} clean images)",
            f"- Best val Dice: {b['best_val_dice']}",
            "",
            "Qualitative overlays: `artifacts/reports/figures/segmentation/unet_overlays.png`.",
            "",
            "> Track B trains on MVTec *test* defects we re-split ourselves, so it is an "
            "educational supervised baseline, **not** an official MVTec result. Track A is the "
            "benchmark-compatible number.",
        ]
    (REP / "SEGMENTATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/segmentation.yaml")
    ap.add_argument("--category", default=None)
    ap.add_argument("--skip-unet", action="store_true")
    ap.add_argument("--skip-patchcore", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.category:
        cfg["category"] = args.category
    if args.epochs is not None:
        cfg["unet"]["epochs"] = args.epochs
    REP.mkdir(parents=True, exist_ok=True)

    out = {"category": cfg["category"], "image_size": cfg["image_size"]}
    if not args.skip_patchcore:
        print("=== Track A: PatchCore ===")
        out["track_a_patchcore"] = track_a(cfg)
        print(json.dumps(out["track_a_patchcore"], indent=2))
    if not args.skip_unet:
        print("=== Track B: U-Net ===")
        out["track_b_unet"] = track_b(cfg)
        print(json.dumps(out["track_b_unet"], indent=2))

    (REP / "segmentation_metrics.json").write_text(json.dumps(out, indent=2),
                                                   encoding="utf-8")
    _write_report(cfg, out.get("track_a_patchcore"), out.get("track_b_unet"))
    print(f"\nWrote {REP/'segmentation_metrics.json'} and SEGMENTATION_REPORT.md")


if __name__ == "__main__":
    main()
