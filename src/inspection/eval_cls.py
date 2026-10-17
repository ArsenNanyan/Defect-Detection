"""STEP 2 (eval) — evaluate a classification checkpoint on a split.

Produces macro-F1, balanced accuracy, per-class report, confusion-matrix figure, and
one-vs-rest AUROC (when probabilities permit). The test split must be used ONCE, at the
very end — not for tuning or checkpoint selection.

Examples
--------
    python -m inspection.eval_cls --checkpoint artifacts/checkpoints/resnet18_full_best.pt --split val
    python -m inspection.eval_cls --checkpoint artifacts/checkpoints/resnet18_full_best.pt --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader  # noqa: E402

from .data import NEUClassificationDataset  # noqa: E402
from .models import build_model  # noqa: E402
from .transforms import build_transforms  # noqa: E402


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model, _ = build_model(ckpt["model_name"], ckpt["num_classes"], ckpt["transfer_mode"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def run_eval(checkpoint: str, split: str, raw_root: str, splits_dir: str,
             out_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(checkpoint, device)
    _, eval_tf = build_transforms(ckpt["image_size"], ckpt["pretrained"])
    ds = NEUClassificationDataset(split, raw_root, splits_dir, eval_tf)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                    pin_memory=device.type == "cuda")
    names = ckpt["class_names"]

    y_true, y_pred, probs = [], [], []
    for x, y in dl:
        p = F.softmax(model(x.to(device)), dim=1).cpu().numpy()
        probs.append(p)
        y_pred.extend(p.argmax(1).tolist())
        y_true.extend(y.tolist())
    probs = np.concatenate(probs, axis=0)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=names,
                                   output_dict=True, zero_division=0)
    try:
        auroc_ovr = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        auroc_ovr = None

    run = Path(checkpoint).stem.replace("_best", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    _plot_cm(cm, names, out_dir / f"confusion_{run}_{split}.png",
             title=f"{run} — {split} (macro-F1={macro_f1:.3f})")

    result = dict(checkpoint=checkpoint, split=split, model=ckpt["model_name"],
                  transfer_mode=ckpt["transfer_mode"], n_samples=len(ds),
                  macro_f1=round(macro_f1, 4), balanced_accuracy=round(bal_acc, 4),
                  auroc_ovr_macro=(round(auroc_ovr, 4) if auroc_ovr else None),
                  per_class={k: report[k] for k in names}, device=device.type)
    (out_dir / f"eval_{run}_{split}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_class"}, indent=2))
    return result


def _plot_cm(cm, names, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--raw-root", default="data/raw/neu_det")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out", type=Path, default=Path("artifacts/reports"))
    args = ap.parse_args()
    run_eval(args.checkpoint, args.split, args.raw_root, args.splits_dir, args.out)


if __name__ == "__main__":
    main()
