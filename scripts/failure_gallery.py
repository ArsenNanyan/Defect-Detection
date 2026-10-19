"""Portfolio polish — qualitative failure-case gallery on the NEU-DET test split.

* Classification: every misclassified test image (EfficientNet-B0), true vs predicted.
* Detection: test images with a false positive (pred box matching no GT) or a miss
  (GT box matching no pred), drawn with GT (green) and pred (red) boxes.

Outputs:
    artifacts/reports/figures/failures/classification_failures.png
    artifacts/reports/figures/failures/detection_failures.png
    artifacts/reports/failure_cases.json   (counts + file lists)

Usage (PYTHONPATH=src):
    python scripts/failure_gallery.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inspection import CLASS_NAMES  # noqa: E402
from inspection.data import NEUClassificationDataset  # noqa: E402
from inspection.models import build_model  # noqa: E402
from inspection.transforms import build_transforms  # noqa: E402

CKPT = Path("artifacts/checkpoints")
FIG = Path("artifacts/reports/figures/failures")
RAW = Path("data/raw/neu_det")
PROC = Path("data/processed/neu_det")


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@torch.no_grad()
def classification_failures() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(CKPT / "efficientnet_b0_full_best.pt", map_location=device,
                    weights_only=False)
    model, _ = build_model(ck["model_name"], ck["num_classes"], ck["transfer_mode"])
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    _, eval_tf = build_transforms(ck["image_size"], ck["pretrained"])
    ds = NEUClassificationDataset("test", transform=eval_tf)

    fails = []
    for i in range(len(ds)):
        x, y = ds[i]
        pred = int(model(x.unsqueeze(0).to(device)).argmax(1).item())
        if pred != y:
            fails.append((ds.df.iloc[i]["image_path"], CLASS_NAMES[y], CLASS_NAMES[pred]))

    _grid_cls(fails, FIG / "classification_failures.png")
    return {"n_test": len(ds), "n_failures": len(fails),
            "cases": [{"image": p, "true": t, "pred": pr} for p, t, pr in fails]}


def _grid_cls(fails, path: Path) -> None:
    if not fails:
        return
    n = min(12, len(fails))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    axes = np.atleast_1d(axes).ravel()
    for ax, (rel, t, pr) in zip(axes, fails[:n], strict=False):
        ax.imshow(Image.open(RAW / rel).convert("RGB"))
        ax.set_title(f"true: {t}\npred: {pr}", fontsize=7, color="crimson")
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Classification failures ({len(fails)} total on test)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def detection_failures(iou_thr: float = 0.3, conf: float = 0.25) -> dict:
    from ultralytics import YOLO

    model = YOLO("runs/detect/yolo26n_neu/weights/best.pt")
    img_dir = PROC / "images" / "test"
    lbl_dir = PROC / "labels" / "test"
    imgs = sorted(img_dir.glob("*.jpg"))

    cases = []
    for p in imgs:
        W = H = 200
        gts = []
        lf = lbl_dir / f"{p.stem}.txt"
        if lf.exists():
            for line in lf.read_text().splitlines():
                c, xc, yc, w, h = map(float, line.split())
                gts.append((int(c), [(xc - w / 2) * W, (yc - h / 2) * H,
                                     (xc + w / 2) * W, (yc + h / 2) * H]))
        r = model.predict(str(p), imgsz=512, conf=conf, device="cpu", verbose=False)[0]
        preds = []
        if r.boxes is not None:
            for b in r.boxes:
                preds.append((int(b.cls.item()), [float(v) for v in b.xyxy[0].tolist()],
                              float(b.conf.item())))
        matched_gt = set()
        fp = 0
        for _pc, pb, _conf in preds:
            best = max((_iou(pb, gb) for _gc, gb in gts), default=0.0)
            if best < iou_thr:
                fp += 1
        for gi, (_gc, gb) in enumerate(gts):
            best = max((_iou(pb, gb) for _pc, pb, _c in preds), default=0.0)
            if best >= iou_thr:
                matched_gt.add(gi)
        miss = len(gts) - len(matched_gt)
        if fp or miss:
            cases.append({"image": str(p.relative_to(PROC)), "fp": fp, "miss": miss,
                          "_gts": gts, "_preds": preds})

    cases.sort(key=lambda c: c["fp"] + c["miss"], reverse=True)
    _grid_det(cases[:12], FIG / "detection_failures.png")
    clean = [{k: v for k, v in c.items() if not k.startswith("_")} for c in cases]
    return {"n_test": len(imgs), "n_images_with_failure": len(cases), "cases": clean[:40]}


def _grid_det(cases, path: Path) -> None:
    if not cases:
        return
    n = len(cases)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.8))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cases, strict=False):
        ax.imshow(Image.open(PROC / c["image"]).convert("RGB"))
        for _gc, gb in c["_gts"]:
            ax.add_patch(mpatches.Rectangle((gb[0], gb[1]), gb[2] - gb[0], gb[3] - gb[1],
                         fill=False, edgecolor="lime", lw=1.5))
        for _pc, pb, cf in c["_preds"]:
            ax.add_patch(mpatches.Rectangle((pb[0], pb[1]), pb[2] - pb[0], pb[3] - pb[1],
                         fill=False, edgecolor="red", lw=1.2, linestyle="--"))
        ax.set_title(f'FP={c["fp"]} miss={c["miss"]}', fontsize=7)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Detection failures — GT (green) vs pred (red dashed)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    out = {"classification": classification_failures(),
           "detection": detection_failures()}
    Path("artifacts/reports/failure_cases.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"classification failures: {out['classification']['n_failures']}"
          f"/{out['classification']['n_test']}")
    print(f"detection images with failures: "
          f"{out['detection']['n_images_with_failure']}/{out['detection']['n_test']}")


if __name__ == "__main__":
    main()
