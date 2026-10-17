"""STEP 4 — convert NEU-DET Pascal VOC XML boxes to YOLO format.

Uses the SAME frozen split as classification (data/splits/{train,val,test}.csv) so
detection and classification are comparable on an identical held-out test set.

For each box: xc=((xmin+xmax)/2)/W, yc=((ymin+ymax)/2)/H, w=(xmax-xmin)/W,
h=(ymax-ymin)/H, then clip to [0,1]. Zero-area boxes are rejected (documented rule).
All valid boxes and all six classes are preserved.

Output:
    data/processed/neu_det/images/{train,val,test}/*.jpg   (copied)
    data/processed/neu_det/labels/{train,val,test}/*.txt   (class xc yc w h)
    data/processed/neu_det/neu_det.yaml
    artifacts/reports/figures/yolo_check_<split>.png        (box alignment sanity)

Usage:
    python scripts/voc_to_yolo.py --raw data/raw/neu_det --splits data/splits \
        --out data/processed/neu_det
"""
from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface",
               "rolled-in_scale", "scratches"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}


def convert_boxes(xml_path: Path) -> tuple[list[str], int]:
    """Return (yolo_lines, n_rejected) for one annotation file."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    W = int(float(size.findtext("width")))
    H = int(float(size.findtext("height")))
    lines, rejected = [], 0
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in CLASS_TO_ID:
            rejected += 1
            continue
        bb = obj.find("bndbox")
        xmin = float(bb.findtext("xmin"))
        ymin = float(bb.findtext("ymin"))
        xmax = float(bb.findtext("xmax"))
        ymax = float(bb.findtext("ymax"))
        if xmax <= xmin or ymax <= ymin:  # zero/negative area -> reject
            rejected += 1
            continue
        xc = ((xmin + xmax) / 2) / W
        yc = ((ymin + ymax) / 2) / H
        w = (xmax - xmin) / W
        h = (ymax - ymin) / H
        xc, yc, w, h = (min(max(v, 0.0), 1.0) for v in (xc, yc, w, h))
        lines.append(f"{CLASS_TO_ID[name]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines, rejected


def convert(raw: Path, splits: Path, out: Path) -> dict:
    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)

    stats = {"total_boxes": 0, "rejected_boxes": 0, "images": 0,
             "empty_label_files": 0, "per_split": {}}

    for split in ("train", "val", "test"):
        df = pd.read_csv(splits / f"{split}.csv")
        n_box = 0
        for _, row in df.iterrows():
            img_src = raw / row["image_path"]
            stem = img_src.stem
            img_dst = out / "images" / split / f"{stem}.jpg"
            if img_src.suffix.lower() == ".jpg":
                shutil.copy2(img_src, img_dst)
            else:
                Image.open(img_src).convert("RGB").save(img_dst, "JPEG", quality=95)

            ann = raw / row["annotation_path"]
            lines, rejected = convert_boxes(ann) if row["annotation_path"] else ([], 0)
            (out / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            if not lines:
                stats["empty_label_files"] += 1
            n_box += len(lines)
            stats["rejected_boxes"] += rejected
            stats["images"] += 1
        stats["per_split"][split] = {"images": len(df), "boxes": n_box}
        stats["total_boxes"] += n_box

    yaml_path = out / "neu_det.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {i: c for i, c in enumerate(CLASS_NAMES)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    stats["yaml"] = str(yaml_path)
    return stats


def verify_figure(out: Path, fig_dir: Path, split: str = "val", n: int = 6) -> None:
    """Draw converted YOLO boxes back onto images to confirm alignment."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    imgs = sorted(p for p in img_dir.glob("*.jpg"))[:n]
    cols = 3
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.ravel()
    for ax, img_path in zip(axes, imgs, strict=False):
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        ax.imshow(im)
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                cid, xc, yc, w, h = line.split()
                xc, yc, w, h = float(xc) * W, float(yc) * H, float(w) * W, float(h) * H
                ax.add_patch(mpatches.Rectangle((xc - w / 2, yc - h / 2), w, h,
                             fill=False, edgecolor="lime", lw=1.5))
                ax.text(xc - w / 2, yc - h / 2 - 2, CLASS_NAMES[int(cid)],
                        color="lime", fontsize=6)
        ax.set_title(img_path.name, fontsize=7)
        ax.axis("off")
    for ax in axes[len(imgs):]:
        ax.axis("off")
    fig.suptitle(f"YOLO conversion check — {split}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"yolo_check_{split}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/neu_det"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/neu_det"))
    args = ap.parse_args()
    stats = convert(args.raw, args.splits, args.out)
    verify_figure(args.out, Path("artifacts/reports/figures"), "val")
    verify_figure(args.out, Path("artifacts/reports/figures"), "train")
    import json

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
