"""STEP 5 — train YOLO26n detector on NEU-DET and evaluate on the frozen test split.

Reads configs/detection.yaml. Fine-tunes yolo26n.pt (COCO-pretrained) on the six steel
defect classes using the same immutable split as classification. After training, the best
checkpoint is validated on BOTH the val and the (frozen) test split; metrics are written to
artifacts/reports/detection_metrics.json and appended to artifacts/reports/metrics.csv.

Tuned for an RTX 3050 6 GB laptop GPU (imgsz 512, batch 8). Source images are 200x200, so
larger imgsz only upscales — 320 vs 512 is a worthwhile speed/accuracy ablation later.

Usage:
    python scripts/train_det.py --config configs/detection.yaml
    python scripts/train_det.py --epochs 80 --imgsz 512 --batch 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/detection.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="Resume training. Bare --resume uses the run's last.pt; "
                         "or pass an explicit path to a last.pt.")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.imgsz is not None:
        cfg["imgsz"] = args.imgsz
    if args.batch is not None:
        cfg["batch"] = args.batch

    device = args.device or (0 if torch.cuda.is_available() else "cpu")
    print(f"Training {cfg['model']} on device={device} "
          f"(imgsz={cfg['imgsz']}, batch={cfg['batch']}, epochs={cfg['epochs']})")

    # absolute project dir avoids Ultralytics path-doubling with relative paths
    project = str(Path(cfg["project"]).resolve())

    if args.resume is not None:
        last = (Path(project) / cfg["name"] / "weights" / "last.pt"
                if args.resume == "auto" else Path(args.resume))
        if not last.exists():
            raise FileNotFoundError(f"Cannot resume: {last} not found.")
        print(f"Resuming from {last}")
        model = YOLO(str(last))
        # resume=True restores optimizer state and epoch count from the checkpoint;
        # all other training args are read back from the run's saved args.yaml.
        model.train(resume=True)
    else:
        model = YOLO(cfg["model"])
        model.train(
            data=cfg["data_yaml"],
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            seed=cfg["seed"],
            project=project,
            name=cfg["name"],
            patience=cfg.get("patience", 20),
            device=device,
            exist_ok=True,
            plots=True,
        )

    # use the trainer's own resolved paths (robust across Ultralytics versions)
    run_dir = Path(model.trainer.save_dir)
    best = Path(model.trainer.best)
    best_model = YOLO(str(best))

    results = {}
    for split in ("val", "test"):
        m = best_model.val(data=cfg["data_yaml"], split=split, imgsz=cfg["imgsz"],
                           device=device, plots=True, project=project,
                           name=f"{cfg['name']}_{split}", exist_ok=True)
        results[split] = {
            "map50_95": round(float(m.box.map), 4),
            "map50": round(float(m.box.map50), 4),
            "precision": round(float(m.box.mp), 4),
            "recall": round(float(m.box.mr), 4),
            "per_class_ap50": {m.names[i]: round(float(v), 4)
                               for i, v in zip(m.box.ap_class_index, m.box.ap50,
                                               strict=False)},
        }
        print(f"[{split}] mAP50-95={results[split]['map50_95']} "
              f"mAP50={results[split]['map50']}")

    rep = Path("artifacts/reports")
    rep.mkdir(parents=True, exist_ok=True)
    out = {
        "model": cfg["model"],
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch"],
        "epochs": cfg["epochs"],
        "best_checkpoint": str(best),
        "device": str(device),
        "results": results,
    }
    (rep / "detection_metrics.json").write_text(json.dumps(out, indent=2),
                                                encoding="utf-8")
    # NOTE: detection metrics are task-specific (mAP, not F1) and are intentionally kept
    # separate from the classification metrics.csv. The unified cross-task comparison is
    # assembled in the Step 7 evaluation, which reads this JSON and classification_comparison.csv.

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
