"""STEP 8 — export trained models to ONNX and verify numerical parity.

Exports
-------
* Detector  : YOLO26n best.pt -> detector.onnx  (via Ultralytics export; the ONNX is
              loaded back through the Ultralytics runtime so decoding/NMS is handled).
* Classifier: EfficientNet-B0 best.pt -> classifier.onnx (torch.onnx, dynamic batch).

Parity (on N held-out test images)
----------------------------------
* Classifier: max abs softmax diff and top-1 agreement, PyTorch vs ONNX Runtime.
* Detector  : per-image detection count, label set, and max box-coordinate delta (px),
              PyTorch vs ONNX, at a fixed confidence threshold.

Writes artifacts/reports/onnx_parity.json and copies both .onnx into artifacts/checkpoints/.

Usage:
    python -m inspection.export_onnx --cls-ckpt artifacts/checkpoints/efficientnet_b0_full_best.pt \
        --det-ckpt runs/detect/yolo26n_neu/weights/best.pt --n 30
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from .data import NEUClassificationDataset
from .models import build_model
from .transforms import build_transforms

CKPT_DIR = Path("artifacts/checkpoints")
REP = Path("artifacts/reports")


# ---------------- classifier ----------------
def export_classifier(ckpt_path: str, imgsz: int | None = None) -> tuple[Path, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, _ = build_model(ckpt["model_name"], ckpt["num_classes"], ckpt["transfer_mode"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    size = imgsz or ckpt["image_size"]
    out = CKPT_DIR / "classifier.onnx"
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    meta = {"model_name": ckpt["model_name"], "image_size": size,
            "pretrained": ckpt["pretrained"], "class_names": ckpt["class_names"]}
    return out, meta


@torch.no_grad()
def parity_classifier(ckpt_path: str, onnx_path: Path, meta: dict, n: int) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, _ = build_model(ckpt["model_name"], ckpt["num_classes"], ckpt["transfer_mode"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    _, eval_tf = build_transforms(meta["image_size"], meta["pretrained"])
    ds = NEUClassificationDataset("test", transform=eval_tf)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    max_prob_diff = 0.0
    agree = 0
    n = min(n, len(ds))
    for i in range(n):
        x, _ = ds[i]
        xb = x.unsqueeze(0)
        pt = torch.softmax(model(xb), dim=1).numpy()[0]
        on = _softmax(sess.run(None, {"input": xb.numpy()})[0][0])
        max_prob_diff = max(max_prob_diff, float(np.max(np.abs(pt - on))))
        agree += int(pt.argmax() == on.argmax())
    return {"n": n, "top1_agreement": agree / n,
            "max_abs_softmax_diff": round(max_prob_diff, 6)}


# ---------------- detector ----------------
def export_detector(det_ckpt: str, imgsz: int) -> Path:
    from ultralytics import YOLO

    det = YOLO(det_ckpt)
    onnx_path = Path(det.export(format="onnx", imgsz=imgsz, dynamic=False))
    dst = CKPT_DIR / "detector.onnx"
    shutil.copy2(onnx_path, dst)
    return dst


def parity_detector(det_ckpt: str, onnx_path: Path, imgsz: int, n: int,
                    conf: float = 0.25) -> dict:
    from ultralytics import YOLO

    det_pt = YOLO(det_ckpt)
    det_on = YOLO(str(onnx_path))
    img_dir = Path("data/processed/neu_det/images/test")
    imgs = sorted(img_dir.glob("*.jpg"))[:n]

    max_box_delta = 0.0
    count_mismatch = 0
    label_mismatch = 0
    for p in imgs:
        rp = det_pt.predict(str(p), imgsz=imgsz, conf=conf, verbose=False)[0]
        ro = det_on.predict(str(p), imgsz=imgsz, conf=conf, verbose=False)[0]
        bp = rp.boxes.xyxy.cpu().numpy() if rp.boxes is not None else np.empty((0, 4))
        bo = ro.boxes.xyxy.cpu().numpy() if ro.boxes is not None else np.empty((0, 4))
        if len(bp) != len(bo):
            count_mismatch += 1
        else:
            if len(bp):
                # boxes come back in the same score order; compare element-wise
                max_box_delta = max(max_box_delta, float(np.max(np.abs(bp - bo))))
            lp = sorted(rp.boxes.cls.cpu().numpy().tolist()) if rp.boxes is not None else []
            lo = sorted(ro.boxes.cls.cpu().numpy().tolist()) if ro.boxes is not None else []
            if lp != lo:
                label_mismatch += 1
    return {"n": len(imgs), "conf": conf,
            "count_mismatches": count_mismatch,
            "label_mismatches": label_mismatch,
            "max_box_coord_delta_px": round(max_box_delta, 3)}


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls-ckpt", default="artifacts/checkpoints/efficientnet_b0_full_best.pt")
    ap.add_argument("--det-ckpt", default="runs/detect/yolo26n_neu/weights/best.pt")
    ap.add_argument("--imgsz-det", type=int, default=512)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)

    report = {}
    print("Exporting classifier...")
    cls_onnx, meta = export_classifier(args.cls_ckpt)
    report["classifier"] = {
        "onnx": str(cls_onnx),
        "size_MB": round(cls_onnx.stat().st_size / 1e6, 2),
        "parity": parity_classifier(args.cls_ckpt, cls_onnx, meta, args.n),
    }
    print("Exporting detector...")
    det_onnx = export_detector(args.det_ckpt, args.imgsz_det)
    report["detector"] = {
        "onnx": str(det_onnx),
        "size_MB": round(det_onnx.stat().st_size / 1e6, 2),
        "parity": parity_detector(args.det_ckpt, det_onnx, args.imgsz_det, args.n),
    }
    (REP / "onnx_parity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
