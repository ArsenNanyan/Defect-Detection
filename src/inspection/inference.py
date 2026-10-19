"""Shared inference wrappers around the exported ONNX models (Steps 9-10).

* OnnxClassifier — EfficientNet-B0 classifier.onnx via ONNX Runtime (CPU). Uses the SAME
  eval transform as training so preprocessing matches the checkpoint.
* OnnxDetector — YOLO26n detector.onnx loaded through the Ultralytics runtime, which
  handles letterbox preprocessing, decoding, and NMS (raw ONNX tensors are not boxes).

Both are process-wide singletons in the API (loaded once at startup).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # no pip calls when loading .onnx

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from PIL import Image  # noqa: E402

from . import CLASS_NAMES  # noqa: E402

CKPT_DIR = Path(os.getenv("INSPECTOR_CKPT_DIR", "artifacts/checkpoints"))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class OnnxClassifier:
    def __init__(self, onnx_path: Path | None = None, image_size: int = 224,
                 class_names: list[str] | None = None) -> None:
        self.onnx_path = Path(onnx_path or CKPT_DIR / "classifier.onnx")
        self.image_size = image_size
        self.class_names = class_names or CLASS_NAMES
        self.model_id = "efficientnet_b0-onnx"
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"{self.onnx_path} missing. Run export_onnx first.")
        self.sess = ort.InferenceSession(str(self.onnx_path),
                                         providers=["CPUExecutionProvider"])
        self._build_transform()

    def _build_transform(self) -> None:
        # lazy import keeps torch out of import path unless a classifier is used
        from .transforms import build_transforms

        _, self._eval_tf = build_transforms(self.image_size, pretrained=True)

    def predict(self, image: Image.Image) -> dict:
        t0 = time.perf_counter()
        x = self._eval_tf(image).unsqueeze(0).numpy()
        logits = self.sess.run(None, {"input": x})[0][0]
        probs = _softmax(logits)
        idx = int(probs.argmax())
        dt = (time.perf_counter() - t0) * 1000
        return {
            "model": self.model_id,
            "task": "classification",
            "label": self.class_names[idx],
            "class_id": idx,
            "confidence": float(probs[idx]),
            "probabilities": {n: float(p) for n, p in zip(self.class_names, probs,
                                                          strict=False)},
            "inference_ms": round(dt, 2),
        }


class OnnxDetector:
    def __init__(self, onnx_path: Path | None = None, imgsz: int = 512) -> None:
        self.onnx_path = Path(onnx_path or CKPT_DIR / "detector.onnx")
        self.imgsz = imgsz
        self.model_id = "yolo26n-onnx"
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"{self.onnx_path} missing. Run export_onnx first.")
        from ultralytics import YOLO

        self.model = YOLO(str(self.onnx_path))
        self.class_names = CLASS_NAMES

    def predict(self, image: Image.Image, conf: float = 0.25) -> dict:
        t0 = time.perf_counter()
        res = self.model.predict(image, imgsz=self.imgsz, conf=conf, device="cpu",
                                 verbose=False)[0]
        dets = []
        if res.boxes is not None:
            for b in res.boxes:
                cid = int(b.cls.item())
                dets.append({
                    "class_id": cid,
                    "label": res.names[cid],
                    "confidence": float(b.conf.item()),
                    "xyxy": [round(float(v), 2) for v in b.xyxy[0].tolist()],
                })
        dt = (time.perf_counter() - t0) * 1000
        return {
            "model": self.model_id,
            "task": "detection",
            "conf_threshold": conf,
            "detections": dets,
            "inference_ms": round(dt, 2),
        }


def load_experiment_metrics() -> dict:
    """Small helper for the UI: pull headline metrics from the committed reports."""
    rep = Path("artifacts/reports")
    out = {}
    cls = rep / "classification_comparison.json"
    det = rep / "detection_metrics.json"
    onnx = rep / "benchmark.json"
    if cls.exists():
        out["classification"] = json.loads(cls.read_text())[:5]
    if det.exists():
        out["detection"] = json.loads(det.read_text()).get("results", {})
    if onnx.exists():
        out["latency"] = json.loads(onnx.read_text())
    return out
