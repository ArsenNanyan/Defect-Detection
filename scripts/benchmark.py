"""STEP 8 (benchmark) — latency for classifier and detector, PyTorch vs ONNX.

Batch 1, 10 warmups + >=100 measured runs. Reports p50/p95 (ms) and FPS. Compares:
  * Classifier: CPU PyTorch, CPU ONNX Runtime, GPU PyTorch (if available).
  * Detector:   CPU PyTorch, CPU ONNX (both via Ultralytics end-to-end predict).
Classifier splits preprocess vs model inference; detector is end-to-end
(decode+preprocess+inference+NMS), which is the fair serving number.

Writes artifacts/reports/benchmark.json and appends serving rows to
artifacts/reports/serving_latency.csv.

Usage:
    python scripts/benchmark.py --runs 100 --n-images 50
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Stop Ultralytics from trying to pip-install onnxruntime-gpu when loading an .onnx model.
os.environ["YOLO_AUTOINSTALL"] = "False"

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inspection.models import build_model  # noqa: E402
from inspection.transforms import build_transforms  # noqa: E402

CKPT = Path("artifacts/checkpoints")
REP = Path("artifacts/reports")


def _stats(times_s: list[float]) -> dict:
    ms = sorted(t * 1000 for t in times_s)
    p50 = statistics.median(ms)
    p95 = ms[min(len(ms) - 1, int(round(0.95 * len(ms))) - 1)]
    return {"p50_ms": round(p50, 3), "p95_ms": round(p95, 3),
            "fps": round(1000 / p50, 1) if p50 else None}


def bench_classifier(runs: int, warmup: int = 10) -> dict:
    ckpt = torch.load(CKPT / "efficientnet_b0_full_best.pt", map_location="cpu",
                      weights_only=False)
    model, _ = build_model(ckpt["model_name"], ckpt["num_classes"], ckpt["transfer_mode"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    size = ckpt["image_size"]
    _, eval_tf = build_transforms(size, ckpt["pretrained"])

    img = Image.new("RGB", (200, 200))
    out = {}

    # preprocess timing
    tp = []
    for _ in range(runs):
        t = time.perf_counter()
        _ = eval_tf(img).unsqueeze(0)
        tp.append(time.perf_counter() - t)
    out["preprocess"] = _stats(tp)

    x = eval_tf(img).unsqueeze(0)

    # CPU PyTorch inference
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        tt = []
        for _ in range(runs):
            t = time.perf_counter()
            model(x)
            tt.append(time.perf_counter() - t)
    out["cpu_pytorch_infer"] = _stats(tt)

    # CPU ONNX Runtime
    sess = ort.InferenceSession(str(CKPT / "classifier.onnx"),
                                providers=["CPUExecutionProvider"])
    xn = x.numpy()
    for _ in range(warmup):
        sess.run(None, {"input": xn})
    to = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {"input": xn})
        to.append(time.perf_counter() - t)
    out["cpu_onnx_infer"] = _stats(to)

    # GPU PyTorch (optional)
    if torch.cuda.is_available():
        gm = model.cuda()
        xg = x.cuda()
        with torch.no_grad():
            for _ in range(warmup):
                gm(xg)
            torch.cuda.synchronize()
            tg = []
            for _ in range(runs):
                t = time.perf_counter()
                gm(xg)
                torch.cuda.synchronize()
                tg.append(time.perf_counter() - t)
        out["gpu_pytorch_infer"] = _stats(tg)
    return out


def bench_detector(runs: int, imgsz: int, warmup: int = 10) -> dict:
    from ultralytics import YOLO

    sample = next(Path("data/processed/neu_det/images/test").glob("*.jpg"))
    out = {}
    for tag, path in (("cpu_pytorch", "runs/detect/yolo26n_neu/weights/best.pt"),
                      ("cpu_onnx", str(CKPT / "detector.onnx"))):
        m = YOLO(path)
        for _ in range(warmup):
            m.predict(str(sample), imgsz=imgsz, device="cpu", verbose=False)
        tt = []
        for _ in range(runs):
            t = time.perf_counter()
            m.predict(str(sample), imgsz=imgsz, device="cpu", verbose=False)
            tt.append(time.perf_counter() - t)
        out[tag + "_e2e"] = _stats(tt)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--imgsz-det", type=int, default=512)
    ap.add_argument("--n-images", type=int, default=50)  # kept for API symmetry
    args = ap.parse_args()
    REP.mkdir(parents=True, exist_ok=True)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    report = {
        "device": device_name,
        "onnx_providers": ort.get_available_providers(),
        "runs": args.runs,
        "classifier": bench_classifier(args.runs),
        "detector": bench_detector(min(args.runs, 100), args.imgsz_det),
    }
    (REP / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
