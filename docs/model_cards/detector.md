# Model Card — Surface Defect Detector

## Overview
- **Model:** YOLO26n (Ultralytics, COCO-pretrained) fine-tuned for 6 steel-defect classes.
- **Task:** Object detection — localize and classify defects with bounding boxes.
- **Version:** 0.1.0 · **Framework:** Ultralytics 8.4 / PyTorch 2.6 → ONNX.
- **License note:** Ultralytics YOLO is **AGPL-3.0 / Enterprise** — review before any
  closed-source commercial use.

## Intended use
- **In scope:** locating defects on steel-surface images; the "where" half of the demo;
  the primary ONNX deployment path.
- **Out of scope:** production acceptance without revalidation; non-steel domains; guaranteed
  recall of safety-critical defects. Source images are 200×200, so boxes are coarse.

## Training data
- NEU-DET Pascal VOC boxes → YOLO format (4,189 boxes, 0 rejected), **same immutable split**
  as the classifier. All boxes and all six classes preserved. imgsz 512 (200² upscaled),
  batch 8, 80 epochs, seed 42. See [DETECTION_REPORT](../../artifacts/reports/DETECTION_REPORT.md).

## Evaluation (frozen test split, 278 images, once)
- **mAP@0.5 0.782**, mAP@0.5:0.95 0.449, precision 0.760, recall 0.734 (val ≈ test → no leakage).
- Per-class AP@0.5: patches 0.936, scratches 0.935, inclusion 0.825, pitted_surface 0.777,
  rolled-in_scale 0.686, **crazing 0.536** (weakest — diffuse cracks, ambiguous box extent).
- **Latency (ONNX, CPU, end-to-end, batch 1):** ~17 ms p50. **Size:** 9.8 MB / 2.5 M params.

## Limitations & failure modes
- 121/278 test images have a false positive or a missed box (recall ~0.73). See
  [failure gallery](../../artifacts/reports/figures/failures/detection_failures.png).
- **crazing** is hard to localize though easy to classify (F1 1.0) — a *what-vs-where* gap.
- Confidence threshold trades precision for recall; the demo exposes it as a live control.

## Ethics & deployment
- In inspection, **false negatives (missed defects) usually cost more than false alarms** —
  a deployment would lower the threshold to raise recall, accepting more false positives.
- NEU-DET license to be verified at source; images kept out of Git; weights derived from
  COCO-pretrained Ultralytics weights (AGPL-3.0).
