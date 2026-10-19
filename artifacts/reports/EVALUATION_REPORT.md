# Unified Evaluation & Model Selection (Step 7)

Final cross-task evaluation on the **frozen NEU-DET test split** (278 images), used once.
Classification and detection share the same immutable split (seed 42); segmentation
(MVTec AD) is a **separate dataset and protocol** and is deliberately **not** ranked here.

> Task metrics are not interchangeable: a classification macro-F1 and a detection mAP
> measure different objectives (recognition vs localization). The table reports each task
> on its own terms.

## Cross-task summary (test split)

| Model | Task | Key test metric | Params (M) | Size (MB, ONNX) | CPU ONNX p50 | Notes |
|---|---|---|---|---|---|---|
| **EfficientNet-B0 (full)** | classification | **macro-F1 0.986**, AUROC 0.9995 | 4.0 | 16.0 | 4.7 ms | selected classifier |
| Custom CNN (scratch) | classification | macro-F1 0.953, AUROC 0.9997 | 0.09 | — | — | from-scratch baseline |
| **YOLO26n** | detection | **mAP@0.5 0.782**, mAP@0.5:0.95 0.449 | 2.5 | 9.8 | 17.3 ms (e2e) | selected detector |
| *(PatchCore / U-Net)* | segmentation | *pending — MVTec AD* | — | — | — | separate protocol |

Latency: batch 1, CPUExecutionProvider, i7-13620H (see `benchmark.json`).

## Classification — selected model, per-class (test)

| Class | F1 | Precision | Recall |
|---|---|---|---|
| crazing | 1.000 | 1.000 | 1.000 |
| patches | 1.000 | 1.000 | 1.000 |
| pitted_surface | 1.000 | 1.000 | 1.000 |
| rolled-in_scale | 1.000 | 1.000 | 1.000 |
| inclusion | 0.961 | 0.925 | 1.000 |
| scratches | 0.953 | 1.000 | 0.911 |

- **Val 1.000 → test 0.986**: the perfect validation score was slightly optimistic (the
  270-image val set was saturated), as flagged during the sweep. Test is the honest number.
- Residual errors: a few non-inclusion images predicted as *inclusion* (precision 0.925),
  and ~9% of *scratches* missed (recall 0.911).

## Detection — selected model, per-class AP@0.5 (test)

patches 0.936 · scratches 0.935 · inclusion 0.825 · pitted_surface 0.777 ·
rolled-in_scale 0.686 · **crazing 0.536** (weakest).

## The key cross-task finding: *what* vs *where*

**crazing is the easiest class to classify (F1 1.000) but the hardest to detect (AP 0.54).**
A crazing image is globally a distinctive crack-network texture, so a classifier that sees
the whole 200×200 image recognizes it trivially — but the cracks have no crisp boundary, so
an IoU-based detector struggles to place a box. The mirror image is why the project ships
both models: a **classifier answers "what defect is present"**, a **detector answers "where
it is"**, and neither substitutes for the other. (Localization is also simply harder overall:
detection mAP ~0.78 vs classification F1 ~0.99.)

## Model-selection logic
- **Classifier:** EfficientNet-B0 (full fine-tune) — tied for best on validation at the
  smallest size (4 M params / 16 MB ONNX), confirmed best-in-class on test (0.986). Chosen
  over ResNet-50 (equal accuracy, ~4× larger) for deployment efficiency.
- **Detector:** YOLO26n — only detector trained; 2.5 M params, 9.8 MB ONNX, 17 ms CPU e2e.
- **Serving:** CPU ONNX Runtime (8× / 3.2× faster than CPU PyTorch; no GPU needed in prod).

## Industrial cost note
False negatives (missed defects) dominate inspection cost. Classifier scratches recall is
0.911 and detector recall ~0.73; a deployment would lower thresholds to trade precision for
recall, and the demo exposes the detector confidence as a live control.

## Still open
- **Segmentation (Step 6):** MVTec AD, Track A (PatchCore normal-only) + Track B
  (supervised U-Net, explicitly non-standard). Requires a manual MVTec download.
- Portfolio polish: model cards, >=20 qualitative failure cases, 60-90 s demo video.
