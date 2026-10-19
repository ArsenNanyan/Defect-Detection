# Model Card — Surface Defect Classifier

## Overview
- **Model:** EfficientNet-B0 (ImageNet-pretrained), 6-way classification head, full fine-tune.
- **Task:** Single-label classification of a 200×200 steel-surface image into one of six
  defect types: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches.
- **Version:** 0.1.0 · **Framework:** PyTorch 2.6 (CUDA 12.4) → exported to ONNX.
- **Owner:** portfolio project (Industrial Surface Defect Inspector).

## Intended use
- **In scope:** demonstrating defect-type recognition on NEU-DET-like steel textures;
  education; the "what defect" half of the inspection demo.
- **Out of scope:** production quality gating without revalidation; non-steel surfaces;
  clean/defective screening (the training data has no genuine "normal" class); any safety-
  or cost-critical decision without human review.

## Training data
- **NEU-DET**, 1,800 images, 6 balanced classes (300 each), 200×200 grayscale→RGB.
- One immutable stratified **70/15/15** split (seed 42, near-duplicate-grouped, hash +
  group leakage asserts). Single label = authoritative class subfolder. 123 multi-class
  images kept and flagged. See [DATA_REPORT](../../artifacts/reports/DATA_REPORT.md).
- Preprocessing: resize 224², ImageNet normalization. Train aug: flips, ±10° rotation,
  mild brightness/contrast (defect-preserving).

## Evaluation
- **Frozen test split (278 images, used once):** macro-F1 **0.986**, balanced acc 0.985,
  OvR AUROC 0.9995. Per-class F1: crazing/patches/pitted/rolled-in 1.000, inclusion 0.961,
  scratches 0.953. Selected over ResNet-50 (equal accuracy, ~4× larger).
- Validation 1.000 was saturated; test 0.986 is the honest number.
- **Latency (ONNX, CPU, batch 1):** ~4.7 ms p50. **Size:** 16 MB ONNX / 4.0 M params.

## Limitations & failure modes
- 4/278 test errors — mainly non-inclusion images predicted as inclusion (precision 0.925)
  and ~9% of scratches missed (recall 0.911). See
  [failure gallery](../../artifacts/reports/figures/failures/classification_failures.png).
- Classification gives **no location** — deep features discard it (see
  [FEATURE_MAP_REPORT](../../artifacts/reports/FEATURE_MAP_REPORT.md)); use the detector for "where".
- No out-of-distribution / abstention mechanism; will confidently label unseen surfaces.

## Ethics, license, provenance
- NEU-DET license must be verified at the original source; a Kaggle mirror is not proof.
  Raw images are kept out of Git.
- Confidence is not calibrated; do not read softmax as a true probability for risk decisions.
