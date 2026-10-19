# Model Card — Anomaly Localization / Segmentation (MVTec AD)

Two tracks on a **separate dataset and protocol** from NEU-DET; never ranked against it.

## Track A — PatchCore (recommended, benchmark-compatible)
- **Model:** PatchCore-lite — ResNet-18 (ImageNet) layer2+3 features, random 10% coreset
  memory bank, kNN patch distance, Gaussian-smoothed anomaly map.
- **Task:** unsupervised, **normal-only** anomaly detection + pixel localization.
- **Training data:** MVTec AD `grid`, train/good only (264 normal images). Threshold-free
  AUROC → nothing tuned on test.
- **Evaluation (grid test, 78 images):** image AUROC **0.927**, pixel AUROC **0.962**.
- **Intended use:** defect-free-only screening where defects are unknown/rare — the realistic
  industrial setting. **Out of scope:** categories unlike the fitted normal set.
- **Limitation:** below published PatchCore (~0.98, WideResNet-50 + greedy coreset); backbone
  and coreset are one-line upgrades.

## Track B — U-Net (educational, **NONSTANDARD**)
- **Model:** U-Net (BCE + soft Dice), one logit/pixel.
- **Task:** *supervised* binary defect segmentation.
- **Training data:** a **self-constructed** whole-image-disjoint split of MVTec `grid`
  **test** defects (45 train / 11 val / 11 test) + some good images. This is **not** the
  official unsupervised benchmark and is labeled NONSTANDARD everywhere.
- **Evaluation:** test Dice **0.421**, IoU 0.279, clean false-positive pixel rate **0.0**,
  best val Dice 0.633 (threshold 0.7 picked on val).
- **Limitation:** modest Dice — tiny labeled set + severe class imbalance (thin grid defects).
  The takeaway: **normal-only PatchCore localizes better than a label-starved U-Net.**

## Ethics, license, provenance
- **MVTec AD is CC BY-NC-SA 4.0 (non-commercial).** Images and MVTec-derived figures are
  **not** committed to the repo; downloaded locally for evaluation only.
- Obtained via a public HF mirror (`foersben/mvtec-ad`) as the official links were down;
  the official page remains the licensing reference.
- See [SEGMENTATION_REPORT](../../artifacts/reports/SEGMENTATION_REPORT.md).
