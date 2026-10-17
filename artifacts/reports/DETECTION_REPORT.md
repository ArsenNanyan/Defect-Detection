# NEU-DET Detection — YOLO26n Report (Step 5)

Fine-tuned **YOLO26n** (COCO-pretrained, 2.51 M params) on the six steel-defect classes
using the **same immutable split** as classification. 80 epochs, imgsz 512, batch 8,
seed 42, on an RTX 3050 6 GB. Training was interrupted at epoch 69 and **resumed** from
`last.pt` to 80 (optimizer state restored) — a reproducibility note, not a separate run.

Reproduce:
```bash
python scripts/voc_to_yolo.py --raw data/raw/neu_det --splits data/splits --out data/processed/neu_det
python scripts/train_det.py  --config configs/detection.yaml
# (resume an interrupted run:) python scripts/train_det.py --config configs/detection.yaml --resume
```

## Headline metrics

| Split | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| validation | 0.790 | 0.463 | 0.761 | 0.722 |
| **test (frozen, single eval)** | **0.782** | **0.449** | 0.760 | 0.734 |

Val ≈ test (0.790 vs 0.782) → the model generalizes; the immutable split shows no leakage
or overfitting. The model checkpoint is **~26 MB**.

## Per-class AP@0.5 (test split)

| Class | Test AP@0.5 | Note |
|---|---|---|
| patches | 0.936 | large, high-contrast — easiest |
| scratches | 0.935 | strong linear structure |
| inclusion | 0.825 | |
| pitted_surface | 0.777 | |
| rolled-in_scale | 0.686 | subtle, low contrast |
| **crazing** | **0.536** | hardest — diffuse network cracks with ill-defined box extents |

## Reading the result
- **crazing** is the clear weak point (0.54). Its defect is a diffuse crack network, so
  box boundaries are inherently ambiguous and IoU-based AP is penalized — a known NEU-DET
  difficulty, not a pipeline bug.
- **patches / scratches** are near-ceiling: compact or strongly-oriented structures that
  localize cleanly.
- Localizing (detection mAP ~0.78) is a genuinely harder task than classifying
  (near-perfect F1 in Step 2) — exactly why the project includes both: a classifier says
  *what*, the detector says *where*.

## Figures (`artifacts/reports/figures/detection/`)
- `training_curves.png` — loss + metric curves across all 80 epochs.
- `val_PR_curve.png`, `test_PR_curve.png` — precision-recall per class.
- `val_confusion_normalized.png`, `test_confusion_normalized.png` — class confusions.

## Industrial-cost note
In inspection, a **missed defect (false negative)** is usually costlier than a false alarm.
The current recall is ~0.73; a deployment would lower the confidence threshold to trade
precision for recall, and the FastAPI/Streamlit demo exposes that threshold as a control.
